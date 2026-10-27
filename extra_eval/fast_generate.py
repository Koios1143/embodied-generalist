"""``LeoAgent.generate`` with a shared-prefix KV cache (opt-in eval speedup).

LEO's input is ``[role + situation][ego-view prompt, image][objects prompt, object tokens]
[USER: question ASSISTANT:][BOS]``. ``LeoAgent.generate`` hands the whole sequence to a
beam search, so the prefix (everything up to the object tokens) is prefilled once per beam
of every question. Here every distinct prefix in the batch is prefilled once. Its KV cache
is then handed to the beam search, which only encodes the question part. The tokens,
attention mask and position ids (cumsum of the mask, as in LLaMA's
``prepare_inputs_for_generation``) are the same as in ``LeoAgent.generate``, so the
computation is the same up to floating-point reduction order.

Questions share a prefix only when their situation text, object order, point subsampling
and anchor pose are all identical. That holds for questions on the same scene with a fixed
object order and no situation, e.g. ScanQA / Beacon3D without the question-object hint and
with per-scene subsampling (``--scene_seed``). Otherwise each question gets its own prefix,
and the saving is the per-beam duplication only.
"""

import types

import torch

from model.utils import maybe_autocast


def _tokens(tok, texts, padding_side, **kw):
    tok.padding_side = padding_side
    return tok(texts, return_tensors='pt', padding='longest', **kw)


def build_prefix_and_question(model, data_dict):
    """``LeoAgent.build_right_justified_sequence`` split into the prefix
    ``[pre][mid1][img][mid2][obj]`` (left padded) and the question ``[post]`` (no BOS)."""
    device = model.device
    tok = model.llm_tokenizer
    embed = model.llm_model.get_input_embeddings()
    pre = _tokens(tok, data_dict['prompt_before_obj'], 'left').to(device)
    mid1 = _tokens(tok, data_dict['prompt_middle_1'], 'left').to(device)
    mid2 = _tokens(tok, data_dict['prompt_middle_2'], 'left').to(device)
    tok.truncation_side = 'left'
    post = _tokens(tok, data_dict['prompt_after_obj'], 'right', truncation=True,
                   max_length=model.max_context_len).to(device)
    assert mid1.attention_mask.all() and mid2.attention_mask.all()
    img_tokens = data_dict['img_tokens'].to(device)
    img_masks = data_dict['img_masks'].to(device).reshape(-1, 1).repeat(1, img_tokens.size(1))
    obj_tokens = data_dict['obj_tokens'].to(device)
    obj_masks = data_dict['obj_masks'].to(device)
    assert not model.clip_txt_guidance
    mid1_mask = mid1.attention_mask[:, 1:].clone()
    mid1_mask[~img_masks.any(1)] = 0   # no image: also mask the ego-view prompt
    pre_embeds = embed(pre.input_ids)
    # upstream assembles the sequence in the text-embedding dtype (fp16)
    prefix = torch.cat([pre_embeds, embed(mid1.input_ids[:, 1:]), img_tokens,
                        embed(mid2.input_ids[:, 1:]), obj_tokens], dim=1).to(pre_embeds.dtype)
    prefix_mask = torch.cat([pre.attention_mask, mid1_mask, img_masks, mid2.attention_mask[:, 1:], obj_masks],
                            dim=1).long()
    questions = [embed(ids[1:][m[1:].bool()]) for ids, m in zip(post.input_ids, post.attention_mask)]
    return prefix, prefix_mask, questions


def unique_prefixes(prefix, prefix_mask):
    """Indices of the distinct prefixes and, per question, which one it uses."""
    uniq, owner = [], []
    for i in range(prefix.shape[0]):
        for u, j in enumerate(uniq):
            if torch.equal(prefix_mask[i], prefix_mask[j]) and torch.equal(prefix[i], prefix[j]):
                owner.append(u)
                break
        else:
            owner.append(len(uniq))
            uniq.append(i)
    return uniq, owner


@torch.no_grad()
def generate_prefix_cached(model, data_dict, num_beams=5, max_length=256, min_length=1,
                           repetition_penalty=3.0, length_penalty=1):
    """Drop-in for ``model.generate(data_dict)`` (beam search, LEO's eval defaults)."""
    device = model.device
    if 'obj_tokens' not in data_dict:
        data_dict = model.pcd_encoder(data_dict)
    data_dict['obj_tokens'] = model.pcd_proj(data_dict['obj_tokens'].to(device))
    data_dict['img_tokens'] = model.img_proj(model.img_encoder(data_dict['img_fts']))
    prefix, prefix_mask, questions = build_prefix_and_question(model, data_dict)
    uniq, owner = unique_prefixes(prefix, prefix_mask)
    data_dict['n_prefill'] = len(uniq)

    # prefill each distinct prefix once
    u_mask = prefix_mask[uniq]
    u_pos = (u_mask.cumsum(-1) - 1).masked_fill(u_mask == 0, 1)
    with maybe_autocast(model):
        out = model.llm_model(inputs_embeds=prefix[uniq], attention_mask=u_mask, position_ids=u_pos,
                              use_cache=True, return_dict=True)
    idx = torch.tensor([owner[i] for i in range(len(owner)) for _ in range(num_beams)], device=device)
    past = tuple(tuple(t.index_select(0, idx) for t in layer) for layer in out.past_key_values)
    del out

    # question part: [left padding][question tokens][BOS]
    bos = model.llm_model.get_input_embeddings()(
        torch.tensor([[model.llm_tokenizer.bos_token_id]], device=device))[0]
    questions = [torch.cat([q, bos], 0) for q in questions]
    bs, q_len = len(questions), max(q.shape[0] for q in questions)
    inputs_embeds = prefix.new_zeros((bs, q_len, prefix.shape[2]))
    q_mask = torch.zeros((bs, q_len), dtype=torch.long, device=device)
    for b, q in enumerate(questions):
        inputs_embeds[b, q_len - q.shape[0]:] = q
        q_mask[b, q_len - q.shape[0]:] = 1
    attention_mask = torch.cat([prefix_mask, q_mask], dim=1)

    # LLaMA's prepare_inputs_for_generation drops inputs_embeds once a cache is given; feed
    # them on the first step, then one token at a time (position ids as upstream)
    llm = model.llm_model
    first = [True]

    def prepare(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        pos = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)
        if first[0]:
            first[0] = False
            n = inputs_embeds.shape[1]
            return {'inputs_embeds': inputs_embeds, 'position_ids': pos[:, -n:], 'past_key_values': past_key_values,
                    'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask}
        return {'input_ids': input_ids[:, -1:], 'position_ids': pos[:, -1:], 'past_key_values': past_key_values,
                'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask}

    # LEO's LoRA wrapper is a plain PeftModel: generate() runs on the LlamaForCausalLM itself
    base = llm.get_base_model() if hasattr(llm, 'get_base_model') else llm
    # bound method (generate() checks ``__func__``); the instance attribute shadows the class one
    base.prepare_inputs_for_generation = types.MethodType(prepare, base)
    try:
        with maybe_autocast(model):
            outputs = llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                   past_key_values=past, do_sample=False, num_beams=num_beams,
                                   max_length=max_length, min_length=min_length,
                                   repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                   num_return_sequences=1)
    finally:
        del base.prepare_inputs_for_generation

    tok = model.llm_tokenizer
    outputs[outputs == tok.unk_token_id] = tok.eos_token_id
    data_dict['output_txt'] = [t.strip() for t in tok.batch_decode(outputs, skip_special_tokens=True)]
    return data_dict
