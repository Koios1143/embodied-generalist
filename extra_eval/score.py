#!/usr/bin/env python3
"""Score a ``run_extra_eval.sh`` output dir (CPU only; METEOR needs Java).

Shards are merged and checked against the benchmark size. Then:

* Scan2Cap: LEO's ``CaptionEvaluator`` protocol. Predictions are formatted as
  ``sos <caption> eos`` and scored against LEO's ``scanrefer_corpus.json`` with LEO's
  ``evaluator/ngram_metrics`` (CIDEr, BLEU-4, METEOR, ROUGE-L). An object whose best Mask3D
  proposal has IoU < k scores as ``sos eos``. Reported @0.25 and @0.5.
* ScanQA, SQA3D: LEO's ``clean_answer`` + ``answer_match`` (EM, refined EM). SQA3D also per
  question type (What/Is/How/Can/Which/Others). ScanQA also gets the n-gram metrics, as in
  ``ScanQAEvaluator``.
* Real-3DQA: the official ``evaluate_debiased_exact_match.py`` (EM, refined EM) on every
  split, and ``evaluate_rotation_robustness.py`` (One..Four, VRS) on the rotation splits.

An empty generation becomes ``<empty>`` before matching. Otherwise the refined match counts
it as correct, because the empty string is contained in every answer. The number of empty
generations is reported.

Writes ``<result_dir>/<bench>/scores.json`` (+ Real-3DQA prediction files) and
``<result_dir>/summary.json``.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / 'real3dqa'))

EXPECTED = {'scan2cap': 2068, 'scanqa': 4675, 'scanqa_nohint': 4675, 'sqa3d': 3519,
            'real3dqa_debiased_test': 1485, 'real3dqa_rotation_0': 750, 'real3dqa_rotation_90': 750,
            'real3dqa_rotation_180': 750, 'real3dqa_rotation_270': 750}
ROTATIONS = ['rotation_0', 'rotation_90', 'rotation_180', 'rotation_270']
SQA_TYPES = ['what', 'is', 'how', 'can', 'which', 'others']
EMPTY = '<empty>'


def load_preds(pred_dir, name):
    files = sorted(glob.glob(os.path.join(pred_dir, f'{name}.shard*of*.jsonl')))
    if not files:
        return None
    n_shards = {f.rsplit('of', 1)[1].split('.')[0] for f in files}
    assert len(n_shards) == 1 and len(files) == int(n_shards.pop()), f'{name}: incomplete shards {files}'
    rows = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            assert r['idx'] not in rows, f'{name}: duplicate idx {r["idx"]}'
            rows[r['idx']] = r
    assert sorted(rows) == list(range(EXPECTED[name])), f'{name}: {len(rows)} items, expected {EXPECTED[name]}'
    return [rows[i] for i in range(len(rows))]


def ngram_scores(gts, preds):
    from evaluator.ngram_metrics.bleu.bleu import Bleu
    from evaluator.ngram_metrics.cider.cider import Cider
    from evaluator.ngram_metrics.meteor.meteor import Meteor
    from evaluator.ngram_metrics.rouge.rouge import Rouge
    return {'CIDEr': 100 * Cider().compute_score(gts, preds)[0],
            'BLEU-4': 100 * Bleu().compute_score(gts, preds)[0][-1],
            'METEOR': 100 * Meteor().compute_score(gts, preds)[0],
            'ROUGE-L': 100 * Rouge().compute_score(gts, preds)[0]}


def score_scan2cap(rows, corpus):
    gt_all = json.load(open(corpus))
    res = {'n': len(rows), 'iou_ge_0.25': sum(r['max_iou'] >= 0.25 for r in rows),
           'iou_ge_0.5': sum(r['max_iou'] >= 0.5 for r in rows)}
    for thr in (0.25, 0.5):
        preds = {r['corpus_key']: [('sos ' + r['pred'] + ' eos').replace('. ', ' . ')]
                 if r['max_iou'] >= thr else ['sos eos'] for r in rows}
        gts = {k: gt_all[k] for k in preds}
        res.update({f'{k}@{thr}': v for k, v in ngram_scores(gts, preds).items()})
    return res


def qa_match(rows):
    from data.data_utils import clean_answer
    from evaluator.scanqa_eval import ScanQAEvaluator
    match = ScanQAEvaluator.answer_match   # identical to SQA3DEvaluator.answer_match
    out = []
    for r in rows:
        pred = clean_answer(r['pred'])
        empty = not pred.strip()
        em, em_r = match(None, EMPTY if empty else pred, [clean_answer(a) for a in r['answers']])
        out.append((pred, em, em_r, empty))
    return out


def score_scanqa(rows):
    m = qa_match(rows)
    res = {'n': len(rows), 'EM': 100 * sum(x[1] for x in m) / len(m), 'EM-R': 100 * sum(x[2] for x in m) / len(m),
           'empty_preds': sum(x[3] for x in m)}
    from data.data_utils import clean_answer
    res.update(ngram_scores({i: [clean_answer(a) for a in r['answers']] for i, r in enumerate(rows)},
                            {i: [x[0]] for i, x in enumerate(m)}))
    return res


def score_sqa3d(rows):
    m = qa_match(rows)
    res = {'n': len(rows), 'EM': 100 * sum(x[1] for x in m) / len(m), 'EM-R': 100 * sum(x[2] for x in m) / len(m),
           'empty_preds': sum(x[3] for x in m)}
    for t, name in enumerate(SQA_TYPES):
        sel = [x for x, r in zip(m, rows) if r['sqa_type'] == t]
        res[f'EM_{name}'] = 100 * sum(x[1] for x in sel) / max(len(sel), 1)
        res[f'EM-R_{name}'] = 100 * sum(x[2] for x in sel) / max(len(sel), 1)
        res[f'n_{name}'] = len(sel)
    return res


def score_real3dqa(pred_dir, out, name):
    import evaluate_debiased_exact_match as dem
    import evaluate_rotation_robustness as rot
    res, files = {'empty_preds': {}}, {}
    for split in ['debiased_test'] + ROTATIONS:
        rows = load_preds(pred_dir, f'real3dqa_{split}')
        if rows is None:
            return None
        recs = []
        for r in rows:
            empty = not dem.clean_answer(r['pred']).strip()
            res['empty_preds'][split] = res['empty_preds'].get(split, 0) + empty
            # int ids: the rotation script int()s the suffix-stripped ids of 90/180/270 only
            recs.append({'question_id': int(r['question_id']), 'scene_id': r['scene_id'],
                         'response_gt': r['answers'], 'response_pred': EMPTY if empty else r['pred'],
                         'prompt': r['prompt']})
        files[split] = out / f'pred_{split}.json'
        json.dump(recs, open(files[split], 'w'), indent=1)
        res[split] = dem.evaluate_em(str(files[split]))
    res['rotation'] = rot.quick_consistency_score([str(files[s]) for s in ROTATIONS])
    log = []   # the same numbers through the unmodified scripts' CLIs, for the record
    for cmd in [[sys.executable, str(HERE / 'real3dqa' / 'evaluate_debiased_exact_match.py'),
                 str(files['debiased_test']), '--name', name],
                [sys.executable, str(HERE / 'real3dqa' / 'evaluate_rotation_robustness.py')]
                + [str(files[s]) for s in ROTATIONS] + ['--name', name]]:
        log.append(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    (out / 'official_scripts.txt').write_text('\n'.join(log).replace(str(REPO) + '/', ''))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pred_dir', required=True, help='run_extra_eval.sh output dir')
    ap.add_argument('--result_dir', default='extra_eval/results')
    ap.add_argument('--corpus', required=True, help="LEO's scan2cap/scanrefer_corpus.json")
    ap.add_argument('--benches', default='scan2cap,scanqa,scanqa_nohint,sqa3d,real3dqa')
    ap.add_argument('--name', default='LEO')
    args = ap.parse_args()
    os.chdir(REPO)   # METEOR's jar path is relative to evaluator/
    result_dir = Path(args.result_dir)
    summary_file = result_dir / 'summary.json'
    summary = json.load(open(summary_file)) if summary_file.exists() else {}
    for bench in args.benches.split(','):
        out = result_dir / bench
        out.mkdir(parents=True, exist_ok=True)
        if bench == 'real3dqa':
            res = score_real3dqa(args.pred_dir, out, args.name)
        else:
            rows = load_preds(args.pred_dir, bench)
            res = None if rows is None else {'scan2cap': lambda r: score_scan2cap(r, args.corpus),
                                             'scanqa': score_scanqa, 'scanqa_nohint': score_scanqa,
                                             'sqa3d': score_sqa3d}[bench](rows)
        if res is None:
            print(f'[{bench}] no predictions in {args.pred_dir}, skipped')
            continue
        timing = [json.load(open(f)) for f in glob.glob(os.path.join(args.pred_dir, (f'{bench}_*.shard*.timing.json' if bench == 'real3dqa' else f'{bench}.shard*.timing.json')))]
        res = {'pred_dir': os.path.relpath(os.path.abspath(args.pred_dir), REPO),
               'run': {k: v for k, v in timing[0].items() if k not in ('items', 'wall_s', 'generate_s', 'prefills')}
               if timing else {}, **res}
        json.dump(res, open(out / 'scores.json', 'w'), indent=1)
        summary[bench] = res
        print(f'[{bench}]', json.dumps(res, indent=1))
    json.dump(summary, open(summary_file, 'w'), indent=1)


if __name__ == '__main__':
    main()
