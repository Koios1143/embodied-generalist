#!/usr/bin/env python3
"""LEO (sft_noact) inference on the Beacon3D QA sets (ScanNet, 3RScan, MultiScan).

Each question is run exactly like LEO's own ScanQA / 3RScanQA evaluation (``LeoScanQA`` /
``Leo3RScanQA`` with ``pc_type=gt``): ground-truth object point clouds, the question's
``object_ids`` first, then the remaining objects in id order up to ``max_obj_len`` (60), the
prompt from ``LeoBase.get_prompts(question, situation="")``, no ego-view image, and
``LeoAgent.generate`` with the trainer's defaults (beam 5, repetition penalty 3.0).
``--no_hint`` drops the question-object priority (objects purely in id order), as a control.

Scan data: ScanNet from LEO's ``pcd_with_global_alignment`` (``LeoBase.load_scannet``),
3RScan from LEO's ``3RScan-ours-align`` (``LeoBase.load_rscan``), MultiScan from SceneVerse's
``MultiScan/{pcd_with_global_alignment,instance_id_to_label}`` (same tuple format; loaded the
way ``load_rscan`` does). Writes one jsonl row per question to ``--out``.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common.misc import default_collate  # noqa: E402
from data.dataset_wrapper import LeoObjPadDatasetWrapper  # noqa: E402
from data.datasets import LeoBase  # noqa: E402
from model.leo_agent import LeoAgent  # noqa: E402

# The released scan files are pickled numpy arrays; torch >= 2.6 loads with weights_only=True,
# so allow exactly the numpy globals needed to rebuild arrays instead of disabling it.
torch.serialization.add_safe_globals(
    [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
    + [type(np.dtype(t)) for t in (np.float32, np.float64, np.int64, np.int32, np.uint8, np.bool_)])


class Beacon3DQA(LeoBase):
    def __init__(self, items, domain, args, hint=True):
        super().__init__()
        self.items, self.domain, self.hint = items, domain, hint
        self.split = 'val'   # no rotation augmentation
        self.num_points, self.max_obj_len = args.num_points, args.max_obj_len
        self.scannet_base, self.rscan_base = args.scannet_base, args.rscan_base
        self.sceneverse_dir = args.sceneverse_dir
        self.scan_data = {}

    def load_sceneverse(self, scan_id):
        """``load_rscan`` for a SceneVerse ``<domain>/pcd_with_global_alignment`` scene."""
        base = os.path.join(self.sceneverse_dir, 'MultiScan')
        points, colors, instance_labels = torch.load(
            os.path.join(base, 'pcd_with_global_alignment', f'{scan_id}.pth'))[:3]
        pcds = np.concatenate([points, colors / 127.5 - 1], 1)
        inst_to_label = torch.load(os.path.join(base, 'instance_id_to_label', f'{scan_id}.pth'))
        return {'obj_pcds': {i: pcds[instance_labels == i] for i in inst_to_label.keys()}}

    def load_scene(self, scan_id):
        if scan_id not in self.scan_data:
            self.scan_data[scan_id] = {'scannet': self.load_scannet, '3rscan': self.load_rscan,
                                       'multiscan': self.load_sceneverse}[self.domain](scan_id)
        return self.scan_data[scan_id]['obj_pcds']

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        obj_pcds = self.load_scene(item['scene_id'])
        insts = [int(i) for i in item['object_ids']] if self.hint else []
        missing = [i for i in insts if i not in obj_pcds]
        assert not missing, f"{item['question_id']}: objects {missing} not in scene"
        order = insts + [i for i in obj_pcds.keys() if i not in insts]
        order = order[:self.max_obj_len]
        # fixed per-question seed: point subsampling in preprocess_pcd is random
        random.seed(index)
        np.random.seed(index)
        obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(
            [obj_pcds[i].copy() for i in order], return_anchor=False)
        data_dict = self.get_prompts(instruction=item['question'], situation="")
        data_dict.update({
            'source': self.domain,
            'scene_id': item['scene_id'],
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_locs': anchor_loc,
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'output_gt': item['answers'],
            'idx': index,
            'n_objs_scene': len(obj_pcds),
            'targets_in_input': all(int(i) in order for i in item['object_ids']),
        })
        return self.check_output_and_fill_dummy(data_dict)


def build_model(args):
    with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base=None):
        cfg = compose('default', overrides=[
            f'llm.cfg_path={args.llm}', f'vision3d.backbone.path={args.pointnetpp}'])
    model = LeoAgent(cfg)
    sd = torch.load(args.ckpt, map_location='cpu')
    res = model.load_state_dict(sd, strict=False)
    # everything not in the release checkpoint must be a frozen pretrained part
    frozen = ('llm_model.base_model.model.', 'pcd_encoder.obj_encoder.', 'img_encoder.')
    bad = [k for k in res.missing_keys if not k.startswith(frozen) or 'lora' in k]
    assert not res.unexpected_keys and not bad, (res.unexpected_keys, bad)
    print(f"loaded {len(sd)} tensors from {args.ckpt}; {len(res.missing_keys)} frozen keys from pretrained", flush=True)
    return model.cuda().eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--beacon_root', required=True, help='Beacon3D repo (data/<domain>/<domain>_qa.json)')
    ap.add_argument('--domains', default='scannet,3rscan,multiscan')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ckpt', required=True, help='sft_noact.pth')
    ap.add_argument('--llm', required=True, help='Vicuna-7B v1.1 dir')
    ap.add_argument('--pointnetpp', required=True, help='pointnetpp_vil3dref.pth')
    ap.add_argument('--scannet_base', default='', help='dir with scan_data/pcd_with_global_alignment')
    ap.add_argument('--rscan_base', default='', help='dir with 3RScan-ours-align')
    ap.add_argument('--sceneverse_dir', default='', help='dir with MultiScan/')
    ap.add_argument('--no_hint', action='store_true', help='control: no question-object priority')
    ap.add_argument('--num_points', type=int, default=1024)
    ap.add_argument('--max_obj_len', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='debug: first N questions per domain')
    args = ap.parse_args()
    PartialState()   # accelerate's logger needs an initialised state
    torch.manual_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    model = build_model(args)
    suffix = '_nohint' if args.no_hint else ''
    for domain in args.domains.split(','):
        items = json.load(open(os.path.join(args.beacon_root, 'data', domain, f'{domain}_qa.json')))
        if args.limit:
            items = items[:args.limit]
        ds = LeoObjPadDatasetWrapper(Beacon3DQA(items, domain, args, hint=not args.no_hint),
                                     EasyDict(max_obj_len=args.max_obj_len))
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, collate_fn=default_collate)
        out = Path(args.out_dir) / f'beacon3d_{domain}_qa{suffix}.jsonl'
        with open(out, 'w') as f:
            for batch in tqdm(loader, desc=domain):
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch = model.generate(batch)
                for j, pred in enumerate(batch['output_txt']):
                    i = int(batch['idx'][j])
                    it = items[i]
                    f.write(json.dumps({
                        'idx': i, 'question_id': it['question_id'], 'scene_id': it['scene_id'],
                        'question': it['question'], 'answers': it['answers'], 'pred': pred,
                        'n_objs_scene': int(batch['n_objs_scene'][j]),
                        'targets_in_input': bool(batch['targets_in_input'][j]),
                    }) + '\n')
        print(f'{domain}: {len(items)} questions -> {out}', flush=True)


if __name__ == '__main__':
    main()
