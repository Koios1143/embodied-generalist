#!/usr/bin/env python3
"""LEO (sft_noact) inference on Scan2Cap, ScanQA, SQA3D and Real-3DQA.

Every benchmark goes through LEO's own eval-time dataset classes, i.e. the configuration of
``configs/task/test_scannet.yaml`` + ``scripts/test_scannet_mask3d.sh``:

* ``scan2cap``  ``LeoScan2Cap`` val. Each object gets the proposal with the highest box IoU,
  then the other proposals with IoU >= 0.25, then the rest. The instruction and situation
  are drawn from LEO's pools, and the anchor is a random point in the object's box. Run
  with ``iou_thres=0.25``, which picks the same proposal as 0.5 whenever one reaches 0.5.
  Each row records the best IoU, so one run scores both @0.25 and @0.5.
* ``scanqa``    ``LeoScanQA`` val. Proposals with IoU >= 0.25 to a question object go first.
  ``--no_hint`` gives the control without that priority, i.e. proposals in Mask3D order.
* ``sqa3d``     ``LeoSQA3D`` test. The situation text and the pose of the situation
  (the anchor token) are inputs.
* ``real3dqa_<split>``  ``LeoSQA3D`` on Real-3DQA (``debiased_test``, ``rotation_{0,90,180,270}``).
  The situation text and the pose come from the split's own file: the rotation splits rotate
  both, while the point cloud stays the same.

``--pc_type pred`` (default) uses LEO's released Mask3D proposals (first 50 per scene);
``gt`` uses ground-truth objects. LEO's eval draws some inputs at random (point subsampling,
pool prompts, Scan2Cap anchor). Item ``i`` is seeded with ``i``, or with the scene id under
``--scene_seed``, which makes questions on one scene share their object tokens.

Speed options (off by default; the default is LEO's own generation, ``LeoAgent.generate``):
``--scene_sort`` processes items scene by scene. ``--prefix_cache`` prefills each distinct
prefix once (``fast_generate.py``). The two only share prefixes across questions when the
object order and subsampling are fixed per scene (``--no_hint --scene_seed``).
Sharding (``--shard k --num_shards n``) splits whole scenes when ``--scene_sort`` is on.
Writes ``<out_dir>/<bench>[_nohint].shard<k>of<n>.jsonl`` (+ ``.timing.json``).
"""

import argparse
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'extra_eval'))
from beacon3d.infer import build_model  # noqa: E402  (also registers numpy safe globals)
from common.misc import default_collate  # noqa: E402
from data.data_utils import construct_bbox_corners, convert_pc_to_box, eval_ref_one_sample  # noqa: E402
from data.dataset_wrapper import LeoObjPadDatasetWrapper  # noqa: E402
from data.datasets import LeoScan2Cap, LeoScanQA, LeoSQA3D  # noqa: E402
from fast_generate import generate_prefix_cached  # noqa: E402

REAL3DQA_SPLITS = ['debiased_test', 'rotation_0', 'rotation_90', 'rotation_180', 'rotation_270']
BENCHES = ['scan2cap', 'scanqa', 'sqa3d'] + [f'real3dqa_{s}' for s in REAL3DQA_SPLITS]


class Scan2CapIoU(LeoScan2Cap):
    """``LeoScan2Cap`` plus the best proposal IoU of the target (1.0 for ``pc_type=gt``)."""

    def best_iou(self, index):
        scan = self.scan_data[self.scan_ids[index]]
        if self.pc_type == 'gt':
            return 1.0
        gt = construct_bbox_corners(*convert_pc_to_box(scan['obj_pcds'][self.lang_data[index]['obj_id']]))
        return max([eval_ref_one_sample(construct_bbox_corners(*convert_pc_to_box(p)), gt)
                    for p in scan['obj_pcds_pred']] + [0.0])

    def __getitem__(self, index):
        d = super().__getitem__(index)
        d['max_iou'] = float(self.best_iou(index))
        return d


class ScanQANoHint(LeoScanQA):
    """``LeoScanQA`` without the question-object priority (objects in Mask3D / id order)."""

    def __getitem__(self, index):
        insts, self.scan_insts[index] = self.scan_insts[index], []
        try:
            return super().__getitem__(index)
        finally:
            self.scan_insts[index] = insts


class Real3DQA(LeoSQA3D):
    """``LeoSQA3D`` on a Real-3DQA split (same jsonl fields as SQA3D's annotations)."""

    def __init__(self, cfg, jsonl):
        self.jsonl = jsonl
        super().__init__(cfg, 'test')

    def load_anno(self, anno_dir):
        self.rows = [json.loads(line) for line in open(self.jsonl)]
        annos = [{'s': [r['situation']], 'q': r['question'], 'a': [r['answer']],
                  'pos': np.array(list(r['position'].values())),
                  'rot': np.array(list(r['rotation'].values()))} for r in self.rows]
        align = torch.load(os.path.join(anno_dir, 'axisAlignment.pth'))
        return [r['scene_id'] for r in self.rows], annos, align


class Seeded(Dataset):
    """Seeds python/numpy RNGs per item (index, or scene id) before LEO's ``__getitem__``."""

    def __init__(self, ds, scene_seed):
        self.ds, self.scene_seed = ds, scene_seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = zlib.crc32(self.ds.scan_ids[i].encode()) if self.scene_seed else i
        random.seed(s)
        np.random.seed(s)
        d = self.ds[i]
        d['idx'] = i
        return d


def build_dataset(bench, args, cfg):
    if bench == 'scan2cap':
        return Scan2CapIoU(cfg, 'val')
    if bench == 'scanqa':
        return (ScanQANoHint if args.no_hint else LeoScanQA)(cfg, 'val')
    if bench == 'sqa3d':
        return LeoSQA3D(cfg, 'test')
    return Real3DQA(cfg, os.path.join(args.real3dqa_dir, bench[len('real3dqa_'):] + '.jsonl'))


def item_meta(bench, ds, args):
    """Per-item ids / references for scoring, in dataset order."""
    if bench == 'scan2cap':
        return [{'corpus_key': k} for k in ds.corpus_cache]
    if bench == 'scanqa':
        items = json.load(open(os.path.join(args.anno, 'scanqa', 'ScanQA_v1.0_val.json')))
        return [{'question_id': it['question_id'], 'answers': [a.strip() for a in it['answers']]} for it in items]
    if bench == 'sqa3d':
        anns = json.load(open(os.path.join(args.anno, 'sqa3d', 'v1_balanced_sqa_annotations_test_scannetv2.json')))
        return [{'question_id': a['question_id'], 'answers': [x['answer'] for x in a['answers']]}
                for a in anns['annotations']]
    return [{'question_id': r['question_id'], 'answers': [r['answer']], 'situation': r['situation'],
             'question': r['question']} for r in ds.rows]


def shard_indices(ds, args):
    idx = list(range(len(ds)))
    if args.limit:
        idx = idx[:args.limit]
    if not args.scene_sort:
        return idx[args.shard::args.num_shards]
    scenes = list(dict.fromkeys(ds.scan_ids[i] for i in idx))
    mine = set(scenes[args.shard::args.num_shards])
    return sorted((i for i in idx if ds.scan_ids[i] in mine), key=lambda i: (scenes.index(ds.scan_ids[i]), i))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--benches', default=','.join(BENCHES))
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ckpt', required=True, help='sft_noact.pth')
    ap.add_argument('--llm', required=True, help='Vicuna-7B v1.1 dir')
    ap.add_argument('--pointnetpp', required=True, help='pointnetpp_vil3dref.pth')
    ap.add_argument('--scannet_base', required=True, help='dir with scan_data/pcd_with_global_alignment and mask/')
    ap.add_argument('--anno', required=True, help="LEO_data annotations/instruction (scan2cap, scanqa, sqa3d)")
    ap.add_argument('--real3dqa_dir', default='', help='Real-3DQA <split>.jsonl dir')
    ap.add_argument('--pc_type', default='pred', choices=['pred', 'gt'])
    ap.add_argument('--no_hint', action='store_true', help='ScanQA control: no question-object priority')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--scene_sort', action='store_true')
    ap.add_argument('--scene_seed', action='store_true')
    ap.add_argument('--prefix_cache', action='store_true')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0, help='debug: first N items per benchmark')
    args = ap.parse_args()
    PartialState()   # accelerate's logger needs an initialised state
    torch.manual_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base=None):
        cfg = compose('default', overrides=[
            'task=test_scannet', f'llm.cfg_path={args.llm}', f'vision3d.backbone.path={args.pointnetpp}',
            f'data.scan_family_base={args.scannet_base}', f'data.instruction_base={args.anno}',
            f'data.scan2cap.pc_type={args.pc_type}', f'data.scanqa.pc_type={args.pc_type}',
            f'data.sqa3d.pc_type={args.pc_type}', 'data.scan2cap.iou_thres=0.25'])
    model = build_model(args)
    generate = (lambda b: generate_prefix_cached(model, b)) if args.prefix_cache else model.generate
    for bench in args.benches.split(','):
        ds = build_dataset(bench, args, cfg)
        meta = item_meta(bench, ds, args)
        assert len(meta) == len(ds), (bench, len(meta), len(ds))
        idx = shard_indices(ds, args)
        wrapped = LeoObjPadDatasetWrapper(Seeded(ds, args.scene_seed), EasyDict(max_obj_len=cfg.dataset_wrapper_args.max_obj_len))
        loader = DataLoader(torch.utils.data.Subset(wrapped, idx), batch_size=args.batch_size,
                            num_workers=args.num_workers, collate_fn=default_collate)
        name = bench + ('_nohint' if args.no_hint and bench == 'scanqa' else '')
        out = Path(args.out_dir) / f'{name}.shard{args.shard}of{args.num_shards}.jsonl'
        t0, gen_time, n_prefill = time.time(), 0.0, 0
        with open(out, 'w') as f:
            for batch in tqdm(loader, desc=name):
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                torch.cuda.synchronize()
                tg = time.time()
                batch = generate(batch)
                torch.cuda.synchronize()
                gen_time += time.time() - tg
                n_prefill += batch.get('n_prefill', len(batch['output_txt']))
                for j, pred in enumerate(batch['output_txt']):
                    i = int(batch['idx'][j])
                    row = {'idx': i, 'scene_id': batch['scene_id'][j], **meta[i], 'pred': pred,
                           'prompt': batch['prompt_after_obj'][j]}
                    if bench == 'scan2cap':
                        row['max_iou'] = float(batch['max_iou'][j])
                    if bench == 'sqa3d' or bench.startswith('real3dqa'):
                        row['sqa_type'] = int(batch['sqa_type'][j])
                    f.write(json.dumps(row) + '\n')
        timing = {'items': len(idx), 'wall_s': time.time() - t0, 'generate_s': gen_time, 'prefills': n_prefill,
                  **{k: getattr(args, k) for k in ('batch_size', 'scene_sort', 'scene_seed', 'prefix_cache',
                                                   'pc_type', 'no_hint')}}
        json.dump(timing, open(str(out).replace('.jsonl', '.timing.json'), 'w'), indent=1)
        print(f'{name}: {len(idx)} items, {timing["wall_s"]:.0f}s (generate {gen_time:.0f}s, '
              f'{n_prefill} prefills) -> {out}', flush=True)


if __name__ == '__main__':
    main()
