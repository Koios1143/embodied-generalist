#!/usr/bin/env python3
"""Compare two ``run_extra_eval.sh`` output dirs item by item (e.g. LEO's ``generate`` vs
``--prefix_cache``): share of identical generations and generate time, per benchmark."""

import argparse
import glob
import json
import os
from collections import defaultdict


def load(d):
    preds, timing = defaultdict(dict), defaultdict(float)
    for f in glob.glob(os.path.join(d, '*.shard*of*.jsonl')):
        name = os.path.basename(f).split('.shard')[0]
        for line in open(f):
            r = json.loads(line)
            preds[name][r['idx']] = r['pred']
        t = json.load(open(f.replace('.jsonl', '.timing.json')))
        timing[name] += t['generate_s']
    return preds, timing


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('ref')
    ap.add_argument('new')
    args = ap.parse_args()
    (pa, ta), (pb, tb) = load(args.ref), load(args.new)
    print(f'{"bench":28s} {"n":>6s} {"identical":>9s} {"gen_ref_s":>9s} {"gen_new_s":>9s} {"speedup":>7s}')
    for name in sorted(set(pa) & set(pb)):
        common = set(pa[name]) & set(pb[name])
        same = sum(pa[name][i] == pb[name][i] for i in common)
        print(f'{name:28s} {len(common):6d} {100 * same / max(len(common), 1):8.1f}% '
              f'{ta[name]:9.1f} {tb[name]:9.1f} {ta[name] / max(tb[name], 1e-9):6.2f}x')


if __name__ == '__main__':
    main()
