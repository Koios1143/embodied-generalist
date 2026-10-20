#!/usr/bin/env python3
"""Score ``beacon3d/infer.py`` outputs with Beacon3D's official QA evaluator. CPU only.

Per domain (ScanNet, 3RScan, MultiScan) and run variant (``""`` = LEO's protocol,
``_nohint`` = the no-question-object control): writes ``infer_qa.json`` in the evaluator's
input format, runs the unmodified ``evaluate_qa.py`` with the GPT judge (``--qa_model``,
default gpt-4o-mini-2024-07-18; needs OPENAI_API_KEY) from the Beacon3D dir, keeps its
printout as ``score_qa.txt``, and re-derives the numbers from its processed file (checked
against the printout) so the domains can be pooled. Beacon3D defines no cross-domain score;
``overall`` reports ``pooled`` (all 2448 questions together) and ``mean`` (of the three).
There is no grounding run (LEO has no grounding output), so the grounding-QA chain
analysis is not reported.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DOMAINS = ['scannet', '3rscan', 'multiscan']
TAGS = ['class', 'appearance', 'geometry', 'spatial', 'existence', 'functionality']


def metrics(pq):
    """Beacon3D QA metrics from processed items (``eval_score`` of evaluate_qa.py)."""
    gpt = lambda xs: (float(np.mean(xs)) - 1) / 4 * 100  # noqa: E731
    sc = [it['score'] for it in pq]
    m = {'case': gpt(sc),
         'object': 100 * float(np.mean([all(x >= 4 for x in sc[i:i + 3]) for i in range(0, len(sc), 3)])),
         'EM': 100 * float(np.mean([it['em'] for it in pq])),
         'EM-R': 100 * float(np.mean([it['em_r'] for it in pq]))}
    for t in TAGS:
        xs = [it['score'] for it in pq if it['tag'].lower() == t]
        m[t] = gpt(xs) if xs else None
    m['n_cases'] = len(sc)
    m['n_objects'] = len(range(0, len(sc), 3))
    return m


def score_domain(pred_dir, out, beacon_root, qa_model, domain, variant):
    src = json.load(open(beacon_root / 'data' / domain / f'{domain}_qa.json'))
    rows = sorted((json.loads(l) for l in open(pred_dir / f'beacon3d_{domain}_qa{variant}.jsonl')),
                  key=lambda r: r['idx'])
    assert [r['idx'] for r in rows] == list(range(len(src))), f'{domain}{variant}: {len(rows)} vs {len(src)}'
    bdir = out / f'{domain}{variant}'
    bdir.mkdir(parents=True, exist_ok=True)
    infer = [{'question_id': s['question_id'], 'scene_id': s['scene_id'], 'question': s['question'],
              'pred': r['pred'], 'answers': s['answers']} for s, r in zip(src, rows)]
    json.dump(infer, open(bdir / 'infer_qa.json', 'w'), indent=1)
    processed = bdir / 'infer_qa_processed.json'
    processed.unlink(missing_ok=True)  # evaluate_qa.py silently reuses an existing one
    cmd = [sys.executable, 'evaluate_qa.py', '--infer', str(bdir / 'infer_qa.json'), '--domain', domain,
           '--model', qa_model, '--output', str(processed)]
    print('$', ' '.join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=beacon_root, capture_output=True, text=True)
    (bdir / 'score_qa.txt').write_text((p.stdout + ('' if p.returncode == 0 else '\n[stderr]\n' + p.stderr))
                                       .replace(str(REPO) + '/', ''))
    print(p.stdout[-1500:])
    if p.returncode != 0:
        raise SystemExit(f'{domain}{variant}: evaluate_qa.py failed\n{p.stderr[-3000:]}')
    pq = json.load(open(processed))
    res = {'qa_judge': qa_model, 'metrics': metrics(pq),
           'targets_dropped': sum(not r['targets_in_input'] for r in rows),
           'scenes_over_max_obj': sorted({r['scene_id'] for r in rows if r['n_objs_scene'] > 60})}
    official = float(p.stdout.split('GPT-Score =')[1].split()[0])
    assert abs(official - res['metrics']['case']) < 0.01, (official, res['metrics'])
    json.dump(res, open(bdir / 'scores.json', 'w'), indent=1)
    return res, pq


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pred_dir', required=True, help='infer.py --out_dir')
    ap.add_argument('--result_dir', default=str(HERE / 'results'))
    ap.add_argument('--beacon_root', default=str(HERE / 'Beacon3D'))
    ap.add_argument('--qa_model', default='gpt-4o-mini-2024-07-18',
                    help='GPT judge (upstream default is gpt-4o-2024-11-20)')
    ap.add_argument('--variants', default=',_nohint', help="comma list of '' and/or '_nohint'")
    args = ap.parse_args()
    pred_dir, out, beacon_root = Path(args.pred_dir), Path(args.result_dir).resolve(), Path(args.beacon_root).resolve()
    summary_path = out / 'summary.json'
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    for variant in args.variants.split(','):
        if not all((pred_dir / f'beacon3d_{d}_qa{variant}.jsonl').exists() for d in DOMAINS):
            print(f'skip variant {variant!r}: predictions missing')
            continue
        per, pooled = {}, []
        for d in DOMAINS:
            res, pq = score_domain(pred_dir, out, beacon_root, args.qa_model, d, variant)
            summary[f'{d}{variant}'] = res
            per[d] = res['metrics']
            pooled += pq
        overall = {'qa_judge': args.qa_model, 'pooled': metrics(pooled),
                   'mean': {k: float(np.mean([per[d][k] for d in DOMAINS if per[d][k] is not None]))
                            for k in per['scannet'] if not k.startswith('n_')}}
        (out / f'overall{variant}').mkdir(parents=True, exist_ok=True)
        json.dump(overall, open(out / f'overall{variant}' / 'scores.json', 'w'), indent=1)
        summary[f'overall{variant}'] = overall
    json.dump(summary, open(summary_path, 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
