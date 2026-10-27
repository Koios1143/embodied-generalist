# LEO Benchmark Evaluation

This is a fork of [LEO](https://github.com/embodied-generalist/embodied-generalist)
("An Embodied Generalist Agent in 3D World", ICML 2024) whose purpose is to **evaluate the
released LEO checkpoint on benchmarks the paper does not report**, under one reproducible
setup, so its scores can be compared with other models on equal footing. Wired in: the QA
half of [Beacon3D](https://beacon-3d.github.io/) (CVPR 2025; ScanNet, 3RScan, MultiScan),
[Real-3DQA](https://arxiv.org/abs/2603.23523), and a re-run of the paper's own Scan2Cap,
ScanQA and SQA3D evaluations as a sanity check of the setup. LEO generates text only (it
has no object-ID or box output), so Beacon3D grounding is not evaluated.

The upstream model, training and data code are untouched apart from a one-line torch-2
compatibility fix in `common/misc.py`; everything this fork adds lives under
[`beacon3d/`](beacon3d/) and [`extra_eval/`](extra_eval/), and the per-question outputs
behind the numbers below are committed in their `results/` directories.

Upstream LEO's own README is in the git history (`git show 567f2d8:README.md`).

## Results

Released checkpoint `sft_noact.pth` (instruction-tuned, no embodied acting) from
[huangjy-pku/LEO_data](https://huggingface.co/datasets/huangjy-pku/LEO_data), Vicuna-7B
v1.1, PointNet++ backbone, no training or fine-tuning of any kind. All numbers are
percentages.

### Beacon3D QA

GPT judge: **`gpt-4o-mini-2024-07-18`** (the evaluator's default is `gpt-4o-2024-11-20`).
*Case* = GPT-Score ((mean 1–5 score − 1) / 4); *object* = share of objects whose three
questions all score ≥ 4; EM / EM-R from the same evaluator; per-type columns are GPT-Score.

| Domain | #Q | Case | Object | EM | EM-R | Class | App. | Geo. | Spa. | Exi. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ScanNet | 1155 | **52.94** | **11.17** | 49.00 | 52.21 | 19.14 | 63.96 | 53.16 | 51.89 | 54.29 |
| 3RScan | 816 | **45.77** | **1.47** | 41.91 | 46.45 | 22.28 | 46.02 | 49.09 | 49.34 | 45.02 |
| MultiScan | 477 | **28.56** | **0.63** | 25.58 | 27.88 | 7.66 | 33.58 | 58.33 | 27.94 | 28.39 |
| Overall (pooled) | 2448 | 45.80 | 5.88 | 42.08 | 45.55 | 16.55 | 52.21 | 52.05 | 46.16 | 45.56 |
| Overall (mean of 3) | | 42.43 | 4.42 | 38.83 | 42.18 | 16.36 | 47.85 | 53.53 | 43.06 | 42.57 |

ScanNet also has 3 *functionality* questions (66.67). Beacon3D defines no cross-domain
score; *pooled* treats all 2,448 questions (816 objects) as one set, *mean* averages the
three domain numbers. Without a grounding run there is no grounding–QA chain analysis.

### Read this before quoting the Beacon3D numbers

- **Protocol = LEO's own ScanQA / 3RScanQA evaluation.** LEO is fed ground-truth object
  point clouds, and, exactly as `LeoScanQA` / `Leo3RScanQA` do at eval time, the objects in
  the question's `object_ids` are placed **first** in the object sequence, followed by the
  rest of the scene in id order, up to `max_obj_len = 60`. This is how LEO's published QA
  numbers are produced, so the table above uses it, but the question-relevant objects are
  annotation information the model would not have otherwise.
- **No-hint control.** Same run with the objects purely in id order (`--no_hint`):

  | Domain | Case | Object | EM | EM-R |
  | --- | --- | --- | --- | --- |
  | ScanNet | 45.15 | 7.01 | 41.82 | 44.59 |
  | 3RScan | 42.46 | 1.84 | 39.71 | 43.50 |
  | MultiScan | 26.26 | 0.63 | 22.85 | 26.21 |
  | Overall (pooled) | 40.57 | 4.04 | 37.42 | 40.65 |
  | Overall (mean of 3) | 37.96 | 3.16 | 34.79 | 38.10 |

  Every Beacon3D scene but one has ≤ 60 objects, so the control sees exactly the same
  objects and only their order changes. The exception is 3RScan `a0905fd9-…` (73 objects),
  where 18 questions lose their target object. The 7.8-point drop on ScanNet therefore
  comes almost entirely from **position**: LEO has learned that the first object tokens are
  the ones the question is about. Results in [`beacon3d/results/*_nohint/`](beacon3d/results/).
- **No ego-view image**, as in LEO's own 3D QA evaluation (`img_masks = 0`).
- **Scan data.** ScanNet: LEO's `pcd_with_global_alignment` release; 3RScan: LEO's
  `3RScan-ours-align` release; MultiScan (LEO has none): SceneVerse's
  `MultiScan/pcd_with_global_alignment`, which has the same `(points, colors,
  instance_labels)` format and is loaded the way LEO loads 3RScan. All Beacon3D
  `object_ids` exist in these files with the same class labels (checked for every
  question). The 3RScan instance-to-label tables are identical to SceneVerse's.
- **Determinism.** LEO subsamples 1,024 points per object at random; `infer.py` seeds
  `random` / `numpy` with the question index, so a rerun reproduces the same inputs.
  Decoding is LEO's trainer default (beam 5, repetition penalty 3.0, no sampling).
- **Environment port.** The release pins torch 1.12.1+cu113, which has no Blackwell
  kernels; this run used torch 2.7.1+cu128 with every other package at the version in
  `requirements.txt` (transformers 4.28.1, peft 0.5.0, accelerate 0.26.1).

### Scan2Cap, ScanQA, SQA3D, Real-3DQA

LEO's own dataset classes and eval protocol (`task=test_scannet`, `pc_type=pred`: Mask3D
proposals from `LEO_data/mask.zip`, as in the paper, "following 3D-VisTA"). Outputs in
[`extra_eval/results/`](extra_eval/results/).

| Benchmark | Metric | This run | LEO paper (Table 4) |
| --- | --- | --- | --- |
| Scan2Cap val (2068 obj.) | C / B-4 / M / R @0.5 | 71.66 / 37.95 / 27.77 / 57.94 | 72.4 / 38.2 / 27.9 / 58.1 |
| | C / B-4 / M / R @0.25 | 76.73 / 41.04 / 29.16 / 61.61 | – |
| ScanQA val (4675 Q) | C / B-4 / M / R | 101.80 / 15.50 / 19.92 / 48.49 | 101.4 / 13.2 / 20.0 / 49.2 |
| | EM / EM-R | 24.21 / 47.32 | 24.5 / 47.6 |
| SQA3D test (3519 Q) | EM / EM-R | 49.79 / 52.43 | 50.0 / 52.4 |

| Real-3DQA | EM / EM-R | One / Two / Three / Four | VRS |
| --- | --- | --- | --- |
| This run | 15.02 / 19.60 | 45.2 / 7.6 / 1.6 / 0.4 | 13.70 |
| Real-3DQA paper (LEO row) | 14.3 / 19.1 | 46.9 / 8.1 / 1.6 / 0.4 | 14.3 |

- **Scan2Cap**: LEO's split and its own `scanrefer_corpus.json` references. The target
  object goes first, followed by the proposals that overlap it; a prediction counts only
  if the best Mask3D proposal reaches the IoU threshold (one run, scored at both thresholds).
  Only 1695 / 2068 targets have a proposal at IoU ≥ 0.5. Every number is within 1 point
  of the paper except ScanQA B-4 (+2.3; computed with LEO's own n-gram evaluator, cause
  not investigated).
- **ScanQA hint.** As in Beacon3D, `LeoScanQA` puts the proposals overlapping the
  question's annotated objects (IoU ≥ 0.25) first. No-hint control (`--no_hint`):
  C 67.61, EM / EM-R 15.02 / 30.35, i.e. −34 CIDEr / −9 EM.
- **SQA3D / Real-3DQA** feed the situation text and LEO's situation pose (anchor
  position + orientation, which rotates every object token). Each Real-3DQA rotation split
  carries its own rotated situation text *and* quaternion. Real-3DQA scores come from its
  official scripts (vendored in `extra_eval/real3dqa/`); empty answers become `<empty>`.
- **Determinism**: random object subsampling and Scan2Cap instructions are seeded per item.

### Faster evaluation (`FAST=1`, opt-in)

`extra_eval/fast_generate.py` replaces `LeoAgent.generate` with a version that prefills
each distinct prompt prefix (situation, ego-view, object tokens) once and hands its KV
cache to the beam search, which then encodes only the question. Upstream prefills the full
sequence once per beam. Same tokens, masks and position ids; differences come only from
floating-point reduction order, which flips near-tied beams. Full-run comparison against
the reference above (2 GPUs, `--scene_sort --prefix_cache`, same seeds):

| Benchmark | Identical outputs | Generate time | Score change |
| --- | --- | --- | --- |
| Scan2Cap | 79.4% | 1.38× faster | C@0.5 71.66 → 71.66 |
| ScanQA / no-hint | 95.5% / 96.1% | 1.56× / 1.52× | EM 24.21 → 24.28 / 15.02 → 14.89 |
| SQA3D | 98.4% | 1.18× | EM 49.79 → 49.93 |
| Real-3DQA debiased / rotations | 98.2% / 95.5–96.4% | 1.44× / 1.57–1.72× | EM 15.02 → 15.22, VRS 13.70 → 13.97 |

The whole job took 78 min on two RTX PRO 6000 (reference + fast). Questions share a prefix
only when the situation, object order, point subsampling and pose are all identical, e.g.
ScanQA without the hint plus `--scene_seed` (one subsample per scene). That gives one
prefill per scene, but changing the seeding alone moves ~20% of answers, so it is not used
for reported numbers. The same flags (`--prefix_cache`, `--scene_seed`) exist in
`beacon3d/infer.py` via `EXTRA_ARGS`; the Beacon3D numbers above were produced without them.

## Repository layout

```
beacon3d/
├── Beacon3D/              vendored Beacon3D evaluators + data (bda1203, Apache-2.0; see VENDORED.md)
├── build_env.sh           uv venv: torch 2.7.1+cu128, requirements.txt, PointNet++/PointNext ops
├── download.sh            LEO weights, Vicuna-7B, ScanNet + 3RScan scan data from Hugging Face
├── infer.py / run_infer.sh   LEO inference on the Beacon3D QA sets (protocol + no-hint control)
├── score.py / score.sh    Beacon3D evaluate_qa.py with the GPT judge, per domain + overall
└── results/
    ├── <domain>[_nohint]/ infer_qa.json, infer_qa_processed.json, score_qa.txt, scores.json
    ├── overall[_nohint]/  scores.json (pooled and mean)
    ├── preds/             raw infer.py output (one jsonl row per question)
    └── summary.json
extra_eval/
├── download.sh            LEO_data annotations + Mask3D masks, Real-3DQA test jsonl
├── infer.py / run_extra_eval.sh   Scan2Cap / ScanQA (+no-hint) / SQA3D / Real-3DQA inference, one shard per GPU
├── fast_generate.py       prefix-KV-cache generate (--prefix_cache, FAST=1)
├── check_prefix_cache.py  compare two runs: identical outputs + speedup
├── score.py / score.sh    LEO's evaluators + official Real-3DQA scripts
├── real3dqa/              vendored Real-3DQA evaluation scripts (see VENDORED.md)
└── results/               <bench>/scores.json, preds/<bench>.jsonl, summary.json
env.sh.example             every path / setting the scripts read
```

## Setup

### External dependencies (not in this repo)

| What | Where it comes from | Variable |
| --- | --- | --- |
| `sft_noact.pth`, `pointnetpp_vil3dref.pth` | HF dataset `huangjy-pku/LEO_data` | `LEO_ROOT` (via `download.sh`) |
| Vicuna-7B v1.1 | HF model `huangjy-pku/vicuna-7b` | `LEO_ROOT` (via `download.sh`) |
| ScanNet / 3RScan point clouds | `pcd_with_global_alignment.zip`, `3RScan-ours-align.zip` in `LEO_data` | `LEO_ROOT` (via `download.sh`) |
| MultiScan point clouds | [SceneVerse](https://github.com/scene-verse/SceneVerse) `MultiScan/` | `SCENEVERSE_DIR` |
| GPT judge | OpenAI API | `OPENAI_API_KEY` |
| LEO annotations, Mask3D masks | `annotations.zip`, `mask.zip` in `LEO_data` | `LEO_ANNO`, `LEO_ROOT` (via `extra_eval/download.sh`) |
| Real-3DQA test sets | HF dataset of [Real-3DQA](https://arxiv.org/abs/2603.23523) | `REAL3DQA_DIR` (via `extra_eval/download.sh`) |
| Java ≥ 8 (METEOR) | any JRE | `JAVA_HOME` |

### Environment

```bash
cp env.sh.example env.sh        # fill in paths; env.sh is gitignored
bash beacon3d/build_env.sh      # on a GPU node with internet; compiles the point-cloud ops
bash beacon3d/download.sh       # ~20 GB download, ~39 GB in $LEO_ROOT once unpacked
```

`build_env.sh` needs CUDA ≥ 12.8 for Blackwell (`LEO_CUDA_HOME`). It compiles PointNet++
from a temporary copy (upstream `setup.py` hard-codes architectures CUDA 12 cannot target)
and pins `setuptools<81` (`openai-clip` imports `pkg_resources`).

## Running

```bash
# 1. inference on one GPU, LEO protocol then the no-hint control (~40 min in total on one RTX PRO 6000)
bash beacon3d/run_infer.sh                       # -> outputs/beacon3d_<timestamp>/
#    DOMAINS=scannet LIMIT=16 VARIANTS=hint OUT=outputs/smoke bash beacon3d/run_infer.sh   # smoke test

# 2. score with the GPT judge (QA_MODEL, default gpt-4o-mini-2024-07-18)
PREDS_DIR=outputs/beacon3d_<timestamp> bash beacon3d/score.sh
#    -> beacon3d/results/{<domain>,<domain>_nohint,overall,overall_nohint}/, summary.json
```

Scan2Cap / ScanQA / SQA3D / Real-3DQA:

```bash
bash extra_eval/download.sh                      # annotations, Mask3D masks, Real-3DQA
bash extra_eval/run_extra_eval.sh                # all GPUs, -> outputs/extra_eval_<timestamp>/ (~45 min on 2 GPUs)
#    FAST=1 bash extra_eval/run_extra_eval.sh    # prefix cache, ~1.4x faster (see above)
#    BENCHES=scanqa LIMIT=32 OUT=outputs/smoke bash extra_eval/run_extra_eval.sh   # smoke test
PREDS_DIR=outputs/extra_eval_<timestamp> bash extra_eval/score.sh   # CPU; -> extra_eval/results/
```

`score.py` deletes the evaluator's `infer_qa_processed.json` before every call:
Beacon3D's `evaluate_qa.py` silently reuses an existing one instead of re-judging. It also
re-derives the case score from the processed file and asserts it matches the evaluator's
printout.

## Citation

```bibtex
@inproceedings{huang2024embodied,
  title={An Embodied Generalist Agent in 3D World},
  author={Huang, Jiangyong and Yong, Silong and Ma, Xiaojian and Linghu, Xiongkun and Li, Puhao and Wang, Yan and Li, Qing and Zhu, Song-Chun and Jia, Baoxiong and Huang, Siyuan},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2024}
}

@inproceedings{huang2025unveiling,
  title={Unveiling the Mist over 3D Vision-Language Understanding: Object-centric Evaluation with Chain-of-Analysis},
  author={Huang, Jiangyong and Jia, Baoxiong and Wang, Yan and Zhu, Ziyu and Linghu, Xiongkun and Li, Qing and Zhu, Song-Chun and Huang, Siyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## License

LEO is released under the MIT license ([`LICENSE`](LICENSE)). The vendored Beacon3D code
and data are Apache-2.0 ([`beacon3d/Beacon3D/LICENSE`](beacon3d/Beacon3D/LICENSE)).
