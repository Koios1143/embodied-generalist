# LEO Benchmark Evaluation

This is a fork of [LEO](https://github.com/embodied-generalist/embodied-generalist)
("An Embodied Generalist Agent in 3D World", ICML 2024) whose purpose is to **evaluate the
released LEO checkpoint on benchmarks the paper does not report**, under one reproducible
setup, so its scores can be compared with other models on equal footing. Wired in so far:
the QA half of [Beacon3D](https://beacon-3d.github.io/) (CVPR 2025; ScanNet, 3RScan,
MultiScan). LEO generates text only (it has no object-ID or box output), so Beacon3D
grounding is not evaluated.

The upstream model, training and data code are untouched apart from a one-line torch-2
compatibility fix in `common/misc.py`; everything this fork adds lives under
[`beacon3d/`](beacon3d/), and the per-question outputs behind the numbers below are
committed in [`beacon3d/results/`](beacon3d/results/).

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

### Read this before quoting the numbers

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
