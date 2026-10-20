#!/bin/bash
# Score a beacon3d/run_infer.sh output dir with Beacon3D's QA evaluator (CPU only).
#   PREDS_DIR=outputs/beacon3d_<ts> bash beacon3d/score.sh
# OPENAI_API_KEY (env.sh) is needed for the GPT judge (QA_MODEL, default
# gpt-4o-mini-2024-07-18); RESULT_DIR defaults to beacon3d/results.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${LEO_ENV:-.venv}/bin/python"
$PY beacon3d/score.py --pred_dir "${PREDS_DIR:?set PREDS_DIR}" --result_dir "${RESULT_DIR:-beacon3d/results}" \
    --qa_model "${QA_MODEL:-gpt-4o-mini-2024-07-18}" --variants "${VARIANTS:-,_nohint}"
