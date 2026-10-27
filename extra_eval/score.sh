#!/bin/bash
# Score an extra_eval/run_extra_eval.sh output dir (CPU only; METEOR needs Java, see JAVA_HOME in env.sh).
#   PREDS_DIR=outputs/extra_eval_<ts> bash extra_eval/score.sh
# RESULT_DIR defaults to extra_eval/results; BENCHES limits which benchmarks are (re)scored.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${LEO_ENV:-.venv}/bin/python"
$PY extra_eval/score.py --pred_dir "${PREDS_DIR:?set PREDS_DIR}" --result_dir "${RESULT_DIR:-extra_eval/results}" \
    --corpus "${LEO_ANNO:?}/scan2cap/scanrefer_corpus.json" \
    --benches "${BENCHES:-scan2cap,scanqa,scanqa_nohint,sqa3d,real3dqa}"
