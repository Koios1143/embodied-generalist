#!/bin/bash
# LEO (sft_noact) on the Beacon3D QA sets, single GPU: LEO's protocol, then the no-hint control.
#   bash beacon3d/run_infer.sh
#   DOMAINS=scannet LIMIT=16 bash beacon3d/run_infer.sh   # quick check
# Paths come from env.sh (see env.sh.example):
#   LEO_ENV         venv from build_env.sh                       (default .venv)
#   LEO_ROOT        download target of download.sh               (ckpt/, vicuna-7b/, scannet/, rscan/3RScan-base/)
#   SCENEVERSE_DIR  SceneVerse scan data with MultiScan/
#   OUT             output dir                                   (default outputs/beacon3d_<timestamp>)
#   VARIANTS        "hint nohint" (default) or one of them
#   EXTRA_ARGS      passed to infer.py, e.g. "--prefix_cache" or "--prefix_cache --scene_seed"
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${LEO_ENV:-.venv}/bin/python"
ROOT="${LEO_ROOT:?set LEO_ROOT}"
OUT="${OUT:-outputs/beacon3d_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
echo "out=$OUT"
for v in ${VARIANTS:-hint nohint}; do
    $PY beacon3d/infer.py --beacon_root beacon3d/Beacon3D --out_dir "$OUT" \
        --domains "${DOMAINS:-scannet,3rscan,multiscan}" --limit "${LIMIT:-0}" \
        --batch_size "${BATCH_SIZE:-16}" $([ "$v" = nohint ] && echo --no_hint) ${EXTRA_ARGS:-} \
        --ckpt "$ROOT/ckpt/sft_noact.pth" --llm "$ROOT/vicuna-7b" --pointnetpp "$ROOT/ckpt/pointnetpp_vil3dref.pth" \
        --scannet_base "$ROOT/scannet" --rscan_base "$ROOT/rscan/3RScan-base" --sceneverse_dir "${SCENEVERSE_DIR:?}" || exit 1
done
ls "$OUT"
echo INFER_DONE
