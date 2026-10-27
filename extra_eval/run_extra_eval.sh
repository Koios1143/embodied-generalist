#!/bin/bash
# LEO (sft_noact) on Scan2Cap, ScanQA (+ no-hint control), SQA3D and Real-3DQA, one shard per GPU.
#   bash extra_eval/run_extra_eval.sh
#   BENCHES=scanqa LIMIT=64 bash extra_eval/run_extra_eval.sh      # quick check
#   FAST=1 bash extra_eval/run_extra_eval.sh                        # scene-sorted + prefix KV cache
# Paths come from env.sh (see env.sh.example):
#   LEO_ENV, LEO_ROOT (ckpt/, vicuna-7b/, scannet/{scan_data,mask}), LEO_ANNO, REAL3DQA_DIR
# Options:
#   OUT          output dir (default outputs/extra_eval_<timestamp>)
#   BENCHES      default scan2cap,scanqa,sqa3d,real3dqa_{debiased_test,rotation_0,_90,_180,_270}
#   NOHINT       1 (default) also runs the ScanQA no-hint control (when scanqa is in BENCHES)
#   PC_TYPE      pred (default, LEO's Mask3D proposals) or gt
#   NUM_GPUS     shards (default: visible GPUs);  BATCH_SIZE (default 16);  LIMIT (debug)
#   FAST         1 = --scene_sort --prefix_cache (same inputs, prefix prefilled once)
#   EXTRA_ARGS   passed through to infer.py (e.g. --scene_seed)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${LEO_ENV:-.venv}/bin/python"
ROOT="${LEO_ROOT:?set LEO_ROOT}"
OUT="${OUT:-outputs/extra_eval_$(date +%Y%m%d_%H%M%S)}"
BENCHES="${BENCHES:-scan2cap,scanqa,sqa3d,real3dqa_debiased_test,real3dqa_rotation_0,real3dqa_rotation_90,real3dqa_rotation_180,real3dqa_rotation_270}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
FAST_ARGS=$([ "${FAST:-0}" = 1 ] && echo --scene_sort --prefix_cache)
mkdir -p "$OUT"
echo "out=$OUT gpus=$NUM_GPUS fast=${FAST:-0}"

run() {  # $1 = benches, rest = extra infer.py args
    local benches=$1 pids=() ok=0; shift
    for k in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$k $PY extra_eval/infer.py --benches "$benches" --out_dir "$OUT" \
            --shard $k --num_shards "$NUM_GPUS" --pc_type "${PC_TYPE:-pred}" \
            --batch_size "${BATCH_SIZE:-16}" --limit "${LIMIT:-0}" \
            --ckpt "$ROOT/ckpt/sft_noact.pth" --llm "$ROOT/vicuna-7b" --pointnetpp "$ROOT/ckpt/pointnetpp_vil3dref.pth" \
            --scannet_base "$ROOT/scannet" --anno "${LEO_ANNO:?}" --real3dqa_dir "${REAL3DQA_DIR:-}" \
            $FAST_ARGS ${EXTRA_ARGS:-} "$@" > "$OUT/infer_$(echo "$benches$*" | tr -c 'a-z0-9_' _ | cut -c1-60)_shard$k.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || ok=1; done
    return $ok
}

run "$BENCHES" || exit 1
if [[ ",$BENCHES," == *,scanqa,* && "${NOHINT:-1}" = 1 ]]; then
    run scanqa --no_hint || exit 1
fi
ls "$OUT"
grep -h "items," "$OUT"/infer_*.log
echo INFER_DONE
