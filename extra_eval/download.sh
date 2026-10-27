#!/bin/bash
# Extra inputs of extra_eval/ on top of beacon3d/download.sh (weights, Vicuna, ScanNet scans):
#   bash extra_eval/download.sh
# From HF dataset huangjy-pku/LEO_data: annotations.zip (Scan2Cap / ScanQA / SQA3D in LEO's
# format) -> $LEO_ROOT/annotations, mask.zip (Mask3D proposals) -> $LEO_ROOT/scannet/mask.
# Real-3DQA (HF dataset Oliver-Ma/Real-3DQA, data/test/*.jsonl) -> $REAL3DQA_DIR. Existing files are kept.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
ROOT="${LEO_ROOT:?set LEO_ROOT}"
HF="${HF_CLI:-huggingface-cli}"
$HF download huangjy-pku/LEO_data --repo-type dataset --local-dir "$ROOT/zips" --include annotations.zip mask.zip
[ -d "$ROOT/annotations/instruction" ] || unzip -q "$ROOT/zips/annotations.zip" -d "$ROOT"
[ -d "$ROOT/scannet/mask" ] || unzip -q "$ROOT/zips/mask.zip" -d "$ROOT/scannet"
if [ -n "${REAL3DQA_DIR:-}" ] && [ ! -f "$REAL3DQA_DIR/debiased_test.jsonl" ]; then
    $HF download Oliver-Ma/Real-3DQA --repo-type dataset --local-dir "$REAL3DQA_DIR/hf" --include 'data/test/*.jsonl'
    cp -n "$REAL3DQA_DIR"/hf/data/test/*.jsonl "$REAL3DQA_DIR/"
fi
ls "$ROOT/annotations/instruction" "$ROOT/scannet/mask" | head
echo DOWNLOAD_DONE
