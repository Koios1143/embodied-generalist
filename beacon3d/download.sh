#!/bin/bash
# Download LEO's released weights and the scan data the Beacon3D QA run needs into $LEO_ROOT:
#   bash beacon3d/download.sh
# From HF dataset huangjy-pku/LEO_data: sft_noact.pth (instruction-tuned LEO), the PointNet++
# backbone, ScanNet point clouds (pcd_with_global_alignment) and 3RScan-ours-align; from HF
# model huangjy-pku/vicuna-7b: the Vicuna-7B v1.1 LLM. Paths come from env.sh:
#   LEO_ROOT  download / unpack target          HF_CLI  huggingface-cli binary (default: on PATH)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
ROOT="${LEO_ROOT:?set LEO_ROOT}"
HF="${HF_CLI:-huggingface-cli}"
mkdir -p "$ROOT"/{ckpt,vicuna-7b,scannet/scan_data,rscan}
$HF download huangjy-pku/LEO_data --repo-type dataset --local-dir "$ROOT/ckpt" \
    --include sft_noact.pth pointnetpp_vil3dref.pth
$HF download huangjy-pku/vicuna-7b --local-dir "$ROOT/vicuna-7b"
$HF download huangjy-pku/LEO_data --repo-type dataset --local-dir "$ROOT/zips" \
    --include pcd_with_global_alignment.zip 3RScan-ours-align.zip
[ -d "$ROOT/scannet/scan_data/pcd_with_global_alignment" ] || unzip -q "$ROOT/zips/pcd_with_global_alignment.zip" -d "$ROOT/scannet/scan_data"
[ -d "$ROOT/rscan/3RScan-base/3RScan-ours-align" ] || unzip -q "$ROOT/zips/3RScan-ours-align.zip" -d "$ROOT/rscan"
ls "$ROOT"/ckpt "$ROOT"/vicuna-7b "$ROOT"/scannet/scan_data "$ROOT"/rscan
echo DOWNLOAD_DONE
