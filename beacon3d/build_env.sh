#!/bin/bash
# LEO inference env as a uv venv, ported to current GPUs: the release pins torch 1.12.1+cu113,
# which has no kernels for Blackwell (sm_120), so this uses torch 2.7.1+cu128 with the rest of
# requirements.txt unchanged, and compiles the PointNet++ ops for the visible GPU.
#   bash beacon3d/build_env.sh
# Run on a GPU node with internet access (compiles CUDA code). Paths come from env.sh:
#   LEO_ENV        where the venv goes                    (default .venv)
#   LEO_CUDA_HOME  CUDA 12.x toolkit, >= 12.8             (default /usr/local/cuda)
#   TORCH_CUDA_ARCH_LIST  GPU archs to compile for        (default: the visible GPU's)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
UV="${UV:-uv}"
ENV="${LEO_ENV:-.venv}"
export CUDA_HOME="${LEO_CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-16}"

$UV python install 3.9
[ -x "$ENV/bin/python" ] || $UV venv "$ENV" -p 3.9 --seed
PY="$ENV/bin/python"
PIP="$UV pip install --python $PY"
$PIP torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$PIP -r requirements.txt
$PIP peft==0.5.0 --no-deps   # as in the README, to escape its install_requires
$PIP openai   # Beacon3D QA judge (beacon3d/score.py)
$PIP "setuptools<81"   # openai-clip imports pkg_resources, removed in setuptools 81
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$($PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')}"
echo "CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# PointNet++ ops (top-level package `pointnet2._ext`). Upstream setup.py hard-codes
# TORCH_CUDA_ARCH_LIST to archs CUDA 12 cannot target, so build a copy that keeps ours.
BUILD="$(mktemp -d)"
cp -r model/pointnetpp "$BUILD/"
sed -i 's/^os.environ\["TORCH_CUDA_ARCH_LIST"\] = .*/os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "")/' "$BUILD/pointnetpp/setup.py"
(cd "$BUILD/pointnetpp" && $PY setup.py install)
rm -rf "$BUILD"
# PointNext ops: unused by LEO's default PointNet++, but model/pcd_backbone.py imports them;
# built in place as in the README (the .so is gitignored).
(cd model/pointnext/cpp/pointnet2_batch && $PY setup.py build_ext --inplace && rm -rf build)

PYTHONPATH=. $PY -c "
import torch, transformers, peft, accelerate
import pointnet2._ext
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('transformers', transformers.__version__, 'peft', peft.__version__, 'accelerate', accelerate.__version__)
from model.pointnetpp.pointnetpp import PointNetPP; from model.leo_agent import LeoAgent; import data.datasets
print('leo imports ok')
"
echo ENV_DONE
