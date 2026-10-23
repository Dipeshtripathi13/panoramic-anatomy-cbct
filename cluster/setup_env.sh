#!/bin/bash
# Run ONCE on the cluster, inside an interactive job (not on the login node):
#   srun --pty -p short --cpus-per-task=4 --mem=16G bash
#   bash $PROJ/code/cluster/setup_env.sh
# Creates conda env "dental" (Python 3.11) with the project's pinned packages and a CUDA build of torch.
# Lawrence GPUs are P100 (sm_60), V100 (sm_70) and A100 (sm_80). PyTorch's CUDA 12.8+ wheels dropped
# sm_60/sm_70, so torch comes from the CUDA 12.6 index (override with TORCH_INDEX / TORCH_SPEC).
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
CONDA="$HOME/miniforge3/bin/conda"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.14.1 torchvision==0.29.1}"

"$CONDA" create -y -n dental python=3.11
ENV_PY="$("$CONDA" run -n dental which python)"
"$ENV_PY" -m pip install --upgrade pip
# torch first, from the CUDA index; fall back to the newest torch that index has
if ! "$ENV_PY" -m pip install $TORCH_SPEC --index-url "$TORCH_INDEX"; then
  echo "!! $TORCH_SPEC not on $TORCH_INDEX — installing newest available there (record the version in LOG.md)"
  "$ENV_PY" -m pip install torch torchvision --index-url "$TORCH_INDEX"
fi
# everything else pinned as on the Mac (minus torch/torchvision)
grep -viE '^(torch|torchvision)==' "$CODE/requirements.lock.txt" > /tmp/req_noTorch_$$.txt
"$ENV_PY" -m pip install -r /tmp/req_noTorch_$$.txt
"$ENV_PY" -m pip install -e "$CODE"
"$ENV_PY" -m pip freeze > "$CODE/cluster/requirements.cluster.lock.txt"
"$ENV_PY" -c "import torch, dentalpano; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'archs', torch.cuda.get_arch_list())"
echo "env ready: conda activate dental"
