#!/bin/bash
# GPU smoke test (~10 min): checks CUDA works on this GPU type and times 2 volumes × 3 renders.
#   sbatch cluster/gpu_smoketest.sbatch                       # any GPU
#   sbatch --gres=gpu:volta:1 cluster/gpu_smoketest.sbatch    # force a V100
#SBATCH --job-name=dental-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJ="${PROJ:?export PROJ=<project folder> before sbatch (see cluster/README.md)}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
mkdir -p "$PROJ/renders" data
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
hostname; nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
time python scripts/render_dataset.py --name smoke_$SLURM_JOB_ID --splits train --variants 2 --limit 2
