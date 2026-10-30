#!/bin/bash
# Train a segmentation model on one V100. Arguments go to scripts/train.py.
#   sbatch cluster/train.sbatch --config configs/train/seg_v1.yaml
#   sbatch cluster/train.sbatch --config configs/train/seg_v1.yaml --set run_name=smoke epochs=1 max_train_batches=20
# Runs are written to $PROJ/runs/<run_name>/ (config.json with git hash, log.csv, best.pt, last.pt).
# One-time, on the LOGIN node (compute nodes may have no internet): cache the ImageNet encoder weights:
#   conda activate dental && python -c "import segmentation_models_pytorch as smp; smp.Unet('resnet34', encoder_weights='imagenet')"
#SBATCH --job-name=dental-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
mkdir -p "$PROJ/renders" "$PROJ/runs" data
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/train.py "$@" --set out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2))
