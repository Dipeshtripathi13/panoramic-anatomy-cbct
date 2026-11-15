#!/bin/bash
# Train the CycleGAN baseline (configs/train/cyclegan_v1.yaml) on one V100. Extra arguments go to scripts/train_cyclegan.py.
#   sbatch cluster/cyclegan.sbatch
# Writes $PROJ/runs/<run_name>/cyclegan.pt (+ samples_*.png); segmentation on translated images: see cluster/README.md.
#SBATCH --job-name=dental-cyclegan
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
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
python scripts/train_cyclegan.py --config configs/train/cyclegan_v1.yaml "$@" \
  --set out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2))
