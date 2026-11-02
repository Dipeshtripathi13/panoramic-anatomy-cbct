#!/bin/bash
# Evaluate a trained run on the held-out TEST sets (DENTEX teeth + TF3 nominal renders). Arguments go to scripts/evaluate.py.
#   sbatch cluster/evaluate.sbatch --run $PROJ/runs/seg_v1_dr_full
# Writes <run>/eval_best/: metrics.json, real_per_image.csv, overlays_real.png
#SBATCH --job-name=dental-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
python scripts/evaluate.py "$@" --num-workers $((SLURM_CPUS_PER_TASK - 2))
