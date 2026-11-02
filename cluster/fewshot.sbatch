#!/bin/bash
# Few-shot grid: 2 arms × 6 label fractions = 12 tasks (≈15–25 min each).
#   sbatch --array=0-11%4 cluster/fewshot.sbatch            # INIT defaults to the zero-shot best.pt
# Task i: arm = i / 6 (0 = real-only from ImageNet, 1 = sim→real fine-tune), frac = FRACS[i % 6].
# real_repeat keeps the number of real samples per epoch ≈ 384 for every fraction.
#SBATCH --job-name=dental-fewshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
INIT="${INIT:-$PROJ/runs/seg_v1_dr_full/best.pt}"
SEED="${SEED:-0}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
FRACS=(0.01 0.05 0.10 0.25 0.50 1.00)
i=$SLURM_ARRAY_TASK_ID; frac=${FRACS[$((i % 6))]}
k=$(python -c "import math; print(max(1, math.ceil($frac * 384)))")
rep=$(( (384 + k - 1) / k ))
if [ $((i / 6)) -eq 0 ]; then arm=realonly; init=null; else arm=sim2real; init="$INIT"; [ -f "$init" ]; fi
python scripts/train.py --config configs/train/fewshot.yaml --set \
  run_name="fewshot_${arm}_f${frac}_s${SEED}" real_train_frac=$frac real_repeat=$rep init_from=$init seed=$SEED \
  out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2))
