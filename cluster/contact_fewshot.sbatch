#!/bin/bash
# Few-shot contact grid on CONTACT: 2 arms × 6 label fractions = 12 tasks (~15 min each), then CONTACT test evaluation.
#   sbatch --array=0-11%4 cluster/contact_fewshot.sbatch          (SEED=1 sbatch ... for more seeds)
# Task i: arm = i / 6 (0 = real-only from ImageNet, 1 = sim→real from contact_v1), frac = FRACS[i % 6] of 878 train crops.
#SBATCH --job-name=dental-ctfew
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
INIT="${INIT:-$PROJ/runs/contact_v1/best.pt}"
SEED="${SEED:-0}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
FRACS=(0.01 0.05 0.10 0.25 0.50 1.00)
i=$SLURM_ARRAY_TASK_ID; frac=${FRACS[$((i % 6))]}
k=$(python -c "import math; print(max(1, math.ceil($frac * 878)))")
rep=$(( (878 + k - 1) / k ))
if [ $((i / 6)) -eq 0 ]; then arm=realonly; init=null; else arm=sim2real; init="$INIT"; [ -f "$init" ]; fi
run="ctfew_${arm}_f${frac}_s${SEED}"
python scripts/train.py --config configs/train/contact_fewshot.yaml --set \
  run_name="$run" real_train_frac=$frac real_repeat=$rep init_from=$init seed=$SEED \
  out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2))
python scripts/evaluate_contact.py --run "$PROJ/runs/$run" --overlays 0 --num-workers $((SLURM_CPUS_PER_TASK - 2))
