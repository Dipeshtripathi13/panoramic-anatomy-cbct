#!/bin/bash
# Full-schedule zero-shot comparison, 3 seeds each: tasks 0–2 = seg_v2 (retuned DR), 3–4 = seg_v1 seeds 1–2
# (seg_v1 seed 0 = seg_v1_dr_full, already trained). Train + test-evaluate; ~2.5 h each.
#   sbatch --array=0-4 cluster/seg_v2.sbatch        (seg_v2 needs render set dr_main_v3 first)
#SBATCH --job-name=dental-segv2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -le 2 ]; then cfg=seg_v2; seed=$i; run="seg_v2_dr_tuned_s$seed"
else cfg=seg_v1; seed=$((i - 2)); run="seg_v1_dr_full_s$seed"; fi
python scripts/train.py --config configs/train/$cfg.yaml --set run_name="$run" seed=$seed \
  out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2))
python scripts/evaluate.py --run "$PROJ/runs/$run" --num-workers $((SLURM_CPUS_PER_TASK - 2))
