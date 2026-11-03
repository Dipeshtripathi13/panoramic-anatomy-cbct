#!/bin/bash
# Train + test-evaluate the paired DR ablation: task 0 = reference (all factors), 1–7 = rendered factor off
# (abl_v2_no_<factor>), 8–9 = image-level factor off. ~45–60 min each.
#   sbatch --array=0-9%4 cluster/ablation_train.sbatch            (SEED=1 sbatch ... for more seeds)
#SBATCH --job-name=dental-abltrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
SEED="${SEED:-0}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
RENDERED=(trough_width blur_outside arch_jitter ghosts spine metal contrast)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -eq 0 ]; then
  name=ref; over=()
elif [ "$i" -le 7 ]; then
  f=${RENDERED[$((i - 1))]}; name="no_$f"; over=(synth_train_sets="[abl_v2_no_$f]" synth_train_variants=null)
elif [ "$i" -eq 8 ]; then
  name=no_noise; over=(image_factors="[processing]")
else
  name=no_processing; over=(image_factors="[noise]")
fi
run="abl_${name}_s${SEED}"
python scripts/train.py --config configs/train/ablation.yaml --set run_name="$run" seed=$SEED ${over[@]+"${over[@]}"} \
  out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2))
python scripts/evaluate.py --run "$PROJ/runs/$run" --overlays 0 --num-workers $((SLURM_CPUS_PER_TASK - 2))
