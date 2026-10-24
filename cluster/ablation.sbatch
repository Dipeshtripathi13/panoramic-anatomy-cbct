#!/bin/bash
# Leave-one-factor-out render sets for the DR ablation (contribution 2).
# One array task = (factor, shard): 7 render-level factors × 4 shards = 28 tasks, at most 4 at once:
#   sbatch --array=0-27%4 cluster/ablation.sbatch
# Each set: train volumes × variants 1..4 with that factor at nominal. Same seeds as dr_main, so
# abl_no_<f>/<case>_v<k> differs from dr_main/<case>_v<k> only in factor <f> (paired ablation).
# Image-level factors (noise, processing) need no renders; they are ablated at training time.
#SBATCH --job-name=dental-abl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
FACTORS=(trough_width blur_outside arch_jitter ghosts spine metal contrast)
NSHARDS=4
F="${FACTORS[$((SLURM_ARRAY_TASK_ID / NSHARDS))]}"
SHARD=$((SLURM_ARRAY_TASK_ID % NSHARDS))
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
mkdir -p "$PROJ/renders" data
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
echo "factor=$F shard=$SHARD/$NSHARDS on $(hostname)"
python scripts/render_dataset.py --name "abl_no_$F" --splits train --variants 4 --no-nominal \
  --disable "$F" --shard "$SHARD" --nshards "$NSHARDS"
