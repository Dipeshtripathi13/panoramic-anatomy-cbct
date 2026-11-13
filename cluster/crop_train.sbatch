#!/bin/bash
# Train the third-molar ROI model (configs/train/crop_v1.yaml) from the zero-shot panoramic model, then evaluate
# it on the CONTACT test set. Extra arguments go to scripts/train.py (e.g. --set run_name=... seed=1).
#   sbatch cluster/crop_train.sbatch
#SBATCH --job-name=dental-crop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
INIT="${INIT:-$PROJ/runs/seg_v1_dr_full/best.pt}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
[ -f "$INIT" ]
log=$(mktemp)
python scripts/train.py --config configs/train/crop_v1.yaml --set init_from="$INIT" "$@" \
  --set out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2)) | tee "$log"
run=$(sed -n 's/.*→ //p' "$log" | tail -n 1)
python scripts/evaluate_contact.py --run "$run" --num-workers $((SLURM_CPUS_PER_TASK - 2))
