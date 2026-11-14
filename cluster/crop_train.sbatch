#!/bin/bash
# Train a third-molar ROI model, then evaluate it on the CONTACT test set. Extra arguments go to scripts/train.py.
#   sbatch cluster/crop_train.sbatch                                   (crop_v1, from the zero-shot panoramic model)
#   CONFIG=configs/train/contact_v1.yaml INIT=/home/santosh_lab/shared/dipesh/dental/runs/crop_v1/best.pt \
#     sbatch cluster/crop_train.sbatch                                 (contact head, from crop_v1)
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
CONFIG="${CONFIG:-configs/train/crop_v1.yaml}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
[ -f "$INIT" ]
log=$(mktemp)
python scripts/train.py --config "$CONFIG" --set init_from="$INIT" "$@" \
  --set out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2)) | tee "$log"
run=$(sed -n 's/.*→ //p' "$log" | tail -n 1)
python scripts/evaluate_contact.py --run "$run" --num-workers $((SLURM_CPUS_PER_TASK - 2))
