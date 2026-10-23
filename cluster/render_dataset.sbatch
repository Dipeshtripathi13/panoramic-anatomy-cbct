#!/bin/bash
# Render a synthetic dataset on one GPU. All arguments are passed to scripts/render_dataset.py.
# Resumable: resubmitting the same command skips finished samples.
#   sbatch cluster/render_dataset.sbatch --name nominal_test --splits test --variants 0
#   sbatch --time=2-00:00:00 cluster/render_dataset.sbatch --name dr_main --splits train val --variants 8
# Renders go to $PROJ/renders/<name> (lab storage), not to the 50 GB home.
#SBATCH --job-name=dental-render
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJ="${PROJ:?export PROJ=<project folder> before sbatch (see cluster/README.md)}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
mkdir -p "$PROJ/renders" data
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
git rev-parse HEAD 2>/dev/null || echo "(code synced by rsync; see LOG.md for commit)"
python scripts/render_dataset.py "$@"
