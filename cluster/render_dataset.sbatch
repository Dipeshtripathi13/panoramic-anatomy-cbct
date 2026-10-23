#!/bin/bash
# Render a synthetic dataset on one GPU. All arguments are passed to scripts/render_dataset.py.
# Resumable: resubmitting the same command skips finished samples.
#   sbatch cluster/render_dataset.sbatch --name nominal_test --splits test --variants 0
#   sbatch --time=2-00:00:00 cluster/render_dataset.sbatch --name dr_main --splits train val --variants 8
# Parallel over GPUs with a job array (each task renders every n-th volume; %4 = at most 4 at once):
#   sbatch --array=0-3%4 cluster/render_dataset.sbatch --name dr_main --splits train val --variants 8
# Renders go to $PROJ/renders/<name> (lab storage), not to the 50 GB home.
#SBATCH --job-name=dental-render
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
mkdir -p "$PROJ/renders" data
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
git rev-parse HEAD 2>/dev/null || echo "(code synced by rsync; see LOG.md for commit)"
SHARD=()
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && SHARD=(--shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT")
python scripts/render_dataset.py "$@" "${SHARD[@]}"
