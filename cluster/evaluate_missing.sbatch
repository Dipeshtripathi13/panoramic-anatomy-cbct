#!/bin/bash
# Test-evaluate every run under $PROJ/runs whose best.pt has no eval_best/metrics.json yet (one GPU, sequential),
# then write $PROJ/runs/summary.csv.
#   sbatch cluster/evaluate_missing.sbatch [PATTERN]        (PATTERN defaults to '*', e.g. 'fewshot_*')
#SBATCH --job-name=dental-evalall
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
for run in "$PROJ"/runs/${1:-*}/; do
  run=${run%/}
  [ -f "$run/best.pt" ] && [ ! -f "$run/eval_best/metrics.json" ] || continue
  echo "== $run"
  python scripts/evaluate.py --run "$run" --overlays 0 --num-workers $((SLURM_CPUS_PER_TASK - 2))
done
python scripts/collect_results.py --runs "$PROJ/runs"
