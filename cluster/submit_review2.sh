#!/bin/bash
# Reviewer round 2 experiments. Run on the cluster from $PROJ/code:  bash cluster/submit_review2.sh
# 1. Initialisation seeds: few-shot from TDI and slab zero-shot seeds 1 and 2 (seed 0 = existing runs), budgets
#    4/39/96/384 images (array 0,2,3,5), fine-tuning seeds 0–2 → 48 jobs, ~1 h each incl. DENTEX + Tufts evaluation
# 2. Mean-teacher real-only baseline at the same budgets, seeds 0–2 → 12 jobs, ~3–4 h each
# 3. Real→sim CycleGAN test-time baseline (NeBLa-style) → 1 job, ~1 h
set -euo pipefail
sb() { sbatch --parsable "$@"; }
ids=()
for arm in sim2real slab; do
  for init in 1 2; do
    for s in 0 1 2; do
      ids+=($(MODE=initfew ARM=$arm INIT=$init SEED=$s sb --array=0,2,3,5 cluster/baselines.sbatch))
    done
  done
done
for s in 0 1 2; do ids+=($(MODE=mtfew SEED=$s sb --time=07:00:00 --array=0,2,3,5 cluster/baselines.sbatch)); done
ids+=($(MODE=r2s sb --array=0 cluster/baselines.sbatch))
echo "submitted ${#ids[@]} array jobs: ${ids[*]}"
