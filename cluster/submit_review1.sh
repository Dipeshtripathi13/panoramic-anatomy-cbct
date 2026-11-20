#!/bin/bash
# Reviewer round 1 experiments, chained with Slurm dependencies. Run on the cluster from $PROJ/code:
#   bash cluster/submit_review1.sh
# 1. curved-slab renders (4 shards) → slab zero-shot seeds 0–2 → slab few-shot (3 seeds × 6 fractions)
# 2. SimCLR pretraining on unlabelled DENTEX → SSL real-only few-shot (3 seeds × 6 fractions)
# 3. mandible external test of every zero-shot model (after the slab models exist)
set -euo pipefail
sb() { sbatch --parsable "$@"; }
render=$(sb --array=0-3%4 cluster/render_dataset.sbatch --name slab_v1 --beam slab --splits train val --variants 8)
slab=$(MODE=slab sb --dependency=afterok:$render --array=0-2 cluster/baselines.sbatch)
for s in 0 1 2; do MODE=slabfew SEED=$s sb --dependency=afterok:$slab --array=0-5 cluster/baselines.sbatch; done
ssl=$(MODE=ssl sb --array=0 cluster/baselines.sbatch)
for s in 0 1 2; do MODE=sslfew SEED=$s sb --dependency=afterok:$ssl --array=0-5 cluster/baselines.sbatch; done
MODE=mandible sb --dependency=afterok:$slab --array=0 cluster/baselines.sbatch
echo "submitted: render $render, slab $slab, ssl $ssl (+ few-shot and mandible jobs)"
