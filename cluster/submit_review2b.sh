#!/bin/bash
# Reviewer round 2, part b (independent of submit_review2.sh; run from $PROJ/code):  bash cluster/submit_review2b.sh
# 1. Supervised mandible reference: 5-fold CV on the Abdi set, ImageNet vs TDI zero-shot init (seed-matched),
#    all training images vs 5 → 12 tasks × 5 folds, ~1.5 h per task
# 2. CycleGAN seeds 1–2 → segmentation on their translated renders (± image DR) → DENTEX/Tufts/mandible evaluation
# 3. Real→sim test-time baseline with the two new translators (seed 0 translator: MODE=r2s task 0 in submit_review2.sh)
set -euo pipefail
sb() { sbatch --parsable "$@"; }
mand=$(MODE=mandcv sb --array=0-11 cluster/baselines.sbatch)
cyc=$(MODE=cycgan sb --time=24:00:00 --array=0-1 cluster/baselines.sbatch)
seg=$(MODE=cycseg sb --time=06:00:00 --dependency=afterok:$cyc --array=0-3 cluster/baselines.sbatch)
r2s=$(MODE=r2s sb --dependency=afterok:$cyc --array=1-2 cluster/baselines.sbatch)
echo "submitted: mandible CV $mand, CycleGAN seeds $cyc, seg on translated $seg, real→sim $r2s"
