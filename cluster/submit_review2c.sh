#!/bin/bash
# Reviewer round 2, part c (run from $PROJ/code):  bash cluster/submit_review2c.sh
# 1. Supervised mandible reference (the MODE=mandcv branch was missing from baselines.sbatch in part b): 12 tasks × 5 folds
# 2. Tufts + mandible test of the seed-0 CycleGAN model trained with image DR (seg_cyclegan_dr), not evaluated before
set -euo pipefail
sb() { sbatch --parsable "$@"; }
mand=$(MODE=mandcv sb --array=0-11 cluster/baselines.sbatch)
ext=$(MODE=exteval RUNS="seg_cyclegan_dr" sb --time=01:00:00 --array=0 cluster/baselines.sbatch)
echo "submitted: mandible CV $mand, external test $ext"
