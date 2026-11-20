#!/bin/bash
# Reviewer baselines, each task trains then test-evaluates. MODE (env) selects the experiment:
#   MODE=realonly  SEED=s sbatch --array=0-5 cluster/baselines.sbatch   retuned real-only, 6 label fractions (~2 h each)
#   MODE=nominal         sbatch --array=0-2 cluster/baselines.sbatch   no-DR zero-shot model, seeds 0–2 (~2.5 h each)
#   MODE=nomfew    SEED=s sbatch --array=0-5 cluster/baselines.sbatch   few-shot from seg_nominal_s0 (fewshot.yaml schedule)
#   MODE=s2rlong   SEED=s sbatch --array=0-5 cluster/baselines.sbatch   long-schedule sim→real from seg_v1_dr_full (~2 h each)
#   MODE=slab            sbatch --array=0-2 cluster/baselines.sbatch   curved-slab zero-shot model, seeds 0–2 (~2.5 h each)
#   MODE=slabfew   SEED=s sbatch --array=0-5 cluster/baselines.sbatch   few-shot from seg_slab_s0 (fewshot.yaml schedule)
#   MODE=ssl             sbatch --array=0   cluster/baselines.sbatch   SimCLR on unlabelled DENTEX → runs/ssl_simclr
#   MODE=sslfew    SEED=s sbatch --array=0-5 cluster/baselines.sbatch   real-only (tuned) from the SSL encoder (~2 h each)
#   MODE=mandible        sbatch --array=0   cluster/baselines.sbatch   mandible external test of every zero-shot model
#SBATCH --job-name=dental-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
PROJ="${PROJ:-/home/santosh_lab/shared/dipesh/dental}"
SEED="${SEED:-0}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate dental
export DENTAL_DATA_ROOT="$PROJ/data_raw"
cd "$PROJ/code"
[ -e data/renders ] || ln -s "$PROJ/renders" data/renders
i=$SLURM_ARRAY_TASK_ID
FRACS=(0.01 0.05 0.10 0.25 0.50 1.00)
frac=${FRACS[$i]:-}
if [ -n "$frac" ]; then
  k=$(python -c "import math; print(max(1, math.ceil($frac * 384)))"); rep=$(( (384 + k - 1) / k ))
fi
case "$MODE" in
  realonly) run="realonly_v2_f${frac}_s${SEED}"
            args=(--config configs/train/realonly_v2.yaml --set real_train_frac=$frac real_repeat=$rep seed=$SEED) ;;
  nominal)  SEED=$i; run="seg_nominal_s${SEED}"
            args=(--config configs/train/seg_nominal.yaml --set seed=$SEED) ;;
  nomfew)   init="$PROJ/runs/seg_nominal_s0/best.pt"; [ -f "$init" ]; run="fewshot_nominal_f${frac}_s${SEED}"
            args=(--config configs/train/fewshot.yaml --set real_train_frac=$frac real_repeat=$rep init_from=$init seed=$SEED) ;;
  s2rlong)  init="$PROJ/runs/seg_v1_dr_full/best.pt"; [ -f "$init" ]; run="sim2real_v2_f${frac}_s${SEED}"
            args=(--config configs/train/sim2real_v2.yaml --set real_train_frac=$frac real_repeat=$rep init_from=$init seed=$SEED) ;;
  slab)     SEED=$i; run="seg_slab_s${SEED}"
            args=(--config configs/train/seg_slab.yaml --set seed=$SEED) ;;
  slabfew)  init="$PROJ/runs/seg_slab_s0/best.pt"; [ -f "$init" ]; run="fewshot_slab_f${frac}_s${SEED}"
            args=(--config configs/train/fewshot.yaml --set real_train_frac=$frac real_repeat=$rep init_from=$init seed=$SEED) ;;
  ssl)      python scripts/pretrain_ssl.py --run "$PROJ/runs/ssl_simclr"; exit 0 ;;
  sslfew)   init="$PROJ/runs/ssl_simclr/encoder.pt"; [ -f "$init" ]; run="ssl_realonly_f${frac}_s${SEED}"
            args=(--config configs/train/realonly_v2.yaml --set real_train_frac=$frac real_repeat=$rep init_from=$init seed=$SEED) ;;
  mandible) for r in seg_v1_dr_full seg_v1_dr_full_s1 seg_v1_dr_full_s2 seg_nominal_s0 seg_nominal_s1 seg_nominal_s2 \
                     seg_cyclegan seg_slab_s0 seg_slab_s1 seg_slab_s2; do
              [ -f "$PROJ/runs/$r/best.pt" ] && python scripts/evaluate_mandible.py --run "$PROJ/runs/$r" --overlays 6
            done; exit 0 ;;
  *) echo "unknown MODE=$MODE"; exit 1 ;;
esac
python scripts/train.py "${args[@]}" --set run_name="$run" out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2))
python scripts/evaluate.py --run "$PROJ/runs/$run" --overlays 0 --num-workers $((SLURM_CPUS_PER_TASK - 2))
