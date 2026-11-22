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
#   MODE=initfew ARM=sim2real|slab INIT=1|2 SEED=s sbatch --array=0,2,3,5 cluster/baselines.sbatch
#                        few-shot from initialisation seed INIT of the TDI (seg_v1_dr_full_s*) or slab (seg_slab_s*)
#                        zero-shot model → fewshot_<ARM>_i<INIT>_f*_s*; INIT=0 are the existing fewshot_<ARM>_f*_s* runs.
#                        DENTEX + Tufts evaluation in the same job (~1 h each)
#   MODE=mtfew     SEED=s sbatch --time=07:00:00 --array=0,2,3,5 cluster/baselines.sbatch
#                        semi-supervised real-only: realonly_v2 recipe + mean teacher on unlabelled DENTEX → realonly_mt_f*_s*
#                        (~3–4 h each), DENTEX + Tufts evaluation in the same job
#   MODE=r2s             sbatch --array=0-2 cluster/baselines.sbatch   NeBLa-style test-time baseline: real DENTEX,
#                        Tufts and mandible images translated to sim style by G_BA of CycleGAN seed i (cyclegan_v1,
#                        _s1, _s2), then segmented by the no-DR and randomized zero-shot models → eval_*_best_r2s{,_g1,_g2}
#   MODE=cycgan          sbatch --time=24:00:00 --array=0-1 cluster/baselines.sbatch   CycleGAN seeds 1–2 (~3–5 h)
#   MODE=cycseg          sbatch --time=06:00:00 --array=0-3 cluster/baselines.sbatch   segmentation on renders translated by
#                        CycleGAN seed 1/2, without (tasks 0–1) and with (2–3) image DR → seg_cyclegan{,_dr}_s{1,2}
#   MODE=mandcv          sbatch --array=0-11 cluster/baselines.sbatch  supervised mandible reference: 5-fold CV on the
#                        Abdi set; task = arm (imagenet | TDI zero-shot seed s) × budget (all | 5 images) × seed s,
#                        5 folds per task (~15 min each) → runs/mandcv_<arm>_n<budget>_s<s>_f<fold>
#   MODE=tufts           sbatch --array=0-5 cluster/baselines.sbatch   Tufts external test: task 0 zero-shot models,
#                        tasks 1–5 few-shot arms (realonly_v2, fewshot_sim2real, ssl_realonly, fewshot_slab, fewshot_nominal)
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
  initfew)  case "$ARM" in sim2real) init="$PROJ/runs/seg_v1_dr_full_s${INIT}/best.pt" ;;
                            slab) init="$PROJ/runs/seg_slab_s${INIT}/best.pt" ;; esac
            [ -f "$init" ]; run="fewshot_${ARM}_i${INIT}_f${frac}_s${SEED}"
            args=(--config configs/train/fewshot.yaml --set real_train_frac=$frac real_repeat=$rep init_from=$init seed=$SEED) ;;
  mtfew)    run="realonly_mt_f${frac}_s${SEED}"
            args=(--config configs/train/realonly_mt.yaml --set real_train_frac=$frac real_repeat=$rep seed=$SEED) ;;
  r2s)      gens=(cyclegan_v1 cyclegan_v1_s1 cyclegan_v1_s2); tags=("" _g1 _g2)
            cg="$PROJ/runs/${gens[$i]}/cyclegan.pt"; [ -f "$cg" ]; tag=${tags[$i]}; nw=$((SLURM_CPUS_PER_TASK - 2))
            for r in seg_nominal_s0 seg_nominal_s1 seg_nominal_s2 seg_v1_dr_full seg_v1_dr_full_s1 seg_v1_dr_full_s2; do
              python scripts/evaluate.py --run "$PROJ/runs/$r" --translate "$cg" --tag "$tag" --overlays 6 --num-workers $nw
              python scripts/evaluate_tufts.py --run "$PROJ/runs/$r" --translate "$cg" --tag "$tag" --num-workers $nw
              python scripts/evaluate_mandible.py --run "$PROJ/runs/$r" --translate "$cg" --tag "$tag" --overlays 0
            done; exit 0 ;;
  cycgan)   SEED=$((i + 1)); python scripts/train_cyclegan.py --config configs/train/cyclegan_v1.yaml \
              --set run_name=cyclegan_v1_s$SEED seed=$SEED out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2))
            exit 0 ;;
  cycseg)   SEED=$(( i % 2 + 1 )); G="$PROJ/runs/cyclegan_v1_s$SEED/cyclegan.pt"; [ -f "$G" ]
            if [ $((i / 2)) -eq 0 ]; then run="seg_cyclegan_s$SEED"; extra=(image_factors='[]')
            else run="seg_cyclegan_dr_s$SEED"; extra=(); fi
            args=(--config configs/train/seg_v1.yaml --set synth_translate=$G seed=$SEED ${extra[@]+"${extra[@]}"}) ;;
  ssl)      python scripts/pretrain_ssl.py --run "$PROJ/runs/ssl_simclr"; exit 0 ;;
  sslfew)   init="$PROJ/runs/ssl_simclr/encoder.pt"; [ -f "$init" ]; run="ssl_realonly_f${frac}_s${SEED}"
            args=(--config configs/train/realonly_v2.yaml --set real_train_frac=$frac real_repeat=$rep init_from=$init seed=$SEED) ;;
  mandible) for r in seg_v1_dr_full seg_v1_dr_full_s1 seg_v1_dr_full_s2 seg_nominal_s0 seg_nominal_s1 seg_nominal_s2 \
                     seg_cyclegan seg_slab_s0 seg_slab_s1 seg_slab_s2; do
              [ -f "$PROJ/runs/$r/best.pt" ] && python scripts/evaluate_mandible.py --run "$PROJ/runs/$r" --overlays 6
            done; exit 0 ;;
  tufts)    arms=(zeroshot realonly_v2 fewshot_sim2real ssl_realonly fewshot_slab fewshot_nominal); a=${arms[$i]}
            if [ "$a" = zeroshot ]; then
              runs="seg_v1_dr_full seg_v1_dr_full_s1 seg_v1_dr_full_s2 seg_nominal_s0 seg_nominal_s1 seg_nominal_s2 seg_cyclegan
                    seg_slab_s0 seg_slab_s1 seg_slab_s2"
            else
              runs=$(for f in "${FRACS[@]}"; do for s in 0 1 2; do echo "${a}_f${f}_s${s}"; done; done)
            fi
            for r in $runs; do
              [ -f "$PROJ/runs/$r/best.pt" ] && python scripts/evaluate_tufts.py --run "$PROJ/runs/$r" \
                --num-workers $((SLURM_CPUS_PER_TASK - 2))
            done; exit 0 ;;
  *) echo "unknown MODE=$MODE"; exit 1 ;;
esac
python scripts/train.py "${args[@]}" --set run_name="$run" out_root="$PROJ/runs" num_workers=$((SLURM_CPUS_PER_TASK - 2))
python scripts/evaluate.py --run "$PROJ/runs/$run" --overlays 0 --num-workers $((SLURM_CPUS_PER_TASK - 2))
if [ "$MODE" = cycseg ]; then
  python scripts/evaluate_mandible.py --run "$PROJ/runs/$run" --overlays 0
fi
if [ "$MODE" = initfew ] || [ "$MODE" = mtfew ] || [ "$MODE" = cycseg ]; then
  python scripts/evaluate_tufts.py --run "$PROJ/runs/$run" --num-workers $((SLURM_CPUS_PER_TASK - 2))
fi
