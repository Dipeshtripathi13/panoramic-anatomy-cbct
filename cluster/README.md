# Running on Lawrence (USD HPC)

Docs: https://usdrcg.gitbook.io/docs/lawrence-hpc/submitting-jobs

Facts from the docs and from `sinfo` on 2026-10-23:
- **Scheduler:** Slurm.
- **Login node:** `lawrence.usd.edu` (Duo 2FA). It is for editing and submitting only; never compute there.
- **GPUs** (partition `gpu`, **max 2 days per job**): gpu002–006 have 1× V100 32 GB each (`--gres=gpu:volta:1`, the scripts' default);
  gpu001 has 2× P100 16 GB (`--gres=gpu:pascal:1 --cpus-per-task=12`; the docs' `-B 1:12` is rejected together with our scripts' CPU request). The docs list A100 nodes (gpu007/008), but `sinfo` does not show them.
- **Storage:** home is 50 GB. The lab directory `/home/santosh_lab` is 5 TB (2.1 TB free); only `shared/` is writable,
  so **PROJ = /home/santosh_lab/shared/dipesh/dental** (the scripts' default). Per-job `/scratch/job_$SLURM_JOB_ID`
  is 169 GB SSD and is deleted when the job ends. **Nothing is backed up.**

## Layout on the cluster
```
$PROJ/                 # /home/santosh_lab/shared/dipesh/dental
  code/                # this repo (synced from the Mac)
  data_raw/            # ToothFairy3.zip, CONTACTdataset.zip, Zenodo/*.zip  (read-only, as on the Mac)
  renders/             # synthetic datasets (code/data/renders → here)
  code/logs/           # Slurm logs (submit jobs from $PROJ/code)
  
```

## One-time setup
1. Done: `discover.sh` facts are recorded above.
2. **Mac:** `bash cluster/sync_to_cluster.sh code tf3 small`.
   ToothFairy3 is 27 GB, so use a wired connection if possible.
3. **Cluster:**
   ```
   export PROJ=/home/santosh_lab/shared/dipesh/dental      # add this line to ~/.bashrc
   srun --pty -p short --cpus-per-task=4 --mem=16G bash
   bash $PROJ/code/cluster/setup_env.sh     # builds conda env "dental"
   exit
   ```
4. **Smoke test:**
   ```
   cd $PROJ/code && mkdir -p logs
   sbatch cluster/gpu_smoketest.sbatch
   squeue -u $USER
   cat logs/dental-smoke-<jobid>.out
   ```

## Everyday commands
| what | command |
|---|---|
| submit | `sbatch cluster/render_dataset.sbatch --name nominal_test --splits test --variants 0` |
| my jobs | `squeue -u $USER` |
| cancel | `scancel <jobid>` |
| finished job info | `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` |
| interactive GPU | `srun --pty -p gpu --gres=gpu:volta:1 --cpus-per-task=8 --mem=32G bash` |
| update code after editing on the Mac | Mac: `bash cluster/sync_to_cluster.sh code` |
| train (after renders exist) | `sbatch cluster/train.sbatch --config configs/train/seg_v1.yaml` |
| one-time encoder weight cache (login node) | `conda activate dental && python -c "import segmentation_models_pytorch as smp; smp.Unet('resnet34', encoder_weights='imagenet')"` |

## CycleGAN baseline
```bash
jid=$(sbatch --parsable cluster/cyclegan.sbatch)                       # ~3-5 h
G=$PROJ/runs/cyclegan_v1/cyclegan.pt
sbatch --dependency=afterok:$jid cluster/train.sbatch --config configs/train/seg_v1.yaml \
  --set run_name=seg_cyclegan image_factors='[]' synth_translate=$G      # CycleGAN only
sbatch --dependency=afterok:$jid cluster/train.sbatch --config configs/train/seg_v1.yaml \
  --set run_name=seg_cyclegan_dr synth_translate=$G                      # image DR + CycleGAN
```
