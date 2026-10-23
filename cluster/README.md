# Running on Lawrence (USD HPC)

Docs: https://usdrcg.gitbook.io/docs/lawrence-hpc/submitting-jobs

Facts from the docs (2026-10-23):
- **Scheduler:** Slurm.
- **Login node:** `lawrence.usd.edu` (Duo 2FA). It is for editing and submitting only; never compute there.
- **GPUs** (partition `gpu`): GPU001 has 2× P100 16 GB (`--gres=gpu:pascal:1`, add `-B 1:12`);
  GPU002–006 have 1× V100 32 GB each (`--gres=gpu:volta:1`); GPU007–008 have 1× A100 80 GB each (label: see `discover.sh`).
- **Storage:** home is 50 GB. The lab directory is 5 TB; the project and data go there. Per-job `/scratch/job_$SLURM_JOB_ID`
  is 169 GB SSD and is deleted when the job ends. **Nothing is backed up.**

## Layout on the cluster
```
$PROJ/                 # e.g. <lab dir>/dipesh/dental
  code/                # this repo (synced from the Mac)
  data_raw/            # ToothFairy3.zip, CONTACTdataset.zip, Zenodo/*.zip  (read-only, as on the Mac)
  renders/             # synthetic datasets (code/data/renders → here)
  code/logs/           # Slurm logs (submit jobs from $PROJ/code)
  
```

## One-time setup
1. **Cluster (login node):** run `bash discover.sh` (or paste it). It reports the real lab-dir path and the GPU labels.
2. **Mac:** `export REMOTE_PROJ=<lab dir>/dipesh/dental` then `bash cluster/sync_to_cluster.sh code tf3 small`.
   ToothFairy3 is 27 GB, so use a wired connection if possible.
3. **Cluster:**
   ```
   export PROJ=<lab dir>/dipesh/dental      # add this line to ~/.bashrc
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
