# Project: Learning panoramic anatomy from CBCT

## Goal
Train segmentation models for **real** dental panoramic X-rays using only **synthetic** panoramics
rendered from labelled CBCT (ToothFairy3), with 3D labels projected to 2D. Evaluate sim-to-real
zero-shot and few-shot.

## Planned contributions
1. Open, physics-based panoramic simulator over ToothFairy3 that projects **multi-label** (overlapping)
   masks: inferior alveolar canal (IAC), incisive canals, maxillary sinus, pulp, teeth, crowns, bridges, implants.
2. Panoramic-specific domain randomization with a per-factor ablation: focal-trough width, arch-curve jitter,
   blur outside the trough, ghost images, spine shadow, metal artifacts, noise, contrast.
3. Sim-to-real evaluation on real panoramics vs. baselines: real-only training at 1–100% labels;
   CycleGAN image translation (NeBLa-style).
4. 3D-relation head: predict true third-molar-root ↔ IAC **contact** from the panoramic alone; labels computed
   automatically from each CBCT.

Prior work to go beyond: PXseg (teeth only, CBCT-pretrained); NeBLa / X-Splat / HyDAR-Pano3D (synthetic pano
for 3D reconstruction); SyntheX, AnyCXR (sim-to-real in chest/hip).

## Data — READ-ONLY, never modify/move/extract next to originals. Never commit data/renders/checkpoints.
All raw data are zips in the repo root; paths in `configs/paths.yaml`; read via `src/dentalpano/io.py`
(streams members from the zip, decodes in memory).
- **ToothFairy3** `ToothFairy3.zip`: 532 NIfTI (nnU-Net layout `imagesTr/*_0000.nii.gz`, `labelsTr/*.nii.gz`),
  HU, 0.3 mm iso, 77 classes. Set A = `P` (417), B = `F` (63, full face), C = `S` (52, other scanner).
  Header affine says LPS but true orientation is **RPI** (brief; conversion script on Ditto site) — convert
  in memory only. Images stored float64 (~550 MB decoded) → cast immediately. Pulp ID = tooth ID + 100;
  103/104 = incisive canals, 105 = lingual canal.
- **ToothFairy2** `ToothFairy2_Dataset.zip` (.mha, ~110 GB uncompressed — do not extract). Sets A/B overlap
  TF3 → do not double count; prefer TF3 labels.
- **CONTACT** `CONTACTdataset.zip`: brief describes full panoramics (2700×1249) + XML polygons, but **this copy
  contains 227×227 crops around the third molar with grayscale JPEG masks** (see LOG.md). Train 1034 / Test 444.
  Groups: 1 = overlap+contact, 2 = no overlap/no contact, 3 = overlap/no contact (key hard case).
- **DENTEX** `Zenodo/{training,validation}_data.zip`: real panoramics + quadrant/enumeration/diagnosis boxes;
  unlabelled real images + optional teeth test set.
- Licences mostly CC BY-NC-SA (TF3 dataset.json says CC-BY-SA 4.0).

## Simulator starting points
PX2Tooth repo (MIT) `data_process/` (CBCT→pano projection, labels, arch curves); NeBLa SimPX focal-trough
Beer–Lambert rays (reimplement from paper); X-Splat (arXiv 2607.02099 App. B.2) SimPX constants on TF3:
K=200 samples/ray, 128×256, β=7.5e-7, p_max=0.25, HU clip [0,4000] — **verify before relying on them**;
DiffDRR optional.

## Working rules
- Python 3.11 venv at `.venv` (uv); pinned in `requirements.lock.txt`; package `dentalpano` in `src/` (editable).
- Machine: Apple M4, 16 GB RAM, MPS only (no CUDA) — plan compute accordingly.
- Patient-level splits fixed once in `splits/`, never changed. Fix all seeds. Log each experiment's config + git hash.
- Ask before any job > ~30 min or any large download.
- Keep `LOG.md` updated (decisions, problems, results). Outputs go in `outputs/` (git-ignored).
- Work in phases; stop for review at each checkpoint.

## First-session plan
1. Repo setup ✔  2. Data audit → `outputs/data_audit.md` (**Checkpoint 1**)  3. Orientation check (PNG slices,
one volume per set)  4. CONTACT loader + overlays (5/group) (**Checkpoint 2**)  5. Simulator v0: PX2Tooth
projection + SimPX rays, 3 Set B renders with multi-label masks next to real panoramics; list differences
(**Checkpoint 3**).
