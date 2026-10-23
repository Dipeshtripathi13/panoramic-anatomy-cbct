# LOG

## 2026-10-22 — Session 1

### Step 1: repo setup
- All raw data are zips in the repo root (plus `Zenodo/*.zip`). Free disk ≈ 57 GB; TF2 unzips to ~110 GB, TF3 to ~30 GB.
  **Decision:** never extract; read members straight from the zips (`src/dentalpano/io.py`). Keeps data read-only and uses no disk.
- Env: uv + Python 3.11 `.venv`, torch 2.14.1 (MPS available), nnunetv2 2.8.1, nibabel 5.4.2, SimpleITK 2.5.6. Full pin in `requirements.lock.txt`.
- Hardware: Apple M4, 16 GB RAM, no CUDA. nnU-Net training on MPS will be slow; we may need a GPU machine for contributions 2–4.
- TF3 images are stored float64 (512×512×262 ≈ 550 MB decoded). Cast to float32/int16 on load; limit parallel decoding.
- TF3 header affine = diag(-0.3,-0.3,0.3) → nibabel reads it as LPS. The brief says the true orientation is RPI → check in step 3.
- **Data mismatch:** CONTACT zip has 227×227 RGB crops (third-molar region) + grayscale JPEG masks (molar/canal/overlap gray levels)
  + `{train,test}ImageFeatures.xlsx` (ImageNum, 6 radiographic features, binary `Class`). It has no full 2700×1249 panoramics and no XML polygons.

### Step 2: data audit → `outputs/data_audit.md` (+ per-case CSVs in `outputs/`)
- `scripts/data_audit.py` runs in ~26 s: TF3 headers + full label decode, 5 workers.
- TF3: 532 cases (A 417 / B 63 / C 52 ✔). All 0.3 mm iso, float64 images, uint8 labels, header LPS, image/label affines match.
  Median FOV: A 111×101×51 mm, B 123×123×82 mm, C 111×107×52 mm.
- **Coverage matters for the simulator:** lower jaw, IAC L/R and pharynx are in 100% of cases, incisive canals in ~94%.
  Upper jawbone is in 43%, maxillary sinus in only ~50 cases (44/63 of Set B), and upper pulp mostly in Set B.
  → Full-arch (maxilla + mandible) panoramics are realistic mainly for Set B; Set A/C are mostly mandible-focused.
- Unknown IDs 60, 100, 160 appear in 4 Set C cases (S_0021/0030/0047/0048) → map to background. Odd case name `ToothFairy3S_0000`.
- TF2: 480 cases = exactly TF3 Sets A (P) + B (F), with identical shapes. TF2 adds no cases. **Decision:** ignore TF2.
- CONTACT: contact 1100 / non-contact 378 matches the brief. Masks come in two gray-level encodings (A: 42/213/128, B: 85/170/255);
  decoder in `src/dentalpano/contact.py`. Group 2 vs 3 cannot be recovered exactly: mask overlap gives 72/306 vs expected 101/277.
  15 contact cases have no visible mask overlap.
- DENTEX: 693 + 705 + 634 labelled + 1571 unlabelled train, 50 val (images only). 1 tiny unlabelled image to filter.

### Step 3: orientation check → `outputs/orientation/` (`orientation_check.md`, CSV, 3 PNGs)
- Anatomical tests on all 532 label volumes: L/R (Left vs Right IAC), A/P (incisors vs pharynx),
  S/I (mandible rises posteriorly; upper teeth above lower teeth). Runtime ~5 min.
- **Raw data are RPI** (i→Right, j→Posterior, k→Inferior) in all sets, as the brief says. The header claims LPS.
- **The official Ditto `fix_orientation.py` is wrong for our purpose.** It flips SimpleITK axes 1,2 (= nibabel i,j), which gives LAI:
  A/P fails 100% and the teeth S/I test fails 100%. The result is LPS rotated 180°, so it looks plausible but is upside down and front-back swapped.
- **Decision:** `tf3.fix_orientation` flips i and k → LPS that matches the header. All 532 cases pass every test. Applied in memory only.
- First S/I test (lower teeth vs IAC centroid) was too weak: the IAC climbs into the ramus, so the margin was ~1–15 voxels. Replaced it.
- Caveat: L/R relies on label names, since the anatomy is nearly symmetric. The names agree with the "R" in RPI.

### Step 4: CONTACT loader → `outputs/contact/` (labels.csv, group{1,2,3}.png, contact_without_mask_overlap.png, canal_level_outliers.png)
- Public release check: Kaggle `tugcetoprak92/contact-dataset` v1 (only version, 17.7 MB, CC BY-NC 4.0) = exactly our zip:
  ROIs + masks only. The paper (Sci Rep 2024, PMC11794476) describes full panoramics and XML annotations, but these are not public.
  → Full panoramics would have to be requested from the authors (Tugce Toprak).
- **Decoder bug found via overlays and fixed.** The masks use 5 gray-level families, not 2:
  A 42/212 (+127 overlap), B 85/170 (+255), D 51/255 (+153), E 102/204, F 127/255.
  The first decoder misread D/E/F masks (e.g. train #1414 decoded as ~99% overlap).
  Now: family = best-supported level pair; molar vs canal decided by shape (canal = thinner band, smaller max distance transform).
  Canal level is consistent within each family except 3 masks with swapped levels (train 209, 410, test 1000); shape handles them (checked visually).
- Derived groups: 1100 / 118 / 260 (brief: 1100 / 101 / 277). Still approximate. All E/F masks (48) are non-contact without overlap.
- 15 contact=1 cases have masks that touch but do not overlap → a radiographic "adjacent" case, not a decoding error.

## 2026-10-23 — Session 1 (cont.)

### Step 5: simulator v0 → `outputs/sim_v0/` (per-case npz + png, compare_dentex.png, compare_contact_crops.png, run.json)
- **PX2Tooth** (cloned into git-ignored `external/`; MIT): `Get_Panoramic_Label.py` and `Get_Panoramic_Mean.py` import
  `preprocess.{arch_mask,skeleton,MPR,projection}`, which are **missing from the repo**. Its core cannot be ported verbatim.
  Reimplemented the pipeline it describes: axial projection of the teeth mask → arch curve → samples along the arch →
  slab along the normals (thickness 40 vox) → MPR mean. `Get_Panoramic_Mean.py` instead uses a fixed circular arc and max-projection.
- **NeBLa** repo: model classes only (confirmed). The paper gives the quadratic centre curve f(x)=0.01(x±100)², 21 centres,
  θ = 0.5°/0.6°/1.5°, and 200 samples per ray. The ray geometry is **underspecified**: no initial direction, ray start point,
  or column mapping. My reading (rays tangent to the centre curve) sweeps only ~90° and gives ~100 rays, not 256 columns.
  → Not implemented in v0. Open item.
- **X-Splat constants verified** (arXiv 2607.02099 Table 4, Sec. 2.1, text): K=200, 128×256, β=7.5e-7, p_max=0.25, I0=1,
  Δs=1.35, HU clip [0,4000], ray-length multiplier 0.5, "direct scaling" (β multiplies HU directly), Set B only (60 scans).
  Table 4 labels the angular steps (0.528/0.600/1.509) "rad"; they must be degrees (NeBLa: 0.5°/0.6°/1.5°). Likely a typo in their table.
- **v0 design** (`src/dentalpano/simulator.py`, `configs/sim_v0.yaml`):
  patient-specific arch = 4th-order polynomial fitted to the axial projection of the tooth labels, extended 15 mm, 0.3 mm/column.
  Rays run along the arch normals. Two renderers: MPR mean (±12 mm) and SimPX Beer–Lambert (ray ±0.5·diag/2 ≈ ±43 mm, X-Splat constants).
  Masks: any-hit per channel within the trough (±12 mm; sinus ±20 mm; pharynx whole ray). Overlaps preserved.
  Runtime ~1.5 s/volume. SimPX values 0–0.58, so the X-Splat constants give a sane range.
- Rendered F_010 (32 teeth, no metal), F_005 (bridge, crown, implant), F_020 (crowns, implants). Real references: DENTEX
  quadrant_enumeration train_390/461/582 (seed 0), plus CONTACT ROIs vs simulated molar ROIs (crop placement approximate).
- Visible differences vs real: see Checkpoint 3 summary (FOV, blur, ghosts, spine, open bite, contrast, noise, resolution, metal).
