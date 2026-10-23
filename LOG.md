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
