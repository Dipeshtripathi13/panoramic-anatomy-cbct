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
