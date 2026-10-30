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

### Step 6: rotational OPG simulator v1 → `src/dentalpano/opg.py`, `configs/opg_v1.yaml`, `outputs/opg_v1/`, `outputs/opg_phantom/`
- **Model.** Time-delay-integration (TDI) panoramic. Per time step (0.3 mm of travel along the arch):
  - Central ray through arch point A(s), tilted up 5°. Source 400 mm behind the trough, detector 100 mm in front (M = 1.25).
  - 6 mm vertical slit; every slit pixel casts its own ray.
  - Detector advances M·cos δ per step, so trough points stay fixed and everything else drifts.
  - Transmission is averaged over the exposure and shown as −log. μ = 0.02/mm·(1+HU/1000), clipped at 0 (physical mode); SimPX constants are an option.
  - Torch `grid_sample` on MPS: ~10–14 s per volume (vs 1.5 s for v0).
  - Masks use a per-ray depth window around the focal point, at ≥50% of the exposure, and keep overlaps.
- **Bead-phantom validation** (`scripts/opg_phantom.py`, one render per bead, F_010 arch):
  - **Bug found:** first measurement used overlapping windows → garbage. Fixed by rendering each bead separately.
  - **Normal beams (v0 geometry):** posterior beams are almost pure translation → no depth blur (beads sharp at ±15 mm), no ghosts. Unrealistic.
  - **Fixed r profile (front 30 → back 60 mm, then 20 → 45):** trough narrow anteriorly, wide posteriorly, lingual blurs faster. But beams drift 20–26° oblique posteriorly. On real volumes that smeared every tooth by ~4 mm horizontally (a tooth's buccal and lingual surfaces project to different columns). Confirmed: a 2 mm slit (≈ no TDI motion) was equally blurred.
  - **Fix = "curvature" beam mode (default):** r(s) = arch radius of curvature (σ 5 mm), clipped to [15, 80] mm, plus a pull toward the arch normal k = 0.3/mm. Result:
    - δ = 0–0.2° at incisor, canine and premolar; 8.8° at the molar; max 18° at the arch ends (third-molar zone).
    - Effective r = 23 mm (incisor), 24 (canine), 30 (premolar), 82 (molar).
    - Peak height ≥ 0.7 within about ±6 mm anteriorly (narrow trough). Molars stay sharp out to ±15 mm (wide trough).
    - In-trough beads 0.9–1.2 mm, i.e. sharp at all heights and tilts.
- **Arch fit changed** from polynomial j(i) to polar ρ(θ) (`simulator.fit_arch`). The polynomial ran ~5 mm lingual of the posterior teeth.
  v0 outputs were re-rendered with it.
- **Real renders (F_010/F_005/F_020).** Effects that now come from the physics:
  - spine shadow over the incisors;
  - ghosted opposite rami/spine at the image edges;
  - air-space bands along the palate and tongue;
  - "smile" occlusal curve (tilt);
  - soft-tissue gradients and blur outside the trough.
  Processing (background subtraction σ 12 mm, 85%) is needed for contrast. The raw −log image is kept so processing can be randomised later.
- **Remaining differences vs DENTEX:**
  - FOV: no orbits, nasal cavity or condyles; sinus barely present. This is CBCT coverage.
  - Open bite in CBCT vs bite block.
  - CBCT chin rest visible at the bottom.
  - No noise or detector texture.
  - 0.3 mm vs ~0.1 mm pixels, so CONTACT-scale crops are soft.
  - Molar crop centring is still approximate.
  - Single arch curve for both jaws; the upper arch is more buccal.

### Step 7: fixed splits → `splits/` (see `splits/README.md`)
- TF3: volume-level 70/10/20, stratified by set × metal × third molar. CONTACT: official test kept; val = 15% of official train.
  CONTACT has no patient IDs, so train/val molars may share a patient.
- **DENTEX duplicates:** the same radiograph appears under different names across subsets and within them (638 exact groups).
  Splits are made over duplicate groups.
  - **First attempt was wrong.** A 64-bit 9×8 dHash at Hamming ≤ 6 linked 15,846 "near-duplicate" pairs into giant groups (test = 1886 images).
  - Fixed with a 2048-bit 65×32 dHash. Nearest-neighbour distances are cleanly bimodal: 0 (identical) or ≥ 554 (different patients),
    plus one re-cropped copy at 394 (checked visually). Threshold 400.
  - The wrong files were deleted and regenerated before any use. TF3 and CONTACT files came out byte-identical; only dentex.json changed.
- Verified: no duplicate group in two DENTEX pools; no TF3 or CONTACT overlap.

### Step 8: domain-randomisation layer + dataset renderer
- `src/dentalpano/randomize.py`: factor registry. Each factor has a nominal value and a sampler; ablation = fix one factor at nominal.
  - **Volume factors:** contrast (μ_water 0.016–0.026/mm, bone contrast 0.75–1.3), spine scale 0.3–1.6, metal scale 1–3.
  - **Geometry factors:** trough width (r scale 0.7–1.6), blur outside the trough (slit 3–10 mm), ghosts (far-side weight 0.4–1.3),
    arch jitter (AP ±4, lateral ±2 mm, rotation ±4°, scale ±5%, tilt 0–10°).
  - **Image factors (on the fly):** noise (2e3–5e4 photons, read noise, PSF ≤ 0.4 mm), processing (background σ 6–25 mm,
    amount 0.6–0.95, unsharp 0–1, γ 0.7–1.4).
- **Problems found in the preview** (`outputs/dr_preview/`):
  - The metal mask used HU > 2800, but enamel reaches the scanner ceiling (~3100 HU). Now metal = crown/bridge/implant labels only.
  - The spine mask (bone behind the airway) first caught skull base and mastoid laterally. Now limited to ±25 mm of the airway midline.
  - At HU > 500 the spine mask held only the vertebral cortex. Now closed and hole-filled per slice.
  - The bright central band below the incisors persists with spine = 0. A depth-window test shows it is far-side neck soft tissue
    between the airway lobes. That is physical, so it is kept.
  - The spine is also superimposed on the ramus regions at the image edges, as in real panoramics.
- **Chin rest/bite block:** connected to the body, so it cannot be removed by connectivity.
  Panoramics are cropped vertically (top → mandible lower border + 8 mm). Real images will be cropped the same way.
- `opg.render` additions: far-side weight, FOV mask (Set A volumes don't reach the upper jaw), per-case `annotated` channel flags,
  channels encoded in chunks of 23 bits (44 channels = 12 structures + 32 per-tooth). Mask reduction runs on the depth window only:
  32 s → 14 s per render, with bit-identical masks.
- **Reproducibility:** MPS scatter-add order varies, so meanT differs between runs in 0.005% of pixels by one float16 step. Masks are identical.
- `scripts/render_dataset.py`: v0 nominal + K random variants per volume; resumable; data/renders/<name>/<case>_v<k>.npz (~0.3 MB each).
- **Timing per render:** Set A ~8 s, Set C ~12 s, Set B ~15–17 s → ~71 min per pass over train+val (427 volumes).

### Step 9: cluster (USD Lawrence) preparation — `cluster/`
- User has a Lawrence account (santosh lab; miniforge3 already in home). Docs read: Slurm; `gpu` partition with
  2× P100 16 GB, 5× V100 32 GB, 2× A100 80 GB; home 50 GB, lab dir 5 TB, nothing backed up; per-job /scratch 169 GB.
- **Code changes:** `$DENTAL_DATA_ROOT` overrides the data root (`io.paths`); `opg.device()` prefers CUDA.
- **Added:** `cluster/discover.sh`, `cluster/sync_to_cluster.sh` (Mac → cluster rsync; one SSH master so Duo is asked once),
  `cluster/setup_env.sh` (conda env + torch from the cu126 index), `cluster/gpu_smoketest.sbatch`, `cluster/render_dataset.sbatch`.
- **Torch caution:** PyTorch's CUDA ≥ 12.8 wheels dropped sm_60/sm_70 (P100/V100), hence the cu126 index.
  The cluster torch version may differ from the Mac pin; `setup_env.sh` records the cluster freeze.
- macOS rsync is openrsync (no --append-verify), so the 27 GB TF3 upload restarts if interrupted unless GNU rsync is installed.
- **Cluster setup done (2026-10-23):**
  - `sinfo` facts: gpu partition = 5× V100 (`volta`, gpu002–006) + 1 node with 2× P100 (`pascal`, gpu001), 2-day limit.
    The A100 nodes in the docs are absent.
  - PROJ = /home/santosh_lab/shared/dipesh/dental (only `shared/` is writable in the lab dir).
  - ToothFairy3.zip copied; size verified (29305037433 B). CONTACT zip copied.
  - Env `dental`: torch 2.14.1+cu126 (same version as the Mac pin), archs include sm_60/sm_70.
    Freeze in cluster/requirements.cluster.lock.txt (on the cluster).
  - `short` partition was full at first; an interactive job on `-p nodes --time=01:00:00` started fine.
- **V100 smoke test (job 1311473, gpu003):** CUDA works (driver 545 / CUDA 12.3 runs the cu126 build fine). But 6 renders took 2.3 min
  (~23 s each), slower than the M4 (~14 s). CPU time ≈ wall time, so it was CPU-bound: per-channel `np.isin` over the whole volume
  (44 channels × 2 passes per render) on slower Skylake cores.
  **Fix:** channel bits built with one 256-entry lookup table (bit-identical, 3.8 s → 0.34 s); annotated flags from one bincount;
  `opg.prepare_labels()` caches both per volume across variants. Mac render now ~6.3 s (from ~14 s); the rest is mostly GPU.
- **V100 smoke test after the fix (job 1311477):** 6 renders + 2 volume loads in 0.6 min (was 2.3) → ~5 s per Set B variant.
  Estimates on one V100: test nominal ~15 min; main set (427 vols × 9) ~4 h; full ablation (7 × 374 × 4) ~15 h.
- **Sharding:** `render_dataset.py --shard/--nshards` plus a Slurm job array (`sbatch --array=0-3%4 …`); `index.json` written atomically.
- **Paired ablation:** `randomize.sample` now always draws every sampler, then overrides the disabled factor with nominal.
  With the same seed, an ablation sample differs from the full-DR sample only in that factor.
  (This changed the random draws vs the dr_preview figures; nothing had been rendered with the old scheme except smoke tests.)
- `scripts/qc_renders.py`: contact sheet with image-level DR, masks and FOV; flags renders with no teeth/mandible, little FOV or NaNs.
  Note: in Set B the mandible reaches the bottom of the volume, so the dentate crop removes little of the chin-rest strip.
  → For training, exclude rows below the mandible's lower border + margin from the loss (TODO, training stage).
- P100 submission: `-B 1:12` (from the docs) fails with "Requested node configuration is not available"; `--cpus-per-task=12` works.
- 2026-10-23: submitted dr_main (array 1311480_0-3, V100s) and nominal_test (1311485, P100).
- **First full render attempt failed (jobs 1311480_0–3, 1311490), all tasks after 6–17 min.**
  Cause: 20 volumes crash `fit_arch`: 19 edentulous (no tooth labels → empty arch) and P_507 (2 teeth → NaN arch).
  The first bad volume in each shard reproduces the observed counts exactly (dr_main 1045, nominal_test 56 files).
  - **Fix:** with < 6 teeth, the arch is fitted to the mandibular body in a 15 mm slab below the anterior alveolar crest, plus any teeth.
    With no teeth, the central-ray height = crest + 5 mm. `fit_arch` now raises on degenerate fits and records `source`.
    Checked visually: P_026 and S_0009 (edentulous) and P_507 have arches on the ridge, with canals and mandible rendered.
  - 43 volumes use the fallback (20 with 0 teeth, 23 with 1–5); edentulous patients stay in the data.
  - `render_dataset.py` now catches per-volume errors, writes failures.shard<i>.json, continues, and exits 2 if any failed.
  - The partial dr_main/nominal_test sets are deleted and re-rendered, because some 1–5-tooth volumes had been rendered with the old arch.

## 2026-10-24
### Step 10: dataset renders on Lawrence — done and QC'd
- Re-rendered after the edentulous fix: dr_main = 427 train+val volumes × 9 (v0 nominal + 8 DR) = 3843 renders;
  nominal_test = 105 test volumes. No failures, no NaNs, no missing mandible, no render with FOV < 50%.
  `no_teeth`: 162 renders (18 edentulous volumes × 9) and 2 in the test set, as expected. ~1 h on 4 V100s.
- **Visual QC** (qc.png of both sets, 12 random samples each): teeth/pulp/canals/incisive canals/prostheses align with the image;
  Set A FOV boundary at the top is respected by the masks; DR variety is visible.
  - For training: mask the chin-rest strip out of the loss (Set B bottom).
  - For training: synthetic contrast is flatter than DENTEX → tune the processing ranges against real image statistics.
  - Implant masks are tiny (mostly inside crowns).
### Step 11: decisions (user said yes to all three recommendations)
- **Ablation renders:** `cluster/ablation.sbatch` = one array of 28 tasks (7 render-level factors × 4 shards, %4 at a time).
  Train volumes × v1..v4 with the factor at nominal. Paired with dr_main (same seed function; verified that only the disabled factor differs).
- **Real labels:** the user will label canal/sinus (60 images) and pulp (20) on DENTEX test images.
  - `splits/dentex_annotation_subset.json` (write-once, 1 per duplicate group, quadrant_enumeration so tooth polygons exist: 1704 on the subset).
  - Protocol: `docs/annotation_protocol.md` (label before seeing predictions; visibility flags; expert double-labels the first 15 for agreement).
  - Label Studio config: data/annotation/dentex_ann/label_config.xml.
  - Also recommended: email the CONTACT authors for full panoramics + canal outlines.
- **Training:** custom multi-label (sigmoid per channel) U-Net, because channels overlap and nnU-Net assumes one label per pixel.
  The loss is masked by FOV, annotated flags and the chin-rest strip. nnU-Net stays as the real-only teeth baseline.
- **Annotation guides** (`scripts/annotation_guide.py` → outputs/annotation/guide_1..5.png). They use synthetic train-split renders
  with exact labels plus real practice images from the DENTEX few-shot train pool (never the 60 annotation images).
- **Simulator issue found while making the guides:** in partially edentulous patients with missing posterior teeth (F_003, F_029),
  the teeth-only arch ends early and its straight extension misses the posterior canal, so the canal label is a short stub.
  This affects some dr_main renders. TODO before training: always fit the posterior arch with mandible-body support (teeth ∪ crest slab),
  quantify truncated-canal renders (left/right canal length asymmetry), re-render the affected volumes.
### Step 12: real-label decision (2026-10-24)
- **User decided not to annotate** (not a dental expert). Self-labelled test ground truth would be weak for a paper.
  Manual annotation is ON HOLD until an expert is available.
  `splits/dentex_annotation_subset.json`, `docs/annotation_protocol.md` and the local Label Studio project stay ready for an expert.
  The user's partial trial annotation of image #1 is not used.
- **Real evaluation scope now:** teeth + enumeration (DENTEX test, tooth polygons), canal + third molar (CONTACT test crops),
  contact classification (CONTACT). Sinus / pulp / incisive canal: synthetic test set + qualitative real examples only;
  state this limitation explicitly.
- **To pursue:** ask the PI (Prof. Santosh) for an expert annotator; email the CONTACT authors for full panoramics + canal outlines.
  Public data: no expert panoramic sinus/pulp masks found. MMOral (github.com/isbrycee/OralGPT; built from TED3 + Do et al. 2024)
  lists canal/sinus, but those appear model-validated, not expert ground truth. Not usable as GT without checking.

## 2026-10-30
- **Ablation renders done:** array 1311640, 28/28 COMPLETED (~15 min each); 7 × 1497 files; no failures; renders total 3.0 GB.
### Step 13: arch fit v3, canal truncation fixed (everything must be re-rendered)
- New `scripts/canal_coverage.py`: fraction of 3D canal and tooth voxels within ±12 mm (in-plane) of the arch, per volume.
- **v2 arch (teeth-only polar polynomial + 25 mm straight extension):** canal coverage < 60% on some side in **125/532** volumes
  (< 90% in 193). Partially edentulous jaws: the teeth-only arch covers only the front, and its extensions turn inward
  while the canal continues back into the rami. Teeth coverage ≥ 99.4% everywhere.
- **Attempt 1 (teeth else mandible body, 6 mm deviation rule):** canals 100%, but teeth coverage dropped in 48 volumes (min 71%). Rejected.
- **Final (v3):** per 2° direction, the radius is the middle of the combined 5–95th percentile radial range of TEETH and CANAL
  (mandibular-body median where there is neither). Gaussian smoothing over θ (σ 6°) instead of a polynomial; centre inside
  the mandibular U; 10 mm extension.
  → **canal coverage 100% in 531/532** (P_381: 78% on one side, displaced/impacted teeth widen the radial range); **teeth ≥ 99.3% everywhere**.
  - Beam obliquity rises slightly: median per-volume p95 within |s| ≤ 50 mm goes 4.2° → 6.6°, worst 22° → 23.5°.
  - Arch length median 175 → 203 mm (now reaches into the rami, as on real panoramics).
  - Phantom (F_010): in-trough beads 0.9 mm; incisor r = 18 mm (narrow trough), posterior wide.
  - Renders checked: F_010, F_001, S_0015, P_204, P_026 show full canals and intact teeth.
- **All existing renders (dr_main, nominal_test, abl_no_*) used the v2 arch → re-render everything as *_v2 sets.**
  Old sets are kept until the user deletes them.
