# Learning Panoramic Anatomy from CBCT: Physics-Based Simulation and Domain Randomization for Zero- and Few-Shot Segmentation of Dental Panoramic Radiographs

*Dipesh Tripathi¹, [co-authors to be confirmed]*
*¹ [Affiliation to be confirmed], University of South Dakota*

*Draft of 2026-11-17. Every number is reproduced by `scripts/make_figures.py` from the run outputs (tables in
`outputs/paper/tables.md`, figures in `outputs/paper/figures/`). Figures whose name ends in `_DATA` show licensed
dataset images and are kept out of version control.*

---

## Abstract

Panoramic radiographs are the most common dental imaging exam, but annotating them is expensive. A panoramic image is
also a projection: structures overlap, so a 2D label cannot say whether two structures that overlap in the image
actually touch in 3D. We ask how far a segmentation model can get on real panoramics when it never sees a real labelled
image, only panoramics simulated from labelled cone-beam CT (CBCT). We built an open simulator of rotational panoramic
imaging. It models the moving narrow beam, the focal trough and the blur outside it, ghost images and the spine shadow.
We applied it to 532 ToothFairy3 CBCT volumes and projected their 3D labels into overlapping 2D masks for teeth (by FDI
number), pulp, the inferior alveolar canal (IAC), the incisive canal, the maxillary sinus and restorations. A domain
randomization scheme varies the panoramic-specific factors, metal, contrast, noise and vendor-style processing.

A U-Net trained only on these renders segments and numbers teeth on 187 real DENTEX panoramics. Over three training
seeds it reaches a teeth Dice of 0.755 ± 0.020, detects 87% of teeth, and gives 90.0% ± 0.1% of detected teeth the
correct FDI number. Fine-tuned on four labelled real images, it reaches a numbering accuracy of 0.933, against 0.454
for a tuned real-only model trained on the same images. Real-only training needs about ten times more labels to catch
up, and the two converge at around 100 labelled images. Pretraining on simulated panoramics without randomization fails
zero-shot (teeth Dice 0.28) and helps less in the few-shot setting. A paired, per-factor ablation over three seeds shows that
image-level noise and processing matter most for transfer, while the panoramic geometry factors mainly help the model
tell teeth apart. Physics-based randomization clearly beats CycleGAN image translation, because translation alters the
anatomy the labels describe (numbering accuracy 0.900 vs 0.104).

Finally, we compute the true 3D contact between the third-molar root and the IAC from each CBCT and train a contact
classifier on synthetic crops alone. On real CONTACT crops, it separates contact from no contact with AUC 0.596. The
predicted 2D overlap is at chance (AUC 0.51). The signal is weak but present. Fine-tuning it helps only with very
few real labels (nine crops); from 44 crops a tuned real-only model is as good or better. The simulator, splits and code will be released.

---

## 1. Introduction

The panoramic radiograph, or orthopantomogram, is the standard first-line image in dentistry. A single low-dose
exposure shows both dental arches, the mandible, the inferior alveolar canals (IAC) and the lower maxillary sinuses.
It is used to plan extractions and implants, find caries and periapical disease, and assess the relation between an
impacted third molar and the IAC before surgery [Mallya & Lam 2018]. Deep learning on panoramics has advanced quickly
for tooth detection and numbering [Tuzoff et al. 2019; Jader et al. 2018], diagnosis [Hamamci et al. 2023], and
segmentation of the canal and other structures [Vinayahalingam et al. 2019; Cha et al. 2021; Aung et al. 2024]. But
these methods depend on pixel-level labels drawn by dentists on 2D images. Such labels are expensive and inconsistent
between datasets. For thin or low-contrast structures such as the canal, pulp and sinus floor, they are also hard to
draw at all. Public panoramic datasets therefore mostly label teeth. We found no public expert segmentations of the
sinus, pulp or incisive canal on panoramics.

A panoramic image also has a problem that annotation cannot fix. It is a curved, blurred projection, so structures at
different depths overlap. Whether a third-molar root actually touches the IAC is the question that decides the risk of
nerve injury during extraction. It cannot be read reliably from the panoramic itself. The classical radiographic signs
of Rood and Shehab [1990] are only moderately predictive. Studies that use CBCT as the reference standard report that
both clinicians and CNNs find the task hard on panoramics [Fukuda et al. 2020; Jeon et al. 2023; Ulusoy et al. 2025].
A 2D label drawn on the panoramic cannot carry the 3D truth.

Cone-beam CT offers both things the panoramic lacks. Public CBCT datasets now come with dense voxel labels: ToothFairy3
provides 532 volumes with 77 classes, including every tooth, its pulp, both IACs and the incisive canals
[ToothFairy3 2025; Cipriano et al. 2022]. If a realistic panoramic can be simulated from such a volume, every label
can be projected exactly, overlaps included, and 3D relations such as root–canal contact can be computed directly.
This is the logic of in-silico training. SyntheX showed that realistic simulation from CT can replace real training
data for X-ray analysis [Gao et al. 2023]. Domain randomization makes models robust to the remaining gap by varying the
simulator widely [Tobin et al. 2017; Billot et al. 2023]. In dentistry, simulated panoramics from CBCT have so far been
used mainly as an intermediate for 3D reconstruction [Song et al. 2021; Park et al. 2024; Szczepański et al. 2026; Yue
et al. 2026], or as pretraining data for tooth segmentation [Wang et al. 2025]. Nobody has yet tested how far a
panoramic simulator alone can carry multi-structure segmentation and 3D-relation prediction on real images, or which
parts of the simulation matter.

This paper addresses that question. Our contributions are:

1. **An open, physics-based panoramic simulator over ToothFairy3.** It models the rotational narrow-beam acquisition,
   including the focal trough, blur outside it, ghosts and the spine shadow. It projects overlapping multi-label masks for
   32 teeth by FDI number, pulp, both IACs, incisive canals, the maxillary sinus, crowns, bridges and implants. Public
   real labels exist only for the teeth and the canal, so these two are validated on real images; the other structures
   are evaluated on held-out synthetic data.
2. **Panoramic-specific domain randomization with a paired per-factor ablation.** Nine factor groups are switched off
   one at a time with all other random draws held fixed, over three seeds. Follow-up experiments retune the most
   influential ranges.
3. **A sim-to-real evaluation on real panoramics.** It covers zero-shot transfer, few-shot fine-tuning against real-only
   training at 1–100% of the labels, and a CycleGAN image-translation baseline in the style of NeBLa.
4. **A 3D-relation head.** It predicts true third-molar–IAC contact from the 2D image alone, trained with contact labels
   computed automatically from each CBCT and tested on real CONTACT crops with CBCT-confirmed labels.

The main findings are as follows. First, a model trained on simulation alone segments and numbers teeth on real
panoramics. With a handful of real labels it matches real-only training that uses about ten times more, although the
advantage disappears once about 100 labelled images are available. Randomization is what makes this work:
unrandomized simulation does not transfer. Second, image-level realism (noise and vendor processing) dominates transfer, while geometry
helps tooth identity. Third, physics-based randomization preserves label correctness, whereas unpaired translation
does not. Fourth, a contact signal learned from CBCT transfers weakly to real images, while 2D overlap carries none;
it is a useful starting point only when almost no real contact labels exist.

## 2. Related Work

### 2.1 Deep learning on panoramic radiographs

Early work detected and numbered teeth with two-stage detectors [Tuzoff et al. 2019] and segmented them with Mask
R-CNN [Jader et al. 2018], often on private data. Public benchmarks followed. The Tufts Dental Database provides 1,000
panoramics with expert tooth and abnormality labels [Panetta et al. 2022]. DENTEX, a MICCAI 2023 challenge, provides
hierarchically labelled quadrants, FDI enumeration and diagnoses, plus a large unlabelled pool [Hamamci et al. 2023].
Non-tooth anatomy has received less attention and smaller datasets. Vinayahalingam et al. [2019] segmented third molars
and the IAC on 81 panoramics. Cha et al. [2021] performed panoptic segmentation of the sinus, jaws, canal, teeth and
implants on 51 images. Aung et al. [2024] studied canal segmentation across devices. All of these rely on 2D expert
labels drawn on the radiograph. Our work instead derives labels from CBCT and needs no 2D annotation for training.

### 2.2 Third-molar–canal relation

Rood and Shehab [1990] described radiographic signs on panoramics, such as darkening of the root, interruption of the
canal's white line and diversion of the canal, that are associated with nerve injury. CBCT is the reference standard
for actual contact. Several deep-learning studies classify the relation directly from panoramic crops using CBCT-derived
labels [Fukuda et al. 2020; Jeon et al. 2023], with accuracies around 0.7–0.8 on held-out data. The CONTACT dataset
[Ulusoy et al. 2025] is the first public panoramic dataset of this kind. It has 1,478 third-molar–canal pairs from 1,010
patients, semantic masks, expert-scored radiographic features and CBCT-confirmed contact labels. Its authors report that
learned models and clinicians alike find the "overlap without contact" case hardest. All prior approaches learn contact
from real labelled panoramics. We learn it from CBCT geometry alone and use CONTACT only for testing and few-shot
experiments.

### 2.3 Simulated X-rays and sim-to-real transfer

Digitally reconstructed radiographs (DRRs) from CT have long been used in registration. DeepDRR added realistic
material decomposition, scatter and noise, and showed that models trained on such DRRs transfer to real fluoroscopy,
while models trained on naive DRRs do not [Unberath et al. 2018, 2019]. DiffDRR made DRR rendering differentiable
[Gopalakrishnan & Golland 2022]. SyntheX scaled this idea and found that realistic simulation combined with strong
domain randomization can match or exceed real-data training for hip imaging, surgical tools and COVID-19 lesions [Gao et
al. 2023]. AnyCXR applies multi-stage randomization to chest X-ray anatomy at arbitrary projection angles [Dong et al.
2025]. Domain randomization itself originates in robotics [Tobin et al. 2017]. In medical imaging, SynthSeg showed that
fully randomizing contrast and resolution yields brain-MRI segmentation that works on any contrast [Billot et al. 2023].
Panoramic imaging differs from these settings because the image is not a single projection. A narrow beam rotates
around the patient while the detector moves, so a curved layer is sharp and everything else is smeared. Most of what
makes a panoramic look like a panoramic comes from this geometry. To our knowledge, the effect of randomizing it has not
been measured.

### 2.4 Simulated panoramics from CBCT

Simulated panoramics have mostly served 3D reconstruction from a single panoramic. Oral-3D [Song et al. 2021] and
X2Teeth [Liang et al. 2021] reconstruct bone or teeth from CBCT-derived panoramics. PX2Tooth [Ma et al. 2024]
reconstructs tooth point clouds and builds paired data by resampling CBCT along the dental arch. NeBLa [Park et al.
2024] simulates panoramics with Beer–Lambert attenuation along rays following panoramic rotation (SimPX). At test time
it translates real panoramics into the simulated style with a learned translation module. X-Splat [Szczepański et al.
2026] and HyDAR-Pano3D [Yue et al. 2026] train on simulated panoramic–CBCT pairs. X-Splat uses SimPX-style rendering on
ToothFairy data. For 2D analysis, PXseg [Wang et al. 2025] pretrains tooth segmentation and numbering on CBCT-derived
panoramics and then trains on real labelled images, reporting a 4.6% gain in tooth Dice over no pretraining.
These works either target teeth only, depend on real labels or a translation step at test time, or do not model the
panoramic blur and ghosting. Our no-randomization baseline (Section 4) approximates PXseg-style pretraining within our
pipeline. None studies which simulation factors matter. Our simulator models the rotational
time-delay integration geometry explicitly. We evaluate zero-shot transfer for many structures at once and ablate each
factor.

### 2.5 Unpaired image translation

An alternative to making the simulator realistic is to make its images look real afterwards, with unpaired translation
such as CycleGAN [Zhu et al. 2017]. Translation can close the appearance gap, but distribution-matching losses are known
to add or remove image content when the two domains' content statistics differ [Cohen et al. 2018]. Labels attached to
the source image then no longer match the translated one. We compare against this approach directly.

## 3. Methods

### 3.1 Data

**ToothFairy3** [ToothFairy3 2025] contains 532 CBCT volumes at 0.3 mm isotropic resolution in Hounsfield units, with
voxel labels for 77 classes. These include the jaws, every tooth by FDI number, each tooth's pulp, both IACs, the
incisive and lingual canals, the maxillary sinuses, and crowns, bridges and implants. The volumes come in three sets:
A (417 volumes, mostly mandible-centred field of view), B (63, full face) and C (52, a different scanner). The voxel
data are stored in RPI order although the header states LPS. We verified this with anatomical tests on all 532 label
volumes and correct it in memory. We found that the officially distributed reorientation script yields an
upside-down, front-to-back mirrored volume, so we did not use it. Upper-jaw structures are present mainly in Set B:
the maxillary sinus appears in about 50 volumes and the upper jawbone in 43% of them. Each volume carries per-channel
"annotated" flags and a field-of-view mask, so the loss ignores structures that a volume does not contain. We split the
volumes once into 374 training, 53 validation and 105 test volumes, stratified by set, metal and third-molar presence.

**DENTEX** [Hamamci et al. 2023] supplies real panoramics with FDI-enumerated tooth polygons, plus 2,098 unlabelled
images. The same radiograph appears under different file names across and within its subsets. We grouped exact
duplicates with a 2048-bit difference hash and made all splits over duplicate groups. This gives a few-shot training
pool of 384 labelled images, 62 validation images and a test set. Evaluation uses one image per duplicate group among
the test images that have tooth polygons: 187 images.

**CONTACT** [Ulusoy et al. 2025] provides 1,478 third-molar region crops from 1,010 patients, at 227 × 227 pixels,
with molar and canal masks, six expert-scored radiographic signs and a CBCT-confirmed contact label. Each crop belongs
to one of three groups: (1) overlap with contact, (2) no overlap and no contact, (3) overlap without contact. We keep
the official test set of 444 crops (330 / 34 / 80 by group) and hold out 15% of the official training set as validation
(878 train, 156 validation). The release has no patient identifiers, so the training and validation crops may share
patients. The test set is unaffected.

All datasets are used under their non-commercial licences.

### 3.2 Rotational panoramic simulator

A panoramic unit rotates a narrow vertical slit beam and a detector around the patient. The detector moves past the slit
at a speed matched to the beam's sweep along the arch. Points on a curved focal layer, the focal trough, project to the
same detector location throughout the exposure and are sharp. Points in front of or behind it drift across the detector
and are blurred horizontally. Structures on the far side of the head, such as the opposite ramus and the spine, are
smeared into ghost images. We simulate this time-delay integration (TDI) process directly.

**Arch curve.** For each volume we fit the focal-trough centre curve in polar form ρ(θ) around a centre inside the
mandibular U. At every 2° direction, the radius is the middle of the combined 5th–95th-percentile radial range of the
teeth and canal voxels. Where neither exists, we use the mandibular body. The curve is smoothed over θ (Gaussian, σ = 6°)
and extended 10 mm at each end. An earlier fit to the teeth alone ended too early in partially edentulous jaws and cut
off the posterior canal in 125 of 532 volumes. The combined fit keeps 100% of the canal within ±12 mm of the curve in
531 volumes, and at least 99.3% of the tooth voxels in every volume. For edentulous volumes, the arch is fitted to the
alveolar ridge.

**Beam geometry.** Time is parametrised by arc length s along the arch, in steps of 0.3 mm. At each step the central
ray passes through the arch point A(s), tilted upward by 5°. The source lies 400 mm behind the trough and the detector
100 mm in front, for a magnification of M = 1.25. A 6 mm slit is modelled by casting one ray per slit pixel. The
rotation centre for each step is set from the local radius of curvature of the arch, clipped to [15, 80] mm and pulled
toward the arch normal. This keeps the beams nearly perpendicular to the arch at the incisors and premolars, and
obliquely through the molar region as in real units. The detector advances by M·cos δ per step, where δ is the beam
obliquity, so that points on the trough stay fixed on the detector. The result is a narrow trough anteriorly and a
wide one posteriorly. We checked this with a bead phantom. Beads inside the trough render 0.9–1.2 mm wide at all
heights. Off-trough beads blur and drift as expected, and the far side produces ghosts.

**Attenuation and image formation.** Volumes are converted to linear attenuation μ = μ_water(1 + HU/1000), clipped at
zero, with μ_water = 0.02 mm⁻¹ nominally. Rays are sampled through the volume with trilinear interpolation, using
`grid_sample` on the GPU. The detector records transmission T = exp(−∫μ ds), averaged over the exposure. The image is
−log T, and the raw mean transmission is stored so that noise and processing can be applied later. Rendering takes
about 5 s per image on a V100 GPU.

**Label projection.** For each detector pixel and channel, we record whether the ray hits the structure within a depth
window around the focal point, for at least 50% of the exposure. Using the depth window rather than the whole ray keeps
far-side structures such as the opposite canal out of the masks. Channels are projected independently, so overlaps are
preserved. A tooth and the canal it overlaps are both labelled at the same pixel. We project 43 channels: 11 structure
channels (teeth union, pulp, left and right IAC, incisive canals, sinus, crown, bridge, implant, mandible, maxilla) and
32 FDI tooth channels. Panoramics are cropped vertically from the top of the field of view to the lower border of the
mandible plus 8 mm. This removes most of the CBCT chin rest. The images are resampled to a 256 × 640 canvas at 0.3 mm
per pixel.

### 3.3 Domain randomization

Every randomization factor has a nominal value, used for the validation and test renders, and a sampling range for
training. The factors act at three levels (Table 1). *Volume factors* change the attenuation before projection: tissue
contrast, spine density, and metal density (restoration labels scaled by a factor). *Geometry factors* change the
acquisition: trough width (a scale on the rotation radius), slit width (blur outside the trough), ghost weight (the
far-side contribution), and arch-curve jitter (shift, rotation, scale and head tilt). *Image factors* are applied on
the fly during training: Poisson photon noise, Gaussian read noise and detector blur, and vendor-style processing
(large-kernel background subtraction, unsharp masking, gamma).

| Factor | Level | Parameters (nominal → training range) |
|---|---|---|
| Trough width | geometry | rotation-radius scale 1 → log-U(0.7, 1.6) |
| Blur outside trough | geometry | slit width 6 → U(3, 10) mm |
| Ghosts | geometry | far-side weight 1 → U(0.4, 1.3) |
| Arch jitter | geometry | AP shift ±4 mm, lateral ±2 mm, rotation ±4°, scale ±5%, tilt 5° → U(0°, 10°) |
| Spine | volume | spine attenuation scale 1 → U(0.3, 1.6) |
| Metal | volume | restoration attenuation scale 1 → log-U(1, 3) |
| Contrast | volume | μ_water 0.02 → U(0.016, 0.026) mm⁻¹; bone contrast 1 → U(0.75, 1.3) |
| Noise | image | photons ∞ → log-U(2·10³, 5·10⁴); read noise ≤ 1%; PSF σ ≤ 0.4 mm |
| Processing | image | background σ 12 → U(6, 25) mm, amount 0.85 → U(0.6, 0.95); unsharp ≤ 1; γ → log-U(0.7, 1.4) |

*Table 1. Randomization factors. Geometry and volume factors are fixed when a render is made. Image factors are drawn
again for every training sample.*

For each training volume we render a nominal image (v0) and eight randomized variants (v1–v8), giving 3,843 training
and validation renders. Random draws come from a generator seeded per volume and variant, and every sampler is always
drawn, even when its factor is switched off. Switching a factor off therefore changes only that factor; all other
parameters of the render stay identical. This allows a *paired* ablation: each ablated render set differs from the
reference only in the removed factor.

### 3.4 Segmentation model and training

We use a U-Net [Ronneberger et al. 2015] with a ResNet-34 encoder [He et al. 2016] initialised from ImageNet, one input
channel and 43 sigmoid outputs (segmentation_models_pytorch). Channels overlap, so each is a separate binary problem.
For the same reason we did not use nnU-Net [Isensee et al. 2021], which assumes one label per pixel. The loss is
per-pixel binary cross-entropy plus a per-image, per-channel soft Dice term. Both are restricted to pixels where the
channel's label is valid, according to the field of view and annotated flags. Dice applies only where the structure is
present. On real DENTEX images, only the teeth union and the 32 FDI channels are supervised.

Augmentation consists of a random crop of 80–100% of the image and a horizontal flip. The flip swaps left and right
channels: the canals, and FDI quadrants 1↔2 and 3↔4. Real images are framed with one fixed relative crop, derived from
tooth extents in the few-shot training pool, so that they match the synthetic canvas. We train with AdamW (learning rate
10⁻³, weight decay 10⁻⁴), a one-cycle schedule, batch size 16 and mixed precision, for 60 epochs on one V100 GPU (about
2.3 hours). For the zero-shot model, the checkpoint is selected on the mean Dice over synthetic validation renders only.
Real images are never used for training or selection.

### 3.5 Third-molar crops and the contact head

CONTACT provides crops, not whole panoramics, so we train a crop model on matching synthetic crops. From the projected
masks we locate each lower third molar (FDI 38 or 48) and the canal on the same side. We then cut a square crop of
75 pixels (about 22.5 mm) around the molar–canal junction. This size matches the third-molar extent in CONTACT, about
172 × 140 pixels of 227. During training the crop scale is jittered by ×0.8–1.25 and the position by ±12%, with random
flips. The model has two channels, molar and canal, and is initialised from the panoramic model. Its molar head is the
mean of the tooth-38 and tooth-48 weights, and its canal head the mean of the two canal heads. Synthetic crops come
from 0.3 mm renders and are softer than CONTACT's crops. For evaluation we therefore downsample each real crop to the
synthetic resolution and resize it back (matched resolution), and we also report native resolution.

**3D contact labels.** For every lower third molar in ToothFairy3 we compute the minimum Euclidean distance between the
tooth and the canal label on the same side. We use a Euclidean distance transform with the true voxel spacing, within
a 15 mm margin. We define contact a priori as the labels touching, including diagonally: distance ≤ 0.52 mm, or √3 times
the 0.3 mm voxel size. Of 583 lower third molars, 195 (33%) are in contact: 411 / 58 / 114 molars in the training,
validation and test volumes. No manual labelling is involved.

**Contact head.** A classification head on the encoder bottleneck (global pooling, dropout 0.2, one logit) is trained
with binary cross-entropy on these labels, alongside the segmentation loss (weight 0.5). The model is initialised from
the crop model and selected on synthetic validation contact AUC. As a no-learning baseline, we also read contact from
the predicted 2D molar–canal overlap fraction, which is the information a panoramic shows directly.

## 4. Experimental Setup

**Zero-shot.** The simulation-only model is evaluated on the 187 DENTEX test images and on nominal renders of the 105
held-out ToothFairy3 test volumes.

**No-randomization baseline.** To isolate the effect of randomization, we train the same model on the nominal renders
only (v0 of each training volume, no image-level randomization), for the same number of optimisation steps as the main
model and with the same selection rule, using three seeds. This is the closest analogue within our pipeline to
pretraining on unrandomized CBCT-derived panoramics, as in PXseg [Wang et al. 2025].

**Few-shot.** Each arm uses 1, 5, 10, 25, 50 or 100% of the 384-image pool (4 to 384 images), with three seeds that
change the image subset and the training run, and selects checkpoints on the 62 real validation images. *Sim→real*
fine-tunes the zero-shot model (seed 0) with learning rate 3·10⁻⁴ for 40 epochs. *No-randomization sim→real* fine-tunes
the no-randomization model (seed 0) with the same schedule. *Real-only* trains the same network from ImageNet
initialisation. A model trained from ImageNet needs a longer schedule than one being fine-tuned: with the fine-tuning
schedule, real-only per-tooth Dice was still rising at the last epoch. We therefore tuned the real-only arm separately
(learning rate 10⁻³, 120 epochs, validation every third epoch), and report only this tuned baseline.

**Paired ablation.** The reference uses variants v1–v4 of every training volume with all factors on, trained for 40
epochs. Each ablation run replaces these with the paired render set in which one factor is at its nominal value. For
the image-level factors, the factor is simply not applied. Each configuration is trained with three seeds and compared
with the reference trained with the same seed.

**CycleGAN baseline.** Following the NeBLa strategy of bridging appearance by learned translation, we train a standard
CycleGAN [Zhu et al. 2017] (ResNet-9 generator, 70 × 70 PatchGAN, least-squares loss, cycle weight 10, identity 0.5) for
40,000 iterations between nominal synthetic renders and the 2,098 unlabelled DENTEX images. None of these share a
duplicate group with the test set. We then train the segmentation model, with the zero-shot schedule, on synthetic
renders passed through the frozen generator together with their original masks. We do this both without and with our
image-level randomization.

**CONTACT.** Zero-shot crop and contact models are tested on the 444 official test crops. For few-shot contact, we
compare fine-tuning the contact model (learning rate 3·10⁻⁴, 30 epochs) with training from ImageNet on 1–100% of the
878 training crops (masks and contact labels), with three seeds and selection on validation contact AUC. As on DENTEX,
the real-only arm uses a tuned schedule (learning rate 10⁻³, 90 epochs, validation every third epoch). With the
fine-tuning schedule it was undertrained: the tuned arm is better at almost every budget (canal Dice +0.03 to +0.14,
contact AUC +0.05 with all crops).

**Metrics.** On DENTEX we report Dice of the teeth union and mean Dice over FDI channels present in the ground truth.
DENTEX polygons are coarse: straight-edged, and including the gaps between teeth. We therefore also report three
boundary-insensitive tooth-level measures. A ground-truth tooth counts as *detected* if at least 50% of its polygon is
covered by predicted tooth pixels. *Numbering accuracy* is the fraction of detected teeth whose largest-overlap predicted
FDI channel is correct. *Presence F1* measures, per FDI number, whether a tooth is predicted (area ≥ 300 pixels) when
present. On CONTACT we report molar and canal Dice, and the AUC of contact (groups 1 vs 2 + 3), overall and for the hard
comparison group 1 vs 3. Confidence intervals are 95% percentile bootstraps over test images (2,000 resamples).

## 5. Results

### 5.1 Zero-shot transfer

Trained only on simulated panoramics, the model segments the teeth on the 187 real DENTEX test images with a Dice of
0.755 ± 0.020 over three training seeds (Table 2). It also numbers them: mean FDI Dice is 0.582 ± 0.019, 87.1% of
annotated teeth are detected, and 90.0% ± 0.1% of detected teeth receive the correct FDI number. Numbering accuracy is
almost identical across seeds, while Dice and detection vary more. The model never saw a real image or a real label.

Visual inspection explains much of the remaining Dice gap (`fig_dentex_overlays_DATA`). The predictions follow tooth
contours and leave interproximal gaps empty. The DENTEX polygons are straight-edged and include those gaps. Dice
therefore partly measures agreement with an annotation style, which is why we also report the tooth-level measures.
On the 105 held-out synthetic test volumes the model reaches a teeth Dice of 0.882, IAC 0.745 / 0.763 (left / right),
sinus 0.709, pulp 0.668 and incisive canal 0.460. On real images the predicted canals are anatomically plausible: they
run below the molar roots and rise into the ramus. Real canal accuracy is measured on CONTACT in Section 5.5.

All downstream experiments (few-shot, crops, contact) start from training seed 0, which happens to be the best of the
three seeds on DENTEX. We report its numbers with bootstrap confidence intervals alongside the mean, so that the
starting point of those experiments is clear.

| Measure (DENTEX test, 187 images) | Mean ± sd, 3 seeds | Seed 0 [95% CI] |
|---|---|---|
| Teeth Dice | 0.755 ± 0.020 | 0.774 [0.765, 0.783] |
| FDI Dice (mean over teeth present) | 0.582 ± 0.019 | 0.604 [0.582, 0.626] |
| Tooth detection rate | 0.871 ± 0.038 | 0.912 [0.895, 0.928] |
| Numbering accuracy | 0.900 ± 0.001 | 0.899 [0.880, 0.918] |
| Presence F1 | 0.929 ± 0.013 | 0.943 |

*Table 2. Zero-shot results of the simulation-only model. Seed 0 initialises every downstream experiment.*

### 5.2 Few-shot: simulation replaces most labels when labels are scarce

Simulation pretraining helps most when real labels are scarce (Table 3, `fig_fewshot_dentex`). With four labelled
images, sim→real reaches FDI Dice 0.702 ± 0.006 and numbering accuracy 0.933 ± 0.006. The tuned real-only model reaches
0.169 ± 0.042 and 0.454 ± 0.073 on the same images. Real-only training needs about ten times more labels to catch up:
with 39 images it reaches FDI Dice 0.696 and numbering 0.923, close to what sim→real achieves with four. With 20 images
sim→real is still clearly ahead (FDI 0.720 vs 0.620, numbering 0.935 vs 0.869). From about 100 labelled images the two
are equivalent; with all 384, real-only is marginally ahead on FDI Dice (0.779 vs 0.774) and level on numbering
(0.952 vs 0.950). Simulation therefore replaces most real labels in the low-data regime but adds little once a
moderately sized labelled set exists. Teeth-union Dice behaves similarly but with smaller gaps. The outline of the
dentition in DENTEX's annotation style is learned from few images; the identity of each tooth is not.

Randomization matters here too. Fine-tuning from the no-randomization model also helps at small budgets (FDI 0.619
with four images), so even unrandomized simulation carries useful anatomical information. But it is below the
randomized model at every budget, and from 39 images it falls behind real-only training (FDI 0.751 vs 0.779 with all
384 images). Pretraining on unrandomized renders seems to leave the network committed to the simulator's appearance,
which then has to be unlearned.

| Labelled images | Teeth Dice RO / NoDR / DR | FDI Dice RO / NoDR / DR | Numbering RO / NoDR / DR |
|---|---|---|---|
| 4 (1%) | 0.793 ± 0.009 / 0.839 ± 0.006 / **0.860 ± 0.013** | 0.169 ± 0.042 / 0.619 ± 0.011 / **0.702 ± 0.006** | 0.454 ± 0.073 / 0.886 ± 0.011 / **0.933 ± 0.006** |
| 20 (5%) | 0.865 ± 0.006 / 0.862 ± 0.007 / **0.875 ± 0.003** | 0.620 ± 0.020 / 0.662 ± 0.006 / **0.720 ± 0.007** | 0.869 ± 0.016 / 0.901 ± 0.003 / **0.935 ± 0.002** |
| 39 (10%) | 0.881 ± 0.000 / 0.872 ± 0.002 / 0.881 ± 0.004 | 0.696 ± 0.006 / 0.685 ± 0.007 / **0.736 ± 0.005** | 0.923 ± 0.005 / 0.911 ± 0.003 / **0.940 ± 0.001** |
| 96 (25%) | 0.888 ± 0.001 / 0.880 ± 0.001 / 0.886 ± 0.002 | 0.738 ± 0.007 / 0.708 ± 0.002 / **0.749 ± 0.002** | 0.941 ± 0.007 / 0.920 ± 0.003 / 0.940 ± 0.002 |
| 192 (50%) | 0.890 ± 0.001 / 0.886 ± 0.001 / 0.890 ± 0.001 | 0.756 ± 0.005 / 0.734 ± 0.004 / 0.762 ± 0.002 | 0.946 ± 0.003 / 0.933 ± 0.002 / 0.949 ± 0.001 |
| 384 (100%) | 0.894 ± 0.002 / 0.890 ± 0.001 / 0.894 ± 0.000 | 0.779 ± 0.002 / 0.751 ± 0.004 / 0.774 ± 0.002 | 0.952 ± 0.002 / 0.941 ± 0.002 / 0.950 ± 0.002 |

*Table 3. Few-shot results on the DENTEX test set, mean ± sd over three seeds. RO = real-only (ImageNet initialisation,
tuned schedule); NoDR = fine-tuned from the no-randomization model; DR = fine-tuned from the randomized zero-shot model.
Bold marks a clear best (difference larger than the spread across seeds).*

### 5.3 Which randomization factors matter

The paired ablation (Table 4, `fig_ablation`) separates the factors into three groups. The reference reaches a teeth
Dice of 0.734 ± 0.004 and numbering accuracy 0.911 ± 0.004 with the shorter ablation schedule.

*Image-level noise is essential.* Without it, teeth Dice drops by 0.436 [−0.547, −0.344], while accuracy on the
synthetic test set stays the same or improves. Without noise the model fits the simulator but does not transfer.
*Vendor-style processing* comes second (−0.066 [−0.095, −0.023]). Both effects appear in every seed.

*The panoramic geometry factors* (trough width, ghosts, blur outside the trough, arch jitter, spine) change teeth Dice
by at most ±0.03. Removing any of them lowers numbering accuracy, by 0.008 to 0.022, and the drop appears in all three
seeds for trough width, ghosts, blur outside the trough, arch jitter and spine. These factors help the model tell
neighbouring teeth apart rather than outline them.

*Metal randomization, as configured, hurts.* Removing it raises teeth Dice by 0.033 [0.022, 0.047] in every seed, at a
small cost in numbering (−0.015).

| Factor removed | Δ teeth Dice, mean [min, max] | Δ numbering, mean [min, max] | Same sign in 3/3 seeds |
|---|---|---|---|
| Noise | −0.436 [−0.547, −0.344] | −0.062 [−0.181, +0.026] | teeth |
| Processing | −0.066 [−0.095, −0.023] | −0.021 [−0.038, +0.006] | teeth |
| Spine | −0.011 [−0.038, +0.009] | −0.008 [−0.015, −0.001] | numbering |
| Arch jitter | −0.009 [−0.043, +0.016] | −0.018 [−0.039, −0.000] | numbering |
| Contrast | +0.001 [−0.015, +0.021] | −0.008 [−0.019, +0.005] | – |
| Trough width | +0.006 [−0.004, +0.015] | −0.022 [−0.028, −0.012] | numbering |
| Ghosts | +0.016 [−0.010, +0.033] | −0.020 [−0.031, −0.011] | numbering |
| Blur outside trough | +0.026 [+0.013, +0.047] | −0.018 [−0.026, −0.007] | teeth, numbering |
| Metal | +0.033 [+0.022, +0.047] | −0.015 [−0.027, +0.005] | teeth |

*Table 4. Paired per-factor ablation on the DENTEX test set. Each Δ is relative to the reference trained with the same
seed.*

**Retuning the ranges.** The ablation suggested two changes, which we tested under the same protocol (Table 5,
`fig_dr_followup`). Narrowing the metal range from ×1–3 to ×0.8–1.5 recovers most of the metal-off gain (teeth +0.024)
with a smaller numbering cost. Widening the image-level ranges (photon count, read noise, detector blur, background
subtraction, unsharp masking, gamma) improves teeth Dice by 0.031 and FDI Dice by 0.033 in every seed, with numbering
unchanged. It is the only change that improves outline and per-tooth Dice at no cost to enumeration.

| Variant | Teeth Dice | FDI Dice | Numbering | Δ teeth [min, max] | Δ numbering |
|---|---|---|---|---|---|
| Reference | 0.734 ± 0.004 | 0.569 ± 0.015 | 0.911 ± 0.004 | – | – |
| Metal off | 0.767 ± 0.009 | 0.592 ± 0.007 | 0.895 ± 0.015 | +0.033 [+0.022, +0.047] | −0.015 |
| Metal ×0.8–1.5 | 0.758 ± 0.005 | 0.584 ± 0.002 | 0.901 ± 0.006 | +0.024 [+0.014, +0.031] | −0.009 |
| Wider image ranges | 0.766 ± 0.011 | 0.601 ± 0.019 | 0.914 ± 0.009 | +0.031 [+0.020, +0.042] | +0.003 |
| Metal off + wider image ranges | 0.779 ± 0.009 | 0.599 ± 0.007 | 0.894 ± 0.012 | +0.044 [+0.036, +0.058] | −0.017 |

*Table 5. Follow-up experiments on the randomization ranges (DENTEX test, three seeds, shorter schedule).*

We then retrained the full model with the narrower metal range and the wider image ranges (seg_v2), three seeds each,
and compared it with three seeds of the original recipe (Table 7). On real panoramics the retuned model is better and
more stable: teeth Dice 0.773 ± 0.008 vs 0.755 ± 0.020, and detection 0.914 vs 0.871, with unchanged numbering.
But it is worse on the structures that lack real labels. On synthetic test volumes the IAC falls from 0.746 / 0.755 to
0.721 / 0.721 and implants from 0.687 to 0.573. The canal loss carries over to real images. A crop model trained with
the retuned recipe reaches a CONTACT canal Dice of 0.397 ± 0.084, against 0.479 ± 0.039 for the original, and it is
lower in every seed pair (Table 8). Stronger image-level randomization thus trades the thin, low-contrast canal against
the teeth. We keep the original recipe as our main model. The ablation's conclusions are drawn on teeth and should not
be generalised to every structure.

### 5.4 Randomization versus no randomization and image translation

Without randomization, simulation does not transfer at all (Table 6). Trained on nominal renders only, the model
reaches a teeth Dice of 0.277 ± 0.070 on DENTEX, detects 8% of the teeth and numbers 19% of those correctly, even though
it fits the synthetic data well. This mirrors the ablation, where removing image-level noise alone cost 0.44 Dice.

The CycleGAN-translated renders look convincingly real (`fig_cyclegan_samples_DATA`), yet segmentation trained on them
also transfers far worse than physics-based randomization (`fig_cyclegan`): teeth Dice 0.562 vs 0.755, and numbering
accuracy 0.104 vs 0.900. Adding our image-level randomization before translation does not help (0.509 / 0.216). The
samples show why. The unpaired translation changes the anatomy: bright restorations disappear, and tooth outlines and
gaps shift, so the projected labels no longer describe the image. Per-tooth identity suffers most, because it depends
on exactly the details the translation alters. A simulator whose labels stay correct under every randomization avoids
this failure. One caveat: this is a single run of the standard architecture, without structure-preserving losses.

| Training images | Runs | Teeth Dice | FDI Dice | Detection | Numbering |
|---|---|---|---|---|---|
| Nominal renders, no randomization | 3 | 0.277 ± 0.070 | 0.013 ± 0.003 | 0.079 ± 0.019 | 0.190 ± 0.138 |
| Physics-based randomization (ours) | 3 | **0.755 ± 0.020** | **0.582 ± 0.019** | **0.871 ± 0.038** | **0.900 ± 0.001** |
| CycleGAN-translated renders | 1 | 0.562 | 0.125 | 0.463 | 0.104 |
| Randomization, then CycleGAN | 1 | 0.509 | 0.145 | 0.344 | 0.216 |

*Table 6. Zero-shot results by source of training images (DENTEX test).**Table 6. Zero-shot comparison with unpaired translation (DENTEX test).*

| Model (3 seeds) | DENTEX teeth | DENTEX FDI | Detection | Numbering | Synth. IAC L / R | Synth. pulp | Synth. implant |
|---|---|---|---|---|---|---|---|
| Original ranges | 0.755 ± 0.020 | 0.582 ± 0.019 | 0.871 ± 0.038 | 0.900 ± 0.001 | 0.746 / 0.755 | 0.669 | 0.687 |
| Retuned ranges | 0.773 ± 0.008 | 0.598 ± 0.013 | 0.914 ± 0.014 | 0.895 ± 0.016 | 0.721 / 0.721 | 0.655 | 0.573 |

*Table 7. Full-schedule zero-shot models, original vs retuned randomization ranges, three seeds each.*

### 5.5 Third molar, canal and contact on real crops

**Segmentation.** The zero-shot crop model segments the third molar on the 444 CONTACT test crops with a Dice of 0.611
and the canal with 0.436 (seed 0, which initialises the contact experiments). Crop models vary considerably with the
training seed. Over three seeds, molar Dice is 0.672 ± 0.058 and canal Dice 0.479 ± 0.039, and seed 0 is the weakest
(Table 8). Molar outlines are often tight (per-image Dice 0.85–0.95). Failures mostly come from segmenting the
neighbouring second molar. CONTACT's canal masks are drawn as wide bands between the cortical lines, while our projected
label is the canal lumen, which caps canal Dice.

| Crop model (3 seeds) | Molar Dice | Canal Dice | Canal Dice, group 3 | Canal Dice per seed |
|---|---|---|---|---|
| Original ranges | 0.672 ± 0.058 | 0.479 ± 0.039 | 0.490 ± 0.036 | 0.436, 0.510, 0.492 |
| Retuned ranges | 0.662 ± 0.085 | 0.397 ± 0.084 | 0.399 ± 0.069 | 0.424, 0.465, 0.303 |

*Table 8. Zero-shot crop models on the CONTACT test set (matched resolution).*

**2D overlap cannot detect contact.** The predicted molar–canal overlap fraction gives a contact AUC of 0.510, and
0.468 for contact vs overlap without contact (group 1 vs 3). Both are at chance, which is the radiological problem
CONTACT was built to study.

**The contact head learns a weak but real signal from CBCT alone.** Trained only on synthetic crops with labels from
3D geometry, the head reaches an AUC of 0.596 [0.538, 0.656] on the real test set, and 0.575 for group 1 vs 3 (Table 9).
For reference, the six expert-scored radiographic signs supplied with CONTACT give an AUC of 0.54 when summed, and 0.72
for the best single sign (interruption of the canal's white line). Looser contact definitions transfer worse
(≤ 1 mm: 0.564; ≤ 2 mm: 0.515), and the validation set agrees, so we keep the a-priori definition. Validation AUC on
CONTACT (0.77–0.78) is well above test AUC. The official test set appears intrinsically harder: even the ground-truth
mask overlap separates contact with AUC 0.717 on test against 0.763 on validation, and the test crops are blurrier.

| Model (zero-shot) | Molar Dice | Canal Dice | Contact AUC [95% CI] | Group 1 vs 3 AUC |
|---|---|---|---|---|
| Crop model, 2D overlap readout | 0.611 | 0.436 | 0.510 | 0.468 |
| Contact head, contact ≤ 0.52 mm | 0.604 | 0.478 | **0.596 [0.538, 0.656]** | **0.575** |
| Contact head, contact ≤ 1.0 mm | 0.624 | 0.412 | 0.564 [0.508, 0.622] | 0.554 |
| Contact head, contact ≤ 2.0 mm | 0.536 | 0.453 | 0.515 [0.457, 0.576] | 0.521 |

*Table 9. Zero-shot contact prediction on the CONTACT test set (444 crops, matched resolution).*

**Few-shot contact.** CBCT pretraining helps only when real labels are very scarce (Table 10, `fig_fewshot_contact`).
With nine labelled crops it reaches a contact AUC of 0.582 ± 0.010, against 0.445 ± 0.043 for the tuned real-only model,
and a canal Dice of 0.610 vs 0.495. Below chance, the real-only model at this budget has learned a spurious cue. With 44
and 88 crops the contact AUC of the two arms is within one standard deviation (0.601 vs 0.589; 0.621 vs 0.594), and
real-only training already segments the canal better (0.760 vs 0.684 with 44 crops). From 220 crops real-only training
is ahead on every measure, and clearly so from 439, reaching a contact AUC of 0.745 ± 0.030 with all 878 crops against 0.672 ± 0.018. The
synthetic contact prior does not carry over to CONTACT's harder test split once real labels are available, and
fine-tuning from it even costs canal accuracy. A plausible cause is the canal definition: our labels mark the canal
lumen, CONTACT's masks mark a wider band (see above), and a model started from the narrower definition does not fully
adapt within the fine-tuning schedule.

| Labelled crops | Contact AUC RO / S2R | Group 1 vs 3 RO / S2R | Canal Dice RO / S2R |
|---|---|---|---|
| 9 (1%) | 0.445 ± 0.043 / **0.582 ± 0.010** | 0.446 ± 0.033 / **0.539 ± 0.009** | 0.495 ± 0.064 / **0.610 ± 0.013** |
| 44 (5%) | 0.589 ± 0.012 / 0.601 ± 0.004 | 0.579 ± 0.027 / 0.565 ± 0.005 | **0.760 ± 0.029** / 0.684 ± 0.025 |
| 88 (10%) | 0.594 ± 0.089 / 0.621 ± 0.014 | 0.559 ± 0.103 / 0.565 ± 0.016 | **0.811 ± 0.008** / 0.757 ± 0.009 |
| 220 (25%) | **0.658 ± 0.028** / 0.627 ± 0.022 | 0.597 ± 0.034 / 0.569 ± 0.014 | **0.839 ± 0.007** / 0.765 ± 0.045 |
| 439 (50%) | **0.722 ± 0.027** / 0.649 ± 0.008 | **0.683 ± 0.034** / 0.589 ± 0.006 | **0.858 ± 0.004** / 0.807 ± 0.011 |
| 878 (100%) | **0.745 ± 0.030** / 0.672 ± 0.018 | **0.697 ± 0.038** / 0.612 ± 0.016 | **0.866 ± 0.006** / 0.833 ± 0.002 |

*Table 10. Few-shot contact on the CONTACT test set, mean ± sd over three seeds. RO = real-only with the tuned
schedule (ImageNet initialisation); S2R = fine-tuned contact model. Bold: the better arm where the means differ by more
than one standard deviation of either arm.*

## 6. Discussion

**What simulation buys.** The clearest result is tooth enumeration. A model that has only seen simulated panoramics
numbers 90% of detected real teeth correctly. Four real images raise this to 93%, a level that real-only training
reaches only with about 40 or more labelled images. Two features of the simulated data probably explain this. First,
every render has a complete, exact FDI label for every tooth in the volume, including teeth that are hard to see
because they overlap or are blurred. Second, the projection preserves the 3D order of the arch. Real-only training with
few labels learns the outline of the dentition but not which tooth is which, and that is the information simulation
supplies. The benefit is limited to the low-data regime. Once about 100 labelled images are available, a well-tuned
real-only model does as well. For teeth, which already have public labels, simulation is therefore mainly a way to need
far fewer of them. Its larger promise is for structures without any public labels, which our evaluation can only
partly test.

**What makes a panoramic simulator work.** The ablation suggests a division of labour. The image-level factors (noise,
detector blur and vendor processing) decide whether anything transfers at all. Without noise, real teeth Dice
collapses to about 0.30 even though synthetic accuracy is unchanged. The geometry factors, which are what make a
panoramic distinct, matter less for the outline but consistently help tooth identity. Our reading is that blur, ghosts
and trough variation create the confusions between neighbouring teeth that occur on real images, and the model learns
to resolve them. Metal randomization at its original strength made restorations unrealistically dominant and hurt the
outline. These per-factor effects are small (a few Dice points), and we needed three seeds and paired renders to
separate them from noise. Unpaired ablations with a single seed would have been misleading here.

**Teeth are not the whole story.** Retuning the ranges for teeth hurt the canal, on synthetic and real images alike.
Different structures want different randomization. Strong noise and processing wash out a thin, low-contrast canal that
a tooth survives easily. Simulation studies that tune or ablate only on the structure with real labels may be optimising
the wrong objective for the rest. Validating each structure, or randomizing per structure, is a natural next step.

**Contact from geometry.** A panoramic cannot show 3D contact directly, and the predicted 2D overlap indeed carries no
information. A head trained on contact labels computed from CBCT recovers a weak but significant signal (AUC 0.60,
confidence interval above 0.5) from the image alone, and it does so without a single real contact label. It is not
clinically useful at this level. Its value is that it shows the 3D relation leaves a learnable trace in the projection,
and that this trace can be learned from simulation. As a pretraining prior it pays off only at the smallest budget:
from 44 real crops a tuned real-only model is as good, and from 220 it is clearly better. Our
contact definition and CONTACT's may not coincide: we use labels touching on the canal lumen, CONTACT uses a
CBCT-confirmed relation. CONTACT also counts far more contact (74%) than our synthetic population (33%). The threshold
analysis rules out the simplest explanation, a looser threshold, but other differences in definition may remain.

**Translation versus simulation.** CycleGAN produced the most realistic-looking images and the worst segmentation. For
dense-label tasks this argues against bridging the sim-to-real gap by changing the image after the fact. Changing it
inside the simulator keeps the labels correct by construction.

## 7. Limitations

Real-image evaluation covers teeth (DENTEX) and the third molar, canal and contact (CONTACT). The sinus, pulp and
incisive canal are evaluated on synthetic data only. We found no public expert panoramic labels for them, and we did not
produce our own. DENTEX tooth polygons are coarse, and Dice rewards learning their style, which favours models trained
on DENTEX labels. CONTACT is distributed as crops without patient identifiers, so our validation split may share
patients with training, and its official test set appears harder than its training pool.

Most results use three seeds, and the seed-to-seed spread is not negligible. The panoramic model we report and use for
initialisation is the best of three seeds on DENTEX (0.774 against a mean of 0.755), and the crop model we use is the
weakest of three on CONTACT. The few-shot and contact experiments vary the fine-tuning seed but start from this one
model. In both few-shot experiments the real-only arm was tuned but the sim→real arm kept the fine-tuning schedule;
a longer or per-budget schedule might narrow the gap at large budgets, where real-only training is ahead. The
CycleGAN baseline is a single run of the standard architecture. ToothFairy3's field of view limits the
simulation: most volumes do not include the upper jaw, orbits or condyles, the patients bite in an open position rather
than on a bite block, and the simulated resolution (0.3 mm) is coarser than a real panoramic's.

## 8. Conclusion

A physics-based panoramic simulator over labelled CBCT, with randomization of the panoramic geometry and image
formation, is enough to train a model that segments and numbers teeth on real panoramics without any real labels. With
a handful of real images, it matches real-only training on about ten times as many labels, although the advantage
disappears once about 100 labelled images are available. Without randomization, simulation does not transfer. The paired ablation shows
which parts of the simulation matter, and that the answer differs between structures. Simulated labels also carry
information no 2D annotation can: a contact head trained on 3D geometry alone detects true third-molar–canal contact
above chance, where the visible overlap cannot, although real contact labels overtake it from a few dozen crops. We will release the simulator, the render configuration, the splits
and the trained models to support work on panoramic anatomy beyond the teeth.

## Data and code availability

ToothFairy3, DENTEX and CONTACT are public under their respective licences (mostly CC BY-NC-SA). The splits, simulator
and training code will be released upon publication. Renders and checkpoints can be regenerated from the configuration
files and fixed seeds.

## References

- Aung, M.T.Z., Lim, S.-H., Han, J., Yang, S., Kang, J.-H., Kim, J.-E., Huh, K.-H., Yi, W.-J., Heo, M.-S., Lee, S.-S.
  Deep learning-based automatic segmentation of the mandibular canal on panoramic radiographs: a multi-device study.
  *Imaging Science in Dentistry* 54(1):81, 2024. doi:10.5624/isd.20230245
- Billot, B., Greve, D.N., Puonti, O., et al. SynthSeg: Segmentation of brain MRI scans of any contrast and resolution
  without retraining. *Medical Image Analysis* 86:102789, 2023. doi:10.1016/j.media.2023.102789
- Cha, J.-Y., Yoon, H.-I., Yeo, I.-S., Huh, K.-H., Han, J.-S. Panoptic segmentation on panoramic radiographs: deep
  learning-based segmentation of various structures including maxillary sinus and mandibular canal. *Journal of
  Clinical Medicine* 10(12):2577, 2021. doi:10.3390/jcm10122577
- Cipriano, M., Allegretti, S., Bolelli, F., et al. Deep segmentation of the mandibular canal: a new 3D annotated dataset
  of CBCT volumes. *IEEE Access* 10:11500–11510, 2022. doi:10.1109/ACCESS.2022.3144840
- Cohen, J.P., Luck, M., Honari, S. Distribution matching losses can hallucinate features in medical image translation.
  *MICCAI* 2018, LNCS 11070, pp. 529–536. doi:10.1007/978-3-030-00928-1_60
- Dong, Z., Wu, W., Hao, J., Chen, T., Weng, Z., Zhou, B. AnyCXR: Human anatomy segmentation of chest X-ray at any acquisition position using multi-stage domain
  randomized synthetic data with imperfect annotations and conditional joint annotation regularization learning.
  arXiv:2512.17263, 2025.
- Fukuda, M., Ariji, Y., Kise, Y., Nozawa, M., Kuwada, C., Funakoshi, T., Muramatsu, C., Fujita, H., Katsumata, A.,
  Ariji, E. Comparison of 3 deep learning neural networks for classifying the relationship between the mandibular third
  molar and the mandibular canal on panoramic radiographs. *Oral Surgery, Oral Medicine, Oral Pathology and Oral
  Radiology* 130(3):336–343, 2020. doi:10.1016/j.oooo.2020.04.005
- Gao, C., Killeen, B.D., Hu, Y., et al. Synthetic data accelerates the development of generalizable learning-based
  algorithms for X-ray image analysis (SyntheX). *Nature Machine Intelligence* 5:294–308, 2023. arXiv:2206.06127
- Gopalakrishnan, V., Golland, P. Fast auto-differentiable digitally reconstructed radiographs for solving inverse
  problems in intraoperative imaging. *MICCAI 2022 Workshop on Clinical Image-Based Procedures (CLIP)*.
  arXiv:2208.12737
- Hamamci, I.E., Er, S., Durugol, O.F., et al. DENTEX: Dental enumeration and tooth pathosis detection benchmark for
  panoramic X-ray. arXiv:2305.19112, 2023 (MICCAI 2023 challenge).
- He, K., Zhang, X., Ren, S., Sun, J. Deep residual learning for image recognition. *CVPR* 2016, pp. 770–778.
- Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H. nnU-Net: a self-configuring method for deep
  learning-based biomedical image segmentation. *Nature Methods* 18:203–211, 2021.
- Jader, G., Fontineli, J., Ruiz, M., Abdalla, K., Pithon, M., Oliveira, L. Deep instance segmentation of teeth in
  panoramic X-ray images. *SIBGRAPI* 2018, pp. 400–407. doi:10.1109/SIBGRAPI.2018.00058
- Jeon, K.J., Choi, H., Lee, C., Han, S.-S. Automatic diagnosis of true proximity between the mandibular canal and the
  third molar on panoramic radiographs using deep learning. *Scientific Reports* 13:22022, 2023.
  doi:10.1038/s41598-023-49512-4
- Liang, Y., Song, W., Yang, J., Qiu, L., Wang, K., He, L. X2Teeth: 3D teeth reconstruction from a single panoramic
  radiograph. *MICCAI* 2021. arXiv:2108.13004
- Ma, W., Wu, H., Xiao, Z., Feng, Y., Wu, J., Liu, Z. PX2Tooth: Reconstructing the 3D point cloud teeth from a single panoramic X-ray. *MICCAI* 2024.
  arXiv:2411.03725
- Mallya, S.M., Lam, E.W.N. *White and Pharoah's Oral Radiology: Principles and Interpretation*, 8th ed. Elsevier, 2018.
- Panetta, K., Rajendran, R., Ramesh, A., Rao, S.P., Agaian, S. Tufts Dental Database: a multimodal panoramic X-ray
  dataset for benchmarking diagnostic systems. *IEEE Journal of Biomedical and Health Informatics* 26(4):1650–1659, 2022.
- Park, S., Kim, S., Kwon, D., Jang, Y., Song, I.-S., Baek, S.J. NeBLa: Neural Beer–Lambert for 3D reconstruction of oral structures from
  panoramic radiographs. *AAAI* 2024, 38(5):4433–4441. arXiv:2304.04027
- Ronneberger, O., Fischer, P., Brox, T. U-Net: convolutional networks for biomedical image segmentation. *MICCAI* 2015,
  pp. 234–241.
- Rood, J.P., Shehab, B.A. The radiological prediction of inferior alveolar nerve injury during third molar surgery.
  *British Journal of Oral and Maxillofacial Surgery* 28(1):20–25, 1990. doi:10.1016/0266-4356(90)90005-6
- Song, W., Liang, Y., Yang, J., Wang, K., He, L. Oral-3D: Reconstructing the 3D structure of oral cavity from panoramic
  X-ray. *AAAI* 2021, 35(1):566–573.
- Szczepański, T., Płotka, S., Grzeszczyk, M.K., Trzciński, T., Sitek, A. X-Splat: Gaussian splatting for 3D CBCT
  generation from single panoramic radiograph. arXiv:2607.02099, 2026.
- Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., Abbeel, P. Domain randomization for transferring deep neural
  networks from simulation to the real world. *IROS* 2017, pp. 23–30.
- ToothFairy3 challenge organisers (University of Modena and Reggio Emilia). ToothFairy3: multi-structure CBCT
  segmentation challenge, MICCAI 2025 (ODIN workshop). https://toothfairy3.grand-challenge.org/
- Tuzoff, D.V., Tuzova, L.N., Bornstein, M.M., et al. Tooth detection and numbering in panoramic radiographs using
  convolutional neural networks. *Dentomaxillofacial Radiology* 48(4):20180051, 2019.
- Ulusoy, A.C., Toprak, T., Selver, M.A., Güneri, P., İlhan, B. Panoramic radiographic features for machine learning
  based detection of mandibular third molar root and inferior alveolar canal contact. *Scientific Reports* 15:4178,
  2025. doi:10.1038/s41598-024-82915-5
- Unberath, M., Zaech, J.-N., Lee, S.C., et al. DeepDRR – a catalyst for machine learning in fluoroscopy-guided
  procedures. *MICCAI* 2018, pp. 98–106. Extended as: Unberath, M., Zaech, J.-N., Gao, C., et al. Enabling machine
  learning in X-ray-based procedures via realistic simulation of image formation. *International Journal of Computer
  Assisted Radiology and Surgery* 14(9):1517–1528, 2019.
- Vinayahalingam, S., Xi, T., Bergé, S., Maal, T., de Jong, G. Automated detection of third molars and mandibular nerve
  by deep learning. *Scientific Reports* 9:9007, 2019. doi:10.1038/s41598-019-45487-3
- Wang, R., Cheng, F., Dai, G., Zhang, J., Fan, C., Yu, J., Li, J., Jiang, F. PXseg: automatic tooth segmentation,
  numbering and abnormal morphology detection based on CBCT and panoramic radiographs. *BMC Oral Health* 25, 2025.
  doi:10.1186/s12903-025-06356-w
- Yue, Y., Schmid, J., Li, X., Delamare, E., Kim, J. HyDAR-Pano3D: A hybrid disentangled anatomical recovery framework
  for panoramic-to-3D reconstruction. arXiv:2605.20827, 2026.
- Zhu, J.-Y., Park, T., Isola, P., Efros, A.A. Unpaired image-to-image translation using cycle-consistent adversarial
  networks. *ICCV* 2017, pp. 2223–2232.
