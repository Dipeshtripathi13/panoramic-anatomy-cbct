# Learning panoramic anatomy from CBCT

Code for *Learning panoramic anatomy from CBCT: physics-based simulation and domain randomization for zero- and
few-shot segmentation of dental panoramic radiographs* (Dipesh Tripathi, University of South Dakota).

A rotational panoramic (orthopantomograph) simulator renders panoramics from labelled cone-beam CT (ToothFairy3) and
projects the 3D labels into overlapping 2D masks: 32 teeth by FDI number, pulp, both inferior alveolar canals, incisive
canals, maxillary sinus, crowns, bridges, implants, mandible and maxilla. Models trained only on these renders are
tested on real panoramics (DENTEX, CONTACT, Tufts, and an external mandible set).

## What is here

| Path | Contents |
|---|---|
| `src/dentalpano/opg.py` | Rotational time-delay-integration panoramic simulator (focal trough, slit blur, ghosts) |
| `src/dentalpano/randomize.py` | Panoramic domain randomization (geometry, volume and image factors; paired draws) |
| `src/dentalpano/tf3.py` | ToothFairy3 loading and the orientation fix (voxel data are RPI although the header says LPS) |
| `src/dentalpano/train/` | Datasets, U-Net model, masked multi-label loss, crop/contact datasets, CycleGAN |
| `scripts/` | Rendering, training, evaluation, statistics and figure scripts (each has a usage docstring) |
| `configs/` | Data paths and every training configuration used in the paper |
| `splits/` | The fixed patient-level splits (ToothFairy3, DENTEX duplicate-grouped splits, CONTACT) |
| `cluster/` | Slurm scripts used on our GPU cluster (paths are examples; set `PROJ` for your system) |
| `paper/latex/` | Manuscript source; `bash paper/build.sh` regenerates figures and builds the PDF |

## Install

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.lock.txt && uv pip install -e .
```

A CUDA GPU is recommended for rendering and training (Apple MPS works for small tests).

## Data

No data are redistributed here. Download each dataset from its provider, accept its licence, and place the zips in the
repository root (or point `$DENTAL_DATA_ROOT` at their folder). File names are listed in `configs/paths.yaml`. The
code reads the zips in place and never extracts them.

- **ToothFairy3**: https://toothfairy3.grand-challenge.org/ (`ToothFairy3.zip`)
- **DENTEX**: https://zenodo.org/records/7812323 (`Zenodo/training_data.zip`, `Zenodo/validation_data.zip`)
- **CONTACT**: Ulusoy et al., *Scientific Reports* 15:4178 (2025) (`CONTACTdataset.zip`)
- **Mandible test set**: Abdi & Kasaei, Mendeley Data, doi:10.17632/hxt48yk462.2
  (`AbdiMandible_DentalPanoramicXrays.zip`)
- **Tufts Dental Database**: Panetta et al., *IEEE JBHI* 26(4), 2022; access on request
  (`Tufts/Radiographs.zip`, `Tufts/Segmentation.zip`)

Do not use the reorientation script distributed with ToothFairy; it yields an upside-down, front-to-back mirrored
volume. `dentalpano.tf3.fix_orientation` applies the correct flip in memory.

## Reproduce

```bash
# render the training set (train + val volumes, nominal + 8 randomized variants each) and the nominal test set
python scripts/render_dataset.py --name dr_main_v2 --splits train val --variants 8
python scripts/render_dataset.py --name nominal_test_v2 --splits test --variants 0
# zero-shot model, then evaluation on DENTEX test and synthetic test volumes
python scripts/train.py --config configs/train/seg_v1.yaml
python scripts/evaluate.py --run outputs/runs/seg_v1_dr_full
# paired ablation: re-render with one factor at nominal, e.g.
python scripts/render_dataset.py --name abl_v2_no_ghosts --disable ghosts --variants 4
# tables, figures and paired statistics of the paper
python scripts/make_figures.py && python scripts/stats_paired.py
```

All random draws are seeded per volume and variant; every training run writes its configuration and git hash next to
its log. The Slurm scripts in `cluster/` contain the exact commands used for every experiment in the paper.

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences (mostly non-commercial); renders and models trained on
them inherit those terms.
