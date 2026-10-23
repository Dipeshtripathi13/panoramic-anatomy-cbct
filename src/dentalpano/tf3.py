"""ToothFairy3 loading with a verified orientation fix applied in memory.

TF3 voxel data are stored RPI (index i→patient Right, j→Posterior, k→Inferior;
verified on all 532 label volumes by scripts/orientation_check.py) while the
header affine claims LPS. `fix_orientation` flips i and k so the data become
LPS and the header affine is truthful.

NOTE: the official Ditto script (ditto.ing.unimore.it/static/toothfairy3/fix_orientation.py)
flips SimpleITK array axes 1 and 2 (= nibabel axes 0 and 1, i and j), which yields
LAI — a 180° rotation of LPS (upside down, front-back swapped). Kept as
`official_fix` for comparison only; do not use it.
"""
from __future__ import annotations

import json

import numpy as np

from . import io as dio

ZIP_KEY = "toothfairy3"
UNKNOWN_IDS = (60, 100, 160)  # stray IDs in 4 Set C cases, not in dataset.json → background


def label_names() -> dict[int, str]:
    zf = dio.open_zip(ZIP_KEY)
    return {v: k for k, v in json.loads(zf.read("dataset.json"))["labels"].items()}


def case_ids(zf=None) -> list[str]:
    zf = zf or dio.open_zip(ZIP_KEY)
    return sorted(n[9:-7] for n in zf.namelist() if n.startswith("labelsTr/") and n.endswith(".nii.gz"))


def case_set(case: str) -> str:
    return {"P": "A", "F": "B", "S": "C"}[case[11]]


def fix_orientation(vol: np.ndarray) -> np.ndarray:
    """RPI-stored → LPS (matches header)."""
    return np.ascontiguousarray(vol[::-1, :, ::-1])


def official_fix(vol: np.ndarray) -> np.ndarray:
    """What the Ditto script does (→ LAI). For comparison only."""
    return np.ascontiguousarray(vol[::-1, ::-1, :])


def load_label(case: str, fix: bool = True, zf=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (uint8 label volume, 4x4 affine)."""
    zf = zf or dio.open_zip(ZIP_KEY)
    nii = dio.load_nifti(zf, f"labelsTr/{case}.nii.gz")
    lab = np.asarray(nii.dataobj, dtype=np.uint8)
    lab[np.isin(lab, UNKNOWN_IDS)] = 0
    return (fix_orientation(lab) if fix else lab), nii.affine


def load_image(case: str, fix: bool = True, zf=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (float32 HU volume, 4x4 affine). Stored as float64 on disk; cast immediately."""
    zf = zf or dio.open_zip(ZIP_KEY)
    nii = dio.load_nifti(zf, f"imagesTr/{case}_0000.nii.gz")
    img = np.asarray(nii.dataobj, dtype=np.float32)
    return (fix_orientation(img) if fix else img), nii.affine
