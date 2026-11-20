"""Read-only access to the raw datasets, directly from their zip archives.

Nothing is ever extracted to disk next to the raw data. Volumes are decoded in
memory; headers are read by streaming only the first bytes of each member.
"""
from __future__ import annotations

import gzip
import io
import os
import zipfile
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import yaml

REPO = Path(__file__).resolve().parents[2]


@lru_cache
def paths() -> dict:
    """Dataset paths from configs/paths.yaml; $DENTAL_DATA_ROOT overrides `root` (e.g. on the cluster)."""
    cfg = yaml.safe_load((REPO / "configs" / "paths.yaml").read_text())
    root = Path(os.environ.get("DENTAL_DATA_ROOT", cfg["root"]))
    if not root.is_absolute():                     # relative roots are relative to the repository
        root = REPO / root
    for key, val in cfg.items():
        if isinstance(val, dict):
            cfg[key] = {k: root / v for k, v in val.items()}
    return cfg


def open_zip(dataset: str, key: str = "zip") -> zipfile.ZipFile:
    return zipfile.ZipFile(paths()[dataset][key])


# ---------------------------------------------------------------- NIfTI (TF3)
def nifti_header(zf: zipfile.ZipFile, member: str) -> nib.Nifti1Header:
    """Parse a .nii.gz header by decompressing only its first 352 bytes."""
    with zf.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz:
        return nib.Nifti1Header.from_fileobj(io.BytesIO(gz.read(352)))


def load_nifti(zf: zipfile.ZipFile, member: str) -> nib.Nifti1Image:
    """Fully decode a .nii.gz member in memory."""
    data = gzip.decompress(zf.read(member))
    return nib.Nifti1Image.from_bytes(data)


# ---------------------------------------------------------------- MetaImage (TF2)
def mha_header(zf: zipfile.ZipFile, member: str) -> dict:
    """Parse the text header of a .mha member (stops at ElementDataFile)."""
    hdr = {}
    with zf.open(member) as f:
        for _ in range(64):
            line = f.readline().decode("latin-1").strip()
            if " = " not in line:
                continue
            k, v = line.split(" = ", 1)
            hdr[k] = v
            if k == "ElementDataFile":
                break
    return hdr
