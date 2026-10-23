"""CONTACT dataset (as shipped in CONTACTdataset.zip): 227x227 third-molar crops.

Masks are single grayscale JPEGs. Five gray-level families occur (JPEG noise on top);
each has two "body" levels (molar, canal — order varies) and sometimes an overlap level:
  A: 42 / 212, overlap 127      B: 85 / 170, overlap 255      D: 51 / 255, overlap 153
  E: 102 / 204 (overlap 153)    F: 127 / 255 (no overlap level: one structure painted over)
`decode_mask` picks the family by pixel support, snaps pixels to its levels, mode-filters
JPEG speckle, then decides which body level is the canal by shape: the canal is a thin
band (small max distance-transform), the molar a thick blob. Overlap pixels belong to both.

Groups (from the brief): 1 = contact (implies overlap), 2 = no overlap & no
contact, 3 = overlap & no contact. The xlsx `Class` is the binary contact label;
overlap is derived from the mask.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from . import io as dio

FAMILIES = {  # name: (body level 1, body level 2, overlap level or None)
    "A": (42, 212, 127), "B": (85, 170, 255), "D": (51, 255, 153), "E": (102, 204, 153), "F": (127, 255, None),
}
SPLITS = {"train": ("Train/Train Images v2", "Train/Train Labels v2", "trainImageFeatures.xlsx"),
          "test": ("Test/Test Images v2", "Test/Test Labels v2", "testImageFeatures.xlsx")}
MIN_OVERLAP_PX = 20  # below this, overlap is treated as JPEG/annotation noise


def detect_scheme(gray: np.ndarray) -> str:
    near = lambda v: int((np.abs(gray.astype(int) - v) <= 8).sum())
    return max(FAMILIES, key=lambda k: near(FAMILIES[k][0]) + near(FAMILIES[k][1]))


def _mode3(lab: np.ndarray, n: int) -> np.ndarray:
    counts = np.stack([ndimage.uniform_filter((lab == k).astype(np.float32), 3) for k in range(n)])
    return counts.argmax(0)


def decode_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Return (molar, canal, family, canal gray level) with molar/canal as boolean masks."""
    fam = detect_scheme(gray)
    b1, b2, ov = FAMILIES[fam]
    levels = [0, b1, b2] + ([ov] if ov is not None else [])
    idx = _mode3(np.abs(gray.astype(int)[..., None] - np.array(levels)).argmin(-1), len(levels))
    overlap = idx == 3
    m1, m2 = (idx == 1) | overlap, (idx == 2) | overlap
    thick = lambda m: ndimage.distance_transform_edt(m).max() if m.any() else 0.0
    if thick(m1) >= thick(m2):
        return m1, m2, fam, b2
    return m2, m1, fam, b1


def load_case(zf, split: str, num: int) -> dict:
    img_dir, lab_dir, _ = SPLITS[split]
    img = np.array(Image.open(zf.open(f"{img_dir}/{num}.jpg")).convert("L"))
    gray = np.array(Image.open(zf.open(f"{lab_dir}/{num}.jpg")).convert("L"))
    molar, canal, scheme, canal_level = decode_mask(gray)
    return {"image": img, "molar": molar, "canal": canal, "scheme": scheme, "canal_level": canal_level}


def label_table(zf=None) -> pd.DataFrame:
    """One row per crop: split, ImageNum, features, contact, overlap_px, group, scheme."""
    zf = zf or dio.open_zip("contact")
    rows = []
    for split, (_, _, xlsx) in SPLITS.items():
        df = pd.read_excel(zf.open(xlsx))
        df.insert(0, "split", split)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True).rename(columns={"Class": "contact"})
    stats = []
    for split, num in zip(df.split, df.ImageNum):
        c = load_case(zf, split, num)
        ov = c["molar"] & c["canal"]
        stats.append((int(ov.sum()), int(c["molar"].sum()), int(c["canal"].sum()), c["scheme"], c["canal_level"]))
    df[["overlap_px", "molar_px", "canal_px", "scheme", "canal_level"]] = pd.DataFrame(stats, index=df.index)
    df["overlap"] = df.overlap_px >= MIN_OVERLAP_PX
    df["group"] = np.select([df.contact == 1, ~df.overlap], [1, 2], default=3)
    return df
