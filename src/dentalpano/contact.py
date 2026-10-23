"""CONTACT dataset (as shipped in CONTACTdataset.zip): 227x227 third-molar crops.

Masks are single grayscale JPEGs with one of two encodings (JPEG noise on top):
  scheme A: molar=42,  canal=213, overlap≈128 (mean of the two)
  scheme B: canal=85,  molar=170, overlap=255 (sum of the two)
`decode_mask` detects the scheme, snaps pixels to its levels, mode-filters away
JPEG speckle, and returns boolean (molar, canal) masks; overlap = molar & canal.

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

SCHEMES = {
    # level -> (molar, canal)
    "A": {0: (0, 0), 42: (1, 0), 213: (0, 1), 128: (1, 1)},
    "B": {0: (0, 0), 170: (1, 0), 85: (0, 1), 255: (1, 1)},
}
SPLITS = {"train": ("Train/Train Images v2", "Train/Train Labels v2", "trainImageFeatures.xlsx"),
          "test": ("Test/Test Images v2", "Test/Test Labels v2", "testImageFeatures.xlsx")}
MIN_OVERLAP_PX = 20  # below this, overlap is treated as JPEG/annotation noise


def detect_scheme(gray: np.ndarray) -> str:
    near = lambda v: int((np.abs(gray.astype(int) - v) <= 6).sum())
    return "A" if near(42) + near(213) > near(85) + near(170) else "B"


def _mode3(lab: np.ndarray, n: int) -> np.ndarray:
    counts = np.stack([ndimage.uniform_filter((lab == k).astype(np.float32), 3) for k in range(n)])
    return counts.argmax(0)


def decode_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    scheme = detect_scheme(gray)
    levels = np.array(list(SCHEMES[scheme]))
    idx = _mode3(np.abs(gray.astype(int)[..., None] - levels).argmin(-1), len(levels))
    table = np.array(list(SCHEMES[scheme].values()), dtype=bool)
    molar, canal = table[idx, 0], table[idx, 1]
    return molar, canal, scheme


def load_case(zf, split: str, num: int) -> dict:
    img_dir, lab_dir, _ = SPLITS[split]
    img = np.array(Image.open(zf.open(f"{img_dir}/{num}.jpg")).convert("L"))
    gray = np.array(Image.open(zf.open(f"{lab_dir}/{num}.jpg")).convert("L"))
    molar, canal, scheme = decode_mask(gray)
    return {"image": img, "molar": molar, "canal": canal, "scheme": scheme}


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
        stats.append((int(ov.sum()), int(c["molar"].sum()), int(c["canal"].sum()), c["scheme"]))
    df[["overlap_px", "molar_px", "canal_px", "scheme"]] = pd.DataFrame(stats, index=df.index)
    df["overlap"] = df.overlap_px >= MIN_OVERLAP_PX
    df["group"] = np.select([df.contact == 1, ~df.overlap], [1, 2], default=3)
    return df
