"""Shared display helpers for panoramic renders and real reference images."""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from . import io as dio

COLORS = {"teeth": "yellow", "pulp": "cyan", "canal_L": "red", "canal_R": "orange", "incisive_canal": "magenta",
          "sinus": "deepskyblue", "crown": "white", "bridge": "lightgray", "implant": "springgreen", "pharynx": "lime"}


def disp(a: np.ndarray, lo: float = 1, hi: float = 99.5) -> np.ndarray:
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / (p1 - p0), 0, 1)


def draw_masks(ax, masks: dict) -> list[str]:
    present = []
    for k, c in COLORS.items():
        if k in masks and masks[k].any():
            ax.contour(masks[k], levels=[0.5], colors=[c], linewidths=0.6)
            present.append(k)
    return present


def dentex_images(subset: str, n: int, rng) -> list[tuple[str, np.ndarray]]:
    zf = dio.open_zip("dentex", "train_zip")
    names = sorted(x for x in zf.namelist() if x.startswith(subset) and x.endswith(".png"))
    pick = [names[i] for i in sorted(rng.choice(len(names), n, replace=False))]
    return [(x.split("/")[-1], np.array(Image.open(zf.open(x)).convert("L"))) for x in pick]


def molar_crop(image: np.ndarray, masks: dict, side: str, crop_mm: float, px_mm: float):
    """Square crop centred between the lower third (else second) molar and its canal, like CONTACT ROIs."""
    canal = masks["canal_L" if side == "L" else "canal_R"]
    for tid in ((38, 37) if side == "L" else (48, 47)):
        m = masks.get(f"tooth{tid}")
        if m is not None and m.any():
            break
    else:
        return None, None
    rc, cc = ndimage.center_of_mass(m)
    near = canal.copy()
    near[:, : max(0, int(cc) - 60)] = False
    near[:, int(cc) + 60:] = False
    if near.any():
        rc = 0.5 * (rc + ndimage.center_of_mass(near)[0])
    h = int(round(crop_mm / px_mm / 2))
    img = np.pad(disp(image), h)
    crop = img[int(rc): int(rc) + 2 * h, int(cc): int(cc) + 2 * h]
    return np.array(Image.fromarray((crop * 255).astype(np.uint8)).resize((227, 227), Image.BILINEAR)), tid
