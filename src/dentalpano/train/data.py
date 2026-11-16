"""Datasets for multi-label panoramic segmentation.

Every item is (image [1,H,W] in [0,1], target [C,H,W] in {0,1}, weight [C,H,W] in {0,1}).
`weight` says where a channel's label is valid: outside the CBCT field of view, for channels that
are not annotated in a volume, or for channels without real ground truth it is 0, and the loss and
metrics ignore those pixels.

SynthDataset — renders from scripts/render_dataset.py; image-level DR (noise, processing) is
               applied on the fly from the stored mean transmission; random crop/scale/flip.
DentexDataset — real DENTEX panoramics with tooth polygons (quadrant_enumeration subset):
               teeth union + per-tooth FDI channels; fixed relative crop for every image.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .. import io as dio
from .. import randomize as DR

STRUCTURES = ["teeth", "pulp", "canal_L", "canal_R", "incisive_canal", "sinus", "crown", "bridge", "implant",
              "mandible", "maxilla"]
FDI = [q * 10 + t for q in (1, 2, 3, 4) for t in range(1, 9)]
TOOTH_CHANNELS = [f"tooth{t}" for t in FDI]
IMAGE_FACTORS = [f for f, (level, _) in DR.FACTORS.items() if level == "image"]
# fixed crop applied to every real panoramic (x0, y0, x1, y1 as fractions); from tooth extents in the
# DENTEX few-shot TRAIN pool (teeth span x 0.18–0.86, y 0.07–0.92) — see LOG.md
REAL_CROP = (0.10, 0.15, 0.90, 0.97)


def flip_name(c: str) -> str:
    """Channel that a structure becomes after a left-right image flip."""
    if c in ("canal_L", "canal_R"):
        return "canal_R" if c == "canal_L" else "canal_L"
    if c.startswith("tooth"):
        q, t = int(c[5]), c[6]
        return f"tooth{ {1: 2, 2: 1, 3: 4, 4: 3}[q] }{t}"
    return c


def _resize(x: torch.Tensor, size, mode: str) -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(x[None], size=size, mode="nearest")[0]
    return F.interpolate(x[None], size=size, mode="bilinear", align_corners=False)[0]


def augment_geometry(img, tgt, wgt, channels, rng, crop_frac=(0.8, 1.0), flip_p=0.5):
    """Random crop (width/height fractions) + horizontal flip with left/right channel swap."""
    _, H, W = img.shape
    fh, fw = rng.uniform(*crop_frac), rng.uniform(*crop_frac)
    h, w = max(16, int(H * fh)), max(16, int(W * fw))
    r0, c0 = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
    img, tgt, wgt = (t[:, r0:r0 + h, c0:c0 + w] for t in (img, tgt, wgt))
    if rng.random() < flip_p:
        perm = [channels.index(flip_name(c)) if flip_name(c) in channels else i for i, c in enumerate(channels)]
        img, tgt, wgt = img.flip(-1), tgt.flip(-1)[perm], wgt.flip(-1)[perm]
    return img, tgt, wgt


class SynthDataset(Dataset):
    def __init__(self, render_sets: list[str], cases: list[str] | None, channels: list[str], size=(256, 640),
                 train: bool = True, image_factors: list[str] | None = None, variants: list[int] | None = None,
                 seed: int = 0, root: Path | None = None, dr_overrides: dict | None = None):
        root = root or dio.REPO / "data" / "renders"
        self.files, self.meta, self.dr_overrides = [], {}, dr_overrides
        wanted = set(cases) if cases is not None else None
        for name in render_sets:
            meta = json.loads((root / name / "index.json").read_text())
            self.meta[name] = meta
            for f in sorted((root / name).glob("*_v*.npz")):
                case, v = f.stem.rsplit("_v", 1)
                if (wanted is None or case in wanted) and (variants is None or int(v) in variants):
                    self.files.append((name, f))
        if not self.files:
            raise ValueError(f"no renders found in {render_sets}")
        self.channels, self.size, self.train, self.seed = channels, tuple(size), train, seed
        self.image_factors = IMAGE_FACTORS if image_factors is None else image_factors
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        name, f = self.files[i]
        meta = self.meta[name]
        d = np.load(f)
        rng = np.random.default_rng([self.seed, self.epoch, i] if self.train else [self.seed, i])
        n = int(d["mask_shape"][0])
        m = np.unpackbits(d["masks"], axis=0)[:n].astype(bool)
        idx = [meta["channels"].index(c) for c in self.channels]
        r0, r1 = (int(v) for v in d["crop_rows"])
        p = DR.sample(rng, set(self.image_factors), self.dr_overrides) if self.train else DR.nominal()
        img = DR.image_from_transmission(d["meanT"].astype(np.float32), p, meta["px_mm"], rng)[r0:r1]
        fov = d["fov"][r0:r1]
        ann = d["annotated"][idx]
        tgt = torch.from_numpy(m[idx, r0:r1].astype(np.float32))
        wgt = torch.from_numpy((fov[None] & ann[:, None, None]).astype(np.float32))
        img = torch.from_numpy(img)[None]
        if self.train:
            img, tgt, wgt = augment_geometry(img, tgt, wgt, self.channels, rng)
        return (_resize(img, self.size, "bilinear"), _resize(tgt, self.size, "nearest"),
                _resize(wgt, self.size, "nearest"), f.name)


@lru_cache(maxsize=1)
def _dentex_ann():
    zf = dio.open_zip("dentex", "train_zip")
    j = json.loads(zf.read("training_data/quadrant_enumeration/train_quadrant_enumeration.json"))
    by_img = {}
    for a in j["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    return {im["file_name"]: (im, by_img.get(im["id"], [])) for im in j["images"]}


def normalise_real(img: np.ndarray) -> np.ndarray:
    lo, hi = (float(v) for v in np.percentile(img, [0.5, 99.5]))
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


class DentexDataset(Dataset):
    """Real panoramics from DENTEX quadrant_enumeration with tooth polygons → teeth + FDI channels."""

    def __init__(self, names: list[str], channels: list[str], size=(256, 640), train: bool = False, seed: int = 0,
                 crop=REAL_CROP):
        ann = _dentex_ann()
        self.items = [n.split("/")[-1] for n in names if n.startswith("train:quadrant_enumeration/")]
        missing = [n for n in self.items if n not in ann]
        assert not missing, f"no annotations for {missing[:3]}"
        self.channels, self.size, self.train, self.seed, self.crop = channels, tuple(size), train, seed, crop
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        fname = self.items[i]
        im_meta, anns = _dentex_ann()[fname]
        zf = dio.open_zip("dentex", "train_zip")
        img = np.array(Image.open(zf.open(f"training_data/quadrant_enumeration/xrays/{fname}")).convert("L"))
        H, W = img.shape
        x0, y0, x1, y1 = int(self.crop[0] * W), int(self.crop[1] * H), int(self.crop[2] * W), int(self.crop[3] * H)
        masks = {c: Image.new("L", (W, H), 0) for c in ("teeth", *TOOTH_CHANNELS)}
        for a in anns:
            fdi = (a["category_id_1"] + 1) * 10 + (a["category_id_2"] + 1)
            for poly in a["segmentation"]:
                if len(poly) >= 6:
                    for c in ("teeth", f"tooth{fdi}"):
                        ImageDraw.Draw(masks[c]).polygon([float(v) for v in poly], fill=1)
        tgt = np.zeros((len(self.channels), y1 - y0, x1 - x0), np.float32)
        wgt = np.zeros_like(tgt)
        for k, c in enumerate(self.channels):
            if c in masks:                      # real ground truth exists only for teeth / FDI teeth
                tgt[k] = np.asarray(masks[c], np.float32)[y0:y1, x0:x1]
                wgt[k] = 1.0
        img = torch.from_numpy(normalise_real(img[y0:y1, x0:x1]))[None]
        tgt, wgt = torch.from_numpy(tgt), torch.from_numpy(wgt)
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            img, tgt, wgt = augment_geometry(img, tgt, wgt, self.channels, rng, crop_frac=(0.85, 1.0))
        return (_resize(img, self.size, "bilinear"), _resize(tgt, self.size, "nearest"),
                _resize(wgt, self.size, "nearest"), fname)
