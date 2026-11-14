"""Third-molar ROI crops (CONTACT-style) for canal / third-molar segmentation.

CONTACT ships 227×227 crops around a lower third molar with molar + canal masks (both sides, not flipped
to one orientation). Matching synthetic crops are cut from the full renders around tooth 38 (+ canal_L) or
48 (+ canal_R). Scale: CONTACT molars span ~172×140 px of 227 and synthetic lower third molars ~54×51 px at
0.3 mm, so a CONTACT crop ≈ 75 px ≈ 23 mm of the render (LOG.md); training jitters scale ±25% and position.
Both domains are intensity-normalised per crop (normalise_real) so they share the same input statistics.

Items are (image [1,S,S], target [2,S,S], weight [2,S,S], name, contact); channels = CROP_CHANNELS.
contact = 1/0 root↔canal contact (synthetic: 3D distance from scripts/contact_labels.py ≤ contact_mm;
CONTACT: the xlsx Class), -1 if unknown.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset

from .. import contact as C
from .. import io as dio
from .. import randomize as DR
from .data import IMAGE_FACTORS, _resize, augment_geometry, normalise_real

CROP_CHANNELS = ["molar", "canal"]
CONTACT_MM = 0.52     # labels touch, 26-connected at 0.3 mm (sqrt(3) * 0.3); see LOG.md


def contact3d(path: Path | None = None) -> dict:
    """case → side → min 3D tooth–canal distance (mm), from data/derived/contact3d.json (if present)."""
    path = path or dio.REPO / "data" / "derived" / "contact3d.json"
    if not path.exists():
        return {}
    return {c: {s: v["min_dist_mm"] for s, v in x.items() if v} for c, x in json.loads(path.read_text())["cases"].items()}


def contact_table() -> dict[str, int]:
    """'<official split>/<ImageNum>' → binary contact label, for every CONTACT crop."""
    import pandas as pd
    zf = dio.open_zip("contact")
    out = {}
    for split, (_, _, xlsx) in C.SPLITS.items():
        df = pd.read_excel(zf.open(xlsx))
        out.update({f"{split}/{n}": int(c) for n, c in zip(df.ImageNum, df.Class)})
    return out
SIDES = {"L": ("tooth38", "canal_L"), "R": ("tooth48", "canal_R")}


def _masks(d) -> np.ndarray:
    return np.unpackbits(d["masks"], axis=0)[: int(d["mask_shape"][0])].astype(bool)


def crop_centre(molar: np.ndarray, canal: np.ndarray, reach: int = 60) -> tuple[float, float]:
    """Molar centroid, pulled halfway (in rows) toward the canal segment beneath it (as viz.molar_crop)."""
    rc, cc = ndimage.center_of_mass(molar)
    near = canal.copy()
    near[:, : max(0, int(cc) - reach)] = False
    near[:, int(cc) + reach:] = False
    if near.any():
        rc = 0.5 * (rc + ndimage.center_of_mass(near)[0])
    return rc, cc


def cut(a: np.ndarray, r0: int, c0: int, s: int) -> np.ndarray:
    """a[r0:r0+s, c0:c0+s] with zero padding outside the array."""
    out = np.zeros((s, s), a.dtype)
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + s, a.shape[0]), min(c0 + s, a.shape[1])
    if re > rs and ce > cs:
        out[rs - r0: re - r0, cs - c0: ce - c0] = a[rs:re, cs:ce]
    return out


class SynthCropDataset(Dataset):
    def __init__(self, render_sets: list[str], cases: list[str] | None, size: int = 256, train: bool = True,
                 image_factors: list[str] | None = None, variants: list[int] | None = None, seed: int = 0,
                 crop_px: float = 75.0, scale=(0.8, 1.25), shift: float = 0.12, root: Path | None = None,
                 contact_mm: float = CONTACT_MM):
        root = root or dio.REPO / "data" / "renders"
        wanted = set(cases) if cases is not None else None
        self.meta, self.items, sides = {}, [], {}
        for name in render_sets:
            meta = json.loads((root / name / "index.json").read_text())
            self.meta[name] = meta
            for f in sorted((root / name).glob("*_v*.npz")):
                case, v = f.stem.rsplit("_v", 1)
                if (wanted is not None and case not in wanted) or (variants is not None and int(v) not in variants):
                    continue
                if case not in sides:              # labels are the same for every variant: check the first one
                    d = np.load(f)
                    m, ann, ch = _masks(d), d["annotated"], meta["channels"]
                    sides[case] = [s for s, (t, c) in SIDES.items()
                                   if ann[ch.index(t)] and ann[ch.index(c)] and m[ch.index(t)].sum() >= 50]
                self.items += [(name, f, s) for s in sides[case]]
        if not self.items:
            raise ValueError(f"no third-molar crops found in {render_sets}")
        self.size, self.train, self.seed = (size, size), train, seed
        self.image_factors = IMAGE_FACTORS if image_factors is None else image_factors
        self.crop_px, self.scale, self.shift = crop_px, scale, shift
        self.dist, self.contact_mm = contact3d(), contact_mm
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        name, f, side = self.items[i]
        meta = self.meta[name]
        d = np.load(f)
        m = _masks(d)
        rng = np.random.default_rng([self.seed, self.epoch, i] if self.train else [self.seed, i])
        tooth, canal_ch = SIDES[side]
        molar, canal = m[meta["channels"].index(tooth)], m[meta["channels"].index(canal_ch)]
        p = DR.sample(rng, set(self.image_factors)) if self.train else DR.nominal()
        img = DR.image_from_transmission(d["meanT"].astype(np.float32), p, meta["px_mm"], rng)
        rc, cc = crop_centre(molar, canal)
        s = self.crop_px
        if self.train:
            s *= float(np.exp(rng.uniform(*np.log(self.scale))))
            rc, cc = rc + rng.uniform(-self.shift, self.shift) * s, cc + rng.uniform(-self.shift, self.shift) * s
        s = int(round(s))
        r0, c0 = int(round(rc - s / 2)), int(round(cc - s / 2))
        fov = cut(d["fov"].astype(bool), r0, c0, s)
        im = normalise_real(cut(img, r0, c0, s)) * fov
        tgt = np.stack([cut(molar, r0, c0, s), cut(canal, r0, c0, s)]).astype(np.float32)
        wgt = np.broadcast_to(fov, tgt.shape).astype(np.float32)
        img_t, tgt_t, wgt_t = torch.from_numpy(im)[None], torch.from_numpy(tgt), torch.from_numpy(wgt.copy())
        if self.train and rng.random() < 0.5:     # CONTACT has both sides unflipped; channels are side-agnostic
            img_t, tgt_t, wgt_t = img_t.flip(-1), tgt_t.flip(-1), wgt_t.flip(-1)
        dist = self.dist.get(f.stem.rsplit("_v", 1)[0], {}).get(side)
        contact = -1.0 if dist is None else float(dist <= self.contact_mm)
        return (_resize(img_t, self.size, "bilinear"), _resize(tgt_t, self.size, "nearest"),
                _resize(wgt_t, self.size, "nearest"), f"{f.stem}_{side}", contact)


class ContactDataset(Dataset):
    """Real CONTACT crops; ids '<official split>/<ImageNum>' as in splits/contact.json.
    match_px: first downsample the image (area) to match_px × match_px, i.e. to the synthetic crop's pixel
    count (~0.3 mm pixels), so real input has the same resolution as the renders; masks stay full resolution."""

    def __init__(self, ids: list[str], size: int = 256, train: bool = False, seed: int = 0, match_px: int | None = None):
        self.ids, self.size, self.train, self.seed, self.match_px = list(ids), (size, size), train, seed, match_px
        self.labels = contact_table()
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        split, num = self.ids[i].split("/")
        c = C.load_case(dio.open_zip("contact"), split, int(num))
        img = torch.from_numpy(normalise_real(c["image"]))[None]
        if self.match_px:
            img = F.interpolate(img[None], size=(self.match_px, self.match_px), mode="area")[0]
        tgt = torch.from_numpy(np.stack([c["molar"], c["canal"]]).astype(np.float32))
        wgt = torch.ones_like(tgt)
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            img, tgt, wgt = augment_geometry(img, tgt, wgt, CROP_CHANNELS, rng, crop_frac=(0.85, 1.0))
        return (_resize(img, self.size, "bilinear"), _resize(tgt, self.size, "nearest"),
                _resize(wgt, self.size, "nearest"), self.ids[i], float(self.labels[self.ids[i]]))
