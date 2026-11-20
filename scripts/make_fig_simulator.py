"""Paper Fig. 1: one ToothFairy3 volume through the simulator.

(a) nominal render, (b) the same render with its projected, overlapping labels (FDI teeth, pulp, canals, incisive
canal, sinus, restorations), (c) three randomized training renders of the same volume (geometry + volume factors fixed
at render time, image factors drawn as in training). Shows dataset-derived images → saved as *_DATA (never committed).

Usage: python scripts/make_fig_simulator.py [--case ToothFairy3F_003] [--set local_train]
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from dentalpano import io as dio
from dentalpano import randomize as DR
from dentalpano.train.data import IMAGE_FACTORS

OUT = dio.REPO / "outputs" / "paper" / "figures"
# structure colours (outline + light fill); teeth get a cycling 4-tone palette by FDI position
STRUCT = [("canal_L", "#e8423f", "Inferior alveolar canal"), ("canal_R", "#e8423f", None),
          ("incisive_canal", "#f2a516", "Incisive canal"), ("sinus", "#2fb5c9", "Maxillary sinus"),
          ("pulp", "#c84fd1", "Pulp"), ("crown", "#ffffff", "Crown / bridge / implant"),
          ("bridge", "#ffffff", None), ("implant", "#ffffff", None)]
TOOTH = ["#2a78d6", "#1baf7a", "#7b61d9", "#3fa3e0"]


def image(d, meta, p, rng):
    r0, r1 = (int(v) for v in d["crop_rows"])
    return DR.image_from_transmission(d["meanT"].astype(np.float32), p, meta["px_mm"], rng)[r0:r1], (r0, r1)


def masks(d, meta, rows):
    n = int(d["mask_shape"][0])
    m = np.unpackbits(d["masks"], axis=0)[:n].astype(bool)[:, rows[0]:rows[1]]
    return {c: m[i] for i, c in enumerate(meta["channels"])}


def overlay(ax, img, m):
    rgb = np.repeat(img[..., None], 3, axis=2).astype(np.float32)
    def paint(mask, hexcol, a):
        col = np.array([int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        rgb[mask] = (1 - a) * rgb[mask] + a * col
    for q in range(1, 5):                                # FDI teeth, colour by position in the quadrant
        for k in range(1, 9):
            c = f"tooth{q}{k}"
            if c in m and m[c].any():
                paint(m[c], TOOTH[(k - 1) % 4], 0.35)
    for c, col, _ in STRUCT:
        if m[c].any():
            paint(m[c], col, 0.55)
    ax.imshow(np.clip(rgb, 0, 1), interpolation="bilinear")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="ToothFairy3F_009")
    ap.add_argument("--set", default="local_train")
    a = ap.parse_args()
    root = dio.REPO / "data" / "renders" / a.set
    meta = json.loads((root / "index.json").read_text())
    d0 = np.load(root / f"{a.case}_v0.npz")
    img0, rows = image(d0, meta, DR.nominal(), np.random.default_rng(0))
    m0 = masks(d0, meta, rows)
    variants = sorted(root.glob(f"{a.case}_v[1-9].npz"))
    rand = []
    for j, f in enumerate((variants * 3)[:3]):           # geometry/volume factors are baked into each render;
        rng = np.random.default_rng([7, j])              # image factors are drawn here, as in training
        rand.append(image(np.load(f), meta, DR.sample(rng, set(IMAGE_FACTORS)), rng)[0])

    ar = img0.shape[0] / img0.shape[1]                  # rows are sized to the image aspect → no dead space
    fig = plt.figure(figsize=(7.2, 7.2 * ar * (1 / 2 + 1 / 3) + 0.75))
    gs = fig.add_gridspec(2, 6, height_ratios=[3, 2], hspace=0.16, wspace=0.03)
    axes = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])] + [fig.add_subplot(gs[1, 2 * i:2 * i + 2]) for i in range(3)]
    axes[0].imshow(img0, cmap="gray", interpolation="bilinear")
    overlay(axes[1], img0, m0)
    for ax, im in zip(axes[2:], rand):
        ax.imshow(im, cmap="gray", interpolation="bilinear")
    for ax, t in zip(axes, ["(a) Nominal render", "(b) Projected labels (overlapping)", "(c) Randomized renders",
                            "", ""]):
        ax.set_axis_off()
        if t:
            ax.set_title(t, fontsize=8, loc="left")
    groups = {}                                          # legend: one entry per label, only if present here
    for c, col, lab in STRUCT:
        groups.setdefault(lab or last, []).append(c)
        last = lab or last
    handles = [Patch(color=TOOTH[0], label="Teeth (32 FDI channels)")] + \
              [Patch(color=dict((l, c) for _, c, l in STRUCT if l)[lab].replace("#ffffff", "#d9d9d9"), label=lab)
               for lab, cs in groups.items() if any(m0[c].any() for c in cs)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_simulator_DATA.png", dpi=300, bbox_inches="tight")
    print("→", OUT / "fig_simulator_DATA.png")


if __name__ == "__main__":
    main()
