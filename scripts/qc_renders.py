"""Quality-check a render set: contact sheet of random samples (image-level DR applied, masks, FOV)
plus basic stats (count, sizes, failed/empty masks).

Usage: python scripts/qc_renders.py --name dr_main [--n 12]   → data/renders/<name>/qc.png, qc.json
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import io as dio  # noqa: E402
from dentalpano import randomize as DR  # noqa: E402
from dentalpano.viz import draw_masks  # noqa: E402


def load(path, channels):
    d = np.load(path)
    m = np.unpackbits(d["masks"], axis=0)[: d["mask_shape"][0]].astype(bool)
    return d, {c: m[i] for i, c in enumerate(channels)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = dio.REPO / "data" / "renders" / args.name
    meta = json.loads((root / "index.json").read_text())
    files = sorted(root.glob("*_v*.npz"))
    rng = np.random.default_rng(args.seed)

    stats = {"files": len(files), "volumes": len({f.name.rsplit("_v", 1)[0] for f in files}),
             "no_teeth": [], "no_mandible": [], "fov_frac_lt_0.5": [], "nan": []}
    for f in files:
        d, m = load(f, meta["channels"])
        if not m["teeth"].any():
            stats["no_teeth"].append(f.name)
        if not m["mandible"].any():
            stats["no_mandible"].append(f.name)
        if d["fov"].mean() < 0.5:
            stats["fov_frac_lt_0.5"].append(f.name)
        if not np.isfinite(d["meanT"].astype(np.float32)).all():
            stats["nan"].append(f.name)
    (root / "qc.json").write_text(json.dumps(stats, indent=1))
    print({k: (v if isinstance(v, int) else len(v)) for k, v in stats.items()})

    pick = rng.choice(len(files), min(args.n, len(files)), replace=False)
    cols = 3
    rows = int(np.ceil(len(pick) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(8 * cols, 3.6 * rows), squeeze=False)
    for a, i in zip(ax.ravel(), pick):
        d, m = load(files[i], meta["channels"])
        p = DR.sample(rng)                           # fresh image-level DR, as in training
        img = DR.image_from_transmission(d["meanT"].astype(np.float32), p, meta["px_mm"], rng)
        r0, r1 = d["crop_rows"]
        a.imshow(img[r0:r1], cmap="gray", vmin=0, vmax=1)
        draw_masks(a, {k: v[r0:r1] for k, v in m.items()})
        if not d["fov"].all():
            a.contour(d["fov"][r0:r1], levels=[0.5], colors=["red"], linewidths=0.8)
        a.set_title(files[i].name, fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(root / "qc.png", dpi=60)
    print("wrote", root / "qc.png")


if __name__ == "__main__":
    main()
