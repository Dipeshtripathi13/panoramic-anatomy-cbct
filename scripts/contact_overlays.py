"""Step 4: CONTACT label table + mask overlays.

Writes outputs/contact/labels.csv and overlay PNGs: 5 random cases per group (seeded),
plus a panel of contact=1 cases whose masks show (near-)zero molar/canal overlap.

Usage: .venv/bin/python scripts/contact_overlays.py [--seed 0 --n 5]
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import contact  # noqa: E402
from dentalpano import io as dio  # noqa: E402

OUT = dio.REPO / "outputs" / "contact"
GROUP_NAMES = {1: "G1 overlap + contact", 2: "G2 no overlap, no contact (derived)",
               3: "G3 overlap, no contact (derived)"}


def overlay(img: np.ndarray, molar: np.ndarray, canal: np.ndarray) -> np.ndarray:
    rgb = np.repeat(img[..., None] / 255.0, 3, -1)
    for m, col in ((molar & ~canal, (1, 0.85, 0)), (canal & ~molar, (1, 0.15, 0.15)), (molar & canal, (1, 0, 1))):
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array(col)
    return rgb


def panel(zf, rows, title: str, path) -> None:
    fig, axes = plt.subplots(2, len(rows), figsize=(2.6 * len(rows), 5.6), squeeze=False)
    for c, r in enumerate(rows):
        case = contact.load_case(zf, r.split, r.ImageNum)
        axes[0, c].imshow(case["image"], cmap="gray")
        axes[1, c].imshow(overlay(case["image"], case["molar"], case["canal"]))
        axes[0, c].set_title(f"{r.split} #{r.ImageNum} (scheme {r.scheme})", fontsize=8)
        axes[1, c].set_title(f"contact={r.contact}  overlap={r.overlap_px}px", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(f"{title} — yellow = molar, red = canal, magenta = overlap", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    zf = dio.open_zip("contact")
    df = contact.label_table(zf)
    df.to_csv(OUT / "labels.csv", index=False)
    print(df.groupby(["split", "group"]).size().unstack(fill_value=0))

    rng = np.random.default_rng(args.seed)
    for g, name in GROUP_NAMES.items():
        sub = df[df.group == g]
        pick = sub.iloc[np.sort(rng.choice(len(sub), args.n, replace=False))]
        panel(zf, list(pick.itertuples()), name, OUT / f"group{g}.png")
    odd = df[(df.contact == 1) & ~df.overlap]
    panel(zf, list(odd.itertuples()), f"contact=1 but mask overlap < {contact.MIN_OVERLAP_PX}px ({len(odd)} cases)",
          OUT / "contact_without_mask_overlap.png")
    print("saved to", OUT)


if __name__ == "__main__":
    main()
