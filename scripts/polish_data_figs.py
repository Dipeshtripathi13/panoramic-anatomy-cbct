"""Journal versions of the dataset-image figures (outputs/paper/figures/*_DATA.png → *_DATA_paper.png).

The evaluation overlays carry debug titles (file names, per-image Dice) and a dozen panels each. This cuts the panels
out of those PNGs (panels are the large non-white blocks; title text rows are sparse and fall away) and recomposes a
subset with a proper legend, so no checkpoint or inference is needed. Dataset images → *_DATA, never committed.

Usage: python scripts/polish_data_figs.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

from dentalpano import io as dio

FIG = dio.REPO / "outputs" / "paper" / "figures"


def runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def panels(path, frac=0.4, min_len=40) -> list[list[np.ndarray]]:
    """Grid of panel crops (rows of columns) from a matplotlib image grid saved on white."""
    img = np.asarray(Image.open(path).convert("RGB"))
    ink = (img < 235).any(axis=2)
    out = []
    for r0, r1 in runs(ink.mean(1) > frac, min_len):
        out.append([img[r0:r1, c0:c1] for c0, c1 in runs(ink[r0:r1].mean(0) > frac, min_len)])
    return out


def compose(crops, ncols, name, handles=None, col_titles=None, width=7.2):
    nrows = int(np.ceil(len(crops) / ncols))
    ar = np.mean([c.shape[0] / c.shape[1] for c in crops])
    top = 0.22 if col_titles else 0.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, width / ncols * ar * nrows + top + (0.35 if handles else 0)),
                             squeeze=False, gridspec_kw={"wspace": 0.02, "hspace": 0.03})
    for ax, c in zip(axes.flat, crops):
        ax.imshow(c, interpolation="lanczos")
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, t in zip(axes[0], col_titles or []):
        ax.set_title(t, fontsize=8)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, -0.01), handlelength=1.8, columnspacing=1.2)
    fig.savefig(FIG / f"{name}_paper.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("→", FIG / f"{name}_paper.png", f"({len(crops)} panels)")


def line(color, label, ls="-"):
    return Line2D([], [], color=color, lw=1.5, ls=ls, label=label)


def main() -> None:
    g = panels(FIG / "fig_dentex_overlays_DATA.png")
    compose([c for row in g[:2] for c in row], 3, "fig_dentex_overlays_DATA",
            [line("#33cc33", "Ground-truth teeth"), line("#e6e600", "Predicted teeth"),
             line("#ff9900", "Predicted canal (right)"), line("#e62e2e", "Predicted canal (left)"),
             line("#33aaee", "Predicted sinus")])

    g = panels(FIG / "fig_contact_overlays_DATA.png")
    compose([c for row in g[:2] for c in row], 4, "fig_contact_overlays_DATA",
            [line("#33cc33", "Ground-truth molar"), line("#33cc33", "Ground-truth canal", "--"),
             line("#e6e600", "Predicted molar"), line("#e62e2e", "Predicted canal")])

    g = panels(FIG / "fig_cyclegan_samples_DATA.png", frac=0.5)
    compose([c for row in g[:3] for c in row], 4, "fig_cyclegan_samples_DATA",
            col_titles=["Synthetic", "Synthetic → real style", "Real", "Real → synthetic style"])


if __name__ == "__main__":
    main()
