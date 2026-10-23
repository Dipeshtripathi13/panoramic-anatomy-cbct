"""Preview the domain-randomisation factors (one factor at a time + fully random samples).

Outputs (outputs/dr_preview/): factors.png (rows = factors, cols = low / nominal / high),
random.png (fully randomised renders with masks), setA_fov.png, timing.json.
Panoramics are cropped vertically to the dentate region (top → mandible lower border + 8 mm),
which also removes the CBCT chin rest.

Usage: .venv/bin/python scripts/dr_preview.py [--case ToothFairy3F_010 --setA ToothFairy3P_001]
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import io as dio  # noqa: E402
from dentalpano import opg, tf3  # noqa: E402
from dentalpano import randomize as DR  # noqa: E402
from dentalpano.simulator import SimConfig, fit_arch  # noqa: E402
from dentalpano.viz import draw_masks  # noqa: E402

OUT = dio.REPO / "outputs" / "dr_preview"


def crop_rows(r: dict, margin_mm: float = 8.0) -> slice:
    mand = r["masks"].get("mandible")
    if mand is None or not mand.any():
        return slice(None)
    bottom = np.nonzero(mand.any(1))[0].max() + int(margin_mm / r["pixel_mm_at_trough"])
    return slice(0, min(bottom, mand.shape[0]))


def extremes(name: str) -> tuple[dict, dict]:
    """Low/high values of a factor's parameters (2nd/98th percentile of its sampler)."""
    rng = np.random.default_rng(0)
    lo, hi = {}, {}
    for p, (_, sampler) in DR.FACTORS[name][1].items():
        v = np.array([sampler(rng) for _ in range(2000)])
        lo[p], hi[p] = np.percentile(v, 2), np.percentile(v, 98)
    return lo, hi


def render(img, lab, arch, spine, p):
    cfg = DR.opg_config(p)
    mu = DR.attenuation(img, lab, p, spine)
    t0 = time.time()
    r = opg.render(None, lab, cfg, arch=DR.jitter_arch(arch, p), mu=mu)
    return r, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="ToothFairy3F_010")
    ap.add_argument("--setA", default="ToothFairy3P_001")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    base = opg.OPGConfig()

    img, _ = tf3.load_image(args.case)
    lab, _ = tf3.load_label(args.case)
    arch = fit_arch(lab, SimConfig(column_mm=base.column_mm, arch_extend_mm=base.arch_extend_mm))
    spine = DR.spine_mask(img, lab)
    nom = DR.nominal()
    r_nom, t_nom = render(img, lab, arch, spine, nom)
    rows = crop_rows(r_nom)
    px = base.column_mm
    times = [t_nom]
    show = lambda ax, im, title: (ax.imshow(im, cmap="gray", vmin=0, vmax=1), ax.set_title(title, fontsize=8), ax.axis("off"))

    names = list(DR.FACTORS)
    fig, ax = plt.subplots(len(names), 3, figsize=(18, 2.9 * len(names)))
    nom_img = DR.image_from_transmission(r_nom["meanT"], nom, px, rng)
    for i, name in enumerate(names):
        lo, hi = extremes(name)
        level = DR.FACTORS[name][0]
        for j, vals in ((0, lo), (2, hi)):
            p = {**nom, **vals}
            if level == "image":
                im = DR.image_from_transmission(r_nom["meanT"], p, px, rng)
                rr = r_nom
            else:
                rr, t = render(img, lab, arch, spine, p)
                times.append(t)
                im = DR.image_from_transmission(rr["meanT"], nom, px, rng)
            show(ax[i, j], im[crop_rows(rr)], f"{name} ({level}) " + ", ".join(f"{k}={v:.3g}" for k, v in vals.items()))
        show(ax[i, 1], nom_img[rows], f"{name}: nominal")
    fig.tight_layout()
    fig.savefig(OUT / "factors.png", dpi=55)
    plt.close(fig)

    fig, ax = plt.subplots(4, 1, figsize=(12, 18))
    samples = []
    for i in range(4):
        p = DR.sample(rng)
        rr, t = render(img, lab, arch, spine, p)
        times.append(t)
        im = DR.image_from_transmission(rr["meanT"], p, px, rng)
        sl = crop_rows(rr)
        ax[i].imshow(im[sl], cmap="gray", vmin=0, vmax=1)
        draw_masks(ax[i], {k: v[sl] for k, v in rr["masks"].items()})
        ax[i].set_title(f"random sample {i}", fontsize=9)
        ax[i].axis("off")
        samples.append({k: (None if not np.isfinite(v) else round(float(v), 4)) for k, v in p.items()})
    fig.tight_layout()
    fig.savefig(OUT / "random.png", dpi=60)
    plt.close(fig)
    del img, lab, spine

    imgA, _ = tf3.load_image(args.setA)
    labA, _ = tf3.load_label(args.setA)
    archA = fit_arch(labA, SimConfig(column_mm=base.column_mm, arch_extend_mm=base.arch_extend_mm))
    rA, t = render(imgA, labA, archA, DR.spine_mask(imgA, labA), nom)
    times.append(t)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(DR.image_from_transmission(rA["meanT"], nom, px, rng), cmap="gray", vmin=0, vmax=1)
    ax.contour(rA["fov"], levels=[0.5], colors=["red"], linewidths=1)
    draw_masks(ax, rA["masks"])
    ax.set_title(f"{args.setA} (Set A): red = field-of-view boundary; annotated channels: "
                 + ", ".join(k for k, v in rA["annotated"].items() if v and not k.startswith("tooth")), fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "setA_fov.png", dpi=70)
    plt.close(fig)

    json.dump({"render_seconds": [round(x, 1) for x in times], "mean_s": round(float(np.mean(times)), 1),
               "random_samples": samples, "spine_voxels": int(DR.spine_mask is not None)},
              open(OUT / "timing.json", "w"), indent=1)
    print("mean render s:", round(float(np.mean(times)), 1), "n =", len(times))


if __name__ == "__main__":
    main()
