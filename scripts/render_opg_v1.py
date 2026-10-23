"""Render the rotational OPG simulator (v1) on Set B volumes and compare with real panoramics.

Outputs (outputs/opg_v1/): <case>.npz (image, meanT, mask_*, soft_*), <case>.png,
compare_dentex.png, compare_v0_v1.png, compare_contact_crops.png, run.json.

Usage: .venv/bin/python scripts/render_opg_v1.py [--config configs/opg_v1.yaml]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import contact, opg, tf3  # noqa: E402
from dentalpano import io as dio  # noqa: E402
from dentalpano.viz import dentex_images, disp, draw_masks, molar_crop  # noqa: E402

OUT = dio.REPO / "outputs" / "opg_v1"


def show(ax, r, masks=False, title="", key="processed"):
    ax.imshow(r[key] if key == "processed" else disp(r[key], 0.5, 99.8), cmap="gray")
    if masks:
        draw_masks(ax, r["masks"])
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(dio.REPO / "configs" / "opg_v1.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    ocfg = opg.OPGConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["opg"].items()})
    for tid in (37, 38, 47, 48):
        ocfg.channels[f"tooth{tid}"] = ([tid, tid + 100], 12.0)

    results, timing = {}, {}
    for case in cfg["cases"]:
        img, _ = tf3.load_image(case)
        lab, _ = tf3.load_label(case)
        t0 = time.time()
        r = opg.render(img, lab, ocfg)
        timing[case] = round(time.time() - t0, 1)
        del img, lab
        np.savez_compressed(OUT / f"{case}.npz", image=r["image"], meanT=r["meanT"], processed=r["processed"],
                            **{f"mask_{k}": v for k, v in r["masks"].items()},
                            **{f"soft_{k}": v.astype(np.float16) for k, v in r["soft_masks"].items()})
        fig, ax = plt.subplots(3, 1, figsize=(12, 15))
        show(ax[0], r, key="image", title=f"{case} — rotational OPG v1, raw −log(mean T) ({timing[case]} s)")
        show(ax[1], r, title="processed (background subtraction σ=12 mm, 85%)")
        show(ax[2], r, masks=True, title="projected multi-label masks (±trough depth window, ≥50% of exposure)")
        fig.tight_layout()
        fig.savefig(OUT / f"{case}.png", dpi=80)
        plt.close(fig)
        results[case] = r
        print(case, r["image"].shape, f"{timing[case]} s",
              {k: int(v.sum()) for k, v in r["masks"].items() if not k.startswith("tooth")})

    real = dentex_images(cfg["real"]["dentex_subset"], cfg["real"]["n_dentex"], rng)
    fig, ax = plt.subplots(len(results), 3, figsize=(24, 4.6 * len(results)))
    for row, ((case, r), (rname, rimg)) in enumerate(zip(results.items(), real)):
        show(ax[row, 0], r, title=f"{case} — OPG v1")
        show(ax[row, 1], r, masks=True, title=f"{case} — OPG v1 + masks")
        ax[row, 2].imshow(rimg, cmap="gray")
        ax[row, 2].set_title(f"real DENTEX {rname}", fontsize=9)
        ax[row, 2].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "compare_dentex.png", dpi=60)
    plt.close(fig)

    case0 = cfg["cases"][0]
    v0 = OUT.parent / "sim_v0" / f"{case0}.npz"
    if v0.exists():
        d = np.load(v0)
        fig, ax = plt.subplots(3, 1, figsize=(12, 15))
        ax[0].imshow(disp(d["mpr"]), cmap="gray")
        ax[0].set_title(f"{case0} — v0 MPR (no geometry)")
        ax[1].imshow(d["simpx"], cmap="gray", vmin=0, vmax=np.percentile(d["simpx"], 99.8))
        ax[1].set_title("v0 SimPX (normal rays, no TDI)")
        show(ax[2], results[case0], title="v1 rotational OPG (TDI, rotating centre, tilt, magnification)")
        for a in ax[:2]:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / "compare_v0_v1.png", dpi=70)
        plt.close(fig)

    zf = dio.open_zip("contact")
    table = contact.label_table(zf)
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.4))
    k = 0
    for case, r in results.items():
        for side in ("R", "L"):
            crop, tid = molar_crop(r["processed"], r["masks"], side, cfg["real"]["contact_crop_mm"], ocfg.column_mm)
            if crop is not None:
                ax[0, k].imshow(crop, cmap="gray")
                ax[0, k].set_title(f"sim {case[-6:]} tooth {tid}", fontsize=9)
            k += 1
    for j, g in enumerate((1, 1, 2, 2, 3, 3)):
        sub = table[table.group == g]
        rr = sub.iloc[rng.integers(len(sub))]
        ax[1, j].imshow(contact.load_case(zf, rr.split, rr.ImageNum)["image"], cmap="gray")
        ax[1, j].set_title(f"CONTACT {rr.split} #{rr.ImageNum} (G{g})", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("Top: simulated lower-molar ROIs (OPG v1). Bottom: real CONTACT ROIs")
    fig.tight_layout()
    fig.savefig(OUT / "compare_contact_crops.png", dpi=70)
    plt.close(fig)

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=dio.REPO).stdout)
    json.dump({"config": cfg, "git": git, "dirty": dirty, "dentex": [n for n, _ in real], "seconds": timing},
              open(OUT / "run.json", "w"), indent=2)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
