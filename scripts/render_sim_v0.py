"""Step 5: render simulator v0 on 3 Set B volumes and compare with real panoramics.

Outputs (outputs/sim_v0/): <case>.npz (mpr, simpx, mask_<channel>), <case>.png,
compare_dentex.png, compare_contact_crops.png, run.json (config + git hash).

Usage: .venv/bin/python scripts/render_sim_v0.py [--config configs/sim_v0.yaml]
"""
from __future__ import annotations

import argparse
import json
import subprocess

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import contact, tf3  # noqa: E402
from dentalpano import io as dio  # noqa: E402
from dentalpano import simulator as S  # noqa: E402
from dentalpano.viz import COLORS, dentex_images, disp, draw_masks, molar_crop  # noqa: E402

OUT = dio.REPO / "outputs" / "sim_v0"
def case_figure(case, r, path):
    fig, ax = plt.subplots(3, 1, figsize=(11, 13))
    ax[0].imshow(disp(r["mpr"]), cmap="gray")
    ax[0].set_title(f"{case}: PX2Tooth-style MPR (mean HU, ±{S.SimConfig().mpr_halfwidth_mm:.0f} mm slab)")
    ax[1].imshow(r["simpx"], cmap="gray", vmin=0, vmax=max(1e-3, np.percentile(r["simpx"], 99.8)))
    ax[1].set_title(f"SimPX Beer–Lambert (ray ±{r['ray_half_mm']:.0f} mm, K=200)")
    ax[2].imshow(r["simpx"], cmap="gray", vmin=0, vmax=max(1e-3, np.percentile(r["simpx"], 99.8)))
    present = draw_masks(ax[2], r["masks"])
    ax[2].set_title("projected multi-label masks: " + ", ".join(f"{k} ({COLORS[k]})" for k in present), fontsize=8)
    for a in ax:
        a.set_xlabel("patient right  ←  arch arclength (0.3 mm/col)  →  patient left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(dio.REPO / "configs" / "sim_v0.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    simcfg = S.SimConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["sim"].items()})
    for tid in (37, 38, 47, 48):
        simcfg.channels[f"tooth{tid}"] = ([tid], 12.0)

    results = {}
    for case in cfg["cases"]:
        img, _ = tf3.load_image(case)
        lab, _ = tf3.load_label(case)
        r = S.render(img, lab, simcfg)
        del img, lab
        np.savez_compressed(OUT / f"{case}.npz", mpr=r["mpr"], simpx=r["simpx"],
                            **{f"mask_{k}": v for k, v in r["masks"].items()})
        case_figure(case, r, OUT / f"{case}.png")
        results[case] = r
        print(case, r["simpx"].shape, f"arch {r['arch']['length_mm']:.0f} mm",
              {k: int(v.sum()) for k, v in r["masks"].items() if not k.startswith("tooth")})

    # ---- side by side with real DENTEX panoramics
    real = dentex_images(cfg["real"]["dentex_subset"], cfg["real"]["n_dentex"], rng)
    fig, ax = plt.subplots(len(results), 3, figsize=(24, 4.4 * len(results)))
    for row, ((case, r), (rname, rimg)) in enumerate(zip(results.items(), real)):
        ax[row, 0].imshow(disp(r["mpr"]), cmap="gray")
        ax[row, 0].set_title(f"{case} — MPR (PX2Tooth-style)")
        ax[row, 1].imshow(r["simpx"], cmap="gray", vmin=0, vmax=np.percentile(r["simpx"], 99.8))
        draw_masks(ax[row, 1], r["masks"])
        ax[row, 1].set_title(f"{case} — SimPX + projected masks")
        ax[row, 2].imshow(rimg, cmap="gray")
        ax[row, 2].set_title(f"real DENTEX {rname} ({rimg.shape[1]}×{rimg.shape[0]})")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "compare_dentex.png", dpi=60)
    plt.close(fig)

    # ---- third-molar crops vs CONTACT ROIs
    zf = dio.open_zip("contact")
    table = contact.label_table(zf)
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.4))
    k = 0
    for case, r in results.items():
        for side in ("R", "L"):
            crop, tid = molar_crop(r["simpx"], r["masks"], side, cfg["real"]["contact_crop_mm"], simcfg.voxel_mm)
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
    fig.suptitle(f"Top: simulated lower 2nd/3rd-molar ROIs ({cfg['real']['contact_crop_mm']:.0f} mm, resized to 227²). "
                 "Bottom: real CONTACT ROIs (227²)")
    fig.tight_layout()
    fig.savefig(OUT / "compare_contact_crops.png", dpi=70)
    plt.close(fig)

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=dio.REPO).stdout)
    json.dump({"config": cfg, "git": git, "dirty": dirty, "dentex": [n for n, _ in real]},
              open(OUT / "run.json", "w"), indent=2)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
