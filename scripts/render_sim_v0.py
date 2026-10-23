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
from PIL import Image
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import contact, tf3  # noqa: E402
from dentalpano import io as dio  # noqa: E402
from dentalpano import simulator as S  # noqa: E402

OUT = dio.REPO / "outputs" / "sim_v0"
COLORS = {"teeth": "yellow", "pulp": "cyan", "canal_L": "red", "canal_R": "orange", "incisive_canal": "magenta",
          "sinus": "deepskyblue", "crown": "white", "bridge": "lightgray", "implant": "springgreen", "pharynx": "lime"}


def disp(a: np.ndarray, lo=1, hi=99.5) -> np.ndarray:
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / (p1 - p0), 0, 1)


def draw_masks(ax, masks):
    for k, c in COLORS.items():
        if masks[k].any():
            ax.contour(masks[k], levels=[0.5], colors=[c], linewidths=0.6)


def case_figure(case, r, path):
    fig, ax = plt.subplots(3, 1, figsize=(11, 13))
    ax[0].imshow(disp(r["mpr"]), cmap="gray")
    ax[0].set_title(f"{case}: PX2Tooth-style MPR (mean HU, ±{S.SimConfig().mpr_halfwidth_mm:.0f} mm slab)")
    ax[1].imshow(r["simpx"], cmap="gray", vmin=0, vmax=max(1e-3, np.percentile(r["simpx"], 99.8)))
    ax[1].set_title(f"SimPX Beer–Lambert (ray ±{r['ray_half_mm']:.0f} mm, K=200)")
    ax[2].imshow(r["simpx"], cmap="gray", vmin=0, vmax=max(1e-3, np.percentile(r["simpx"], 99.8)))
    draw_masks(ax[2], r["masks"])
    present = [k for k in COLORS if r["masks"][k].any()]
    ax[2].set_title("projected multi-label masks: " + ", ".join(f"{k} ({COLORS[k]})" for k in present), fontsize=8)
    for a in ax:
        a.set_xlabel("patient right  ←  arch arclength (0.3 mm/col)  →  patient left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def dentex_images(cfg, rng):
    zf = dio.open_zip("dentex", "train_zip")
    names = sorted(n for n in zf.namelist() if n.startswith(cfg["real"]["dentex_subset"]) and n.endswith(".png"))
    pick = [names[i] for i in sorted(rng.choice(len(names), cfg["real"]["n_dentex"], replace=False))]
    return [(n.split("/")[-1], np.array(Image.open(zf.open(n)).convert("L"))) for n in pick]


def molar_crop(r, side: str, crop_mm: float, vx: float):
    """Square crop centred between the lower third (or second) molar and its canal, like CONTACT ROIs."""
    canal = r["masks"]["canal_L" if side == "L" else "canal_R"]
    for tid in ((38, 37) if side == "L" else (48, 47)):
        m = r["masks"][f"tooth{tid}"]
        if m.any():
            break
    else:
        return None, None
    rc, cc = ndimage.center_of_mass(m)
    near = canal.copy()
    near[:, : max(0, int(cc) - 60)] = False
    near[:, int(cc) + 60:] = False
    if near.any():
        rc = 0.5 * (rc + ndimage.center_of_mass(near)[0])
    h = int(round(crop_mm / vx / 2))
    r0, c0 = int(rc) - h, int(cc) - h
    img = disp(r["simpx"])
    pad = np.pad(img, h)
    crop = pad[r0 + h: r0 + 3 * h, c0 + h: c0 + 3 * h]
    return np.array(Image.fromarray((crop * 255).astype(np.uint8)).resize((227, 227), Image.BILINEAR)), tid


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
    real = dentex_images(cfg, rng)
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
            crop, tid = molar_crop(r, side, cfg["real"]["contact_crop_mm"], simcfg.voxel_mm)
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
