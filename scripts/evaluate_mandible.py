"""External real-image test of a non-tooth structure: the mandible (Abdi & Kasaei, 116 panoramics, two dentists).

Each image gets exactly the zero-shot real-image protocol: the fixed relative framing crop (REAL_CROP), percentile
normalisation, resize to the model canvas. Ground truth (Segmentation1/2 = the image masked to the mandible → nonzero,
holes filled) is cropped with the same box and resized (nearest) to the canvas, so Dice is measured on the framed
region the model sees; rami and condyles outside the crop are not scored (ToothFairy3 rarely contains them).
Reported: Dice of the predicted `mandible` channel vs each dentist and the mean of the two, the two
dentists' agreement on the same region (ceiling), and the Dice of a trivial all-foreground prediction (the mandible
fills much of the framed region, so this floor is high), with 95% bootstrap CIs over images.

Usage: python scripts/evaluate_mandible.py --run runs/seg_v1_dr_full [--ckpt best] [--overlays 6]
Output: <run>/eval_mandible_<ckpt>/{metrics.json, per_image.csv, overlays.png}
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from dentalpano import io as dio
from dentalpano.train.data import REAL_CROP, _resize, normalise_real
from dentalpano.train.model import build_model, device


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(v), 4) for v in np.percentile(m, [2.5, 97.5])]


def load_case(zf, i: int, size):
    g = np.array(Image.open(io.BytesIO(zf.read(f"Images/{i}.png"))).convert("L"))
    H, W = g.shape
    x0, y0, x1, y1 = int(REAL_CROP[0] * W), int(REAL_CROP[1] * H), int(REAL_CROP[2] * W), int(REAL_CROP[3] * H)
    img = _resize(torch.from_numpy(normalise_real(g[y0:y1, x0:x1]))[None], size, "bilinear")
    gts = []
    for r in (1, 2):
        s = np.array(Image.open(io.BytesIO(zf.read(f"Segmentation{r}/{i}.png"))).convert("L")) > 0
        s = ndimage.binary_fill_holes(s)[y0:y1, x0:x1]
        gts.append(_resize(torch.from_numpy(s.astype(np.float32))[None], size, "nearest")[0].numpy() > 0.5)
    return img, gts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--overlays", type=int, default=6)
    ap.add_argument("--translate", type=Path, help="CycleGAN checkpoint: translate real images to sim style (G_BA) "
                    "before segmenting (NeBLa-style test-time baseline) → eval_<ckpt>_r2s")
    args = ap.parse_args()

    ck = torch.load(args.run / f"{args.ckpt}.pt", map_location="cpu", weights_only=False)
    cfg, channels = ck["cfg"], ck["channels"]
    size = tuple(cfg["size"])
    dev = device()
    model = build_model(len(channels), cfg.get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    if args.translate:                          # real → sim translation at test time
        from dentalpano.train.cyclegan import Translated, load_generator
        model = Translated(load_generator(args.translate, dev, key="G_BA"), model).eval()
    mi = channels.index("mandible")
    zf = dio.open_zip("abdi_mandible")
    ids = sorted(int(Path(n).stem) for n in zf.namelist() if n.startswith("Images/") and n.endswith(".png"))

    rows, shown = [], []
    for i in ids:
        img, (g1, g2) = load_case(zf, i, size)
        with torch.no_grad():
            out = model(img[None].to(dev))
            out = out[0] if isinstance(out, tuple) else out
        pred = (torch.sigmoid(out[0, mi]) > 0.5).cpu().numpy()
        full = np.ones_like(g1)                     # trivial baseline: the mandible fills much of the framed region
        rows.append({"id": i, "dice_r1": dice(pred, g1), "dice_r2": dice(pred, g2), "dice_r1_r2": dice(g1, g2),
                     "dice_all_fg": (dice(full, g1) + dice(full, g2)) / 2})
        if len(shown) < args.overlays:
            shown.append((img[0].numpy(), pred, g1))

    out_dir = args.run / f"eval_mandible_{args.ckpt}{'_r2s' if args.translate else ''}"
    out_dir.mkdir(exist_ok=True)
    keys = ["dice_r1", "dice_r2", "dice_r1_r2", "dice_all_fg"]
    with open(out_dir / "per_image.csv", "w") as f:
        f.write("id," + ",".join(keys) + "\n")
        for r in rows:
            f.write(f"{r['id']}," + ",".join(f"{r[k]:.4f}" for k in keys) + "\n")
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    mean_pred = (arr["dice_r1"] + arr["dice_r2"]) / 2
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    metrics = {"run": str(args.run), "ckpt": args.ckpt, "epoch": ck["epoch"], "git": git, "n": len(rows),
               "dice_mandible": {"mean": round(float(mean_pred.mean()), 4), "ci95": boot_ci(mean_pred)},
               "dice_vs_rater1": round(float(arr["dice_r1"].mean()), 4),
               "dice_vs_rater2": round(float(arr["dice_r2"].mean()), 4),
               "all_foreground_baseline_dice": round(float(arr["dice_all_fg"].mean()), 4),
               "interrater_dice": {"mean": round(float(arr["dice_r1_r2"].mean()), 4), "ci95": boot_ci(arr["dice_r1_r2"])}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))

    if shown:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(shown), 1, figsize=(7, 2.9 * len(shown)))
        for ax, (im, p, g) in zip(np.atleast_1d(axes), shown):
            ax.imshow(im, cmap="gray")
            ax.contour(g, [0.5], colors="#33cc33", linewidths=1)
            ax.contour(p, [0.5], colors="#e6e600", linewidths=1)
            ax.set_axis_off()
        fig.savefig(out_dir / "overlays.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
