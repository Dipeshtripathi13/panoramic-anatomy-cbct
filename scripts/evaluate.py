"""Evaluate a trained checkpoint on held-out TEST data (run once per final model; never used for selection).

Real:  DENTEX test images with tooth polygons (quadrant_enumeration subset), one image per duplicate group
       → per-image Dice for the teeth union and the 32 FDI channels (mean over teeth present in GT or prediction),
       with 95% bootstrap CIs over images.
Synth: TF3 test volumes from a nominal render set (default nominal_test_v2) → per-channel Dice.
Metrics are computed at the model canvas (cfg size, default 256×640), threshold 0.5.

Outputs in <run>/eval_<ckpt>/: metrics.json, real_per_image.csv, overlays_real.png
Usage: python scripts/evaluate.py --run runs/seg_v1_dr_full [--ckpt best] [--synth-set nominal_test_v2]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import io as dio  # noqa: E402
from dentalpano.train.data import TOOTH_CHANNELS, DentexDataset, SynthDataset  # noqa: E402
from dentalpano.train.model import build_model, device, dice_scores  # noqa: E402


def dentex_test_names() -> list[str]:
    """One quadrant_enumeration image per duplicate group among the test groups."""
    sp = json.loads((dio.REPO / "splits" / "dentex.json").read_text())
    test = set(sp["test"])
    out = []
    for grp in sp["groups"]:
        qe = sorted(n for n in grp if n in test and n.startswith("train:quadrant_enumeration/"))
        if qe:
            out.append(qe[0])
    return sorted(out)


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [round(float(v), 4) for v in np.percentile(means, [2.5, 97.5])]


@torch.no_grad()
def predict(model, loader, dev):
    for img, tgt, wgt, names in loader:
        logits = model(img.to(dev)).float()
        yield img, tgt, wgt, names, logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--synth-set", default="nominal_test_v2")
    ap.add_argument("--synth-split", default="test", help="TF3 split to evaluate ('all' = every volume; debugging)")
    ap.add_argument("--overlays", type=int, default=12)
    ap.add_argument("--limit", type=int, help="first N real test images only (debugging)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    ck = torch.load(args.run / f"{args.ckpt}.pt", map_location="cpu", weights_only=False)
    cfg, channels = ck["cfg"], ck["channels"]
    size = tuple(cfg["size"])
    dev = device()
    model = build_model(len(channels), cfg.get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    out = args.run / f"eval_{args.ckpt}"
    out.mkdir(exist_ok=True)
    git = __import__("subprocess").run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                       cwd=dio.REPO).stdout.strip()
    metrics = {"run": str(args.run), "ckpt": args.ckpt, "epoch": ck["epoch"], "git": git, "size": size}
    mk = lambda ds: DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # --- real: DENTEX test ---
    names = dentex_test_names()[: args.limit]
    real = DentexDataset(names, channels, size, train=False)
    ti, fdi_idx = channels.index("teeth"), [channels.index(c) for c in TOOTH_CHANNELS]
    rows, shown = [], []
    for img, tgt, wgt, fn, logits in predict(model, mk(real), dev):
        d = dice_scores(logits, tgt.to(dev), wgt.to(dev)).numpy()
        for k in range(len(fn)):
            fd = d[k, fdi_idx]
            rows.append({"image": fn[k], "dice_teeth": float(d[k, ti]),
                         "dice_fdi_mean": float(np.nanmean(fd)) if np.isfinite(fd).any() else float("nan"),
                         "n_teeth_gt": int((tgt[k, fdi_idx].sum((1, 2)) > 0).sum())})
            if len(shown) < args.overlays:
                p = torch.sigmoid(logits[k]).cpu().numpy() > 0.5
                shown.append((fn[k], img[k, 0].numpy(), tgt[k, ti].numpy() > 0.5,
                              {c: p[channels.index(c)] for c in ("teeth", "canal_L", "canal_R", "sinus")
                               if c in channels}))
    with open(out / "real_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for key in ("dice_teeth", "dice_fdi_mean"):
        x = np.array([r[key] for r in rows], float)
        metrics[f"real_{key}"] = {"mean": round(float(np.nanmean(x)), 4), "median": round(float(np.nanmedian(x)), 4),
                                  "ci95": bootstrap_ci(x), "n": int(np.isfinite(x).sum())}

    # --- synthetic: TF3 test volumes, nominal renders ---
    root = dio.REPO / "data" / "renders" / args.synth_set
    if (root / "index.json").exists():
        test_cases = (None if args.synth_split == "all"
                      else json.loads((dio.REPO / "splits" / "toothfairy3.json").read_text())[args.synth_split])
        synth = SynthDataset([args.synth_set], test_cases, channels, size, train=False)
        scores = [dice_scores(lg, t.to(dev), w.to(dev)) for _, t, w, _, lg in predict(model, mk(synth), dev)]
        s = torch.cat(scores).numpy()
        metrics["synth_set"], metrics["synth_split"], metrics["synth_n"] = args.synth_set, args.synth_split, len(synth)
        metrics["synth_dice"] = {c: round(float(np.nanmean(s[:, k])), 4) for k, c in enumerate(channels)
                                 if np.isfinite(s[:, k]).any()}
    else:
        print(f"[skip] synthetic test: {root} not found")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k != "synth_dice"}, indent=1))
    if "synth_dice" in metrics:
        print("synth test:", {c: metrics["synth_dice"][c] for c in ("teeth", "pulp", "canal_L", "canal_R",
                                                                     "incisive_canal", "sinus")
                              if c in metrics["synth_dice"]})

    # --- overlays: GT teeth (green) vs predicted teeth (yellow) + predicted canals/sinus (no real GT) ---
    if shown:
        cols = 3
        fig, axes = plt.subplots(int(np.ceil(len(shown) / cols)), cols, figsize=(cols * 6, 2.6 * np.ceil(len(shown) / cols)))
        colors = {"teeth": "yellow", "canal_L": "red", "canal_R": "orange", "sinus": "deepskyblue"}
        for ax, (fn, im, gt, pr) in zip(np.ravel(axes), shown):
            ax.imshow(im, cmap="gray")
            if gt.any():
                ax.contour(gt, levels=[0.5], colors=["lime"], linewidths=0.6)
            for c, m in pr.items():
                if m.any():
                    ax.contour(m, levels=[0.5], colors=[colors[c]], linewidths=0.6)
            r = next(r for r in rows if r["image"] == fn)
            ax.set_title(f"{fn}  teeth Dice {r['dice_teeth']:.2f}", fontsize=8)
        for ax in np.ravel(axes):
            ax.axis("off")
        fig.suptitle("DENTEX test — green: GT teeth; yellow: pred teeth; red/orange: pred canal L/R; blue: pred sinus",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "overlays_real.png", dpi=110)
    print("→", out)


if __name__ == "__main__":
    main()
