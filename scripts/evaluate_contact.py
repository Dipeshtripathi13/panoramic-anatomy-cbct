"""Evaluate a crop model (task: crop) on the CONTACT official TEST set (444 third-molar crops).

Per image: Dice for molar and canal (CONTACT masks; scored at the model canvas, threshold 0.5), by CONTACT group
(1 = contact, 2 = no overlap / no contact, 3 = overlap / no contact), with 95% bootstrap CIs.
2D relation readout from the predicted masks (no extra training): predicted molar∩canal overlap
→ (a) overlap vs the GT-mask overlap, (b) AUC of the overlap fraction for the binary contact label.
Every metric is reported with and without resolution matching (CONTACT image → match_px before resizing).

Outputs in <run>/eval_contact_<ckpt>/: metrics.json, per_image.csv, overlays.png
Usage: python scripts/evaluate_contact.py --run runs/crop_v1 [--ckpt best]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import contact as C  # noqa: E402
from dentalpano import io as dio  # noqa: E402
from dentalpano.train.crops import ContactDataset  # noqa: E402
from dentalpano.train.model import build_model, device, dice_scores  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import bootstrap_ci  # noqa: E402


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann–Whitney AUC (ties count half)."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
    return float(gt)


def test_labels() -> dict[str, int]:
    zf = dio.open_zip("contact")
    df = pd.read_excel(zf.open(C.SPLITS["test"][2]))
    return {f"test/{n}": int(c) for n, c in zip(df.ImageNum, df.Class)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overlays", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    ck = torch.load(args.run / f"{args.ckpt}.pt", map_location="cpu", weights_only=False)
    cfg, channels = ck["cfg"], ck["channels"]
    assert channels == ["molar", "canal"], channels
    dev = device()
    model = build_model(len(channels), cfg.get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    out = args.run / f"eval_contact_{args.ckpt}"
    out.mkdir(exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    ids = json.loads((dio.REPO / "splits" / "contact.json").read_text())["test"][: args.limit]
    contact = test_labels()
    metrics = {"run": str(args.run), "ckpt": args.ckpt, "epoch": ck["epoch"], "git": git, "n": len(ids)}
    rows, shown = {}, []
    variants = {"matched": cfg.get("real_match_px") or 75, "native": None}
    for vname, mpx in variants.items():
        ds = ContactDataset(ids, cfg["size"], match_px=mpx)
        with torch.no_grad():
            for img, tgt, wgt, names in DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers):
                logits = model(img.to(dev)).float()
                d = dice_scores(logits, tgt.to(dev), wgt.to(dev)).numpy()
                p = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                t = tgt.numpy() > 0.5
                for k, n in enumerate(names):
                    gt_ov = int((t[k, 0] & t[k, 1]).sum())
                    pr_ov = int((p[k, 0] & p[k, 1]).sum())
                    r = rows.setdefault(n, {"id": n, "contact": contact[n]})
                    r["gt_overlap"] = int(gt_ov * (227 / cfg["size"]) ** 2 >= C.MIN_OVERLAP_PX)
                    r["group"] = 1 if r["contact"] == 1 else (2 if not r["gt_overlap"] else 3)
                    r.update({f"{vname}_dice_molar": float(d[k, 0]), f"{vname}_dice_canal": float(d[k, 1]),
                              f"{vname}_pred_overlap_px": pr_ov,
                              f"{vname}_pred_overlap_frac": pr_ov / max(int(p[k, 1].sum()), 1)})
                    if vname == "matched" and len(shown) < args.overlays:
                        shown.append((n, r["group"], img[k, 0].numpy(), t[k], p[k], float(d[k, 0]), float(d[k, 1])))
    rows = list(rows.values())
    with open(out / "per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    grp = np.array([r["group"] for r in rows])
    lab = np.array([r["contact"] for r in rows])
    gov = np.array([r["gt_overlap"] for r in rows])
    metrics["groups"] = {int(g): int((grp == g).sum()) for g in (1, 2, 3)}
    min_px = C.MIN_OVERLAP_PX * (cfg["size"] / 227) ** 2
    for vname in variants:
        m = {}
        for c in ("molar", "canal"):
            x = np.array([r[f"{vname}_dice_{c}"] for r in rows], float)
            m[f"dice_{c}"] = {"mean": round(float(np.nanmean(x)), 4), "ci95": bootstrap_ci(x),
                              **{f"group{g}": round(float(np.nanmean(x[grp == g])), 4) for g in (1, 2, 3)
                                 if (grp == g).any()}}
        pov = np.array([r[f"{vname}_pred_overlap_px"] for r in rows]) >= min_px
        frac = np.array([r[f"{vname}_pred_overlap_frac"] for r in rows])
        m["overlap_acc"] = round(float((pov == gov.astype(bool)).mean()), 4)
        m["contact_auc_from_overlap_frac"] = round(auc(frac, lab), 4)
        m["contact_auc_group1_vs_3"] = round(auc(frac[grp != 2], lab[grp != 2]), 4)
        metrics[vname] = m
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))

    if shown:
        cols = 4
        nr = int(np.ceil(len(shown) / cols))
        fig, axes = plt.subplots(nr, cols, figsize=(cols * 3.2, nr * 3.4))
        for ax, (n, g, im, t, p, dm, dc) in zip(np.ravel(axes), shown):
            ax.imshow(im, cmap="gray")
            for mask, col, ls in ((t[0], "lime", "-"), (t[1], "lime", "--"), (p[0], "yellow", "-"), (p[1], "red", "-")):
                if mask.any():
                    ax.contour(mask, levels=[0.5], colors=[col], linewidths=0.8, linestyles=ls)
            ax.set_title(f"{n} g{g}  molar {dm:.2f} canal {dc:.2f}", fontsize=7)
        for ax in np.ravel(axes):
            ax.axis("off")
        fig.suptitle("CONTACT test (matched res.) — green: GT molar (solid) / canal (dashed); yellow: pred molar; red: pred canal",
                     fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "overlays.png", dpi=110)
    print("→", out)


if __name__ == "__main__":
    main()
