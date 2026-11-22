"""External real test on the Tufts Dental Database (1,000 panoramics; Panetta et al. 2022).

Same real-image protocol as DENTEX: fixed relative framing crop (REAL_CROP), percentile normalisation, resize to the
model canvas; ground truth is cropped with the same box and resized (nearest). Tufts tooth masks follow the tooth
contours (DENTEX polygons are coarse), so teeth Dice here measures boundary quality more fairly.
Per-tooth labels: teeth_polygon.json, Universal numbering → FDI (1–8 → 18–11, 9–16 → 21–28, 17–24 → 38–31,
25–32 → 41–48); primary teeth (A–T) have no model channel.
Main set: images with at least one permanent tooth and no primary teeth. Metrics as in evaluate.py: teeth Dice
(vs the Tufts teeth mask), mean FDI Dice over teeth present, detection, numbering accuracy, presence F1. Secondary: Dice
of the predicted jaw region (mandible ∪ maxilla ∪ teeth) vs the Tufts maxillomandibular mask, with the Dice of a
trivial all-foreground prediction as its floor.

Usage: python scripts/evaluate_tufts.py --run runs/seg_v1_dr_full [--ckpt best]
Output: <run>/eval_tufts_<ckpt>/{metrics.json, per_image.csv}
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import bootstrap_ci, ratio_ci, tooth_level  # noqa: E402

from dentalpano import io as dio  # noqa: E402
from dentalpano.train.data import FDI, REAL_CROP, TOOTH_CHANNELS, _resize, normalise_real  # noqa: E402
from dentalpano.train.model import build_model, device  # noqa: E402


def universal_to_fdi(u: int) -> int:
    if u <= 8:
        return 19 - u
    if u <= 16:
        return 12 + u
    if u <= 24:
        return 55 - u
    return 16 + u


class TuftsDataset(Dataset):
    def __init__(self, entries: list[dict], size):
        self.entries, self.size = entries, tuple(size)
        self.zr = self.zs = None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, k: int):
        if self.zr is None:                               # one zip handle per worker
            self.zr, self.zs = dio.open_zip("tufts", "radiographs"), dio.open_zip("tufts", "segmentation")
        e = self.entries[k]
        i = e["External ID"].split(".")[0]
        g = np.array(Image.open(io.BytesIO(self.zr.read(f"Radiographs/{i}.JPG"))).convert("L"))
        H, W = g.shape
        x0, y0, x1, y1 = int(REAL_CROP[0] * W), int(REAL_CROP[1] * H), int(REAL_CROP[2] * W), int(REAL_CROP[3] * H)
        mask = lambda sub: np.array(Image.open(io.BytesIO(self.zs.read(f"Segmentation/{sub}/{i}.jpg")))
                                    .convert("L")) > 127
        fdi = np.zeros((len(FDI), H, W), bool)
        for o in e["Label"]["objects"]:
            if not o["title"].isdigit():
                continue
            c = Image.new("1", (W, H))
            d = ImageDraw.Draw(c)
            for pg in o["polygons"]:
                if len(pg) >= 3:
                    d.polygon([tuple(p) for p in pg], fill=1)
            fdi[FDI.index(universal_to_fdi(int(o["title"])))] |= np.array(c)
        gt = np.concatenate([mask("teeth_mask")[None], mask("maxillomandibular")[None], fdi])[:, y0:y1, x0:x1]
        img = _resize(torch.from_numpy(normalise_real(g[y0:y1, x0:x1]))[None], self.size, "bilinear")
        gt = _resize(torch.from_numpy(gt.astype(np.float32)), self.size, "nearest") > 0.5
        return img, gt, i


def dice(a, b) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--translate", type=Path, help="CycleGAN checkpoint: translate real images to sim style (G_BA) "
                    "before segmenting (NeBLa-style test-time baseline) → eval_<ckpt>_r2s<tag>")
    ap.add_argument("--tag", default="", help="suffix of the r2s output folder (one per translator)")
    args = ap.parse_args()

    ck = torch.load(args.run / f"{args.ckpt}.pt", map_location="cpu", weights_only=False)
    cfg, channels = ck["cfg"], ck["channels"]
    dev = device()
    model = build_model(len(channels), cfg.get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    if args.translate:                          # real → sim translation at test time
        from dentalpano.train.cyclegan import Translated, load_generator
        model = Translated(load_generator(args.translate, dev, key="G_BA"), model).eval()
    ti, fdi_idx = channels.index("teeth"), [channels.index(c) for c in TOOTH_CHANNELS]
    jaw_idx = [channels.index(c) for c in ("teeth", "mandible", "maxilla")]

    poly = json.loads(dio.open_zip("tufts", "segmentation").read("Segmentation/teeth_polygon.json"))
    main_set = [e for e in poly if e["Label"]["objects"]
                and all(o["title"].isdigit() for o in e["Label"]["objects"])][: args.limit]
    loader = DataLoader(TuftsDataset(main_set, cfg["size"]), batch_size=args.batch_size, num_workers=args.num_workers)
    rows = []
    with torch.no_grad():
        for img, gt, ids in loader:
            out = model(img.to(dev))
            out = out[0] if isinstance(out, tuple) else out
            p = (torch.sigmoid(out.float()) > 0.5).cpu().numpy()
            gt = gt.numpy()
            for k in range(len(ids)):
                teeth_gt, jaw_gt, fdi_gt = gt[k, 0], gt[k, 1], gt[k, 2:]
                fd = [dice(p[k, fdi_idx[j]], fdi_gt[j]) for j in range(len(FDI)) if fdi_gt[j].any()]
                rows.append({"image": ids[k], "dice_teeth": dice(p[k, ti], teeth_gt),
                             "dice_fdi_mean": float(np.mean(fd)) if fd else float("nan"),
                             "dice_jaw": dice(p[k, jaw_idx].any(0), jaw_gt),
                             "dice_jaw_all_fg": dice(np.ones_like(jaw_gt), jaw_gt),
                             **tooth_level(p[k, fdi_idx], fdi_gt, p[k, ti])})

    out = args.run / f"eval_tufts_{args.ckpt}{'_r2s' + args.tag if args.translate else ''}"
    out.mkdir(exist_ok=True)
    with open(out / "per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    m = {"run": str(args.run), "ckpt": args.ckpt, "epoch": ck["epoch"], "git": git, "n_images": len(rows)}
    col = lambda key: np.array([r[key] for r in rows], float)
    for key in ("dice_teeth", "dice_fdi_mean", "dice_jaw", "dice_jaw_all_fg"):
        x = col(key)
        m[key] = {"mean": round(float(np.nanmean(x)), 4), "ci95": bootstrap_ci(x)}
    tp, fp, fn_ = col("pres_tp"), col("pres_fp"), col("pres_fn")
    for name, num, den in (("tooth_detection_rate", col("n_detected"), col("n_teeth_gt")),
                           ("tooth_numbering_acc", col("n_number_ok"), col("n_detected"))):
        m[name] = {"value": round(float(num.sum() / max(den.sum(), 1)), 4), "ci95": ratio_ci(num, den)}
    m["presence_f1"] = round(float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn_.sum(), 1)), 4)
    m["n_teeth_gt"] = int(col("n_teeth_gt").sum())
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
