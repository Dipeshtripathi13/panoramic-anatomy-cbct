"""Supervised in-domain reference for the mandible external test (Abdi & Kasaei, 116 panoramics, two dentists).

5-fold cross-validation over splits/abdi_mandible_folds.json: train on four folds (or `--n-train` images drawn from
them), test on the held-out fold, so every image is tested exactly once per arm. Same real-image protocol as
evaluate_mandible.py (fixed framing crop, normalisation, 256×640 canvas, region outside the crop not scored).
Training label = dentist 1; Dice is reported against the mean of both dentists, as for the zero-shot models.

Arms (`--init`): `imagenet` = U-Net/ResNet-34 from ImageNet; a checkpoint path = the zero-shot model, whose
`mandible` head initialises the single output channel (sim→real). Both arms use the same schedule (AdamW lr 1e-3,
one-cycle, batch 8, a fixed number of steps, random crop 85–100% + horizontal flip + intensity jitter) and keep the
last checkpoint: there is no validation split, so nothing is selected on the test fold.

Usage: python scripts/train_mandible_cv.py --fold 0 --init imagenet [--n-train 5] --out runs/mandcv_imagenet_n5_f0
Output: <out>/{config.json, per_image.csv, metrics.json}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_mandible import dice, load_case  # noqa: E402
from train import init_weights  # noqa: E402

from dentalpano import io as dio  # noqa: E402
from dentalpano.train.model import build_model, device  # noqa: E402

SIZE = (256, 640)


def augment(img: torch.Tensor, gt: torch.Tensor, rng) -> tuple[torch.Tensor, torch.Tensor]:
    _, H, W = img.shape
    h, w = int(H * rng.uniform(0.85, 1.0)), int(W * rng.uniform(0.85, 1.0))
    y, x = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
    img = F.interpolate(img[None, :, y:y + h, x:x + w], size=SIZE, mode="bilinear", align_corners=False)[0]
    gt = F.interpolate(gt[None, :, y:y + h, x:x + w], size=SIZE, mode="nearest")[0]
    if rng.random() < 0.5:                       # the mandible is one structure: no left/right channel swap
        img, gt = img.flip(-1), gt.flip(-1)
    m = img.mean()
    img = ((img - m) * rng.uniform(0.8, 1.2) + m + rng.uniform(-0.1, 0.1)).clamp(0, 1) ** rng.uniform(0.8, 1.25)
    return img, gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--init", default="imagenet", help="'imagenet' or a zero-shot best.pt")
    ap.add_argument("--n-train", type=int, default=0, help="0 = all images of the four training folds")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    (args.out / "config.json").write_text(json.dumps({**vars(args), "out": str(args.out), "git": git}, indent=1))

    folds = json.loads((dio.REPO / "splits" / "abdi_mandible_folds.json").read_text())["folds"]
    test_ids = folds[args.fold]
    train_ids = sorted(i for k, f in enumerate(folds) if k != args.fold for i in f)
    if args.n_train:
        train_ids = sorted(np.random.default_rng(args.seed).choice(train_ids, args.n_train, replace=False).tolist())
    zf = dio.open_zip("abdi_mandible")
    train = []
    for i in train_ids:
        img, (g1, _) = load_case(zf, i, SIZE)
        train.append((img, torch.from_numpy(g1.astype(np.float32))[None]))

    dev = device()
    model = build_model(1, "resnet34", pretrained=args.init == "imagenet")
    if args.init != "imagenet":
        init_weights(model, torch.load(args.init, map_location="cpu", weights_only=False), ["mandible"],
                     {"mandible": ["mandible"]})
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    for step in range(args.steps):
        batch = [augment(*train[k], rng) for k in rng.integers(0, len(train), args.batch)]
        x, y = torch.stack([b[0] for b in batch]).to(dev), torch.stack([b[1] for b in batch]).to(dev)
        logits = model(x)
        p = torch.sigmoid(logits)
        soft_dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
        loss = F.binary_cross_entropy_with_logits(logits, y) + soft_dice.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0:
            print(f"step {step + 1}: loss {loss.item():.4f}", flush=True)

    model.eval()
    rows = []
    with torch.no_grad():
        for i in test_ids:
            img, (g1, g2) = load_case(zf, i, SIZE)
            pred = (torch.sigmoid(model(img[None].to(dev))[0, 0]) > 0.5).cpu().numpy()
            rows.append({"id": i, "dice_r1": dice(pred, g1), "dice_r2": dice(pred, g2)})
    with open(args.out / "per_image.csv", "w") as f:
        f.write("id,dice_r1,dice_r2\n")
        f.writelines(f"{r['id']},{r['dice_r1']:.4f},{r['dice_r2']:.4f}\n" for r in rows)
    d = np.array([(r["dice_r1"] + r["dice_r2"]) / 2 for r in rows])
    m = {"fold": args.fold, "init": args.init, "n_train": len(train_ids), "n_test": len(rows), "git": git,
         "dice_mandible": round(float(d.mean()), 4)}
    (args.out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m))


if __name__ == "__main__":
    main()
