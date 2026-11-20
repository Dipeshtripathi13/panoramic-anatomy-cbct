"""In-domain self-supervised pretraining baseline (reviewer item: label-efficient real-only baseline).

SimCLR on the 2,098 unlabelled DENTEX panoramics (duplicate groups disjoint from the test set; no labels used),
starting from the same ImageNet-initialised ResNet-34 encoder (1 input channel) as every other arm. Images use the same
fixed framing crop and canvas as DentexDataset and are cached in memory once. Two views per image: random resized
square crop (scale 0.2–1) → 224 px, horizontal flip, brightness/contrast/gamma jitter, Gaussian blur, Gaussian noise.
Projection head: global average pool → MLP 512-512-128; NT-Xent loss, temperature 0.2.

The output `encoder.pt` holds only encoder weights in the train.py checkpoint format, so
`init_from=<run>/encoder.pt` restores the encoder and leaves the decoder freshly initialised, as for ImageNet.

Usage: python scripts/pretrain_ssl.py --run runs/ssl_simclr [--epochs 400] [--batch 128]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dentalpano import io as dio
from dentalpano.train.cyclegan import DentexImages
from dentalpano.train.model import build_model, device


def augment(x: torch.Tensor, out_px: int = 224) -> torch.Tensor:
    """x: (B, 1, H, W) in [0, 1] on the device → one random view per image."""
    B, _, H, W = x.shape
    views = []
    for b in range(B):
        area = H * W * np.random.uniform(0.2, 1.0)
        side = int(min(math.sqrt(area), H, W))
        i, j = np.random.randint(0, H - side + 1), np.random.randint(0, W - side + 1)
        v = F.interpolate(x[b:b + 1, :, i:i + side, j:j + side], size=(out_px, out_px), mode="bilinear",
                          align_corners=False)
        if np.random.rand() < 0.5:
            v = v.flip(-1)
        views.append(v)
    v = torch.cat(views)
    c = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.6, 1.4)              # contrast
    m = v.mean((2, 3), keepdim=True)
    v = ((v - m) * c + m + torch.empty_like(c).uniform_(-0.2, 0.2)).clamp(0, 1)  # + brightness
    v = v ** torch.exp(torch.empty_like(c).uniform_(np.log(0.7), np.log(1.4)))   # gamma
    if np.random.rand() < 0.5:                                                     # blur (whole batch, σ random)
        s = np.random.uniform(0.1, 2.0)
        k = torch.arange(-4, 5, device=x.device, dtype=v.dtype)
        g = torch.exp(-k ** 2 / (2 * s * s)); g /= g.sum()
        v = F.conv2d(F.pad(v, (4, 4, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
        v = F.conv2d(F.pad(v, (0, 0, 4, 4), mode="reflect"), g.view(1, 1, -1, 1))
    v = v + torch.randn_like(v) * torch.empty_like(c).uniform_(0, 0.03)
    return v.clamp(0, 1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N images (smoke tests)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    (out / "config.json").write_text(json.dumps({**vars(args), "git": git}, indent=1))

    names = json.loads((dio.REPO / "splits" / "dentex.json").read_text())["unlabelled"][: args.limit]
    ds = DentexImages(names)
    t0 = time.time()
    cache = torch.stack([ds[i][0] for i in range(len(ds))])                      # (N, 1, 256, 640) in [0, 1]
    print(f"cached {len(cache)} images in {time.time() - t0:.0f} s", flush=True)

    dev = device()
    enc = build_model(1, "resnet34", True).encoder.to(dev)
    proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 128)).to(dev)
    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(cache) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.05)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    embed = lambda v: proj(enc(v)[-1].mean((2, 3)))
    log = open(out / "log.csv", "w")
    log.write("epoch,minutes,loss\n")
    for epoch in range(args.epochs):
        perm = torch.randperm(len(cache))
        tot = 0.0
        for s in range(steps_per_epoch):
            x = cache[perm[s * args.batch:(s + 1) * args.batch]].to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=use_amp):
                loss = nt_xent(embed(augment(x)), embed(augment(x)), args.tau)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        log.write(f"{epoch + 1},{(time.time() - t0) / 60:.2f},{tot / steps_per_epoch:.4f}\n")
        log.flush()
        if (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            print(f"epoch {epoch + 1}: loss {tot / steps_per_epoch:.4f}", flush=True)
            torch.save({"model": {f"encoder.{k}": v.cpu() for k, v in enc.state_dict().items()}, "channels": [],
                        "epoch": epoch + 1, "git": git}, out / "encoder.pt")
    print("→", out / "encoder.pt")


if __name__ == "__main__":
    main()
