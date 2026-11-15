"""Train the CycleGAN baseline: synthetic panoramics (A) ↔ real DENTEX panoramics (B), unpaired.

A = renders of TF3 TRAIN volumes (image-level DR off: nominal appearance, so the GAN alone supplies realism);
B = DENTEX 'unlabelled' split (duplicate groups disjoint from the test set; no labels used). Random square crops for
training; the generator is fully convolutional and is applied to whole 256×640 panoramics afterwards.

Outputs in <out_root>/<run_name>/: config.json (+ git), log.csv, cyclegan.pt (G_AB, G_BA, arch), samples_<it>.png
Usage: python scripts/train_cyclegan.py --config configs/train/cyclegan_v1.yaml [--set key=value ...]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import io as dio  # noqa: E402
from dentalpano.train.cyclegan import DentexImages, Discriminator, Generator, ImagePool  # noqa: E402
from dentalpano.train.data import SynthDataset  # noqa: E402
from dentalpano.train.model import device  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import parse_overrides  # noqa: E402


def forever(loader):
    while True:
        yield from loader


def rand_crop(x: torch.Tensor, s: int) -> torch.Tensor:
    """Independent random s×s crop (and horizontal flip) per image of a [B,1,H,W] batch."""
    out = []
    for im in x:
        r, c = random.randint(0, im.shape[-2] - s), random.randint(0, im.shape[-1] - s)
        im = im[:, r:r + s, c:c + s]
        out.append(im.flip(-1) if random.random() < 0.5 else im)
    return torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action="append", default=[])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg.update(parse_overrides([kv for group in args.set for kv in (group or [])]))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    out = Path(cfg.get("out_root", dio.REPO / "outputs" / "runs")) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "log.csv").exists() and not cfg.get("overwrite", False):
        raise SystemExit(f"{out} already has a log.csv — pick another run_name or pass overwrite=true")
    (out / "log.csv").unlink(missing_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    (out / "config.json").write_text(json.dumps({**cfg, "git": git}, indent=1, default=str))

    split_tf3 = json.loads((dio.REPO / "splits" / "toothfairy3.json").read_text())
    split_dx = json.loads((dio.REPO / "splits" / "dentex.json").read_text())
    size = tuple(cfg["size"])
    synth = SynthDataset(cfg["synth_sets"], split_tf3["train"], ["teeth"], size, train=True,
                         image_factors=cfg.get("image_factors", []), seed=cfg["seed"])
    real = DentexImages(split_dx["unlabelled"], size)
    nw, bs = cfg.get("num_workers", 4), cfg["batch_size"]
    mk = lambda ds: DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True,
                               persistent_workers=nw > 0)
    it_a, it_b = forever(mk(synth)), forever(mk(real))
    fixed_a = torch.stack([SynthDataset(cfg.get("sample_sets", cfg["synth_sets"]), split_tf3["val"], ["teeth"], size,
                                        train=False, variants=[0])[i][0] for i in range(4)])
    fixed_b = torch.stack([real[i][0] for i in range(4)])

    dev = device()
    arch = {"ngf": cfg.get("ngf", 64), "n_blocks": cfg.get("n_blocks", 9)}
    G_AB, G_BA = Generator(**arch).to(dev), Generator(**arch).to(dev)
    D_A, D_B = Discriminator().to(dev), Discriminator().to(dev)
    opt_g = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=cfg["lr"], betas=(0.5, 0.999))
    n_it, decay_from = cfg["iterations"], cfg["iterations"] // 2        # constant lr, then linear decay to 0
    lam = lambda i: 1.0 if i < decay_from else max(0.0, 1 - (i - decay_from) / (n_it - decay_from))
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lam)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lam)
    pool_a, pool_b = ImagePool(), ImagePool()
    mse, l1 = torch.nn.MSELoss(), torch.nn.L1Loss()
    lc, li, s = cfg.get("lambda_cycle", 10.0), cfg.get("lambda_identity", 0.5), cfg["crop"]
    gan = lambda pred, real_label: mse(pred, torch.full_like(pred, float(real_label)))

    t0, acc = time.time(), []
    for it in range(n_it):
        a = rand_crop(next(it_a)[0], s).to(dev)
        b = rand_crop(next(it_b)[0], s).to(dev)
        # generators
        fake_b, fake_a = G_AB(a), G_BA(b)
        loss_gan = gan(D_B(fake_b), 1) + gan(D_A(fake_a), 1)
        loss_cyc = l1(G_BA(fake_b), a) + l1(G_AB(fake_a), b)
        loss_id = l1(G_AB(b), b) + l1(G_BA(a), a)
        loss_g = loss_gan + lc * loss_cyc + lc * li * loss_id
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        # discriminators
        pb, pa = pool_b.query(fake_b), pool_a.query(fake_a)
        loss_d = 0.5 * (gan(D_B(b), 1) + gan(D_B(pb), 0)) + 0.5 * (gan(D_A(a), 1) + gan(D_A(pa), 0))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
        sch_g.step()
        sch_d.step()
        acc.append([loss_gan.item(), loss_cyc.item(), loss_id.item(), loss_d.item()])

        if (it + 1) % cfg.get("log_every", 500) == 0 or it + 1 == n_it:
            m = np.mean(acc, 0)
            acc = []
            row = {"it": it + 1, "minutes": round((time.time() - t0) / 60, 2), "lr": sch_g.get_last_lr()[0],
                   "g_gan": m[0], "cycle": m[1], "identity": m[2], "d": m[3]}
            new = not (out / "log.csv").exists()
            with open(out / "log.csv", "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if (it + 1) % cfg.get("save_every", 5000) == 0 or it + 1 == n_it:
            torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "arch": arch, "it": it + 1, "cfg": cfg,
                        "git": git}, out / "cyclegan.pt")
            G_AB.eval(), G_BA.eval()
            with torch.no_grad():
                ta, tb = G_AB(fixed_a.to(dev)).cpu(), G_BA(fixed_b.to(dev)).cpu()
            G_AB.train(), G_BA.train()
            fig, axes = plt.subplots(4, 4, figsize=(20, 9))
            for k in range(4):
                for j, (im, title) in enumerate(((fixed_a[k], "synthetic"), (ta[k], "→ real style"),
                                                 (fixed_b[k], "real"), (tb[k], "→ synthetic style"))):
                    axes[k, j].imshow(im[0], cmap="gray", vmin=0, vmax=1)
                    axes[k, j].set_title(title, fontsize=8)
                    axes[k, j].axis("off")
            fig.tight_layout()
            fig.savefig(out / f"samples_{it + 1:06d}.png", dpi=70)
            plt.close(fig)
    print("→", out)


if __name__ == "__main__":
    main()
