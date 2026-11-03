"""Train the multi-label panoramic segmentation model.

Model selection uses SYNTHETIC validation only (zero-shot setting); real DENTEX validation is
evaluated and logged but never used to choose checkpoints or hyperparameters. Real-data training
(few-shot / real-only baselines) is enabled with `real_train_frac` > 0.

Outputs in <out_root>/<run_name>/: config.json (config + git hash), log.csv, best.pt, last.pt.
Usage: python scripts/train.py --config configs/train/seg_v1.yaml [--set key=value ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

from dentalpano import io as dio
from dentalpano.train.data import STRUCTURES, TOOTH_CHANNELS, DentexDataset, SynthDataset
from dentalpano.train.model import build_model, device, dice_scores, masked_loss


def parse_overrides(pairs):
    out = {}
    for kv in pairs or []:
        k, v = kv.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def channel_list(spec):
    if spec == "structures":
        return list(STRUCTURES)
    if spec == "structures+teeth":
        return list(STRUCTURES) + list(TOOTH_CHANNELS)
    return list(spec)


def evaluate(model, loader, dev, channels, max_batches=None):
    model.eval()
    scores, losses = [], []
    with torch.no_grad():
        for b, (img, tgt, wgt, _) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            img, tgt, wgt = img.to(dev), tgt.to(dev), wgt.to(dev)
            logits = model(img)
            losses.append(masked_loss(logits, tgt, wgt)[0].item())
            scores.append(dice_scores(logits, tgt, wgt))
    s = torch.cat(scores).numpy()
    per = {c: float(np.nanmean(s[:, k])) if np.isfinite(s[:, k]).any() else float("nan") for k, c in enumerate(channels)}
    return float(np.mean(losses)), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action="append", default=[],
                    help="override config keys, e.g. epochs=2 batch_size=4 (may be repeated)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg.update(parse_overrides([kv for group in args.set for kv in (group or [])]))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    out = Path(cfg.get("out_root", dio.REPO / "outputs" / "runs")) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    channels = channel_list(cfg["channels"])
    (out / "config.json").write_text(json.dumps({**cfg, "channels_resolved": channels, "git": git}, indent=1, default=str))

    split_tf3 = json.loads((dio.REPO / "splits" / "toothfairy3.json").read_text())
    split_dx = json.loads((dio.REPO / "splits" / "dentex.json").read_text())
    size = tuple(cfg["size"])
    train_sets = []
    if cfg.get("synth_train_sets"):
        train_sets.append(SynthDataset(cfg["synth_train_sets"], split_tf3["train"], channels, size, train=True,
                                       image_factors=cfg.get("image_factors"), seed=cfg["seed"],
                                       variants=cfg.get("synth_train_variants")))
    if cfg.get("real_train_frac", 0) > 0:
        names = sorted(split_dx["fewshot_train"])
        rng = np.random.default_rng(cfg["seed"])
        k = max(1, math.ceil(cfg["real_train_frac"] * len(names)))
        pick = sorted(rng.choice(len(names), k, replace=False).tolist())
        real = DentexDataset([names[i] for i in pick], channels, size, train=True, seed=cfg["seed"])
        reps = max(1, int(cfg.get("real_repeat", 1)))
        train_sets.extend([real] * reps)
    train_ds = ConcatDataset(train_sets)
    synth_val = SynthDataset(cfg["synth_val_sets"], split_tf3["val"], channels, size, train=False,
                             variants=[0]) if cfg.get("synth_val_sets") else None
    real_val = None
    if cfg.get("real_val", True):
        real_val = DentexDataset(split_dx["val"], channels, size, train=False)
        if cfg.get("real_val_limit"):
            real_val = Subset(real_val, range(min(cfg["real_val_limit"], len(real_val))))

    nw = cfg.get("num_workers", 4)
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=nw,
                                        drop_last=shuffle, persistent_workers=nw > 0)
    train_loader = mk(train_ds, True)
    val_loaders = {"real": mk(real_val, False)} if real_val is not None else {}
    if synth_val is not None:
        val_loaders["synth"] = mk(synth_val, False)

    dev = device()
    model = build_model(len(channels), cfg.get("encoder", "resnet34"), cfg.get("pretrained", True))
    if cfg.get("init_from"):                    # fine-tune from an earlier run's checkpoint (e.g. sim-pretrained)
        ck = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
        assert ck["channels"] == channels, "init_from checkpoint has different channels"
        model.load_state_dict(ck["model"])
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    steps = cfg["epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=max(steps, 1), pct_start=0.05)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = out / "log.csv"
    best, t0 = -1.0, time.time()
    sel_key = cfg.get("select_on", "synth")
    for epoch in range(cfg["epochs"]):
        for ds in train_sets:
            ds.epoch = epoch
        model.train()
        tl = []
        for b, (img, tgt, wgt, _) in enumerate(train_loader):
            if cfg.get("max_train_batches") and b >= cfg["max_train_batches"]:
                break
            img, tgt, wgt = img.to(dev), tgt.to(dev), wgt.to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(img)
            loss, parts = masked_loss(logits.float(), tgt, wgt, cfg.get("dice_weight", 1.0))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl.append(loss.item())
        row = {"epoch": epoch, "train_loss": float(np.mean(tl)), "minutes": round((time.time() - t0) / 60, 2)}
        for name, loader in val_loaders.items():
            vl, per = evaluate(model, loader, dev, channels, cfg.get("max_val_batches"))
            row[f"{name}_loss"] = vl
            for c, v in per.items():
                if np.isfinite(v):
                    row[f"{name}_dice_{c}"] = round(v, 4)
            finite = [v for v in per.values() if np.isfinite(v)]
            row[f"{name}_dice_mean"] = round(float(np.mean(finite)), 4) if finite else float("nan")
        score = row.get(f"{sel_key}_dice_mean", float("nan"))
        if np.isfinite(score) and score > best:
            best = score
            torch.save({"model": model.state_dict(), "channels": channels, "epoch": epoch, "cfg": cfg}, out / "best.pt")
        torch.save({"model": model.state_dict(), "channels": channels, "epoch": epoch, "cfg": cfg}, out / "last.pt")
        new = not log_path.exists()
        with open(log_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(row), extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        print(json.dumps({k: row[k] for k in row if k in ("epoch", "train_loss", "minutes") or k.endswith("dice_mean")
                          or k.endswith("dice_teeth") or k.startswith("synth_dice_canal")}), flush=True)
    print("best", sel_key, "dice_mean:", best, "→", out)


if __name__ == "__main__":
    main()
