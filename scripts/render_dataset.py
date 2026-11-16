"""Render a synthetic panoramic dataset from ToothFairy3 with domain randomisation.

For every volume in the chosen split(s): variant 0 is nominal (no randomisation) and variants
1..K are randomised with the enabled factors (geometry and volume factors are baked into the
render; image-level factors — noise, processing — are applied later, on the fly, from the
stored mean transmission). Resumable: existing outputs are skipped.

Each sample: data/renders/<name>/<case>_v<k>.npz with
  meanT (float16), fov (bool), masks (packed bits, channel order in index.json),
  crop_rows (dentate crop: top → mandible lower border + 8 mm), params (json), annotated flags.

Usage:
  .venv/bin/python scripts/render_dataset.py --name dr_full --splits train val --variants 8
  .venv/bin/python scripts/render_dataset.py --name nominal_test --splits test --variants 0
  .venv/bin/python scripts/render_dataset.py --name abl_no_ghosts --disable ghosts --variants 4
  --limit N renders only the first N volumes (smoke tests).
  --shard i --nshards n renders volumes i, i+n, i+2n, … (one Slurm array task per shard; same output folder).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback

import numpy as np

from dentalpano import io as dio
from dentalpano import opg, tf3
from dentalpano import randomize as DR
from dentalpano.simulator import SimConfig, fit_arch

ROOT = dio.REPO / "data" / "renders"
EXTRA_CHANNELS = {f"tooth{t}": ([t, t + 100], 12.0) for t in (q * 10 + i for q in (1, 2, 3, 4) for i in range(1, 9))}


def crop_rows(masks: dict, px_mm: float, margin_mm: float = 8.0) -> list[int]:
    mand = masks.get("mandible")
    if mand is None or not mand.any():
        return [0, int(next(iter(masks.values())).shape[0])]
    bottom = int(np.nonzero(mand.any(1))[0].max() + margin_mm / px_mm)
    return [0, min(bottom, mand.shape[0])]


def seed_for(base: int, case: str, k: int) -> int:
    return int(hashlib.sha256(f"{base}/{case}/{k}".encode()).hexdigest()[:8], 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--variants", type=int, default=8, help="random variants per volume (plus nominal v0)")
    ap.add_argument("--disable", nargs="*", default=[], help="factors fixed at nominal (ablation)")
    ap.add_argument("--override", nargs="*", default=[],
                    help="sampling-range overrides param=u:lo:hi or param=logu:lo:hi (e.g. metal_scale=logu:0.8:1.5)")
    ap.add_argument("--no-nominal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    out = ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((dio.REPO / "splits" / "toothfairy3.json").read_text())
    cases = [c for s in args.splits for c in split[s]][: args.limit][args.shard:: args.nshards]
    enabled = set(DR.FACTORS) - set(args.disable)
    base = opg.OPGConfig()
    base.channels.update(EXTRA_CHANNELS)
    channels = list(base.channels)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts"], capture_output=True, text=True,
                                cwd=dio.REPO).stdout)
    index_path = out / "index.json"
    args.overrides = dict(kv.split("=", 1) for kv in args.override)
    meta = {"name": args.name, "splits": args.splits, "variants": args.variants, "enabled": sorted(enabled),
            "overrides": args.overrides,
            "seed": args.seed, "channels": channels, "git": git, "dirty": dirty, "px_mm": base.column_mm}
    if index_path.exists():
        old = json.loads(index_path.read_text())
        same = lambda d: (d["enabled"], d["seed"], d["channels"], d.get("overrides") or {})   # older sets: no overrides
        assert same(old) == same(meta), "existing render set was made with different settings"
    tmp = out / f".index.{args.shard}.tmp"                 # atomic write: shards may start together
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(index_path)

    ks = ([] if args.no_nominal else [0]) + list(range(1, args.variants + 1))
    t_start = time.time()
    failures = {}
    for n, case in enumerate(cases):
        todo = [k for k in ks if not (out / f"{case}_v{k}.npz").exists()]
        if not todo:
            continue
        try:
            render_volume(case, todo, out, args, base, enabled, channels)
        except Exception as e:                         # one bad volume must not kill the shard
            failures[case] = f"{type(e).__name__}: {e}"
            print(f"!! {case} failed: {failures[case]}", flush=True)
            traceback.print_exc()
        el = time.time() - t_start
        print(f"[{n + 1}/{len(cases)}] {case} ({len(todo)} renders)  elapsed {el / 60:.1f} min", flush=True)
    if failures:
        (out / f"failures.shard{args.shard}.json").write_text(json.dumps(failures, indent=1))
        print(f"{len(failures)} volume(s) failed — see failures.shard{args.shard}.json")
        raise SystemExit(2)


def render_volume(case, todo, out, args, base, enabled, channels) -> None:
    img, _ = tf3.load_image(case)
    lab, _ = tf3.load_label(case)
    spine = DR.spine_mask(img, lab)
    arch = fit_arch(lab, SimConfig(column_mm=base.column_mm, arch_extend_mm=base.arch_extend_mm))
    labels = opg.prepare_labels(lab, base)          # once per volume, shared by all variants
    for k in todo:
        rng = np.random.default_rng(seed_for(args.seed, case, k))
        p = DR.nominal() if k == 0 else DR.sample(rng, enabled, args.overrides)
        r = opg.render(None, lab, DR.opg_config(p, base), arch=DR.jitter_arch(arch, p), mu=DR.attenuation(img, lab, p, spine),
                       labels=labels)
        masks = np.stack([r["masks"][c] for c in channels])
        np.savez_compressed(
            out / f"{case}_v{k}.npz", meanT=r["meanT"].astype(np.float16), fov=r["fov"],
            masks=np.packbits(masks, axis=0), mask_shape=np.array(masks.shape),
            crop_rows=np.array(crop_rows(r["masks"], base.column_mm)),
            params=json.dumps({a: (None if not np.isfinite(b) else b) for a, b in p.items()}),
            annotated=np.array([r["annotated"][c] for c in channels]),
            arch_source=arch.get("source", "teeth"))


if __name__ == "__main__":
    main()
