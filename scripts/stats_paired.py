"""Paired hierarchical bootstrap for the paper's main comparisons → outputs/paper/stats.md.

Each comparison is arm A − arm B on the same test images. One bootstrap draw resamples the training seeds of each arm
(with replacement, independently) and then the test images (with replacement, shared by both arms, so the pairing by
image is kept); the statistic is recomputed on the resampled set. Reported: mean difference, 95% percentile interval,
and the two-sided bootstrap p-value (2 × the smaller tail fraction around zero). 10,000 draws, seed 0.

Usage: python scripts/stats_paired.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dentalpano import io as dio

RUNS = dio.REPO / "outputs" / "runs"
OUT = dio.REPO / "outputs" / "paper" / "stats.md"
FRACS = ["0.01", "0.05", "0.10", "0.25", "0.50", "1.00"]
N_BOOT = 10_000


def _load(runs: list[str], sub: str, key: str) -> list[dict]:
    """Per-image tables as {column: array} (plus "_index"), sorted by image so arms align."""
    out = []
    for r in runs:
        df = pd.read_csv(RUNS / r / sub / ("real_per_image.csv" if sub == "eval_best" else "per_image.csv"))
        df = df.set_index(key).sort_index()
        out.append({"_index": df.index.to_numpy(), **{c: df[c].to_numpy(float) for c in df.columns}})
    return out


def load_dentex(runs: list[str]) -> list[dict]:
    return _load(runs, "eval_best", "image")


def load_contact(runs: list[str]) -> list[dict]:
    return _load(runs, "eval_contact_best", "id")


def load_tufts(runs: list[str]) -> list[dict]:
    return _load(runs, "eval_tufts_best", "image")


def load_mandible(runs: list[str]) -> list[dict]:
    return _load(runs, "eval_mandible_best", "id")


def fdi(d, idx):
    return np.nanmean(d["dice_fdi_mean"][idx])


def numbering(d, idx):
    return d["n_number_ok"][idx].sum() / max(d["n_detected"][idx].sum(), 1)


def teeth(d, idx):
    return np.nanmean(d["dice_teeth"][idx])


def mandible(d, idx):
    return ((d["dice_r1"][idx] + d["dice_r2"][idx]) / 2).mean()


def auc(d, idx, col="matched_contact_prob", g13=False):
    y, s, g = d["contact"][idx], d[col][idx], d["group"][idx]
    if g13:
        y, s = y[np.isin(g, [1, 3])], s[np.isin(g, [1, 3])]
    return roc_auc_score(y, s)


def canal(d, idx):
    return d["matched_dice_canal"][idx].mean()


def compare(a: list[pd.DataFrame], b: list[pd.DataFrame], stat, rng) -> tuple[float, float, float, float]:
    assert all((x["_index"] == a[0]["_index"]).all() for x in a + b), "arms must share the same test images"
    n = len(a[0]["_index"])
    full = np.arange(n)
    point = np.mean([stat(x, full) for x in a]) - np.mean([stat(x, full) for x in b])
    diffs = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = rng.integers(0, n, n)
        sa = rng.integers(0, len(a), len(a))
        sb = rng.integers(0, len(b), len(b))
        diffs[k] = np.mean([stat(a[i], idx) for i in sa]) - np.mean([stat(b[i], idx) for i in sb])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return point, lo, hi, p


def fmt(r) -> str:
    point, lo, hi, p = r
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}], p {'< 0.001' if p < 1e-3 else f'= {p:.3f}'}"


def main() -> None:
    rng = np.random.default_rng(0)
    lines = ["# Paired hierarchical bootstrap (seeds × test images, 10,000 draws)", ""]

    lines += ["## Zero-shot (DENTEX test, 187 images)", "| Comparison | Teeth Dice | FDI Dice | Numbering |",
              "|---|---|---|---|"]
    dr = load_dentex(["seg_v1_dr_full", "seg_v1_dr_full_s1", "seg_v1_dr_full_s2"])
    for lab, other in [("Randomized − no randomization", [f"seg_nominal_s{s}" for s in range(3)]),
                       ("Randomized − CycleGAN", ["seg_cyclegan"])]:
        o = load_dentex(other)
        lines.append(f"| {lab} | " + " | ".join(fmt(compare(dr, o, f, rng)) for f in (teeth, fdi, numbering)) + " |")

    lines += ["", "## Few-shot DENTEX: sim→real (randomized) − real-only (tuned)",
              "| Images | FDI Dice | Numbering |", "|---|---|---|"]
    for f in FRACS:
        a = load_dentex([f"fewshot_sim2real_f{f}_s{s}" for s in range(3)])
        b = load_dentex([f"realonly_v2_f{f}_s{s}" for s in range(3)])
        k = max(1, int(np.ceil(float(f) * 384)))
        lines.append(f"| {k} | {fmt(compare(a, b, fdi, rng))} | {fmt(compare(a, b, numbering, rng))} |")

    lines += ["", "## Simulator: TDI (ours) − curved summed slab, both randomized (zero-shot, 3 seeds each)",
              "| Test set | Teeth Dice | FDI Dice | Numbering |", "|---|---|---|---|"]
    slab = [f"seg_slab_s{s}" for s in range(3)]
    lines.append("| DENTEX | " + " | ".join(fmt(compare(dr, load_dentex(slab), f, rng)) for f in (teeth, fdi, numbering))
                 + " |")
    zs = ["seg_v1_dr_full", "seg_v1_dr_full_s1", "seg_v1_dr_full_s2"]
    a, b = load_tufts(zs), load_tufts(slab)
    lines.append("| Tufts | " + " | ".join(fmt(compare(a, b, f, rng)) for f in (teeth, fdi, numbering)) + " |")
    lines.append(f"| Mandible (Abdi) | {fmt(compare(load_mandible(zs), load_mandible(slab), mandible, rng))} | | |")

    def few_table(title, loader, arm_a, arm_b, n_pool=384):
        nonlocal lines
        lines += ["", f"## {title}", "| Images | FDI Dice | Numbering |", "|---|---|---|"]
        for f in FRACS:
            a = loader([f"{arm_a}_f{f}_s{s}" for s in range(3)])
            b = loader([f"{arm_b}_f{f}_s{s}" for s in range(3)])
            k = max(1, int(np.ceil(float(f) * n_pool)))
            lines.append(f"| {k} | {fmt(compare(a, b, fdi, rng))} | {fmt(compare(a, b, numbering, rng))} |")

    few_table("Few-shot DENTEX: sim→real TDI − sim→real slab", load_dentex, "fewshot_sim2real", "fewshot_slab")
    few_table("Few-shot DENTEX: sim→real − real-only with in-domain SSL init", load_dentex, "fewshot_sim2real",
              "ssl_realonly")
    few_table("Few-shot DENTEX: SSL init − ImageNet init (real-only)", load_dentex, "ssl_realonly", "realonly_v2")
    few_table("Tufts (trained on DENTEX labels): sim→real − real-only", load_tufts, "fewshot_sim2real", "realonly_v2")
    few_table("Tufts: sim→real − SSL real-only", load_tufts, "fewshot_sim2real", "ssl_realonly")
    few_table("Tufts: sim→real TDI − sim→real slab", load_tufts, "fewshot_sim2real", "fewshot_slab")

    lines += ["", "## Few-shot CONTACT: sim→real (lr 1e-3, 90 ep) − real-only (same schedule)",
              "| Crops | Contact AUC | Group 1 vs 3 AUC | Canal Dice |", "|---|---|---|---|"]
    for f in FRACS:
        a = load_contact([f"ctfew_sim2real_v3_f{f}_s{s}" for s in range(3)])
        b = load_contact([f"ctfew_realonly_v2_f{f}_s{s}" for s in range(3)])
        k = max(1, int(np.ceil(float(f) * 878)))
        lines.append(f"| {k} | {fmt(compare(a, b, auc, rng))} | "
                     f"{fmt(compare(a, b, lambda d, i: auc(d, i, g13=True), rng))} | {fmt(compare(a, b, canal, rng))} |")

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
