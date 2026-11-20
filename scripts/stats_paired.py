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


def load_dentex(runs: list[str]) -> list[pd.DataFrame]:
    return [pd.read_csv(RUNS / r / "eval_best" / "real_per_image.csv").set_index("image").sort_index() for r in runs]


def load_contact(runs: list[str]) -> list[pd.DataFrame]:
    return [pd.read_csv(RUNS / r / "eval_contact_best" / "per_image.csv").set_index("id").sort_index() for r in runs]


def fdi(df, idx):
    return df["dice_fdi_mean"].to_numpy()[idx].mean()


def numbering(df, idx):
    return df["n_number_ok"].to_numpy()[idx].sum() / max(df["n_detected"].to_numpy()[idx].sum(), 1)


def teeth(df, idx):
    return df["dice_teeth"].to_numpy()[idx].mean()


def auc(df, idx, col="matched_contact_prob", g13=False):
    d = df.iloc[idx]
    if g13:
        d = d[d["group"].isin([1, 3])]
    return roc_auc_score(d["contact"], d[col])


def canal(df, idx):
    return df["matched_dice_canal"].to_numpy()[idx].mean()


def compare(a: list[pd.DataFrame], b: list[pd.DataFrame], stat, rng) -> tuple[float, float, float, float]:
    assert all((x.index == a[0].index).all() for x in a + b), "arms must share the same test images"
    n = len(a[0])
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
