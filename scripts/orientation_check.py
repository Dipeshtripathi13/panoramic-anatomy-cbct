"""Step 3: verify the TF3 orientation fix (in memory only).

1. Anatomical test on every label volume, raw vs fixed, in index space where the
   header says i→Left, j→Posterior, k→Superior:
     L/R: Left IAC (3) centroid i  >  Right IAC (4) centroid i
     A/P: lower incisors (31,32,41,42) centroid j  <  pharynx (7) centroid j
     S/I: mandible (1) rises posteriorly (ramus/condyles): corr(j, k) over its voxels has the
          sign expected for j→Posterior, k→Superior (> 0); and, when both exist, upper teeth
          (11–28) centroid k > lower teeth (31–48) centroid k
     (A first S/I test, lower-teeth vs IAC centroid, was too weak: the IAC climbs into the
     ramus, so the margin is only a few voxels.)
2. PNG slices (axial / coronal / sagittal) with label contours for one volume per set,
   raw vs fixed → outputs/orientation/.

Usage: .venv/bin/python scripts/orientation_check.py [--workers 5]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import io as dio  # noqa: E402
from dentalpano import tf3  # noqa: E402

OUT = dio.REPO / "outputs" / "orientation"
LOWER_TEETH = [q * 10 + t for q in (3, 4) for t in range(1, 9)]
UPPER_TEETH = [q * 10 + t for q in (1, 2) for t in range(1, 9)]
FRONT = [31, 32, 41, 42]


def centroid(lab: np.ndarray, ids) -> np.ndarray | None:
    m = np.isin(lab, ids)
    return np.array(ndimage.center_of_mass(m)) if m.any() else None


def tests(lab: np.ndarray) -> dict:
    c3, c4 = centroid(lab, [3]), centroid(lab, [4])
    cf, c7 = centroid(lab, FRONT), centroid(lab, [7])
    ct, cu = centroid(lab, LOWER_TEETH), centroid(lab, UPPER_TEETH)
    _, j, k = np.nonzero(lab[::4, ::4, ::4] == 1)
    return {
        "LR_ok": None if c3 is None or c4 is None else bool(c3[0] > c4[0]),
        "AP_ok": None if cf is None or c7 is None else bool(cf[1] < c7[1]),
        "SI_ok": bool(np.corrcoef(j, k)[0, 1] > 0),
        "SI_teeth_ok": None if ct is None or cu is None else bool(cu[2] > ct[2]),
    }


def check_case(case: str) -> dict:
    raw, _ = tf3.load_label(case, fix=False)
    row = {"case": case, "set": tf3.case_set(case)}
    row.update({f"raw_{k}": v for k, v in tests(raw).items()})
    row.update({f"official_{k}": v for k, v in tests(tf3.official_fix(raw)).items()})
    row.update({f"fix_{k}": v for k, v in tests(tf3.fix_orientation(raw)).items()})
    return row


COLORS = {1: "#d0d0d0", 2: "#a0a0ff", 3: "#ff3030", 4: "#ff9900", 5: "#30c0ff", 6: "#3080ff",
          7: "#80ff80", 103: "#ff00ff", 104: "#cc00cc", "teeth": "#ffff00", "pulp": "#00ffcc"}


def contour(ax, mask2d, color):
    if mask2d.any():
        ax.contour(mask2d.T, levels=[0.5], colors=[color], linewidths=0.8)


def draw(ax, img2d, lab2d, title, xlabel, ylabel):
    ax.imshow(np.clip(img2d, -500, 2500).T, cmap="gray", origin="lower")
    for k, c in COLORS.items():
        if k == "teeth":
            m = (lab2d >= 11) & (lab2d <= 48)
        elif k == "pulp":
            m = (lab2d >= 111) & (lab2d <= 148)
        else:
            m = lab2d == k
        contour(ax, m, c)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def figure(case: str) -> None:
    img_raw, _ = tf3.load_image(case, fix=False)
    lab_raw, _ = tf3.load_label(case, fix=False)
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for r, (name, fn) in enumerate([("raw (as stored)", lambda v: v),
                                    ("official Ditto flip", tf3.official_fix),
                                    ("our fix (flip i,k)", tf3.fix_orientation)]):
        img, lab = fn(img_raw), fn(lab_raw)
        ci = np.round(centroid(lab, [3, 4])).astype(int)         # through the canals
        ct = np.round(centroid(lab, LOWER_TEETH) if centroid(lab, LOWER_TEETH) is not None else ci).astype(int)
        c3 = np.round(centroid(lab, [3])).astype(int)
        # array shown with .T and origin='lower' → first index on x axis, increasing rightward
        draw(axes[r, 0], img[:, :, ct[2]], lab[:, :, ct[2]], f"{name}: axial k={ct[2]}",
             "i → (header: patient Left)", "j → (header: Posterior)")
        draw(axes[r, 1], img[:, ci[1], :], lab[:, ci[1], :], f"{name}: coronal j={ci[1]}",
             "i → (header: patient Left)", "k → (header: Superior)")
        draw(axes[r, 2], img[c3[0], :, :], lab[c3[0], :, :], f"{name}: sagittal i={c3[0]} (Left IAC)",
             "j → (header: Posterior)", "k → (header: Superior)")
    fig.suptitle(f"{case} (Set {tf3.case_set(case)}) — red = Left IAC (3), orange = Right IAC (4), "
                 "yellow = teeth, cyan = pulp, green = pharynx, magenta = incisive canals, blue = sinus/upper jaw",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"{case}.png", dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--reuse", action="store_true", help="reuse orientation_tests.csv")
    ap.add_argument("--cases", nargs="*", default=["ToothFairy3P_001", "ToothFairy3F_001", "ToothFairy3S_0001"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.reuse:
        df = pd.read_csv(OUT / "orientation_tests.csv")
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            df = pd.DataFrame(ex.map(check_case, tf3.case_ids(), chunksize=4))
        df.to_csv(OUT / "orientation_tests.csv", index=False)
    lines = ["# Orientation check", "", "Fraction of cases passing each anatomical test (n evaluable):", "",
             "| set | test | raw | official Ditto flip (i,j) | our fix (i,k) |", "|---|---|---|---|---|"]
    for s, g in df.groupby("set"):
        for t in ("LR_ok", "AP_ok", "SI_ok", "SI_teeth_ok"):
            cell = lambda c: f"{g[c].dropna().astype(bool).mean():.3f} ({g[c].notna().sum()})"
            lines.append(f"| {s} | {t} | {cell('raw_' + t)} | {cell('official_' + t)} | {cell('fix_' + t)} |")
    tests_ = ("LR_ok", "AP_ok", "SI_ok", "SI_teeth_ok")
    bad = df[~df[[f"fix_{t}" for t in tests_]].fillna(True).astype(bool).all(axis=1)]
    lines += ["", f"Cases failing any test after fix: {len(bad)}"] + [f"- {r.case}: " + ", ".join(
        t for t in tests_ if r[f"fix_{t}"] == False) for _, r in bad.iterrows()]  # noqa: E712
    (OUT / "orientation_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    for case in args.cases:
        figure(case)
        print("saved", OUT / f"{case}.png")


if __name__ == "__main__":
    main()
