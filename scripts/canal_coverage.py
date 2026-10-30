"""Measure how much of each 3D canal (and of the teeth) lies inside the focal-trough band of the fitted arch.

For every volume and side: fraction of canal voxels (labels 3 = left, 4 = right) whose in-plane distance
to the arch curve is ≤ band_mm. Low values mean the panoramic canal mask is truncated (the canal leaves
the trough, typically posteriorly where the arch extension misses the ramus).

Usage: .venv/bin/python scripts/canal_coverage.py [--out outputs/canal_coverage.csv] [--workers 4]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dentalpano import io as dio
from dentalpano import tf3
from dentalpano.simulator import arch_distance_map
from dentalpano.simulator import TOOTH_IDS, SimConfig, fit_arch

BAND_MM = 12.0


def coverage(case: str) -> dict:
    lab, _ = tf3.load_label(case)
    arch = fit_arch(lab, SimConfig(column_mm=0.3, arch_extend_mm=10.0))
    dist = arch_distance_map(arch, lab.shape[:2]) * 0.3
    row = {"case": case, "set": tf3.case_set(case), "arch_source": arch["source"]}
    for side, lid in (("L", 3), ("R", 4)):
        ii, jj, _ = np.nonzero(lab == lid)
        row[f"cov_{side}"] = float((dist[ii, jj] <= BAND_MM).mean()) if len(ii) else np.nan
    ii, jj, _ = np.nonzero(np.isin(lab, TOOTH_IDS))
    row["cov_teeth"] = float((dist[ii, jj] <= BAND_MM).mean()) if len(ii) else np.nan
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(dio.REPO / "outputs" / "canal_coverage.csv"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()
    cases = args.cases or tf3.case_ids()
    with ProcessPoolExecutor(args.workers) as ex:
        df = pd.DataFrame(ex.map(coverage, cases, chunksize=4))
    df["cov_min"] = df[["cov_L", "cov_R"]].min(axis=1)
    df.to_csv(args.out, index=False)
    q = df.cov_min.quantile([0, .05, .1, .25, .5]).round(3).to_dict()
    print("min-side coverage quantiles:", q)
    for t in (0.6, 0.8, 0.9):
        print(f"volumes with a side < {t}: {(df.cov_min < t).sum()}")
    print("teeth coverage quantiles:", df.cov_teeth.quantile([0, .01, .05, .25, .5]).round(3).to_dict())
    print(df.sort_values("cov_min").head(6).to_string(index=False))
    print(df.sort_values("cov_teeth").head(6).to_string(index=False))


if __name__ == "__main__":
    main()
