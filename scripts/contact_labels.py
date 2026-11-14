"""3D third-molar ↔ inferior alveolar canal relation from every ToothFairy3 label volume (contribution 4).

For each lower third molar present (38 with the left canal, 48 with the right canal) we compute the minimum 3D
distance (mm) between tooth voxels and canal voxels, plus how much of the tooth lies within 0.5 / 1.0 mm of the
canal. Distances are capped at the search margin (15 mm). Orientation does not matter (distances only), so raw
voxel order is used. The contact threshold is chosen later; the raw distance is stored.

Usage: python scripts/contact_labels.py [--shard i --nshards n] → data/derived/contact3d[.shard<i>].json
       python scripts/contact_labels.py --merge                → data/derived/contact3d.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy import ndimage

from dentalpano import io as dio
from dentalpano import tf3

PAIRS = {"L": (38, 3), "R": (48, 4)}   # side: (tooth label, canal label)
MARGIN_MM = 15.0
OUT = dio.REPO / "data" / "derived"


def relation(lab: np.ndarray, tooth: int, canal: int, spacing: np.ndarray) -> dict | None:
    t = lab == tooth
    if t.sum() < 50:
        return None
    idx = np.argwhere(t)
    m = np.ceil(MARGIN_MM / spacing).astype(int)
    lo, hi = np.maximum(idx.min(0) - m, 0), np.minimum(idx.max(0) + m + 1, lab.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    c = lab[sl] == canal
    if not c.any():
        return {"min_dist_mm": MARGIN_MM, "frac_within_0.5mm": 0.0, "frac_within_1mm": 0.0,
                "tooth_vox": int(t.sum()), "canal_in_margin": False}
    d = ndimage.distance_transform_edt(~c, sampling=spacing)[t[sl]]
    return {"min_dist_mm": round(float(min(d.min(), MARGIN_MM)), 3),
            "frac_within_0.5mm": round(float((d <= 0.5).mean()), 4), "frac_within_1mm": round(float((d <= 1.0).mean()), 4),
            "tooth_vox": int(t.sum()), "canal_in_margin": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.merge:
        res = {}
        for f in sorted(OUT.glob("contact3d.shard*.json")):
            res.update(json.loads(f.read_text())["cases"])
        n = len(tf3.case_ids())
        assert len(res) == n, f"merged {len(res)} of {n} cases"
        (OUT / "contact3d.json").write_text(json.dumps({"margin_mm": MARGIN_MM, "cases": res}, indent=0))
        print("merged", len(res), "→", OUT / "contact3d.json")
        return

    zf = dio.open_zip("toothfairy3")
    cases = tf3.case_ids(zf)[args.shard:: args.nshards][: args.limit]
    res, t0 = {}, time.time()
    for i, case in enumerate(cases):
        lab, aff = tf3.load_label(case, fix=False, zf=zf)
        spacing = np.linalg.norm(aff[:3, :3], axis=0)
        res[case] = {s: relation(lab, tooth, canal, spacing) for s, (tooth, canal) in PAIRS.items()}
        print(f"{i + 1}/{len(cases)} {case} {res[case]} ({time.time() - t0:.0f}s)", flush=True)
    name = "contact3d.json" if args.nshards == 1 and not args.limit else f"contact3d.shard{args.shard}.json"
    (OUT / name).write_text(json.dumps({"margin_mm": MARGIN_MM, "cases": res}, indent=0))
    print("→", OUT / name)


if __name__ == "__main__":
    main()
