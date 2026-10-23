"""Validate the OPG geometry with a bead phantom on a real patient arch.

Beads (r = 0.6 mm, dense) sit on the patient-RIGHT half of the arch of `--case` at 4 arch
positions × 9 offsets t along the normal (lingual < 0 < buccal). Each offset gets its own
height so beads don't overlap. Per bead we measure horizontal/vertical FWHM in the
panoramic, which gives the focal-trough width; the empty left half shows ghosts.

Outputs: outputs/opg_phantom/{phantom.png, fwhm.png, fwhm.csv, summary.md}
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import io as dio  # noqa: E402
from dentalpano import opg, tf3  # noqa: E402
from dentalpano.simulator import TOOTH_IDS, SimConfig, fit_arch  # noqa: E402

OUT = dio.REPO / "outputs" / "opg_phantom"
OFFSETS_MM = [-15, -10, -6, -3, 0, 3, 6, 10, 15]
REGIONS_MM = {"incisor": 0.0, "canine": 18.0, "premolar": 32.0, "molar": 50.0}  # arclength from midline


def fwhm(profile: np.ndarray) -> float:
    p = profile - profile.min()
    if p.max() <= 0:
        return np.nan
    above = np.nonzero(p >= p.max() / 2)[0]
    return float(above[-1] - above[0] + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="ToothFairy3F_010")
    ap.add_argument("--slit-mm", type=float, default=None)
    ap.add_argument("--beam", default="curvature", choices=["curvature", "rotating", "normal"])
    ap.add_argument("--r-front", type=float, default=None)
    ap.add_argument("--r-back", type=float, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = OUT / (args.tag or args.beam)
    out.mkdir(parents=True, exist_ok=True)
    cfg = opg.OPGConfig(beam=args.beam)
    if args.slit_mm:
        cfg.slit_mm = args.slit_mm
    if args.r_front:
        cfg.r_front_mm = args.r_front
    if args.r_back:
        cfg.r_back_mm = args.r_back
    vx = cfg.voxel_mm

    lab, _ = tf3.load_label(args.case)
    arch = fit_arch(lab, SimConfig(voxel_mm=vx, column_mm=cfg.column_mm, arch_extend_mm=cfg.arch_extend_mm))
    zc = float(np.nonzero(np.isin(lab, TOOTH_IDS))[2].mean())
    P, N = arch["points"], arch["normals"]
    mid = int(np.argmin(P[:, 1]))                    # most anterior arch point = midline
    right_dir = -1 if P[0, 0] < P[-1, 0] else 1      # index direction toward patient right (low i)

    ii, jj, kk = np.ogrid[: lab.shape[0], : lab.shape[1], : lab.shape[2]]
    beads = []
    for reg, s_mm in REGIONS_MM.items():
        f = mid + right_dir * int(round(s_mm / cfg.column_mm))
        for t in OFFSETS_MM:
            beads.append({"region": reg, "t_mm": t, "frame": f, "centre": P[f] + (t / vx) * N[f]})

    def bead_volume(centres):
        mu = np.zeros(lab.shape, np.float32)
        for c in centres:
            r2 = ((ii - c[0]) ** 2 + (jj - c[1]) ** 2 + (kk - zc) ** 2) * vx ** 2
            mu[r2 <= 0.6 ** 2] = 0.1
        return mu

    # one render per bead → unambiguous FWHM / position
    rows = []
    for b in beads:
        res = opg.render(None, None, cfg, arch=arch, mu=bead_volume([b["centre"]]))
        img = res["image"]
        pr, pc = np.unravel_index(np.argmax(img), img.shape)
        col = res["frame_col"][b["frame"]]
        row = res["z_top_mm"] / cfg.column_mm
        rows.append({"region": b["region"], "t_mm": b["t_mm"], "frame": b["frame"],
                     "fwhm_h_mm": fwhm(img[pr, :]) * cfg.column_mm, "fwhm_v_mm": fwhm(img[:, pc]) * cfg.column_mm,
                     "dx_mm": (pc - col) * cfg.column_mm, "dy_mm": (pr - row) * cfg.column_mm,
                     "peak": float(img.max()), "integral": float(img.sum())})
    # all beads together, for the figure
    res = opg.render(None, None, cfg, arch=arch, mu=bead_volume([b["centre"] for b in beads]))
    img = res["image"]
    df = pd.DataFrame(rows)
    df.to_csv(out / "fwhm.csv", index=False)

    geo = res["geometry"]
    lines = [f"# OPG bead phantom ({args.case} arch, beam={cfg.beam}, r front/back {cfg.r_front_mm}/{cfg.r_back_mm} mm, slit {cfg.slit_mm} mm = {res['slit_cols']} cols, M = {cfg.magnification:.2f}, "
             f"tilt {cfg.tilt_deg}°)", "", "Image-plane values are in mm at trough scale (0.3 mm/px). "
             "Ideal bead image: 1.2 mm.", "",
             df.pivot(index="t_mm", columns="region", values="fwhm_h_mm").round(2).to_markdown(), "",
             "Vertical FWHM (mm):", "", df.pivot(index="t_mm", columns="region", values="fwhm_v_mm").round(2).to_markdown(), "",
             "Horizontal position relative to the frame's trough column, dx (mm):", "", df.pivot(index="t_mm", columns="region", values="dx_mm").round(2).to_markdown()]
    trough = {}
    for reg, g in df.groupby("region"):
        ok = g[g.fwhm_h_mm <= 2 * g.loc[g.t_mm == 0, "fwhm_h_mm"].iloc[0]].t_mm
        trough[reg] = (ok.min(), ok.max())
    lines += ["", "Peak height (fraction of in-trough peak):", "",
              df.assign(rel=df.peak / df.groupby("region").peak.transform(lambda p: p[df.loc[p.index, "t_mm"] == 0].iloc[0]))
              .pivot(index="t_mm", columns="region", values="rel").round(2).to_markdown()]
    lines += ["", "Focal trough (offsets where horizontal FWHM ≤ 2× its value at t=0): "
              + ", ".join(f"{k}: [{a}, {b}] mm" for k, (a, b) in trough.items())]
    lines += ["", "Beam obliquity δ (deg) / effective rotation radius r (mm) at the bead positions: " + ", ".join(
        f"{reg} {geo['delta_deg'][g.frame.iloc[0]]:.1f}° / {geo['r'][g.frame.iloc[0]]:.0f}" for reg, g in df.groupby("region")),
        f"Max |δ| over the dentate arch (|s| ≤ 55 mm): {geo['delta_deg'][np.abs(geo['s']) <= 55].max():.1f}°"]
    lines += ["", "Beads are at the occlusal height; ghosts need structures behind the rotation centre "
              "(see real-volume renders)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(2, 1, figsize=(14, 9))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.95))
    ax[0].set_title("bead phantom panoramic (beads on patient-right half; patient right = image left)")
    ax[1].imshow(np.log1p(200 * img), cmap="magma")
    ax[1].set_title("log scale (shows ghosts / smear)")
    fig.tight_layout()
    fig.savefig(out / "phantom.png", dpi=80)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.isin(lab, TOOTH_IDS).any(2).T * 1.0, cmap="gray", origin="lower",
              extent=[0, lab.shape[0] * vx, 0, lab.shape[1] * vx])
    ax.plot(geo["A"][:, 0], geo["A"][:, 1], "y-", lw=1, label="focal trough centre (arch)")
    if cfg.beam != "normal":
        ax.plot(geo["C"][:, 0], geo["C"][:, 1], "c.", ms=1, label="instantaneous rotation centre")
    for f in range(0, len(geo["A"]), 25):
        a, uu = geo["A"][f], geo["u"][f]
        ax.plot([a[0] - 140 * uu[0], a[0] + 40 * uu[0]], [a[1] - 140 * uu[1], a[1] + 40 * uu[1]], "r-", lw=0.4, alpha=0.6)
    ax.set_xlim(0, lab.shape[0] * vx)
    ax.set_ylim(lab.shape[1] * vx, 0)
    ax.set_xlabel("i (mm) → patient left")
    ax.set_ylabel("j (mm) → posterior")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"beam geometry ({cfg.beam}); red = central rays every 7.5 mm")
    fig.tight_layout()
    fig.savefig(out / "geometry.png", dpi=80)
    fig, ax = plt.subplots(figsize=(6, 4))
    for reg, g in df.groupby("region"):
        ax.plot(g.t_mm, g.fwhm_h_mm, "o-", label=reg)
    ax.set_xlabel("offset from arch t (mm; − lingual, + buccal)")
    ax.set_ylabel("horizontal FWHM (mm)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fwhm.png", dpi=90)


if __name__ == "__main__":
    main()
