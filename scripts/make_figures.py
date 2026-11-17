"""Results figures + tables for the paper draft, computed from outputs/runs (copied from the cluster).

Figures (PNG + PDF) → outputs/paper/figures/; tables (markdown) → outputs/paper/tables.md.
Figures that show dataset images (overlays, CycleGAN samples) stay in outputs/ (git-ignored; CC BY-NC-SA data).
Usage: python scripts/make_figures.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import io as dio  # noqa: E402

RUNS = dio.REPO / "outputs" / "runs"
OUT = dio.REPO / "outputs" / "paper"
FIG = OUT / "figures"
SEEDS = (0, 1, 2)
FRACS = ["0.01", "0.05", "0.10", "0.25", "0.50", "1.00"]
# reference categorical palette (dataviz skill, light mode): slot 1 blue = ours, slot 2 orange = baseline
C_OURS, C_BASE, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def stats(vals) -> tuple[float, float]:
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ---------------------------------------------------------------- DENTEX few-shot
def dentex_fewshot(tables: list[str]) -> None:
    metrics = [("real_dice_teeth", "mean", "Teeth Dice"), ("real_dice_fdi_mean", "mean", "Per-tooth (FDI) Dice"),
               ("real_tooth_numbering_acc", "value", "Numbering accuracy")]
    n_img = [max(1, int(np.ceil(float(f) * 384))) for f in FRACS]
    zs = jload(RUNS / "seg_v1_dr_full" / "eval_best" / "metrics.json")
    res = {}
    for arm in ("realonly", "sim2real"):
        for f in FRACS:
            js = [jload(RUNS / f"fewshot_{arm}_f{f}_s{s}" / "eval_best" / "metrics.json") for s in SEEDS]
            res[arm, f] = [j for j in js if j]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (key, sub, title) in zip(axes, metrics):
        for arm, col, lab in (("realonly", C_BASE, "Real-only (ImageNet init)"), ("sim2real", C_OURS, "Sim→real (ours)")):
            m = np.array([stats([j[key][sub] for j in res[arm, f]]) for f in FRACS])
            ax.plot(n_img, m[:, 0], color=col, lw=2, marker="o", ms=4, label=lab)
            ax.fill_between(n_img, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], color=col, alpha=0.15, lw=0)
        ax.axhline(zs[key][sub], color=INK2, lw=1.2, ls="--", label="Zero-shot (sim only, 0 labels)")
        ax.set_xscale("log")
        ax.set_xticks(n_img, [str(n) for n in n_img])
        ax.minorticks_off()
        ax.set_xlabel("Labelled real images")
        ax.set_title(title, color=INK, fontsize=8.5)
    axes[0].set_ylabel("DENTEX test (187 images)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "fig_fewshot_dentex")

    rows = ["| Labelled images | " + " | ".join(f"{t} RO / S2R" for *_, t in metrics) + " |",
            "|---|" + "---|" * len(metrics),
            "| 0 (zero-shot) | " + " | ".join(f"– / {zs[k][s]:.3f}" for k, s, _ in metrics) + " |"]
    for f, n in zip(FRACS, n_img):
        cells = []
        for k, s, _ in metrics:
            a = stats([j[k][s] for j in res["realonly", f]])
            b = stats([j[k][s] for j in res["sim2real", f]])
            cells.append(f"{a[0]:.3f}±{a[1]:.3f} / **{b[0]:.3f}±{b[1]:.3f}**" if b[0] > a[0]
                         else f"**{a[0]:.3f}±{a[1]:.3f}** / {b[0]:.3f}±{b[1]:.3f}")
        rows.append(f"| {n} ({float(f):.0%}) | " + " | ".join(cells) + " |")
    tables += ["### Table A — DENTEX few-shot (test, mean ± sd over 3 seeds)", *rows, ""]


# ---------------------------------------------------------------- DR ablation
ABL = [("no_noise", "Noise"), ("no_processing", "Processing"), ("no_metal", "Metal"), ("no_ghosts", "Ghosts"),
       ("no_contrast", "Contrast"), ("no_arch_jitter", "Arch jitter"), ("no_blur_outside", "Blur outside trough"),
       ("no_spine", "Spine"), ("no_trough_width", "Trough width")]


def ablation(tables: list[str]) -> None:
    get = lambda name, s: jload(RUNS / f"abl_{name}_s{s}" / "eval_best" / "metrics.json")
    ref = {s: get("ref", s) for s in SEEDS}
    keys = [("real_dice_teeth", "mean", "Δ teeth Dice"), ("real_tooth_numbering_acc", "value", "Δ numbering accuracy")]
    d = {}
    for name, _ in ABL:
        for k, sub, _ in keys:
            d[name, k] = np.array([get(name, s)[k][sub] - ref[s][k][sub] for s in SEEDS])
    order = sorted(ABL, key=lambda x: d[x[0], "real_dice_teeth"].mean())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    y = np.arange(len(order))
    lim = {"real_dice_teeth": (-0.11, 0.06), "real_tooth_numbering_acc": (-0.2, 0.04)}
    for ax, (k, sub, title) in zip(axes, keys):
        lo_ax, hi_ax = lim[k]
        clip = lambda v: np.clip(v, lo_ax, hi_ax)
        mean = np.array([d[n, k].mean() for n, _ in order])
        ax.barh(y, clip(mean), height=0.6, color=C_OURS, zorder=2)
        ax.hlines(y, clip([d[n, k].min() for n, _ in order]), clip([d[n, k].max() for n, _ in order]),
                  color=INK, lw=1, zorder=3)
        for yy, (n, _), m in zip(y, order, mean):
            vals = d[n, k]
            ax.plot(clip(vals), [yy] * len(vals), "o", ms=2.5, color=INK, zorder=4)
            if m < lo_ax:          # bar runs off the axis: say so and give the value
                ax.text(lo_ax + 0.003, yy, f"{m:+.3f} (off axis)", va="center", ha="left", fontsize=6.5,
                        color="white", zorder=5)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(lo_ax, hi_ax)
        ax.set_xticks([-0.1, -0.05, 0, 0.05] if k == "real_dice_teeth" else [-0.2, -0.1, 0])
        ax.set_title(f"{title} when the factor is switched off", fontsize=8.5, color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [lab for _, lab in order])
    fig.text(0.5, -0.04, "Paired ablation vs. all factors on (DENTEX test; bar = mean, whisker/dots = 3 seeds; "
             "values beyond the axis are clipped).", ha="center", color=INK2, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_ablation")

    r_t = stats([ref[s]["real_dice_teeth"]["mean"] for s in SEEDS])
    r_n = stats([ref[s]["real_tooth_numbering_acc"]["value"] for s in SEEDS])
    rows = [f"Reference (all factors on): teeth Dice {r_t[0]:.3f}±{r_t[1]:.3f}, numbering {r_n[0]:.3f}±{r_n[1]:.3f}.", "",
            "| Factor off | Δ teeth Dice mean [min, max] | Δ numbering mean [min, max] | consistent sign (3/3) |", "|---|---|---|---|"]
    for n, lab in order:
        a, b = d[n, "real_dice_teeth"], d[n, "real_tooth_numbering_acc"]
        cons = ", ".join(t for t, v in (("teeth", a), ("numbering", b)) if (v > 0).all() or (v < 0).all()) or "–"
        rows.append(f"| {lab} | {a.mean():+.3f} [{a.min():+.3f}, {a.max():+.3f}] | "
                    f"{b.mean():+.3f} [{b.min():+.3f}, {b.max():+.3f}] | {cons} |")
    tables += ["### Table B — DR ablation (DENTEX test, Δ vs. reference, 3 seeds)", *rows, ""]


# ---------------------------------------------------------------- DR follow-up (changed ranges, same paired protocol)
FOLLOW = [("no_metal", "Metal off"), ("metal_narrow", "Metal ×0.8–1.5 (was ×1–3)"),
          ("image_strong", "Wider image-level ranges"), ("no_metal_image_strong", "Metal off + wider image ranges")]


def followup(tables: list[str]) -> None:
    get = lambda name, s: jload(RUNS / f"abl_{name}_s{s}" / "eval_best" / "metrics.json")
    if not all(get(n, s) for n, _ in FOLLOW for s in SEEDS):
        return
    ref = {s: get("ref", s) for s in SEEDS}
    keys = [("real_dice_teeth", "mean", "Δ teeth Dice"), ("real_dice_fdi_mean", "mean", "Δ per-tooth (FDI) Dice"),
            ("real_tooth_numbering_acc", "value", "Δ numbering accuracy")]
    d = {(n, k): np.array([get(n, s)[k][sub] - ref[s][k][sub] for s in SEEDS]) for n, _ in FOLLOW for k, sub, _ in keys}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9), sharey=True)
    y = np.arange(len(FOLLOW))[::-1]
    for ax, (k, sub, title) in zip(axes, keys):
        mean = np.array([d[n, k].mean() for n, _ in FOLLOW])
        ax.barh(y, mean, height=0.6, color=C_OURS, zorder=2)
        ax.hlines(y, [d[n, k].min() for n, _ in FOLLOW], [d[n, k].max() for n, _ in FOLLOW], color=INK, lw=1, zorder=3)
        for yy, (n, _) in zip(y, FOLLOW):
            ax.plot(d[n, k], [yy] * len(SEEDS), "o", ms=2.5, color=INK, zorder=4)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-0.05, 0.07)
        ax.set_xticks([-0.04, 0, 0.04])
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [lab for _, lab in FOLLOW])
    fig.text(0.5, -0.06, "Δ vs. the ablation reference (DENTEX test; bar = mean, whisker/dots = 3 seeds).",
             ha="center", color=INK2, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_dr_followup")

    absval = lambda n, k, sub: stats([get(n, s)[k][sub] for s in SEEDS])
    syn = lambda n, c: np.mean([get(n, s)["synth_dice"][c] for s in SEEDS])
    rows = ["| Variant | Teeth Dice | FDI Dice | Numbering | Δ teeth [min, max] | Δ numbering [min, max] | synth implant Dice |",
            "|---|---|---|---|---|---|---|"]
    for n, lab in [("ref", "Reference"), *FOLLOW]:
        cells = [f"{m:.3f}±{s:.3f}" for m, s in (absval(n, k, sub) for k, sub, _ in keys)]
        if n == "ref":
            delta = ["–", "–"]
        else:
            a, b = d[n, "real_dice_teeth"], d[n, "real_tooth_numbering_acc"]
            delta = [f"{a.mean():+.3f} [{a.min():+.3f}, {a.max():+.3f}]", f"{b.mean():+.3f} [{b.min():+.3f}, {b.max():+.3f}]"]
        rows.append(f"| {lab} | " + " | ".join(cells + delta) + f" | {syn(n, 'implant'):.3f} |")
    tables += ["### Table F — DR follow-up (DENTEX test, 3 seeds, paired with the ablation reference)", *rows, ""]


# ---------------------------------------------------------------- CycleGAN
def cyclegan(tables: list[str]) -> None:
    runs = [("seg_v1_dr_full", "Physics DR (ours)", C_OURS), ("seg_cyclegan", "CycleGAN", C_BASE),
            ("seg_cyclegan_dr", "DR + CycleGAN", C_3)]
    keys = [("real_dice_teeth", "mean", "Teeth\nDice"), ("real_dice_fdi_mean", "mean", "FDI\nDice"),
            ("real_tooth_detection_rate", "value", "Detection"), ("real_tooth_numbering_acc", "value", "Numbering"),
            ("real_presence_f1", None, "Presence\nF1")]
    js = {r: jload(RUNS / r / "eval_best" / "metrics.json") for r, *_ in runs}
    val = lambda j, k, s: j[k][s] if s else j[k]
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    x, w = np.arange(len(keys)), 0.26
    for i, (r, lab, col) in enumerate(runs):
        ax.bar(x + (i - 1) * (w + 0.02), [val(js[r], k, s) for k, s, _ in keys], width=w, color=col, label=lab, zorder=2)
    ax.set_xticks(x, [t for *_, t in keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("DENTEX test")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save(fig, "fig_cyclegan")
    s = RUNS / "cyclegan_v1" / "samples_040000.png"
    if s.exists():
        shutil.copy(s, FIG / "fig_cyclegan_samples_DATA.png")
    rows = ["| Training images | " + " | ".join(t.replace("\n", " ") for *_, t in keys) + " |", "|---|" + "---|" * len(keys)]
    for r, lab, _ in runs:
        rows.append(f"| {lab} | " + " | ".join(f"{val(js[r], k, s):.3f}" for k, s, _ in keys) + " |")
    tables += ["### Table C — Physics DR vs CycleGAN translation (zero-shot, DENTEX test, single run)", *rows, ""]


# ---------------------------------------------------------------- CONTACT
def contact(tables: list[str]) -> None:
    n_img = [max(1, int(np.ceil(float(f) * 878))) for f in FRACS]
    res = {(arm, f): [j["matched"] for s in SEEDS
                      if (j := jload(RUNS / f"ctfew_{arm}_f{f}_s{s}" / "eval_contact_best" / "metrics.json"))]
           for arm in ("realonly", "sim2real") for f in FRACS}
    zs = jload(RUNS / "contact_v1" / "eval_contact_best" / "metrics.json")["matched"]
    crop = jload(RUNS / "crop_v1" / "eval_contact_best" / "metrics.json")["matched"]
    keys = [(lambda m: m["head_contact_auc"], "Contact AUC", zs["head_contact_auc"]),
            (lambda m: m["head_contact_auc_group1_vs_3"], "Contact AUC, group 1 vs 3", zs["head_contact_auc_group1_vs_3"]),
            (lambda m: m["dice_canal"]["mean"], "Canal Dice", zs["dice_canal"]["mean"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (fn, title, z) in zip(axes, keys):
        for arm, col, lab in (("realonly", C_BASE, "Real-only (ImageNet init)"),
                              ("sim2real", C_OURS, "Sim→real (CBCT contact labels)")):
            m = np.array([stats([fn(x) for x in res[arm, f]]) for f in FRACS])
            ax.plot(n_img, m[:, 0], color=col, lw=2, marker="o", ms=4, label=lab)
            ax.fill_between(n_img, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], color=col, alpha=0.15, lw=0)
        ax.axhline(z, color=INK2, lw=1.2, ls="--", label="Zero-shot (sim only)")
        if "AUC" in title:
            ax.axhline(0.5, color=INK2, lw=0.8, ls=":", label="Chance")
        ax.set_xscale("log")
        ax.set_xticks(n_img, [str(n) for n in n_img])
        ax.minorticks_off()
        ax.set_xlabel("Labelled real crops")
        ax.set_title(title, color=INK, fontsize=8.5)
    axes[0].set_ylabel("CONTACT test (444 crops)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "fig_fewshot_contact")
    for r, name in (("crop_v1", "fig_contact_overlays_DATA.png"), ("seg_v1_dr_full", "fig_dentex_overlays_DATA.png")):
        src = RUNS / r / ("eval_contact_best" if r == "crop_v1" else "eval_best") / \
            ("overlays.png" if r == "crop_v1" else "overlays_real.png")
        if src.exists():
            shutil.copy(src, FIG / name)

    rows = ["| Model | molar Dice | canal Dice | contact AUC [CI] | group 1 vs 3 AUC |", "|---|---|---|---|---|",
            f"| crop_v1 (sim only), 2D overlap readout | {crop['dice_molar']['mean']:.3f} | {crop['dice_canal']['mean']:.3f} | "
            f"{crop['contact_auc_from_overlap_frac']:.3f} | {crop['contact_auc_group1_vs_3']:.3f} |",
            f"| contact_v1 (sim only, CBCT labels), contact head | {zs['dice_molar']['mean']:.3f} | {zs['dice_canal']['mean']:.3f} | "
            f"{zs['head_contact_auc']:.3f} {zs['head_contact_auc_ci95']} | {zs['head_contact_auc_group1_vs_3']:.3f} |"]
    for t in ("1.0", "2.0"):
        j = jload(RUNS / f"contact_v1_t{t}" / "eval_contact_best" / "metrics.json")
        if j:
            m = j["matched"]
            rows.append(f"| contact_v1, contact := ≤ {t} mm (sensitivity) | {m['dice_molar']['mean']:.3f} | "
                        f"{m['dice_canal']['mean']:.3f} | {m['head_contact_auc']:.3f} {m['head_contact_auc_ci95']} | "
                        f"{m['head_contact_auc_group1_vs_3']:.3f} |")
    rows += ["", "| Labelled crops | contact AUC RO / S2R | g1 vs 3 RO / S2R | canal Dice RO / S2R |", "|---|---|---|---|"]
    for f, n in zip(FRACS, n_img):
        c = [tuple(stats([fn(x) for x in res[arm, f]]) for arm in ("realonly", "sim2real")) for fn, *_ in keys]
        rows.append(f"| {n} ({float(f):.0%}) | " + " | ".join(f"{a[0]:.3f}±{a[1]:.3f} / {b[0]:.3f}±{b[1]:.3f}" for a, b in c) + " |")
    tables += ["### Table D — CONTACT (official test, 444 crops; matched resolution; few-shot 3 seeds)", *rows, ""]


def zero_shot(tables: list[str]) -> None:
    j = jload(RUNS / "seg_v1_dr_full" / "eval_best" / "metrics.json")
    sd = j["synth_dice"]
    tables += ["### Table E — Zero-shot panoramic model (seg_v1, trained on synthetic only)",
               "| | value [95% CI] |", "|---|---|",
               f"| DENTEX teeth Dice | {j['real_dice_teeth']['mean']:.3f} {j['real_dice_teeth']['ci95']} |",
               f"| DENTEX FDI Dice | {j['real_dice_fdi_mean']['mean']:.3f} {j['real_dice_fdi_mean']['ci95']} |",
               f"| DENTEX tooth detection | {j['real_tooth_detection_rate']['value']:.3f} {j['real_tooth_detection_rate']['ci95']} |",
               f"| DENTEX numbering accuracy | {j['real_tooth_numbering_acc']['value']:.3f} {j['real_tooth_numbering_acc']['ci95']} |",
               f"| DENTEX presence F1 | {j['real_presence_f1']:.3f} |",
               "| Synthetic test (105 CBCT volumes) | " + ", ".join(f"{c} {sd[c]:.3f}" for c in
                                                         ("teeth", "pulp", "canal_L", "canal_R", "incisive_canal", "sinus")) + " |",
               ""]


def zero_shot_seeds(tables: list[str]) -> None:
    arms = [("seg_v1 (original ranges)", ["seg_v1_dr_full", "seg_v1_dr_full_s1", "seg_v1_dr_full_s2"]),
            ("seg_v2 (retuned ranges)", [f"seg_v2_dr_tuned_s{s}" for s in SEEDS])]
    res = {lab: [jload(RUNS / r / "eval_best" / "metrics.json") for r in rs] for lab, rs in arms}
    if not all(all(js) for js in res.values()):
        return
    real = [("real_dice_teeth", "mean", "teeth Dice"), ("real_dice_fdi_mean", "mean", "FDI Dice"),
            ("real_tooth_detection_rate", "value", "detection"), ("real_tooth_numbering_acc", "value", "numbering")]
    syn = ["teeth", "pulp", "canal_L", "canal_R", "incisive_canal", "sinus", "implant"]
    cols = [(lambda j, k=k, s=s: j[k][s], f"DENTEX {t}") for k, s, t in real] + \
           [(lambda j, c=c: j["synth_dice"][c], f"synth {c}") for c in syn]
    rows = ["| Model | " + " | ".join(t for _, t in cols) + " |", "|---|" + "---|" * len(cols)]
    for lab, js in res.items():
        rows.append(f"| {lab} | " + " | ".join("{:.3f}±{:.3f}".format(*stats([f(j) for j in js])) for f, _ in cols) + " |")
    tables += ["### Table G — Zero-shot, full schedule, 3 seeds (mean ± sd; DENTEX test + synthetic test)", *rows, ""]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    tables: list[str] = []
    zero_shot(tables)
    zero_shot_seeds(tables)
    dentex_fewshot(tables)
    ablation(tables)
    followup(tables)
    cyclegan(tables)
    contact(tables)
    (OUT / "tables.md").write_text("\n".join(tables))
    print("\n".join(tables))
    print("→", OUT)


if __name__ == "__main__":
    main()
