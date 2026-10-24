"""Visual guide for annotating real panoramics (canal, sinus, pulp).

Synthetic panoramics from TRAIN-split CBCTs carry exact 3D-derived labels, so they show where each
structure lies. Real practice images come from DENTEX's few-shot TRAIN pool — never from the 60 test
images being annotated (splits/dentex_annotation_subset.json).

Outputs in outputs/annotation/: guide_1_patients.png, guide_2_canal.png, guide_3_sinus.png,
guide_4_pulp.png, guide_5_real_practice.png
Usage: .venv/bin/python scripts/annotation_guide.py
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dentalpano import io as dio  # noqa: E402
from dentalpano import opg, tf3  # noqa: E402
from dentalpano import randomize as DR  # noqa: E402

OUT = dio.REPO / "outputs" / "annotation"
COL = {"Canal_R": "#FF8C00", "Canal_L": "#FF0000", "Sinus_R": "#1E90FF", "Sinus_L": "#00BFFF", "Pulp": "#00FFCC"}
CHANNELS = {"Canal_R": ([4], 12.0), "Canal_L": ([3], 12.0), "Sinus_R": ([6], 20.0), "Sinus_L": ([5], 20.0),
            "Pulp": ([t + 100 for t in range(11, 49)], 12.0), "mandible": ([1], 12.0)}
PATIENTS = {"ToothFairy3F_009": "all 32 teeth", "ToothFairy3F_004": "crowns / bridge (metal = bright white)",
            "ToothFairy3P_026": "no teeth (edentulous), small scan"}
SINUS_CASES = ["ToothFairy3F_009", "ToothFairy3F_027", "ToothFairy3F_011", "ToothFairy3F_004"]


def render(case: str) -> dict:
    img, _ = tf3.load_image(case)
    lab, _ = tf3.load_label(case)
    cfg = opg.OPGConfig(channels=dict(CHANNELS))
    p = DR.nominal()
    r = opg.render(None, lab, DR.opg_config(p, cfg), mu=DR.attenuation(img, lab, p, DR.spine_mask(img, lab)))
    m = r["masks"]
    bottom = np.nonzero(m["mandible"].any(1))[0].max() + 25 if m["mandible"].any() else r["image"].shape[0]
    sl = slice(0, min(bottom, r["image"].shape[0]))
    return {"img": DR.image_from_transmission(r["meanT"], p, 0.3, np.random.default_rng(0))[sl],
            "m": {k: v[sl] for k, v in m.items()}}


def overlay(img, masks, keys, alpha=0.45):
    rgb = np.repeat(img[..., None], 3, -1).astype(float)
    for k in keys:
        mk = masks[k]
        rgb[mk] = (1 - alpha) * rgb[mk] + alpha * np.array(to_rgb(COL[k]))
    return rgb


def box(mask, pad_r=(40, 25), pad_c=(30, 30), shape=None):
    r = np.nonzero(mask.any(1))[0]
    c = np.nonzero(mask.any(0))[0]
    H, W = shape
    return slice(max(0, r.min() - pad_r[0]), min(H, r.max() + pad_r[1])), slice(max(0, c.min() - pad_c[0]), min(W, c.max() + pad_c[1]))


def pair(ax_plain, ax_lab, img, masks, keys, title, outline=True):
    ax_plain.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax_plain.set_title(title + " — try first", fontsize=10)
    ax_lab.imshow(img, cmap="gray", vmin=0, vmax=1)
    for k in keys:
        if masks[k].any():
            if outline:
                ax_lab.contour(masks[k], levels=[0.5], colors=[COL[k]], linewidths=1.6)
            else:
                ax_lab.imshow(np.ma.masked_where(~masks[k], np.ones_like(img)), cmap=matplotlib.colors.ListedColormap([COL[k]]),
                              alpha=0.45, vmin=0, vmax=1)
    ax_lab.set_title(title + " — answer", fontsize=10)
    for a in (ax_plain, ax_lab):
        a.axis("off")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {c: render(c) for c in dict.fromkeys(list(PATIENTS) + SINUS_CASES + ["ToothFairy3F_027"])}

    # 1 — whole panoramics, four kinds of patient
    fig, ax = plt.subplots(len(PATIENTS), 2, figsize=(20, 4.8 * len(PATIENTS)))
    for i, (c, desc) in enumerate(PATIENTS.items()):
        d = cache[c]
        ax[i, 0].imshow(d["img"], cmap="gray", vmin=0, vmax=1)
        ax[i, 0].set_title(f"{desc}: unlabelled", fontsize=11)
        ax[i, 1].imshow(overlay(d["img"], d["m"], [k for k in COL if d["m"][k].any()]))
        ax[i, 1].set_title("labels: orange Canal_R · red Canal_L · blue Sinus_R · light-blue Sinus_L · cyan Pulp", fontsize=11)
        ax[i, 1].text(5, 18, "IMAGE LEFT = PATIENT RIGHT", color="yellow", fontsize=10, weight="bold")
        for a in ax[i]:
            a.axis("off")
    fig.suptitle("Guide 1 — where each label goes, three different patients (synthetic, exact labels from 3D CBCT)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "guide_1_patients.png", dpi=60)
    plt.close(fig)

    # 2 — canal close-ups (both sides, three patients)
    cases = ["ToothFairy3F_009", "ToothFairy3F_027", "ToothFairy3P_026"]
    fig, ax = plt.subplots(len(cases) * 2, 2, figsize=(16, 7 * len(cases)))
    row = 0
    for c in cases:
        d = cache[c]
        for side in ("Canal_R", "Canal_L"):
            if not d["m"][side].any():
                continue
            rs, cs = box(d["m"][side], shape=d["img"].shape)
            pair(ax[row, 0], ax[row, 1], d["img"][rs, cs], {side: d["m"][side][rs, cs]}, [side],
                 f"{c[-6:]} {side} ({'image LEFT' if side == 'Canal_R' else 'image RIGHT'})")
            row += 1
    for a in ax[row:].ravel():
        a.axis("off")
    fig.suptitle("Guide 2 — mandibular canal: dark band between two thin white lines, below the molar roots,\n"
                 "from the ramus (back) running down and forward to below the premolars", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "guide_2_canal.png", dpi=60)
    plt.close(fig)

    # 3 — sinus close-ups
    fig, ax = plt.subplots(len(SINUS_CASES), 2, figsize=(18, 4.5 * len(SINUS_CASES)))
    for i, c in enumerate(SINUS_CASES):
        d = cache[c]
        both = d["m"]["Sinus_R"] | d["m"]["Sinus_L"]
        H = d["img"].shape[0]
        rs = slice(0, int(H * 0.55))
        pair(ax[i, 0], ax[i, 1], d["img"][rs], {k: d["m"][k][rs] for k in ("Sinus_R", "Sinus_L")}, ["Sinus_R", "Sinus_L"],
             f"{c[-6:]} upper jaw" + ("" if both.any() else " (no sinus in scan)"))
    fig.suptitle("Guide 3 — maxillary sinus: dark air space ABOVE the upper premolar/molar roots, one each side.\n"
                 "CBCT scans cut it off at the top — on real panoramics it is much taller (up toward the eye socket)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "guide_3_sinus.png", dpi=60)
    plt.close(fig)

    # 4 — pulp close-ups: lower molars/premolars and upper front teeth
    d = cache["ToothFairy3F_009"]
    H, W = d["img"].shape
    crops = {"lower right molars/premolars (image left)": (slice(int(H * .35), int(H * .8)), slice(int(W * .1), int(W * .42))),
             "upper front teeth": (slice(int(H * .05), int(H * .45)), slice(int(W * .35), int(W * .65))),
             "lower left molars/premolars (image right)": (slice(int(H * .35), int(H * .8)), slice(int(W * .58), int(W * .9)))}
    fig, ax = plt.subplots(len(crops), 2, figsize=(16, 6 * len(crops)))
    for i, (t, (rs, cs)) in enumerate(crops.items()):
        pair(ax[i, 0], ax[i, 1], d["img"][rs, cs], {"Pulp": d["m"]["Pulp"][rs, cs]}, ["Pulp"], t)
    fig.suptitle("Guide 4 — pulp: the darker core of each tooth — chamber in the crown + a thin dark canal down each root.\n"
                 "One polygon per tooth (first 20 images only)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "guide_4_pulp.png", dpi=60)
    plt.close(fig)

    # 5 — real practice images (DENTEX few-shot TRAIN pool, never the annotation test set)
    split = json.loads((dio.REPO / "splits" / "dentex.json").read_text())
    ann = set(json.loads((dio.REPO / "splits" / "dentex_annotation_subset.json").read_text())["images"])
    pool = [n for n in split["fewshot_train"] if n not in ann]
    pick = [pool[i] for i in np.random.default_rng(1).choice(len(pool), 3, replace=False)]
    zf = dio.open_zip("dentex", "train_zip")
    fig, ax = plt.subplots(3, 3, figsize=(24, 15))
    for i, n in enumerate(pick):
        im = np.array(Image.open(zf.open("training_data/" + n.split(":", 1)[1].replace("/", "/xrays/", 1))).convert("L"))
        H, W = im.shape
        ax[i, 0].imshow(im, cmap="gray")
        ax[i, 0].set_title(f"{n.split('/')[-1]} — whole image (R marker = image left)", fontsize=10)
        ax[i, 1].imshow(im[int(H * .45):int(H * .95), int(W * .1):int(W * .45)], cmap="gray")
        ax[i, 1].set_title("zoom: lower jaw, patient RIGHT → find Canal_R", fontsize=10)
        ax[i, 2].imshow(im[int(H * .05):int(H * .55), int(W * .12):int(W * .45)], cmap="gray")
        ax[i, 2].set_title("zoom: upper jaw, patient RIGHT → find Sinus_R", fontsize=10)
        for a in ax[i]:
            a.axis("off")
    fig.suptitle("Guide 5 — practice on real panoramics (DENTEX training pool, not your 60 test images). No answers: compare with guides 1–4",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "guide_5_real_practice.png", dpi=60)
    plt.close(fig)
    print("wrote guides to", OUT)


if __name__ == "__main__":
    main()
