"""Create the fixed train/val/test splits ONCE. Refuses to overwrite existing split files.

ToothFairy3 : patient(=volume)-level, stratified by set (A/B/C) × metal × third-molar presence; 70/10/20.
CONTACT     : official Train/Test kept (test = test). val = 15% of official train, stratified by
              derived group. No patient IDs are released (1010 patients / 1478 molars), so molars
              of one patient may straddle train/val — documented, cannot be fixed.
DENTEX      : images grouped by exact identity (zip CRC) and near-duplicates (2048-bit dHash of a
              65×32 thumbnail, Hamming ≤ 400), because the same radiograph appears in several
              subsets under different names. Nearest-neighbour distances are bimodal: 0 (identical)
              vs ≥ 554 (different patients), with one re-cropped copy at 394 (checked visually).
              (A first attempt with a 64-bit 9×8 hash merged unrelated panoramics into giant groups.) Whole groups go to one split. Labelled quadrant_enumeration groups are split
              60/10/30 (train = few-shot pool). Every other group not touching val/test is
              "unlabelled" (for unsupervised adaptation); images < 1000×500 are dropped.

Usage: .venv/bin/python scripts/make_splits.py
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from dentalpano import contact
from dentalpano import io as dio

SPLITS = dio.REPO / "splits"
SEED = 0


def write_once(name: str, obj) -> None:
    path = SPLITS / name
    if path.exists():
        raise SystemExit(f"{path} exists — splits are fixed and must never change.")
    path.write_text(json.dumps(obj, indent=1))
    print("wrote", path)


def stratified(keys: pd.Series, fracs: dict, rng) -> pd.Series:
    """Assign split names within each stratum, preserving fractions."""
    out = pd.Series(index=keys.index, dtype=object)
    for _, idx in keys.groupby(keys).groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        n = len(idx)
        bounds = np.cumsum([round(f * n) for f in fracs.values()])
        bounds[-1] = n
        start = 0
        for name, b in zip(fracs, bounds):
            out[idx[start:b]] = name
            start = b
    return out


# ------------------------------------------------------------------ ToothFairy3
def tf3_splits(rng) -> dict:
    df = pd.read_csv(dio.REPO / "outputs" / "audit_tf3.csv")
    lc = df.label_counts.apply(ast.literal_eval)
    metal = lc.apply(lambda d: any(k in d for k in (8, 9, 10)))
    m3 = lc.apply(lambda d: any(k in d for k in (38, 48)))
    key = df.set + "_" + metal.map({True: "metal", False: "nometal"}) + "_" + m3.map({True: "m3", False: "nom3"})
    df["split"] = stratified(key, {"train": 0.7, "val": 0.1, "test": 0.2}, rng)
    out = {s: sorted(df.case[df.split == s]) for s in ("train", "val", "test")}
    print("TF3", pd.crosstab(df.set, df.split).to_dict())
    return out


# ------------------------------------------------------------------ CONTACT
def contact_splits(rng) -> dict:
    zf = dio.open_zip("contact")
    t = contact.label_table(zf)
    crc = collections.defaultdict(list)
    for i in zf.infolist():
        if "Images" in i.filename:
            crc[i.CRC].append(i.filename)
    dups = [v for v in crc.values() if len(v) > 1]
    tr = t[t.split == "train"].copy()
    tr["s"] = stratified(tr.group.astype(str), {"train": 0.85, "val": 0.15}, rng)
    ident = lambda d: [f"{r.split}/{r.ImageNum}" for r in d.itertuples()]
    out = {"train": ident(tr[tr.s == "train"]), "val": ident(tr[tr.s == "val"]), "test": ident(t[t.split == "test"]),
           "note": "ids are '<official split>/<ImageNum>'; no patient IDs available",
           "exact_duplicate_images": dups}
    print("CONTACT", {k: len(v) for k, v in out.items() if k in ("train", "val", "test")}, "exact dup groups:", len(dups))
    return out


# ------------------------------------------------------------------ DENTEX
def dhash(args) -> tuple[str, np.ndarray, int, int]:
    key, member, name = args
    zf = dio.open_zip("dentex", key)
    im = Image.open(zf.open(member)).convert("L")
    w, h = im.size
    a = np.asarray(im.resize((65, 32), Image.BILINEAR), dtype=np.float32)
    return name, (a[:, 1:] > a[:, :-1]).ravel(), w, h


def dentex_splits(rng) -> dict:
    items = []
    crc = {}
    for key in ("train_zip", "val_zip"):
        zf = dio.open_zip("dentex", key)
        for i in zf.infolist():
            if i.filename.endswith(".png") and ".ipynb_checkpoints" not in i.filename:
                name = f"{key.split('_')[0]}:{i.filename.split('/')[1]}/{i.filename.split('/')[-1]}"
                items.append((key, i.filename, name))
                crc[name] = i.CRC
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(dhash, items, chunksize=16))
    names = [r[0] for r in res]
    B = np.stack([r[1] for r in res]).astype(np.float32)
    size = {r[0]: (r[2], r[3]) for r in res}

    # union-find over exact CRC matches and near-duplicate dHash
    parent = list(range(len(names)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    by_crc = collections.defaultdict(list)
    for i, n in enumerate(names):
        by_crc[crc[n]].append(i)
    for idx in by_crc.values():
        for j in idx[1:]:
            union(idx[0], j)
    ham = B @ (1 - B).T + (1 - B) @ B.T
    near = np.argwhere(np.triu(ham <= 400, 1))
    for a, b in near:
        union(a, b)
    groups = collections.defaultdict(list)
    for i, n in enumerate(names):
        groups[find(i)].append(n)
    groups = list(groups.values())
    n_exact = sum(len(v) > 1 for v in by_crc.values())

    labelled = [g for g in groups if any(n.startswith("train:quadrant_enumeration/") for n in g)]
    val_zip = [g for g in groups if any(n.startswith("val:") for n in g)]
    order = rng.permutation(len(labelled))
    n = len(labelled)
    cut1, cut2 = round(0.6 * n), round(0.7 * n)
    split_of = {}
    for rank, gi in enumerate(order):
        split_of[id(labelled[gi])] = "train" if rank < cut1 else ("val" if rank < cut2 else "test")
    # DENTEX's own validation images are held out with test (their labels are not in our zip)
    for g in val_zip:
        split_of.setdefault(id(g), "test")
    out = {"fewshot_train": [], "val": [], "test": [], "unlabelled": [], "dropped_small": [], "groups": []}
    for g in groups:
        s = split_of.get(id(g))
        out["groups"].append(sorted(g))
        if s is None:
            for nme in g:
                (out["dropped_small"] if min(size[nme]) < 500 or size[nme][0] < 1000 else out["unlabelled"]).append(nme)
            continue
        key = {"train": "fewshot_train", "val": "val", "test": "test"}[s]
        out[key].extend(nme for nme in g if nme.startswith("train:quadrant_enumeration/") or s == "test")
    for k in ("fewshot_train", "val", "test", "unlabelled", "dropped_small"):
        out[k] = sorted(out[k])
    out["note"] = ("names are '<zip>:<subset>/<file>'. Splits are made over duplicate groups; fewshot_train/val list "
                   "only quadrant_enumeration members (labels = that subset's JSON); test lists every member of "
                   "test groups (incl. DENTEX val images); unlabelled = all members of untouched groups.")
    out["stats"] = {"images": len(names), "groups": len(groups), "exact_dup_groups": n_exact,
                    "near_dup_pairs": int(len(near)), "labelled_groups": n}
    print("DENTEX", {k: len(out[k]) for k in ("fewshot_train", "val", "test", "unlabelled", "dropped_small")}, out["stats"])
    return out


def main() -> None:
    SPLITS.mkdir(exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=dio.REPO).stdout.strip()
    meta = {"seed": SEED, "git": git}
    write_once("toothfairy3.json", {**meta, **tf3_splits(np.random.default_rng(SEED))})
    write_once("contact.json", {**meta, **contact_splits(np.random.default_rng(SEED + 1))})
    write_once("dentex.json", {**meta, **dentex_splits(np.random.default_rng(SEED + 2))})
    digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SPLITS.glob("*.json"))}
    write_once("SHA256SUMS.json", digest)


if __name__ == "__main__":
    main()
