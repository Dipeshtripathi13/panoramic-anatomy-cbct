"""Select and export the real-panoramic annotation set (DENTEX test split) for canal / sinus / pulp labels.

- Picks N radiographs from splits/dentex.json["test"], one per duplicate group, only from the
  quadrant_enumeration subset (which also has tooth polygons → same images serve the teeth eval).
- Writes splits/dentex_annotation_subset.json ONCE (refuses to overwrite) — these are TEST images:
  never train or tune on them.
- Exports the PNGs to data/annotation/dentex_ann/images/ (git-ignored) and the Label Studio
  labelling config to data/annotation/dentex_ann/label_config.xml.
Protocol: docs/annotation_protocol.md

Usage: .venv/bin/python scripts/make_annotation_set.py [--n 60]
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from dentalpano import io as dio

OUT = dio.REPO / "data" / "annotation" / "dentex_ann"
SUBSET = dio.REPO / "splits" / "dentex_annotation_subset.json"

LABEL_CONFIG = """<View>
  <Header value="Image left = patient RIGHT (look for the R marker). Label only what you can see."/>
  <Image name="img" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <PolygonLabels name="poly" toName="img" strokeWidth="2" pointSize="small" opacity="0.35">
    <Label value="Canal_R" background="#FF8C00"/>
    <Label value="Canal_L" background="#FF0000"/>
    <Label value="Sinus_R" background="#1E90FF"/>
    <Label value="Sinus_L" background="#00BFFF"/>
    <Label value="Pulp" background="#00FFCC"/>
  </PolygonLabels>
  <Header value="Canal visibility (patient right / left)"/>
  <Choices name="canal_R_vis" toName="img" choice="single" showInline="true">
    <Choice value="R: clear"/><Choice value="R: partial"/><Choice value="R: not visible"/>
  </Choices>
  <Choices name="canal_L_vis" toName="img" choice="single" showInline="true">
    <Choice value="L: clear"/><Choice value="L: partial"/><Choice value="L: not visible"/>
  </Choices>
  <Header value="Image"/>
  <Choices name="quality" toName="img" choice="single" showInline="true">
    <Choice value="quality ok"/><Choice value="quality poor"/>
  </Choices>
  <Choices name="pulp_done" toName="img" choice="single" showInline="true">
    <Choice value="pulp labelled"/><Choice value="pulp skipped"/>
  </Choices>
  <TextArea name="notes" toName="img" placeholder="notes (optional)" rows="2"/>
</View>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = json.loads((dio.REPO / "splits" / "dentex.json").read_text())
    test = set(d["test"])
    candidates = []
    for grp in d["groups"]:
        members = [m for m in grp if m in test and m.startswith("train:quadrant_enumeration/")]
        if members:
            candidates.append(sorted(members)[0])          # one representative per duplicate group
    candidates.sort()
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(candidates, args.n, replace=False).tolist())

    if SUBSET.exists():
        old = json.loads(SUBSET.read_text())
        assert old["images"] == pick, "annotation subset exists with different images — never change it"
    else:
        SUBSET.write_text(json.dumps({"seed": args.seed, "n": args.n, "source": "dentex.json test, quadrant_enumeration, 1 per group",
                                      "images": pick, "pulp_subset": pick[:20],
                                      "note": "TEST images: never train or tune on them"}, indent=1))
        print("wrote", SUBSET)

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    zf = dio.open_zip("dentex", "train_zip")
    for name in pick:
        member = "training_data/" + name.split(":", 1)[1].replace("/", "/xrays/", 1)
        target = img_dir / name.split(":", 1)[1].replace("/", "__")
        if not target.exists():
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    (OUT / "label_config.xml").write_text(LABEL_CONFIG)
    print(f"exported {len(pick)} images to {img_dir}; pulp on the first 20 (alphabetical) = splits file 'pulp_subset'")


if __name__ == "__main__":
    main()
