# Annotation protocol — real panoramic test labels (canal, sinus, pulp)

> **Status (2026-10-24): on hold.** Intended for an expert annotator (dentist / oral radiologist). The project author is
> not labelling these images.

**Images:** 60 DENTEX test radiographs, listed in `splits/dentex_annotation_subset.json`, exported to
`data/annotation/dentex_ann/images/`. They are **test** images: never train, tune or choose hyperparameters on them.

**Rules that keep the evaluation valid**
1. Label **before** looking at any model prediction on these images. Never edit a label after seeing model output.
2. Label what is **visible on the radiograph**. Do not infer anatomy you cannot see.
3. Write down the time spent and anything unusual in the per-image *notes* box.
4. Expert check: a dentist or oral radiologist independently labels the first 15 images (alphabetical). We report agreement
   (Dice and 95th-percentile surface distance) between the two labellers.

**Orientation:** image left = patient **right** (DENTEX images carry an *R* marker on the left).

## Structures

### Mandibular canal — `Canal_R`, `Canal_L` (one polygon per side)
- Outline the canal **lumen as a band**: trace its upper cortical border forward, then its lower border back, so the polygon
  encloses the dark canal between the two white lines.
- **Extent:** from where it is first visible in the ramus (mandibular foramen region) to the mental foramen.
  Do not extend into the incisive canal.
- **Gaps:** where it disappears (for example behind a root), do not bridge the gap. Make separate polygons and set visibility to *partial*.
- Set the per-side visibility choice on every image:
  - **clear:** both borders visible over most of the length;
  - **partial:** visible in pieces, or only one border;
  - **not visible:** no polygon for that side.

### Maxillary sinus — `Sinus_R`, `Sinus_L`
- Outline the **radiolucent sinus cavity** as visible: floor (over the molar and premolar roots), anterior wall, and the
  posterior/superior walls as far as they are visible.
- If the top of the sinus is not visible or ambiguous, close the polygon along the highest clearly visible border and add a note.
- Leave it unlabelled when you cannot identify the floor.

### Pulp — `Pulp` (first 20 images only; set *pulp labelled* / *pulp skipped*)
- One polygon per tooth: pulp chamber + root canals as visible, as one connected outline where possible.
- Teeth with root-canal fillings: outline the radiopaque filling as pulp and add a note.
- Skip teeth whose pulp you cannot distinguish.

## Tool (Label Studio, local)
```
uv tool install label-studio          # or: pip install label-studio  (one-time, local)
label-studio start                    # opens http://localhost:8080
```
1. Create a project, then *Settings → Labeling Interface → Code*: paste `data/annotation/dentex_ann/label_config.xml`.
2. *Import*: drag in the 60 PNGs from `data/annotation/dentex_ann/images/`.
3. Polygons: choose a label, click points, and click the first point to close. Hold Ctrl/Cmd to add points while zoomed.
4. Export when done (or as a weekly backup): *Export → JSON* → save to `data/annotation/dentex_ann/export_<date>.json`.
