# Fixed splits — never change

Created once by `scripts/make_splits.py` (seed 0). Checksums are in `SHA256SUMS.json`.
The script refuses to overwrite.

| dataset | unit | train | val | test | notes |
|---|---|---|---|---|---|
| ToothFairy3 | volume (= patient) | 374 (A 293 / B 44 / C 37) | 53 (42/6/5) | 105 (82/13/10) | stratified by set × metal × third molar |
| CONTACT | molar ROI | 878 | 156 | 444 (official test) | val = 15% of official train, stratified by group; **no patient IDs** (1010 patients / 1478 molars) |
| DENTEX | radiograph (duplicate group) | 384 few-shot | 62 | 481 | + 2098 unlabelled; dup groups never straddle splits. `dentex_annotation_subset.json`: 60 of the test images for manual canal/sinus/pulp labels |

DENTEX contains the same radiograph under different names across (and within) subsets: 638 exact-duplicate groups
plus one re-cropped copy. `dentex.json["groups"]` lists every group.
ToothFairy2 is not split; it is fully contained in TF3 Sets A+B and is not used.
