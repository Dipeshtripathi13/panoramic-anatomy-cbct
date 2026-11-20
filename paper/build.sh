#!/bin/bash
# Build the Computers in Biology and Medicine manuscript (elsarticle) → paper/latex/main.pdf.
# Figures are regenerated into outputs/paper/figures and copied here; *_DATA figures and the PDF contain licensed
# dataset images and are git-ignored.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/make_figures.py > /dev/null
$PY scripts/make_fig_simulator.py > /dev/null
$PY scripts/polish_data_figs.py > /dev/null
mkdir -p paper/latex/figures
for f in fig_simulator_DATA.png fig_dentex_overlays_DATA_paper.png fig_fewshot_dentex.pdf fig_ablation.pdf \
         fig_cyclegan_samples_DATA_paper.png fig_contact_overlays_DATA_paper.png fig_fewshot_contact.pdf; do
  cp outputs/paper/figures/$f paper/latex/figures/
done
cd paper/latex && tectonic main.tex
