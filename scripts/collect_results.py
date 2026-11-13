"""Gather every run under a runs/ directory into one table: training summary from log.csv and test metrics from
eval_<ckpt>/metrics.json (blank if not evaluated yet).

Usage: python scripts/collect_results.py --runs /path/to/runs [--ckpt best] → <runs>/summary.csv (+ printed)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLS = ["run", "epochs_done", "best_epoch", "val_real_teeth_at_best", "test_real_teeth_dice", "test_real_teeth_ci",
        "test_real_fdi_dice", "test_detection", "test_numbering", "test_presence_f1", "test_synth_teeth",
        "test_synth_canal_L", "test_synth_canal_R", "test_synth_pulp", "test_synth_sinus", "test_synth_incisive"]


def summarise(run: Path, ckpt: str) -> dict:
    row = {"run": run.name}
    log = run / "log.csv"
    if log.exists():
        rows = list(csv.DictReader(open(log)))
        row["epochs_done"] = len(rows)
        sel = json.loads((run / "config.json").read_text()).get("select_on", "synth")
        key = f"{sel}_dice_mean"
        scored = [r for r in rows if r.get(key) not in (None, "", "nan")]
        if scored:
            b = max(scored, key=lambda r: float(r[key]))
            row["best_epoch"], row["val_real_teeth_at_best"] = b["epoch"], b.get("real_dice_teeth", "")
    m = run / f"eval_{ckpt}" / "metrics.json"
    if m.exists():
        j = json.loads(m.read_text())
        g = lambda k, f="mean": j.get(k, {}).get(f, "")
        row.update(test_real_teeth_dice=g("real_dice_teeth"), test_real_teeth_ci=j.get("real_dice_teeth", {}).get("ci95", ""),
                   test_real_fdi_dice=g("real_dice_fdi_mean"), test_detection=g("real_tooth_detection_rate", "value"),
                   test_numbering=g("real_tooth_numbering_acc", "value"), test_presence_f1=j.get("real_presence_f1", ""))
        sd = j.get("synth_dice", {})
        for c, k in (("teeth", "teeth"), ("canal_L", "canal_L"), ("canal_R", "canal_R"), ("pulp", "pulp"),
                     ("sinus", "sinus"), ("incisive", "incisive_canal")):
            row[f"test_synth_{c}"] = sd.get(k, "")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--ckpt", default="best")
    args = ap.parse_args()
    rows = [summarise(r, args.ckpt) for r in sorted(args.runs.iterdir()) if (r / "config.json").exists()]
    with open(args.runs / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["run"], {k: r.get(k, "") for k in COLS[1:8]})
    print("→", args.runs / "summary.csv")


if __name__ == "__main__":
    main()
