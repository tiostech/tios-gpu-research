"""Tabulate Kumo metrics: python research/pjm/results.py [/scratch/outputs/pjm-fe]"""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/outputs/pjm-fe")
rows = []
for f in sorted(root.glob("*/val_*_metrics.json")):
    m = json.loads(f.read_text())
    rows.append((f.parent.name, m["mode"], m["n"], m["mae_median"],
                 m.get("rmse_median", float("nan")), m["rmse_mean"],
                 m["pinball_q10_q90"], m["coverage_80"]))
print(f"{'run':32} {'mode':18} {'n':>5} {'MAE':>6} {'RMSEmed':>8} {'RMSEmean':>8} {'pinball':>7} {'cov80':>6}")
for r in sorted(rows, key=lambda r: (r[1], r[3])):
    print(f"{r[0]:32} {r[1]:18} {r[2]:5d} {r[3]:6.3f} {r[4]:8.3f} {r[5]:8.3f} {r[6]:7.3f} {r[7]:6.3f}")
