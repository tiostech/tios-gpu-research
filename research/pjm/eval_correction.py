"""Score a self-correction run against its base forecast.

  python research/pjm/eval_correction.py <base_predictions.parquet> <correction_run_dir> [--mode walk_forward]

corrected = base_q500 + lam * r_q500. lam is chosen on the screening days (every 3rd day,
same as --eval-every 3 runs) and reported on the remaining holdout days, plus lam = 1.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("base", type=Path)
p.add_argument("corr", type=Path)
p.add_argument("--mode", default="walk_forward")
args = p.parse_args()

base = pd.read_parquet(args.base).assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts")
corr = pd.read_parquet(args.corr / f"val_{args.mode}_predictions.parquet")
corr = corr.assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts") if "ts" in corr else corr
b = base.loc[corr.index] if corr.index.dtype.kind == "M" else None
assert b is not None, "correction predictions need a ts column"
y = b["y_true"]
ok = y.notna()
days = corr.index.floor("D")
day_num = (days - days.min()).days
screen = ((day_num % 3) == 0) & ok
hold = ((day_num % 3) != 0) & ok


def score(q, m):
    e = y[m] - q[m]
    return np.mean(np.abs(e)), np.sqrt(np.mean(e**2))


lams = np.round(np.arange(0, 1.51, 0.1), 2)
best = min(lams, key=lambda lam: score(b["q500"] + lam * corr["q500"], screen)[0])
print(f"lambda chosen on screen days: {best}")
for name, m in (("ALL", ok), ("SCREEN", screen), ("HOLD", hold)):
    bm, br = score(b["q500"], m)
    line = f"{name:6} base MAE {bm:.3f} RMSE {br:.2f}"
    for lam in (1.0, best):
        cm, cr = score(b["q500"] + lam * corr["q500"], m)
        line += f" | lam={lam}: MAE {cm:.3f} ({cm - bm:+.3f}) RMSE {cr:.2f} ({cr - br:+.2f})"
    print(line)
