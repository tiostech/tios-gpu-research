"""Score RT runs against a reference on the whole predictive distribution, not just the median (2025 only).

  uv run python research/pjm_rt/score_dist.py v0_congfc_rt_daily v0_congfc10_rt_daily [--ref v0_rt_daily]

For each run vs the reference (outputs/pjm-fe/<run>), on held-out days (day number not a multiple of 3), the full
year, and both without Mondays (the pure-shadow congestion forecast is null on 2025 Mondays): daily-paired gains with
a day-block bootstrap (2000 draws) for MAE of the median, pinball loss at q05 / q90 / q95, the average pinball loss
over the 7 stored quantiles, and MAE of the mean. Positive gain = the run is better than the reference.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--ref", default="v0_rt_daily")
a = p.parse_args()
Y0 = pd.Timestamp("2025-01-01")
idx = pd.date_range(Y0, "2025-12-31 23:00", freq="h")
QS = (("q050", .05), ("q100", .1), ("q250", .25), ("q500", .5), ("q750", .75), ("q900", .9), ("q950", .95))


def load(r):
    d = pd.read_parquet(ROOT / f"outputs/pjm-fe/{r}/val_walk_forward_predictions.parquet")
    return d.assign(ts=pd.to_datetime(d["ts"])).set_index("ts").reindex(idx)


ref = load(a.ref)
rt = ref["y_true"]
day = pd.Series(idx.normalize(), idx)
dn = pd.Series((idx.normalize() - Y0).days, idx)
mon = pd.Series(idx.dayofweek == 0, idx)


def pin(pr, q, t):
    e = rt - pr[q]
    return np.maximum(t * e, (t - 1) * e)


LOSS = {
    "MAE of median": lambda pr: (rt - pr["q500"]).abs(),
    "pinball q05": lambda pr: pin(pr, "q050", .05),
    "pinball q90": lambda pr: pin(pr, "q900", .9),
    "pinball q95": lambda pr: pin(pr, "q950", .95),
    "avg pinball 7 q": lambda pr: sum(pin(pr, q, t) for q, t in QS) / len(QS),
    "MAE of mean": lambda pr: (rt - pr["mean"]).abs(),
}
for r in a.runs:
    pr = load(r)
    ok = rt.notna() & ref["q500"].notna() & pr["q500"].notna()
    print(f"== {r} vs {a.ref}")
    for nm, m in (("HOLD", ok & (dn % 3 != 0)), ("HOLD non-Mon", ok & (dn % 3 != 0) & ~mon), ("ALL", ok), ("ALL non-Mon", ok & ~mon)):
        rng = np.random.default_rng(0)
        out = []
        for lab, f in LOSS.items():
            db, dr = f(ref)[m].groupby(day[m]).mean(), f(pr)[m].groupby(day[m]).mean()
            d = (db - dr).to_numpy()
            se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
            out.append(f"{lab} {db.mean():.2f}->{dr.mean():.2f} {d.mean():+.3f}+-{se:.3f}{'*' if d.mean() > 2 * se else '!' if d.mean() < -2 * se else ''}")
        print(f"  {nm:13} " + " | ".join(out))
print("* = gain above 2 SE; ! = loss above 2 SE")
