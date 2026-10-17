"""Tabulate Kumo runs against the daily baseline on the same timestamps.

  python research/pjm/results.py [outputs_root] [baseline_predictions.parquet]

Because each walk-forward day's forecast depends only on the context before it,
the full daily baseline restricted to a run's timestamps is the exact baseline
for --eval-every runs (paired comparison: d_MAE = run - baseline, < 0 is better).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/outputs/pjm-fe")
base_path = Path(
    sys.argv[2] if len(sys.argv) > 2
    else "/scratch/outputs/pjm/val_walk_forward_predictions.parquet"
)
base = pd.read_parquet(base_path).set_index("ts")


def score(y, q500, mean):
    ok = ~np.isnan(y)
    y, q500, mean = y[ok], q500[ok], mean[ok]
    return (np.mean(np.abs(y - q500)), np.sqrt(np.mean((y - q500) ** 2)),
            np.sqrt(np.mean((y - mean) ** 2)), ok.sum())


def boot_se(run_res, base_res, days, reps=2000, seed=0):
    """Day-block bootstrap SEs of MAE and RMSE differences (run - base)."""
    d = pd.DataFrame({"ra": np.abs(run_res), "ba": np.abs(base_res),
                      "rs": run_res**2, "bs": base_res**2, "day": days}).dropna()
    per_day = d.groupby("day")[["ra", "ba", "rs", "bs"]].sum()
    per_day["n"] = d.groupby("day").size()
    per_day = per_day.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_day), (reps, len(per_day)))
    t = per_day[idx].sum(axis=1)  # (reps, 5)
    dmae = (t[:, 0] - t[:, 1]) / t[:, 4]
    drmse = np.sqrt(t[:, 2] / t[:, 4]) - np.sqrt(t[:, 3] / t[:, 4])
    return float(np.std(dmae)), float(np.std(drmse))


print(f"{'run':30} {'mode':24} {'n':>5} {'MAE':>6} {'RMSEmed':>8} {'RMSEmean':>8}"
      f" | {'baseMAE':>7} {'dMAE':>6} {'(se)':>6} {'dRMSEmed':>8} {'(se)':>6}")
for f in sorted(root.glob("*/val_*_predictions.parquet")):
    run = pd.read_parquet(f)
    mode = f.name.removeprefix("val_").removesuffix("_predictions.parquet")
    y = run["y_true"].to_numpy(float)
    mae, rmse_med, rmse_mean, n = score(y, run["q500"].to_numpy(), run["mean"].to_numpy())
    b = base.loc[run["ts"]]
    bmae, brmse_med, _, _ = score(y, b["q500"].to_numpy(), b["mean"].to_numpy())
    se_mae, se_rmse = boot_se(y - run["q500"].to_numpy(), y - b["q500"].to_numpy(),
                              pd.to_datetime(run["ts"]).dt.floor("D").to_numpy())
    print(f"{f.parent.name:30} {mode:24} {n:5d} {mae:6.3f} {rmse_med:8.3f} {rmse_mean:8.3f}"
          f" | {bmae:7.3f} {mae - bmae:+6.3f} {se_mae:6.3f} {rmse_med - brmse_med:+8.3f} {se_rmse:6.3f}")
