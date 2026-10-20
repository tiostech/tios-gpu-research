"""Build the self-correction dataset: target = base model's out-of-sample error.

  uv run python research/pjm/make_correction.py <base_predictions.parquet> <base_variant> <name>
      [--correct-from 2025-01-01]

base_predictions: walk-forward predictions of the base model (ts, y_true, mean, q050..q950),
covering a warm-up year (context for the correction layer) and the correction year.
base_variant: the feature variant the base used (for fundamentals / price history).

Target r = y - base_q500; the corrected forecast is base_q500 + predicted r.
Features favour *changes* and the base model's own state over plain magnitudes:
  * base forecast state: median, spread, skew, change vs yesterday's actual / 7-day level
  * recent base errors (fair: D-1 actuals are known before the D-1 morning deadline)
  * 24h / 168h changes of fundamentals, weather, outages
Rows before --correct-from are train (context), the rest val (walk-forward).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from make_features import daily_lag, lag

ROOT = Path(__file__).resolve().parents[2] / "data" / "pjm-fe"

p = argparse.ArgumentParser()
p.add_argument("base_predictions", type=Path)
p.add_argument("base_variant")
p.add_argument("name")
p.add_argument("--correct-from", default="2025-01-01")
args = p.parse_args()

base = pd.read_parquet(args.base_predictions)
base["ts"] = pd.to_datetime(base["ts"])
base = base.set_index("ts").sort_index()
feat = pd.concat(pd.read_parquet(ROOT / args.base_variant / f"{s}.parquet") for s in ("train", "val", "test"))
feat["ts"] = pd.to_datetime(feat["ts"])
feat = feat.set_index("ts").sort_index()

# Full hourly grid over the base window so lags are exact.
idx = pd.date_range(base.index[0], base.index[-1], freq="h")
f = feat.reindex(idx)
b = base.reindex(idx)
y = f["y"]
q500 = b["q500"]

out = pd.DataFrame(index=idx)
out["r"] = y - q500  # target

# Base forecast state
out["base_q500"] = q500
out["base_band80"] = b["q900"] - b["q100"]
out["base_iqr"] = b["q750"] - b["q250"]
out["base_skew"] = (b["q900"] - q500) - (q500 - b["q100"])
out["base_mean_minus_med"] = b["mean"] - q500
out["base_vs_y_lag24"] = q500 - f["y_lag24"]
out["base_vs_y_7d"] = q500 - f["y_samehour_7d_mean"]
out["base_daymean_vs_y_d1"] = q500.groupby(idx.floor("D")).transform("mean") - f["y_d1_mean"]

# Recent base errors (D-1 and earlier)
r = out["r"]
out["r_lag24"] = lag(r, 24)
out["r_lag48"] = lag(r, 48)
out["r_samehour_7d_mean"] = pd.concat([lag(r, 24 * d) for d in range(1, 8)], axis=1).mean(axis=1)
out["r_d1_mean"] = daily_lag(r, 1, "mean")
out["r_d1_absmean"] = daily_lag(r.abs(), 1, "mean")
out["r_d1_max"] = daily_lag(r, 1, "max")
out["r_d1_min"] = daily_lag(r, 1, "min")
out["r_d2_mean"] = daily_lag(r, 2, "mean")
out["r_7d_mean"] = pd.concat([daily_lag(r, d, "mean") for d in range(1, 8)], axis=1).mean(axis=1)
out["r_7d_absmean"] = pd.concat([daily_lag(r.abs(), d, "mean") for d in range(1, 8)], axis=1).mean(axis=1)

# Changes in fundamentals / weather / outages
for c in ("net_load_delta24", "thermal_stress_delta24", "wx_temp_delta24", "wx_feelslike_drop4h",
          "y_d1_vs_d2", "y_lag24_vs_7d"):
    out[c] = f[c]
for c, h in (("pjm_load", 24), ("pjm_load", 168), ("pjm_wind", 24), ("pjm_solar", 24),
             ("pjm_gen_on_outage", 24), ("supply_cushion", 24), ("thermal_stress_all_miso", 24),
             ("feelslike_weightedavg", 24), ("load_ens_std", 24)):
    out[f"{c}_d{h}"] = f[c] - lag(f[c], h)
out["gas_pct_d24"] = f["smoothed_gas"] / lag(f["smoothed_gas"], 24) - 1
out["forced_out_d24"] = f["out_forced_outages"] - lag(f["out_forced_outages"], 24)
out["supply_cushion_ratio"] = f["supply_cushion_ratio"]
out["hour"] = idx.hour
out["dow"] = idx.dayofweek

out = out.replace([np.inf, -np.inf], np.nan)
out = out[q500.notna()].rename_axis("ts").reset_index()
dst = ROOT / args.name
dst.mkdir(parents=True, exist_ok=True)
cut = pd.Timestamp(args.correct_from)
out[out["ts"] < cut].to_parquet(dst / "train.parquet", index=False)
out[out["ts"] >= cut].to_parquet(dst / "val.parquet", index=False)
print(f"{args.name}: {out.shape[1] - 2} features, train {(out['ts'] < cut).sum()}, "
      f"val {(out['ts'] >= cut).sum()} -> {dst}")
