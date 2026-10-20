"""Day-level dataset: one row per delivery day, so Kumo's context can hold the full 2018-2024 history.

  uv run python research/pjm/make_daily.py <hourly_variant> <name> [--target mean|peak]

Target y = the day's mean price (`mean`) or the mean over hour-beginning 16-20 (`peak`, the block
with most of the error). Features = daily means of the hourly variant's features (all fair at the
D-1 cutoff), plus min/max of the stress drivers. Split: train < 2025-01-01 <= val; ts = the date.
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "pjm-fe"
STRESS = ["supply_cushion_ratio", "supply_cushion", "thermal_stress_all_pjm", "net_load", "pjm_load",
          "feelslike_weightedavg", "wx_temp_mean", "load_ens_std"]

p = argparse.ArgumentParser()
p.add_argument("variant")
p.add_argument("name")
p.add_argument("--target", default="mean", choices=["mean", "peak"])
args = p.parse_args()

df = pd.concat(pd.read_parquet(ROOT / args.variant / f"{s}.parquet") for s in ("train", "val", "test"))
df["ts"] = pd.to_datetime(df["ts"])
day = df["ts"].dt.floor("D")
feats = [c for c in df.columns if c not in ("ts", "y") and pd.api.types.is_numeric_dtype(df[c])]
out = df[feats].groupby(day).mean()
pk = df["ts"].dt.hour.between(16, 20)
for c in STRESS:
    if c in df:
        out[f"{c}_daymax"] = df[c].groupby(day).max()
        out[f"{c}_daymin"] = df[c].groupby(day).min()
        out[f"{c}_peakmean"] = df.loc[pk, c].groupby(day[pk]).mean()
y = df["y"] if args.target == "mean" else df["y"].where(pk)
out["y"] = y.groupby(day).mean()
out = out.rename_axis("ts").reset_index()
cut = pd.Timestamp("2025-01-01")
dst = ROOT / args.name
dst.mkdir(parents=True, exist_ok=True)
out[out["ts"] < cut].to_parquet(dst / "train.parquet", index=False)
out[(out["ts"] >= cut) & (out["ts"] < pd.Timestamp("2026-01-01"))].to_parquet(dst / "val.parquet", index=False)
print(f"{args.name}: {out.shape[1] - 2} features, train {(out['ts'] < cut).sum()} days, "
      f"val {((out['ts'] >= cut) & (out['ts'] < pd.Timestamp('2026-01-01'))).sum()} days")
