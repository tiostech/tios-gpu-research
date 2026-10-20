"""Re-split a feature variant so the walk-forward ("val") covers a different window.

  uv run python research/pjm/resplit.py <variant> <new_name> --val-start 2024-01-01 --val-end 2025-12-31

Rows before --val-start become train (context); rows after --val-end are dropped.
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "pjm-fe"

p = argparse.ArgumentParser()
p.add_argument("variant")
p.add_argument("name")
p.add_argument("--val-start", required=True)
p.add_argument("--val-end", required=True)
args = p.parse_args()

df = pd.concat(pd.read_parquet(ROOT / args.variant / f"{s}.parquet") for s in ("train", "val", "test"))
ts = pd.to_datetime(df["ts"])
out = ROOT / args.name
out.mkdir(parents=True, exist_ok=True)
start, end = pd.Timestamp(args.val_start), pd.Timestamp(args.val_end) + pd.Timedelta(days=1)
df[ts < start].to_parquet(out / "train.parquet", index=False)
df[(ts >= start) & (ts < end)].to_parquet(out / "val.parquet", index=False)
print(f"{args.name}: train {(ts < start).sum()} rows, val {((ts >= start) & (ts < end)).sum()} rows")
