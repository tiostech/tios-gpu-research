"""Average the quantile forecasts of several Kumo runs into one prediction file.

  python research/pjm/ensemble.py OUT_DIR RUN_DIR [RUN_DIR ...] [--mode walk_forward]

Each RUN_DIR holds val_<mode>_predictions.parquet; quantiles and mean are averaged per ts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("runs", type=Path, nargs="+")
p.add_argument("--mode", default="walk_forward")
args = p.parse_args()

name = f"val_{args.mode}_predictions.parquet"
preds = [pd.read_parquet(r / name).set_index("ts") for r in args.runs]
cols = [c for c in preds[0].columns if c != "y_true"]
ens = sum(d[cols] for d in preds) / len(preds)
ens.insert(0, "y_true", preds[0]["y_true"])

args.out.mkdir(parents=True, exist_ok=True)
ens.reset_index().to_parquet(args.out / name, index=False)

ok = ens["y_true"].notna()
y, q = ens.loc[ok, "y_true"], ens.loc[ok, "q500"]
m = {"n": int(ok.sum()), "mae_median": float(np.mean(np.abs(y - q))),
     "rmse_median": float(np.sqrt(np.mean((y - q) ** 2))),
     "rmse_mean": float(np.sqrt(np.mean((y - ens.loc[ok, "mean"]) ** 2))),
     "coverage_80": float(((y >= ens.loc[ok, "q100"]) & (y <= ens.loc[ok, "q900"])).mean()),
     "members": [str(r) for r in args.runs]}
(args.out / f"val_{args.mode}_metrics.json").write_text(json.dumps(m, indent=2))
print(json.dumps(m))
