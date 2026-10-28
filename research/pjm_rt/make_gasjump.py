"""RT dataset with gas-price jump features (fair: built from blended_gasprice, the model's gas input at the cutoff).

  uv run python research/pjm_rt/make_gasjump.py

Base v0_x5rt_25 (RT target, 5 congestion PCs + MISO / SPP DA forecasts; val = 2025, test = 2026). For day D, with
g(D) = day-D mean of blended_gasprice:
  gas_rel30  log(g(D) / median of g over D-30..D-1)   (the jump generators' offers pass through)
  gas_chg1   log(g(D) / g(D-1))
-> v0_x5rt_gasjump_25
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/pjm-fe"
S = {s: pd.read_parquet(D / "v0_x5rt_25" / f"{s}.parquet") for s in ("train", "val", "test")}
allr = pd.concat(S.values(), ignore_index=True)
day = pd.to_datetime(allr["ts"]).dt.normalize()
g = allr["blended_gasprice"].groupby(day).mean().asfreq("D")
rel = np.log(g / g.shift(1).rolling(30, min_periods=15).median())
chg = np.log(g / g.shift(1))
out = D / "v0_x5rt_gasjump_25"
out.mkdir(exist_ok=True)
for s, df in S.items():
    d = pd.to_datetime(df["ts"]).dt.normalize()
    df["gas_rel30"], df["gas_chg1"] = d.map(rel).to_numpy(), d.map(chg).to_numpy()
    df.to_parquet(out / f"{s}.parquet", index=False)
    print(s, len(df), "nulls", int(df.gas_rel30.isna().sum()), int(df.gas_chg1.isna().sum()),
          "| 2025 range" if s == "val" else "", (f"{df.gas_rel30.min():.2f}..{df.gas_rel30.max():.2f}" if s == "val" else ""))
