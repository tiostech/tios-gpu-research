"""RT-target copies of PJM energy-model datasets: y = rt_energy_price instead of da_energy_price.

  uv run python research/pjm_rt/make_rt.py v0 v0_custom_all_met v0_actuals_all [--data DATA]

For each data/pjm-fe/<src>/{train,val,test}.parquet writes data/pjm-fe/<src>_rt/ with every feature unchanged and
y = PJM RT system energy price (rt_energy_price from PjmFlexDaep20261001$collect_data(), data/pjm_raw.parquet; the
same pjm_energy_prices column as shiny's rt_energy_prices()). Checks first that v0's y equals da_energy_price
from the same pull, so the alignment is right. Run RT walk-forwards with --context-gap-hours 24: D-1's RT prices
are incomplete at the D-1 11:00 EPT cutoff, so the context must end with D-2.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("src", nargs="+")
p.add_argument("--data", default=str(ROOT / "data"))
a = p.parse_args()
DATA = Path(a.data)

raw = pd.read_parquet(DATA / "pjm_raw.parquet")
raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts")

for src in a.src:
    dst = DATA / "pjm-fe" / f"{src}_rt"
    dst.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        df = pd.read_parquet(DATA / "pjm-fe" / src / f"{s}.parquet")
        ts = pd.to_datetime(df["ts"])
        da = raw["da_energy_price"].reindex(ts).to_numpy()
        ok = df["y"].notna().to_numpy() & ~np.isnan(da)
        match = (np.abs(df["y"].to_numpy()[ok] - da[ok]) < 1e-6).mean()
        assert match > 0.999, (src, s, match)
        df["y"] = raw["rt_energy_price"].reindex(ts).to_numpy()
        df.to_parquet(dst / f"{s}.parquet", index=False)
        print(f"{src}_rt/{s}: rows {len(df)}, y==DA check {match:.4f}, RT missing {df['y'].isna().sum()}, "
              f"RT mean {df['y'].mean():.2f} max {df['y'].max():.0f}")
