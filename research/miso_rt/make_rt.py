"""RT-target copy of a MISO dataset: y = rt_energy_price instead of da_energy_price (as research/pjm_rt/make_rt.py).

  uv run python research/miso/make_original.py data/miso/miso_raw.parquet m0_original_bp24 2024-01-01 2026-01-01
  uv run python research/miso_rt/make_rt.py m0_original_bp24      # -> data/miso-fe/m0_original_bp24_rt

Every feature unchanged; y = MISO RT system energy price (miso_energy_prices.rt_energy_price, the column behind
shiny's rt_energy_prices(); data/miso/miso_energy_prices.parquet from research/miso_rt/collect_rt.R). Checks first
that the source y equals da_energy_price from the same table, so the (dt, hr) alignment is right. Run RT
walk-forwards with --context-gap-hours 24: D-1's RT prices are incomplete at the D-1 bid deadline, so the context
must end with D-2.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
src = sys.argv[1]
e = pd.read_parquet(ROOT / "data/miso/miso_energy_prices.parquet")
e["ts"] = pd.to_datetime(e["dt"]) + pd.to_timedelta(e["hr"] - 1, unit="h")
e = e.drop_duplicates("ts").set_index("ts")
dst = ROOT / "data/miso-fe" / f"{src}_rt"
dst.mkdir(parents=True, exist_ok=True)
for s in ("train", "val", "test"):
    df = pd.read_parquet(ROOT / "data/miso-fe" / src / f"{s}.parquet")
    ts = pd.to_datetime(df["ts"])
    da = e["da_energy_price"].reindex(ts).to_numpy()
    ok = df["y"].notna().to_numpy() & ~np.isnan(da)
    match = (np.abs(df["y"].to_numpy()[ok] - da[ok]) < 1e-3).mean()
    assert match > 0.99, (src, s, match)
    df["y"] = e["rt_energy_price"].reindex(ts).to_numpy()
    df.to_parquet(dst / f"{s}.parquet", index=False)
    print(f"{src}_rt/{s}: rows {len(df)}, y==DA check {match:.4f}, RT missing {df['y'].isna().sum()}, "
          f"RT mean {df['y'].mean():.2f} max {df['y'].max():.0f}")
