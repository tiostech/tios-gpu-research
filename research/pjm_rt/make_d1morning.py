"""RT datasets with D-1 morning RT information (idea 1 from the DA-tightness discussion).

  uv run python research/pjm_rt/make_d1morning.py

At the 11:00 EPT cutoff on D-1, D-1's RT for hours beginning 00..08 is known (the 08:00 hour ends at 09:00).
Features for each row on day D, from D-1 hours beginning 00..08 (data/pjm_raw.parquet):
  d1m_rt_mean, d1m_rt_max    D-1 morning RT average / max
  d1m_rt_last                D-1 08:00 RT (the latest known hour)
  d1m_rtda_mean              D-1 morning RT minus DA (D-1's DA is known: it cleared on D-2)
The matching target-side change is --context-gap-hours 15 (context keeps rows before D-1 09:00).
Base: v0_congfc5_xmkt_rt26 (RT target, 5 congestion PCs + MISO / SPP DA forecast medians), split so val = 2025 and
test = 2026 (2026 is filled too, for a later test run):
  v0_x5rt_25       base, val 2025
  v0_x5rt_d1m_25   base + d1m_* features, val 2025
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/pjm-fe"
raw = pd.read_parquet(ROOT / "data/pjm_raw.parquet")
raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts").sort_index()
mo = raw[raw.index.hour < 9]
g = mo.groupby(mo.index.normalize())
f = pd.DataFrame({"d1m_rt_mean": g["rt_energy_price"].mean(), "d1m_rt_max": g["rt_energy_price"].max(),
                  "d1m_rt_last": mo[mo.index.hour == 8]["rt_energy_price"].groupby(mo[mo.index.hour == 8].index.normalize()).last(),
                  "d1m_rtda_mean": (g["rt_energy_price"].mean() - g["da_energy_price"].mean())})
f.index = f.index + pd.Timedelta(days=1)                   # D-1's morning -> features of day D
base = {s: pd.read_parquet(D / "v0_congfc5_xmkt_rt26" / f"{s}.parquet") for s in ("train", "val")}
full_val = base["val"]
split = {"train": base["train"], "val": full_val[pd.to_datetime(full_val["ts"]) < "2026-01-01"],
         "test": full_val[pd.to_datetime(full_val["ts"]) >= "2026-01-01"]}
for name, add in (("v0_x5rt_25", False), ("v0_x5rt_d1m_25", True)):
    out = D / name
    out.mkdir(exist_ok=True)
    for s, df in split.items():
        df = df.reset_index(drop=True)
        if add:
            day = pd.to_datetime(df["ts"]).dt.normalize()
            for c in f.columns:
                df[c] = day.map(f[c]).to_numpy()
        df.to_parquet(out / f"{s}.parquet", index=False)
        print(name, s, len(df), df["ts"].min(), df["ts"].max(),
              "" if not add else "nulls " + ", ".join(f"{c} {int(df[c].isna().sum())}" for c in f.columns))
