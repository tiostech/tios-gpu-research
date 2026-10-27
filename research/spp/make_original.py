"""SPP "original" Kumo dataset: the baseline rf730d features of SppFlexDaep20260701 + time features.

  uv run python research/spp/make_original.py [data/spp/spp_raw.parquet] [name] [val_start] [val_end]

val_start (default 2025-01-01) moves the train / val boundary earlier for backpublishing, e.g. 2023-10-01 for
the PJM cross-market features (research/pjm_xmkt/), so a daily walk-forward forecasts every day from then on.
val_end (default 2026-01-01) moves the val / test boundary: 2026-01-01 2027-01-01 forecasts 2026 (the test year;
only for the 2026 production comparison the user asked for, 2026-10-27).

Mirrors the PJM original (data/pjm-fe/v0): target y = da_energy_price, the RF's 19 regressors (incl. its season/hour time features), plus
ts (hour-beginning, SPP market-local clock; Kumo expands it to epoch/hour/dow/month/doy), dt, hr and
dt_ordinal. Splits as PJM: train < 2025-01-01 <= val < 2026-01-01 <= test.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/spp/spp_raw.parquet"
name = sys.argv[2] if len(sys.argv) > 2 else "s0_original"
val_start = pd.Timestamp(sys.argv[3]) if len(sys.argv) > 3 else pd.Timestamp("2025-01-01")
val_end = pd.Timestamp(sys.argv[4]) if len(sys.argv) > 4 else pd.Timestamp("2026-01-01")

RF_FEATURES = [  # rf730d_predict formula in spp-flex-daep-20260701.R; its days_from_target_dt
    # (target-relative age of a row) is covered by dt_ordinal / ts epoch, which Kumo sees directly.
    "spp_load", "spp_wind", "spp_solar", "netload_all_spp", "netload_deriv", "wind_deriv",
    "solar_deriv", "netload_north_miso", "netload_sps_z3", "spp_gen_on_outage", "spp_nuclear_goo",
    "feelslike_spp_popwt", "windspeed_okc_wichita", "spp_pct_of_dispatch", "spp_gas",
    "season_sin", "season_cos", "hr_numeric", "utc_hr",
]

raw = pd.read_parquet(src)
df = pd.DataFrame({
    "ts": pd.to_datetime(raw["dt"]) + pd.to_timedelta(raw["hr"] - 1, unit="h"),
    "dt": pd.to_datetime(raw["dt"]).dt.date,
    "hr": raw["hr"].astype(float),
    "y": raw["da_energy_price"],
})
df[RF_FEATURES] = raw[RF_FEATURES]
df["dt_ordinal"] = pd.to_datetime(df["dt"]).map(pd.Timestamp.toordinal).astype(float)
df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
df = df[df["ts"] >= "2017-12-31"]

dst = ROOT / "data/spp-fe" / name
dst.mkdir(parents=True, exist_ok=True)
a, b = val_start, val_end
for split, m in (("train", df["ts"] < a), ("val", (df["ts"] >= a) & (df["ts"] < b)), ("test", df["ts"] >= b)):
    df[m].to_parquet(dst / f"{split}.parquet", index=False)
    print(f"{split}: {m.sum()} rows, {df.loc[m, 'ts'].min()} .. {df.loc[m, 'ts'].max()}, "
          f"y missing {df.loc[m, 'y'].isna().sum()}")
print("feature NaN share (train):")
print(df[df["ts"] < a][RF_FEATURES].isna().mean().round(3).to_string())
