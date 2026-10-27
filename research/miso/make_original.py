"""MISO "original" Kumo dataset: the baseline Lucas RF features of MisoFlexDaep20251101 + time features.

  uv run python research/miso/make_original.py [data/miso/miso_raw.parquet] [name] [val_start]

val_start (default 2025-01-01) moves the train / val boundary earlier for backpublishing, e.g. 2023-10-01 for
the PJM cross-market features (research/pjm_xmkt/), so a daily walk-forward forecasts every day from then on.

Mirrors the PJM original (data/pjm-fe/v0): target y = da_energy_price, the RF's 12 regressors, plus
ts (hour-beginning, MISO fixed-EST clock; Kumo expands it to epoch/hour/dow/month/doy), dt, hr and
dt_ordinal. Splits as PJM: train < 2025-01-01 <= val < 2026-01-01 <= test.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/miso/miso_raw.parquet"
name = sys.argv[2] if len(sys.argv) > 2 else "m0_original"
val_start = pd.Timestamp(sys.argv[3]) if len(sys.argv) > 3 else pd.Timestamp("2025-01-01")

RF_FEATURES = [  # lucas_rf_predict formula in miso-flex-daep-20251101.R
    "thermal_stress_all_miso", "sdi_macro", "miso_load", "miso_wind", "miso_solar",
    "blended_gas_price", "illiois_basin", "solarduckcurve_all_miso_lead1_diff",
    "solar_all_lead1_diff", "duck_curve_south_sub_north_sub_central_miso",
    "solar_all_lag1_diff", "miso_feelslike",
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

dst = ROOT / "data/miso-fe" / name
dst.mkdir(parents=True, exist_ok=True)
a, b = val_start, pd.Timestamp("2026-01-01")
for split, m in (("train", df["ts"] < a), ("val", (df["ts"] >= a) & (df["ts"] < b)), ("test", df["ts"] >= b)):
    df[m].to_parquet(dst / f"{split}.parquet", index=False)
    print(f"{split}: {m.sum()} rows, {df.loc[m, 'ts'].min()} .. {df.loc[m, 'ts'].max()}, "
          f"y missing {df.loc[m, 'y'].isna().sum()}")
print("feature NaN share (train):")
print(df[df["ts"] < a][RF_FEATURES].isna().mean().round(3).to_string())
