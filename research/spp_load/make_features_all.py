"""SPP ALL hourly load datasets for Kumo (target y = spp_hourly_load zone ALL, MW).

  uv run python research/spp_load/make_features_all.py [A0_weather|A1_iso|A2_with_meteo|all]

Inputs: data/spp_load/ (collect.R + collect_all.R). Forecast inputs are single D-1 ~09:00-09:22
vintages; actual loads only as lags >= 48h. Run walk-forwards with --context-gap-hours 24.

  A0_weather    population-weighted feels-like of the 26 cities / weights in
                ep-forecast.daep-20260701.spp.json (feelslike_spp_popweighted), its lags, rolling and
                daily stats, CDD/HDD; per-city feels-like for the 12 SPP-footprint cities; temp /
                dew point / cloud cover for the 6 largest SPP cities; OKC/Wichita wind speed (config's
                windspeed_okc_wichita); calendar / holidays / daylight; SPP ALL actual load lags >= 48h.
  A1_iso        A0 + SPP ISO load forecast (ALL; level, shape, day-over-day change) + Meteologica SPS.
  A2_with_meteo A1 + Meteologica ALL forecast (level, shape). (ALL equals the sum of its zonal forecasts
                exactly in 2025, so the zone sum is not added.)
Splits: train < 2025-01-01 <= val < 2026-01-01.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from make_features import daily, daily_lag, daylight, holidays, lag, pivot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/spp_load"
# feelslike_spp_popweighted weights (ep-forecast.daep-20260701.spp.json), keyed by Maxar location_id.
POPWT = {21: 1, 20: 1, 142: 0.9, 27: 0.9, 28: 0.8, 25: 0.8, 6: 0.7, 2: 0.7, 208: 0.6, 26: 0.6, 7: 0.6,
         209: 0.5, 152: 0.5, 89: 0.5, 189: 0.4, 135: 0.4, 159: 0.4, 60: 0.3, 153: 0.3, 59: 0.3,
         177: 0.2, 122: 0.2, 146: 0.2, 236: 0.1, 32: 0.1, 161: 0.1}
SPP_CITIES = {21: "omaha", 20: "okc", 142: "kc", 27: "tulsa", 28: "wichita", 25: "sioux_falls", 6: "fargo",
              2: "amarillo", 208: "topeka", 26: "springfield", 7: "fayetteville", 209: "roswell"}
BIG6 = [21, 20, 142, 27, 28, 2]


def build():
    load = pivot(pd.read_parquet(SRC / "load.parquet"), "zone", ["load_mwh"])
    met = pivot(pd.read_parquet(SRC / "meteologica.parquet"), "region", ["power_mw"])
    mz = pivot(pd.read_parquet(SRC / "meteologica_zones.parquet"), "region", ["power_mw"])
    iso = pd.read_parquet(SRC / "iso_fc.parquet").assign(
        ts=lambda d: pd.to_datetime(d["dt"]) + pd.to_timedelta(d["hr"].astype(int) - 1, unit="h")
    ).groupby("ts")["iso_load_fc"].mean()
    mx = pd.read_parquet(SRC / "maxar_all.parquet")
    wcols = ["temp", "dew_point", "rel_humidity", "wind_speed", "tscv", "feelslike"]
    mx[wcols] = mx[wcols].astype(float)
    st = pivot(mx, "location_id", wcols)

    idx = pd.date_range("2017-12-01", "2025-12-31 23:00", freq="h")
    y = load["load_mwh__ALL"].reindex(idx)
    col = lambda c, k: st[f"{c}__{k}"].reindex(idx) if f"{c}__{k}" in st else pd.Series(np.nan, idx)

    A0 = pd.DataFrame(index=idx)
    num = sum(w * col("feelslike", k).fillna(0) for k, w in POPWT.items())
    den = sum(w * col("feelslike", k).notna() for k, w in POPWT.items())
    fl = num / den.replace(0, np.nan)
    A0["feelslike_spp_popwt"] = fl
    for h in (24, 48, 72):
        A0[f"fl_popwt_lag{h}"] = lag(fl, h)
    A0["fl_popwt_roll24"], A0["fl_popwt_roll72"] = fl.rolling(24).mean(), fl.rolling(72).mean()
    for how in ("mean", "std", "max", "min"):
        A0[f"fl_popwt_day_{how}"] = daily(fl, how)
    A0["fl_popwt_day_lag1"], A0["fl_popwt_day_lag2"] = daily_lag(fl, 1, "mean"), daily_lag(fl, 2, "mean")
    A0["fl_popwt_delta24"] = fl - lag(fl, 24)
    A0["cdd"], A0["hdd"] = (fl - 65).clip(lower=0), (65 - fl).clip(lower=0)
    A0["cdd_day"], A0["hdd_day"] = daily(A0["cdd"], "mean"), daily(A0["hdd"], "mean")
    for k, name in SPP_CITIES.items():
        A0[f"fl_{name}"] = col("feelslike", k)
    for k in BIG6:
        n = SPP_CITIES[k]
        A0[f"temp_{n}"], A0[f"dew_{n}"], A0[f"tscv_{n}"] = col("temp", k), col("dew_point", k), col("tscv", k)
    A0["windspeed_okc_wichita"] = (col("wind_speed", 20) + col("wind_speed", 28)) / 2
    A0["hr"] = idx.hour + 1
    A0["dow"] = idx.dayofweek
    yd, nd = idx.dayofyear.to_numpy(), np.where(idx.is_leap_year, 366, 365)
    A0["season_sin"], A0["season_cos"] = np.sin(2 * np.pi * (yd - 80.75) / nd), np.cos(2 * np.pi * (yd - 80.75) / nd)
    day = idx.floor("D")
    hol = holidays(range(2017, 2027))
    A0["is_holiday"] = day.isin(hol).astype(int)
    A0["is_holiday_adjacent"] = (day.isin(hol - pd.Timedelta(days=1)) | day.isin(hol + pd.Timedelta(days=1))).astype(int)
    thanks = pd.DatetimeIndex([d for d in hol if d.month == 11])
    A0["is_thanksgiving_week"] = np.isin(day, np.concatenate([thanks + pd.Timedelta(days=k) for k in range(4)])).astype(int)
    A0["is_xmas_newyear_week"] = (((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 1))).astype(int)
    A0["covid_start"] = ((day > "2020-03-15") & (day < "2020-05-31")).astype(int)
    A0 = A0.join(daylight(idx))
    for h in (48, 72, 168):
        A0[f"all_lag{h}"] = lag(y, h)
    A0["all_samehour_7d_mean"] = pd.concat([lag(y, 24 * d) for d in range(2, 9)], axis=1).mean(axis=1)
    for how in ("mean", "max", "min"):
        A0[f"all_d2_{how}"] = daily_lag(y, 2, how)

    A1 = pd.DataFrame(index=idx)
    A1["iso_load_fc"] = iso.reindex(idx)
    A1["iso_load_fc_shape"] = A1["iso_load_fc"] / daily(A1["iso_load_fc"], "mean")
    A1["iso_fc_delta24"] = A1["iso_load_fc"] - lag(A1["iso_load_fc"], 24)
    A1["met_sps_fc"] = mz["power_mw__SPS"].reindex(idx)

    A2 = pd.DataFrame(index=idx)
    A2["met_all_fc"] = met["power_mw__ALL"].reindex(idx)
    A2["met_all_fc_shape"] = A2["met_all_fc"] / daily(A2["met_all_fc"], "mean")

    base = pd.DataFrame({"ts": idx, "y": y.to_numpy()}, index=idx)
    variants = {"A0_weather": [A0], "A1_iso": [A0, A1], "A2_with_meteo": [A0, A1, A2]}
    return base, variants, met["power_mw__ALL"].reindex(idx)


def write(name, df):
    dst = ROOT / "data/spp-load-fe" / name
    dst.mkdir(parents=True, exist_ok=True)
    df = df[df["ts"] >= "2018-01-01"].replace([np.inf, -np.inf], np.nan)
    df = df.loc[:, df.notna().any()]
    a, b = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
    df[df["ts"] < a].to_parquet(dst / "train.parquet", index=False)
    df[(df["ts"] >= a) & (df["ts"] < b)].to_parquet(dst / "val.parquet", index=False)
    v = df[(df["ts"] >= a) & (df["ts"] < b)]
    print(f"{name}: {df.shape[1] - 2} features, train {(df['ts'] < a).sum()}, val {len(v)}, "
          f"y missing val {v['y'].isna().sum()}, val cols >5% NaN: "
          f"{[c for c in v.columns if v[c].isna().mean() > 0.05]}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    base, variants, met = build()
    for name, parts in variants.items():
        if which in ("all", name):
            write(name, pd.concat([base] + parts, axis=1).reset_index(drop=True))
    pd.DataFrame({"ts": met.index, "meteologica": met.to_numpy()}).to_parquet(
        SRC / "meteologica_all.parquet", index=False)
