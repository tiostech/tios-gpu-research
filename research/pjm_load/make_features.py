"""PJM ALL hourly load datasets for Kumo (target y = pjm_hourly_load zone ALL, MW).

  uv run python research/pjm_load/make_features.py [P0_weather|P2_with_meteo|all] [--val-start 2023-10-01]

Inputs: data/pjm_load/ (research/pjm_load/collect.R). Forecast inputs are the latest vintage issued by
D-1 10:30 EPT; actual loads only as lags >= 48h. Run walk-forwards with --context-gap-hours 24.

  P0_weather    population-weighted feels-like of the 18 cities / weights of blended_feelslike in
                ep-forecast.dafc-20261001.pjm.json (= production's feelslike_weightedavg), its lags,
                rolling and daily stats, CDD/HDD; per-city feels-like; temp / dew point / cloud cover for
                the 8 largest cities; calendar / holidays / daylight; PJM ALL actual load lags >= 48h.
  P2_with_meteo P0 + Meteologica regional load forecasts (MIDATL, SOUTH, WEST and 5 big zones) + the
                Meteologica ALL forecast (level, shape, day-over-day change).
The ISO forecast (pjm_load_forecast) has no fair D-1 vintage (stamped after 10:30 EPT) and is not used.
Splits: default train < 2025-01-01 <= val < 2026-01-01 (data/pjm-load-fe/<variant>/). With --val-start,
val = val-start .. 2025-12-31 and the variant is written as <variant>_bp (backpublish).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spp_load"))
from make_features import daily, daily_lag, holidays, lag, pivot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/pjm_load"
# blended_feelslike weights (ep-forecast.dafc-20261001.pjm.json), keyed by Maxar location_id.
POPWT = {3: 0.0562, 9: 0.07, 10: 0.0545, 14: 0.02, 15: 0.07, 18: 0.0767, 20: 0.0767, 21: 0.0542,
         27: 0.0389, 29: 0.012, 30: 0.02, 38: 0.02, 70: 0.0734, 74: 0.024, 114: 0.1097, 120: 0.02,
         162: 0.12, 164: 0.0838}
CITY = {3: "cmh", 9: "dca", 10: "fwa", 14: "avp", 15: "bwi", 18: "orf", 20: "ric", 21: "crw", 27: "roa",
        29: "tol", 30: "cxy", 38: "abe", 70: "cak", 74: "cle", 114: "iad", 120: "ipt", 162: "ord",
        164: "pit", 11: "ewr", 13: "phl"}
BIG8 = [162, 114, 13, 164, 3, 20, 11, 15]
REGIONS = ["MIDATL", "SOUTH", "WEST", "WEST_CE", "WEST_AEP", "SOUTH_DOM", "WEST_ATSI", "MIDATL_PS"]
DC_LAT, DC_LON = 38.9, -77.0


def daylight(idx):
    """Approximate Washington DC sunrise/sunset (NOAA formula, local standard time UTC-5)."""
    doy = idx.floor("D").dayofyear.to_numpy()
    g = 2 * np.pi / 365 * (doy - 1)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    lat = np.radians(DC_LAT)
    ha = np.degrees(np.arccos(np.cos(np.radians(90.833)) / (np.cos(lat) * np.cos(decl))
                              - np.tan(lat) * np.tan(decl)))
    noon = 720 - 4 * DC_LON - eqt - 300  # minutes after local standard midnight (UTC-5)
    dst = idx.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").map(
        lambda t: bool(t.dst()) if t is not pd.NaT else False).to_numpy(bool)
    rise, sset = (noon - 4 * ha) / 60 + dst, (noon + 4 * ha) / 60 + dst
    h = idx.hour.to_numpy()
    return pd.DataFrame({"day_duration": sset - rise, "is_day": (((h + 1) > rise) & (h < sset)).astype(int)},
                        index=idx)


def build():
    load = pivot(pd.read_parquet(SRC / "load.parquet"), "zone", ["load_mwh"])
    met = pivot(pd.read_parquet(SRC / "meteologica.parquet"), "region", ["power_mw"])
    mx = pd.read_parquet(SRC / "maxar.parquet")
    wcols = ["temp", "dew_point", "feelslike", "wind_speed", "tscv"]
    mx[wcols] = mx[wcols].astype(float)
    st = pivot(mx, "location_id", wcols)

    idx = pd.date_range("2017-12-01", "2025-12-31 23:00", freq="h")
    y = load["load_mwh__ALL"].astype(float).reindex(idx)
    col = lambda c, k: st[f"{c}__{k}"].reindex(idx) if f"{c}__{k}" in st else pd.Series(np.nan, idx)  # noqa: E731

    P0 = pd.DataFrame(index=idx)
    num = sum(w * col("feelslike", k).fillna(0) for k, w in POPWT.items())
    den = sum(w * col("feelslike", k).notna() for k, w in POPWT.items())
    fl = num / den.replace(0, np.nan)
    P0["feelslike_popwt"] = fl
    for h in (24, 48, 72):
        P0[f"fl_popwt_lag{h}"] = lag(fl, h)
    P0["fl_popwt_roll24"], P0["fl_popwt_roll72"] = fl.rolling(24).mean(), fl.rolling(72).mean()
    for how in ("mean", "std", "max", "min"):
        P0[f"fl_popwt_day_{how}"] = daily(fl, how)
    P0["fl_popwt_day_lag1"], P0["fl_popwt_day_lag2"] = daily_lag(fl, 1, "mean"), daily_lag(fl, 2, "mean")
    P0["fl_popwt_delta24"] = fl - lag(fl, 24)
    P0["cdd"], P0["hdd"] = (fl - 65).clip(lower=0), (65 - fl).clip(lower=0)
    P0["cdd_day"], P0["hdd_day"] = daily(P0["cdd"], "mean"), daily(P0["hdd"], "mean")
    for k in POPWT:
        P0[f"fl_{CITY[k]}"] = col("feelslike", k)
    for k in BIG8:
        n = CITY[k]
        P0[f"temp_{n}"], P0[f"dew_{n}"], P0[f"tscv_{n}"] = col("temp", k), col("dew_point", k), col("tscv", k)
    P0["wind_mean"] = sum(col("wind_speed", k) for k in BIG8) / len(BIG8)
    P0["hr"] = idx.hour + 1
    P0["dow"] = idx.dayofweek
    yd, nd = idx.dayofyear.to_numpy(), np.where(idx.is_leap_year, 366, 365)
    P0["season_sin"], P0["season_cos"] = np.sin(2 * np.pi * (yd - 80.75) / nd), np.cos(2 * np.pi * (yd - 80.75) / nd)
    day = idx.floor("D")
    hol = holidays(range(2017, 2027))
    P0["is_holiday"] = day.isin(hol).astype(int)
    P0["is_holiday_adjacent"] = (day.isin(hol - pd.Timedelta(days=1)) | day.isin(hol + pd.Timedelta(days=1))).astype(int)
    thanks = pd.DatetimeIndex([d for d in hol if d.month == 11])
    P0["is_thanksgiving_week"] = np.isin(day, np.concatenate([thanks + pd.Timedelta(days=k) for k in range(4)])).astype(int)
    P0["is_xmas_newyear_week"] = (((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 1))).astype(int)
    P0["covid_start"] = ((day > "2020-03-15") & (day < "2020-05-31")).astype(int)
    P0 = P0.join(daylight(idx))
    for h in (48, 72, 168):
        P0[f"all_lag{h}"] = lag(y, h)
    P0["all_samehour_7d_mean"] = pd.concat([lag(y, 24 * d) for d in range(2, 9)], axis=1).mean(axis=1)
    for how in ("mean", "max", "min"):
        P0[f"all_d2_{how}"] = daily_lag(y, 2, how)

    P2 = pd.DataFrame(index=idx)
    for r in REGIONS:
        P2[f"met_{r.lower()}_fc"] = met[f"power_mw__{r}"].reindex(idx)
    m = met["power_mw__ALL"].reindex(idx)
    P2["met_all_fc"] = m
    P2["met_all_fc_shape"] = m / daily(m, "mean")
    P2["met_all_fc_delta24"] = m - lag(m, 24)

    base = pd.DataFrame({"ts": idx, "y": y.to_numpy()}, index=idx)
    variants = {"P0_weather": [P0], "P2_with_meteo": [P0, P2]}
    return base, variants, m, fl


def write(name, df, val_start):
    dst = ROOT / "data/pjm-load-fe" / name
    dst.mkdir(parents=True, exist_ok=True)
    df = df[df["ts"] >= "2018-01-01"].replace([np.inf, -np.inf], np.nan)
    df = df.loc[:, df.notna().any()]
    a, b = pd.Timestamp(val_start), pd.Timestamp("2026-01-01")
    df[df["ts"] < a].to_parquet(dst / "train.parquet", index=False)
    v = df[(df["ts"] >= a) & (df["ts"] < b)]
    v.to_parquet(dst / "val.parquet", index=False)
    print(f"{name}: {df.shape[1] - 2} features, train {(df['ts'] < a).sum()}, val {len(v)}, "
          f"y missing val {v['y'].isna().sum()}, val cols >5% NaN: "
          f"{[c for c in v.columns if v[c].isna().mean() > 0.05]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", nargs="?", default="all")
    p.add_argument("--val-start", default="2025-01-01")
    args = p.parse_args()
    base, variants, met, fl = build()
    suffix = "" if args.val_start == "2025-01-01" else "_bp"
    for name, parts in variants.items():
        if args.which in ("all", name):
            write(name + suffix, pd.concat([base] + parts, axis=1).reset_index(drop=True), args.val_start)
    pd.DataFrame({"ts": met.index, "meteologica": met.to_numpy()}).to_parquet(SRC / "meteologica_all.parquet", index=False)
    # check: config-weighted feels-like reproduces production's feelslike_weightedavg (data/pjm-fe/v0)
    v0 = pd.concat(pd.read_parquet(ROOT / "data/pjm-fe/v0" / f"{s}.parquet") for s in ("train", "val"))
    v0 = v0.assign(ts=pd.to_datetime(v0["ts"])).set_index("ts")["feelslike_weightedavg"]
    d = (fl.reindex(v0.index) - v0).dropna()
    print(f"feelslike_popwt vs v0 feelslike_weightedavg: n {len(d)}, |diff| median {d.abs().median():.3f}, "
          f"p99 {d.abs().quantile(.99):.3f}; 2025 corr {np.corrcoef(fl.reindex(v0.index)['2025'].fillna(0), v0['2025'].fillna(0))[0, 1]:.4f}")
