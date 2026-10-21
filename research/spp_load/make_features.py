"""OKGE hourly load datasets for Kumo (target y = spp_hourly_load zone OKGE, MW).

  uv run python research/spp_load/make_features.py [L0_branch|L1_extra|L2_with_meteo|all]

Inputs: data/spp_load/*.parquet (research/spp_load/collect.R). Every forecast input is the single
D-1 ~09:00 vintage stored in its table; actual loads are used only with lags >= 48h (at the D-1
morning cutoff, D-1 load is only partly known).

  L0_branch     features of SppOkgeLoadWnorm20260716 (spp-load-fc-research branch): station-weighted
                Maxar temp / rel_humidity / wind_speed / tscv, their 24-72h lags and rolling means,
                within-day sd / daily means and 1-2 day lags, SPP ISO load forecast (ALL) and its
                intraday shape, SPP ALL actual load lag 48/72 and D-2/D-3 means, calendar (dow,
                season, holidays, Thanksgiving / Christmas weeks, COVID), OKC daylight.
  L1_extra      L0 + OKGE own load history (lag 48/72/168, D-2 mean/max/min, same-hour 7d mean),
                per-station feels-like / dew point / heat index / wind chill / precip for the two
                main stations, CDD/HDD, Meteologica SPP-ALL forecast, ISO forecast day-over-day change.
  L2_with_meteo L1 + the Meteologica OKGE forecast (power_mw) and its intraday shape as features
                (its 80% band is empty in 2025, so not used).
Splits: train < 2025-01-01 <= val < 2026-01-01 (walk-forward over val).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/spp_load"
WEIGHTS = {20: 0.814769752, 13: 0.062065320, 203: 0.120202027, 205: 0.002962901}  # branch weights
OKC_LAT, OKC_LON = 35.4676, -97.5164


def ts_of(df):
    return pd.to_datetime(df["dt"]) + pd.to_timedelta(df["hr"].astype(int) - 1, unit="h")


def pivot(df, key, cols):
    df = df.assign(ts=ts_of(df))
    p = df.pivot_table(index="ts", columns=key, values=cols, aggfunc="mean")
    p.columns = [f"{c}__{k}" for c, k in p.columns]
    return p


def lag(s, h):
    return s.shift(h)


def daily(s, how):
    return s.groupby(s.index.floor("D")).transform(how)


def daily_lag(s, d, how):
    day = s.groupby(s.index.floor("D")).agg(how)
    return pd.Series(day.shift(d).reindex(s.index.floor("D")).to_numpy(), index=s.index)


def holidays(years):
    from pandas.tseries.holiday import (USFederalHolidayCalendar, USLaborDay, USMartinLutherKingJr,
                                        USMemorialDay, USPresidentsDay, USThanksgivingDay)
    out = []
    for y in years:
        a, b = f"{y}-01-01", f"{y}-12-31"
        out += [pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-07-04"), pd.Timestamp(f"{y}-12-25")]
        for rule in (USMartinLutherKingJr, USPresidentsDay, USMemorialDay, USLaborDay, USThanksgivingDay):
            out += list(rule.dates(a, b))
    return pd.DatetimeIndex(out)


def daylight(idx):
    """Approximate OKC sunrise/sunset (NOAA formula, local standard time UTC-6)."""
    day = idx.floor("D")
    doy = day.dayofyear.to_numpy()
    g = 2 * np.pi / 365 * (doy - 1)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    lat = np.radians(OKC_LAT)
    ha = np.degrees(np.arccos(np.cos(np.radians(90.833)) / (np.cos(lat) * np.cos(decl))
                              - np.tan(lat) * np.tan(decl)))
    noon = 720 - 4 * OKC_LON - eqt - 360  # minutes after local standard midnight (UTC-6)
    dst = idx.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT").map(
        lambda t: bool(t.dst()) if t is not pd.NaT else False).to_numpy(bool)
    rise, sset = (noon - 4 * ha) / 60 + dst, (noon + 4 * ha) / 60 + dst
    h = idx.hour.to_numpy()
    is_day = ((h + 1) > rise) & (h < sset)
    return pd.DataFrame({"day_duration": (sset - rise), "is_day": is_day.astype(int)}, index=idx)


def build():
    load = pivot(pd.read_parquet(SRC / "load.parquet"), "zone", ["load_mwh"])
    met = pivot(pd.read_parquet(SRC / "meteologica.parquet"), "region", ["power_mw", "lower_80", "upper_80"])
    iso = pd.read_parquet(SRC / "iso_fc.parquet").assign(ts=lambda d: ts_of(d)).groupby("ts")["iso_load_fc"].mean()
    mx = pd.read_parquet(SRC / "maxar.parquet")
    wcols = ["temp", "dew_point", "rel_humidity", "wchl", "heat_index", "wind_speed", "tscv", "precip",
             "prob_of_precip", "feelslike"]
    mx[wcols] = mx[wcols].astype(float)
    st = pivot(mx, "location_id", wcols)

    idx = pd.date_range("2017-12-01", "2025-12-31 23:00", freq="h")
    f = pd.DataFrame(index=idx)
    f["y"] = load["load_mwh__OKGE"].reindex(idx)

    # Station-weighted weather (weights renormalized over stations present each hour).
    for c in ("temp", "rel_humidity", "wind_speed", "tscv", "feelslike"):
        num = sum(w * st[f"{c}__{k}"].reindex(idx).fillna(0) for k, w in WEIGHTS.items())
        den = sum(w * st[f"{c}__{k}"].reindex(idx).notna() for k, w in WEIGHTS.items())
        f[f"w_{c}"] = num / den.replace(0, np.nan)
    t, rh = f["w_temp"], f["w_rel_humidity"]

    L0 = pd.DataFrame(index=idx)
    for c in ("w_temp", "w_rel_humidity", "w_wind_speed", "w_tscv"):
        L0[c] = f[c]
    for h in (24, 48, 72):
        L0[f"temp{h}"], L0[f"rel_humidity{h}"] = lag(t, h), lag(rh, h)
    L0["temp_roll24"], L0["rh_roll24"] = t.rolling(24).mean(), rh.rolling(24).mean()
    L0["temp_roll72"], L0["rh_roll72"] = t.rolling(72).mean(), rh.rolling(72).mean()
    for c, s in (("temp", t), ("rh", rh), ("wind", f["w_wind_speed"]), ("tscv", f["w_tscv"])):
        L0[f"{c}_day_mean"], L0[f"{c}_day_sd"] = daily(s, "mean"), daily(s, "std")
    L0["temp_day_lag1"], L0["temp_day_lag2"] = daily_lag(t, 1, "mean"), daily_lag(t, 2, "mean")
    L0["rh_day_lag1"], L0["rh_day_lag2"] = daily_lag(rh, 1, "mean"), daily_lag(rh, 2, "mean")
    L0["iso_load_fc"] = iso.reindex(idx)
    L0["iso_load_fc_shape"] = L0["iso_load_fc"] / daily(L0["iso_load_fc"], "mean")
    all_act = load["load_mwh__ALL"].reindex(idx)
    L0["all_load_lag48"], L0["all_load_lag72"] = lag(all_act, 48), lag(all_act, 72)
    L0["all_load_d2_mean"], L0["all_load_d3_mean"] = daily_lag(all_act, 2, "mean"), daily_lag(all_act, 3, "mean")
    day = idx.floor("D")
    hol = holidays(range(2017, 2027))
    L0["hr"] = idx.hour + 1
    L0["dow"] = idx.dayofweek
    yd = idx.dayofyear.to_numpy()
    nd = np.where(idx.is_leap_year, 366, 365)
    L0["season_sin"], L0["season_cos"] = np.sin(2 * np.pi * (yd - 80.75) / nd), np.cos(2 * np.pi * (yd - 80.75) / nd)
    L0["is_holiday"] = day.isin(hol).astype(int)
    L0["is_holiday_adjacent"] = (day.isin(hol - pd.Timedelta(days=1)) | day.isin(hol + pd.Timedelta(days=1))).astype(int)
    thanks = pd.DatetimeIndex([d for d in hol if d.month == 11])
    L0["is_thanksgiving_week"] = np.isin(day, np.concatenate([thanks + pd.Timedelta(days=k) for k in range(4)])).astype(int)
    L0["is_xmas_newyear_week"] = (((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 1))).astype(int)
    L0["covid_start"] = ((day > "2020-03-15") & (day < "2020-05-31")).astype(int)
    L0 = L0.join(daylight(idx))

    L1 = pd.DataFrame(index=idx)
    y = f["y"]
    for h in (48, 72, 168):
        L1[f"okge_lag{h}"] = lag(y, h)
    L1["okge_samehour_7d_mean"] = pd.concat([lag(y, 24 * d) for d in range(2, 9)], axis=1).mean(axis=1)
    for how in ("mean", "max", "min"):
        L1[f"okge_d2_{how}"] = daily_lag(y, 2, how)
    for k in (20, 203):
        for c in ("feelslike", "dew_point", "heat_index", "wchl", "wind_speed", "tscv", "precip", "prob_of_precip"):
            L1[f"{c}_{k}"] = st[f"{c}__{k}"].reindex(idx)
    fl = f["w_feelslike"]
    L1["w_feelslike"] = fl
    L1["cdd"], L1["hdd"] = (fl - 65).clip(lower=0), (65 - fl).clip(lower=0)
    L1["cdd_day"], L1["hdd_day"] = daily(L1["cdd"], "mean"), daily(L1["hdd"], "mean")
    L1["feelslike_delta24"] = fl - lag(fl, 24)
    L1["met_all_fc"] = met["power_mw__ALL"].reindex(idx)
    L1["iso_fc_delta24"] = L0["iso_load_fc"] - lag(L0["iso_load_fc"], 24)
    L1["okge_share_d2"] = daily_lag(y, 2, "mean") / daily_lag(all_act, 2, "mean")

    L2 = pd.DataFrame(index=idx)
    L2["met_okge_fc"] = met["power_mw__OKGE"].reindex(idx)
    L2["met_okge_fc_shape"] = L2["met_okge_fc"] / daily(L2["met_okge_fc"], "mean")

    base = pd.DataFrame({"ts": idx, "y": f["y"].to_numpy()}, index=idx)
    variants = {"L0_branch": [L0], "L1_extra": [L0, L1], "L2_with_meteo": [L0, L1, L2]}
    return base, variants, met["power_mw__OKGE"].reindex(idx)


def write(name, df):
    dst = ROOT / "data/spp-load-fe" / name
    dst.mkdir(parents=True, exist_ok=True)
    df = df[df["ts"] >= "2018-01-01"].replace([np.inf, -np.inf], np.nan)
    df = df.loc[:, df.notna().any()]
    a, b = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
    df[df["ts"] < a].to_parquet(dst / "train.parquet", index=False)
    df[(df["ts"] >= a) & (df["ts"] < b)].to_parquet(dst / "val.parquet", index=False)
    print(f"{name}: {df.shape[1] - 2} features, train {(df['ts'] < a).sum()}, val {((df['ts'] >= a) & (df['ts'] < b)).sum()}, "
          f"y missing val {df.loc[df['ts'] >= a, 'y'].isna().sum()}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    base, variants, met = build()
    for name, parts in variants.items():
        if which in ("all", name):
            write(name, pd.concat([base] + parts, axis=1).reset_index(drop=True))
    out = ROOT / "data/spp_load/meteologica_okge.parquet"
    pd.DataFrame({"ts": met.index, "meteologica": met.to_numpy()}).to_parquet(out, index=False)
