"""PJM ALL hourly wind / solar datasets for Kumo (target y = pjm_windgen_outputs / pjm_solargen_outputs ALL, MW).

  uv run python research/pjm_renew/make_features.py [wind|solar|all] [--val-start 2023-10-01] [--check]

Inputs: data/pjm_renew/ (research/pjm_renew/collect.R). Scenario / Maxar inputs are the latest vintage issued by
D-1 10:30 EPT; ISO forecasts and Meteologica wind / solar the latest by D-1 11:00 EPT (bid deadline; Meteologica
wind / solar keep one 10:23-10:59 vintage per hour, which is exactly v0's pjm_wind / pjm_solar). Actuals only as lags
>= 48h. Run walk-forwards with --context-gap-hours 24.

  wind/W2   Meteologica ALL + regional + zonal wind forecasts (level, 24h change, share of a capacity proxy;
            ALL 80% band), ISO wind forecast (same transforms + staleness), ECMWF-ENS / GEFS scenario
            average / std / q5 / q95, Meteologica's recent error (D-2 mean, 7 days ending D-2), Maxar wind speed
            at 13 wind-cluster stations + its cube clipped at a rated-speed proxy, calendar, actual lags >= 48h,
            capacity proxy = 30-day max of actuals lagged 48h.
  solar/S2  Meteologica ALL + regional solar (ALL 80% band), ISO solar ALL + BTM_ALL (+ staleness), ECMWF-ENS
            scenario, Meteologica's recent error, sun elevation and clear-sky proxy at a PJM centroid and three
            regional centroids + day length, Maxar sky cover / temperature at 14 solar-cluster stations,
            calendar, actual lags >= 48h, same-hour max over the 30 days ending D-2 (+ Meteologica as a share).
Splits: default train < 2025-01-01 <= val < 2026-01-01 (data/pjm-renew-fe/<target>/<variant>/). With --val-start,
val = val-start .. 2025-12-31 and the variant is written as <variant>_bp (backpublish).
Also writes data/pjm_renew/bench.parquet (actuals + Meteologica / ISO ALL forecasts) for scoring.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spp_load"))
from make_features import daily, daily_lag, lag, pivot, ts_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/pjm_renew"
WIND_ST = {165: "pnt", 172: "rfd", 162: "ord", 126: "laf", 10: "fwa", 175: "sbn", 29: "tol", 80: "day",
           164: "pit", 93: "eri", 21: "crw", 120: "ipt", 14: "avp"}
SOLAR_ST = {20: "ric", 71: "cho", 18: "orf", 114: "iad", 43: "acy", 12: "ttn", 15: "bwi", 102: "ged", 3: "cmh",
            80: "day", 30: "cxy", 5: "ind", 126: "laf", 165: "pnt"}
CENTROIDS = {"pjm": (39.5, -79.0), "west": (40.6, -85.0), "midatl": (40.2, -75.8), "south": (37.6, -77.6)}
RATED_MPH = 25.0  # surface wind speed above which turbines are roughly at rated output (proxy)
IDX = pd.date_range("2017-12-01", "2025-12-31 23:00", freq="h")


def load_fc(name, key, cols, cutoff_hm):
    """Pivot a fair-vintage forecast table; also return hours between the vintage and the cutoff."""
    d = pd.read_parquet(SRC / f"{name}.parquet")
    d[cols] = d[cols].astype(float)
    p = pivot(d, key, cols).reindex(IDX)
    cut = pd.to_datetime(d["dt"]) - pd.Timedelta(days=1) + pd.Timedelta(cutoff_hm)
    d["stale_h"] = (cut - pd.to_datetime(d["forecast_time"])).dt.total_seconds() / 3600
    stale = pivot(d, key, ["stale_h"]).reindex(IDX)
    return p, stale


def utc_offset(idx):
    """EPT wall-clock hour-beginning -> hours to add to get UTC (4 in EDT, 5 in EST)."""
    loc = idx.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    off = pd.Series(loc.map(lambda t: -t.utcoffset().total_seconds() / 3600 if t is not pd.NaT else np.nan), idx)
    return off.ffill().bfill().to_numpy()


def sun(idx, lat, lon):
    """Solar elevation (deg) at mid-hour and a clear-sky GHI proxy (Meinel air-mass model, W/m2)."""
    t = idx + pd.Timedelta(minutes=30)
    utc_h = t.hour.to_numpy() + t.minute.to_numpy() / 60 + utc_offset(idx)
    doy = t.dayofyear.to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (utc_h - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    solar_min = utc_h * 60 + eqt + 4 * lon
    ha = np.radians(solar_min / 4 - 180)
    la = np.radians(lat)
    sin_el = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    el = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
    am = 1 / np.maximum(sin_el, 0.01)
    ghi = np.where(sin_el > 0, 1361 * sin_el * 0.7 ** (am ** 0.678), 0.0)
    daylen = 2 * np.degrees(np.arccos(np.clip(-np.tan(la) * np.tan(decl), -1, 1))) / 15
    return el, ghi, daylen


def actual_lags(P, y, tag):
    for h in (48, 72, 168):
        P[f"{tag}_lag{h}"] = lag(y, h)
    P[f"{tag}_samehour_7d_mean"] = pd.concat([lag(y, 24 * d) for d in range(2, 9)], axis=1).mean(axis=1)
    for how in ("mean", "max"):
        P[f"{tag}_d2_{how}"] = daily_lag(y, 2, how)


def recent_error(P, y, fc, tag):
    """Mean actual minus forecast over D-2, and over the 7 days ending D-2 (both known at the D-1 cutoff)."""
    err = y - fc
    P[f"{tag}_err_d2"] = daily_lag(err, 2, "mean")
    P[f"{tag}_err_7d"] = pd.concat([daily_lag(err, d, "mean") for d in range(2, 9)], axis=1).mean(axis=1)


def calendar(P):
    P["hr"] = IDX.hour + 1
    yd, nd = IDX.dayofyear.to_numpy(), np.where(IDX.is_leap_year, 366, 365)
    P["season_sin"], P["season_cos"] = np.sin(2 * np.pi * (yd - 80.75) / nd), np.cos(2 * np.pi * (yd - 80.75) / nd)


def read_inputs():
    act = {t: pd.read_parquet(SRC / f"{t}_act.parquet") for t in ("wind", "solar")}
    y = {t: act[t].assign(ts=ts_of(act[t])).groupby("ts")["power_mw"].mean().astype(float).reindex(IDX)
         for t in act}
    met_w, met_w_st = load_fc("met_wind", "region", ["power_mw", "lower_80", "upper_80"], "11:00:00")
    met_s, met_s_st = load_fc("met_solar", "region", ["power_mw", "lower_80", "upper_80"], "11:00:00")
    iso_w, iso_w_st = load_fc("iso_wind", "region", ["power_mw"], "11:00:00")
    iso_s, iso_s_st = load_fc("iso_solar", "region", ["power_mw"], "11:00:00")
    iso_l, iso_l_st = load_fc("iso_load", "zone", ["load_mwh"], "11:00:00")
    sc_w, _ = load_fc("scen_wind", "region", ["average", "std_dev", "quantile_5", "quantile_95"], "10:30:00")
    sc_s, _ = load_fc("scen_solar", "region", ["average", "std_dev", "quantile_5", "quantile_95"], "10:30:00")
    mx = pd.read_parquet(SRC / "maxar.parquet")
    wc = ["temp", "wind_speed", "wind_dir", "tscv", "rel_humidity", "prob_of_precip"]
    mx[wc] = mx[wc].astype(float)
    st = pivot(mx, "location_id", wc).reindex(IDX)
    return dict(y=y, met_w=met_w, met_s=met_s, iso_w=iso_w, iso_s=iso_s, iso_l=iso_l, sc_w=sc_w, sc_s=sc_s, st=st,
                stale={"met_wind": met_w_st, "met_solar": met_s_st, "iso_wind": iso_w_st, "iso_solar": iso_s_st,
                       "iso_load": iso_l_st})


def build_wind(I):
    y, met, iso, sc, st = I["y"]["wind"], I["met_w"], I["iso_w"], I["sc_w"], I["st"]
    P = pd.DataFrame(index=IDX)
    cap = lag(y, 48).rolling(24 * 30, min_periods=24 * 7).max()
    P["cap_proxy"] = cap
    m = met["power_mw__Meteologica"]
    regions = [c.split("__")[1] for c in met if c.startswith("power_mw__Meteologica_")]
    P["met_all_fc"], P["met_all_lo80"], P["met_all_hi80"] = m, met["lower_80__Meteologica"], met["upper_80__Meteologica"]
    P["met_all_delta24"], P["met_all_share"] = m - lag(m, 24), m / cap
    for r in regions:
        P[f"met_{r.removeprefix('Meteologica_').lower().replace('-', '')}_fc"] = met[f"power_mw__{r}"]
    i = iso["power_mw__ALL"]
    P["iso_all_fc"], P["iso_all_delta24"], P["iso_all_share"] = i, i - lag(i, 24), i / cap
    P["iso_stale_h"] = I["stale"]["iso_wind"]["stale_h__ALL"]
    P["iso_minus_met"] = i - m
    for src in ("ECMWF-ENS", "GEFS"):
        tag = src.split("-")[0].lower()
        for c, nm in (("average", "avg"), ("std_dev", "std"), ("quantile_5", "q5"), ("quantile_95", "q95")):
            P[f"scen_{tag}_{nm}"] = sc[f"{c}__{src}"]
    recent_error(P, y, m, "met")
    ws = []
    for k, n in WIND_ST.items():
        w = st[f"wind_speed__{k}"]
        P[f"ws_{n}"] = w
        P[f"ws3_{n}"] = w.clip(upper=RATED_MPH) ** 3
        ws.append(w)
    W = pd.concat(ws, axis=1)
    P["ws_mean"], P["ws3_mean"] = W.mean(axis=1), W.clip(upper=RATED_MPH).pow(3).mean(axis=1)
    calendar(P)
    actual_lags(P, y, "wind")
    return P


def build_solar(I):
    y, met, iso, sc, st = I["y"]["solar"], I["met_s"], I["iso_s"], I["sc_s"], I["st"]
    P = pd.DataFrame(index=IDX)
    m = met["power_mw__Meteologica"]
    P["met_all_fc"], P["met_all_lo80"], P["met_all_hi80"] = m, met["lower_80__Meteologica"], met["upper_80__Meteologica"]
    P["met_all_delta24"] = m - lag(m, 24)
    for r, nm in (("Meteologica_West", "west"), ("Meteologica_Mid-Atlantic", "midatl"), ("Meteologica_South", "south")):
        P[f"met_{nm}_fc"] = met[f"power_mw__{r}"]
    i = iso["power_mw__ALL"]
    P["iso_all_fc"], P["iso_all_delta24"], P["iso_btm_fc"] = i, i - lag(i, 24), iso["power_mw__BTM_ALL"]
    P["iso_stale_h"] = I["stale"]["iso_solar"]["stale_h__ALL"]
    P["iso_minus_met"] = i - m
    for c, nm in (("average", "avg"), ("std_dev", "std"), ("quantile_5", "q5"), ("quantile_95", "q95")):
        P[f"scen_ecmwf_{nm}"] = sc[f"{c}__ECMWF-ENS"]
    recent_error(P, y, m, "met")
    for nm, (la, lo) in CENTROIDS.items():
        el, ghi, dl = sun(IDX, la, lo)
        P[f"sun_el_{nm}"], P[f"clearsky_{nm}"] = el, ghi
        if nm == "pjm":
            P["day_length"] = dl
    for k, n in SOLAR_ST.items():
        P[f"tscv_{n}"], P[f"temp_{n}"] = st[f"tscv__{k}"], st[f"temp__{k}"]
    P["tscv_mean"] = pd.concat([st[f"tscv__{k}"] for k in SOLAR_ST], axis=1).mean(axis=1)
    calendar(P)
    actual_lags(P, y, "solar")
    shmax = pd.concat([lag(y, 24 * d) for d in range(2, 32)], axis=1).max(axis=1)
    P["samehour_30d_max"] = shmax
    P["met_share_of_max"] = m / shmax.where(shmax > 50)
    return P


def write(target, name, P, y, val_start):
    dst = ROOT / "data/pjm-renew-fe" / target / name
    dst.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.DataFrame({"ts": IDX, "y": y.to_numpy()}, index=IDX), P], axis=1).reset_index(drop=True)
    df = df[df["ts"] >= "2018-01-01"].replace([np.inf, -np.inf], np.nan)
    df = df.loc[:, df.notna().any()]
    a, b = pd.Timestamp(val_start), pd.Timestamp("2026-01-01")
    df[df["ts"] < a].to_parquet(dst / "train.parquet", index=False)
    v = df[(df["ts"] >= a) & (df["ts"] < b)]
    v.to_parquet(dst / "val.parquet", index=False)
    print(f"{target}/{name}: {df.shape[1] - 2} features, train {(df['ts'] < a).sum()}, val {len(v)}, "
          f"y missing val {v['y'].isna().sum()}, val cols >5% NaN: "
          f"{[c for c in v.columns if v[c].isna().mean() > 0.05]}")


def check(I):
    """Gate 1: Meteologica pulls reproduce v0, actuals match the forecasts' definition, 2025 benchmarks."""
    v0 = pd.concat(pd.read_parquet(ROOT / "data/pjm-fe/v0" / f"{s}.parquet") for s in ("train", "val", "test"))
    v0 = v0.assign(ts=pd.to_datetime(v0["ts"])).drop_duplicates("ts").set_index("ts")
    for t, met, col in (("wind", I["met_w"], "pjm_wind"), ("solar", I["met_s"], "pjm_solar")):
        m = met["power_mw__Meteologica"]
        d = (m.reindex(v0.index) - v0[col])
        dd = d[(d.index >= "2023-10-01") & (d.index < "2026-01-01")]
        print(f"[gate1] {t}: Meteologica pull vs v0.{col}: n {dd.notna().sum()}, exact {(dd.abs() < 1e-6).mean():.4f}, "
              f"|diff| mean {dd.abs().mean():.2f}, v0 NaN where pull has value {(v0[col].isna() & m.reindex(v0.index).notna()).sum()}"
              f" (all years exact {(d.abs() < 1e-6).mean():.4f})")
    for t, met, iso in (("wind", I["met_w"], I["iso_w"]), ("solar", I["met_s"], I["iso_s"])):
        y = I["y"][t]
        for yr in (2023, 2024, 2025):
            s = slice(f"{yr}-01-01", f"{yr}-12-31 23:00")
            yy, mm, ii = y[s], met["power_mw__Meteologica"][s], iso["power_mw__ALL"][s]
            ok = yy.notna() & mm.notna() & ii.notna()
            if t == "solar":
                ok &= (yy > 10) | (mm > 10)
            print(f"[gate1] {t} {yr}: mean actual {yy[ok].mean():.0f} met {mm[ok].mean():.0f} ({100 * (mm[ok].mean() / yy[ok].mean() - 1):+.1f}%) "
                  f"iso {ii[ok].mean():.0f} ({100 * (ii[ok].mean() / yy[ok].mean() - 1):+.1f}%) | MAE met {(yy - mm)[ok].abs().mean():.0f} "
                  f"iso {(yy - ii)[ok].abs().mean():.0f} | nMAE met {(yy - mm)[ok].abs().mean() / yy[ok].mean():.3f} "
                  f"iso {(yy - ii)[ok].abs().mean() / yy[ok].mean():.3f} | bias(act-fc) met {(yy - mm)[ok].mean():+.0f} "
                  f"iso {(yy - ii)[ok].mean():+.0f} | n {ok.sum()}{' (daylight)' if t == 'solar' else ''}")
    # ISO load vs Meteologica load, 2025 (context)
    load = pd.read_parquet(ROOT / "data/pjm_load/load.parquet")
    yl = load[load["zone"] == "ALL"].assign(ts=lambda d: ts_of(d)).groupby("ts")["load_mwh"].mean().reindex(IDX)
    il = I["iso_l"]["load_mwh__RTO COMBINED"]
    ml = v0["pjm_load"].reindex(IDX)
    s = slice("2025-01-01", "2025-12-31 23:00")
    ok = yl[s].notna() & il[s].notna() & ml[s].notna()
    print(f"[gate1] load 2025: MAE met {(yl[s] - ml[s])[ok].abs().mean():.0f} iso {(yl[s] - il[s])[ok].abs().mean():.0f} "
          f"bias met {(yl[s] - ml[s])[ok].mean():+.0f} iso {(yl[s] - il[s])[ok].mean():+.0f} n {ok.sum()}")
    for k, st in I["stale"].items():
        c = st.iloc[:, 0]
        win = c[(c.index >= "2023-10-01")]
        days = pd.Series(win.notna().to_numpy(), win.index).groupby(win.index.floor("D")).any()
        print(f"[gate1] {k}: 2023-10..2025 days with fair vintage {days.mean():.3f} ({days.sum()}/{len(days)}); "
              f"hours before cutoff median {win.median():.2f}, p95 {win.quantile(.95):.2f}; by year "
              + ", ".join(f"{yr}: {c[str(yr)].notna().mean():.3f}" for yr in range(2018, 2026)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", nargs="?", default="all")
    p.add_argument("--val-start", default="2025-01-01")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    I = read_inputs()
    if args.check:
        check(I)
    suffix = "" if args.val_start == "2025-01-01" else "_bp"
    if args.which in ("all", "wind"):
        write("wind", "W2" + suffix, build_wind(I), I["y"]["wind"], args.val_start)
    if args.which in ("all", "solar"):
        write("solar", "S2" + suffix, build_solar(I), I["y"]["solar"], args.val_start)
    bench = pd.DataFrame({"ts": IDX, "wind": I["y"]["wind"].to_numpy(), "solar": I["y"]["solar"].to_numpy(),
                          "met_wind": I["met_w"]["power_mw__Meteologica"].to_numpy(),
                          "met_solar": I["met_s"]["power_mw__Meteologica"].to_numpy(),
                          "iso_wind": I["iso_w"]["power_mw__ALL"].to_numpy(),
                          "iso_solar": I["iso_s"]["power_mw__ALL"].to_numpy(),
                          "iso_load": I["iso_l"]["load_mwh__RTO COMBINED"].to_numpy()})
    bench.to_parquet(SRC / "bench.parquet", index=False)
