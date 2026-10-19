"""Build PJM feature variants for Kumo-Tabular walk-forward experiments.

  uv run python research/pjm/make_features.py <variant> [--src DIR] [--out DIR]

Reads the original train/val/test parquet (src), concatenates them into one
hourly series so lags cross split boundaries, adds the variant's features and
writes <out>/<variant>/{train,val,test}.parquet with the same split rows.

Leakage rules (forecast for day D is made the morning of D-1):
  * y (DA price) of D-1 is known -> lags >= 24h are fair.
  * Feature columns are fair forecasts per row, so same-row values are fair.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load(src: Path) -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(src / f"{split}.parquet")
        df["_split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").set_index("ts")
    full = pd.date_range(df.index[0], df.index[-1], freq="h")
    assert df.index.is_unique and len(full) == len(df), "expected a gapless hourly grid"
    return df


def lag(s: pd.Series, hours: int) -> pd.Series:
    return s.shift(hours, freq="h").reindex(s.index)


def daily_lag(s: pd.Series, days: int, how: str) -> pd.Series:
    """Daily aggregate of s, `days` days back, broadcast to every hour of the day."""
    day = s.index.floor("D")
    agg = s.groupby(day).agg(how)
    agg.index = agg.index + pd.Timedelta(days=days)
    return pd.Series(agg.reindex(day).to_numpy(), index=s.index)


def add_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y"]
    for h in (24, 48, 72, 168):
        df[f"y_lag{h}"] = lag(y, h)
    df["y_samehour_7d_mean"] = pd.concat(
        [lag(y, 24 * d) for d in range(1, 8)], axis=1
    ).mean(axis=1)
    for how in ("mean", "max", "min"):
        df[f"y_d1_{how}"] = daily_lag(y, 1, how)
    df["y_d1_shape"] = df["y_lag24"] / df["y_d1_mean"]
    return df


def add_fundamental_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df["net_load"] = df["pjm_load"] - df["pjm_wind"].fillna(0) - df["pjm_solar"].fillna(0)
    df["net_load_delta24"] = df["net_load"] - lag(df["net_load"], 24)
    df["thermal_stress_delta24"] = df["thermal_stress_all_pjm"] - lag(
        df["thermal_stress_all_pjm"], 24
    )
    gas_lag = lag(df["smoothed_gas"], 24)
    df["heat_rate_lag24"] = df["y_lag24"] / gas_lag
    # Anchor-and-scale, like the production scaled method: yesterday's price
    # scaled by the change in gas.
    df["y_lag24_gas_scaled"] = df["y_lag24"] * df["smoothed_gas"] / gas_lag
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """NERC holidays (observed) and weekend flags."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    cal = USFederalHolidayCalendar()
    nerc = {"New Year's Day", "Memorial Day", "Independence Day", "Labor Day",
            "Thanksgiving Day", "Christmas Day"}
    days = df.index.floor("D")
    rules = [r for r in cal.rules if r.name in nerc]
    holidays = pd.DatetimeIndex([])
    for r in rules:
        holidays = holidays.union(r.dates(days.min(), days.max()))
    df["is_holiday"] = days.isin(holidays).astype(float)
    df["is_weekend_or_holiday"] = ((days.dayofweek >= 5) | days.isin(holidays)).astype(float)
    return df


def add_price_history(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y"]
    df["y_d2_mean"] = daily_lag(y, 2, "mean")
    df["y_d1_std"] = daily_lag(y, 1, "std")
    df["y_7d_mean"] = pd.concat([daily_lag(y, d, "mean") for d in range(1, 8)], axis=1).mean(axis=1)
    df["y_samehour_7d_max"] = pd.concat([lag(y, 24 * d) for d in range(1, 8)], axis=1).max(axis=1)
    df["y_samehour_7d_min"] = pd.concat([lag(y, 24 * d) for d in range(1, 8)], axis=1).min(axis=1)
    df["y_d1_vs_d2"] = df["y_d1_mean"] - df["y_d2_mean"]
    df["y_lag24_vs_7d"] = df["y_lag24"] - df["y_samehour_7d_mean"]
    return df


def raw(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Columns from the full collect_data() pull (research/pjm/collect_raw.R)."""
    r = pd.read_parquet(ROOT / "data" / "pjm_raw.parquet")
    r = r.assign(ts=pd.to_datetime(r["time"])).set_index("ts")
    return r[cols].reindex(df.index)


def cfg(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Columns from the full dafc-20261001 config pull (research/pjm/collect_config.R)."""
    r = pd.read_parquet(ROOT / "data" / "pjm_config.parquet")
    r = r.assign(ts=pd.to_datetime(r["time"])).set_index("ts")
    return r[cols].reindex(df.index)


MF = ".meteologica.forecast."


def extra(df: pd.DataFrame, name: str, key: str | None = None) -> pd.DataFrame:
    """A table from research/pjm/collect_extra.R (fair D-1 vintages) on df's hourly
    index, keyed columns spread wide as <col>__<key>. dt/hr are hour-ending EPT."""
    r = pd.read_parquet(ROOT / "data" / "pjm_extra" / f"{name}.parquet")
    r["ts"] = pd.to_datetime(r["dt"]) + pd.to_timedelta(r["hr"] - 1, unit="h")
    r = r.drop(columns=["dt", "hr", "forecast_time"])
    if key:
        r[key] = r[key].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
        r = r.drop_duplicates(["ts", key]).pivot(index="ts", columns=key)
        r.columns = [f"{c}__{k}" for c, k in r.columns]
    else:
        r = r.drop_duplicates("ts").set_index("ts")
    return r.reindex(df.index)


def add_outage_split(df: pd.DataFrame) -> pd.DataFrame:
    """Planned/maintenance/forced outage MW (D-1 06:00 vintage, daily values)."""
    x = extra(df, "outages", "region")
    for c in ("planned_outages_mw", "maintenance_outages_mw", "forced_outages_mw",
              "renewable_capacity_on_outage_mw"):
        df[f"out_{c.removesuffix('_mw')}"] = x[f"{c}__all"]
    df["out_forced_share"] = x["forced_outages_mw__all"] / x["capacity_on_outage_mw__all"]
    df["out_forced_delta1d"] = df["out_forced_outages"] - lag(df["out_forced_outages"], 24)
    df["out_forced_western"] = x["forced_outages_mw__western"]
    df["out_forced_dominion"] = x["forced_outages_mw__mid_atlantic_dominion"]
    return df


def add_load_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """Meteologica ensemble spread and ensemble-vs-point load forecast disagreement."""
    sl = extra(df, "scen_load", "region")
    df["load_ens_std"] = sl["std_dev__all"]
    df["load_ens_p95_p5"] = sl["quantile_95__all"] - sl["quantile_5__all"]
    df["load_ens_upside"] = sl["quantile_95__all"] - sl["quantile_50__all"]
    sw = extra(df, "scen_wind", "region")
    df["wind_ens_std_ecmwf"] = sw["std_dev__ecmwf_ens"]
    df["wind_ens_std_gefs"] = sw["std_dev__gefs"]
    df["wind_ens_p5_ecmwf"] = sw["quantile_5__ecmwf_ens"]
    df["wind_ecmwf_vs_gefs"] = sw["average__ecmwf_ens"] - sw["average__gefs"]
    ss = extra(df, "scen_solar", "region")
    df["solar_ens_std"] = ss["std_dev__ecmwf_ens"]
    # pjm_load_forecast (ISO) is mostly stamped ~10:47 EPT, after the 10:30 cutoff,
    # with no earlier vintage kept -> not used.
    ml = extra(df, "met_load", "region")
    df["load_met_band80"] = ml["upper_80__all"] - ml["lower_80__all"]
    df["load_ens_mean_minus_point"] = sl["average__all"] - df["pjm_load"]
    return df


CITIES = {3: "cmh", 9: "dca", 10: "fwa", 11: "ewr", 13: "phl", 15: "bwi", 18: "orf",
          20: "ric", 21: "crw", 27: "roa", 70: "akr", 74: "cle", 114: "iad",
          162: "ord", 164: "pit"}


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Maxar city forecasts: cross-city aggregates, degree days, big-city temps, swings."""
    x = extra(df, "maxar", "location_id")
    temp = x[[f"temp__{i}" for i in CITIES]]
    df["wx_temp_mean"] = temp.mean(axis=1)
    df["wx_temp_min"] = temp.min(axis=1)
    df["wx_temp_max"] = temp.max(axis=1)
    df["wx_temp_spread"] = temp.std(axis=1)
    df["wx_dew_mean"] = x[[f"dew_point__{i}" for i in CITIES]].mean(axis=1)
    df["wx_wind_mean"] = x[[f"wind_speed__{i}" for i in CITIES]].mean(axis=1)
    df["wx_wchl_min"] = x[[f"wchl__{i}" for i in CITIES]].min(axis=1)
    df["wx_heat_index_max"] = x[[f"heat_index__{i}" for i in CITIES]].max(axis=1)
    df["wx_hdd"] = (65 - df["wx_temp_mean"]).clip(lower=0)
    df["wx_cdd"] = (df["wx_temp_mean"] - 65).clip(lower=0)
    day = df.index.floor("D")
    df["wx_day_temp_min"] = df["wx_temp_mean"].groupby(day).transform("min")
    df["wx_day_temp_max"] = df["wx_temp_mean"].groupby(day).transform("max")
    df["wx_temp_delta24"] = df["wx_temp_mean"] - lag(df["wx_temp_mean"], 24)
    df["wx_feelslike_drop4h"] = df["feelslike_weightedavg"] - lag(df["feelslike_weightedavg"], 4)
    for i in (162, 114, 164, 18, 13):  # ORD, IAD, PIT, ORF, PHL
        df[f"wx_temp_{CITIES[i]}"] = x[f"temp__{i}"]
    return df


def add_op_summary(df: pd.DataFrame) -> pd.DataFrame:
    """PJM daily operating summary for D-1 (published the morning of D-1)."""
    r = pd.read_parquet(ROOT / "data" / "pjm_extra" / "op_summary.parquet")
    r = r.assign(day=pd.to_datetime(r["dt"]) + pd.Timedelta(days=1)).drop_duplicates("day")
    r = r.set_index("day")
    day = df.index.floor("D")
    for c in ("operating_reserve_mw", "unscheduled_steam_capacity_mw",
              "scheduled_tie_flow_total_mw", "load_forecast_mw"):
        df[f"ops_d1_{c.removesuffix('_mw')}"] = r[c].reindex(day).to_numpy()
    df["ops_d1_reserve_ratio"] = df["ops_d1_operating_reserve"] / df["ops_d1_load_forecast"]
    return df


def add_regional(df: pd.DataFrame) -> pd.DataFrame:
    regional = [f"load_{r}{MF}pjm" for r in ("southern", "western", "midatlantic")]
    regional += [f"wind_{r}{MF}pjm" for r in ("western", "midatlantic", "south")]
    regional += [f"solar_{r}{MF}pjm" for r in ("midatlantic", "south", "west")]
    x = cfg(df, regional + ["thermal_capacity_all.eia.forecast.pjm"])
    for c in regional:
        df[c.replace(MF, "_")] = x[c]
    df["pjm_thermal_capacity"] = x["thermal_capacity_all.eia.forecast.pjm"]
    net_load = df["pjm_load"] - df["pjm_wind"].fillna(0) - df["pjm_solar"].fillna(0)
    df["supply_cushion"] = df["pjm_thermal_capacity"] - df["pjm_gen_on_outage"] - net_load
    df["supply_cushion_ratio"] = df["supply_cushion"] / net_load
    return df


def add_fuels(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"gas_price_{h}.ice.forecast.pjm" for h in ("domsouth", "tetco", "transco_z6_ny", "chicago")]
    cols += ["central_app", "illiois_basin", "northern_app"]
    x = cfg(df, cols)
    for c in cols:
        df[c.removesuffix(".ice.forecast.pjm")] = x[c]
    return df


def add_neighbors(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for m in ("miso", "spp"):
        cols += [f"load_all{MF}{m}", f"wind_all{MF}{m}", f"solar_all{MF}{m}",
                 f"genonoutage_region_all.iso.forecast.{m}", f"thermal_capacity_all.eia.forecast.{m}"]
    for m in ("nyiso", "isone"):
        cols += [f"load_all{MF}{m}", f"wind_all{MF}{m}", f"solar_all{MF}{m}"]
    for m in ("tva", "soco"):
        cols += [f"load_all{MF}{m}", f"solar_all{MF}{m}"]
    x = cfg(df, cols)
    for c in cols:
        df[c.replace(MF, "_").replace(".iso.forecast.", "_").replace(".eia.forecast.", "_")] = x[c]
    return df


def onpeak_daily_mean(s: pd.Series) -> pd.Series:
    """Mean over HE 8-23 (hour-beginning 07-22) of each day, broadcast to the day."""
    day = s.index.floor("D")
    on = s.where((s.index.hour >= 7) & (s.index.hour <= 22))
    return on.groupby(day).transform("mean")


def add_ice(df: pd.DataFrame) -> pd.DataFrame:
    """West Hub ICE next-day on-peak price (fair as of the bid deadline; Tue-Fri only)
    and the production ICE-RF anchor: ICE less expected West Hub congestion."""
    x = raw(df, ["west_ice_price", "west_congestion_fc", "west_da_congestion"])
    df["west_ice_price"] = x["west_ice_price"]
    cong_fc = onpeak_daily_mean(x["west_congestion_fc"])
    cong_yday = onpeak_daily_mean(lag(x["west_da_congestion"], 24))
    df["ice_pred_da_onpeak"] = x["west_ice_price"] - (cong_fc + cong_yday) / 2
    df["is_onpeak"] = ((df.index.hour >= 7) & (df.index.hour <= 22)).astype(float)
    return df


def add_rt(df: pd.DataFrame) -> pd.DataFrame:
    """RT price history known before the D-1 deadline (lags >= 48h, D-2 and earlier)."""
    x = raw(df, ["rt_energy_price", "da_energy_price", "is_spike"])
    rt = x["rt_energy_price"]
    df["rt_lag48"] = lag(rt, 48)
    df["rt_lag168"] = lag(rt, 168)
    df["rt_d2_mean"] = daily_lag(rt, 2, "mean")
    df["rt_d2_max"] = daily_lag(rt, 2, "max")
    df["rt_da_spread_d2"] = daily_lag(rt - x["da_energy_price"], 2, "mean")
    df["spikes_d2_d8"] = pd.concat(
        [daily_lag(x["is_spike"], d, "sum") for d in range(2, 9)], axis=1
    ).sum(axis=1)
    return df


def add_congestion(df: pd.DataFrame) -> pd.DataFrame:
    x = raw(df, ["west_congestion_fc", "west_da_congestion", "solarduckcurve_lag1_diff_pjm"])
    df["west_congestion_fc"] = x["west_congestion_fc"]
    df["west_da_congestion_lag24"] = lag(x["west_da_congestion"], 24)
    df["solarduckcurve_lag1_diff_pjm"] = x["solarduckcurve_lag1_diff_pjm"]
    return df


def add_deltas_only(df: pd.DataFrame) -> pd.DataFrame:
    """Fundamental deltas without keeping the price-lag block (y_lag24 is needed
    internally for the heat-rate / gas-scaled anchor features)."""
    df["y_lag24"] = lag(df["y"], 24)
    return add_fundamental_deltas(df)


FUNDAMENTALS = [
    "pjm_load", "pjm_wind", "pjm_solar", "pjm_gen_on_outage", "blended_gasprice",
    "feelslike_weightedavg", "thermal_stress_all_pjm", "thermal_stress_all_miso",
    "thermal_stress_all_spp", "north_external_stress", "south_external_stress",
    "iir_daily_nuclear_outage",
]


def add_broad_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Day-over-day and week-over-week changes of every fundamental, plus hourly ramps."""
    for c in FUNDAMENTALS:
        df[f"{c}_d24"] = df[c] - lag(df[c], 24)
        df[f"{c}_d168"] = df[c] - lag(df[c], 168)
    for c in ("pjm_load", "net_load", "pjm_solar", "thermal_stress_all_pjm"):
        df[f"{c}_ramp1"] = df[c] - lag(df[c], 1)
    return df


def drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["dt", "dt_ordinal", "hr"])


VARIANTS = {
    "v0": [],
    "v1_price_lags": [add_price_lags],
    "v2_lags_deltas": [add_price_lags, add_fundamental_deltas],
    "v3_lags_deltas_slim": [add_price_lags, add_fundamental_deltas, drop_redundant],
    "v4_v3_calendar": [add_price_lags, add_fundamental_deltas, drop_redundant, add_calendar],
    "v5_v3_history": [add_price_lags, add_fundamental_deltas, drop_redundant, add_price_history],
    "v6_deltas_only": [add_deltas_only, drop_redundant],
    "v7_v5_broad_deltas": [add_price_lags, add_fundamental_deltas, drop_redundant,
                           add_price_history, add_broad_deltas],
    "r1_ice": [add_ice],
    "r2_rt": [add_rt],
    "r3_cong": [add_congestion],
    "c1_regional": [add_regional],
    "c2_fuels": [add_fuels],
    "c3_neighbors": [add_neighbors],
    "v8_v5_regional": [add_price_lags, add_fundamental_deltas, drop_redundant,
                       add_price_history, add_regional],
    "x1_outage_split": [add_outage_split],
    "x2_load_uncertainty": [add_load_uncertainty],
    "x3_weather": [add_weather],
    "x4_op_summary": [add_op_summary],
    "x5_all_extra": [add_outage_split, add_load_uncertainty, add_weather, add_op_summary],
    "x6_v8_all_extra": [add_price_lags, add_fundamental_deltas, drop_redundant, add_price_history,
                        add_regional, add_outage_split, add_load_uncertainty, add_weather,
                        add_op_summary],
    "x7_v8_weather": [add_price_lags, add_fundamental_deltas, drop_redundant, add_price_history,
                      add_regional, add_weather],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=list(VARIANTS))
    p.add_argument("--src", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "pjm-fe")
    args = p.parse_args()

    df = load(args.src)
    for fn in VARIANTS[args.variant]:
        df = fn(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop(columns=df.columns[df.isna().all()]).reset_index()

    out = args.out / args.variant
    out.mkdir(parents=True, exist_ok=True)
    for split, part in df.groupby("_split"):
        part.drop(columns="_split").to_parquet(out / f"{split}.parquet", index=False)
    cols = [c for c in df.columns if c not in ("_split", "ts", "y")]
    print(f"{args.variant}: {len(cols)} features -> {out}")


if __name__ == "__main__":
    main()
