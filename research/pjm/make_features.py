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


def drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["dt", "dt_ordinal", "hr"])


VARIANTS = {
    "v0": [],
    "v1_price_lags": [add_price_lags],
    "v2_lags_deltas": [add_price_lags, add_fundamental_deltas],
    "v3_lags_deltas_slim": [add_price_lags, add_fundamental_deltas, drop_redundant],
    "v4_v3_calendar": [add_price_lags, add_fundamental_deltas, drop_redundant, add_calendar],
    "v5_v3_history": [add_price_lags, add_fundamental_deltas, drop_redundant, add_price_history],
    "r1_ice": [add_ice],
    "r2_rt": [add_rt],
    "r3_cong": [add_congestion],
    "c1_regional": [add_regional],
    "c2_fuels": [add_fuels],
    "c3_neighbors": [add_neighbors],
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
    df = df.replace([np.inf, -np.inf], np.nan).reset_index()

    out = args.out / args.variant
    out.mkdir(parents=True, exist_ok=True)
    for split, part in df.groupby("_split"):
        part.drop(columns="_split").to_parquet(out / f"{split}.parquet", index=False)
    cols = [c for c in df.columns if c not in ("_split", "ts", "y")]
    print(f"{args.variant}: {len(cols)} features -> {out}")


if __name__ == "__main__":
    main()
