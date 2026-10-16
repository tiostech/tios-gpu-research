"""Walk-forward forecasting with Mistral Timestral (time-series foundation model).

Uses the same train/val files and features as kumo_forecast.py for a like-for-
like comparison.  Requires a regular time grid in --time-col (e.g. hourly with
DST gaps present as rows with a missing target).

For each val day, Timestral reads the target history (train + earlier val
days, up to --max-context steps) and forecasts that day.  Every non-target
column, after the same datetime expansion as Kumo, is passed as a known-future
covariate, i.e. its value is available for the forecast day, exactly as Kumo
sees the query rows' features.  String columns become categorical covariates.

Outputs: val_timestral_predictions.parquet and val_timestral_metrics.json,
with the same metrics as kumo_forecast.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster

from kumo_forecast import SAVE_QUANTILES, metrics, prepare, read_split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--target", default="y")
    p.add_argument("--time-col", required=True)
    p.add_argument("--drop", nargs="*", default=[], help="Columns to ignore")
    p.add_argument(
        "--size", default="209M", choices=["5M", "28M", "123M", "209M"]
    )
    p.add_argument("--max-context", type=int, default=32_768)
    p.add_argument(
        "--no-covariates",
        action="store_true",
        help="Forecast from the target history alone",
    )
    p.add_argument("--days", type=int, help="Only forecast the first N val days")
    args = p.parse_args()

    splits = {s: read_split(args.data_dir, s) for s in ("train", "val")}
    for name, df in splits.items():
        if df is None:
            p.error(f"no {name}.parquet/{name}.csv in {args.data_dir}")
    full = pd.concat(
        [splits["train"].assign(_val=False), splits["val"].assign(_val=True)]
    ).drop(columns=args.drop)
    full[args.time_col] = pd.to_datetime(full[args.time_col])
    full = full.sort_values(args.time_col).reset_index(drop=True)
    times, is_val = full[args.time_col], full.pop("_val").to_numpy()
    steps = times.diff().dropna().unique()
    if len(steps) != 1:
        p.error(f"--time-col must be a regular grid, found steps {steps[:5]}")

    y = full[args.target].to_numpy(dtype=np.float32)
    features = prepare(full.drop(columns=[args.target]), args.time_col)
    features = features.loc[:, features.nunique(dropna=False) > 1]
    categorical = features.select_dtypes(exclude="number")
    numeric = features.drop(columns=categorical.columns)
    known = numeric.to_numpy(dtype=np.float32).T  # (C, T)
    known_cat = np.stack(
        [pd.factorize(categorical[c])[0] for c in categorical]
    ).astype(np.float32) if len(categorical.columns) else None
    if known_cat is not None:
        known_cat[known_cat < 0] = np.nan  # factorize marks missing as -1
    if args.no_covariates:
        known = known_cat = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimestralForecaster.from_pretrained(
        f"mistralai/Timestral-{args.size}", device=device
    )
    levels = np.array(model.forecast(np.zeros(64), 1).quantile_levels)
    names = [f"q{round(q * 1000):03d}" for q in levels]

    val_idx = np.flatnonzero(is_val)
    if args.days:
        first_days = times[val_idx].dt.floor("D").drop_duplicates()[: args.days]
        val_idx = val_idx[times[val_idx].dt.floor("D").isin(first_days).to_numpy()]
    days = pd.Series(val_idx).groupby(times[val_idx].dt.floor("D").to_numpy())
    parts = []
    for i, (day, idx) in enumerate(days, 1):
        start, end = idx.iloc[0], idx.iloc[-1] + 1
        lo = max(0, start - args.max_context)
        kwargs = {}
        if known is not None:
            kwargs["known_covariates"] = known[None, :, lo:end]
        if known_cat is not None:
            kwargs["categorical_known_covariates"] = known_cat[None, :, lo:end]
        forecast = model.forecast(
            y[None, None, lo:start],
            prediction_length=end - start,
            clip_nonnegative=False,
            **kwargs,
        )
        parts.append(forecast.quantiles[0, 0].cpu().numpy())  # (h, Q)
        print(f"timestral: {i}/{len(days)} days ({pd.Timestamp(day):%Y-%m-%d})")

    quantiles = pd.DataFrame(np.concatenate(parts), columns=names)
    # Approximate the mean by integrating the quantile function over 1-99%.
    mean = np.trapezoid(quantiles.to_numpy(), levels, axis=1) / (
        levels[-1] - levels[0]
    )
    y_val = y[val_idx].astype(float)
    out = pd.DataFrame({args.time_col: times[val_idx].to_numpy(), "y_true": y_val})
    out["mean"] = mean
    out = pd.concat([out, quantiles[SAVE_QUANTILES]], axis=1)

    has_y = ~np.isnan(y_val)
    results = metrics(
        y_val[has_y], quantiles[has_y].reset_index(drop=True), mean[has_y]
    )
    results["n_missing_target"] = int((~has_y).sum())
    print(f"val (timestral): {json.dumps(results)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_dir / "val_timestral_predictions.parquet")
    (args.out_dir / "val_timestral_metrics.json").write_text(
        json.dumps(
            {
                "mode": "timestral_walk_forward",
                **vars(args),
                "covariates": [] if known is None else list(features.columns),
                **results,
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
