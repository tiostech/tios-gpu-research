"""Zero-shot forecasting with NVIDIA Kumo-Tabular (in-context regression).

Expects a directory with train/val files (parquet or csv) sharing one schema:
a numerical target column plus feature columns (lags, calendar, exogenous,
...).  Kumo-Tabular has no fitting step: it reads labeled "context" rows and
predicts "query" rows in a single forward pass.  Only val is forecast.

  * static (default): all of val is predicted using train as context.
  * --walk-forward: val is predicted one day at a time.  After each day is
    forecast, its actual targets are appended to the context for the next day.

The context is the most recent --max-context rows (sorted by --time-col if
given).  Datetime columns are expanded into numeric features, since the model
only accepts numerical/categorical inputs.

Outputs: val_<mode>_predictions.parquet (mean + selected quantiles) and
val_<mode>_metrics.json (MAE of median, RMSE of mean, pinball loss, 80%
coverage).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sdm

SAVE_QUANTILES = ["q050", "q100", "q250", "q500", "q750", "q900", "q950"]
PINBALL_QUANTILES = [f"q{q:03d}" for q in range(100, 1000, 100)]


def read_split(data_dir: Path, name: str) -> pd.DataFrame | None:
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = data_dir / f"{name}{ext}"
        if path.exists():
            return reader(path)
    return None


def prepare(df: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
    df = df.copy()
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
    for col in df.columns[df.dtypes.map(pd.api.types.is_datetime64_any_dtype)]:
        ts = df.pop(col)
        df[f"{col}_epoch"] = ts.astype("int64") // 10**9
        df[f"{col}_hour"] = ts.dt.hour
        df[f"{col}_dow"] = ts.dt.dayofweek
        df[f"{col}_month"] = ts.dt.month
        df[f"{col}_doy"] = ts.dt.dayofyear
    return df


def to_table(df, stypes, device):
    return sdm.TableTensor.from_pandas(df=df, stypes=stypes, device=device)


def predict(model, context, query, target, args, device) -> pd.DataFrame:
    """Fit on the context once (KV cache), then predict query in batches."""
    stypes = sdm.infer_stypes(context, overrides={target: "numerical"})
    ctx = to_table(context, stypes, device)
    feature_stypes = {k: v for k, v in stypes.items() if k != target}
    x_query = query.drop(columns=[target], errors="ignore")

    outs = []
    with torch.amp.autocast(
        device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        model.fit(
            x=ctx.drop_columns(target),
            y=ctx[:, target],
            num_estimators=args.num_estimators,
            # Re-seeded per fit so each walk-forward day is reproducible.
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
        for start in range(0, len(x_query), args.batch_size):
            batch = x_query.iloc[start : start + args.batch_size]
            outs.append(
                model.predict(to_table(batch, feature_stypes, device)).to_pandas()
            )
    model.clear()
    return pd.concat(outs, ignore_index=True)


def metrics(y: np.ndarray, quantiles: pd.DataFrame) -> dict[str, float]:
    mean = quantiles.mean(axis=1).to_numpy()
    pinball = np.mean(
        [
            np.mean(
                np.maximum(
                    (q := int(c[1:]) / 1000) * (y - quantiles[c]),
                    (q - 1) * (y - quantiles[c]),
                )
            )
            for c in PINBALL_QUANTILES
        ]
    )
    lo, hi = quantiles["q100"].to_numpy(), quantiles["q900"].to_numpy()
    return {
        "n": len(y),
        "mae_median": float(np.mean(np.abs(y - quantiles["q500"]))),
        "rmse_mean": float(np.sqrt(np.mean((y - mean) ** 2))),
        "pinball_q10_q90": float(pinball),
        "coverage_80": float(np.mean((y >= lo) & (y <= hi))),
    }


def smoke_data() -> dict[str, pd.DataFrame]:
    """Noisy daily+weekly seasonal hourly series with lag features."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=24 * 120, freq="h")
    t = np.arange(len(ts))
    y = (
        10 * np.sin(2 * np.pi * t / 24)
        + 5 * np.sin(2 * np.pi * t / (24 * 7))
        + rng.normal(0, 1, len(t))
    )
    df = pd.DataFrame({"ts": ts, "y": y})
    for lag in (24, 48, 168):
        df[f"y_lag{lag}"] = df["y"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    n = len(df)
    return {"train": df.iloc[: int(n * 0.8)], "val": df.iloc[int(n * 0.8) :]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--target", default="y")
    p.add_argument("--time-col", help="Sort/expand this datetime column")
    p.add_argument("--drop", nargs="*", default=[], help="Columns to ignore")
    p.add_argument("--size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--max-context", type=int, default=10_000)
    p.add_argument("--num-estimators", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0, help="Ensemble RNG seed")
    p.add_argument(
        "--walk-forward",
        action="store_true",
        help="Forecast val day by day, adding each day's actuals to the context",
    )
    p.add_argument("--smoke", action="store_true", help="Run on toy data")
    args = p.parse_args()

    if args.smoke:
        args.target, args.time_col = "y", "ts"
        splits = smoke_data()
    else:
        if args.data_dir is None:
            p.error("--data-dir is required unless --smoke")
        splits = {s: read_split(args.data_dir, s) for s in ("train", "val")}
        for name, df in splits.items():
            if df is None:
                p.error(f"no {name}.parquet/{name}.csv in {args.data_dir}")
    if args.walk_forward and not args.time_col:
        p.error("--walk-forward requires --time-col")

    for name, df in splits.items():
        df = df.drop(columns=args.drop)
        if args.time_col:
            df[args.time_col] = pd.to_datetime(df[args.time_col])
            df = df.sort_values(args.time_col)
        splits[name] = df.reset_index(drop=True)
    train, val = splits["train"], splits["val"]

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sdm.models.KumoTabular(task="regression", size=args.size, device=device)

    def forecast(context: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
        return predict(
            model,
            prepare(context.tail(args.max_context), args.time_col),
            prepare(query, args.time_col),
            args.target,
            args,
            device,
        )

    if args.walk_forward:
        mode = "walk_forward"
        context, parts = train, []
        days = list(val.groupby(val[args.time_col].dt.floor("D"), sort=True))
        for i, (day, rows) in enumerate(days, 1):
            parts.append(forecast(context, rows))
            context = pd.concat([context, rows]).tail(args.max_context)
            print(f"walk-forward: {i}/{len(days)} days ({day:%Y-%m-%d})")
        quantiles = pd.concat(parts, ignore_index=True)
    else:
        mode = "static"
        quantiles = forecast(train, val)

    out = pd.DataFrame({"mean": quantiles.mean(axis=1)})
    out = pd.concat([out, quantiles[SAVE_QUANTILES]], axis=1)
    results = {}
    if args.target in val and val[args.target].notna().all():
        out.insert(0, "y_true", val[args.target].to_numpy())
        results = metrics(val[args.target].to_numpy(), quantiles)
        print(f"val ({mode}): {json.dumps(results)}")
    if args.time_col:
        out.insert(0, args.time_col, val[args.time_col].to_numpy())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_dir / f"val_{mode}_predictions.parquet")
    (args.out_dir / f"val_{mode}_metrics.json").write_text(
        json.dumps({"mode": mode, **vars(args), **results}, indent=2, default=str)
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
