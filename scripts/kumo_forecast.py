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
from datetime import date
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
    # Parquet date columns load as object dtype holding datetime.date values.
    for col in df.columns[df.dtypes == object]:
        values = df[col].dropna()
        if len(values) and values.map(lambda v: isinstance(v, date)).all():
            df[col] = pd.to_datetime(df[col])
    for col in df.columns[df.dtypes.map(pd.api.types.is_datetime64_any_dtype)]:
        ts = df.pop(col)
        df[f"{col}_epoch"] = ts.astype("datetime64[s]").astype("int64")
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


def target_transform(spec: str):
    """(forward(y, df), inverse(q, df)) for --target-transform; all are monotone
    in y, so inverse-transforming each predicted quantile is exact."""
    kind, _, col = spec.partition(":")
    if kind == "none":
        return (lambda y, df: y), (lambda q, df: q)
    if kind == "asinh":  # compresses price spikes; ~linear below the scale
        scale = float(col or 20.0)
        return (lambda y, df: np.arcsinh(y / scale)), (lambda q, df: np.sinh(q) * scale)
    if kind == "log":
        return (lambda y, df: np.log(y)), (lambda q, df: np.exp(q))
    if kind == "ratio":  # e.g. ratio:smoothed_gas -> implied heat rate
        return (lambda y, df: y / df[col]), (lambda q, df: q * df[col])
    if kind == "logratio":
        return (lambda y, df: np.log(y / df[col])), (lambda q, df: np.exp(q) * df[col])
    if kind == "diff":  # e.g. diff:y_lag24 -> predict the change
        return (lambda y, df: y - df[col]), (lambda q, df: q + df[col])
    raise ValueError(f"unknown --target-transform {spec!r}")


def select_context(history, query, args) -> pd.DataFrame:
    """--context-select: 'recent' = last --max-context rows; 'knn' = for each
    query row its nearest history rows (standardized features), unioned and
    topped up with the most recent rows; 'mix' = half recent, half knn."""
    history = history.tail(args.context_pool)
    n = args.max_context
    if args.context_select == "recent" or len(history) <= n:
        return history.tail(n)
    n_knn = n if args.context_select == "knn" else n // 2
    cols = [c for c in query.columns
            if c not in (args.target, args.time_col) and pd.api.types.is_numeric_dtype(query[c])
            and pd.api.types.is_numeric_dtype(history[c])]
    h = history[cols].to_numpy(float)
    mu, sd = np.nanmean(h, 0), np.nanstd(h, 0) + 1e-9
    h = np.nan_to_num((h - mu) / sd)
    q = np.nan_to_num((query[cols].to_numpy(float) - mu) / sd)
    d = ((q**2).sum(1)[:, None] - 2 * q @ h.T + (h**2).sum(1)[None, :])  # [Q, H]
    # Round-robin over query rows by rank until n_knn unique rows are picked.
    order = np.argsort(d, axis=1)
    picked, seen = [], np.zeros(len(history), bool)
    for rank in range(order.shape[1]):
        for i in order[:, rank]:
            if not seen[i]:
                seen[i] = True
                picked.append(i)
        if len(picked) >= n_knn:
            break
    seen[:] = False
    seen[picked[:n_knn]] = True
    for i in range(len(history) - 1, -1, -1):  # top up with the most recent
        if seen.sum() >= n:
            break
        seen[i] = True
    return history[seen]


def metrics(
    y: np.ndarray, quantiles: pd.DataFrame, mean: np.ndarray | None = None
) -> dict[str, float]:
    if mean is None:  # Mean of 999 evenly spaced quantiles.
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
        "rmse_median": float(np.sqrt(np.mean((y - quantiles["q500"]) ** 2))),
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
    p.add_argument(
        "--step-days",
        type=int,
        default=1,
        help="Walk forward N days at a time (faster screening; 1 = daily)",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Only forecast every Nth step (context still gets every step's "
        "actuals), i.e. an unbiased 1/N sample of the daily walk-forward",
    )
    p.add_argument(
        "--target-transform",
        default="none",
        help="none | asinh[:scale] | log | ratio:COL | logratio:COL | diff:COL",
    )
    p.add_argument("--context-select", default="recent", choices=["recent", "knn", "mix"])
    p.add_argument("--context-pool", type=int, default=10_000,
                   help="History rows that 'knn'/'mix' may select from")
    p.add_argument("--feature-frac", type=float, default=1.0,
                   help="Random-subspace member: keep this fraction of feature columns")
    p.add_argument("--feature-seed", type=int, default=0, help="RNG seed for --feature-frac")
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
    if args.feature_frac < 1:
        # Same column subset for the whole run; the time column (calendar features) is always kept.
        cols = [c for c in train.columns if c not in (args.target, args.time_col)]
        rng = np.random.default_rng(args.feature_seed)
        keep = set(rng.choice(cols, size=max(1, round(len(cols) * args.feature_frac)), replace=False))
        dropped = [c for c in cols if c not in keep]
        train, val = train.drop(columns=dropped), val.drop(columns=dropped)
        print(f"feature-frac {args.feature_frac} seed {args.feature_seed}: "
              f"kept {len(keep)}/{len(cols)} features")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sdm.models.KumoTabular(task="regression", size=args.size, device=device)

    fwd, inv = target_transform(args.target_transform)

    def forecast(context: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
        context = select_context(context.dropna(subset=[args.target]), query, args).copy()
        context[args.target] = fwd(context[args.target], context)
        context = context.replace([np.inf, -np.inf], np.nan).dropna(subset=[args.target])
        q = predict(
            model,
            prepare(context, args.time_col),
            prepare(query, args.time_col),
            args.target,
            args,
            device,
        )
        q = q.apply(lambda c: inv(c.to_numpy(), query.reset_index(drop=True)))
        return q

    if args.walk_forward:
        mode = "walk_forward"
        if args.step_days > 1:
            mode += f"_{args.step_days}d"
        if args.eval_every > 1:
            mode += f"_every{args.eval_every}"
        context, parts = train, []
        day = val[args.time_col].dt.floor("D")
        day_num = (day - day.iloc[0]).dt.days
        steps = list(val.groupby(day_num // args.step_days, sort=True))
        scored = []
        for i, (_, rows) in enumerate(steps, 1):
            if (i - 1) % args.eval_every == 0:
                parts.append(forecast(context, rows))
                scored.append(rows.index)
                last = rows[args.time_col].iloc[-1]
                print(f"walk-forward: {i}/{len(steps)} steps (through {last:%Y-%m-%d})")
            context = pd.concat([context, rows]).tail(max(args.max_context, args.context_pool))
        quantiles = pd.concat(parts, ignore_index=True)
        val = val.loc[np.concatenate(scored)].reset_index(drop=True)
    else:
        mode = "static"
        quantiles = forecast(train, val)

    out = pd.DataFrame({"mean": quantiles.mean(axis=1)})
    out = pd.concat([out, quantiles[SAVE_QUANTILES]], axis=1)
    results = {}
    if args.target in val:
        y = val[args.target].to_numpy(dtype=float)
        out.insert(0, "y_true", y)
        # Score only rows with an actual (e.g. DST gaps have no target).
        has_y = ~np.isnan(y)
        results = metrics(y[has_y], quantiles[has_y].reset_index(drop=True))
        results["n_missing_target"] = int((~has_y).sum())
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
