# tios-gpu-research

GPU research sandbox. First project: zero-shot forecasting with NVIDIA
[Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular)
([API](https://nvidia.github.io/structured-data-models/api/generated/sdm.models.KumoTabular)).

## On the GPU box

Salt runs `just init` (`uv sync` and pre-download of the model weights). Then:

```bash
just smoke                                   # GPU check and toy forecast
just pull s3://<bucket>/<prefix>/my-task     # -> /scratch/data/my-task
just forecast my-task --target y --time-col ts [--walk-forward] [--drop id_col ...] [--max-context 20000]
just push s3://<bucket>/<prefix>/my-task     # -> .../my-task/outputs/<UTC timestamp>/
```

Results are written to `/scratch/outputs/my-task/`: `val_<mode>_predictions.parquet`
(mean + quantiles) and `val_<mode>_metrics.json` (metrics + run arguments), where
`<mode>` is `static` or `walk_forward`. `/scratch` is wiped when the instance stops.

## Data layout

`train` and `val` (`.parquet` or `.csv`) in one directory, both with the same
columns: a numerical target plus features. Kumo-Tabular is a tabular in-context
model, not a sequence model, so build the forecasting features yourself (lags,
rolling stats, calendar, exogenous). Each row is one (entity, time) to predict.

- Only val is forecast. Any `test` file is ignored.
- Default (`static`): all of val is predicted using train as context.
- `--walk-forward`: val is predicted one day at a time. After each day, its actual
  targets join the context for the next day. Lag features must only use data
  available before the forecast day, or the backtest leaks.
- With `--time-col`, the context is the most recent `--max-context` rows.
- Ensembling is random; `--seed` (default 0) makes runs repeatable.
- Datetime columns are expanded into numeric features. String columns are
  treated as categorical. The model uses at most 500 features.
