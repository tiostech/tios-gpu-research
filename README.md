# tios-gpu-research

GPU research sandbox. First project: zero-shot forecasting with NVIDIA
[Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular)
([API](https://nvidia.github.io/structured-data-models/api/generated/sdm.models.KumoTabular)).

## On the GPU box

Salt runs `just init` (`uv sync` and pre-download of the model weights). Then:

```bash
just smoke                                   # GPU check and toy forecast
just pull s3://<bucket>/<prefix>/my-task     # -> /scratch/data/my-task
just forecast my-task --target y --time-col ts [--drop id_col ...] [--max-context 20000]
```

Results are written to `/scratch/outputs/my-task/`: `{val,test}_predictions.parquet`
(mean + quantiles) and `metrics.json`. `/scratch` is wiped when the instance stops.

## Data layout

`train`, `val`, and `test` (`.parquet` or `.csv`) in one directory, all with the same
columns: a numerical target plus features. Kumo-Tabular is a tabular in-context
model, not a sequence model, so build the forecasting features yourself (lags,
rolling stats, calendar, exogenous). Each row is one (entity, time) to predict.

- val is predicted using train as context. test is predicted using train + val.
- With `--time-col`, the context is the most recent `--max-context` rows.
- Datetime columns are expanded into numeric features. String columns are
  treated as categorical. The model uses at most 500 features.
