# Plan: PJM Kumo stakeholder artifact (energy + load)

Written 2026-10-21 at the end of the feature-engineering session, for a fresh-context agent.
Read this whole file first, then `research/pjm-kumo-feature-engineering.md` (PJM price log),
`research/spp-load-kumo.md` (load-forecasting lessons, incl. the leak) and `research/miso-kumo.md`.

## Deliverable
One private claude.ai Artifact (HTML page; Plotly from cdnjs) with seven sections:
1. PJM production DAEP forecast vs Kumo "original baseline": MAE / RMSE, 2025.
2. PJM ALL load: Meteologica ALL vs a Kumo load forecast (weather + ISO fc + Meteologica fc + ...), 2025.
3. Kumo original baseline vs "baseline + custom PJM load forecast" (Kumo load fc from step 2
   swapped into `pjm_load`), 2025.
4. Interactive time series: all energy forecasts vs actual DAEP.
5. Interactive time series: load forecasts vs actual PJM load.
6. Plain-language explanation of Kumo-Tabular vs traditional ML (our production RFs).
7. Specs of every model in the page (lookback, context, features, estimators, vintages).

## Ground rules learned so far (do not relearn the hard way)
- **Target is the DA system energy component** (`da_energy_price`, via shiny `da_energy_prices()` ->
  `pjm_energy_prices`), not West Hub LMP. Production `flex.daep.*` targets the same series.
- **Fairness cutoff:** forecast for day D uses only data available by **D-1 10:30 EPT** (PJM DA bids
  close 11:00; shiny uses -30 min). PJM forecast tables keep MANY vintages: always take
  `row_number() over (partition by dt, hr[, key] order by forecast_time desc) = 1` with
  `forecast_time <= timestamp(date_sub(dt, interval 1 day), '10:30:00')` (see `research/pjm/collect_extra.R`).
  `forecast_time` is market-local (EPT). (SPP tables, by contrast, store one D-1 ~09:00 vintage.)
- **Context leak for load:** walk-forward context must not contain D-1 actual load (unknown at the
  cutoff). Use `scripts/kumo_forecast.py --context-gap-hours 24`. Actual-load features: lags >= 48h only.
  Prices are fine without the gap (D-1 DA prices publish on D-2).
- **Evaluation convention:** score the median (`q500`). Report full 2025 ("ALL") and HOLD = the 243
  days of 2025 whose day number (from 2025-01-01) is not a multiple of 3; paired day-block bootstrap
  of the daily MAE difference (+- SE, win rate). Template: `research/miso/compare_prod.py`,
  `research/spp_load/compare.py`.
- **Compute:** box `ssh tiostech@10.0.92.46` (A10G, GPU-saturated; ~6 s per walk-forward day at
  est8 / 10k context / ~60 features). Repo at `~/tios-gpu-research` (branch `pjm-feature-engineering`,
  `git pull` after pushing). Run jobs sequentially; launch with a script file + `nohup ... < /dev/null &`
  (don't `pkill -f` with a pattern that matches your own ssh command line). Data under
  `/scratch/data/<proj>/`, outputs `/scratch/outputs/<proj>/`, logs `/scratch/logs/`.
- **SQL:** read-only on `slave`, through the user's SSH tunnel (127.0.0.1:3309) — ask the user to
  open it. Helper: `cd ~/dev/tios-quant-shiny && Rscript ~/dev/tios-gpu-research/research/pjm/sql.R "<sql>" [out.parquet]`.
  MySQL 8: `lead` is reserved.
- **S3:** the box cannot write S3. Copy outputs locally and `aws s3 sync` from the user's machine only
  when the user asks. Prefix: `s3://tios-quantforecasts-rspec/gpu-research/kumo-tabular/`.
- No PRT data (subscription ending). Don't seed-average.

## Existing assets
- Kumo original baseline (PJM): `s3://.../kumo-tabular/pjm-fe/outputs/baseline_v0_daily/` (= box
  `/scratch/outputs/pjm/val_walk_forward_predictions.parquet`). Full 2025 MAE 4.672 / RMSE 9.49; HOLD
  4.608 / 9.35. Config: Kumo-Tabular large, 8 estimators, seed 0, `--max-context 10000`, daily
  walk-forward over 2025, data `data/pjm-fe/v0` (= user's original train/val/test).
- Production PJM forecast: `data/pjm_extra/prod_flex_daep.parquet` (`pjm_tios_forecasts`, name
  `flex.daep.20261001`, point forecast, 2025-01-01 onward, stamped D-1 10:30-10:35; user confirmed it
  is published fairly). 2025: MAE 5.426 / RMSE 11.68; HOLD 5.380 / 11.44.
  -> Section 1 needs no new runs: recompute with a small script (reuse `research/pjm/analyze_blend.py`
  logic). Also show seasons / hour blocks and the paired bootstrap.
- Best PJM energy ensemble for reference (optional series in section 4): ensC (`.../pjm-fe/outputs/ensC_daily/`),
  2025 MAE 4.391 / RMSE 8.86.

## Section 2 — Kumo PJM ALL load forecast (new work)
**What `pjm_load` is today:** in `PjmFlexDaep20261001$collect_data()` (tios-quant-shiny
`lib/trader-forecasting/pjm-flex-daep-20261001.R`, ~line 339) `pjm_load = load_all.meteologica.forecast.pjm`,
i.e. the Meteologica PJM ALL load forecast imported through the config framework (fair vintage). So the
section-2 benchmark is that same Meteologica ALL series. Verify by matching `data/pjm-fe/v0`'s `pjm_load`
to a fresh pull of `pjm_meteologica_load_forecasts` (region ALL, fair vintage) on 2025 — they should agree
(near-)exactly; if not, use whatever reproduces `v0.pjm_load` as the benchmark and say so.

1. **Find the tables** (ask for the tunnel): PJM actual hourly load ALL (find the history importer used by
   `tios_macro_data_import(market="pjm", importer_keyword="load", vendor_type="iso", data_type="history")`,
   or `SHOW TABLES LIKE 'pjm%load%'`), `pjm_meteologica_load_forecasts` (region ALL + regional splits),
   `pjm_load_forecast` (ISO; zones `RTO COMBINED`, regions), `pjm_maxar_weather_forecasts`
   (the 15 location ids in `research/pjm/make_features.py` CITIES / `collect_extra.R`), plus the
   weather-weighting used by `feelslike_weightedavg` (see the PJM daep config json under
   `lib/trader-forecasting/jsons/`). Pull 2017-12..2025-12 with fair vintages (`collect_extra.R` pattern).
   **Caution:** the ISO load forecast was mostly stamped ~10:47 EPT (after the cutoff) in our earlier pull.
   With the fair-vintage query you get the latest earlier vintage (likely the previous day's issue) —
   check how stale it is and report it; drop it if it's > ~1 day older than the Meteologica vintage.
2. **Features** (mirror `research/spp_load/make_features_all.py`, which won for SPP ALL):
   - Weather: pop-weighted feels-like (config weights) + lags 24/48/72, rolling 24/72, daily
     mean/sd/max/min, day lags, delta24, CDD/HDD; per-city feels-like; temp / dew / cloud for the biggest
     cities; calendar (dow, season sin/cos, holidays, Thanksgiving / Christmas weeks, COVID), daylight.
   - Actual load lags >= 48h (lag48/72/168, same-hour 7d mean from lag 48, D-2 mean/max/min).
   - ISO load forecast (level, shape, d24) if fair; Meteologica regional load forecasts.
   - Meteologica ALL forecast (level, shape) — the SPP/OKGE winners all had the vendor forecast as an input.
3. **Run:** `--context-gap-hours 24`, est8, seed 0, `--max-context 10000`, target y = actual PJM ALL load.
   Compare with Meteologica ALL (and ISO) via `research/spp_load/compare.py --bench <parquet>`.
   Expect the vendor-as-input variant to win (SPP ALL: -21% MAE; OKGE: -7.5%). Optional quick extra:
   a no-vendor variant (weather + calendar + lags only) for the explainer.

## Section 3 — baseline with the custom load forecast (backpublish)
- The baseline's 2025 walk-forward only ever uses the **last 10,000 context rows** (~14 months). At the
  first 2025 step the context starts ~2023-11-08. So the custom load forecast must exist, out of sample,
  **from ~2023-10-01 through 2025-12-31** (earlier rows never enter the context; keep their original
  `pjm_load` or set NULL — state which). Do NOT use in-sample fitted values for the 2023-24 rows.
- Backpublish = the section-2 load model run as one daily walk-forward from 2023-10-01 to 2025-12-31
  (`research/pjm/resplit.py`-style split: train < 2023-10-01, val = 2023-10-01..2025-12-31),
  ~820 steps, ~1.5-2 h. The 2025 part of that run IS the section-2 result (same model, same
  procedure) — run once, score 2025 for section 2, reuse all of it for section 3.
- Build `data/pjm-fe/v0_customload`: copy `v0`, replace `pjm_load` with the backpublished q500
  for 2023-10-01..2025-12-31. Check `thermal_stress_all_pjm`, `solarduckcurve.forecast.pjm` and
  `southwest_east_diff_duckcurve_pjm` — they are also derived from Meteologica load in the production
  config. Run two variants: (a) swap `pjm_load` only; (b) also recompute the load-derived features from the
  custom forecast if their formulas are clear in the config jsons (else report (a) only and note it).
- Run each with exactly the baseline settings (est8, seed 0, 10k, daily walk-forward 2025, no gap —
  it's a price target). Compare to the baseline with the paired bootstrap.
- Leakage check before running: for every 2025 day D, the load fc used as a feature must come from a load
  model whose context ended at D-2 (gap 24) — true by construction of the walk-forward; assert it.

## Sections 4-5 — interactive charts
- Plotly UMD from cdnjs (pinned version), data inlined as compact JSON (round to 2 dp; ~8,760 rows x
  ~6 series is ~1-2 MB, fine). Range slider + range buttons (1w / 1m / 3m / all), legend toggles,
  hover with unified x. Daily-mean toggle so the full year is readable.
- Section 4 series: actual DAEP, production `flex.daep.20261001`, Kumo baseline, Kumo baseline + custom
  load (best of a/b), optionally ensC (best Kumo ensemble so far), all hourly 2025.
- Section 5 series: actual PJM ALL load, Meteologica ALL, ISO (if fair), Kumo load forecast.
- Optional small panel: rolling 7-day MAE per forecast.

## Section 6 — explainer for non-technical readers (draft points)
- Production model (RF / boosted trees): a model is *trained* on a fixed window (e.g. last 90 days),
  learning fixed rules from features to price; retrained each day; good at the typical relationship, can't
  "look up" situations it never trained on in that window.
- Kumo-Tabular: a large pre-trained "foundation model" for tables. It is not trained on our data at all;
  each day we hand it ~10,000 recent hours (features + actual outcome) as *context* plus tomorrow's 24 rows,
  and it predicts by recognising which past hours look like tomorrow (in-context learning, like showing an
  analyst a stack of comparable days). New data is used immediately; no tuning; returns a full distribution
  (quantiles), not just a point.
- Trade-offs: needs a GPU; can't extrapolate beyond levels it has seen in the context (why weather-only
  load models under-forecast growing load — vendor/ISO forecasts fix that); more columns isn't better.
- Use one concrete PJM example from the results (e.g. a spike day) rather than abstractions.

## Section 7 — specs table (one row per model)
Columns: model, target, method, training/context window, retrain cadence, features (count + list link),
data vintage / cutoff, estimators / seed, scored period. Rows at least: production flex.daep.20261001
(read its RF/blender lookbacks from the shiny class: rf 90/730-day windows, scaled method, blender — check
the 20261001 file), Kumo original baseline, Kumo baseline + custom load (a/b), Meteologica ALL (vendor,
D-1 vintage), ISO load fc, Kumo PJM load model (+ no-vendor variant if run), ensC (if shown).

## Order of work and checkpoints
1. Section 1 numbers (local, minutes). 2. Ask for tunnel; table discovery + fair-vintage pull; verify
`v0.pjm_load` == Meteologica ALL. 3. Build load features; quick 2025 `--eval-every 3` screen of 2 variants
(with / without vendor fc) to pick the model (~10-15 min each). 4. Backpublish run 2023-10..2025-12 of the
chosen variant (~2 h). 5. Score section 2. 6. Build v0_customload a/b, run both (~35 min each), score
section 3. 7. Build the page; one preview pass; publish. 8. Log everything in
`research/pjm-kumo-feature-engineering.md` (new section) and commit/push to `pjm-feature-engineering`;
offer an S3 sync of new outputs (`pjm-load-fe/`, `pjm-fe/outputs/v0_customload*`).
Report honestly if section 3 shows no gain — the baseline's `pjm_load` is already a good vendor
forecast, and price error is dominated by a few spike days (see the price log), so a better load fc may
move MAE only a little.
