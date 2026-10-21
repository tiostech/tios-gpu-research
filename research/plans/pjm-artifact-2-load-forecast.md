# Artifact 2 spec: Kumo PJM load forecast, and feeding it into the Kumo energy forecast

Read `research/plans/common.md` first, then `research/spp-load-kumo.md` (the SPP/OKGE load work this
builds on — including the D-1 context leak and its fix). Needs SQL (ask for the tunnel) and ~4 h of GPU.

## Audience and message
Stakeholders and the trading desk. Two questions:
(a) Can Kumo beat Meteologica's PJM ALL load forecast? (SPP precedent: yes, -21% MAE for SPP ALL and
-7.5% for OKGE, when the vendor/ISO forecasts are Kumo *inputs*; weather-only Kumo loses.)
(b) Does swapping that better load forecast into the Kumo energy model's `pjm_load` input improve the
DAEP forecast? Report honestly if not — the baseline's `pjm_load` is already a good vendor forecast and
price error is dominated by a few spike days.

## Key facts
- In `PjmFlexDaep20261001$collect_data()` (`~/dev/tios-quant-shiny/lib/trader-forecasting/pjm-flex-daep-20261001.R`,
  ~line 339) `pjm_load = load_all.meteologica.forecast.pjm` — the Meteologica PJM ALL load forecast
  imported through the config framework. So Meteologica ALL is both the load benchmark and the series
  replaced in the energy model. Verify: a fresh fair-vintage pull of `pjm_meteologica_load_forecasts`
  (region ALL) should match `data/pjm-fe/v0`'s `pjm_load` on 2025; if it doesn't, use whatever
  reproduces `v0.pjm_load` as the benchmark and say so.
- `thermal_stress_all_pjm`, `solarduckcurve.forecast.pjm` (+ its lead diff) and
  `southwest_east_diff_duckcurve_pjm` are also derived from Meteologica load in the production config.
- The baseline's 2025 walk-forward only uses the **last 10,000 context rows**: at the first 2025 step
  the context starts ~2023-11-08. So the custom load forecast is needed out of sample only from
  **~2023-10-01 to 2025-12-31**.

## Work plan
1. **Table discovery + pull** (tunnel). PJM actual hourly load ALL (find the history importer used by
   `tios_macro_data_import(market = "pjm", importer_keyword = "load", vendor_type = "iso",
   data_type = "history")`, or `SHOW TABLES LIKE 'pjm%load%'`); `pjm_meteologica_load_forecasts`
   (ALL + regions); `pjm_load_forecast` (ISO; `RTO COMBINED` + regions); `pjm_maxar_weather_forecasts`
   (the 15 location ids in `research/pjm/make_features.py` CITIES); the weights behind
   `feelslike_weightedavg` (PJM daep config json under `lib/trader-forecasting/jsons/`).
   2017-12 .. 2025-12, fair vintages (`collect_extra.R` pattern). Audit every table: no row used for day D
   issued after D-1 10:30 EPT. **ISO caution:** it was mostly stamped ~10:47 EPT in an earlier pull; the
   fair query then returns an older vintage — measure how stale and drop it if > ~1 day older than
   Meteologica's.
2. **Load features** (`research/pjm_load/make_features.py`, mirroring `research/spp_load/make_features_all.py`):
   pop-weighted feels-like (config weights) + lags 24/48/72, rolling 24/72, daily mean/sd/max/min, day
   lags, delta24, CDD/HDD; per-city feels-like; temp / dew / cloud for the big cities; calendar (dow,
   season sin/cos, holidays, Thanksgiving / Christmas weeks, COVID), daylight; actual load lags >= 48h
   (48/72/168, same-hour 7d mean from lag 48, D-2 mean/max/min); ISO forecast (if fair); Meteologica
   regional forecasts; and in the main variant the Meteologica ALL forecast (level, shape).
3. **Screen** two variants on 2025 with `--eval-every 3 --context-gap-hours 24` (~15 min each):
   P0 no vendor/ISO (weather + calendar + lags) and P2 with ISO + Meteologica. Pick the winner (expect P2).
4. **Backpublish** the winner: one daily walk-forward, train < 2023-10-01, val 2023-10-01 .. 2025-12-31
   (`research/pjm/resplit.py` style), `--context-gap-hours 24`, est8, seed 0, `--max-context 10000`
   (~820 steps, ~1.5-2 h). Its 2025 part is the section-A result; all of it feeds section B. Never use
   in-sample fitted values.
5. **Score the load forecast** vs Meteologica ALL (and ISO) on 2025: `research/spp_load/compare.py --bench`.
6. **Energy model with the custom load:** `data/pjm-fe/v0_customload_a` = v0 with `pjm_load` replaced by
   the backpublished q500 for 2023-10-01..2025-12-31 (earlier rows: keep the original — they never
   enter the context — and say so); `_b` also recomputes the load-derived features above, if their
   formulas are clear in the config jsons (otherwise report (a) only). Assert for each 2025 day that the
   load value came from a load-model step whose context ended by D-2. Run both with exactly the baseline
   settings (est8, seed 0, 10k, daily 2025, no gap; ~35 min each). Compare with the baseline (paired bootstrap).

## Artifact sections
1. **Headline.** Load: Kumo vs Meteologica ALL (MAE / RMSE / MAPE, ALL + HOLD, gain +- SE). Energy:
   baseline vs baseline + custom load (a/b).
2. **Load scorecard + breakdown.** By season / hour block; bias (SPP showed both vendor and Kumo
   under-forecast in 2025 due to load growth — check PJM); optional P0 vs P2 to show why the vendor
   forecast must be an input.
3. **Load time series (Plotly).** Hourly 2025: actual PJM ALL load, Meteologica ALL, ISO (if fair),
   Kumo load. Range slider/buttons, unified hover, daily-mean toggle.
4. **Energy impact.** Scorecard baseline vs custom-load a/b; where it helps (season / hour / spike days).
5. **Energy time series (Plotly).** Actual DAEP, Kumo baseline, Kumo baseline + custom load
   (production optional, off by default).
6. **How it works (short).** Two-stage pipeline diagram (weather + vendor/ISO forecasts -> Kumo load
   model -> load forecast -> Kumo energy model), the walk-forward / leak rule in one plain sentence, and a
   link to Artifact 1 for what Kumo-Tabular is.
7. **Model specs.** Rows: Meteologica ALL (vendor, D-1 vintage), ISO load fc (vintage used), Kumo load
   P0/P2, Kumo original baseline, baseline + custom load a/b. Columns: target, method, context window,
   cadence, inputs, cutoff, estimators/seed, scored period.

## Deliverables / bookkeeping
- Code: `research/pjm_load/` (collect, features, compare), `data/pjm-load-fe/`, `data/pjm-fe/v0_customload_*`.
- Outputs on the box: `/scratch/outputs/pjm-load-fe/`, `/scratch/outputs/pjm-fe/v0_customload_*`.
- Log in `research/pjm-kumo-feature-engineering.md` (new section) or a new `research/pjm-load-kumo.md`;
  commit/push; offer S3 sync.
