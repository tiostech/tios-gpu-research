# Shared context for the PJM Kumo stakeholder artifacts

Written 2026-10-21 for fresh-context agents working on the `pjm-artifact-*.md` specs in this folder
(1: production vs Kumo, 2: load forecast, 4: wind and solar forecasts).
Read this file, then the artifact spec, then `research/pjm-kumo-feature-engineering.md` (PJM price
log); for artifact 2 also `research/spp-load-kumo.md` (load lessons, incl. the leak).

## Ground rules learned so far (do not relearn the hard way)
- **Target is the DA system energy component** (`da_energy_price`, via shiny `da_energy_prices()` ->
  `pjm_energy_prices`), not West Hub LMP. Production `flex.daep.*` targets the same series.
- **Fairness cutoff:** a forecast for day D uses only data available by **D-1 10:30 EPT** (PJM DA bids
  close 11:00; shiny uses -30 min). PJM forecast tables keep MANY vintages: always take
  `row_number() over (partition by dt, hr[, key] order by forecast_time desc) = 1` with
  `forecast_time <= timestamp(date_sub(dt, interval 1 day), '10:30:00')` (see `research/pjm/collect_extra.R`).
  `forecast_time` is market-local (EPT). (SPP tables, by contrast, store one D-1 ~09:00 vintage.)
  **Exception (user decision, 2026-10-23):** PJM ISO forecasts (`pjm_load_forecast`, ISO wind / solar) may
  use vintages up to **D-1 11:00 EPT**, the day-ahead bid deadline (10:00 CT). The ISO load forecast is
  usually stamped ~10:47 EPT, which is fair under this rule.
- **Context leak for load targets:** walk-forward context must not contain D-1 actual load (unknown at the
  cutoff). Use `scripts/kumo_forecast.py --context-gap-hours 24`; actual-load features only as lags >= 48h.
  Price targets don't need the gap (D-1 DA prices publish on D-2).
- **Evaluation convention:** score the median (`q500`). Report full 2025 ("ALL") and HOLD = the 243
  days of 2025 whose day number (from 2025-01-01) is not a multiple of 3; paired day-block bootstrap of
  the daily MAE difference (+- SE, win rate). Templates: `research/miso/compare_prod.py`,
  `research/spp_load/compare.py`, `research/pjm/analyze_blend.py`.
- **Compute:** box `ssh tiostech@10.0.92.46` (A10G, compute-saturated; ~6 s per walk-forward day at
  est8 / 10k context / ~60 features). Repo `~/tios-gpu-research` (branch `pjm-feature-engineering`;
  `git pull` there after pushing). Run jobs sequentially; launch via a script file +
  `nohup bash script.sh > /dev/null 2>&1 < /dev/null &` (piping the script over ssh works when scp
  doesn't). Never `pkill -f` a pattern that also matches your own ssh command line. Data
  `/scratch/data/<proj>/`, outputs `/scratch/outputs/<proj>/`, logs `/scratch/logs/` (+ `queue.log`).
- **SQL:** read-only on `slave`, through the user's SSH tunnel (127.0.0.1:3309) — ask the user to open it
  (it closes periodically). Helper: `cd ~/dev/tios-quant-shiny && Rscript ~/dev/tios-gpu-research/research/pjm/sql.R "<sql>" [out.parquet]`.
  MySQL 8: `lead` is a reserved word.
- **S3:** the box cannot write S3. Copy outputs locally and `aws s3 sync` from the user's machine only
  when the user asks. Prefix `s3://tios-quantforecasts-rspec/gpu-research/kumo-tabular/`.
- No PRT data (subscription ending). Don't seed-average.
- **Artifact:** private claude.ai Artifact (HTML), Plotly UMD from cdnjs (pinned version), data inlined
  as compact JSON (2 dp). Follow the Artifact tool's design guidance (quickstart first). The earlier
  leaderboard page (https://claude.ai/artifact/S7KiKPorbudFE3CRx1GNc7) shows the house style so far.
- Log results in `research/pjm-kumo-feature-engineering.md` (new section per artifact), commit and push to
  `pjm-feature-engineering`, offer an S3 sync of new outputs.

## Existing assets
- **Kumo original baseline (PJM):** `s3://.../kumo-tabular/pjm-fe/outputs/baseline_v0_daily/` (= box
  `/scratch/outputs/pjm/val_walk_forward_predictions.parquet`). 2025 MAE 4.672 / RMSE 9.49; HOLD
  4.608 / 9.35. Kumo-Tabular large, 8 estimators, seed 0, `--max-context 10000` (last 10k hours,
  ~14 months), daily walk-forward over 2025, data `data/pjm-fe/v0` (= the user's original
  train/val/test; 18 production-RF features + `ts`, `dt`, `hr`, `dt_ordinal`).
- **Production PJM forecast:** `data/pjm_extra/prod_flex_daep.parquet` (`pjm_tios_forecasts`, name
  `flex.daep.20261001`, point forecast, 2025-01-01 onward, stamped D-1 10:30-10:35; user confirmed it is
  published fairly; no bands populated). 2025 MAE 5.426 / RMSE 11.68; HOLD 5.380 / 11.44.
  Error corr with Kumo ~0.82; blending it into Kumo does not help (picked weight 0-0.15).
- **Best Kumo PJM ensemble:** ensC (`.../pjm-fe/outputs/ensC_daily/`, average of v0_ctx20k,
  c1_ctx20k, e16_x9, e16_v8_knn40k): 2025 MAE 4.391 / RMSE 8.86; HOLD 4.371 / 8.93.
- Same result in other markets (2025, "original" Kumo = production RF features + time features):
  MISO 3.385 / 6.86 vs production 3.716 / 8.45; SPP 3.933 / 6.53 vs 4.446 / 7.29
  (`research/miso-kumo.md`, `research/spp-kumo.md`).
- **Custom load forecast (Artifact 2, https://claude.ai/artifact/4TGPBZGrscdy1zawsqqoWM):** Kumo P2 (weather +
  Meteologica inputs, `--context-gap-hours 24`) beats Meteologica ALL by 8% MAE in 2025 (10% on HOLD), but
  swapping it into the energy model (`v0_customload_b`, derived inputs recomputed) gives no price gain
  (HOLD -0.033 +- 0.055). Backpublish `.../pjm-load-fe/outputs/P2_bp_gap24/` (2023-10-01 .. 2025-12-31).
  The user asked to build only the "with recomputed derived inputs" variant.
- **2026 test set (Artifact 3, https://claude.ai/artifact/84nZyh3RiGz6Zq5KhFbcyv):** original baseline on
  2026-01-01 .. 2026-10-08 (`.../pjm-fe/outputs/baseline_v0_test_daily/`; data = /scratch/data/pjm with
  train = train + val, val = test): MAE 9.478 vs production 10.202, win 69%; the late-January cold snap
  (prices to $1,501) is the one week production did better.
