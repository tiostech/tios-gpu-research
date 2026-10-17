# PJM DA energy price — Kumo-Tabular feature engineering

Goal: lower val-set (2025) MAE / RMSE of Kumo-Tabular walk-forward forecasts of
PJM DA energy price (`y` = `da_energy_price`), by feature engineering only.

## Setup (keep fixed unless noted)
- Model: Kumo-Tabular large, 8 estimators, seed 0, `--max-context 10000`,
  `--walk-forward` (daily), target `y`, time col `ts` (hour-beginning, EPT wall clock).
- Splits: train 2017-12-31 .. 2024-12-31, val 2025 (8760 h, 1 missing target on DST).
- Instance: `ssh tiostech@10.0.92.46` (A10G 24GB), repo at /ebsmount/www/tios-gpu-research.
  Full daily walk-forward of baseline took ~35 min.
- Metrics: script prints `mae_median` (MAE of q500) and `rmse_mean`; the R comparison
  in tios-quant-shiny uses RMSE of q500 as well. Report both MAE(q500), RMSE(q500), RMSE(mean).

## Workflow (how to run things)
- Build variants locally: `uv run python research/pjm/make_features.py <variant>` (reads
  repo-root train/val/test parquet = original dataset; writes data/pjm-fe/<variant>/).
  Variant `v0` reproduces the original val exactly (checked).
- Copy to box: `rsync -az data/pjm-fe/ tiostech@10.0.92.46:/scratch/data/pjm-fe/`
- Box repo is checked out on branch `pjm-feature-engineering`; `git pull` there after pushing.
- Run sequentially (g5.xlarge = 4 vCPU / 15 GB RAM; concurrent runs are CPU-bound and
  slower in total): `nohup research/pjm/queue.sh "v1_price_lags 7" "v2 1 v2_daily" < /dev/null &`
  Logs: /scratch/logs/<name>_<step>d.log and /scratch/logs/queue.log.
- Tabulate: `python3 research/pjm/results.py` on the box.
- `uv` is only on PATH in login shells on the box (scripts export it).
- ssh + background job: redirect stdin (`< /dev/null`) or the ssh call hangs.
- DON'T screen with `--step-days 7`: stale context biases it (v0: MAE 5.82 at 7d vs 4.67
  daily) and it flatters price-lag features. Use `--eval-every 3` instead: exact daily
  walk-forward forecasting only every 3rd day (~12 min). Since each day's forecast depends only
  on prior context (fixed seed), the full daily baseline restricted to the same timestamps is an
  exact paired baseline; results.py prints dMAE / dRMSE vs it.
- DB: MySQL via user's SSH tunnel on 127.0.0.1:3309 (`dbset("slave")`); ask user to open it.
  tios-docs rule: read-only queries on slave only, ask before querying.

## Data source
- Original dataset columns = subset of `PjmFlexDaep20261001$collect_data()` in
  ~/dev/tios-quant-shiny/lib/trader-forecasting/pjm-flex-daep-20261001.R
  (call via `PjmFlexDaep20261001$new()$do_collect_data(start, end)` after sourcing
  includes/common_include.R, `dbset("slave")`).
- Per that file, features are fair *forecasts* (Meteologica load/wind/solar, ISO outage fc,
  stress indices, fuel prices) for both train and forecast rows -> low leakage risk.

## Leakage rules (day-ahead: forecast for day D is made before the D-1 morning bid deadline)
- DA price of D-1 is known (cleared on D-2 afternoon) -> `y` lags >= 24h OK.
- RT price: D-1 is only partially known at the deadline -> use RT lags >= 48h
  (or daily aggregates of D-2 and earlier).
- West Hub ICE next-day price: fair as of deadline in `ice_price_computor` (Tue-Fri only).
- West Hub DA congestion: lag >= 24h. `west_congestion_fc`: a forecast, OK.
- Anything derived from `y` (heat_rate, spikes) must be lagged >= 24h.

## Baselines (2025 val, same 8759 scored hours)
| run | MAE(q500) | RMSE(q500) | RMSE(mean) | notes |
|---|---|---|---|---|
| flex.daep.20261001 (production) | 5.43 | 11.68 | - | from pjm_tios_forecasts |
| Kumo walk-forward, original features | 4.67 | 9.49 | 9.75 | user GPU run 2026-10-16 |
| Timestral-209M + covariates, 32k ctx | 4.88 | - | 10.18 | user GPU run |

## Experiment log
(newest last; one line per run: variant, features changed, metrics, takeaway)
- 2026-10-17 v0 @ step 7d: MAE 5.82, RMSE(med) 12.17 -> showed 7d screening is biased; abandoned.
- Screening = `--eval-every 3` (122 days, 2928 h), paired vs daily baseline; day-block bootstrap
  SE of dMAE ~0.11 -> only effects >~0.25 are detectable in a screen; confirm with full daily runs.
- v0 every3: MAE 4.799 = baseline subsample exactly (dMAE 0.000) -> paired method validated.
- v1_price_lags every3: MAE 4.892 (dMAE +0.093 +- 0.110), dRMSE(med) +0.30 -> price lags do NOT
  help (daily-refreshed context already carries recent prices); slight harm, within noise.
- Raw pull (data/pjm_raw.parquet) matches original dataset exactly on 2025. West Hub ICE next-day
  price: Tue-Fri only (55% of 2025 hours), corr 0.966 with DA on-peak mean -> strong anchor.

## Ideas backlog
- y lags 24/48/168, same-hour-yesterday, daily mean/max of D-1, rolling 7d mean
- lagged heat rate (y / smoothed_gas) lag 24/168
- RT lags 48h+, D-2 RT-DA spread, spike counts
- West Hub ICE price (anchor), west_congestion_fc, west DA congestion lag 24
- pjm_thermal_capacity_all, solarduckcurve_lag1_diff_pjm (collected but unused)
- net load = load - wind - solar; net load ramps; reserve margin proxies
- calendar: holidays, weekday/weekend flags
- drop redundant (dt, dt_ordinal, hr) features
- max-context / num-estimators tuning (secondary)
