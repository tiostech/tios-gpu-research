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
