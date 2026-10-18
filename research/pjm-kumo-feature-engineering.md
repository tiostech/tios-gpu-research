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
- Batch 1 (every3, dMAE+-se / dRMSEmed+-se vs baseline):
  v2_lags_deltas -0.063+-0.12 / -0.78+-0.41; v3 (drop dt/dt_ordinal/hr) -0.067 / -0.71;
  v4 (+calendar) -0.099 / -0.80; v5 (+price history) -0.108+-0.14 / -0.92+-0.51.
  -> fundamental deltas (net_load_delta24, thermal_stress_delta24, heat_rate_lag24,
  y_lag24_gas_scaled) drive an RMSE gain (~1.9 se); MAE gains within noise. v6 = deltas only queued.
- Batch 2 (raw pull, each on top of v0): r1_ice +0.016+-0.06 / -0.23+-0.26 (ICE days MAE 5.46->5.42,
  non-ICE days 3.97->4.05: small, not a lever); r2_rt +0.018 / +0.26; r3_cong +0.043 / +0.13 -> no gain.
- Batch 3 (every3, dMAE+-se / dRMSEmed+-se): c1_regional -0.080+-0.12 / +0.02; c2_fuels -0.008 / -0.14;
  c3_neighbors +0.153 / +1.33 (bad); v6_deltas_only -0.053 / -0.55; v7_v5_broad_deltas -0.085 / -0.48;
  v0_ctx20k -0.146+-0.11 / -0.71+-0.36; v5_ctx20k +0.263 / +1.70 (one blow-up day, 2025-01-22 MAE 72).
- KEY: 2025 errors are dominated by a few spike days (01-19, 01-22, 06-24, 07-30). Robust view
  (dMAE excluding the 5 worst baseline days / day win-rate): delta variants v2-v6 are +0.09..+0.12
  WORSE on typical days and win only on spike days; v0_ctx20k +0.04 (spike-only too);
  c1_regional is the only one better on typical days (-0.110, win 0.57).
- Ensembles (mean of q500 across runs, every3): v0_ctx20k+v7 MAE 4.556 (-0.243), RMSE 8.97;
  v0_ctx20k+c1+v7 -0.241, excl5 -0.132, win 0.63; all 9 decent runs -0.185 (no selection).
  -> averaging different feature sets is the biggest lever so far. Caveat: pair selection is in-sample
  and part could be seed-variance reduction -> control: v0 seed 1 averaged with v0 seed 0.
- Batch 4 (every3, dMAE+-se, excl5, RMSEmed):
  v0_seed1 +0.096+-0.054 (!) -> SEED NOISE ~0.1 MAE for a single run; the day-bootstrap SE does not
  include it, so single-run differences < ~0.15 are not trustworthy.
  Context curve v0: 10k 0, 15k -0.042, 20k -0.146, 30k -0.013 -> non-monotonic, 20k likely partly luck.
  v8_v5_regional -0.212+-0.14 (RMSE 8.83); c1_ctx20k -0.221+-0.13 (RMSE 8.97).
  Ensembles: v0+v0_seed1 +0.040 (seed averaging gives nothing) -> ensemble gains come from
  feature-set diversity. Best: v0_ctx20k + c1_ctx20k + v8 MAE 4.474 (-0.325+-0.117, excl5 -0.130,
  RMSE 8.73 vs 9.77); v0_ctx20k+c1_ctx20k+v7 -0.282.
- CONFIRMED (full daily 2025, 8759 h; dMAE+-se vs baseline 4.672 / RMSEmed 9.494):
  v0_ctx20k_daily 4.630 (-0.042+-0.07) RMSE 9.06; c1_ctx20k_daily 4.529 (-0.143+-0.08) RMSE 9.02;
  v8_daily 4.570 (-0.102+-0.12) RMSE 9.23;
  ENSEMBLE3 (mean of the 3 runs' quantiles, research/pjm/ensemble.py) MAE 4.459 (-0.213+-0.08,
  -4.6%), RMSE 8.865 (-6.6%), wins 61% of days, excl5 -0.110, 80% coverage 0.879 (base 0.869).
  Holdout (243 days never used for screening/selection): ENSEMBLE3 4.452 vs 4.608 (-0.157+-0.10),
  RMSE 8.93 vs 9.36, wins 62% of days (~3.7 binomial se) -> real, modest improvement.
  Single runs shrink a lot from screen to holdout (seed/selection noise); the ensemble holds up best.
  Predictions: box /scratch/outputs/pjm-fe/ens3_daily, S3 .../kumo-tabular/pjm-fe/.
- Current best = ENSEMBLE3. Next ideas: more diverse members (different context sizes / feature
  sets / seeds per member), weight members, re-screen features as ensemble members not alone.
- Config pull (data/pjm_config.parquet, 68 cols): regional PJM load/wind/solar, gas hubs, coal
  basins, thermal capacity, neighbor (MISO/SPP/NYISO/ISONE/TVA/SOCO) fundamentals.
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
