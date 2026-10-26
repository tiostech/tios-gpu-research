# PJM DA energy price — Kumo-Tabular feature engineering

Goal: lower val-set (2025) MAE / RMSE of Kumo-Tabular walk-forward forecasts of
PJM DA energy price (`y` = `da_energy_price`), by feature engineering only.

## Setup (keep fixed unless noted)
- Model: Kumo-Tabular large, 8 estimators, seed 0, `--max-context 10000`,
  `--walk-forward` (daily), target `y`, time col `ts` (hour-beginning, EPT wall clock).
- Splits: train 2017-12-31 .. 2024-12-31, val 2025 (8760 h, 1 missing target on DST).
- Instance: `ssh tiostech@10.0.92.46` (A10G 24GB), repo at /ebsmount/www/tios-gpu-research.
  Full daily walk-forward of baseline took ~35 min. GPU is compute-saturated during runs
  (100% util, ~257/300 W, fp16 autocast already on; CPU/disk idle) -> runtime scales with
  context rows x estimators x features; only a bigger GPU or fewer est/rows speeds it up.
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
- Target `y` = `da_energy_price` = PJM DA *system energy* component (not West Hub LMP): shiny's
  `da_energy_prices()` (lib/trader-forecasting/tools.R) reads `pjm_energy_prices.da_energy_price`,
  falling back to West Hub `lmp - congestion - loss` for missing days. Same for MISO.
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

## Current best (full daily 2025 walk-forward; HOLD = 243 days never used for selection)
| model | MAE all | RMSE all | MAE HOLD | RMSE HOLD |
|---|---|---|---|---|
| original Kumo baseline (v0, est8, seed 0) | 4.672 | 9.49 | 4.608 | 9.35 |
| e16_x9 single (v8 + lean weather + forced outages + load-ens spread) | 4.462 | 9.13 | 4.442 | 9.06 |
| ens3 = mean(v0_ctx20k, c1_ctx20k, v8) | 4.459 | 8.86 | 4.452 | 8.93 |
| ensB = mean(v0_ctx20k, c1_ctx20k, e16_x9) | 4.421 | 8.82 | 4.404 | 8.81 |
| ensC = ensB + e16_v8_knn40k | 4.391 | 8.86 | 4.371 | 8.93 |

## Methodology / consistency (checked 2026-10-20)
- Every run uses the same daily walk-forward as the original baseline: forecast day D from context
  before D, append D's actuals, step to D+1 (scripts/kumo_forecast.py --walk-forward).
- `--eval-every 3` runs are the same daily walk-forward but only *forecast* every 3rd day; their
  forecasts are bit-identical to a full daily run on those days. Verified: v0 rerun on current code
  (v0_repro_check) == the user's original daily baseline on all 2928 scored hours (max |diff| 0.0).
- Screen numbers (every-3rd day, baseline 4.799) and full-year numbers (baseline 4.672) are on
  different day sets: only compare within a section. Final claims use full daily runs + HOLD days.
- Settings that vary by run (estimators 8/16, context 10k/20k, recent/knn) are in the run name
  and val_*_metrics.json args. The original baseline (est8 seed 0) is ~0.08 MAE luckier than a
  typical seed. 7-day-step runs (first day only) were biased and abandoned.

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
  Predictions: box /scratch/outputs/pjm-fe/ens3_daily (+ members), copied to local outputs/pjm-fe/.
  Box instance role has NO s3:PutObject (AccessDenied on pjm-fe/ and pjm/outputs/) -> uploads must be
  done by the user; feature data is reproducible via make_features.py.
- Current best = ENSEMBLE3. Next ideas: more diverse members (different context sizes / feature
  sets / seeds per member), weight members, re-screen features as ensemble members not alone.
- Config pull (data/pjm_config.parquet, 68 cols): regional PJM load/wind/solar, gas hubs, coal
  basins, thermal capacity, neighbor (MISO/SPP/NYISO/ISONE/TVA/SOCO) fundamentals.
- Raw pull (data/pjm_raw.parquet) matches original dataset exactly on 2025. West Hub ICE next-day
  price: Tue-Fri only (55% of 2025 hours), corr 0.966 with DA on-peak mean -> strong anchor.
- Round 2 (2026-10-19, every3; dMAE vs v0 seed0 = 4.799):
  SEED CORRECTION: v0 seed1 4.895, v0 est16 seed0 4.873, est16 seed1 4.894 -> v0 seed0 was lucky;
  a typical v0 is ~4.88. est16 cuts seed spread 0.10 -> 0.02 (no accuracy gain) -> screen with
  --num-estimators 16 from now on, baseline = mean of e_est16 seeds 0/1.
  Target transforms (on v0): asinh:20 4.855, log 4.864, logratio:smoothed_gas 4.834 (RMSE 9.20 but
  excl5 +0.29, erratic) -> no. v8 diff:y_lag24 4.682 vs v8 4.587 -> no.
  Context: knn40k 4.719 (RMSE 9.23), mix40k 4.738 -> modest gain (~-0.15 vs typical v0).
  New SQL data on v0: x1 outage split 4.812, x2 load/wind ens spread 4.800 (RMSE 9.49), x3 Maxar
  weather 4.765 (excl5 -0.107, win 0.55, but RMSE 10.33), x4 op summary 4.864,
  x5 all four 4.635 (RMSE 9.48) -> together ~-0.25 vs typical v0.
  ISO load forecast dropped (stamped ~10:47 EPT > cutoff). PRT excluded (subscription ending).
- Round 3 (est16, every3; dMAE vs typical v0 = mean of 2 est16 seeds, MAE 4.879, RMSE ~10.1;
  excl5 = dMAE without the 5 worst baseline days):
  e16_v8 4.573 (-0.31+-0.15, excl5 -0.10, RMSE 8.92) | e16_x7 v8+weather 4.520 (-0.36, -0.18, 9.47)
  e16_x8 v8+lean weather 4.534 (-0.35, -0.18, 9.28, win .56) | e16_x9 x8+forced out+load ens 4.500
  (-0.38+-0.18, -0.20, 9.26, win .57) = BEST SINGLE | e16_x6 v8+all extra 4.557 (-0.32, -0.21, 10.21)
  e16_x5 v0+all extra 4.609 | e16_x6_knn40k 4.699 (10.37) | e16_v8_knn40k 4.592 (8.95) | e16_x6_ctx20k 4.561
  ICE (all Mon-Fri incl. Monday contracts traded Fri, NULL weekends; + west congestion fc and D-1 DA
  congestion as separate features): e16_i2 v8+ICE 4.616 (RMSE 9.53), e16_i1 x9+ICE 4.547 (9.66) -> no.
  Spike days dominate: 2025-01-22, 06-24, 07-30 decide most RMSE differences (x6 doubled 07-30 error).
- e16_x7_daily (full 2025, est16): MAE 4.481 / RMSE 9.19 (orig baseline 4.672 / 9.49); holdout 243 days
  4.461 vs 4.608, RMSE 9.05 vs 9.35, win .54. ens3 holdout 4.452 / 8.93 / win .62 for comparison.

## Synthesis (2026-10-19): what moved the needle and why (thesis)
Bottom line: relative to a *typical* baseline run (~4.88 on screen days; the original seed-0 run was
~0.08 lucky), the best single model (x9/x7) gains ~0.35 MAE / ~7%, about what the first 3-way ensemble got;
but the single-model path took ~40 runs while ensembling got it for free, and every variant sits within
~0.1-0.2 MAE of the others -- inside 1-2 bootstrap SE once spike days are included.
1. Kumo already has the main information. pjm_load / gas / thermal stress / outages / feels-like are
   themselves forecasts that embed weather and system state; new sources (city temps, outage split,
   ensemble spread, ICE) are largely re-expressions of it. ICE in particular trades minutes before our
   cutoff on the same fundamentals the model sees, and adds nothing over price history + fundamentals.
2. In-context learning predicts by analogy to context rows. Gains come from features that make the
   right rows look similar (regional load/wind/solar -> locational analogs; fundamental deltas ->
   'tightening vs loosening' days), not from features with marginal correlation to price.
3. Column dilution. Columns are attended to roughly symmetrically and the readout has a fixed width, so
   every extra column weakens the few that matter on spike days. Hence the recurring pattern: more
   features -> better typical days, worse spikes (x6, i1, c3 neighbors, x7 vs x8).
4. Spike days are not forecastable from day-ahead fundamentals alone (scarcity bidding, unit trips,
   gas-pipeline constraints in real time); with only a handful of analogs in 10k-20k context the median
   shrinks toward normal. These ~3 days drive most of RMSE and most of the noise in every comparison.
5. Model-side knobs: target transforms change little because Kumo standardizes the target and the
   median is roughly transform-equivariant; kNN context helps a weak feature set (v0) but not v8, whose
   price-history features already supply the analog information; more estimators reduce seed variance
   (0.10 -> 0.02) but not bias.
6. Why ensembling works: members built on different feature sets / contexts retrieve different analogs,
   so their errors on spike days are only weakly correlated; averaging quantiles cancels member-specific
   misses (bagging-like variance reduction over the ICL estimator), which no single feature set does.
Drivers summary (what kind of change moved results):
| type of change | effect | examples |
|---|---|---|
| features giving locational / regime analogs | helps typical days | c1 regional, weather aggregates (x8) |
| relative-change features (deltas vs yesterday) | helps spike days / RMSE | v2-v5 fundamental deltas, v8 |
| more columns restating existing info | neutral, dilutes spikes | ICE, op summary, full weather, outage split alone |
| raw cross-market magnitudes | hurts | c3 neighbors (MISO/SPP/NYISO/ISONE raw) |
| price lags alone | hurts slightly | v1 (context already carries recent prices) |
| target transforms | neutral alone; mild ensemble diversity at best | asinh, log, logratio gas |
| context size / kNN | helps weak feature sets, noisy; diversity in ensembles | ctx20k, knn40k |
| estimators 8 -> 16 | lower seed noise, no accuracy gain | e_est16 |
| ensembling diverse members | most reliable gain | ens3, ensB, ensC |

Lesson for future screens: use est16, a 2-seed baseline, report excl5 + win rate, and treat < ~0.15
MAE as noise unless it survives a full daily holdout.

## Ensembling round (queued 2026-10-19 evening)
Offline on screen days (vs ens3 = v0_ctx20k + c1_ctx20k + v8, MAE 4.474 / RMSE 8.73):
  swap v8 -> e16_x9: 4.455 / 8.85; + e16_v8_knn40k: 4.429 / 8.70 (win .65);
  c1_ctx20k + x9 + v8knn + logratio-target: 4.437 / 8.59; mean of 14 runs 4.468 / 8.90.
  Greedy forward selection on half the days overfits (spike days split unevenly) but repeatedly picks
  the logratio:smoothed_gas target run -> target-transform diversity is a useful member axis.
Full daily members queued (est16): e16_x9_daily, e16_v8_knn40k_daily, e16_x9_logratio_daily,
  e16_c1_ctx20k_daily (+ existing v0_ctx20k_daily, c1_ctx20k_daily, v8_daily, e16_x7_daily).
Plan: evaluate on the 243 holdout days only (members/weights chosen on screen days): equal-weight
  candidates, median vs mean, and walk-forward performance weighting (trailing-N-day member MAE).
RESULTS (2026-10-20, full daily 2025; HOLD = 243 days never used for selection; orig baseline HOLD
  4.608 / 9.35; vs-ens3 diffs are paired day-bootstrap):
  singles HOLD: e16_x9_daily 4.442 / 9.06 (best single; ~= ens3 4.452 / 8.93); e16_x7 4.461; c1_ctx20k
  4.504; e16_c1_ctx20k 4.563 (est8 c1_ctx20k was a lucky seed); v8 4.561; v8_knn40k 4.679 / 10.12;
  x9_logratio 4.776 / 10.47 (weak alone).
  B = v0_ctx20k + c1_ctx20k + e16_x9: HOLD 4.404 / 8.81 (-0.048+-0.014 vs ens3, win .62) -> most
      reliable gain; best holdout RMSE.
  C = B + e16_v8_knn40k: HOLD 4.371 / 8.93 (-0.080+-0.038, win .58) -> best holdout MAE (-5.1% vs orig).
  E = C + x9_logratio: 4.396 / 9.01; D (no v0_ctx20k): 4.404 / 9.16 -> logratio member does not hold up.
  all-8 mean 4.387 / 8.96; all-8 median 4.380 / 8.88.
  Walk-forward inverse-MAE weighting (trailing 30/60 days, p=1/2): 4.386-4.389 -> no gain over equal.
  Takeaway: better members (x9) and context diversity (knn) add ~0.05-0.08 over ens3; weighting,
  medians and more members do not. Predictions: box /scratch/outputs/pjm-fe/ens{B,C}_daily.

## Self-correction layer (2026-10-20) -- fail-fast verdict: NO GAIN
Design: base = x9 (est8, ctx10k, seed 0; best accuracy/compute) walk-forward daily over 2024-2025
  (`resplit.py x9_v8_lean_extra x9_wf2y`, run base_x9_wf2y, ~85 min; 2-yr MAE 3.636 / RMSE 7.53).
  Correction = Kumo-Tabular (same settings) on target r = y - base_q500, context = 2024 out-of-sample
  residuals, walk-forward daily over 2025 (appends realised residuals). 38 features
  (`make_correction.py`): base state (q500, 80% band, IQR, skew, mean-median, q500 vs y_lag24 / 7-day),
  lagged residuals (r lag24/48, same-hour 7d, D-1 mean/absmean/max/min, D-2, 7-day), and 24h/168h
  changes in load, wind, solar, outages, cushion, gas %, temps, MISO stress, load-ensemble std.
  Corrected = base_q500 + lam * r_q500; lam picked on screen days (`eval_correction.py`). Run ~33 min.
Results 2025 (base x9 est8 HOLD 4.442 / 9.13, same as e16_x9):
| | ALL MAE / RMSE | SCREEN | HOLD |
|---|---|---|---|
| base | 4.470 / 9.25 | 4.526 / 9.48 | 4.442 / 9.13 |
| lam=1 | 4.519 / 9.18 | 4.525 / 9.04 | 4.515 / 9.25 |
| lam=0.5 (chosen) | 4.445 / 9.12 | 4.488 / 9.17 | 4.424 / 9.09 |
  HOLD daily gain lam=0.5: +0.018 +- 0.030 (win .51) -> noise; lam=1: -0.073 +- 0.062 (win .45) -> hurts.
Why it fails (consistent with the pre-run diagnostics):
- Base residuals are nearly white: corr(r, r_lag24) -0.09, day-mean lag-1 -0.07, 2025 mean bias +0.13.
  The walk-forward base already conditions on yesterday's prices, so there is no persistent error to learn.
- Predicted corrections are small and mostly negative (median -0.2, sd 2.6 vs residual sd 9.25; corr with
  the realised residual 0.18). The only learnable signal is "wide base band -> actual lands below
  median" (spike-shaped upper tail); the layer shrinks forecasts on those hours, which helps a few
  spike-prone screen days (screen RMSE -0.45) but not the holdout.
- One year (8.8k rows) of residual context with ~3 spike days gives the ICL model too few analogs for
  the error regimes that matter; the correction for typical hours is ~0.
- Contrast with markets where this worked: there the base was presumably a static (non-walk-forward)
  model with drift / regime bias. Kumo walk-forward is itself an online, in-context learner, so the
  'learn from recent errors' role is already played by the appended context.
Not pursued further. A correction on an *ensemble* (ensC) base would face the same white residuals.

## Random-subspace (mtry) members (2026-10-20) -- fail-fast verdict: NO GAIN
Design: `--feature-frac 0.6 --feature-seed 1..8` on x9 (39 of 65 features per member, fixed per run;
  calendar from ts always kept), est4, `--eval-every 3`; members m_x9_f60_s{1..8}, ~6 min each.
  Scored on screen days (script: research/pjm/eval_mtry.py, run on the box; quantile-average = mean of q500s).
Results (screen days; e16_x9 4.500 / 9.26; ensC 4.429 / 8.70):
  members: 4.54 - 5.19 MAE (median ~4.77), RMSE 9.2 - 12.3 -> every member but one worse than full x9.
  mtry mean of 2 / 4 / 8: 4.891 / 4.760 / 4.672 (RMSE 10.05) -> vs e16_x9 -0.172 +- 0.065 (win .46).
  ensC with x9 swapped for mtry8: 4.477 / 8.88 (-0.048 +- 0.016 vs ensC); ensC + mtry8: 4.454 (-0.025).
  Member errors correlate 0.94 pairwise (0.99 between mtry8 and x9).
Why: x9's 65 columns are already a curated set in which a handful (price history, gas, load/net-load
  deltas, thermal stress) carry most of the signal; a 60% draw drops at least one of them in most
  members (spike RMSE up to 12.3), and the members still retrieve nearly the same analogs, so the errors
  stay ~0.94 correlated and averaging recovers only part of the loss. Feature bagging pays off for weak,
  high-variance learners with many redundant columns (trees); an ICL model with few strong columns gets
  bias, not diversity. Useful diversity here comes from *different information or context* (c1 regional,
  ctx20k, kNN), not random column removal. A milder frac (0.8-0.9) would at best approach x9.

## Error breakdown + production-forecast blend (2026-10-20) -- `research/pjm/analyze_blend.py`
Production DAFC: `pjm_tios_forecasts` name 'flex.daep.20261001' (point forecast only, no bands populated;
  2025-01-01 onward, stamped 10:30-10:35 D-1; user confirms it is published fairly) ->
  data/pjm_extra/prod_flex_daep.parquet. No pre-2025 history, so it cannot be a Kumo feature (no context).
Where ensC's 2025 error lives (MAE of q500):
  season: DJF 5.04, JJA 4.74, SON 4.22, MAM 3.58 (prod 6.47 / 5.79 / 4.72 / 4.74).
  hour block (HB): 16-20 = 7.66 MAE and 36% of all abs error in 21% of hours; 00-05 2.49; 06-09 4.45.
  supply-cushion ratio: tightest 10% of hours = 13.0 MAE, 30% of abs error; loosest 10% = 1.4.
  concentration: top 3 days = 6% of abs error / 26% of SSE; top 10 = 16% / 54%; top 30 = 29% / 69%.
  top days come in episodes (Jan 18-22 cold snap, Jun 24-25, Jul 28-30 heat): under-forecast on the
  first spike day(s), over-forecast after the peak passes (Jan 21-22, Jun 25, Jul 30) -> anchored on
  recent prices. On average the effect is small (bias after a >8 under-forecast day: -2.4, n=11;
  corr(bias, D-1 price jump) -0.05), so it is an episode-level, not a systematic, error.
  Overall bias ~0 (+0.05); spike misses split 53/47 under/over.
Production forecast vs Kumo (HOLD 243 days): prod 5.380 / 11.44 (ALL 5.426 / 11.68) -- worse than the
  original Kumo baseline (4.608 / 9.35) in every season and hour block. Error corr with ensC 0.82.
  Blend w*prod + (1-w)*kumo, w picked on SCREEN: w=0 for ensC/ensB/ens3; x9 w=0.05 (+0.014 +- 0.009);
  50/50 blends lose 0.11-0.24. Prod beat ensC on a few spike days (Jan 22 19.7 vs 29.5, Jun 25 8.3 vs
  17.5) but was far worse on others (Jan 20 67 vs 23, Jun 24 43 vs 24) -> no usable diversity.
Verdict: no gain from the production blend. The breakdown says the remaining error is evening-peak,
  tight-cushion hours on ~10-30 days -> next work should target those (day-level model with full-history
  context; stress-enriched context selection), not more columns.

## Day-level model (#2) and analog context (#3) (2026-10-20) -- verdict: NO SIGNIFICANT GAIN
#2 Day-level models (`make_daily.py`; one row per day, 89 features = daily means of x9 + min/max/peak-mean
  of stress drivers; context = all 2018-2024 days; est16, daily walk-forward 2025; `eval_daylevel.py`):
  d1_x9_mean (target = day mean): HOLD-day MAE of the day level 3.382 vs ensC's implied level 3.341
    (50/50 3.242; error corr 0.87). Worst-30-day level MAE 12.9 vs ensC 14.1 (better Jan 18/21, Jun 24-25;
    worse Jul 29: 105 vs actual 164, ensC 127).
  d2_x9_peak (target = mean of HB 16-20): 7.071 vs ensC-implied 6.729 (50/50 6.659).
  Re-levelling ensC hourly q500 toward the day models (lam on SCREEN; A = all hours by day-mean gap,
    B = HB16-20 by peak gap, C = B + off-peak hours so the day mean matches A):
    ensC C lam=0.4: HOLD 4.326 / 9.08 (+0.045 +- 0.059 vs 4.371 / 8.93, win .59); A lam=0.3: 4.336 / 8.94.
    x9 C lam=0.4: 4.365 / 9.13 (+0.077 +- 0.060 vs 4.442 / 9.06). lam=1 loses ~0.22 in every scheme.
  -> Full-history context does not make the day level more accurate on its own; it is a mildly
     diversifying level signal (corr .87), worth ~+0.05 MAE at the cost of RMSE, inside noise.
#3 Analog context (`--context-select analog`: recent (1-f)*10k rows + f*10k older rows nearest to the
  query day on supply_cushion_ratio, thermal_stress_all_pjm, net_load, feelslike, smoothed_gas + hour;
  pool = all history from 2018; x9, est16, every3; vs e16_x9 4.500 / 9.26):
  a1 f=0.3: 4.636 / 9.85 (-0.136 +- 0.097); top-10 spike days 16.71 vs 15.87; other days 3.558 vs 3.485.
  a2 f=0.5: 4.530 / 9.17 (-0.030 +- 0.093); top-10 spike days 16.34 vs 15.87; other days 3.476 vs 3.485.
  Error corr with x9 0.97-0.98. In ensC: + a2 4.410 / 8.71 (+0.019 +- 0.026); x9 -> a2 4.425 / 8.67 (+0.004).
  -> Giving the model older stressed analogs does not fix spike days (slightly worse): older analogs
     (e.g. 2022 high-gas winter for Jan 22 2025) come from a different price regime (gas level, capacity
     mix), so their prices mislead as much as they inform. kNN-type selection again acts only as weak
     ensemble diversity.
Overall reading: the spike-day turning points are not recoverable from day-ahead fundamentals with more
  or better-chosen history. Both remaining levers (#2 re-level, #3 analog member) are within noise.

### Round 2 (2026-10-19), from tios-docs survey; priority order
Model-side (queued 10-19): target transforms (asinh:20, log, logratio:smoothed_gas, v8 diff:y_lag24),
  context selection knn/mix from a 40k-row pool, 16 estimators (x2 seeds to measure seed noise).
1. Gen outage split: pjm_totalgen_on_outages (planned/maintenance/forced MW, forecast_time) +
   pjm_iir_aggregated_outage_forecasts other fuel keys (filter is_unfair).
2. Weather detail: pjm_maxar_weather_forecasts raw temp/dewpoint/wind by city (pjm_locations),
   cross-city spread, 4h feels-like drop, HDD/CDD; Meteologica scenario load/wind/solar std_dev
   (forecast uncertainty). Check vintage semantics (data_sources.type) before use.
3. Load forecast disagreement: Meteologica vs ISO (pjm_load_forecast, forecast_time vintages) and
   revisions; pjm_op_summaries (peak load fc, operating reserve, unscheduled steam).
4. Vendor price forecast: pjm_prt_lmp_forecasts West Hub DA (as-of via *_forecast_time) as a feature.
5. ICE strips available every day (nextweek 66 / balweek 67) to cover non-ICE days (Sat-Mon).
6. Congestion/constraints: pjm_hourly_congestion_forecasts for other hubs/zones; high-kV transmission
   outage counts as-of from pjm_transmission_outages_versions.
7. Cross-market: MISO Indy Hub DA lags (fair), miso_operating_margin_forecasts, MISO weighted feels-like.
8. "Joint" MISO: Kumo-Tabular is single-target (supports_multi_target=False). Options: stack the
   production MISO flex.daep forecast (fair at ~09:05 CT < PJM 09:30 CT cutoff) but live history only
   from 2025-11 -> needs run_forecast_backtest for 2017-2025; or a Kumo stage-1 MISO forecast made
   walk-forward OOS over all history (expensive); or row-stack PJM+MISO rows with a market flag.
Timing: PJM cutoff 09:30 CT (deadline 10:00 CT); MISO data in fixed EST; forecast_time is market-local.

### Round 1
- y lags 24/48/168, same-hour-yesterday, daily mean/max of D-1, rolling 7d mean
- lagged heat rate (y / smoothed_gas) lag 24/168
- RT lags 48h+, D-2 RT-DA spread, spike counts
- West Hub ICE price (anchor), west_congestion_fc, west DA congestion lag 24
- pjm_thermal_capacity_all, solarduckcurve_lag1_diff_pjm (collected but unused)
- net load = load - wind - solar; net load ramps; reserve margin proxies
- calendar: holidays, weekday/weekend flags
- drop redundant (dt, dt_ordinal, hr) features
- max-context / num-estimators tuning (secondary)

## Stakeholder artifact 1: production vs Kumo-Tabular (2026-10-21)
Page: https://claude.ai/artifact/SPd1KUEWoAyqLpFArYYLJw (private). Built by `research/pjm/artifact1_data.py`
(+ `artifact1_template.html`; output outputs/pjm-fe/artifact1/) from existing predictions; no GPU, no SQL.
Production flex.daep.20261001 vs original Kumo baseline (same 17 RF inputs + smoothed gas + time columns):
  HOLD 5.380 / 11.44 -> 4.608 / 9.35 (MAE -14.3%, RMSE -18.3%), daily gain +0.772 +- 0.176, wins 70% of days;
  ALL 5.426 / 11.68 -> 4.672 / 9.49 (-13.9%), +0.754 +- 0.146, win 69%. ensC (reference) HOLD 4.371 / 8.93, win 74%.
  Kumo lower MAE in every season (-8% SON .. -17.5% MAM) and hour block (-5% HB21-23 .. -23% HB06-09).
  Example day 2025-01-20 (cold snap): prod daily MAE 67.0 vs Kumo 38.8; Kumo hit the 07:00 spike (257 vs 261 actual).
  Prod better on 2025-01-21/22/18 (Kumo stayed high after the peak). Blend: err corr 0.85 (ensC 0.82);
  picked prod weight 0.15 for baseline -> HOLD 4.591 vs 4.608 (noise), 0 for ensC; 50/50 worse (4.720).
Production model structure (pjm-flex-daep-20261001.R): rf730d (ranger 500 trees, 17 features, last 730 d) +
  scaled method (lm heat rate ~ poly5(thermal stress), 730 d; D-1 same-hour price x ratio), blended hourly by a
  ranger blender (25 features, 730 d); ICE RF and RT-despike RF computed with alloc 0.
Artifact 1 v2 (2026-10-21): (1) "cannot extrapolate" corrected. Check (artifact1_data.py, baseline v0 vs the max
  price in each day's 10k-hour context): median q500 above the context max on 3 days of 2025 (2025-01-21: 365 vs
  308), q950 above it on 11 days (up to 2.4x). 2025 set a new context high on 4 days; the median exceeded the
  old high on 01-20/01-21 but stayed below it on 06-24 (297 vs old high 312, actual 404) and 07-29 (374 vs 404,
  actual 537) -> can extrapolate, but reluctantly (unlike RFs, bounded by the training max).
  (2) Added a "How the 2025 days were used" section: walk-forward everywhere + 122 selection / 243 held-out days;
  only research variants/ensembles were chosen on selection days; baseline and production involve no selection.

## Artifact 2: Kumo PJM ALL load forecast and its effect on the price model (2026-10-21)

Page: https://claude.ai/artifact/4TGPBZGrscdy1zawsqqoWM (private). Scripts in `research/pjm_load/`
(collect.R, make_features.py, make_customload.py, artifact2_data.py, artifact2_template.html).

**Load (target pjm_hourly_load ALL, walk-forward daily, --context-gap-hours 24, actual load only as lags >= 48h).**
Benchmark Meteologica ALL (D-1 ~10:20 EPT vintage = v0's pjm_load). ISO pjm_load_forecast dropped: no fair vintage.
- Screens (every 3rd day of 2025, 122 days): Meteologica 1297 MW MAE; P0 weather/calendar/lags only 1701 (+31%,
  loses, as in SPP); P2 = P0 + Meteologica regional + ALL forecasts 1234.
- P2 backpublish 2023-10-01..2025-12-31 (P2_bp_gap24): 2025 MAE 1116 vs Meteologica 1215 (-8.2%; held-out days
  1057 vs 1174, -10.0%, gain 117 +- 29 MW, 60% of days won). Pre-2025 1049 vs 1148. 80% band coverage 91%.
  Gain in winter/spring/autumn and every hour block; summer equal (1483 vs 1480). Meteologica's monthly biases up
  to ~1 GW (Jan -986, Sep +901) shrink to <= 0.4 GW. Hourly error correlation 0.84.

**Energy: v0_customload_b** = v0 with pjm_load replaced by the P2 q500 from 2023-10-01 on, plus thermal_stress,
solarduckcurve and its lead1 diff recomputed (formulas checked exact). Baseline settings (est8, 10k context, daily).
- 2025 MAE 4.688 vs baseline 4.672 (gain -0.016 +- 0.043); held-out 4.641 vs 4.608 (-0.033 +- 0.055), 51% days
  won. No measurable price gain. 10 worst baseline days: 5 better, 5 worse, not tracking the load error.
- Load-only variant (v0_customload_a) skipped at the user's request.
- Conclusion: the better load forecast doesn't help the price model; ~100 MW on ~100 GW is noise to it, and there
  is no gain in summer, where load drives spikes.

## Artifact 3: production vs original Kumo baseline on the 2026 test set (2026-10-21)
Page: https://claude.ai/artifact/84nZyh3RiGz6Zq5KhFbcyv (private). `research/pjm/artifact3_data.py` + `artifact3_template.html`.
Run baseline_v0_test_daily: original baseline data (/scratch/data/pjm) with train = train + val (through 2025-12-31),
val = test (2026-01-01 .. 2026-10-09 01:00); est8, seed 0, 10k context, daily walk-forward, no other changes.
Scored on 281 complete days (to 2026-10-08). First look at the test split; nothing was selected on it.
- 2026: prod 10.202 / 28.75 -> Kumo 9.478 / 28.06 (MAE -7.1%, RMSE -2.4%), gain +0.724 +- 0.533, win 69%
  (sign test p 3e-10). 2025 for reference: -13.9%, +0.754 +- 0.146, win 69%.
- Kumo better in every month but January (26.58 vs 23.28). Cold snap 2026-01-23..30 (feels-like -8F, prices to
  $1,501 vs context max $537 from 2025-07-29): Kumo under-forecast 24-27 Jan (daily mean 134/242/348 vs actual
  252/350/509; prod 232/276/398) -> reluctant extrapolation again; q900 reached 1,622 on 27 Jan; Kumo beat prod
  on 28-29 Jan. Excluding that week (273 days): 8.36 -> 7.14 (-14.6%), gain +1.22 +- 0.25, win 70%.
- Hour blocks: Kumo better 00-05/06-09/16-20, tie 21-23, slightly worse 10-15 (8.62 vs 8.46). 80% cov 0.86.
- 2026 prices higher/more volatile (mean $64 vs $47), so both MAEs roughly double vs 2025.

## Artifact 4: Kumo PJM wind / solar forecasts and the price model with all custom inputs (2026-10-23)
Page: https://claude.ai/artifact/4GJv29JH9TMnqZpQAnS2T2 (private). Plan `research/plans/pjm-artifact-4-wind-solar-forecasts.md`.
Scripts in `research/pjm_renew/` (collect.R, make_features.py, make_custominputs.py, score.py, artifact4_data.py,
artifact4_template.html, queue.sh). Nothing run on 2026.

**Data / Gate 1.** Targets pjm_windgen_outputs / pjm_solargen_outputs ALL (one row per hour, no revision history ->
lags >= 48h only). ISO forecasts pjm_windgen_output_forecasts ALL (stamped 10:55 EPT), pjm_solargen_output_forecasts
ALL + BTM_ALL (10:00), pjm_load_forecast RTO COMBINED (~10:47): one vintage per hour, all fair under the 11:00 EPT
rule, 100% of 2023-10..2025 days. **Meteologica wind / solar tables also keep one vintage per hour, stamped
10:23-10:59 EPT (after 10:30 on ~70% of 2023-25 hours); v0's pjm_wind / pjm_solar equal it on every hour.** Under the
10:30 rule only ~20% of days had a value, so these use the 11:00 rule too (= what production feeds the price model).
Scenario ensembles and Maxar keep 10:30. Scales match (2025 wind mean actual 3669 / Met 3669 / ISO 3707 MW; solar
daylight Met -5%, ISO -2%). 2025 benchmarks: wind MAE Met 472 / ISO 499; load MAE Met 1215 / ISO 1547 / Kumo P2 1116.

**Wind W2** (69 features: Met ALL + 3 regional + 8 zonal, ISO, ECMWF-ENS/GEFS scenarios, Met recent error, Maxar
wind speed + capped cube at 13 stations, calendar, lags >= 48h, 30-day max capacity proxy). Screen (122 sel days)
478.7 vs Met 485.5. Backpublish W2_bp_gap24 (2023-10-01..2025-12-31): 2025 ALL 468.7 vs Met 472.0 / ISO 499.2
(gain vs Met +3.4 +- 5.0, win 50%); HOLD 463.6 vs 465.3 (+1.7 +- 6.1) -> **tie with Meteologica**; beats ISO
+38 +- 9. Pre-2025 450.5 vs 452.7. Ramps and top-10% stress hours tied. Cov80 0.88.
**Solar S2** (67 features: Met ALL + 3 regional, ISO ALL + BTM, ECMWF-ENS scenario, Met recent error, sun elevation
/ clear-sky at 4 centroids, Maxar sky cover / temp at 14 stations, calendar, lags, same-hour 30-day max; q clipped
at 0; scored on daylight hours = sun up at any centroid). Screen 453.8 vs Met 506.4. Backpublish S2_bp_gap24:
2025 ALL 461.3 vs Met 530.5 / ISO 687.7 (-13.0%, +66 +- 11, win 61%); **HOLD 465.1 vs 542.6 (-14.3%, +74 +- 13)**;
pre-2025 286 vs 369 (-22%). Mostly bias: Met +248 MW low on avg daylight hour, Kumo +33. Ramp error 147 vs 166,
stress hours 450 vs 517. Cov80 0.86.

**Gate 3 perfect foresight: v0_actuals_all** (v0 with actual load / wind / solar from 2023-10-01, thermal stress,
duck curve, its change and solar_lag1_diff recomputed (all formulas exact on v0), + 6 cols actual minus Met / ISO).
Baseline settings. 2025 ALL 4.635 / 9.32 vs 4.672 / 9.49 (gain +0.038 +- 0.050, win 49%); **HOLD 4.606 / 9.18 vs
4.608 / 9.35 (+0.002 +- 0.064, win 47%) -> below 2 SE, so step 6 (v0_custom_all) was skipped** per the plan.
Reading: the DA price already reflects load / wind / solar about as well as perfect knowledge would; better forecasts
of these (and the custom-minus-ISO mis-commitment columns) cannot help this price model. Run order: the perfect-
foresight run was moved ahead of the backpublishes at the user's request (W2_bp restarted).
