# SPP OKGE hourly load — Kumo-Tabular vs Meteologica

Goal: beat the Meteologica OKGE load forecast with Kumo, using features from the
`spp-load-fc-research` branch of tios-quant-shiny (`SppOkgeLoadWnorm20260716`) plus extras.
Kumo-Tabular large, 8 estimators, seed 0, `--max-context 10000`, daily walk-forward over 2025.

## Data and fairness
- `research/spp_load/collect.R` -> data/spp_load/: spp_hourly_load (OKGE target; ALL/GRDA/CSWS),
  spp_meteologica_load_forecasts (OKGE, ALL), spp_load_forecast (ISO, ALL), spp_maxar_weather_forecasts
  (branch stations 13/20/203/205).
- Each forecast table stores ONE vintage per (dt, hr[, key]). Audit 2018-2025: every row issued before
  D-1 09:30 (Meteologica 09:00-09:22, ISO 09:05, Maxar 06:00/09:00); 0 later; no duplicates. So all
  inputs, and the Meteologica benchmark, reflect the same D-1 morning information.
- Actual loads only as lags >= 48h (D-2 is complete at the D-1 cutoff).
- **Context leak found and fixed (2026-10-20):** the walk-forward context for day D included all of
  D-1's actual load, which is mostly unknown at D-1 09:30 (harmless for DA prices, published D-2).
  `kumo_forecast.py --context-gap-hours 24` drops history newer than the end of D-2. All OKGE results
  below use it unless marked LEAKY.
- Variants (`research/spp_load/make_features.py`): L0_branch (43 branch features), L1_extra (+ OKGE load
  lags, per-station feels-like/dew/heat-index/wind-chill/precip, CDD/HDD, Meteologica SPP-ALL, ISO fc
  change; 75), L2_with_meteo (+ Meteologica OKGE forecast and its shape; 77).

## Results 2025 (`research/spp_load/compare.py`; HOLD = 243 days, blend weight picked on SCREEN)
Meteologica OKGE: MAE 131.7 / RMSE 183.4 / MAPE 2.84% (bias y - fc +21.9).
| run | ALL MAE / RMSE | HOLD MAE | gain vs Meteologica (HOLD) |
|---|---|---|---|
| L0_branch LEAKY (D-1 actuals in context) | 118.8 / 169.0 | 119.1 | +13.0 +- 4.7 (not valid) |
| L0_branch_gap24 | 134.1 / 185.1 | 133.3 | -1.2 +- 4.9 (par) |
| L0_gap24 + Meteologica blend (w_met 0.60) | | 121.0 | +11.1 +- 2.3 (win .63) |
- The whole apparent L0 edge came from the leak (worth ~15 MW MAE, most in HB 00-05: 75.6 vs 101.4).
- Leak-free, Kumo with the branch features is on par with Meteologica, but their errors differ enough
  that a 60/40 Meteologica/Kumo blend beats Meteologica by ~11 MW (8%) on held-out days.
