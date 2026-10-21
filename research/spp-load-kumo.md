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
| L1_extra_gap24 | 130.8 / 182.9 | 129.7 | +2.4 +- 4.6 (par) |
| L1_gap24 + Meteologica blend (w_met 0.55) | | 119.7 | +12.4 +- 2.3 (win .67) |
| **L2_with_meteo_gap24** (Meteologica as a feature) | **121.8 / 172.8** (MAPE 2.63%) | **121.2** | **+10.9 +- 3.2 (win .62)** |
| L2_gap24 + Meteologica blend (w_met 0.30) | | 119.4 | +12.7 +- 2.4 (only +1.8 +- 1.1 vs L2) |
- The whole apparent L0 edge came from the leak (worth ~15 MW MAE, most in HB 00-05: 75.6 vs 101.4).
- Leak-free, Kumo with the branch features is on par with Meteologica, but their errors differ enough
  that a 60/40 Meteologica/Kumo blend beats Meteologica by ~11 MW (8%) on held-out days.
- L2 (Kumo with Meteologica as an input) beats Meteologica on its own: -7.5% MAE on full 2025, better in
  every season (DJF 113 vs 128, SON 120 vs 135; JJA 163 vs 164 ~tie) and every hour block. Further
  blending with Meteologica adds little (+1.8 MW), i.e. L2 already absorbs what Meteologica knows.
- L1 extras (OKGE own lags >= 48h, station detail, CDD/HDD) are worth ~3 MW over L0: small.
- All Kumo runs and Meteologica under-forecast on average (y - fc +15 to +22 MW).

# SPP ALL hourly load — Kumo vs Meteologica ALL (2026-10-21)
Target `spp_hourly_load` zone ALL; benchmark Meteologica ALL (D-1 ~09:20 vintage). All runs
`--context-gap-hours 24`. Inputs: `research/spp_load/collect_all.R` (Maxar weather for the 26 cities of
`feelslike_spp_popweighted` in ep-forecast.daep-20260701.spp.json, Meteologica zonal forecasts; vintage
audit 2018-2025: 0 rows issued after D-1 09:30) + collect.R. Features `research/spp_load/make_features_all.py`:
- A0_weather (66): pop-weighted feels-like (config weights) + lags / rolling / daily stats / CDD/HDD,
  per-city feels-like for 12 SPP cities, temp / dew / cloud for 6 largest, OKC/Wichita wind, calendar,
  daylight, SPP ALL actual load lags >= 48h.
- A1_iso (70): A0 + ISO load forecast (level, shape, d24) + Meteologica SPS (note: a vendor forecast,
  so A1 is not vendor-free).
- A2_with_meteo (72): A1 + Meteologica ALL (level, shape). (Meteologica ALL = exact sum of its zones.)
| 2025 | ALL MAE / RMSE / MAPE | bias (y - fc) | HOLD gain vs Meteologica |
|---|---|---|---|
| Meteologica ALL | 547.0 / 702.4 / 1.58% | +236 | |
| ISO forecast alone | 547.1 / 749.3 | +179 | |
| A0_weather_gap24 | 621.7 / 800.9 / 1.78% | +279 | -77.1 +- 21.8 |
| A1_iso_gap24 | 468.6 / 638.0 / 1.34% | +27 | +72.6 +- 21.4 (win .60) |
| **A2_with_meteo_gap24** | **430.2 / 588.4 / 1.23%** | +78 | **+108.6 +- 16.4 (win .71)** |
| A1 + 35% Meteologica blend | HOLD 437.4 / 588.3 | | +108.0 +- 15.3 |
| A2 + 5% Meteologica blend | HOLD 435.4 | | +1.5 +- 1.0 vs A2 (nothing left) |
- A2: -21% MAE / -16% RMSE vs Meteologica; better in every season (DJF 438 vs 632, SON 318 vs 507,
  MAM 344 vs 409; JJA 620 vs 642) and every hour block.
- A0 (weather only) is worse: load growth puts 2025 levels above much of the 10k-hour context, and
  Kumo extrapolates only reluctantly (bias +279; see the PJM price log for a check of how far its forecasts go beyond the context range). The ISO forecast supplies the current level (A1 bias +27).
- Everyone under-forecasts 2025 (load growth). Next ideas: A1 without SPS (clean ISO-only
  comparison); relative target `--target-transform ratio:all_samehour_7d_mean` to remove the
  growth trend.
