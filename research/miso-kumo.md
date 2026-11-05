# MISO DA energy price — Kumo-Tabular

Same setup as the PJM work (research/pjm-kumo-feature-engineering.md): Kumo-Tabular large,
8 estimators, seed 0, `--max-context 10000`, daily walk-forward over 2025, target `y` =
`da_energy_price`, time col `ts` (hour-beginning, MISO fixed EST). HOLD = the 2025 days whose
day number is not a multiple of 3 (243 days), as for PJM.

## Data
- Target = MISO DA *system energy* component (`miso_energy_prices.da_energy_price` via shiny's
  `da_energy_prices()`; fallback Indiana Hub `lmp - congestion - loss`), not Indiana Hub LMP.
- `research/miso/collect_original.R` -> `MisoFlexDaep20251101$new()$do_collect_data()` (production
  class in ~/dev/tios-quant-shiny/lib/trader-forecasting/miso-flex-daep-20251101.R; fair
  forecast vintages, 25 min before the bid deadline) -> data/miso/miso_raw.parquet
  (2017-12-01 .. 2026-10-20, 77,904 rows, 24 h every day).
- `research/miso/make_original.py` -> data/miso-fe/m0_original: the 12 regressors of the production
  baseline RF (`lucas_rf_predict`: thermal_stress_all_miso, sdi_macro, miso_load/wind/solar,
  blended_gas_price, illiois_basin, solarduckcurve_all_miso_lead1_diff, solar_all_lead1_diff,
  duck_curve_south_sub_north_sub_central_miso, solar_all_lag1_diff, miso_feelslike) + time features
  (ts, dt, hr, dt_ordinal; the RF has none). train 2018-2024 (61,392), val 2025 (8,760),
  test 2026 (7,032). NaN left as NULL (duck_curve_south... 12% missing in train, others < 7%).
- Production forecast: `miso_tios_forecasts` name 'flex.daep.20251101' (2023-11-01 onward; 2025
  issued D-1 09:05 / 10:05) -> data/miso/prod_flex_daep.parquet.

## Results (2026-10-20) -- `research/miso/compare_prod.py`
| 2025 | Kumo m0_original MAE / RMSE | production MAE / RMSE | Kumo gain (paired day bootstrap) |
|---|---|---|---|
| ALL | 3.385 / 6.86 | 3.716 / 8.45 | +0.332 +- 0.100 (win .64) |
| HOLD | 3.455 / 7.34 | 3.908 / 9.51 | +0.452 +- 0.141 (win .64) |
| SCREEN | 3.244 / 5.78 | 3.335 / 5.81 | +0.091 +- 0.102 |
- Kumo beats production in every season (JJA 4.53 vs 5.15, DJF 3.75 vs 4.08) and hour block
  (HB 16-20 5.82 vs 6.55); RMSE gap is largest (spike hours). Coverage of the 80% band 0.83.
- Error corr 0.82. Blend weight picked on SCREEN = 0.45 prod, but on HOLD it is no better than Kumo
  alone (3.461 vs 3.455) and worse on RMSE (7.96 vs 7.34) -> use Kumo alone.
- Bias (y - fc): Kumo +0.24, production +0.39.
Predictions: box /scratch/outputs/miso-fe/m0_original, local outputs/miso-fe/m0_original.

## RT energy price backpublish, 2024-2025 (2026-11-05) -- `research/miso_rt/`
- `collect_rt.R` -> data/miso/miso_energy_prices.parquet (miso_energy_prices dt, hr, da/rt_energy_price,
  2017-12-01 .. 2026-10-20, no gaps). `make_original.py ... m0_original_bp24 2024-01-01 2026-01-01` then
  `make_rt.py m0_original_bp24` -> data/miso-fe/m0_original_bp24_rt: m0_original features, y = RT system energy
  (source y == DA price on every row, so alignment is right).
- Run `m0_rt_bp24_daily`: daily walk-forward 2024-01-01 .. 2025-12-31, `--context-gap-hours 24` (context ends D-2),
  large / 8 est / seed 0 / 10k context. 731 steps in 53 min.
- `score_rt.py`, scored on RT, trading days (non-Monday), MAE / RMSE; gain = paired day bootstrap:
| | Kumo RT q500 | Kumo RT mean | 50/50 RT+DA Kumo | Kumo DA q500 | DA price |
|---|---|---|---|---|---|
| 2024 | 7.63 / 26.6 | 13.09 / 35.2 | 7.99 / 26.0 | 8.94 / 26.1 | 8.60 / 25.6 |
| 2025 | 11.42 / 41.0 | 19.23 / 45.1 | 11.66 / 39.7 | 12.89 / 39.3 | 12.55 / 38.9 |
  Kumo RT vs DA price +0.97 +- 0.14 (2024), +1.13 +- 0.31 (2025); blend vs DA price +0.61 +- 0.11, +0.89 +- 0.18.
  RMSE: DA price slightly best. The quantile-average mean over-forecasts (bias -6 / -9) -> use q500.
  80% coverage 0.85 / 0.87. Bias of q500 +3.4 / +5.5 (under-forecasts, right-skewed RT).
