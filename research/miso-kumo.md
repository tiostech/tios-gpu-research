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
