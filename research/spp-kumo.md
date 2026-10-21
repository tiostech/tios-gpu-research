# SPP DA energy price — Kumo-Tabular

Same setup as PJM / MISO (research/pjm-kumo-feature-engineering.md, research/miso-kumo.md):
Kumo-Tabular large, 8 estimators, seed 0, `--max-context 10000`, daily walk-forward over 2025,
target `y` = `da_energy_price` (SPP DA *system energy* component via shiny's `da_energy_prices()`,
not SPP North Hub LMP), time col `ts` (hour-beginning, market-local). HOLD = 2025 days whose day
number is not a multiple of 3 (243 days).

## Data
- `research/spp/collect_original.R` -> `SppFlexDaep20260701$new()$do_collect_data()` (production
  class ~/dev/tios-quant-shiny/lib/trader-forecasting/spp-flex-daep-20260701.R, config
  daep-20260701; fair vintages 25 min before the bid deadline) -> data/spp/spp_raw.parquet
  (2017-11-30 .. 2026-10-21, 77,910 rows).
- `research/spp/make_original.py` -> data/spp-fe/s0_original: the 19 regressors of the production
  baseline RF (`rf730d_predict`: spp_load/wind/solar, netload_all_spp, netload/wind/solar_deriv,
  netload_north_miso, netload_sps_z3, spp_gen_on_outage, spp_nuclear_goo, feelslike_spp_popwt,
  windspeed_okc_wichita, spp_pct_of_dispatch, spp_gas, season_sin/cos, hr_numeric, utc_hr) + ts, dt,
  hr, dt_ordinal. The RF's `days_from_target_dt` is target-relative (recomputed per forecast day), so
  it is replaced by dt_ordinal / ts epoch. train 2018-2024 (61,392; 7 y missing), val 2025 (8,760;
  1 missing), test 2026 (7,035). NaN left as NULL (netload_north_miso 12%, feelslike 9%, pct_of_dispatch 2%).
- Production forecast: `spp_tios_forecasts` name 'flex.daep.20260701' (2022-01-01 onward; 2025 all
  issued D-1 09:05) -> data/spp/prod_flex_daep.parquet.

## Results (2026-10-20) -- `research/miso/compare_prod.py <preds> data/spp/prod_flex_daep.parquet`
| 2025 | Kumo s0_original MAE / RMSE | production MAE / RMSE | Kumo gain (paired day bootstrap) |
|---|---|---|---|
| ALL | 3.933 / 6.53 | 4.446 / 7.29 | +0.513 +- 0.098 (win .67) |
| HOLD | 3.894 / 6.55 | 4.432 / 7.55 | +0.537 +- 0.132 (win .68) |
| SCREEN | 4.010 / 6.48 | 4.474 / 6.74 | +0.463 +- 0.149 (win .65) |
- Kumo beats production in every season (MAM 4.99 vs 5.79, SON 3.39 vs 3.99) and every hour block.
- Error corr 0.81. Blend w_prod = 0.20 (picked on SCREEN): HOLD 3.853 / 6.52, +0.042 +- 0.028 vs Kumo
  alone (win .55) -> small, ~1.5 SE; 50/50 is worse than Kumo alone.
- Bias (y - fc): Kumo -0.05, production -0.21.
Predictions: box /scratch/outputs/spp-fe/s0_original, local outputs/spp-fe/s0_original.
