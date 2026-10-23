# Artifact 4 plan: Kumo PJM wind and solar forecasts, and feeding all custom forecasts into the Kumo energy forecast

Written 2026-10-23, revised the same day after user review. Read `research/plans/common.md` first, then the
"Artifact 2" section of `research/pjm-kumo-feature-engineering.md` (the load version of this experiment
and its null result on the price side), then `research/pjm_load/` (the code this reuses). Needs SQL (ask
the user to open the tunnel) and roughly 5 h of GPU, run sequentially. **2025 only: run nothing on the
2026 test set.**

## Questions
(a) Can Kumo-Tabular forecast PJM ALL hourly **wind** and **solar** generation better than Meteologica,
the vendor forecasts the production price model uses?
(b) If the custom wind, solar **and** load forecasts all go into the Kumo energy model together, does the
price forecast improve? Artifact 2 swapped in load alone and found nothing (HOLD -0.033 +- 0.055 $/MWh);
this asks whether all three custom inputs together, plus their differences from the **ISO** forecasts,
add up to something. The ISO differences carry a market theory (from the user): PJM commits generation
on its own forecasts, the market anticipates mis-commitment when those forecasts are wrong, and traders
see better forecasts than the ISO's. So a better forecast minus the ISO's may predict the mis-commit, and
the price effect, ahead of time.

**Prior:** in 2025 wind averaged ~3.7 GW and solar ~2.7 GW (peak 11.6 GW) against ~90 GW of load, so
better wind and solar forecasts move thermal stress by a few hundred MW at most. Expect a clear result on
(a) and a likely null on (b). Step 5 runs a perfect-foresight check before the energy run, and the
artifact must say plainly if (b) is null.

## Key facts (verify each before relying on it)
- `PjmFlexDaep20261001$collect_data()` (`~/dev/tios-quant-shiny/lib/trader-forecasting/pjm-flex-daep-20261001.R`,
  ~lines 320-360): `pjm_load = load_all.meteologica.forecast.pjm`, `pjm_wind = wind_all.meteologica.forecast.pjm`,
  `pjm_solar = solar_all.meteologica.forecast.pjm`, and `solar_lag1_diff = lead(solar_all, 1) - solar_all`
  (despite the name it is the *next* hour minus this hour).
- Inputs in `data/pjm-fe/v0` derived from ALL load / wind / solar (formulas from TiosQ / the dafc-20261001
  config; the load ones were verified exact in `research/pjm_load/make_customload.py`):
  - `thermal_stress_all_pjm = cap / (cap - gen_on_outage) * (load - wind - solar)`: recompute as
    `k * (L' - W' - S')` with the row's own `k = ts / (L - W - S)`;
  - `solarduckcurve.forecast.pjm = load - solar`, and `solarduckcurve_lead1_diff = sdc(t) - sdc(t-1)`;
  - `solar_lag1_diff = S(t+1) - S(t)`.
  - `southwest_east_diff_duckcurve_pjm` uses the *regional* Meteologica load / wind / solar and stays
    unchanged (regional custom forecasts are out of scope; say so in the method note).
- Window: the 2025 energy walk-forward only reads the last 10,000 context rows, so every custom forecast
  is needed out of sample from **2023-10-01 to 2025-12-31**.
- Custom load already exists: `outputs/pjm-load-fe/P2_bp_gap24` (that window exactly).
- **ISO forecast cutoff: 11:00 EPT (user decision, 2026-10-23).** `pjm_load_forecast` keeps one vintage
  per hour, usually stamped ~10:47 EPT on D-1 (`forecast_time` is Eastern). That is before PJM's
  day-ahead bid deadline (11:00 EPT = 10:00 CT), so it is fair (Artifact 2 dropped it under the stricter
  10:30 rule). For every ISO forecast (load, wind, solar; model inputs and difference columns) use the
  latest vintage with `forecast_time <= timestamp(date_sub(dt, interval 1 day), '11:00:00')`.
  Meteologica, Maxar and all other inputs keep the 10:30 rule. State both cutoffs in the artifact's
  method note.
- Existing local data: `data/pjm_extra/scen_wind.parquet` (Meteologica scenario wind, ECMWF-ENS + GEFS,
  average / std / q5 / q50 / q95, fair D-1 vintages from 2018-04) and `scen_solar.parquet` (ECMWF-ENS,
  from 2021-07). `data/pjm_config.parquet` has the Meteologica regional wind (western / midatlantic /
  south) and solar (west / midatlantic / south) forecasts. Refresh all of these in step 1.

## Work plan
1. **Table discovery and pull** (tunnel; `research/pjm_renew/collect.R`, modelled on `research/pjm_load/collect.R`).
   - **Actuals (targets):** find PJM hourly wind and solar generation for the RTO. Check what the TiosQ
     history importer uses (`tios_macro_data_import(market = "pjm", importer_keyword = "wind" / "solar",
     vendor_type = "iso", data_type = "history")`), else `SHOW TABLES LIKE 'pjm%wind%'` / `'pjm%solar%'` /
     `'pjm%fuel%'`. Note whether the table keeps revisions (an insert or forecast timestamp); if so, use
     only values available by the time a >=48h lag would be read.
   - **Meteologica:** ALL and regional wind and solar forecasts, fair vintage (latest
     `forecast_time <= D-1 10:30 EPT`), plus any interval columns (`lower_80` / `upper_80`, as in the
     load table) and the scenario tables.
   - **ISO forecasts (required inputs):** PJM's own wind and solar forecasts, RTO and regional if
     available, and the ISO load forecast (`pjm_load_forecast`, `RTO COMBINED`) for the difference
     columns. Use the 11:00 EPT fair-vintage query from Key facts. Measure coverage and staleness (hours
     between the vintage used and the cutoff) by year, and leave days without a fair vintage as missing.
     If fair coverage in 2023-10 .. 2025 is very low (say under half of days), tell the user before going
     on rather than dropping it.
   - **Weather:** `pjm_maxar_weather_forecasts`. Look up `pjm_locations` for stations near the wind
     clusters (northern Illinois / ComEd, western Pennsylvania ridges, West Virginia, Ohio, Indiana) and
     the solar clusters (Virginia / Dominion, New Jersey, Maryland, Ohio, Pennsylvania). Pull
     `wind_speed` (and gusts / direction if present), `tscv` (sky cover), `temp`.
   - Range 2017-11-01 .. 2025-12-31, to `data/pjm_renew/`.
2. **Gate 1: definitions.**
   - The fresh Meteologica ALL pulls must reproduce `v0.pjm_wind` and `v0.pjm_solar` exactly. If they
     don't, use whatever does and say so.
   - The actuals must measure the same thing the forecasts target: compare scale and 2025 bias for
     Meteologica and ISO. PJM metered solar excludes behind-the-meter solar. A persistent 10%+ gap means
     a definition mismatch that has to be resolved before going on.
   - Report Meteologica's and ISO's 2025 MAE / nMAE for both targets as the benchmarks, and the ISO load
     forecast's 2025 MAE next to Meteologica's for context.
3. **Features: vendor versions only** (`research/pjm_renew/make_features.py`; one dataset per target,
   `data/pjm-renew-fe/<target>/<variant>`, with `--val-start 2023-10-01` for the backpublish split).
   No weather-only variants.
   - **Wind W2:**
     - Meteologica ALL and regional wind forecasts (level, 24h change, share of a capacity proxy).
     - ISO wind forecasts (same transforms, plus a staleness column).
     - The ECMWF-ENS and GEFS scenario average / std / q5 / q95.
     - Meteologica's recent error (mean actual minus forecast over D-2, and over the 7 days ending D-2).
     - Maxar wind speed at the wind-cluster stations, plus its cube clipped at a rated-speed proxy.
     - Calendar (hour, season sin/cos).
     - Actual wind lags >= 48h (48 / 72 / 168, same-hour 7-day mean, D-2 mean / max).
     - Capacity proxy = rolling 30-day max of actuals, lagged 48h.
   - **Solar S2:**
     - Meteologica ALL and regional solar forecasts.
     - ISO solar forecasts (plus staleness).
     - Scenario average / std / q5 / q95.
     - Meteologica's recent error.
     - Deterministic sun geometry for each hour: solar elevation and a clear-sky irradiance proxy at a
       PJM centroid and three regional centroids, plus day length.
     - Maxar sky cover and temperature at the solar-cluster stations.
     - Calendar.
     - Actual solar lags >= 48h.
     - The same-hour max over the 30 days ending D-2, as a clear-sky / capacity proxy; plus today's
       Meteologica forecast as a share of it.
   - Lags of 48h or more only for any actual, and every run uses `--context-gap-hours 24` (D-1 actuals
     are not known at the cutoff; same rule as load).
4. **Quick check, then backpublish.**
   - First a 2025 every-3rd-day screen of W2 and S2 (`--eval-every 3 --context-gap-hours 24`, ~15 min
     each) as a bug check. **Gate 2:** each must beat Meteologica on the selection days. If one doesn't,
     look for a leak or bug before spending a backpublish on it.
   - Then backpublish each: one daily walk-forward with train < 2023-10-01 and val 2023-10-01 ..
     2025-12-31, `--context-gap-hours 24`, est8, seed 0, 10k context (~820 steps, ~1.5-2 h each).
   - For solar, clip all quantiles at 0 (a rule fixed before scoring, applied to every hour).
   - Score 2025 vs Meteologica and ISO:
     - MAE, RMSE, nMAE and bias; daylight-only hours for solar;
     - by season and hour block;
     - hour-to-hour ramp error (`solar_lag1_diff` and the duck-curve change are ramps);
     - error in the top 10% thermal-stress hours, where prices spike.
5. **Gate 3: perfect-foresight check** (one energy run, ~35 min). Build v0 with the 2023-10 .. 2025
   load, wind and solar replaced by the *actuals*. Recompute every derived input, and add six difference
   columns, for load, wind and solar each:
   - actual minus Meteologica (`*_actual_minus_met`);
   - actual minus ISO (`*_actual_minus_iso`), with the ISO vintage rule from Key facts.

   This contains everything step 6 uses (with the custom forecasts replaced by the truth), plus the
   Meteologica errors. Run the baseline settings on 2025.
   - This leaks the future, so it is a diagnostic only, never a result. It bounds what any load / wind /
     solar forecast could do for this price model.
   - **If it gains less than 2 SE on held-out days,** skip step 6 and report the bound. Read it as a sign
     that the day-ahead market already prices on load / wind / solar forecasts about as good as perfect
     ones, i.e. at par with anything we could build. Note it; nothing further to act on.
6. **One energy run with all custom forecasts** (only if Gate 3 passes): `v0_custom_all`. Generalise
   `research/pjm_load/make_customload.py` into `research/pjm_renew/make_custominputs.py --load <run>
   --wind <run> --solar <run> [--actuals]` (`--actuals` builds the step 5 dataset):
   - `pjm_load`, `pjm_wind` and `pjm_solar` are replaced by the custom q500s from 2023-10-01 on.
   - Every derived input above is recomputed: thermal stress, duck curve, duck-curve change, solar change.
   - Three columns are added: `load_custom_minus_iso`, `wind_custom_minus_iso` and
     `solar_custom_minus_iso` (custom forecast minus the ISO forecast; the ISO vintage rule is in Key
     facts). There are no custom-minus-Meteologica columns. They are left missing before 2023-10-01
     (those rows never enter the 2025 context) and wherever no ISO vintage exists.
   - Each formula is verified on v0 first, and the script asserts that every replaced value came from a
     step whose context ended by D-2.
   - Run with exactly the baseline settings (est8, seed 0, 10k, daily 2025, no gap). Compare with the
     baseline (paired day-block bootstrap on HOLD and ALL, win rate) and with Artifact 2's
     `v0_customload_b`.
   - It counts as an improvement only if the HOLD gain is above 2 SE. Otherwise report "no measurable
     gain" with the numbers. No further variants unless the user asks.

## Compute budget (sequential on the box)
| Step | Runs | Time |
|---|---|---|
| 4 screens | 2 x ~15 min | ~0.5 h |
| 4 backpublish | 2 x ~1.5-2 h | ~3.5 h |
| 5 perfect-foresight check | 1 | ~35 min |
| 6 energy run | 1 | ~35 min |

About 5 h in all. Chain jobs with a script file and `nohup` (see common.md). Never `pkill -f` a pattern
that matches the ssh command itself.

## Artifact sections (private claude.ai page, same house style as Artifacts 1-3)
1. **Headline:** two short answers, (a) wind and solar vs Meteologica and ISO and (b) price impact,
   with Artifact 2's result as the prior in one sentence.
2. **Wind scorecard and solar scorecard:** MAE / RMSE / nMAE / bias, HOLD and full 2025, gain +- SE,
   days won, with Meteologica and ISO as benchmarks (ISO coverage noted); daylight-only for solar.
3. **Where it helps:** season and hour block; ramp error; top-stress-hour error; monthly bias chart.
4. **Time series (Plotly):** hourly 2025 actual vs Meteologica vs ISO vs Kumo, for wind and for solar
   (tabs), with a range slider and a daily toggle.
5. **What could the price model gain at best:** the perfect-foresight bound in plain language, and what
   a null bound says about the market.
6. **Energy impact** (if step 6 ran): baseline vs `v0_custom_all` (and Artifact 2's load-only result)
   scorecard; spike-day table.
7. **How it works:** flow from weather and vendor / ISO forecasts to the Kumo wind / solar / load models,
   then to the custom inputs and the Kumo energy model; the walk-forward rule in one sentence; the
   mis-commitment theory behind the custom-minus-ISO columns in two sentences; links to Artifacts 1-3.
8. **Specs and method note:** one row per model (Meteologica wind / solar, ISO wind / solar with the
   vintage used, ISO load with the vintage used, Kumo W2 / S2, Kumo load P2, baseline,
   `v0_custom_all`, the perfect-foresight run):
   target, method, context, cadence, inputs, cutoff, estimators / seed, scored period.

## Deliverables and bookkeeping
- Code `research/pjm_renew/` (collect.R, make_features.py, make_custominputs.py, artifact4_data.py,
  artifact4_template.html); data `data/pjm_renew/`, `data/pjm-renew-fe/`, `data/pjm-fe/v0_custom_all`,
  `data/pjm-fe/v0_actuals_all`.
- Box outputs `/scratch/outputs/pjm-renew-fe/`, `/scratch/outputs/pjm-fe/v0_custom_all_daily`,
  `/scratch/outputs/pjm-fe/v0_actuals_all_daily`.
- Log in `research/pjm-kumo-feature-engineering.md` (new "Artifact 4" section), commit and push to
  `pjm-feature-engineering`, offer an S3 sync of the new outputs.
