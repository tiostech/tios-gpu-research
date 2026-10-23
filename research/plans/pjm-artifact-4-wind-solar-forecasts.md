# Artifact 4 plan: Kumo PJM wind and solar forecasts, and feeding them into the Kumo energy forecast

Written 2026-10-23. Read `research/plans/common.md` first, then the "Artifact 2" section of
`research/pjm-kumo-feature-engineering.md` (the load version of this experiment, and its null result on the
price side), then `research/pjm_load/` (the code this reuses). Needs SQL (ask the user to open the tunnel)
and roughly 8-10 h of GPU, run sequentially.

## Questions
(a) Can Kumo-Tabular forecast PJM ALL hourly **wind** and **solar** generation better than Meteologica,
the vendor forecasts the production price model uses?
(b) If those forecasts replace Meteologica's in the Kumo energy model (with the derived inputs
recomputed), does the price forecast improve? And does adding them on top of the custom load forecast
from Artifact 2 help, i.e. do **more** custom inputs add up to something the single load swap didn't?

**Prior, stated up front:** Artifact 2 found that a load forecast 8-10% better than Meteologica's gave no
measurable price gain (-0.033 +- 0.055 $/MWh on held-out days). In 2025 wind averaged ~3.7 GW and
solar ~2.7 GW (peak 11.6 GW) against ~90 GW of load, so a 10-20% better wind or solar forecast moves
thermal stress by a few hundred MW at most. Expect a clear result on (a) and a likely null on (b). The
plan therefore runs a cheap perfect-foresight check (step 6) before spending GPU time on (b), and the
artifact must say plainly if (b) is null.

## Key facts (verify each before relying on it)
- `PjmFlexDaep20261001$collect_data()` (`~/dev/tios-quant-shiny/lib/trader-forecasting/pjm-flex-daep-20261001.R`,
  ~lines 320-360): `pjm_wind = wind_all.meteologica.forecast.pjm`,
  `pjm_solar = solar_all.meteologica.forecast.pjm`, and
  `solar_lag1_diff = lead(solar_all, 1) - solar_all` (despite the name it is the *next* hour minus this hour).
- Inputs derived from ALL wind/solar in `data/pjm-fe/v0` (formulas from TiosQ / the dafc-20261001 config;
  the load ones were verified exact in `research/pjm_load/make_customload.py`):
  - `thermal_stress_all_pjm = cap / (cap - gen_on_outage) * (load - wind - solar)`: recompute as
    `k * (L - W' - S')` with the row's own `k = ts / (L - W - S)`;
  - `solarduckcurve.forecast.pjm = load - solar`, and `solarduckcurve_lead1_diff = sdc(t) - sdc(t-1)`;
  - `solar_lag1_diff = S(t+1) - S(t)`.
  - `southwest_east_diff_duckcurve_pjm` uses the *regional* Meteologica load/wind/solar and stays unchanged
    (regional custom forecasts are out of scope; say so in the method note).
- Window: the 2025 energy walk-forward only reads the last 10,000 context rows, so custom forecasts are
  needed out of sample from **2023-10-01**. Run the backpublish through **2026-10-08** (the end of the
  test split) so the 2026 confirmation in step 9 needs no second run.
- Existing local data: `data/pjm_extra/scen_wind.parquet` (Meteologica scenario wind, ECMWF-ENS + GEFS,
  average / std / q5 / q50 / q95, fair D-1 vintages from 2018-04) and `scen_solar.parquet` (ECMWF-ENS,
  from 2021-07); `data/pjm_config.parquet` has the Meteologica regional wind (western / midatlantic /
  south) and solar (west / midatlantic / south) forecasts. Refresh all of these in step 1 through 2026-10-08.
- Custom load forecast from Artifact 2: `outputs/pjm-load-fe/P2_bp_gap24` (2023-10-01 .. 2025-12-31;
  it would need extending to 2026 for the combined variant on the test set, see step 9).

## Work plan
1. **Table discovery and pull** (tunnel; `research/pjm_renew/collect.R`, modelled on `research/pjm_load/collect.R`).
   - **Actuals (targets):** find PJM hourly wind and solar generation for the RTO. Check what the TiosQ
     history importer uses (`tios_macro_data_import(market = "pjm", importer_keyword = "wind" / "solar",
     vendor_type = "iso", data_type = "history")`), else `SHOW TABLES LIKE 'pjm%wind%'` / `'pjm%solar%'` /
     `'pjm%fuel%'`. Note whether the table keeps revisions (an insert or forecast timestamp); if so use
     only values available by the time a >=48h lag would be read.
   - **Meteologica:** ALL and regional wind and solar forecasts, fair vintage (latest `forecast_time <=
     D-1 10:30 EPT`), plus any interval columns (`lower_80` / `upper_80` as in the load table) and the
     scenario tables.
   - **ISO forecasts:** PJM's own wind and solar forecasts if they exist. Audit the vintage the same way
     as the ISO load forecast, which failed (stamped after 10:30 on ~99% of days). Drop them if not fair.
   - **Weather:** `pjm_maxar_weather_forecasts`. Look up `pjm_locations` for stations near the wind
     clusters (northern Illinois / ComEd, western Pennsylvania ridges, West Virginia, Ohio, Indiana) and
     the solar clusters (Virginia / Dominion, New Jersey, Maryland, Ohio, Pennsylvania). Pull
     `wind_speed` (and gusts / direction if present), `tscv` (sky cover), `temp`.
   - Range 2017-11-01 .. 2026-10-08, to `data/pjm_renew/`.
2. **Gate 1: definitions.**
   - The fresh Meteologica ALL pulls must reproduce `v0.pjm_wind` and `v0.pjm_solar` exactly. If they
     don't, use whatever does and say so.
   - The actuals must measure the same thing Meteologica forecasts: compare scale and 2025 bias.
     PJM metered solar excludes behind-the-meter solar. A persistent 10%+ gap means a definition mismatch
     that has to be resolved before going on.
   - Report Meteologica's 2025 MAE / nMAE for both as the benchmark.
3. **Features** (`research/pjm_renew/make_features.py`; one dataset per target, `data/pjm-renew-fe/<target>/<variant>`).
   - **Wind W0 (no vendor):** Maxar wind speed at the wind-cluster stations, plus its cube clipped at a
     rated-speed proxy (power curve shape), its daily mean / max and 24h change; calendar (hour,
     season sin/cos); actual wind lags >= 48h (48 / 72 / 168, same-hour 7-day mean, D-2 mean / max);
     a capacity proxy = rolling 30-day max of actuals, lagged 48h.
   - **Wind W2 = W0 + vendor:** Meteologica ALL and regional wind forecasts (level, 24h change, share of
     capacity proxy), the ECMWF-ENS and GEFS scenario average / std / q5 / q95, and Meteologica's recent
     error (mean actual minus forecast over D-2 and over the 7 days ending D-2).
   - **Solar S0 (no vendor):** deterministic sun geometry for each hour (solar elevation and a clear-sky
     irradiance proxy at a PJM centroid and three regional centroids, day length); Maxar sky cover and
     temperature at the solar-cluster stations; calendar; actual solar lags >= 48h, plus the same-hour max
     over the 30 days ending D-2 as a clear-sky / capacity proxy and today's elevation times that proxy.
   - **Solar S2 = S0 + vendor:** Meteologica ALL and regional solar forecasts, the scenario average / std
     / q5 / q95, the forecast as a share of the clear-sky proxy, and Meteologica's recent error.
   - Only lags of 48h or more for any actual, and every run uses `--context-gap-hours 24` (the D-1
     actuals are not known at the cutoff; same rule as load).
4. **Screens** on 2025, every 3rd day, `--eval-every 3 --context-gap-hours 24`, est8, seed 0, 10k
   context (~15 min each): W0, W2, S0, S2. Expect the vendor variants to win, as for load. Pick one per
   target on the selection days only. **Gate 2:** backpublish a target only if its winner beats
   Meteologica on the selection days by more than 2 SE. If neither target clears the gate, stop after
   step 5 and write up (a) alone.
5. **Backpublish** each winner: one daily walk-forward with train < 2023-10-01 and val 2023-10-01 ..
   2026-10-08 (`make_features.py --val-start` writes `<variant>_bp`), `--context-gap-hours 24`, est8,
   seed 0, 10k (~1,100 steps, ~2-2.5 h each). For solar, clip all quantiles at 0 (a rule fixed before
   scoring and applied to every hour). Score 2025 vs Meteologica (MAE, RMSE, nMAE, bias; daylight-only
   for solar; by season and hour block; hour-to-hour ramp error, since `solar_lag1_diff` and the
   duck-curve change are ramps; and error in the top 10% thermal-stress hours, where prices spike). 2026
   is scored only after all choices are fixed.
6. **Gate 3: perfect-foresight check** (one energy run, ~35 min). Replace `pjm_wind` and `pjm_solar` in
   v0 with the *actuals* and recompute the derived inputs, then run the baseline settings on 2025. This
   leaks the future and is a diagnostic only, never a result. It bounds what any wind/solar forecast can
   do for this price model. If even perfect wind + solar gains less than 2 SE on held-out days, skip
   step 7, report the bound, and say the energy-side question is closed for wind and solar. Optionally
   run the same check for load (one more run), which would explain Artifact 2's null in the same terms.
7. **Energy model with the custom forecasts** (only if Gate 3 passes). Generalise
   `research/pjm_load/make_customload.py` into `research/pjm_renew/make_custominputs.py --wind <run>
   --solar <run> [--load <run>]`:
   - It replaces the series from 2023-10-01 on and recomputes every derived input above. Only the
     "with recomputed derived inputs" version is built, per the user's instruction on Artifact 2.
   - It verifies each formula on v0 first.
   - It asserts that each replaced value came from a step whose context ended by D-2.
   - Run with exactly the baseline settings (est8, seed 0, 10k, daily 2025, no gap), in this order:
     - `v0_custom_ws`: wind + solar.
     - `v0_custom_lws`: + Artifact 2's custom load. This answers "do more custom inputs help".
     - `v0_custom_w` and `v0_custom_s` separately, only if `ws` shows a gain worth attributing.

   Compare each run with the baseline: paired day-block bootstrap on HOLD and ALL, plus win rate.
   Compare `lws` with Artifact 2's `v0_customload_b` as well.
8. **Decision.** Pick the variant to report on HOLD only. A variant counts as an improvement only if
   the HOLD gain is above 2 SE. Otherwise report "no measurable gain" with the numbers.
9. **2026 confirmation** (after step 8 is fixed; ~25 min per run). Rebuild the test dataset as for
   Artifact 3:
   - Use `/scratch/data/pjm` with train = train + val and val = test, then apply the same replacement.
   - Run the chosen variant against `baseline_v0_test_daily`.
   - The `lws` variant also needs the load backpublish extended through 2026-10-08 (one ~30 min run).
     Do that only if `lws` is the variant chosen.

## Compute budget (sequential on the box)
| Step | Runs | Time |
|---|---|---|
| 4 screens | 4 x ~15 min | ~1 h |
| 5 backpublish | 2 x ~2-2.5 h | ~4.5 h |
| 6 perfect-foresight check | 1 (+1 optional for load) | ~35-70 min |
| 7 energy runs | 2, up to 4 | ~1.2-2.3 h |
| 9 2026 confirmation | 1-2 (+ load extension) | ~0.5-1 h |

Chain jobs with a script file and `nohup` (see common.md). Never `pkill -f` a pattern that matches the
ssh command itself.

## Artifact sections (private claude.ai page, same house style as Artifacts 1-3)
1. **Headline:** two short answers, (a) wind and solar vs Meteologica and (b) price impact, with the
   prior from Artifact 2 stated in one sentence.
2. **Wind scorecard and solar scorecard:** MAE / RMSE / nMAE / bias, HOLD and full 2025, gain +- SE,
   days won; daylight-only for solar; W0 vs W2 and S0 vs S2 screen table (why the vendor forecast has
   to be an input).
3. **Where it helps:** season and hour block; ramp error; top-stress-hour error; monthly bias chart.
4. **Time series (Plotly):** hourly 2025 actual vs Meteologica vs Kumo for wind and for solar (tabs),
   with a range slider and a daily toggle.
5. **What could the price model gain at best:** the perfect-foresight bound, explained in plain language.
6. **Energy impact:** baseline vs `ws` vs `lws` (and `w` / `s` if run) scorecard; spike-day table;
   2026 confirmation if run.
7. **How it works:** flow from weather and vendor forecasts to the Kumo wind / solar / load models,
   then to the custom inputs and the Kumo energy model; the walk-forward rule in one sentence; links to
   Artifacts 1-3.
8. **Specs and method note:** one row per model (Meteologica wind / solar, ISO if used, W0 / W2 / S0 /
   S2, baseline, custom variants): target, method, context, cadence, inputs, cutoff, estimators / seed,
   scored period.

## Deliverables and bookkeeping
- Code `research/pjm_renew/` (collect.R, make_features.py, make_custominputs.py, artifact4_data.py,
  artifact4_template.html); data `data/pjm_renew/`, `data/pjm-renew-fe/`, `data/pjm-fe/v0_custom_*`.
- Box outputs `/scratch/outputs/pjm-renew-fe/`, `/scratch/outputs/pjm-fe/v0_custom_*` (+ `_test`).
- Log in `research/pjm-kumo-feature-engineering.md` (new "Artifact 4" section), commit and push to
  `pjm-feature-engineering`, offer an S3 sync of the new outputs.
