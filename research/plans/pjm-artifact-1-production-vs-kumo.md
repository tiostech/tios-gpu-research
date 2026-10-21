# Artifact 1 spec: PJM production forecast vs Kumo-Tabular

Read `research/plans/common.md` first. No new GPU runs and no SQL are needed: everything below is
built from existing prediction files (see "Existing assets" in common.md).

## Audience and message
Non-technical stakeholders. One message: with the *same inputs* as the production model (its RF
feature set + time columns), an off-the-shelf Kumo-Tabular run beats the production PJM DAEP forecast on
2025, and the gain holds on held-out days, across seasons and hours. Keep it honest: note what the
baseline is (a no-frills run) and that a tuned Kumo ensemble does better still (ensC, optional).

## Sections
1. **Headline + scorecard.** MAE / RMSE for production `flex.daep.20261001` vs Kumo original baseline on
   full 2025 and HOLD (243 days), % improvement, paired-bootstrap gain +- SE and win rate. Optionally a
   third column for ensC, labelled "best Kumo ensemble from the research".
2. **Where the gain comes from.** MAE by season (DJF/MAM/JJA/SON) and hour block (00-05, 06-09, 10-15,
   16-20, 21-23), plus the worst 10 days for production vs Kumo (date, actual daily max, both daily MAEs).
   Note blending production into Kumo doesn't help (error corr 0.82; picked weight ~0).
3. **Interactive time series (Plotly).** Hourly 2025: actual DAEP, production, Kumo baseline (+ ensC,
   off by default). Range slider + buttons (1w / 1m / 3m / all), unified hover, legend toggles, a
   daily-mean toggle so the full year reads clearly. Optional second panel: rolling 7-day MAE per forecast.
4. **What Kumo-Tabular is (plain language).** Draft points:
   - Production (random forests / boosted trees + blender): each day the model is *trained* on a fixed
     window (e.g. last 90 / 730 days), learning fixed rules from features to price, then applied to
     tomorrow.
   - Kumo-Tabular: a large pre-trained "foundation model" for tables. It isn't trained on our data;
     each day we hand it ~10,000 recent hours (features + what actually happened) as *context* plus
     tomorrow's 24 rows, and it predicts by recognising which past hours resemble tomorrow (in-context
     learning — like giving an experienced analyst a stack of comparable days). New data is used
     immediately; no model tuning; it returns a full probability range (quantiles), not just one number.
   - Trade-offs: needs a GPU; can't extrapolate beyond levels seen in its context; more input columns
     isn't automatically better; a handful of extreme spike days still dominate the remaining error.
   - Use one concrete 2025 PJM example from the data (e.g. a cold-snap or heat-wave day).
   - One line on the cross-market check: same result in MISO and SPP (numbers in common.md).
5. **Model specs.** Table, one row per model shown: target, method, training/context window, retrain
   cadence, inputs (count + list), data cutoff, estimators/seed, scored period.
   - Production `flex.daep.20261001`: read the RF / scaled-method / blender structure and lookbacks
     from `~/dev/tios-quant-shiny/lib/trader-forecasting/pjm-flex-daep-20261001.R` (the formula
     blocks around lines 730 and 820 list the RF features; check each model's training window and the
     blender inputs). Point forecast; published D-1 ~10:30 EPT.
   - Kumo original baseline: see common.md (est8, seed 0, last 10k hours, daily walk-forward,
     `data/pjm-fe/v0` columns).
   - ensC (if shown): its four members and their settings from the price log ("Current best" and
     "Ensembling round" sections).
6. **Method note (small print).** Walk-forward procedure, fairness cutoff, energy-component target,
   holdout definition, bootstrap.

## Steps
1. Script `research/pjm/artifact1_data.py`: load baseline, production (ts = dt + hr - 1 h), ensC,
   actuals; compute the scorecard / breakdowns / worst days; write one compact JSON for the page.
2. Quickstart the Artifact tool, build the HTML (scorecard, tables, Plotly charts, explainer, specs),
   one preview pass, publish (private).
3. Log in the price research doc, commit, push. Expect ~1-2 hours of work; no GPU time.
