"""Energy-model datasets with custom (or actual) PJM ALL load / wind / solar in place of Meteologica's.

  uv run python research/pjm_renew/make_custominputs.py --load <run> --wind <run> --solar <run> [--data DATA]
  uv run python research/pjm_renew/make_custominputs.py --actuals [--data DATA]

Generalises research/pjm_load/make_customload.py (v0_customload_b). From 2023-10-01 to 2025-12-31 (the price
baseline's 10k-row context never reaches back before ~2023-11) v0's pjm_load / pjm_wind / pjm_solar
(Meteologica ALL) are replaced, and every input derived from them is recomputed (formulas from TiosQ / the
dafc-20261001 config, each verified exact on v0 before use):
  thermal_stress_all_pjm = cap / (cap - gen_on_outage) * (L - W - S)  -> k * (L' - W' - S'), k = ts / (L - W - S)
  solarduckcurve.forecast.pjm = L - S;  solarduckcurve_lead1_diff.forecast.pjm = sdc(t) - sdc(t - 1)
  solar_lag1_diff = S(t + 1) - S(t)
southwest_east_diff_duckcurve_pjm uses Meteologica's regional forecasts and is unchanged.

  v0_custom_all   custom q500s from backpublished walk-forwards (daily, --context-gap-hours 24, val start
                  <= 2023-10-01), plus load/wind/solar_custom_minus_iso (custom minus ISO forecast, ISO vintage
                  = latest by D-1 11:00 EPT).
  v0_actuals_all  (--actuals; perfect-foresight diagnostic, leaks the future) actuals instead, plus
                  *_actual_minus_met and *_actual_minus_iso.
Difference columns are missing before 2023-10-01 and where no ISO vintage exists.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spp_load"))
from make_features import ts_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
A, B = pd.Timestamp("2023-10-01"), pd.Timestamp("2026-01-01")
COL = {"load": "pjm_load", "wind": "pjm_wind", "solar": "pjm_solar"}

p = argparse.ArgumentParser()
p.add_argument("--load")
p.add_argument("--wind")
p.add_argument("--solar")
p.add_argument("--actuals", action="store_true")
p.add_argument("--data", default=str(ROOT / "data"))
args = p.parse_args()
DATA = Path(args.data)


def custom(run):
    """q500 of a backpublished walk-forward whose context always ended by D-2 (24 h gap, daily steps)."""
    run = Path(run)
    m = json.loads((run / "val_walk_forward_metrics.json").read_text())
    assert m["walk_forward"] and m.get("context_gap_hours", 0) >= 24 and m.get("step_days", 1) == 1, run
    assert m.get("eval_every", 1) == 1 and m.get("context_select", "recent") == "recent", run
    pr = pd.read_parquet(run / "val_walk_forward_predictions.parquet")
    pr = pr.assign(ts=pd.to_datetime(pr["ts"])).set_index("ts")
    # the run's val split (= first forecast day) starts on or before 2023-10-01, so every replaced value is out
    # of sample, made from context that ended at least 24 h before its forecast day (= by the end of D-2)
    assert pr.index.min() <= A, (run, pr.index.min())
    return pr["q500"]


splits = {s: pd.read_parquet(DATA / "pjm-fe/v0" / f"{s}.parquet") for s in ("train", "val", "test")}
full = pd.concat(splits.values(), keys=splits.keys(), names=["split", "i"]).reset_index(level=0)
full["ts"] = pd.to_datetime(full["ts"])
win = ((full["ts"] >= A) & (full["ts"] < B)).to_numpy()
ts = full["ts"]
L, W, S = full["pjm_load"], full["pjm_wind"], full["pjm_solar"]

# ---- verify the derived-input formulas on v0 (inside the replacement window)
sdc = full["solarduckcurve.forecast.pjm"]
checks = {
    "sdc = L - S": (sdc - (L - S)).abs() < 1e-6,
    "sdc_lead1_diff = sdc - sdc(t-1)": (full["solarduckcurve_lead1_diff.forecast.pjm"] - (sdc - sdc.shift(1))).abs() < 1e-6,
    "solar_lag1_diff = S(t+1) - S(t)": (full["solar_lag1_diff"] - (S.shift(-1) - S)).abs() < 1e-6,
}
for k, ok in checks.items():
    frac = ok[win].mean()
    print(f"check {k}: {frac:.4f} of window rows")
    assert frac > 0.995, k
k_ts = full["thermal_stress_all_pjm"] / (L - W - S)
print(f"thermal stress factor k in window: min {k_ts[win].min():.3f} median {k_ts[win].median():.3f} max {k_ts[win].max():.3f}")

# ---- replacement series
bench = pd.read_parquet(DATA / "pjm_renew/bench.parquet").assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts")
iso = {"load": bench["iso_load"], "wind": bench["iso_wind"], "solar": bench["iso_solar"]}
if args.actuals:
    name, tag = "v0_actuals_all", "actual"
    ld = pd.read_parquet(DATA / "pjm_load/load.parquet")
    ld = ld[ld["zone"] == "ALL"]
    new = {"load": ld.assign(ts=ts_of(ld)).groupby("ts")["load_mwh"].mean().astype(float),
           "wind": bench["wind"], "solar": bench["solar"]}
else:
    assert args.load and args.wind and args.solar, "--load, --wind and --solar runs are required"
    name, tag = "v0_custom_all", "custom"
    new = {"load": custom(args.load), "wind": custom(args.wind), "solar": custom(args.solar).clip(lower=0)}

out = full.copy()
for t, c in COL.items():
    v = new[t].reindex(ts[win]).to_numpy()
    miss = np.isnan(v) & full.loc[win, c].notna().to_numpy()
    print(f"{t}: window rows {win.sum()}, replacement missing where v0 has a value: {miss.sum()} (kept Meteologica)")
    assert miss.sum() <= 6, f"{t}: too many missing hours"
    out.loc[win, c] = np.where(np.isnan(v), full.loc[win, c].to_numpy(), v)

L2, W2, S2 = out["pjm_load"], out["pjm_wind"], out["pjm_solar"]
out["thermal_stress_all_pjm"] = np.where(win, k_ts * (L2 - W2 - S2), full["thermal_stress_all_pjm"])
sdc2 = L2 - S2
out["solarduckcurve.forecast.pjm"] = np.where(win, sdc2, sdc)
out["solarduckcurve_lead1_diff.forecast.pjm"] = np.where(win, sdc2 - sdc2.shift(1), full["solarduckcurve_lead1_diff.forecast.pjm"])
out["solar_lag1_diff"] = np.where(win, S2.shift(-1) - S2, full["solar_lag1_diff"])

for t, c in COL.items():
    i = iso[t].reindex(ts).to_numpy()
    out[f"{t}_{tag}_minus_iso"] = np.where(win, out[c] - i, np.nan)
    if args.actuals:
        out[f"{t}_actual_minus_met"] = np.where(win, out[c] - full[c], np.nan)

dst = DATA / "pjm-fe" / name
dst.mkdir(parents=True, exist_ok=True)
for s, part in out.groupby("split", sort=False):
    part = part.drop(columns="split")
    part["ts"] = splits[s]["ts"].to_numpy()  # keep the original dtype
    part.to_parquet(dst / f"{s}.parquet", index=False)
v = out[out["split"] == "val"]
for t, c in COL.items():
    d = (v[c] - splits["val"][c].to_numpy())
    print(f"{name}: 2025 {t} |new - Meteologica| mean {d.abs().mean():.0f} MW, mean {d.mean():+.0f}; "
          f"{t}_{tag}_minus_iso missing in 2025: {v[f'{t}_{tag}_minus_iso'].isna().sum()}")
print(f"{name}: thermal stress changed rows {((out['thermal_stress_all_pjm'] - full['thermal_stress_all_pjm']).abs() > 1e-9).sum()}, "
      f"columns {out.shape[1] - 1} -> {dst}")
