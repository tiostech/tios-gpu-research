"""Energy-model datasets with the Kumo PJM load forecast in place of Meteologica ALL (pjm_load).

  uv run python research/pjm_load/make_customload.py <load_run_dir> [data_root]   (data_root default: ./data;
  on the box: /scratch/data)

<load_run_dir> = backpublished load walk-forward (val 2023-10-01 .. 2025-12-31, daily, --context-gap-hours
24), e.g. outputs/pjm-load-fe/P2_bp_gap24. Its q500 replaces v0's pjm_load for 2023-10-01 .. 2025-12-31
(the price baseline's 10k-row context never reaches back before ~2023-11-08). Earlier rows keep Meteologica.
  v0_customload_a  pjm_load replaced only.
  v0_customload_b  also recomputes the load-derived production features (formulas in
                   ep-forecast.dafc-20261001.pjm.json / TiosQ, verified exact on v0):
                     thermal_stress_all_pjm = cap / (cap - gen_on_outage) * (load - wind - solar)
                       -> rescaled with the row's own factor k = ts / (load - wind - solar)
                     solarduckcurve.forecast.pjm = load - solar
                     solarduckcurve_lead1_diff.forecast.pjm = sdc(t) - sdc(t - 1)
                   southwest_east_diff_duckcurve_pjm uses regional loads only and is unchanged.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
run = Path(sys.argv[1])
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data"
A, B = pd.Timestamp("2023-10-01"), pd.Timestamp("2026-01-01")

args = json.loads((run / "val_walk_forward_metrics.json").read_text())
assert args["walk_forward"] and args.get("context_gap_hours", 0) >= 24 and args.get("step_days", 1) == 1, args
assert not args.get("eval_every") or args["eval_every"] == 1, args
pred = pd.read_parquet(run / "val_walk_forward_predictions.parquet")
pred = pred.assign(ts=pd.to_datetime(pred["ts"])).set_index("ts")["q500"]
# the load run's val split starts on or before 2023-10-01, so every replaced value is out of sample and was
# made from context ending at least 24 h before the forecast day starts (= by the end of D-2)
vs = pd.read_parquet(DATA / "pjm-load-fe/P2_with_meteo_bp/val.parquet", columns=["ts"])["ts"].min()
assert pd.Timestamp(vs) <= A, vs

splits = {s: pd.read_parquet(DATA / "pjm-fe/v0" / f"{s}.parquet") for s in ("train", "val", "test")}
full = pd.concat(splits.values(), keys=splits.keys(), names=["split", "i"]).reset_index(level=0)
full["ts"] = pd.to_datetime(full["ts"])
win = (full["ts"] >= A) & (full["ts"] < B)
new = pred.reindex(full.loc[win, "ts"]).to_numpy()
miss = np.isnan(new) & full.loc[win, "pjm_load"].notna().to_numpy()
print(f"replace window rows {win.sum()}, load preds missing where v0 has pjm_load: {miss.sum()}")
assert miss.sum() <= 2, "load forecast missing hours"
new = np.where(np.isnan(new), full.loc[win, "pjm_load"].to_numpy(), new)

L, W, S = full["pjm_load"], full["pjm_wind"], full["pjm_solar"]
sdc_old = full["solarduckcurve.forecast.pjm"]
chk = ((sdc_old - (L - S)).abs() < 1e-6) | sdc_old.isna()
print(f"check sdc = load - solar: {chk.mean():.4f}")
lead = full["solarduckcurve_lead1_diff.forecast.pjm"]
chk2 = ((lead - (sdc_old - sdc_old.shift(1))).abs() < 1e-6)[win]
print(f"check sdc_lead1_diff = sdc - sdc(t-1) in window: {chk2.mean():.4f}")

a = full.copy()
a.loc[win, "pjm_load"] = new
b = a.copy()
k = full["thermal_stress_all_pjm"] / (L - W - S)
b["thermal_stress_all_pjm"] = np.where(win, k * (b["pjm_load"] - W - S), full["thermal_stress_all_pjm"])
sdc_new = b["pjm_load"] - S
b["solarduckcurve.forecast.pjm"] = np.where(win, sdc_new, sdc_old)
b["solarduckcurve_lead1_diff.forecast.pjm"] = np.where(win, sdc_new - sdc_new.shift(1), lead)

for name, df in (("v0_customload_a", a), ("v0_customload_b", b)):
    out = DATA / "pjm-fe" / name
    out.mkdir(parents=True, exist_ok=True)
    for s, part in df.groupby("split", sort=False):
        part = part.drop(columns="split")
        part["ts"] = splits[s]["ts"].to_numpy()  # keep the original dtype
        part.to_parquet(out / f"{s}.parquet", index=False)
    v = df[df["split"] == "val"]
    d = (v["pjm_load"] - splits["val"]["pjm_load"].to_numpy()).abs()
    print(f"{name}: 2025 |custom - Meteologica| mean {d.mean():.0f} MW; "
          f"thermal stress changed rows {(df['thermal_stress_all_pjm'] != full['thermal_stress_all_pjm']).sum()}")
