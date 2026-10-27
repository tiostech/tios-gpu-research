"""PJM DA datasets with Kumo's backpublished MISO and SPP DA energy forecasts as inputs (+ congestion PCs).

  uv run python research/pjm_xmkt/make_xmkt.py [--base v0_congfc10] [--name v0_congfc10_xmkt]

MISO / SPP DA results for day D come out in the afternoon of D-1, after PJM's 11:00 EPT cutoff, so only a forecast
can be used. The forecasts are the q500s of daily walk-forwards from 2023-10-01 (outputs/miso-fe/m0_original_bp_daily,
outputs/spp-fe/s0_original_bp_daily: the production-RF inputs, Kumo-Tabular large, 8 estimators, seed 0), each made
from inputs at the neighbour's fair vintage (25 min before its own bid deadline: SPP 09:05 CPT = 10:05 EPT; MISO
10:05 EST = 10:05 EPT in winter, 11:05 EDT in summer, 5 minutes after PJM's cutoff in summer).
Alignment: same instant. PJM ts (EPT, prevailing) -> MISO fixed EST, SPP Central prevailing. Columns, filled from
2023-10-01 to 2025-12-31 and missing before (as the other PJM diagnostics):
  miso_da_fc, spp_da_fc    neighbour DA energy forecast (q500) for the same hour
  miso_da_fc_q90, spp_da_fc_q90   their 90th percentiles (upside risk next door)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--base", default="v0_congfc10")
p.add_argument("--name", default="v0_congfc10_xmkt")
a = p.parse_args()
DATA = ROOT / "data"
A, B = pd.Timestamp("2023-10-01"), pd.Timestamp("2026-01-01")
SRC = {"miso": ("outputs/miso-fe/m0_original_bp_daily", "Etc/GMT+5"),
       "spp": ("outputs/spp-fe/s0_original_bp_daily", "America/Chicago")}


def run(path):
    path = ROOT / path
    m = json.loads((path / "val_walk_forward_metrics.json").read_text())
    assert m["walk_forward"] and m.get("step_days", 1) == 1 and m.get("eval_every", 1) == 1, path
    pr = pd.read_parquet(path / "val_walk_forward_predictions.parquet")
    pr = pr.assign(ts=pd.to_datetime(pr["ts"])).drop_duplicates("ts").set_index("ts")
    assert pr.index.min() <= A, (path, pr.index.min())     # every filled value is out of sample
    return pr


P = {k: run(v[0]) for k, v in SRC.items()}
dst = DATA / "pjm-fe" / a.name
dst.mkdir(parents=True, exist_ok=True)
for s in ("train", "val", "test"):
    df = pd.read_parquet(DATA / "pjm-fe" / a.base / f"{s}.parquet")
    ts = pd.DatetimeIndex(pd.to_datetime(df["ts"]))
    win = ((ts >= A) & (ts < B))
    u = ts.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    for k, (_, tz) in SRC.items():
        key = u.tz_convert(tz).tz_localize(None)
        for col, q in ((f"{k}_da_fc", "q500"), (f"{k}_da_fc_q90", "q900")):
            v = P[k][q].reindex(key).to_numpy()
            df[col] = np.where(win, v, np.nan)
    df.to_parquet(dst / f"{s}.parquet", index=False)
    if s != "test":
        print(f"{a.name}/{s}: rows {len(df)}, window rows {int(win.sum())}, missing in window "
              + ", ".join(f"{c} {int(np.isnan(df[c].to_numpy()[win]).sum())}" for c in ("miso_da_fc", "spp_da_fc")))
