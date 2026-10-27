"""PJM DA datasets with Kumo's backpublished MISO and SPP DA energy forecasts as inputs (+ congestion PCs).

  uv run python research/pjm_xmkt/make_xmkt.py [--base v0_congfc10] [--name v0_congfc10_xmkt] [--data D --outputs O]
  (on the GPU box: --data /scratch/data --outputs /scratch/outputs)

MISO / SPP DA results for day D come out in the afternoon of D-1, after PJM's 11:00 EPT cutoff, so only a forecast
can be used. The forecasts are the q500s of daily walk-forwards from 2023-10-01 (outputs/miso-fe/m0_original_bp_daily,
outputs/spp-fe/s0_original_bp_daily: the production-RF inputs, Kumo-Tabular large, 8 estimators, seed 0), each made
from inputs at the neighbour's fair vintage (25 min before its own bid deadline: SPP 09:05 CPT = 10:05 EPT; MISO
10:05 EST = 10:05 EPT in winter, 11:05 EDT in summer, 5 minutes after PJM's cutoff in summer).
Alignment: same instant. PJM ts (EPT, prevailing) -> MISO fixed EST, SPP Central prevailing. Columns, filled from
2023-10-01 to 2025-12-31 and missing before (as the other PJM diagnostics):
  miso_da_fc, spp_da_fc    neighbour DA energy forecast (q500) for the same hour
  miso_da_fc_q90, spp_da_fc_q90   their 90th percentiles (upside risk next door)
  --quantiles 050,500,900,950 (v0_congfc10_xmktq) adds *_da_fc_q05 and *_da_fc_q95 (each neighbour's range).
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
p.add_argument("--data", default=str(ROOT / "data"))
p.add_argument("--end", default="2026-01-01", help="fill up to (exclusive); > 2026-01-01 adds the *_26_daily 2026 runs (test comparison)")
p.add_argument("--test-into-val", action="store_true", help="val = val + test, so one walk-forward covers 2025 and 2026")
p.add_argument("--quantiles", default="500,900", help="neighbour quantiles, per mille: 500 -> {m}_da_fc, else {m}_da_fc_qNN")
p.add_argument("--outputs", default=str(ROOT / "outputs"), help="dir holding miso-fe/ and spp-fe/ runs")
a = p.parse_args()
DATA, OUTS = Path(a.data), Path(a.outputs)
A, B = pd.Timestamp("2023-10-01"), pd.Timestamp(a.end)
SRC = {"miso": ("miso-fe/m0_original_bp_daily", "Etc/GMT+5"),
       "spp": ("spp-fe/s0_original_bp_daily", "America/Chicago")}


def run(path):
    path = OUTS / path
    m = json.loads((path / "val_walk_forward_metrics.json").read_text())
    assert m["walk_forward"] and m.get("step_days", 1) == 1 and m.get("eval_every", 1) == 1, path
    pr = pd.read_parquet(path / "val_walk_forward_predictions.parquet")
    pr = pr.assign(ts=pd.to_datetime(pr["ts"])).drop_duplicates("ts").set_index("ts")
    assert pr.index.min() <= A, (path, pr.index.min())     # every filled value is out of sample
    return pr


P = {k: run(v[0]) for k, v in SRC.items()}
if B > pd.Timestamp("2026-01-01"):                          # continue each backpublish with its 2026 walk-forward
    for k, run26 in (("miso", "miso-fe/m0_original_26_daily"), ("spp", "spp-fe/s0_original_26_daily")):
        p26 = pd.read_parquet(OUTS / run26 / "val_walk_forward_predictions.parquet")
        p26 = p26.assign(ts=pd.to_datetime(p26["ts"])).drop_duplicates("ts").set_index("ts")
        assert p26.index.min() <= pd.Timestamp("2026-01-01 01:00"), (run26, p26.index.min())
        P[k] = pd.concat([P[k][P[k].index < p26.index.min()], p26])
dst = DATA / "pjm-fe" / a.name
dst.mkdir(parents=True, exist_ok=True)
split = {s: pd.read_parquet(DATA / "pjm-fe" / a.base / f"{s}.parquet") for s in ("train", "val", "test")}
if a.test_into_val:
    split["val"], split["test"] = pd.concat([split["val"], split["test"]], ignore_index=True), split["test"].iloc[:0]
for s in ("train", "val", "test"):
    df = split[s]
    ts = pd.DatetimeIndex(pd.to_datetime(df["ts"]))
    win = ((ts >= A) & (ts < B))
    u = ts.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    for k, (_, tz) in SRC.items():
        key = u.tz_convert(tz).tz_localize(None)
        for qq in a.quantiles.split(","):
            q = f"q{int(qq):03d}"
            col = f"{k}_da_fc" if q == "q500" else f"{k}_da_fc_q{int(qq) // 10:02d}"
            v = P[k][q].reindex(key).to_numpy()
            df[col] = np.where(win, v, np.nan)
    df.to_parquet(dst / f"{s}.parquet", index=False)
    if s != "test":
        print(f"{a.name}/{s}: rows {len(df)}, window rows {int(win.sum())}, missing in window "
              + ", ".join(f"{c} {int(np.isnan(df[c].to_numpy()[win]).sum())}" for c in ("miso_da_fc", "spp_da_fc")))
