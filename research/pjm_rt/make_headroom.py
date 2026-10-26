"""RT datasets with committed-capacity headroom (PJM ALL economic max of committed units minus load).

  uv run python research/pjm_rt/make_headroom.py

Source: pjm_gen_capacities (zone ALL; total_committed, economic_max, emergency_max), hourly, data/pjm_rt/gen_cap.parquet.
  v0_head_rt    v0_rt + inputs known by D-1 11:00 EPT, taken from D-1 hours beginning 00-08 (ending by 09:00, which
                allows two hours for publication; assumption, PJM posts these hourly):
                  em_d1am       mean economic max of committed units, D-1 00-08
                  tc_d1am       mean total committed, D-1 00-08
                  headroom_fc   em_d1am - Meteologica hourly load forecast (v0's pjm_load)
                  headroom_fc_net  em_d1am - Meteologica net load (load - wind - solar)
  v0_headpf_rt  v0_head_rt + realised hourly headroom_act = economic_max - actual ALL load (perfect-foresight
                diagnostic; leaks the future)
Every row (context and forecast) uses the same construction, so the model learns the relation it will use.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/spp_load"))
from make_features import ts_of  # noqa: E402

DATA = ROOT / "data"
g = pd.read_parquet(DATA / "pjm_rt/gen_cap.parquet")
g = g.assign(ts=ts_of(g)).groupby("ts")[["total_committed", "economic_max"]].mean().astype(float)
am = g[g.index.hour <= 8]
d1 = am.groupby(am.index.normalize()).mean()
d1.index = d1.index + pd.Timedelta("1D")      # D-1's morning, indexed by D
ld = pd.read_parquet(DATA / "pjm_load/load.parquet")
ld = ld[ld["zone"] == "ALL"]
L = ld.assign(ts=ts_of(ld)).groupby("ts")["load_mwh"].mean().astype(float)

for name, pf in (("v0_head_rt", False), ("v0_headpf_rt", True)):
    dst = DATA / "pjm-fe" / name
    dst.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        df = pd.read_parquet(DATA / "pjm-fe/v0_rt" / f"{s}.parquet")
        ts = pd.to_datetime(df["ts"])
        D = ts.dt.normalize()
        df["em_d1am"] = d1["economic_max"].reindex(D).to_numpy()
        df["tc_d1am"] = d1["total_committed"].reindex(D).to_numpy()
        df["headroom_fc"] = df["em_d1am"] - df["pjm_load"]
        df["headroom_fc_net"] = df["em_d1am"] - (df["pjm_load"] - df["pjm_wind"] - df["pjm_solar"])
        if pf:
            df["headroom_act"] = g["economic_max"].reindex(ts).to_numpy() - L.reindex(ts).to_numpy()
        df.to_parquet(dst / f"{s}.parquet", index=False)
        new = [c for c in df if c in ("em_d1am", "tc_d1am", "headroom_fc", "headroom_fc_net", "headroom_act")]
        print(f"{name}/{s}: rows {len(df)}, missing " + ", ".join(f"{c} {int(df[c].isna().sum())}" for c in new)
              + f"; headroom_fc mean {df['headroom_fc'].mean():.0f}")
