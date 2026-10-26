"""RT datasets with capacity headroom: PJM ALL available generator capacity (economic max, Data Miner day_gen_capacity) minus load.

  uv run python research/pjm_rt/make_headroom.py

Source: pjm_gen_capacities (zone ALL; total_committed, economic_max, emergency_max), hourly, data/pjm_rt/gen_cap.parquet.
Publication: Data Miner day_gen_capacity is posted once a day for the whole previous day (our 10:00 EPT daily
download on 2026-10-26 held data through 2026-10-25 HE24 and nothing of 10-26), so at the D-1 11:00 EPT cutoff the
latest complete day is D-2. total_committed is the RPM capacity commitment (flat for weeks) and is not used.
  v0_head_rt    v0_rt + inputs from D-2 (known by the cutoff):
                  em_d2         mean hourly economic max on D-2
                  em_d2_h       economic max in the same hour on D-2
                  headroom_fc   em_d2_h - Meteologica hourly load forecast (v0's pjm_load)
                  headroom_fc_net  em_d2_h - Meteologica net load (load - wind - solar)
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
d2 = g["economic_max"].groupby(g.index.normalize()).mean()
d2.index = d2.index + pd.Timedelta("2D")      # D-2's mean, indexed by D
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
        df["em_d2"] = d2.reindex(D).to_numpy()
        df["em_d2_h"] = g["economic_max"].reindex(ts - pd.Timedelta("48h")).to_numpy()
        df["headroom_fc"] = df["em_d2_h"] - df["pjm_load"]
        df["headroom_fc_net"] = df["em_d2_h"] - (df["pjm_load"] - df["pjm_wind"] - df["pjm_solar"])
        if pf:
            df["headroom_act"] = g["economic_max"].reindex(ts).to_numpy() - L.reindex(ts).to_numpy()
        df.to_parquet(dst / f"{s}.parquet", index=False)
        new = [c for c in df if c in ("em_d2", "em_d2_h", "headroom_fc", "headroom_fc_net", "headroom_act")]
        print(f"{name}/{s}: rows {len(df)}, missing " + ", ".join(f"{c} {int(df[c].isna().sum())}" for c in new)
              + f"; headroom_fc mean {df['headroom_fc'].mean():.0f}")
