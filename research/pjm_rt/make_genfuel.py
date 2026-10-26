"""RT datasets with actual PJM generation by fuel (perfect-foresight diagnostics; leak the future).

  uv run python research/pjm_rt/make_genfuel.py

Source: pjm_hourly_generation_by_types (Data Miner gen_by_fuel), data/pjm_rt/gen_by_fuel.parquet, from 2023-06.
Columns are filled from 2023-10-01 to 2025-12-31 (as make_custominputs.py: the RT baseline's 10k-hour context never
reaches before ~2023-11) and missing before. Wind and solar are left out: their actuals were tested separately.
  v0_genfuel_pf_rt   v0_rt + actual gen_gas, gen_coal, gen_nuclear, gen_oil, gen_hydro, gen_other (multiple fuels,
                     other renewables, storage) and net_imports = actual ALL load - total generation.
                     Upper bound, partly circular: gas and oil output respond to the RT price itself.
  v0_supply_pf_rt    v0_rt + actual gen_nuclear and net_imports only: supply-side shocks (nuclear trips, neighbours)
                     that are mostly not PJM's own dispatch response, so closer to what a forecast could deliver.
  v0_stack_pf_rt     v0_rt + supply-stack slack, PJM data only (IIR is too stale / imprecise): spare_{gas, coal, oil,
                     hydro, nuclear} = the fuel's highest hourly output over the previous 30 days (practical capacity)
                     minus its actual output; system_slack = available capacity (day_gen_capacity economic max) minus
                     total generation; forced_outages_fc = PJM's forced-outage report (published D-1 06:00, known).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/spp_load"))
from make_features import ts_of  # noqa: E402

DATA = ROOT / "data"
A, B = pd.Timestamp("2023-10-01"), pd.Timestamp("2026-01-01")
g = pd.read_parquet(DATA / "pjm_rt/gen_by_fuel.parquet")
p = g.assign(ts=ts_of(g)).pivot_table(index="ts", columns="fuel_type", values="power_mw", aggfunc="mean")
ld = pd.read_parquet(DATA / "pjm_load/load.parquet")
ld = ld[ld["zone"] == "ALL"]
L = ld.assign(ts=ts_of(ld)).groupby("ts")["load_mwh"].mean()
cols = pd.DataFrame({
    "gen_gas": p["Gas"], "gen_coal": p["Coal"], "gen_nuclear": p["Nuclear"], "gen_oil": p["Oil"], "gen_hydro": p["Hydro"],
    "gen_other": p["Multiple Fuels"] + p["Other Renewables"] + p["Storage"],
    "net_imports": L.reindex(p.index) - p.sum(axis=1),
})
for fu, nm in (("Gas", "gas"), ("Coal", "coal"), ("Oil", "oil"), ("Hydro", "hydro"), ("Nuclear", "nuclear")):
    cols[f"spare_{nm}"] = p[fu].shift(1).rolling(720, min_periods=600).max() - p[fu]
gc = pd.read_parquet(DATA / "pjm_rt/gen_cap.parquet")
cols["system_slack"] = gc.assign(ts=ts_of(gc)).groupby("ts")["economic_max"].mean().reindex(p.index) - p.sum(axis=1)
og = pd.read_parquet(DATA / "pjm_rt/outages_all.parquet")
cols["forced_outages_fc"] = og.assign(ts=ts_of(og)).groupby("ts")["forced_outages_mw"].mean().reindex(p.index)
SETS = {"v0_genfuel_pf_rt": ["gen_gas", "gen_coal", "gen_nuclear", "gen_oil", "gen_hydro", "gen_other", "net_imports"],
        "v0_supply_pf_rt": ["gen_nuclear", "net_imports"],
        "v0_stack_pf_rt": ["spare_gas", "spare_coal", "spare_oil", "spare_hydro", "spare_nuclear", "system_slack", "forced_outages_fc"]}

import sys as _s
only = _s.argv[1:]
for name, use in SETS.items():
    if only and name not in only:
        continue
    dst = DATA / "pjm-fe" / name
    dst.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        df = pd.read_parquet(DATA / "pjm-fe/v0_rt" / f"{s}.parquet")
        ts = pd.to_datetime(df["ts"])
        win = ((ts >= A) & (ts < B)).to_numpy()
        for c in use:
            df[c] = np.where(win, cols[c].reindex(ts).to_numpy(), np.nan)
        df.to_parquet(dst / f"{s}.parquet", index=False)
        print(f"{name}/{s}: rows {len(df)}, window rows {win.sum()}, missing in window "
              + ", ".join(f"{c} {int(np.isnan(df[c].to_numpy()[win]).sum())}" for c in use))
