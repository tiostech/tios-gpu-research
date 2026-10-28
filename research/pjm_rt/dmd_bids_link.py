"""Do DA demand bids explain the DA price's RT-spike edge? (diagnostic; demand bids post with the DA results)

  uv run python research/pjm_rt/dmd_bids_link.py

data/pjm_rt/da_demand_bids.parquet: pjm_da_demands (Data Miner hrl_dmd_bids), RTO total DA demand bid MW, hour ending.
Compared with PJM's load forecast at the 11:00 cutoff (pjm_load in the RT dataset) and actual load (pjm_hourly_load):
  excess_fc  = demand bids - load forecast     (bidding above the public forecast: "informed longs"?)
  excess_act = demand bids - actual load
Within groups of similar Kumo q95 (v0_congfc5_xmkt_rt26_daily, 2025), does excess demand separate RT spikes, and does
the DA price's edge survive controlling for it?
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]


def ts(df):
    return pd.to_datetime(df["dt"]) + pd.to_timedelta(df["hr"].astype(int) - 1, unit="h")


b = pd.read_parquet(ROOT / "data/pjm_rt/da_demand_bids.parquet")
b = b[b["zone"] == "ALL"].assign(ts=ts).drop_duplicates("ts").set_index("ts")["demand_mw"]
a = pd.read_parquet(ROOT / "data/pjm_rt/rt_load_all.parquet").assign(ts=ts).drop_duplicates("ts").set_index("ts")["load_mw"]
ds = pd.concat([pd.read_parquet(ROOT / f"data/pjm-fe/v0_congfc5_xmkt_rt26/{s}.parquet") for s in ("train", "val")])
fc = ds.assign(ts=pd.to_datetime(ds["ts"])).drop_duplicates("ts").set_index("ts")["pjm_load"]
raw = pd.read_parquet(ROOT / "data/pjm_raw.parquet")
raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts")
rng = np.random.default_rng(0)


def within(score, y, groups):
    au, w = [], []
    for _, ix in y.groupby(groups).groups.items():
        yy = y[ix]
        if 0 < yy.sum() < len(yy):
            au.append(roc_auc_score(yy, score[ix])), w.append(yy.sum())
    return np.average(au, weights=w)


for y0, y1 in (("2025-01-01", "2026-01-01"), ("2023-01-01", "2026-01-01")):
    idx = pd.date_range(y0, pd.Timestamp(y1) - pd.Timedelta("1h"), freq="h")
    rt, da = raw["rt_energy_price"].reindex(idx), raw["da_energy_price"].reindex(idx)
    F = pd.DataFrame({"bid": b.reindex(idx), "fc": fc.reindex(idx), "act": a.reindex(idx)})
    F["excess_fc"], F["excess_act"], F["fc_err"] = F.bid - F.fc, F.bid - F.act, F.act - F.fc
    F["excess_fc_pct"] = 100 * F.excess_fc / F.fc
    m = rt.notna() & da.notna() & F.notna().all(axis=1)
    sp = (rt >= 150)[m]
    hm = pd.Series(idx.hour.astype(str) + "_" + idx.month.astype(str), idx)[m]
    print(f"\n== {y0[:4]}-{int(y1[:4]) - 1}: {int(sp.sum())} spike hours / {int(m.sum())}")
    print(f"  demand bids vs forecast: mean {F.excess_fc_pct[m].mean():+.2f}% (sd {F.excess_fc_pct[m].std():.2f}); spike hours {F.excess_fc_pct[m][sp].mean():+.2f}%, "
          f"others {F.excess_fc_pct[m][~sp].mean():+.2f}%")
    for c in ("bid", "excess_fc", "excess_fc_pct", "excess_act", "fc_err"):
        print(f"  {c:13} AUC {roc_auc_score(sp, F[c][m]):.3f} | within hour x month {within(F[c][m], sp, hm):.3f} | "
              f"rank corr with DA price within hour x month {np.nanmean([F[c][m][ix].corr(da[m][ix], method='spearman') for ix in hm.groupby(hm).groups.values()]):+.2f}")
    if y0.startswith("2025"):
        K = pd.read_parquet(ROOT / "outputs/pjm-fe/v0_congfc5_xmkt_rt26_daily/val_walk_forward_predictions.parquet")
        K = K.assign(ts=pd.to_datetime(K["ts"])).drop_duplicates("ts").set_index("ts").reindex(idx)[m]
        dec = pd.Series(pd.qcut(K["q950"].rank(method="first"), 10, labels=False).to_numpy(), sp.index)
        ex = F.excess_fc_pct[m]
        print(f"  within Kumo-q95 deciles: AUC excess_fc_pct {within(ex, sp, dec):.3f} | excess_act {within(F.excess_act[m], sp, dec):.3f} | "
              f"DA price {within(da[m], sp, dec):.3f}")
        ter = pd.Series(pd.qcut(ex.rank(method="first"), 3, labels=False).to_numpy(), sp.index)
        print(f"  DA price AUC within Kumo decile x excess-demand tercile: {within(da[m], sp, dec.astype(str) + '_' + ter.astype(str)):.3f}")
        rc = np.nanmean([ex[dec == k].corr(da[m][dec == k], method="spearman") for k in range(10)])
        print(f"  rank corr DA price vs excess demand within Kumo deciles: {rc:+.2f}")
        top = dec >= 8
        print(f"  in Kumo's top-2 deciles: excess demand on spike hours {ex[top & sp].mean():+.2f}% vs non-spike {ex[top & ~sp].mean():+.2f}%")
