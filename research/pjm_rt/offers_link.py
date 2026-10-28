"""Does DA generator offer behaviour explain the DA price's RT-spike edge? (diagnostic, 2025)

  uv run python research/pjm_rt/offers_link.py

data/pjm_rt/offers (collect_offers.R): DA offer curves for ~350 bidding groups per hour (10 cumulative MW / price
segments). Hourly stack features (hour ending -> hour beginning):
  cap            total MW offered
  mw_ge_100/200  MW offered at >= $100 / $200 (expensive tail of the stack), and as a share of cap
  wavg           MW-weighted mean offer price
  p90, p95       price at 90% / 95% of the aggregated offered MW (how steep the top of the stack is)
  markup         MW-weighted mean over units of log(unit's daily mean offer / its median over the previous 30 days):
                 offers above the unit's own recent norm (behaviour change, incl. fuel cost); markup_gas = gas units only
Within groups of similar Kumo q95 (v0_congfc5_xmkt_rt26_daily), does each separate RT >= 150 hours, how does it rank
with the DA price, and does the DA price's edge survive controlling for it?
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
o = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/pjm_rt/offers").glob("2025-*.parquet"))], ignore_index=True)
o["ts"] = pd.to_datetime(o["dt"]) + pd.to_timedelta(o["hr"].astype(int) - 1, unit="h")
B = o[[f"bid{k}" for k in range(1, 11)]].to_numpy(float)
W = o[[f"mw{k}" for k in range(1, 11)]].to_numpy(float)
valid = W > 0
W = np.where(valid, np.maximum.accumulate(np.where(valid, W, 0), axis=1), 0)
inc = np.diff(np.concatenate([np.zeros((len(W), 1)), W], axis=1), axis=1) * valid
inc = np.clip(inc, 0, None)
o["cap"] = inc.sum(1)
for p in (100, 200):
    o[f"mw_ge_{p}"] = (inc * (B >= p)).sum(1)
o["pw"] = (inc * B).sum(1)
o["unit_mean"] = o["pw"] / o["cap"].replace(0, np.nan)
o["gas"] = o["fuel"].fillna("").str.contains("NG|gas|Gas", regex=True)

g = o.groupby("ts")
H = pd.DataFrame({"cap": g["cap"].sum(), "mw_ge_100": g["mw_ge_100"].sum(), "mw_ge_200": g["mw_ge_200"].sum()})
H["wavg"] = g["pw"].sum() / H["cap"]
H["share_ge_100"], H["share_ge_200"] = H.mw_ge_100 / H.cap, H.mw_ge_200 / H.cap

# price at 90% / 95% of the aggregated stack
seg = pd.DataFrame({"ts": np.repeat(o["ts"].to_numpy(), 10), "p": B.ravel(), "mw": inc.ravel()})
seg = seg[seg.mw > 0].sort_values(["ts", "p"])
seg["cum"] = seg.groupby("ts")["mw"].cumsum() / seg["ts"].map(H["cap"])
for q in (0.90, 0.95):
    H[f"p{int(q * 100)}"] = seg[seg.cum >= q].groupby("ts")["p"].first()

# unit markup vs its own previous-30-day median (daily)
ud = o.assign(day=pd.to_datetime(o["dt"])).groupby(["unit", "day"]).agg(m=("unit_mean", "mean"), cap=("cap", "mean"), gas=("gas", "first"))
ud = ud.reset_index().sort_values(["unit", "day"])
ud["ref"] = ud.groupby("unit")["m"].transform(lambda s: s.shift(1).rolling(30, min_periods=15).median())
ud["lr"] = np.log(ud["m"].clip(lower=1) / ud["ref"].clip(lower=1))
ok = ud["lr"].notna() & (ud["cap"] > 0)
mk = ud[ok].groupby("day").apply(lambda d: np.average(d.lr, weights=d.cap))
mkg = ud[ok & ud.gas].groupby("day").apply(lambda d: np.average(d.lr, weights=d.cap))
H["markup"] = H.index.normalize().map(mk)
H["markup_gas"] = H.index.normalize().map(mkg)

raw = pd.read_parquet(ROOT / "data/pjm_raw.parquet")
raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts")
idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
rt, da = raw["rt_energy_price"].reindex(idx), raw["da_energy_price"].reindex(idx)
K = pd.read_parquet(ROOT / "outputs/pjm-fe/v0_congfc5_xmkt_rt26_daily/val_walk_forward_predictions.parquet")
K = K.assign(ts=pd.to_datetime(K["ts"])).drop_duplicates("ts").set_index("ts").reindex(idx)
H = H.reindex(idx)
print(f"units per hour {o.groupby('ts')['unit'].nunique().median():.0f}, offered MW per hour median {H.cap.median():,.0f} "
      f"(gas share {o[o.gas].cap.sum() / o.cap.sum():.0%})")
m = rt.notna() & da.notna() & K["q950"].notna() & H[["cap", "wavg", "p90", "markup"]].notna().all(axis=1)
sp = (rt >= 150)[m]
dec = pd.Series(pd.qcut(K["q950"][m].rank(method="first"), 10, labels=False).to_numpy(), sp.index)
hm = pd.Series(idx.hour.astype(str) + "_" + idx.month.astype(str), idx)[m]


def within(score, groups):
    au, w = [], []
    for _, ix in sp.groupby(groups).groups.items():
        yy = sp[ix]
        if 0 < yy.sum() < len(yy) and score[ix].nunique() > 1:
            au.append(roc_auc_score(yy, score[ix])), w.append(yy.sum())
    return np.average(au, weights=w)


print(f"{int(sp.sum())} spike hours; DA price AUC within Kumo-q95 deciles {within(da[m], dec):.3f}\n")
print(f"{'feature':13} {'AUC':>6} {'hr x mo':>8} {'in Kumo dec':>11} {'DA|ctl':>7} {'rho(DA) in dec':>14}")
for c in ("cap", "mw_ge_100", "mw_ge_200", "share_ge_100", "share_ge_200", "wavg", "p90", "p95", "markup", "markup_gas"):
    f = H[c][m]
    if f.isna().any():
        f = f.fillna(f.median())
    ter = pd.Series(pd.qcut(f.rank(method="first"), 3, labels=False).to_numpy(), sp.index)
    rho = np.nanmean([f[dec == k].corr(da[m][dec == k], method="spearman") for k in range(10)])
    print(f"{c:13} {roc_auc_score(sp, f):6.3f} {within(f, hm):8.3f} {within(f, dec):11.3f} "
          f"{within(da[m], dec.astype(str) + '_' + ter.astype(str)):7.3f} {rho:+14.2f}")

# how much of the markup is fuel cost? compare with the gas price relative to its own previous-30-day median
ds = pd.concat([pd.read_parquet(ROOT / f"data/pjm-fe/v0_congfc5_xmkt_rt26/{s}.parquet") for s in ("train", "val")])
gp = ds.assign(ts=pd.to_datetime(ds["ts"])).drop_duplicates("ts").set_index("ts")["blended_gasprice"]
gd = gp.groupby(gp.index.normalize()).mean()
grel = np.log(gd / gd.shift(1).rolling(30, min_periods=15).median())
gr = pd.Series(idx.normalize().map(grel), idx)[m]
print(f"\ncorr(markup_gas, log gas / its 30-day median) daily: {pd.Series(mkg).corr(grel.reindex(mkg.index)):.2f}")
print(f"gas_rel (fair: gas feature at the cutoff): AUC within Kumo deciles {within(gr, dec):.3f}; rho(DA) in dec "
      f"{np.nanmean([gr[dec == k].corr(da[m][dec == k], method='spearman') for k in range(10)]):+.2f}")
resid = H["markup_gas"][m] - np.polyval(np.polyfit(gr.fillna(0), H["markup_gas"][m].fillna(0), 1), gr.fillna(0))
print(f"markup_gas net of gas: AUC within Kumo deciles {within(resid, dec):.3f}")
