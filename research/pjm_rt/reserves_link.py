"""Which RT reserve clearing prices go with RT energy spikes? (same hour, perfect foresight; 2023-2025, no 2026)

  uv run python research/pjm_rt/reserves_link.py

Hourly features from data/pjm_rt/reserves (collect_reserves.R), RTO ("ALL") and Mid-Atlantic/Dominion (MAD):
  {sr,pr,tmr}_{rto,mad}_mean / _max   synchronized / primary / 30-min reserve MCP, hourly mean and max of 5-min
  {..}_short                          5-min intervals in shortage
  {..}_margin                         min over the hour of (reserve quantity - requirement)
  rmcp                                regulation MCP (hourly)
Spike = RT energy >= 150. AUC (and AUC within hour-of-day x month strata), rank correlation with RT, and how the
spike rate moves with the MCP. Writes outputs/pjm-rt/reserves/link.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "data/pjm_rt/reserves"
OUT = ROOT / "outputs/pjm-rt/reserves"
TY = {"synchronized": "sr", "primary": "pr", "thirty-minute reserve": "tmr"}
ZN = {"ALL": "rto", "Mid-Atlantic/Dominion": "mad"}


def hourly():
    r = pd.read_parquet(R / "reserves.parquet")
    r["ts"] = pd.to_datetime(r["dt"]) + pd.to_timedelta(r["hr"].astype(int) - 1, unit="h")
    r["key"] = r["res_type"].map(TY) + "_" + r["zone"].map(ZN)
    r["margin"] = r["min_margin"]
    w = r.pivot_table(index="ts", columns="key", values=["mcp_mean", "mcp_max", "n_short", "margin"], aggfunc="first")
    w.columns = [f"{k}_{v.replace('mcp_', '').replace('n_', '')}" for v, k in w.columns]
    g = pd.read_parquet(R / "regulation.parquet")
    g["ts"] = pd.to_datetime(g["dt"]) + pd.to_timedelta(g["hr"].astype(int) - 1, unit="h")
    g = g.drop_duplicates("ts").set_index("ts")
    w["rmcp"] = g["rmcp"].reindex(w.index)
    return w.sort_index()


if __name__ == "__main__":
    X = hourly()
    raw = pd.read_parquet(ROOT / "data/pjm_raw.parquet")
    raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts")
    rows = []
    for y0, y1 in (("2025-01-01", "2026-01-01"), ("2023-01-01", "2026-01-01")):
        idx = pd.date_range(y0, pd.Timestamp(y1) - pd.Timedelta("1h"), freq="h")
        rt, da = raw["rt_energy_price"].reindex(idx), raw["da_energy_price"].reindex(idx)
        F = X.reindex(idx)
        sp = rt >= 150
        strata = pd.Series(idx.hour.astype(str) + "_" + idx.month.astype(str), idx)
        print(f"\n== {y0[:4]}..{int(y1[:4]) - 1}: {int(sp.sum())} spike hours of {int(rt.notna().sum())}")
        for c in sorted(F.columns):
            m = rt.notna() & F[c].notna()
            if m.sum() < 1000 or F[c][m].nunique() < 3:
                continue
            s = -F[c] if c.endswith("_margin") else F[c]      # lower margin = tighter
            auc = roc_auc_score(sp[m], s[m])
            aucs, wts = [], []
            for _, ix in s[m].groupby(strata[m]).groups.items():
                yy = sp[m][ix]
                if 0 < yy.sum() < len(yy):
                    aucs.append(roc_auc_score(yy, s[m][ix])), wts.append(yy.sum())
            sauc = np.average(aucs, weights=wts) if aucs else np.nan
            pos = m & (F[c] > (0 if not c.endswith("_margin") else -np.inf)) if not c.endswith("_margin") else m & (F[c] <= 0)
            rows.append({"period": y0[:4] + "-" + str(int(y1[:4]) - 1), "feature": c, "n": int(m.sum()), "auc": auc, "strat_auc": sauc,
                         "spearman": F[c][m].corr(rt[m], method="spearman"),
                         "share_pos": pos.sum() / m.sum(), "spike_rate_pos": sp[pos].mean(), "spike_rate_else": sp[m & ~pos].mean(),
                         "spikes_covered": (sp & pos).sum() / (sp & m).sum()})
        d = pd.DataFrame([r for r in rows if r["period"].startswith(y0[:4])]).sort_values("strat_auc", ascending=False)
        print(d.drop(columns="period").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        # DA price as the reference signal
        m = rt.notna() & da.notna()
        print(f"  reference: DA price AUC {roc_auc_score(sp[m], da[m]):.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "link.csv", index=False)
