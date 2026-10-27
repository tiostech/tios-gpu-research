"""Data for stakeholder Artifact 7: PJM DA energy forecast on the 2026 test set -- production DAFC vs the original
Kumo baseline vs Kumo with trader congestion forecasts and MISO / SPP DA forecasts.

  uv run python research/pjm_xmkt/artifact7_data.py

Models (all forecast day D by 11:00 EPT on D-1; scored on complete days):
  prod  production flex.daep.20261001 (data/pjm_extra/prod_flex_daep.parquet)
  orig  original Kumo baseline: 2026 = outputs/pjm-fe/baseline_v0_test_daily, 2025 = s3_stage/baseline_v0_daily
  new   v0_congfc5_xmkt26_daily: the baseline's inputs + 5 principal components of the trader congestion forecasts
        (PCA fitted 2023-10..2024-12) + Kumo's MISO and SPP DA energy forecasts (medians of daily walk-forwards from
        2023-10, continued through 2026). One walk-forward from 2025-01-01 to 2026-10-08.
"Trading days" = all days except Mondays (the desk does not trade Mondays). Paired day-block bootstrap (2,000 draws)
of daily MAE gains. 2026 was not used for any research choice; this page is its first scoring for these models.
Writes outputs/pjm-xmkt/artifact7/ JSON + HTML page.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pjm-xmkt/artifact7"
M = ("prod", "orig", "new")

prod = pd.read_parquet(ROOT / "data/pjm_extra/prod_flex_daep.parquet")
prod["ts"] = pd.to_datetime(prod["dt"]) + pd.to_timedelta(prod["hr"] - 1, unit="h")
prod = prod.drop_duplicates("ts").set_index("ts")["forecast"]


def preds(run):
    d = pd.read_parquet(ROOT / "outputs/pjm-fe" / run / "val_walk_forward_predictions.parquet")
    return d.assign(ts=pd.to_datetime(d["ts"])).drop_duplicates("ts").set_index("ts")


new = preds("v0_congfc5_xmkt26_daily")


def frame(orig_run, y0, y1):
    o = preds(orig_run)
    idx = o.index[(o.index >= y0) & (o.index < y1)]
    P = pd.DataFrame({"y": o["y_true"].reindex(idx), "prod": prod.reindex(idx), "orig": o["q500"].reindex(idx),
                      "new": new["q500"].reindex(idx), "orig_lo": o["q100"].reindex(idx), "orig_hi": o["q900"].reindex(idx),
                      "new_lo": new["q100"].reindex(idx), "new_hi": new["q900"].reindex(idx)})
    n = P.groupby(P.index.floor("D"))["y"].transform("count")
    return P[n >= 23]                                     # complete days


T = frame("baseline_v0_test_daily", pd.Timestamp("2026-01-01"), pd.Timestamp("2027-01-01"))
V = frame("s3_stage/baseline_v0_daily", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"))
assert (new["y_true"].reindex(T.index) - T["y"]).abs().max() < 1e-6, "targets disagree"


def paired(P, a, b, m, n=2000):
    """Daily MAE of b minus daily MAE of a (positive = a better), bootstrap SE, share of days a better."""
    day = P.index.floor("D")
    g = ((P["y"] - P[b]).abs() - (P["y"] - P[a]).abs())[m].groupby(day[m]).mean()
    rng = np.random.default_rng(0)
    bs = [g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n)]
    return {"gain": float(g.mean()), "se": float(np.std(bs)), "win": float((g > 0).mean())}


def score(P, m):
    s = {"days": int(P.index.floor("D")[m].nunique()), "hours": int(m.sum()),
         "start": str(P.index[m].min().date()), "end": str(P.index[m].max().date()),
         "ymean": float(P["y"][m].mean()), "ymax": float(P["y"][m].max())}
    for c in M:
        e = (P["y"] - P[c])[m]
        s[c] = {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}
    for c in ("orig", "new"):
        s[c]["cov80"] = float(((P["y"] >= P[f"{c}_lo"]) & (P["y"] <= P[f"{c}_hi"]))[m].mean())
    s["new_vs_prod"], s["new_vs_orig"], s["orig_vs_prod"] = paired(P, "new", "prod", m), paired(P, "new", "orig", m), paired(P, "orig", "prod", m)
    return s


def masks(P):
    ok = P[["y", *M]].notna().all(axis=1)
    return {"trade": ok & (P.index.dayofweek != 0), "all": ok}


mT, mV = masks(T), masks(V)
sc = {"t": {k: score(T, m) for k, m in mT.items()}, "v": {k: score(V, m) for k, m in mV.items()}}
for yr in ("t", "v"):
    for k, s in sc[yr].items():
        print(f"{'2026' if yr == 't' else '2025'} {k:5} {s['days']}d " + " ".join(f"{c} {s[c]['mae']:.3f}/{s[c]['rmse']:.2f}" for c in M)
              + f" | new-prod {s['new_vs_prod']['gain']:+.3f}+-{s['new_vs_prod']['se']:.3f} new-orig {s['new_vs_orig']['gain']:+.3f}"
              f"+-{s['new_vs_orig']['se']:.3f} orig-prod {s['orig_vs_prod']['gain']:+.3f}+-{s['orig_vs_prod']['se']:.3f}")

ok = mT["trade"]
y = T["y"]
day = pd.Series(T.index.floor("D"), T.index)
E = T[list(M)].sub(y, axis=0).abs()


def breakdown(key, order):
    out = []
    for k in order:
        m = ok & (key == k)
        if m.any():
            out.append({"k": k, "days": int(day[m].nunique()), **{c: float(E[c][m].mean()) for c in M},
                        "new_vs_prod": paired(T, "new", "prod", m)["gain"]})
    return out


mo = pd.Series(T.index.strftime("%b"), T.index)
by_month = breakdown(mo, list(dict.fromkeys(mo[ok])))
hb = pd.Series(pd.cut(T.index.hour, [-1, 5, 9, 15, 20, 23], labels=["00-05", "06-09", "10-15", "16-20", "21-23"]).astype(str), T.index)
by_hour = breakdown(hb, ["00-05", "06-09", "10-15", "16-20", "21-23"])
dmae = E[ok].groupby(day[ok]).mean()
ymax, ymean = y[ok].groupby(day[ok]).max(), y[ok].groupby(day[ok]).mean()


def days_tbl(ix):
    return [{"d": d.strftime("%Y-%m-%d"), "ymax": float(ymax[d]), "ymean": float(ymean[d]), **{c: float(dmae.loc[d, c]) for c in M}} for d in ix]


worst = {"prod": days_tbl(dmae["prod"].sort_values(ascending=False).head(10).index),
         "new": days_tbl(dmae["new"].sort_values(ascending=False).head(10).index),
         "prod_better": days_tbl((dmae["new"] - dmae["prod"]).sort_values(ascending=False).head(8).index)}
top10 = dmae["prod"].sort_values(ascending=False).head(10).index
excl = ok & ~day.isin(top10)
excl10 = {"days": int(day[excl].nunique()), **{c: float(E[c][excl].mean()) for c in M},
          "new_vs_prod": paired(T, "new", "prod", excl)}
print("excl prod-worst-10", excl10)
for r in by_month:
    print(f"  {r['k']} " + " ".join(f"{c} {r[c]:.2f}" for c in M))

r2 = lambda s: [None if pd.isna(v) else round(float(v), 2) for v in s]  # noqa: E731
H = T.reindex(pd.date_range(T.index[0], T.index[-1], freq="h"))
roll = E[mT["all"]].groupby(day[mT["all"]]).mean().rolling(7, min_periods=5).mean()
out = {"sc": sc, "by_month": by_month, "by_hour": by_hour, "worst": worst, "excl10": excl10,
       "err_corr": float(np.corrcoef((y - T["new"])[ok], (y - T["prod"])[ok])[0, 1]),
       "hourly": {"t0": str(H.index[0]), "y": r2(H["y"]), **{c: r2(H[c]) for c in M}, "new_lo": r2(H["new_lo"]), "new_hi": r2(H["new_hi"])},
       "daily": {"d": [d.strftime("%Y-%m-%d") for d in roll.index], **{c: r2(roll[c]) for c in M}}}


def rnd(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    return o


OUT.mkdir(parents=True, exist_ok=True)
js = json.dumps(rnd(out), separators=(",", ":"))
(OUT / "artifact7.json").write_text(js)
tpl = Path(__file__).parent / "artifact7_template.html"
if tpl.exists():
    page = OUT / "pjm-da-2026.html"
    page.write_text(tpl.read_text().replace("__DATA__", js))
    print(f"wrote {page} ({page.stat().st_size / 1e3:.0f} kB)")
