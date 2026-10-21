"""Data for stakeholder Artifact 3: production PJM DAEP forecast (flex.daep.20261001) vs the original Kumo
baseline on the 2026 test set.

  uv run python research/pjm/artifact3_data.py

Kumo: outputs/pjm-fe/baseline_v0_test_daily = the original baseline (data/pjm, est8, seed 0, 10k context,
daily walk-forward) run on the test split, with train = train + val (through 2025-12-31) and val = test.
Production: data/pjm_extra/prod_flex_daep.parquet (ts = dt + hr - 1 h). Scored on complete days
2026-01-01 .. last day with 24 actual hours. No day of 2026 was used for any research choice, so all of it is
held out. Paired day-block bootstrap of the daily MAE gain. 2025 numbers (s3_stage/baseline_v0_daily) are
recomputed alongside for comparison. Writes outputs/pjm-fe/artifact3/ JSON + HTML page.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pjm-fe/artifact3"

prod = pd.read_parquet(ROOT / "data/pjm_extra/prod_flex_daep.parquet")
prod["ts"] = pd.to_datetime(prod["dt"]) + pd.to_timedelta(prod["hr"] - 1, unit="h")
prod = prod.drop_duplicates("ts").set_index("ts")["forecast"]


def frame(run):
    d = pd.read_parquet(ROOT / "outputs/pjm-fe" / run / "val_walk_forward_predictions.parquet")
    d = d.assign(ts=pd.to_datetime(d["ts"])).set_index("ts")
    P = pd.DataFrame({"y": d["y_true"], "prod": prod.reindex(d.index), "kumo": d["q500"],
                      "q100": d["q100"], "q900": d["q900"]})
    # complete days only (the test split ends at 01:00 on its last day)
    n = P.groupby(P.index.floor("D"))["y"].transform("size")
    return P[n >= 23]


T = frame("baseline_v0_test_daily")
V = frame("s3_stage/baseline_v0_daily")


def paired(P, m, n=2000):
    day = P.index.floor("D")
    g = ((P["y"] - P["prod"]).abs() - (P["y"] - P["kumo"]).abs())[m].groupby(day[m]).mean()
    rng = np.random.default_rng(0)
    bs = [g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n)]
    return float(g.mean()), float(np.std(bs)), float((g > 0).mean())


def score(P):
    ok = P["y"].notna() & P["prod"].notna() & P["kumo"].notna()
    s = {"days": int(P.index.floor("D")[ok].nunique()), "hours": int(ok.sum()),
         "start": str(P.index[ok].min().date()), "end": str(P.index[ok].max().date())}
    for c in ("prod", "kumo"):
        e = (P["y"] - P[c])[ok]
        s[c] = {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}
    g, se, win = paired(P, ok)
    s["kumo"] |= {"gain": g, "se": se, "win": win, "pct_mae": 100 * (1 - s["kumo"]["mae"] / s["prod"]["mae"]),
                  "pct_rmse": 100 * (1 - s["kumo"]["rmse"] / s["prod"]["rmse"])}
    s["cov80"] = float(((P["y"] >= P["q100"]) & (P["y"] <= P["q900"]))[ok].mean())
    s["ymean"], s["ymax"] = float(P["y"][ok].mean()), float(P["y"][ok].max())
    return s, ok


sc_t, ok = score(T)
sc_v, _ = score(V)
for nm, s in (("2026", sc_t), ("2025", sc_v)):
    print(f"{nm} {s['days']}d prod {s['prod']['mae']:.3f}/{s['prod']['rmse']:.2f} kumo {s['kumo']['mae']:.3f}/"
          f"{s['kumo']['rmse']:.2f} gain {s['kumo']['gain']:+.3f}+-{s['kumo']['se']:.3f} win {s['kumo']['win']:.2f} "
          f"({s['kumo']['pct_mae']:.1f}% / {s['kumo']['pct_rmse']:.1f}%) cov80 {s['cov80']:.2f} ymean {s['ymean']:.1f}")

y = T["y"]
day = pd.Series(T.index.floor("D"), T.index)
E = T[["prod", "kumo"]].sub(y, axis=0).abs()[ok]


def breakdown(key, order):
    t = E.groupby(key[ok]).mean()
    out = []
    for k in order:
        m = ok & (key == k)
        if not m.any():
            continue
        out.append({"k": k, "n": int(m.sum()), "days": int(day[m].nunique()), "prod": float(t.loc[k, "prod"]),
                    "kumo": float(t.loc[k, "kumo"]), "win": paired(T, m)[2]})
    return out


mo = pd.Series(T.index.strftime("%b"), T.index)
months = list(dict.fromkeys(mo[ok]))
by_month = breakdown(mo, months)
hb = pd.Series(pd.cut(T.index.hour, [-1, 5, 9, 15, 20, 23],
                      labels=["00-05", "06-09", "10-15", "16-20", "21-23"]).astype(str), T.index)
by_hour = breakdown(hb, ["00-05", "06-09", "10-15", "16-20", "21-23"])
mm = T.index.month
season = pd.Series(np.select([mm.isin([12, 1, 2]), mm.isin([3, 4, 5]), mm.isin([6, 7, 8])],
                             ["Winter (Jan-Feb)", "Spring (Mar-May)", "Summer (Jun-Aug)"], "Autumn (Sep-Oct)"), T.index)
by_season = breakdown(season, ["Winter (Jan-Feb)", "Spring (Mar-May)", "Summer (Jun-Aug)", "Autumn (Sep-Oct)"])
for r in by_month + by_season + by_hour:
    print(f"  {r['k']:18} prod {r['prod']:6.2f} kumo {r['kumo']:6.2f} win {r['win']:.2f}")

dmae = E.groupby(day[ok]).mean()
ymax = y[ok].groupby(day[ok]).max()
ymean = y[ok].groupby(day[ok]).mean()


def days_tbl(idx):
    return [{"d": d.strftime("%Y-%m-%d"), "ymax": float(ymax[d]), "ymean": float(ymean[d]),
             "prod": float(dmae.loc[d, "prod"]), "kumo": float(dmae.loc[d, "kumo"])} for d in idx]


worst_prod = dmae["prod"].sort_values(ascending=False).head(10).index
worst_kumo = dmae["kumo"].sort_values(ascending=False).head(10).index
conc = {c: float(E[c][day[ok].isin(dmae[c].sort_values(ascending=False).index[:10])].sum() / E[c].sum())
        for c in ("prod", "kumo")}
# where production was better: largest kumo-minus-prod daily gaps
prod_better = (dmae["kumo"] - dmae["prod"]).sort_values(ascending=False).head(5).index
print("worst prod\n", pd.DataFrame(days_tbl(worst_prod)).round(1).to_string(index=False))
print("worst kumo\n", pd.DataFrame(days_tbl(worst_kumo)).round(1).to_string(index=False))
print("prod better\n", pd.DataFrame(days_tbl(prod_better)).round(1).to_string(index=False))
print("conc", conc)
excl = ok & ~day.isin(worst_prod)
excl10 = {c: float((y - T[c])[excl].abs().mean()) for c in ("prod", "kumo")}
print("excl prod top10", excl10)

# the late-January 2026 cold snap dominates the year: stats for that week and for the rest of the test set
EV0, EV1 = pd.Timestamp("2026-01-23"), pd.Timestamp("2026-01-30")
inev = (day >= EV0) & (day <= EV1)
g_rest = paired(T, ok & ~inev)
hist = pd.concat([pd.read_parquet(ROOT / "data/pjm-fe/v0" / f"{s}.parquet", columns=["ts", "y"]) for s in ("train", "val", "test")])
hist = hist.assign(ts=pd.to_datetime(hist["ts"])).set_index("ts")["y"].dropna()
ctx = hist[hist.index < EV0].iloc[-10000:]
evd = T[inev & ok].groupby(day[inev & ok])
event = {"start": str(EV0.date()), "end": str(EV1.date()), "ymax": float(y[inev].max()),
         "ctx_max": float(ctx.max()), "ctx_max_at": str(ctx.idxmax().date()),
         "fl_min": float(pd.read_parquet(ROOT / "data/pjm-fe/v0/test.parquet", columns=["ts", "feelslike_weightedavg"])
                         .assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts")
                         .loc[str(EV0.date()):str(EV1.date()), "feelslike_weightedavg"].min()),
         "days": [{"d": d.strftime("%Y-%m-%d"), "y": float(g["y"].mean()), "kumo": float(g["kumo"].mean()),
                   "prod": float(g["prod"].mean()), "ymax": float(g["y"].max()), "q900max": float(g["q900"].max()),
                   "mae_prod": float(dmae.loc[d, "prod"]), "mae_kumo": float(dmae.loc[d, "kumo"])} for d, g in evd],
         "rest": {"days": int(day[ok & ~inev].nunique()), "gain": g_rest[0], "se": g_rest[1], "win": g_rest[2],
                  "prod": float((y - T["prod"])[ok & ~inev].abs().mean()),
                  "kumo": float((y - T["kumo"])[ok & ~inev].abs().mean())}}
gd = ((y - T["prod"]).abs() - (y - T["kumo"]).abs())[ok].groupby(day[ok]).mean()
from scipy.stats import binomtest  # noqa: E402
sign_p = float(binomtest(int((gd > 0).sum()), len(gd)).pvalue)
print("event", json.dumps({k: v for k, v in event.items() if k != "days"}), "sign test p", sign_p)
print(pd.DataFrame(event["days"]).round(0).to_string(index=False))

roll = dmae.rolling(7, min_periods=7).mean()
r2 = lambda s: [None if pd.isna(v) else round(float(v), 2) for v in s]  # noqa: E731
Tk = T[T.index.floor("D") <= pd.Timestamp(sc_t["end"])]
out = {
    "t": sc_t, "v": sc_v, "by_month": by_month, "by_season": by_season, "by_hour": by_hour,
    "worst_prod": days_tbl(worst_prod), "worst_kumo": days_tbl(worst_kumo), "prod_better": days_tbl(prod_better),
    "conc": conc, "excl10": excl10, "event": event, "sign_p": sign_p,
    "err_corr": float(np.corrcoef((y - T["kumo"])[ok], (y - T["prod"])[ok])[0, 1]),
    "hourly": {"t0": str(Tk.index[0]), "y": r2(Tk["y"]), "prod": r2(Tk["prod"]), "kumo": r2(Tk["kumo"])},
    "daily": {"d": [d.strftime("%Y-%m-%d") for d in roll.index], "roll_prod": r2(roll["prod"]),
              "roll_kumo": r2(roll["kumo"])},
}
assert len(Tk) == len(out["hourly"]["y"]) and (Tk.index == pd.date_range(Tk.index[0], periods=len(Tk), freq="h")).all()


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
(OUT / "artifact3.json").write_text(js)
tpl = Path(__file__).parent / "artifact3_template.html"
if tpl.exists():
    page = OUT / "pjm-test-2026.html"
    page.write_text(tpl.read_text().replace("__DATA__", js))
    print(f"wrote {page} ({page.stat().st_size / 1e3:.0f} kB)")
