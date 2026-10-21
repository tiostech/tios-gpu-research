"""Data for stakeholder Artifact 1: production PJM DAEP forecast (flex.daep.20261001) vs Kumo-Tabular.

  uv run python research/pjm/artifact1_data.py [out.json]

Reads full-daily 2025 predictions from outputs/pjm-fe/s3_stage/<run>/ (baseline_v0_daily, ensC_daily)
and the production forecast from data/pjm_extra/prod_flex_daep.parquet (ts = dt + hr - 1 h).
Writes one compact JSON (scorecard, season / hour-block breakdown, worst days, hourly series, example
day) for the artifact page, and prints the headline numbers. HOLD = 2025 days whose day number from
2025-01-01 is not a multiple of 3 (243 days); paired day-block bootstrap of the daily MAE gain.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/pjm-fe/artifact1/artifact1.json"


def load(run):
    d = pd.read_parquet(ROOT / "outputs/pjm-fe/s3_stage" / run / "val_walk_forward_predictions.parquet")
    return d.assign(ts=pd.to_datetime(d["ts"])).set_index("ts")


base, ensc = load("baseline_v0_daily"), load("ensC_daily")
y = base["y_true"]
assert (y.dropna() == ensc["y_true"].reindex(y.index).dropna()).all()
prod = pd.read_parquet(ROOT / "data/pjm_extra/prod_flex_daep.parquet")
prod["ts"] = pd.to_datetime(prod["dt"]) + pd.to_timedelta(prod["hr"] - 1, unit="h")
prod = prod.drop_duplicates("ts").set_index("ts")
P = pd.DataFrame({"prod": prod["forecast"].reindex(y.index), "kumo": base["q500"], "ensc": ensc["q500"]})

ok = y.notna() & P.notna().all(axis=1)
day = pd.Series(P.index.floor("D"), P.index)
dn = (day - pd.Timestamp("2025-01-01")).dt.days
hold = ok & (dn % 3 != 0)
print(f"rows ok {ok.sum()} / {len(P)}, HOLD days {day[hold].nunique()}")


def mae(c, m):
    return float((y - P[c])[m].abs().mean())


def rmse(c, m):
    return float(np.sqrt(((y - P[c])[m] ** 2).mean()))


def paired(c, ref, m, n=2000):
    g = ((y - P[ref]).abs() - (y - P[c]).abs())[m].groupby(day[m]).mean()
    rng = np.random.default_rng(0)
    bs = [g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n)]
    return float(g.mean()), float(np.std(bs)), float((g > 0).mean()), int(len(g))


score = {}
for sname, m in (("all", ok), ("hold", hold)):
    score[sname] = {"days": int(day[m].nunique()), "hours": int(m.sum())}
    for c in P:
        score[sname][c] = {"mae": mae(c, m), "rmse": rmse(c, m), "bias": float((y - P[c])[m].mean())}
        if c != "prod":
            g, se, win, _ = paired(c, "prod", m)
            score[sname][c] |= {"gain": g, "se": se, "win": win,
                                "pct_mae": 100 * (1 - mae(c, m) / mae("prod", m)),
                                "pct_rmse": 100 * (1 - rmse(c, m) / rmse("prod", m))}
    print(f"{sname:4} prod {score[sname]['prod']['mae']:.3f}/{score[sname]['prod']['rmse']:.2f}  "
          f"kumo {score[sname]['kumo']['mae']:.3f}/{score[sname]['kumo']['rmse']:.2f} "
          f"gain {score[sname]['kumo']['gain']:+.3f}+-{score[sname]['kumo']['se']:.3f} "
          f"win {score[sname]['kumo']['win']:.2f} ({score[sname]['kumo']['pct_mae']:.1f}%)  "
          f"ensC {score[sname]['ensc']['mae']:.3f}/{score[sname]['ensc']['rmse']:.2f}")

E = P.sub(y, axis=0).abs()[ok]
mo = P.index.month
season = pd.Series(np.select([mo.isin([12, 1, 2]), mo.isin([3, 4, 5]), mo.isin([6, 7, 8])],
                             ["Winter (Dec-Feb)", "Spring (Mar-May)", "Summer (Jun-Aug)"],
                             "Autumn (Sep-Nov)"), P.index)
hb = pd.Series(pd.cut(P.index.hour, [-1, 5, 9, 15, 20, 23],
                      labels=["00-05", "06-09", "10-15", "16-20", "21-23"]).astype(str), P.index)


def breakdown(key, order):
    t = E.groupby(key[ok]).mean().reindex(order)
    n = E.groupby(key[ok]).size().reindex(order)
    g = {k: paired("kumo", "prod", ok & (key == k))[:3] for k in order}
    return [{"k": k, "n": int(n[k]), "prod": t.loc[k, "prod"], "kumo": t.loc[k, "kumo"],
             "ensc": t.loc[k, "ensc"], "win": g[k][2]} for k in order]


by_season = breakdown(season, ["Winter (Dec-Feb)", "Spring (Mar-May)", "Summer (Jun-Aug)", "Autumn (Sep-Nov)"])
by_hour = breakdown(hb, ["00-05", "06-09", "10-15", "16-20", "21-23"])

dmae = E.groupby(day[ok]).mean()
ymax = y[ok].groupby(day[ok]).max()
ymean = y[ok].groupby(day[ok]).mean()
worst_prod = dmae["prod"].sort_values(ascending=False).head(10).index
worst_kumo = dmae["kumo"].sort_values(ascending=False).head(10).index


def days_tbl(idx):
    return [{"d": d.strftime("%Y-%m-%d"), "ymax": ymax[d], "ymean": ymean[d], "prod": dmae.loc[d, "prod"],
             "kumo": dmae.loc[d, "kumo"], "ensc": dmae.loc[d, "ensc"], "hold": bool(dn[d] % 3 != 0)}
            for d in idx]


# concentration of error in the worst days (by each model's own ranking)
conc = {}
for c in ("prod", "kumo"):
    err = E[c]
    srt = dmae[c].sort_values(ascending=False).index
    conc[c] = {f"top{k}": float(err[day[ok].isin(srt[:k])].sum() / err.sum()) for k in (10, 30)}

# blend check (weight on production picked on the non-HOLD days, scored on HOLD)
screen = ok & ~hold
ws = np.round(np.arange(0, 1.01, 0.05), 2)
blend_mae = lambda w, m: float((y - (w * P["prod"] + (1 - w) * P["kumo"]))[m].abs().mean())  # noqa: E731
w_pick = float(min(ws, key=lambda w: blend_mae(w, screen)))
blend = {"err_corr": float(np.corrcoef((y - P["kumo"])[ok], (y - P["prod"])[ok])[0, 1]),
         "err_corr_ensc": float(np.corrcoef((y - P["ensc"])[ok], (y - P["prod"])[ok])[0, 1]),
         "w_pick": w_pick, "hold_mae_pick": blend_mae(w_pick, hold), "hold_mae_5050": blend_mae(0.5, hold),
         "hold_mae_kumo": mae("kumo", hold)}

# rolling 7-day MAE per forecast (daily points)
roll = dmae.rolling(7, min_periods=7).mean()

# example day for the explainer: the 2025 day with the biggest production-minus-Kumo daily MAE gap,
# among days with a real price event (daily max >= 150)
gap = (dmae["prod"] - dmae["kumo"])[ymax >= 150]
ex = gap.idxmax()
exm = day == ex
example = {"d": ex.strftime("%Y-%m-%d"), "ymax": ymax[ex], "prod_mae": dmae.loc[ex, "prod"],
           "kumo_mae": dmae.loc[ex, "kumo"], "hold": bool(dn[ex] % 3 != 0),
           "hours": list(range(24)),
           "y": y[exm].round(2).tolist(), "prod": P["prod"][exm].round(2).tolist(),
           "kumo": P["kumo"][exm].round(2).tolist(),
           "q100": base["q100"][exm].round(2).tolist(), "q900": base["q900"][exm].round(2).tolist()}
# feature context for that day vs the 2025 median: what Kumo "saw"
v0 = pd.read_parquet(ROOT / "data/pjm-fe/v0/val.parquet")
v0 = v0.assign(ts=pd.to_datetime(v0["ts"])).set_index("ts")
for col, nm in (("feelslike_weightedavg", "feelslike"), ("pjm_load", "load"),
                ("thermal_stress_all_pjm", "stress"), ("blended_gasprice", "gas")):
    example[nm] = {"day_max": float(v0.loc[exm[exm].index, col].max()),
                   "day_min": float(v0.loc[exm[exm].index, col].min()),
                   "year_median": float(v0[col].median())}
print(f"example day {example['d']}: ymax {example['ymax']:.0f}, prod {example['prod_mae']:.1f} "
      f"vs kumo {example['kumo_mae']:.1f}")

# extrapolation check: Kumo forecasts vs the price range of its own context (last 10k hours before each day)
hist = pd.concat([pd.read_parquet(ROOT / "data/pjm-fe/v0" / f"{s}.parquet") for s in ("train", "val")])
hist = hist.assign(ts=pd.to_datetime(hist["ts"])).set_index("ts")["y"]
xr = []
for d, g in base.groupby(base.index.floor("D")):
    ctx = hist[hist.index < d].dropna().iloc[-10000:]
    xr.append({"d": d, "cmax": ctx.max(), "q500": g["q500"].max(), "q950": g["q950"].max(), "ymax": g["y_true"].max()})
xr = pd.DataFrame(xr).set_index("d")
rec = xr[xr["ymax"] > xr["cmax"]]
extrap = {"days_q500_above": int((xr["q500"] > xr["cmax"]).sum()), "days_q950_above": int((xr["q950"] > xr["cmax"]).sum()),
          "max_q950_ratio": float((xr["q950"] / xr["cmax"]).max()),
          "best": {"d": (bd := (xr["q500"] / xr["cmax"]).idxmax()).strftime("%Y-%m-%d"),
                   "q500": float(xr.loc[bd, "q500"]), "cmax": float(xr.loc[bd, "cmax"])},
          "records": [{"d": d.strftime("%Y-%m-%d"), "cmax": float(v.cmax), "ymax": float(v.ymax), "q500": float(v.q500),
                       "q950": float(v.q950)} for d, v in rec.iterrows()]}
print("extrapolation:", extrap)

r2 = lambda s: [None if pd.isna(v) else round(float(v), 2) for v in s]  # noqa: E731
out = {
    "score": score, "by_season": by_season, "by_hour": by_hour,
    "worst_prod": days_tbl(worst_prod), "worst_kumo": days_tbl(worst_kumo), "conc": conc,
    "blend": blend, "example": example, "extrap": extrap,
    "hourly": {"t0": str(P.index[0]), "y": r2(y), "prod": r2(P["prod"]), "kumo": r2(P["kumo"]),
               "ensc": r2(P["ensc"])},
    "daily": {"d": [d.strftime("%Y-%m-%d") for d in roll.index], "roll_prod": r2(roll["prod"]),
              "roll_kumo": r2(roll["kumo"]), "roll_ensc": r2(roll["ensc"])},
}
OUT.parent.mkdir(parents=True, exist_ok=True)


def rnd(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    return o


OUT.write_text(json.dumps(rnd(out), separators=(",", ":")))
print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")
print(json.dumps(rnd({"by_season": by_season, "by_hour": by_hour, "conc": conc, "blend": blend}), indent=1))
for t, idx in (("worst prod", worst_prod), ("worst kumo", worst_kumo)):
    print(t, pd.DataFrame(days_tbl(idx)).round(1).to_string(index=False))

# page: inline the JSON into the HTML template
tpl = (Path(__file__).parent / "artifact1_template.html").read_text()
page = OUT.with_name("pjm-production-vs-kumo.html")
page.write_text(tpl.replace("__DATA__", json.dumps(rnd(out), separators=(",", ":"))))
print(f"wrote {page} ({page.stat().st_size / 1e3:.0f} kB)")
