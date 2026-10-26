"""Data for stakeholder Artifact 5: forecasting PJM RT system energy price day-ahead with Kumo.

  uv run python research/pjm_rt/artifact5_data.py

Every model forecasts day D's 24 hourly RT prices with information available by D-1 11:00 EPT, scored on 2025 RT
(rt_energy_price, data/pjm_raw.parquet):
  da_base    Kumo original baseline trained on DA (outputs/pjm-fe/s3_stage/baseline_v0_daily), its q500 used as the
             RT forecast
  rt_base    same inputs, trained on RT (outputs/pjm-fe/v0_rt_daily; --context-gap-hours 24)
  rt_custom  + Kumo load / wind / solar replacing Meteologica's, derived inputs recomputed, + Kumo minus ISO and
             Kumo minus Meteologica for each (outputs/pjm-fe/v0_custom_all_met_rt_daily)
  rt_pf      perfect-foresight diagnostic: actual load / wind / solar + actual minus ISO / Meteologica
             (outputs/pjm-fe/v0_actuals_all_rt_daily); leaks the future, a ceiling only
  prod       production flex.daep.20261001 (a DA forecast), for reference
HOLD = 243 days (day number from 2025-01-01 not a multiple of 3); paired day-block bootstrap of daily MAE gains.
Writes outputs/pjm-rt/artifact5/ JSON + HTML page.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/pjm_renew"))
from score import load_run, paired  # noqa: E402

OUT = ROOT / "outputs/pjm-rt/artifact5"
Y0, Y1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
SEASONS = ["Winter (Dec-Feb)", "Spring (Mar-May)", "Summer (Jun-Aug)", "Autumn (Sep-Nov)"]
HBS = ["00-05", "06-09", "10-15", "16-20", "21-23"]
RUNS = {"da_base": "outputs/pjm-fe/s3_stage/baseline_v0_daily", "rt_base": "outputs/pjm-fe/v0_rt_daily",
        "rt_custom": "outputs/pjm-fe/v0_custom_all_met_rt_daily", "rt_pf": "outputs/pjm-fe/v0_actuals_all_rt_daily"}


def season_of(idx):
    mo = idx.month
    return pd.Series(np.select([mo.isin([12, 1, 2]), mo.isin([3, 4, 5]), mo.isin([6, 7, 8])], SEASONS[:3], SEASONS[3]), idx)


def hb_of(idx):
    return pd.Series(pd.cut(idx.hour, [-1, 5, 9, 15, 20, 23], labels=HBS).astype(str), idx)


def r(s, nd=2):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


def rnd(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    return o


raw = pd.read_parquet(ROOT / "data/pjm_raw.parquet")
raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts")
idx = pd.date_range(Y0, Y1 - pd.Timedelta("1h"), freq="h")
rt, da = raw["rt_energy_price"].reindex(idx), raw["da_energy_price"].reindex(idx)

pr = {k: load_run(ROOT / v) for k, v in RUNS.items() if (ROOT / v / "val_walk_forward_predictions.parquet").exists()}
cols = list(pr)
P = pd.DataFrame({k: v["q500"].reindex(idx) for k, v in pr.items()})
Q = {k: v[["q100", "q900"]].reindex(idx) for k, v in pr.items()}
prod = pd.read_parquet(ROOT / "data/pjm_extra/prod_flex_daep.parquet")
prod["ts"] = pd.to_datetime(prod["dt"]) + pd.to_timedelta(prod["hr"] - 1, unit="h")
P["prod"] = prod.drop_duplicates("ts").set_index("ts")["forecast"].reindex(idx)
allc = cols + ["prod"]
ok = rt.notna() & P.notna().all(axis=1)
dn = pd.Series((idx.floor("D") - Y0).days, idx)
masks = {"all": ok, "hold": ok & (dn % 3 != 0)}

score = {}
for nm, m in masks.items():
    F = P[m].assign(y=rt[m])
    s = {"days": int(F.index.floor("D").nunique()), "hours": int(m.sum())}
    for c in allc:
        e = F["y"] - F[c]
        s[c] = {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
                "mae_vs_da": float((da[m] - F[c]).abs().mean())}
        if c in Q:
            s[c]["cov80"] = float(((F["y"] >= Q[c]["q100"][m]) & (F["y"] <= Q[c]["q900"][m])).mean())
        for ref in ("da_base", "rt_base"):
            if c != ref and ref in F and c != "prod":
                s[c][f"vs_{ref}"] = paired(F, c, ref)
    score[nm] = s
    print(f"{nm:4} " + " ".join(f"{c} {s[c]['mae']:.3f}/{s[c]['rmse']:.2f}" for c in allc))
    for c in cols:
        for ref in ("da_base", "rt_base"):
            if f"vs_{ref}" in s[c]:
                g = s[c][f"vs_{ref}"]
                print(f"     {c} vs {ref}: {g['gain']:+.3f} +- {g['se']:.3f} win {g['win']:.2f}")

# breakdowns (full 2025)
E = P[cols].sub(rt, axis=0).abs()[ok]
brk = {}
for nm, key, order in (("season", season_of(idx)[ok], SEASONS), ("hour", hb_of(idx)[ok], HBS)):
    t = E.groupby(key).mean()
    brk[nm] = [{"k": k, **{c: float(t.loc[k, c]) for c in cols}} for k in order]
# calm vs volatile days: daily RT max above / below the 90th percentile of daily maxima
day = pd.Series(idx.floor("D"), idx)
dmax = rt[ok].groupby(day[ok]).max()
hot = dmax >= dmax.quantile(0.9)
hot_h = day.map(hot).fillna(False).astype(bool) & ok
regime = {}
for nm, m in (("calm", ok & ~hot_h), ("volatile", hot_h)):
    regime[nm] = {"days": int(day[m].nunique()), **{c: float((rt - P[c])[m].abs().mean()) for c in allc}}
    for c in cols:
        if c != "rt_base" and "rt_base" in P:
            F = P[m].assign(y=rt[m])
            regime[nm][f"{c}_vs_rt_base"] = paired(F, c, "rt_base")
print("regime", {k: {c: round(v[c], 3) for c in allc} for k, v in regime.items()})
# stressed hours: top 10% thermal stress (Meteologica-based input, same for every model)
v0 = pd.read_parquet(ROOT / "data/pjm-fe/v0/val.parquet").assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts")
stv = v0["thermal_stress_all_pjm"].reindex(idx)
top = (stv >= stv.quantile(0.9)) & ok
stress = {c: float((rt - P[c])[top].abs().mean()) for c in allc} | {"hours": int(top.sum())}
# ten worst RT days for rt_base
dm = P[allc].sub(rt, axis=0).abs()[ok].groupby(day[ok]).mean()
dmean = rt[ok].groupby(day[ok]).mean()
dda = da[ok].groupby(day[ok]).mean()
worst = dm["rt_base" if "rt_base" in dm else cols[0]].sort_values(ascending=False).head(10).index
spike = [{"d": d.strftime("%Y-%m-%d"), "rtmax": float(dmax[d]), "rtmean": float(dmean[d]), "damean": float(dda[d]),
          **{c: float(dm.loc[d, c]) for c in allc}} for d in worst]
excl = ok & ~day.isin(worst)
excl10 = {c: float((rt - P[c])[excl].abs().mean()) for c in allc}
# how far apart RT and DA are (the irreducible part for a DA-trained model)
spread = {"mae_rt_vs_da": float((rt - da)[ok].abs().mean()), "mean_rt": float(rt[ok].mean()), "mean_da": float(da[ok].mean()),
          "max_rt": float(rt[ok].max()), "max_da": float(da[ok].max()), "corr": float(np.corrcoef(rt[ok], da[ok])[0, 1])}
print("spread", spread, "stress", stress, "excl10", excl10)

out = {"score": score, "brk": brk, "regime": regime, "stress": stress, "spike": spike, "excl10": excl10,
       "spread": spread, "cols": cols,
       "hourly": {"t0": str(idx[0]), "rt": r(rt), "da": r(da), **{c: r(P[c]) for c in allc}}}
OUT.mkdir(parents=True, exist_ok=True)
js = json.dumps(rnd(out), separators=(",", ":"))
(OUT / "artifact5.json").write_text(js)
tpl = Path(__file__).parent / "artifact5_template.html"
if tpl.exists():
    page = OUT / "pjm-rt-kumo.html"
    page.write_text(tpl.read_text().replace("__DATA__", js))
    print(f"wrote {page} ({page.stat().st_size / 1e3:.0f} kB)")
