"""Data for stakeholder Artifact 2: Kumo PJM ALL load forecast vs Meteologica, and the energy model with it.

  uv run python research/pjm_load/artifact2_data.py

Load (2025): actual pjm_hourly_load ALL; Meteologica ALL (data/pjm_load/meteologica_all.parquet, D-1 ~10:20
EPT vintage = v0's pjm_load); Kumo load = 2025 part of the backpublished P2 run
(outputs/pjm-load-fe/P2_bp_gap24); P0 vs P2 screens (every 3rd day) from outputs/pjm-load-fe/P*_gap24_e3.
Energy (2025): outputs/pjm-fe/s3_stage/baseline_v0_daily vs outputs/pjm-fe/v0_customload_{a,b}_daily, and
production flex.daep.20261001. HOLD = 243 days (day number from 2025-01-01 not a multiple of 3);
paired day-block bootstrap of the daily MAE gain. Writes outputs/pjm-load-fe/artifact2/ JSON + HTML page.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pjm-load-fe/artifact2"
Y0, Y1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")


def preds(path, name="val_walk_forward_predictions.parquet"):
    d = pd.read_parquet(Path(path) / name)
    d = d.assign(ts=pd.to_datetime(d["ts"])).set_index("ts")
    return d[(d.index >= Y0) & (d.index < Y1)]


def masks(idx, ok):
    day = pd.Series(idx.floor("D"), idx)
    dn = (day - Y0).dt.days
    return day, {"all": ok, "hold": ok & (dn % 3 != 0)}


def paired(y, p, ref, m, day, n=2000):
    g = ((y - ref).abs() - (y - p).abs())[m].groupby(day[m]).mean()
    rng = np.random.default_rng(0)
    bs = [g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n)]
    return float(g.mean()), float(np.std(bs)), float((g > 0).mean())


def stats(y, p, m):
    e = (y - p)[m]
    return {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "mape": float((e.abs() / y[m]).mean() * 100), "bias": float(e.mean())}


def season_of(idx):
    mo = idx.month
    return pd.Series(np.select([mo.isin([12, 1, 2]), mo.isin([3, 4, 5]), mo.isin([6, 7, 8])],
                               ["Winter (Dec-Feb)", "Spring (Mar-May)", "Summer (Jun-Aug)"], "Autumn (Sep-Nov)"), idx)


def hb_of(idx):
    return pd.Series(pd.cut(idx.hour, [-1, 5, 9, 15, 20, 23],
                            labels=["00-05", "06-09", "10-15", "16-20", "21-23"]).astype(str), idx)


SEASONS = ["Winter (Dec-Feb)", "Spring (Mar-May)", "Summer (Jun-Aug)", "Autumn (Sep-Nov)"]
HBS = ["00-05", "06-09", "10-15", "16-20", "21-23"]


def breakdown(y, P, ok, cols, ref, new, day):
    out = {}
    E = P[cols].sub(y, axis=0).abs()[ok]
    for nm, key, order in (("season", season_of(P.index), SEASONS), ("hour", hb_of(P.index), HBS)):
        t = E.groupby(key[ok]).mean()
        out[nm] = [{"k": k, "n": int((key[ok] == k).sum()), **{c: float(t.loc[k, c]) for c in cols},
                    "win": paired(y, P[new], P[ref], ok & (key == k), day)[2]} for k in order]
    return out


def r(s, nd=0):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


def rnd(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    return o


# ------------------------------------------------------------------ load
kl = preds(ROOT / "outputs/pjm-load-fe/P2_bp_gap24")
yl = kl["y_true"]
met = pd.read_parquet(ROOT / "data/pjm_load/meteologica_all.parquet").set_index("ts")["meteologica"]
L = pd.DataFrame({"met": met.reindex(yl.index), "kumo": kl["q500"]})
okl = yl.notna() & L.notna().all(axis=1)
dayl, ml = masks(L.index, okl)
load = {}
for k, m in ml.items():
    load[k] = {"days": int(dayl[m].nunique()), "hours": int(m.sum()),
               "met": stats(yl, L["met"], m), "kumo": stats(yl, L["kumo"], m)}
    g, se, win = paired(yl, L["kumo"], L["met"], m, dayl)
    load[k]["kumo"] |= {"gain": g, "se": se, "win": win, "pct_mae": 100 * (1 - load[k]["kumo"]["mae"] / load[k]["met"]["mae"]),
                        "pct_rmse": 100 * (1 - load[k]["kumo"]["rmse"] / load[k]["met"]["rmse"])}
    print(f"load {k:4} met {load[k]['met']['mae']:.0f}/{load[k]['met']['rmse']:.0f} ({load[k]['met']['mape']:.2f}%) "
          f"kumo {load[k]['kumo']['mae']:.0f}/{load[k]['kumo']['rmse']:.0f} ({load[k]['kumo']['mape']:.2f}%) "
          f"gain {g:+.1f}+-{se:.1f} win {win:.2f} ({load[k]['kumo']['pct_mae']:.1f}%)")
load_brk = breakdown(yl, L, okl, ["met", "kumo"], "met", "kumo", dayl)
e_met, e_k = (yl - L["met"])[okl], (yl - L["kumo"])[okl]
mon = pd.DataFrame({"met": e_met.groupby(e_met.index.month).mean(), "kumo": e_k.groupby(e_k.index.month).mean()})
load_bias = {"month": [int(i) for i in mon.index], "met": r(mon["met"], 1), "kumo": r(mon["kumo"], 1)}
# 80% band coverage of the Kumo load forecast
cov = float(((yl >= kl["q100"]) & (yl <= kl["q900"]))[okl].mean())
# full backpublish period (2023-10 .. 2025) for the context
kbp = pd.read_parquet(ROOT / "outputs/pjm-load-fe/P2_bp_gap24/val_walk_forward_predictions.parquet")
kbp = kbp.assign(ts=pd.to_datetime(kbp["ts"])).set_index("ts").dropna(subset=["y_true"])
mb = met.reindex(kbp.index)
okb = mb.notna()
bp = {"start": str(kbp.index.min().date()), "end": str(kbp.index.max().date()), "hours": int(okb.sum()),
      "met_mae": float((kbp["y_true"] - mb)[okb].abs().mean()),
      "kumo_mae": float((kbp["y_true"] - kbp["q500"])[okb].abs().mean())}
pre = okb & (kbp.index < Y0)
bp |= {"pre_met_mae": float((kbp["y_true"] - mb)[pre].abs().mean()),
       "pre_kumo_mae": float((kbp["y_true"] - kbp["q500"])[pre].abs().mean())}
print("backpublish", bp)

# P0 vs P2 screens (every 3rd day of 2025)
scr = {}
for nm in ("P0", "P2"):
    p = preds(ROOT / f"outputs/pjm-load-fe/{nm}_gap24_e3", "val_walk_forward_every3_predictions.parquet")
    yy = p["y_true"]
    mm = met.reindex(p.index)
    ok = yy.notna() & mm.notna()
    scr[nm] = stats(yy, p["q500"], ok)
    scr["met"] = stats(yy, mm, ok)
    scr["days"] = int(p.index.floor("D")[ok].nunique())
print("screen", {k: (round(v["mae"]), round(v["bias"])) if isinstance(v, dict) else v for k, v in scr.items()})

# ------------------------------------------------------------------ energy
runs = {"base": ROOT / "outputs/pjm-fe/s3_stage/baseline_v0_daily",
        "cla": ROOT / "outputs/pjm-fe/v0_customload_a_daily",
        "clb": ROOT / "outputs/pjm-fe/v0_customload_b_daily"}
ep = {k: preds(v) for k, v in runs.items() if (v / "val_walk_forward_predictions.parquet").exists()}
ye = ep["base"]["y_true"]
P = pd.DataFrame({k: v["q500"] for k, v in ep.items()})
prod = pd.read_parquet(ROOT / "data/pjm_extra/prod_flex_daep.parquet")
prod["ts"] = pd.to_datetime(prod["dt"]) + pd.to_timedelta(prod["hr"] - 1, unit="h")
P["prod"] = prod.drop_duplicates("ts").set_index("ts")["forecast"].reindex(P.index)
oke = ye.notna() & P.notna().all(axis=1)
daye, me = masks(P.index, oke)
energy = {}
for k, m in me.items():
    energy[k] = {"days": int(daye[m].nunique())}
    for c in P:
        s = stats(ye, P[c], m)
        s.pop("mape")
        energy[k][c] = s
        if c in ("cla", "clb"):
            g, se, win = paired(ye, P[c], P["base"], m, daye)
            energy[k][c] |= {"gain": g, "se": se, "win": win}
    print(f"energy {k:4} " + " ".join(f"{c} {energy[k][c]['mae']:.3f}/{energy[k][c]['rmse']:.2f}" for c in P)
          + " | " + " ".join(f"{c} gain {energy[k][c]['gain']:+.3f}+-{energy[k][c]['se']:.3f} win {energy[k][c]['win']:.2f}"
                             for c in ("cla", "clb") if c in P))
ecols = [c for c in ("base", "cla", "clb") if c in P]
best = "clb" if "clb" in P else "cla"
energy_brk = breakdown(ye, P, oke, ecols + ["prod"], "base", best, daye)
dm = P.sub(ye, axis=0).abs()[oke].groupby(daye[oke]).mean()
ymax = ye[oke].groupby(daye[oke]).max()
top = dm["base"].sort_values(ascending=False).head(10).index
dle = (yl - L["met"]).abs()[okl].groupby(dayl[okl]).mean()
dlk = (yl - L["kumo"]).abs()[okl].groupby(dayl[okl]).mean()
spike = [{"d": d.strftime("%Y-%m-%d"), "ymax": float(ymax[d]), **{c: float(dm.loc[d, c]) for c in ecols},
          "load_met": float(dle.get(d, np.nan)), "load_kumo": float(dlk.get(d, np.nan))} for d in top]
excl = oke & ~daye.isin(top)
energy["excl_top10"] = {c: float((ye - P[c])[excl].abs().mean()) for c in ecols}
# how much did the load input actually change on the energy side
chg = {"load_abs_change_2025": float((L["kumo"] - L["met"])[okl].abs().mean()),
       "load_err_corr": float(np.corrcoef(e_met, e_k)[0, 1]),
       "energy_err_corr_a": float(np.corrcoef((ye - P["base"])[oke], (ye - P["cla"])[oke])[0, 1]) if "cla" in P else None}

out = {
    "load": load, "load_brk": load_brk, "load_bias": load_bias, "load_cov80": cov, "bp": bp, "screen": scr,
    "energy": energy, "energy_brk": energy_brk, "spike": spike, "chg": chg, "best": best,
    "load_hourly": {"t0": str(L.index[0]), "y": r(yl), "met": r(L["met"]), "kumo": r(L["kumo"])},
    "energy_hourly": {"t0": str(P.index[0]), "y": r(ye, 2), **{c: r(P[c], 2) for c in P}},
}
OUT.mkdir(parents=True, exist_ok=True)
js = json.dumps(rnd(out), separators=(",", ":"))
(OUT / "artifact2.json").write_text(js)
print(json.dumps(rnd({k: out[k] for k in ("load_brk", "load_bias", "load_cov80", "energy_brk", "spike", "chg")}), indent=None)[:6000])
tpl = Path(__file__).parent / "artifact2_template.html"
if tpl.exists():
    page = OUT / "pjm-load-kumo.html"
    page.write_text(tpl.read_text().replace("__DATA__", js))
    print(f"wrote {page} ({page.stat().st_size / 1e3:.0f} kB)")
