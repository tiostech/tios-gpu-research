"""Data for stakeholder Artifact 4: Kumo PJM wind / solar forecasts vs Meteologica and ISO, and the energy model
with all custom load / wind / solar inputs.

  uv run python research/pjm_renew/artifact4_data.py

Wind / solar (2025 part of the backpublished walk-forwards outputs/pjm-renew-fe/{wind,solar}/{W2,S2}_bp_gap24;
screens {W2,S2}_gap24_e3): scored with research/pjm_renew/score.py (solar daylight-only, quantiles clipped at 0).
Energy (2025): outputs/pjm-fe/s3_stage/baseline_v0_daily vs v0_actuals_all_daily (perfect-foresight bound),
v0_custom_all_daily (if run) and Artifact 2's v0_customload_b_daily; production flex.daep.20261001.
HOLD = 243 days (day number from 2025-01-01 not a multiple of 3); paired day-block bootstrap of daily MAE gains.
Writes outputs/pjm-renew-fe/artifact4/ JSON + HTML page.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import Y0, frame, load_run, paired, scorecard, subsets  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pjm-renew-fe/artifact4"
Y1 = pd.Timestamp("2026-01-01")
SEASONS = ["Winter (Dec-Feb)", "Spring (Mar-May)", "Summer (Jun-Aug)", "Autumn (Sep-Nov)"]
HBS = ["00-05", "06-09", "10-15", "16-20", "21-23"]


def season_of(idx):
    mo = idx.month
    return pd.Series(np.select([mo.isin([12, 1, 2]), mo.isin([3, 4, 5]), mo.isin([6, 7, 8])], SEASONS[:3], SEASONS[3]), idx)


def hb_of(idx):
    return pd.Series(pd.cut(idx.hour, [-1, 5, 9, 15, 20, 23], labels=HBS).astype(str), idx)


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


v0 = pd.concat(pd.read_parquet(ROOT / "data/pjm-fe/v0" / f"{s}.parquet") for s in ("train", "val"))
v0 = v0.assign(ts=pd.to_datetime(v0["ts"])).drop_duplicates("ts").set_index("ts")
stress = v0["thermal_stress_all_pjm"]
s25 = stress[(stress.index >= Y0) & (stress.index < Y1)]
top_stress = s25 >= s25.quantile(0.9)


def renew(target, run, screen):
    F = frame(run, target)
    sub = subsets(F)
    out = {k: scorecard(v) for k, v in sub.items() if len(v)}
    A = sub["all"]
    cols = ["met", "iso", "kumo"]
    E = A[cols].sub(A["y"], axis=0).abs()
    brk = {}
    for nm, key, order in (("season", season_of(A.index), SEASONS), ("hour", hb_of(A.index), HBS)):
        t = E.groupby(key).mean()
        brk[nm] = [{"k": k, "n": int((key == k).sum()), **{c: float(t.loc[k, c]) for c in cols},
                    "win": paired(A[key == k], "kumo", "met")["win"]} for k in order if k in t.index]
    # hour-to-hour ramp error: (next hour - this hour) of forecast vs actual, both hours in the same day, all hours
    Fa = frame(run, target, daylight_only=False)
    Fa = Fa[(Fa.index >= Y0) & (Fa.index < Y1)][["y"] + cols]
    nxt = Fa.reindex(Fa.index + pd.Timedelta("1h"))
    nxt.index = Fa.index
    R = (nxt - Fa)[(Fa.index.hour < 23)].dropna()
    ramp = {c: float((R["y"] - R[c]).abs().mean()) for c in cols}
    ts_m = top_stress.reindex(A.index).fillna(False).to_numpy()
    stress_err = {c: float(E.loc[ts_m, c].mean()) for c in cols} | {"hours": int(ts_m.sum())}
    e = A[cols].rsub(A["y"], axis=0)
    mon = e.groupby(A.index.month).mean()
    bias = {"month": [int(i) for i in mon.index], **{c: r(mon[c], 1) for c in cols}}
    cov = float(((A["y"] >= A["q100"]) & (A["y"] <= A["q900"])).mean()) if "q100" in A else None
    # screen (every 3rd day of 2025 = selection days)
    S = frame(screen, target)
    scr = scorecard(S)
    # hourly 2025 series, all hours
    k = load_run(run)
    k = k[(k.index >= Y0) & (k.index < Y1)]
    b = pd.read_parquet(ROOT / "data/pjm_renew/bench.parquet").assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts")
    q = k["q500"].clip(lower=0) if target == "solar" else k["q500"]
    hourly = {"t0": str(k.index[0]), "y": r(b[target].reindex(k.index)), "met": r(b[f"met_{target}"].reindex(k.index)),
              "iso": r(b[f"iso_{target}"].reindex(k.index)), "kumo": r(q)}
    bp = {"start": str(F.index.min().date()), "end": str(F.index.max().date())}
    print(f"--- {target}")
    for nm in ("all", "hold", "pre2025"):
        if nm in out:
            m, i, kk = out[nm]["met"], out[nm]["iso"], out[nm]["kumo"]
            print(f"{nm:8} MAE met {m['mae']:.1f} iso {i['mae']:.1f} kumo {kk['mae']:.1f} | gain vs met "
                  f"{kk['vs_met']['gain']:+.1f}+-{kk['vs_met']['se']:.1f} win {kk['vs_met']['win']:.2f} | vs iso "
                  f"{kk['vs_iso']['gain']:+.1f}+-{kk['vs_iso']['se']:.1f}")
    print("ramp", ramp, "stress", stress_err, "cov80", cov)
    print("screen", {c: round(scr[c]["mae"], 1) for c in cols})
    return {"score": out, "brk": brk, "ramp": ramp, "stress": stress_err, "bias": bias, "cov80": cov,
            "screen": scr, "hourly": hourly, "bp": bp}


wind = renew("wind", ROOT / "outputs/pjm-renew-fe/wind/W2_bp_gap24", ROOT / "outputs/pjm-renew-fe/wind/W2_gap24_e3")
solar = renew("solar", ROOT / "outputs/pjm-renew-fe/solar/S2_bp_gap24", ROOT / "outputs/pjm-renew-fe/solar/S2_gap24_e3")

# ISO load vs Meteologica load, 2025 (context for the custom-minus-ISO columns)
b = pd.read_parquet(ROOT / "data/pjm_renew/bench.parquet").assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts")
ld = pd.read_parquet(ROOT / "data/pjm_load/load.parquet")
ld = ld[ld["zone"] == "ALL"]
ld = ld.assign(ts=pd.to_datetime(ld["dt"]) + pd.to_timedelta(ld["hr"].astype(int) - 1, unit="h")).groupby("ts")["load_mwh"].mean()
kl = load_run(ROOT / "outputs/pjm-load-fe/P2_bp_gap24")["q500"]
idx = pd.date_range(Y0, Y1 - pd.Timedelta("1h"), freq="h")
LD = pd.DataFrame({"y": ld.reindex(idx), "met": v0["pjm_load"].reindex(idx), "iso": b["iso_load"].reindex(idx),
                   "kumo": kl.reindex(idx)}).dropna()
load = {c: float((LD["y"] - LD[c]).abs().mean()) for c in ("met", "iso", "kumo")} | \
       {f"bias_{c}": float((LD["y"] - LD[c]).mean()) for c in ("met", "iso", "kumo")}
print("load 2025", load)

# ------------------------------------------------------------------ energy
runs = {"base": ROOT / "outputs/pjm-fe/s3_stage/baseline_v0_daily",
        "pf": ROOT / "outputs/pjm-fe/v0_actuals_all_daily",
        "custom": ROOT / "outputs/pjm-fe/v0_custom_all_daily",
        "clb": ROOT / "outputs/pjm-fe/v0_customload_b_daily"}
ep = {}
for k, v in runs.items():
    if (v / "val_walk_forward_predictions.parquet").exists():
        d = load_run(v)
        ep[k] = d[(d.index >= Y0) & (d.index < Y1)]
ye = ep["base"]["y_true"]
P = pd.DataFrame({k: v["q500"] for k, v in ep.items()})
prod = pd.read_parquet(ROOT / "data/pjm_extra/prod_flex_daep.parquet")
prod["ts"] = pd.to_datetime(prod["dt"]) + pd.to_timedelta(prod["hr"] - 1, unit="h")
P["prod"] = prod.drop_duplicates("ts").set_index("ts")["forecast"].reindex(P.index)
oke = ye.notna() & P.notna().all(axis=1)
dn = (P.index.floor("D") - Y0).days
energy = {}
for nm, m in (("all", oke), ("hold", oke & (dn % 3 != 0))):
    Fm = P[m].assign(y=ye[m])
    energy[nm] = {"days": int(Fm.index.floor("D").nunique())}
    for c in P:
        e = Fm["y"] - Fm[c]
        energy[nm][c] = {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean()))}
        if c not in ("base", "prod"):
            energy[nm][c] |= paired(Fm, c, "base")
    print(f"energy {nm:4} " + " ".join(f"{c} {energy[nm][c]['mae']:.3f}/{energy[nm][c]['rmse']:.2f}" for c in P) + " | " +
          " ".join(f"{c} gain {energy[nm][c]['gain']:+.3f}+-{energy[nm][c]['se']:.3f} win {energy[nm][c]['win']:.2f}"
                   for c in P if c not in ("base", "prod")))
day = pd.Series(P.index.floor("D"), P.index)
dm = P.sub(ye, axis=0).abs()[oke].groupby(day[oke]).mean()
ymax = ye[oke].groupby(day[oke]).max()
top = dm["base"].sort_values(ascending=False).head(10).index
ecols = [c for c in ("base", "pf", "custom", "clb") if c in P]
spike = [{"d": d.strftime("%Y-%m-%d"), "ymax": float(ymax[d]), **{c: float(dm.loc[d, c]) for c in ecols}} for d in top]
energy_brk = {}
E = P[ecols].sub(ye, axis=0).abs()[oke]
for nm, key, order in (("season", season_of(P.index)[oke], SEASONS), ("hour", hb_of(P.index)[oke], HBS)):
    t = E.groupby(key).mean()
    energy_brk[nm] = [{"k": k, **{c: float(t.loc[k, c]) for c in ecols}} for k in order]

out = {"wind": wind, "solar": solar, "load": load, "energy": energy, "energy_brk": energy_brk, "spike": spike,
       "ecols": ecols, "energy_hourly": {"t0": str(P.index[0]), "y": r(ye, 2), **{c: r(P[c], 2) for c in P}}}
OUT.mkdir(parents=True, exist_ok=True)
js = json.dumps(rnd(out), separators=(",", ":"))
(OUT / "artifact4.json").write_text(js)
print(json.dumps(rnd({k: out[k] for k in ("energy_brk", "spike")}))[:3000])
tpl = Path(__file__).parent / "artifact4_template.html"
if tpl.exists():
    page = OUT / "pjm-wind-solar-kumo.html"
    page.write_text(tpl.read_text().replace("__DATA__", js))
    print(f"wrote {page} ({page.stat().st_size / 1e3:.0f} kB)")
