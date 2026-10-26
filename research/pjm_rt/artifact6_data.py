"""Data for stakeholder Artifact 6: what perfect foresight tells us about forecasting PJM RT spikes (2025 only).

  uv run python research/pjm_rt/spikes_ramp.py      # writes outputs/pjm-rt/spikes/ramp_auc.csv (feature tests)
  uv run python research/pjm_rt/artifact6_data.py

Pulls together: perfect-foresight ablations (which actual input carries the gain), median vs mean as the point
forecast, how the models behave on spike hours, which realised quantities separate spike hours at equal net load
(ramp_auc.csv), and the capacity-headroom runs. All RT runs: Kumo-Tabular large, 8 estimators, seed 0, daily
walk-forward, --context-gap-hours 24. HOLD = 243 days; paired day-block bootstrap. Writes outputs/pjm-rt/artifact6/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/pjm_renew"))
sys.path.insert(0, str(ROOT / "research/spp_load"))
from score import paired  # noqa: E402
from make_features import ts_of  # noqa: E402

OUT = ROOT / "outputs/pjm-rt/artifact6"
Y0, Y1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
SPIKE = 150.0
idx = pd.date_range(Y0, Y1 - pd.Timedelta("1h"), freq="h")
RUNS = {"da_base": "s3_stage/baseline_v0_daily", "rt_base": "v0_rt_daily", "rt_custom": "v0_custom_all_met_rt_daily",
        "pf_all": "v0_actuals_all_rt_daily", "pf_errs": "v0_pf_errs_rt_daily", "pf_load": "v0_pf_load_rt_daily",
        "pf_solar": "v0_pf_solar_rt_daily", "pf_wind": "v0_pf_wind_rt_daily", "head": "v0_head_rt_daily",
        "headpf": "v0_headpf_rt_daily"}


def auc(score, lab):
    s, l = np.asarray(score, float), np.asarray(lab, bool)
    ok = ~np.isnan(s)
    s, l = s[ok], l[ok]
    rk = pd.Series(s).rank().to_numpy()
    n1, n0 = l.sum(), (~l).sum()
    return float((rk[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rnd(o):
    if isinstance(o, float):
        return None if np.isnan(o) else round(o, 4)
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    return o


raw = pd.read_parquet(ROOT / "data/pjm_raw.parquet")
raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts").reindex(idx)
rt, da = raw["rt_energy_price"], raw["da_energy_price"]
P = {k: pd.read_parquet(ROOT / f"outputs/pjm-fe/{v}/val_walk_forward_predictions.parquet")
     .assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts").reindex(idx) for k, v in RUNS.items()}
ok = rt.notna() & pd.concat([p["q500"] for p in P.values()], axis=1).notna().all(axis=1)
dn = pd.Series((idx.normalize() - Y0).days, idx)
day = pd.Series(idx.normalize(), idx)
hold = ok & (dn % 3 != 0)
spk = rt >= SPIKE
dmax = rt[ok].groupby(day[ok]).max()
vol = day.map(dmax >= dmax.quantile(0.9)).fillna(False).astype(bool)

# ---- ablations (median point forecast)
abl = {}
for nm, m in (("hold", hold), ("all", ok)):
    F = pd.DataFrame({"y": rt[m], **{k: p["q500"][m] for k, p in P.items()}})
    abl[nm] = {}
    for k in P:
        e = F["y"] - F[k]
        r = {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
             "spike_mae": float((rt - P[k]["q500"])[m & spk].abs().mean()),
             "calm_mae": float((rt - P[k]["q500"])[m & ~vol].abs().mean())}
        if k != "rt_base":
            r["vs_rt_base"] = paired(F, k, "rt_base")
        abl[nm][k] = r
    print(nm, {k: round(v["vs_rt_base"]["gain"], 3) for k, v in abl[nm].items() if "vs_rt_base" in v})

# ---- median vs mean
mm = {}
for k in ("da_base", "rt_base", "rt_custom", "pf_all"):
    F = pd.DataFrame({"y": rt[hold], "med": P[k]["q500"][hold], "mean": P[k]["mean"][hold]})
    r = {}
    for pt in ("med", "mean"):
        e = F["y"] - F[pt]
        r[pt] = {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
                 "spike_mae": float((rt - F[pt])[hold & spk].abs().mean()), "calm_mae": float((rt - F[pt])[hold & ~vol].abs().mean())}
    r["mean_vs_med"] = paired(F, "mean", "med")
    mm[k] = r
F = pd.DataFrame({"y": rt[hold], "cu": P["rt_custom"]["mean"][hold], "rb": P["rt_base"]["mean"][hold]})
mm["custom_vs_base_on_mean"] = paired(F, "cu", "rb")

# ---- spike-hour behaviour
sp = {"hours": int((ok & spk).sum()), "days": int(day[ok & spk].nunique()), "rt_mean": float(rt[ok & spk].mean()),
      "rt_max": float(rt[ok].max()), "share_hours": float((ok & spk).mean())}
for k in ("rt_base", "pf_all"):
    p = P[k]
    sp[k] = {"q500": float(p["q500"][ok & spk].mean()), "q950": float(p["q950"][ok & spk].mean()), "mean": float(p["mean"][ok & spk].mean()),
             "auc_q500": auc(p["q500"][ok], spk[ok]), "auc_q950": auc(p["q950"][ok], spk[ok]),
             "q950_hit": float((p["q950"][ok & spk] >= SPIKE).mean()), "q950_false": float((p["q950"][ok & ~spk] >= SPIKE).mean())}
# analogues: context hours (10k ending D-2) with actual net load within 2 GW below this hour's
v = pd.concat([pd.read_parquet(ROOT / f"data/pjm-fe/v0_actuals_all_rt/{s}.parquet") for s in ("train", "val")])
v = v.assign(ts=pd.to_datetime(v["ts"])).drop_duplicates("ts").set_index("ts").sort_index()
net = (v["pjm_load"] - v["pjm_wind"] - v["pjm_solar"])
pos = np.searchsorted(net.index, idx.normalize() - pd.Timedelta("25h"))
netv, actn = net.to_numpy(), net.reindex(idx).to_numpy()
nan_ = np.full(len(idx), np.nan)
for i in np.flatnonzero((ok & spk).to_numpy()):
    p0 = pos[i]
    nan_[i] = (netv[max(0, p0 - 10000):p0] >= actn[i] - 2000).sum()
an = pd.Series(nan_, idx)
sp["analog"] = []
for lo, hi, lab in ((0, 20, "under 20"), (20, 100, "20 to 100"), (100, 1e9, "100 or more")):
    m = ok & spk & (an >= lo) & (an < hi)
    sp["analog"].append({"k": lab, "hours": int(m.sum()), "rt": float(rt[m].mean()), "q500": float(P["pf_all"]["q500"][m].mean()),
                         "q950": float(P["pf_all"]["q950"][m].mean())})
print("spike", {k: v for k, v in sp.items() if k != "analog"}, sp["analog"])

# ---- feature tests (from spikes_ramp.py)
T = pd.read_csv(ROOT / "outputs/pjm-rt/spikes/ramp_auc.csv")
FEAT = {
    "headroom_econ": ("Available-capacity headroom (econ max - load)", "margin", "realised"),
    "load_ramp_err_iso": ("Load ramp miss vs ISO, 1 h", "load", "realised error"),
    "net_ramp_err_iso": ("Net-load ramp miss vs ISO, 1 h", "net", "realised error"),
    "net_ramp_err_met": ("Net-load ramp miss vs Meteologica, 1 h", "net", "realised error"),
    "net_swing6_iso": ("Net-load miss rise over 6 h vs ISO", "net", "realised error"),
    "solar_ramp_err_met": ("Solar ramp miss vs Meteologica, 1 h", "ren", "realised error"),
    "ren_ramp_err_met": ("Wind + solar ramp miss vs Meteologica, 1 h", "ren", "realised error"),
    "ren_ramp3_err_iso": ("Wind + solar ramp miss vs ISO, 3 h", "ren", "realised error"),
    "wind_ramp3_err_met": ("Wind ramp miss vs Meteologica, 3 h", "ren", "realised error"),
    "ren_swing6_met": ("Renewables above-then-below forecast, 6 h", "ren", "realised error"),
    "ren_swing_day_iso": ("Renewables above-then-below forecast, day", "ren", "realised error"),
    "iir_offline_mw": ("IIR offline capacity", "margin", "realised"),
    "gen_on_outage_fc": ("PJM outage forecast (in the model)", "margin", "known at cutoff"),
    "gas_gen_ramp3": ("Gas generation ramp, 3 h", "margin", "realised"),
    "SYMPTOM_oil_gen": ("Oil generation (symptom)", "sym", "symptom"),
}
ft = []
for c, (lab, grp, kind) in FEAT.items():
    r = {"f": c, "lab": lab, "grp": grp, "kind": kind}
    for (ln, sn), t in T.groupby(["label", "strata"]):
        row = t[t["feature"] == c]
        if len(row):
            row = row.iloc[0]
            # report as "higher = more spikes": flip AUC < 0.5 features so the bar reads as strength
            flip = row["auc"] < 0.5
            r[f"{ln}_{sn}"] = {"auc": float(1 - row["auc"] if flip else row["auc"]), "lo": float(1 - row["hi"] if flip else row["lo"]),
                               "hi": float(1 - row["lo"] if flip else row["hi"]), "flip": bool(flip), "stable": bool(row["stable"]),
                               "h1": float(row["H1"]), "h2": float(row["H2"])}
    ft.append(r)

# ---- headroom proxy (daily): D-2 econ max same hour - Meteologica load, daily min
g = pd.read_parquet(ROOT / "data/pjm_rt/gen_cap.parquet")
em = g.assign(ts=ts_of(g)).groupby("ts")["economic_max"].mean().astype(float)
ld = pd.read_parquet(ROOT / "data/pjm_load/load.parquet")
ld = ld[ld["zone"] == "ALL"]
L = ld.assign(ts=ts_of(ld)).groupby("ts")["load_mwh"].mean()
full = pd.date_range("2024-12-01", Y1 - pd.Timedelta("1h"), freq="h")
emf = em.reindex(full)
proxy = (emf.shift(48).reindex(idx) - raw["pjm_load"]).groupby(day).min()
real = (em.reindex(idx) - L.reindex(idx)).groupby(day).min()
Dd = pd.DataFrame({"spike": (rt.groupby(day).max() >= SPIKE), "proxy": proxy, "real": real, "peak": raw["pjm_load"].groupby(day).max()}).dropna()
Dd["q"] = pd.qcut(Dd["peak"], 5, labels=False)


def within(c):
    a = [(auc(-gq[c], gq["spike"]), gq["spike"].sum()) for _, gq in Dd.groupby("q") if 0 < gq["spike"].sum() < len(gq)]
    return float(sum(x * n for x, n in a) / sum(n for _, n in a))


hp = {"auc_proxy": auc(-Dd["proxy"], Dd["spike"]), "auc_real": auc(-Dd["real"], Dd["spike"]), "within_proxy": within("proxy"),
      "within_real": within("real"), "corr": float(Dd["proxy"].corr(Dd["real"])),
      "spike_hr_head_mean": float(real.mean())}
q5 = pd.qcut(Dd["proxy"], 5)
hp["quint"] = [{"k": f"{iv.left / 1e3:.0f}-{iv.right / 1e3:.0f} GW", "share": float(gq["spike"].mean()), "days": int(len(gq))}
               for iv, gq in Dd.groupby(q5, observed=True)]
print("headroom", {k: v for k, v in hp.items() if k != "quint"})

# ---- hourly series for the chart
ser = {"t0": str(idx[0]), "rt": rt, "da": da, "base_med": P["rt_base"]["q500"], "base_q95": P["rt_base"]["q950"],
       "pf_med": P["pf_all"]["q500"], "pf_q95": P["pf_all"]["q950"], "load_med": P["pf_load"]["q500"]}
ser = {k: (v if isinstance(v, str) else [None if pd.isna(x) else round(float(x), 1) for x in v]) for k, v in ser.items()}

out = {"abl": abl, "mm": mm, "sp": sp, "ft": ft, "hp": hp, "ser": ser,
       "n": {"hold_days": int(day[hold].nunique()), "all_days": int(day[ok].nunique()), "spike": SPIKE,
             "n_feat": int(T["feature"].nunique()), "n_tests": int(len(T))}}
OUT.mkdir(parents=True, exist_ok=True)
js = json.dumps(rnd(out), separators=(",", ":"))
(OUT / "artifact6.json").write_text(js)
tpl = Path(__file__).parent / "artifact6_template.html"
if tpl.exists():
    page = OUT / "pjm-rt-spikes.html"
    page.write_text(tpl.read_text().replace("__DATA__", js))
    print(f"wrote {page} ({page.stat().st_size / 1e3:.0f} kB)")
