"""Why perfect foresight of load / wind / solar barely helps RT spikes (2025 only).

  uv run python research/pjm_rt/spikes.py

Part A (models): on spike hours, what do rt_base and rt_pf forecast (q500 / q900 / q950 / mean), and do they rank
spike hours higher (AUC)? Is the pf gain on spike hours or elsewhere?
Part B (conditions): which realised-minus-forecast quantities separate spike hours from other hours (univariate
AUC, stratified by hour-of-day x month so plain seasonality does not count), and the spike rate by error decile.
Spike = RT >= SPIKE $/MWh. Forecast = Meteologica ALL (v0's inputs) or ISO (latest by D-1 11:00 EPT).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/spp_load"))
from make_features import ts_of  # noqa: E402

SPIKE, BIG = 150.0, 300.0
Y0, Y1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
idx = pd.date_range(Y0, Y1 - pd.Timedelta("1h"), freq="h")


def auc(score, lab):
    """P(score on a spike hour > score on a non-spike hour); ties count half."""
    s, l = np.asarray(score, float), np.asarray(lab, bool)
    ok = ~np.isnan(s)
    s, l = s[ok], l[ok]
    rk = pd.Series(s).rank().to_numpy()
    n1, n0 = l.sum(), (~l).sum()
    return (rk[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def strat_auc(score, lab, grp):
    """AUC within strata (weighted by spike count): removes what hour-of-day x month alone explains."""
    num = den = 0.0
    for _, g in pd.DataFrame({"s": score, "l": lab, "g": grp}).groupby("g"):
        if g["l"].sum() and (~g["l"]).sum():
            a = auc(g["s"], g["l"])
            if not np.isnan(a):
                num, den = num + a * g["l"].sum(), den + g["l"].sum()
    return num / den if den else np.nan


raw = pd.read_parquet(ROOT / "data/pjm_raw.parquet")
raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts").reindex(idx)
rt, da = raw["rt_energy_price"], raw["da_energy_price"]
spk = rt >= SPIKE
print(f"2025 hours {rt.notna().sum()}, spike hours (RT >= {SPIKE:.0f}) {spk.sum()} on {idx[spk].normalize().nunique()} days, "
      f">= {BIG:.0f}: {(rt >= BIG).sum()}; DA >= {SPIKE:.0f}: {(da >= SPIKE).sum()}")

# ---------------- Part A: the models on spike hours
pr = {k: pd.read_parquet(ROOT / f"outputs/pjm-fe/{v}/val_walk_forward_predictions.parquet")
      .assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts").reindex(idx)
      for k, v in (("da_base", "s3_stage/baseline_v0_daily"), ("rt_base", "v0_rt_daily"), ("rt_pf", "v0_actuals_all_rt_daily"))}
print("\nA1. Spike hours: average RT vs average forecast; hourly AUC for spike vs not")
print(f"   RT on spike hours: mean {rt[spk].mean():.0f}, median {rt[spk].median():.0f}")
for k, p in pr.items():
    row = " ".join(f"{q} {p[q][spk].mean():6.1f} (AUC {auc(p[q], spk):.3f})" for q in ("q500", "q900", "q950", "mean"))
    print(f"   {k:8} {row}")
for k in ("rt_base", "rt_pf"):
    p = pr[k]
    print(f"   {k}: spike hours with q950 >= {SPIKE:.0f}: {(p['q950'][spk] >= SPIKE).mean():.1%}; non-spike hours with "
          f"q950 >= {SPIKE:.0f}: {(p['q950'][~spk & rt.notna()] >= SPIKE).mean():.1%}")

print("\nA2. Where the perfect-foresight gain comes from (sum of |error| reduction, rt_base -> rt_pf)")
e_b, e_p = (rt - pr["rt_base"]["q500"]).abs(), (rt - pr["rt_pf"]["q500"]).abs()
d = (e_b - e_p)
for nm, m in (("spike hours", spk), ("RT 75-150", (rt >= 75) & ~spk), ("RT < 75", rt < 75)):
    print(f"   {nm:12} hours {m.sum():5}  MAE base {e_b[m].mean():6.2f} pf {e_p[m].mean():6.2f}  share of total gain {d[m].sum() / d.sum():6.1%}")
print("   spike hours, mean signed error (RT - q500): base "
      f"{(rt - pr['rt_base']['q500'])[spk].mean():.0f}, pf {(rt - pr['rt_pf']['q500'])[spk].mean():.0f}")

# daily level: does pf know which days spike?
day = pd.Series(idx.normalize(), idx)
dmax = rt.groupby(day).max()
dspk = dmax >= SPIKE
print(f"\nA3. Day level ({dspk.sum()} spike days): AUC of each model's daily max forecast for 'day has a spike hour'")
for k, p in pr.items():
    print(f"   {k:8} " + " ".join(f"{q} {auc(p[q].groupby(day).max(), dspk):.3f}" for q in ("q500", "q950")))

# ---------------- Part B: realised vs forecast conditions
v0 = pd.read_parquet(ROOT / "data/pjm-fe/v0/val.parquet").assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts").reindex(idx)
pf = pd.read_parquet(ROOT / "data/pjm-fe/v0_actuals_all_rt/val.parquet").assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts").reindex(idx)
bench = pd.read_parquet(ROOT / "data/pjm_renew/bench.parquet").assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts").reindex(idx)
ld = pd.read_parquet(ROOT / "data/pjm_load/load.parquet")
ld = ld[ld["zone"] == "ALL"]
act = pd.DataFrame({"load": ld.assign(ts=ts_of(ld)).groupby("ts")["load_mwh"].mean().reindex(idx),
                    "wind": bench["wind"], "solar": bench["solar"]})
met = pd.DataFrame({"load": v0["pjm_load"], "wind": v0["pjm_wind"], "solar": v0["pjm_solar"]})
iso = pd.DataFrame({"load": bench["iso_load"], "wind": bench["iso_wind"], "solar": bench["iso_solar"]})
for D in (act, met, iso):
    D["net"] = D["load"] - D["wind"] - D["solar"]
    D["ramp"] = D["net"].diff()          # hour-on-hour net-load change
    D["ramp3"] = D["net"].diff(3)        # 3-hour net-load ramp

k_ts = v0["thermal_stress_all_pjm"] / (met["net"])
F = pd.DataFrame(index=idx)
for src, S in (("met", met), ("iso", iso)):
    for c in ("load", "wind", "solar", "net", "ramp", "ramp3"):
        F[f"{c}_err_{src}"] = act[c] - S[c]
    F[f"net_err_{src}_pct"] = (act["net"] - S["net"]) / S["net"]
F["net_err_met_daymax"] = F["net_err_met"].groupby(day).transform("max")
F["net_err_met_daysum"] = F["net_err_met"].groupby(day).transform("sum")
F["net_err_met_lag1"] = F["net_err_met"].shift(1)
F["net_err_met_cum3"] = F["net_err_met"].rolling(3).sum()
F["act_net"] = act["net"]
F["act_net_daymax"] = act["net"].groupby(day).transform("max")
F["act_ramp3"] = act["ramp3"]
F["met_net"] = met["net"]
F["met_net_daymax"] = met["net"].groupby(day).transform("max")
F["thermal_stress_met"] = v0["thermal_stress_all_pjm"]
F["thermal_stress_act"] = pf["thermal_stress_all_pjm"]
F["gen_on_outage"] = v0["pjm_gen_on_outage"]
F["feelslike"] = v0["feelslike_weightedavg"]
F["gas"] = v0["blended_gasprice"]
F["ext_stress_north"] = v0["north_external_stress"]
F["ext_stress_south"] = v0["south_external_stress"]
F["miso_stress"] = v0["thermal_stress_all_miso"]
F["rt_base_q500"] = pr["rt_base"]["q500"]
F["da_price"] = da
grp = pd.Series(idx.hour.astype(str) + "_" + idx.month.astype(str), idx)
ok = rt.notna()

rows = []
for c in F:
    s = F[c][ok]
    rows.append({"feature": c, "auc": auc(s, spk[ok]), "auc_strat": strat_auc(s, spk[ok], grp[ok]),
                 "mean_spike": s[spk[ok]].mean(), "mean_other": s[~spk[ok]].mean(), "sd": s.std()})
T = pd.DataFrame(rows)
T["dist"] = (T["auc_strat"] - 0.5).abs()
T = T.sort_values("dist", ascending=False)
pd.set_option("display.width", 200)
print(f"\nB1. Spike hours vs others. auc_strat = AUC within hour x month cells (0.5 = no signal; <0.5 = lower on spikes)")
print(T.drop(columns="dist").round(3).to_string(index=False))

print("\nB2. Spike rate (RT >= 150) by decile of actual-minus-Meteologica net load error (MW)")
q = pd.qcut(F["net_err_met"][ok], 10)
print(spk[ok].groupby(q, observed=True).agg(["mean", "size"]).round(3).to_string())
print("\nB3. Spike rate by decile of the daily max actual-minus-Meteologica net load error")
q = pd.qcut(F["net_err_met_daymax"][ok], 10)
print(spk[ok].groupby(q, observed=True).agg(["mean", "size"]).round(3).to_string())

# the spikes the model misses most: what were the errors?
worst = (rt - pr["rt_pf"]["q500"])[spk].sort_values(ascending=False).head(25).index
print("\nB4. 25 spike hours rt_pf misses most")
cols = ["net_err_met", "load_err_met", "wind_err_met", "solar_err_met", "ramp_err_met", "act_net", "gen_on_outage"]
W = pd.DataFrame({"rt": rt[worst], "da": da[worst], "pf_q500": pr["rt_pf"]["q500"][worst], "pf_q950": pr["rt_pf"]["q950"][worst]})
W = W.join(F.loc[worst, cols]).round(0)
print(W.to_string())
# context: how unusual are those errors, as percentiles of all 2025 hours
pct = {c: F[c].rank(pct=True) for c in cols}
print("   percentile of each error among 2025 hours, median over these 25:",
      {c: round(float(pct[c][worst].median()), 2) for c in cols})
OUT = ROOT / "outputs/pjm-rt/spikes"
OUT.mkdir(parents=True, exist_ok=True)
T.to_csv(OUT / "feature_auc.csv", index=False)
F.assign(rt=rt, da=da, spike=spk, **{f"{k}_{q}": pr[k][q] for k in pr for q in ("q500", "q950")}).to_parquet(OUT / "hourly.parquet")

# ---------------- Part C: at the same net load, what separates spike hours?
# Strata = 5 GW bands of actual net load x season, so load / wind / solar themselves cannot count.
# PJM operating summary (published ~08:00 EPT on D itself, i.e. after the cutoff: foresight) and the generation
# outage forecast (published 06:00 EPT on D-1, before the cutoff: usable today).
ops = pd.read_parquet(ROOT / "data/pjm_extra/op_summary.parquet").assign(dt=lambda d: pd.to_datetime(d["dt"])).set_index("dt")
og = pd.read_parquet(ROOT / "data/pjm_extra/outages.parquet")
og = og[og["region"] == "ALL"].assign(ts=lambda d: pd.to_datetime(d["dt"]) + pd.to_timedelta(d["hr"] - 1, unit="h")).set_index("ts")
og = og[~og.index.duplicated()].reindex(idx)
dD = pd.Series(idx.normalize(), idx)
for c in ("operating_reserve_mw", "scheduled_tie_flow_total_mw", "scheduled_capacity_total_mw", "unscheduled_steam_capacity_mw"):
    F[f"ops_{c}"] = ops[c].reindex(dD).to_numpy()
F["ops_reserve_pct"] = F["ops_operating_reserve_mw"] / ops["load_forecast_mw"].reindex(dD).to_numpy()
F["ops_cap_minus_act_net_daymax"] = F["ops_scheduled_capacity_total_mw"] - F["act_net_daymax"]
for c in ("forced_outages_mw", "planned_outages_mw", "maintenance_outages_mw", "capacity_on_outage_mw"):
    F[f"out_{c}"] = og[c]
F["hour"] = idx.hour
F["rt_lag_d2_max"] = rt.groupby(day).transform("max").shift(48)     # D-2's RT max, known at the cutoff
seas = pd.Series(np.select([idx.month.isin([6, 7, 8]), idx.month.isin([12, 1, 2])], ["S", "W"], "X"), idx)
band = (act["net"] // 5000).astype("Int64").astype(str) + seas
hi = ok & (act["net"] >= act["net"][spk].quantile(0.1))   # the net-load range where spikes happen
print(f"\nC1. Hours in the spike net-load range (actual net >= {act['net'][spk].quantile(0.1):.0f} MW): {hi.sum()}, "
      f"of which spikes {spk[hi].sum()} ({spk[hi].mean():.1%})")
rows = []
for c in F:
    s = F[c][hi]
    rows.append({"feature": c, "auc_in_band": strat_auc(s, spk[hi], band[hi]), "mean_spike": s[spk[hi]].mean(),
                 "mean_other": s[~spk[hi]].mean()})
C = pd.DataFrame(rows)
C = C.assign(dist=(C["auc_in_band"] - 0.5).abs()).sort_values("dist", ascending=False).drop(columns="dist")
print("   AUC within 5 GW net-load band x season (0.5 = no signal beyond net load)")
print(C.round(3).head(30).to_string(index=False))
C.to_csv(OUT / "feature_auc_in_band.csv", index=False)

print("\nC2. Daily: RT max by operating-reserve quintile (op summary, published D 08:00)")
Dd = pd.DataFrame({"rtmax": rt.groupby(day).max(), "res": ops["operating_reserve_mw"].reindex(dmax.index),
                   "netmax": act["net"].groupby(day).max(), "pf_q500max": pr["rt_pf"]["q500"].groupby(day).max()})
Dd["spikeday"] = Dd["rtmax"] >= SPIKE
print(Dd.groupby(pd.qcut(Dd["res"], 5), observed=True).agg(days=("rtmax", "size"), spike_share=("spikeday", "mean"),
      rtmax_med=("rtmax", "median"), rtmax_mean=("rtmax", "mean"), pf_q500max=("pf_q500max", "mean")).round(2).to_string())
top = Dd["netmax"] >= Dd["netmax"].quantile(0.8)
print(f"   among the top-20% net-load days ({top.sum()}): spike-day AUC of low reserve {auc(-Dd['res'][top], Dd['spikeday'][top]):.3f}, "
      f"of net max {auc(Dd['netmax'][top], Dd['spikeday'][top]):.3f}, of rt_pf daily q500 max {auc(Dd['pf_q500max'][top], Dd['spikeday'][top]):.3f}")
