"""Do renewable ramp misses (and realised margins) separate RT spikes beyond net load? 2025 only.

  uv run python research/pjm_rt/spikes_ramp.py

Hypothesis (trader's eye): spikes follow renewables running above forecast and then falling below it, so the realised
renewable ramp is much steeper than forecast, sometimes with no matching net-load ramp miss.
Labels: spike = RT >= 150; premium = RT - DA >= 75 (RT the DA market did not price).
Each feature is scored by AUC within strata, so what the strata explain does not count:
  band  = 5 GW actual-net-load band x season (removes the net-load level)
  +nre  = band x tercile of the 3-hour net-load ramp error vs ISO (removes the net-load ramp miss too)
  +head = band x tercile of available-capacity headroom (does a fuel breakdown add beyond total headroom?)
95% intervals from a day-block bootstrap (resample days); "H1 / H2" = the same AUC on Jan-Jun and Jul-Dec separately,
a check that the effect is not one episode. Sign convention: error = actual - forecast. A negative renewable ramp
error = renewables fell faster (or rose slower) than forecast.
Actuals from SQL (data/pjm_rt/*, pulled 2026-10-26): pjm_hourly_generation_by_types, pjm_gen_capacities (ALL),
pjm_iir_outage_total_summaries, pjm_hourly_regulation_reserves. Symptoms (oil burn, regulation price) are marked:
they move with the spike but are not causes to forecast.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/spp_load"))
from make_features import ts_of  # noqa: E402

Y0, Y1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
idx = pd.date_range(Y0, Y1 - pd.Timedelta("1h"), freq="h")
NB = 200
rng = np.random.default_rng(0)


def hourly(f, val, **flt):
    d = pd.read_parquet(ROOT / "data/pjm_rt" / f)
    for k, v in flt.items():
        d = d[d[k] == v]
    return d.assign(ts=ts_of(d)).groupby("ts")[val].mean().reindex(idx)


raw = pd.read_parquet(ROOT / "data/pjm_raw.parquet")
raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts").reindex(idx)
rt, da = raw["rt_energy_price"], raw["da_energy_price"]
v0 = pd.read_parquet(ROOT / "data/pjm-fe/v0/val.parquet").assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts").reindex(idx)
bench = pd.read_parquet(ROOT / "data/pjm_renew/bench.parquet").assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts").reindex(idx)
ld = pd.read_parquet(ROOT / "data/pjm_load/load.parquet")
ld = ld[ld["zone"] == "ALL"]
L = ld.assign(ts=ts_of(ld)).groupby("ts")["load_mwh"].mean().reindex(idx)

act = pd.DataFrame({"load": L, "wind": bench["wind"], "solar": bench["solar"]})
fc = {"met": pd.DataFrame({"load": v0["pjm_load"], "wind": v0["pjm_wind"], "solar": v0["pjm_solar"]}),
      "iso": pd.DataFrame({"load": bench["iso_load"], "wind": bench["iso_wind"], "solar": bench["iso_solar"]})}
for D in (act, *fc.values()):
    D["ren"] = D["wind"] + D["solar"]
    D["net"] = D["load"] - D["ren"]

F = pd.DataFrame(index=idx)
for s, D in fc.items():
    e = act - D
    for c in ("ren", "wind", "solar", "load", "net"):
        F[f"{c}_ramp_err_{s}"] = act[c].diff() - D[c].diff()
        F[f"{c}_ramp3_err_{s}"] = act[c].diff(3) - D[c].diff(3)
    F[f"ren_err_{s}"] = e["ren"]
    # above-then-below: how far renewables fell from their best showing vs forecast in the previous 6 hours
    F[f"ren_swing6_{s}"] = e["ren"].shift(1).rolling(6).max() - e["ren"]
    F[f"wind_swing6_{s}"] = e["wind"].shift(1).rolling(6).max() - e["wind"]
    F[f"solar_swing6_{s}"] = e["solar"].shift(1).rolling(6).max() - e["solar"]
    F[f"net_swing6_{s}"] = e["net"] - e["net"].shift(1).rolling(6).min()
day = pd.Series(idx.normalize(), idx)
F["ren_swing_day_iso"] = (act["ren"] - fc["iso"]["ren"]).groupby(day).transform(lambda x: x.cummax() - x)
F["act_ren_ramp3"] = act["ren"].diff(3)

cap = {c: hourly("gen_cap.parquet", c) for c in ("total_committed", "economic_max", "emergency_max")}
F["headroom_econ"] = cap["economic_max"] - act["load"]
F["headroom_committed"] = cap["total_committed"] - act["load"]
F["headroom_emerg"] = cap["emergency_max"] - act["load"]
F["headroom_econ_d1"] = F["headroom_econ"].diff()
F["iir_offline_mw"] = hourly("iir_offline.parquet", "offline_mw")
F["iir_starting_outs_mw"] = hourly("iir_offline.parquet", "startingouts_offline_mw")
gen = pd.read_parquet(ROOT / "data/pjm_rt/gen_by_fuel.parquet")
gen = gen.assign(ts=ts_of(gen)).pivot_table(index="ts", columns="fuel_type", values="power_mw", aggfunc="mean").reindex(idx)
F["gas_gen_ramp3"] = gen["Gas"].diff(3)
F["nuclear_gen"] = gen["Nuclear"]
F["nuclear_vs_d2"] = gen["Nuclear"] - gen["Nuclear"].shift(48)          # nuclear lost since D-2 (trips, derates)
F["coal_gen"] = gen["Coal"]
F["net_imports"] = act["load"] - gen.sum(axis=1)
F["net_imports_vs_d2"] = F["net_imports"] - F["net_imports"].shift(48)
F["hydro_gen"] = gen["Hydro"]
F["SYMPTOM_gas_gen"] = gen["Gas"]
# supply-stack slack from PJM data only (not IIR): a fuel's practical capacity = its highest hourly output over the
# previous 30 days (tracks retirements and long outages); spare = that minus current output
gl = pd.read_parquet(ROOT / "data/pjm_rt/gen_by_fuel.parquet")
gl = gl.assign(ts=ts_of(gl)).pivot_table(index="ts", columns="fuel_type", values="power_mw", aggfunc="mean")
for fu, nm in (("Gas", "gas"), ("Coal", "coal"), ("Oil", "oil"), ("Hydro", "hydro"), ("Nuclear", "nuclear")):
    cap30 = gl[fu].shift(1).rolling(720, min_periods=600).max()
    F[f"spare_{nm}"] = (cap30 - gl[fu]).reindex(idx)
F["spare_thermal"] = F[["spare_gas", "spare_coal", "spare_oil"]].sum(axis=1, min_count=3)
F["spare_gas_pct"] = F["spare_gas"] / (gl["Gas"].shift(1).rolling(720, min_periods=600).max()).reindex(idx)
F["system_slack"] = cap["economic_max"] - gen.sum(axis=1)
og = pd.read_parquet(ROOT / "data/pjm_rt/outages_all.parquet")
og = og.assign(ts=ts_of(og)).groupby("ts")[["forced_outages_mw", "capacity_on_outage_mw"]].mean().reindex(idx)
F["pjm_forced_outages_fc"] = og["forced_outages_mw"]
F["SYMPTOM_oil_gen"] = gen["Oil"]
F["SYMPTOM_reg_price"] = hourly("regulation.parquet", "market_clear_price")
F["gen_on_outage_fc"] = v0["pjm_gen_on_outage"]

ok = rt.notna() & act["net"].notna()
seas = pd.Series(np.select([idx.month.isin([6, 7, 8]), idx.month.isin([12, 1, 2])], ["S", "W"], "X"), idx)
band = (act["net"] // 5000).astype("Int64").astype(str) + seas
nre = pd.qcut(F["net_ramp3_err_iso"], 3, labels=False).astype("Int64").astype(str)
# third control: also hold total headroom (available capacity - load) fixed, in terciles
headq = pd.qcut(F["headroom_econ"], 3, labels=False).astype("Int64").astype(str)
labels = {"spike": rt >= 150, "premium": (rt - da) >= 75}
day_code = pd.Series(pd.factorize(day)[0], idx)
half = pd.Series(np.where(idx.month <= 6, "H1", "H2"), idx)


def wauc(s, l, g, w):
    """Weighted within-stratum AUC (weights = bootstrap multiplicity), averaged over strata by positive weight.
    g = list of index arrays, one per stratum."""
    num = den = 0.0
    for m in g:
        sl, ll, wl = s[m], l[m], w[m]
        wp, wn = wl[ll].sum(), wl[~ll].sum()
        if wp == 0 or wn == 0:
            continue
        o = np.argsort(sl, kind="mergesort")
        ss, lo, wo = sl[o], ll[o], wl[o]
        # weighted count of negatives strictly below + half ties, for each positive
        negw = np.where(~lo, wo, 0.0)
        uniq, inv = np.unique(ss, return_inverse=True)
        neg_at = np.bincount(inv, weights=negw, minlength=len(uniq))
        below = np.concatenate([[0.0], np.cumsum(neg_at)[:-1]])
        a = ((below[inv] + 0.5 * neg_at[inv]) * np.where(lo, wo, 0.0)).sum() / (wp * wn)
        num, den = num + a * wp, den + wp
    return num / den if den else np.nan


def score(feat, lab, strat, m):
    s, l, g = F[feat][m].to_numpy(float), lab[m].to_numpy(bool), strat[m].to_numpy()
    good = ~np.isnan(s)
    s, l, dc = s[good], l[good], day_code[m].to_numpy()[good]
    codes = pd.factorize(pd.Series(g[good]).fillna("na"))[0]
    g = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
    gl = [x for x in g if l[x].any() and (~l[x]).any()]
    g = gl
    a = wauc(s, l, g, np.ones(len(s)))
    nd = day_code.max() + 1
    bs = []
    for _ in range(NB):
        w = np.bincount(rng.integers(0, nd, nd), minlength=nd)[dc].astype(float)
        bs.append(wauc(s, l, g, w))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    hv = half[m].to_numpy()[good]
    hh = [wauc(s, l, [x[hv[x] == h] for x in g], np.ones(len(s))) for h in ("H1", "H2")]
    return a, lo, hi, hh[0], hh[1]


rows = []
for ln, lab in labels.items():
    m = ok if ln == "premium" else ok & (act["net"] >= act["net"][lab & ok].quantile(0.1))
    print(f"{ln}: {int(lab[m].sum())} positive hours on {day[m & lab].nunique()} days, of {int(m.sum())}")
    for sn, strat in (("band", band), ("band+nre", band + "_" + nre), ("band+head", band + "_" + headq)):
        for c in F:
            a, lo, hi, h1, h2 = score(c, lab, strat, m)
            rows.append({"label": ln, "strata": sn, "feature": c, "auc": a, "lo": lo, "hi": hi, "H1": h1, "H2": h2,
                         "mean_pos": F[c][m & lab].mean(), "mean_neg": F[c][m & ~lab].mean()})
T = pd.DataFrame(rows)
T["sig"] = (T["lo"] > 0.5) | (T["hi"] < 0.5)
T["stable"] = np.sign(T["H1"] - 0.5) == np.sign(T["H2"] - 0.5)
OUT = ROOT / "outputs/pjm-rt/spikes"
OUT.mkdir(parents=True, exist_ok=True)
T.to_csv(OUT / "ramp_auc.csv", index=False)
pd.set_option("display.width", 220)
for (ln, sn), t in T.groupby(["label", "strata"], sort=False):
    t = t.assign(d=(t["auc"] - 0.5).abs()).sort_values("d", ascending=False).drop(columns=["label", "strata", "d"])
    print(f"\n== {ln}, strata {sn} (top 22)")
    print(t.head(22).round(3).to_string(index=False))
