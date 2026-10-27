"""Kumo RT model (5 congestion PCs) vs the production EUM model (flex.eum.cvar.20261001) on 2025 RT.

  uv run python research/pjm_rt/compare_eum.py [--run v0_congfc_rt_tail_daily]

EUM (Snowflake TIOSDATA_PRODUCTION.TIOSDATA.PJM_TRADER_FORECAST_SNAPSHOTS, data/pjm_rt/eum_cvar.parquet; one
snapshot per hour stamped D-1 10:35 EPT): forecast (point), upper_80 = upper 5% CVaR (mean of the top 5% of the RT
distribution), lower_80 = lower 5% CVaR. Kumo: q500 / mean (point), cvar_hi050 / cvar_lo050 = mean of quantiles
q951..q999 / q001..q049 of its 999-quantile grid (same definition), q950 = its upper 5% VaR.

1. Point forecasts: MAE / RMSE / bias, daily-paired gains with a day-block bootstrap.
2. Upper CVaR. ES alone has no strictly consistent score, so:
   a. Regression backtest (Bayer & Dimitriadis 2020, auxiliary ES regression): fit VaR_t = a + b c_t and
      ES_t = g + d c_t jointly by minimising the FZ0 loss (Patton, Ziegel & Chen 2019) at alpha = 5% on the upper
      tail; a calibrated CVaR forecast c_t has g = 0, d = 1. Encompassing: ES_t = g + d1 c_eum + d2 c_kumo; the
      forecast that keeps the weight carries the information. Mean FZ0 loss of each fit = its information content
      after a free linear recalibration (lower is better); day-block bootstrap of the loss difference.
   b. Plain checks: share of hours with RT above the CVaR, AUC for spike hours, calibration by forecast decile
      (mean forecast vs realised mean of the top 5% of RT in the decile; a mixture, so only roughly comparable).
   c. Kumo alone also gets the joint FZ0 score with its own VaR (q950); EUM publishes no VaR.
3. Lower CVaR: plain checks only.
Scored on hours where both exist; HOLD = day number not a multiple of 3; 2025 only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--run", default="v0_congfc_rt_tail_daily")
a = p.parse_args()
Y0 = pd.Timestamp("2025-01-01")
idx = pd.date_range(Y0, "2025-12-31 23:00", freq="h")
ALPHA = 0.05
rng = np.random.default_rng(0)

k = pd.read_parquet(ROOT / f"outputs/pjm-fe/{a.run}/val_walk_forward_predictions.parquet")
k = k.assign(ts=pd.to_datetime(k["ts"])).set_index("ts").reindex(idx)
assert "cvar_hi050" in k, "run has no cvar_hi050: re-run with the updated kumo_forecast.py"
e = pd.read_parquet(ROOT / "data/pjm_rt/eum_cvar.parquet")
e = e.assign(ts=pd.to_datetime(e["dt"]) + pd.to_timedelta(e["hr"] - 1, unit="h")).drop_duplicates("ts").set_index("ts").reindex(idx)
rt = k["y_true"]
day = pd.Series(idx.normalize(), idx)
dn = pd.Series((idx.normalize() - Y0).days, idx)
ok = rt.notna() & k["q500"].notna() & e["forecast"].notna() & e["upper_80"].notna()
S = {"HOLD": ok & (dn % 3 != 0), "ALL": ok}
print(f"hours with both: {ok.sum()}; EUM missing {int((rt.notna() & e['forecast'].isna()).sum())}")


def boot(d):
    d = d.to_numpy()
    return d.mean(), np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])


# ---------------- 1. point
print("\n1. Point forecast (RT minus forecast for bias)")
P = {"EUM forecast": e["forecast"], "Kumo median": k["q500"], "Kumo mean": k["mean"]}
for nm, m in S.items():
    print(f"  {nm}: " + " | ".join(f"{c} MAE {(rt - f)[m].abs().mean():.2f} RMSE {np.sqrt(((rt - f)[m] ** 2).mean()):.2f} "
                                    f"bias {(rt - f)[m].mean():+.2f}" for c, f in P.items()))
    for c in ("Kumo median", "Kumo mean"):
        g, se = boot(((rt - P["EUM forecast"]).abs() - (rt - P[c]).abs())[m].groupby(day[m]).mean())
        print(f"     {c} vs EUM: daily MAE gain {g:+.3f} +- {se:.3f}")
        sq = ((rt - P["EUM forecast"]) ** 2 - (rt - P[c]) ** 2)[m].groupby(day[m]).mean()
        g, se = boot(sq)
        print(f"     {c} vs EUM: daily squared-error gain {g:+.1f} +- {se:.1f}")


# ---------------- 2. upper CVaR
def fz0(v, es, y):
    """FZ0 loss for the upper alpha tail of y (Patton-Ziegel-Chen on -y); needs es > 0 (prices here)."""
    v_, e_, y_ = -v, -es, -y
    return -1 / (ALPHA * e_) * (y_ <= v_) * (v_ - y_) + v_ / e_ + np.log(-e_) - 1


def esr(X, y, w=None):
    """Auxiliary ES regression: VaR = X @ b, ES = X @ g, minimise mean FZ0 (weights w). Returns (b, g, loss)."""
    w = np.ones(len(y)) if w is None else w
    n = X.shape[1]
    # start: VaR = 0.6 c, ES = c on the first forecast column, zero weight on others
    st_b, st_g = np.zeros(n), np.zeros(n)
    st_b[1], st_g[1] = 0.6, 1.0
    x0 = np.concatenate([st_b, st_g])

    def f(th):
        v, es = X @ th[:n], X @ th[n:]
        if np.any(es <= 1e-6):
            return 1e9
        return np.average(fz0(v, es, y), weights=w)

    r = minimize(f, x0, method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-6, "fatol": 1e-9})
    r = minimize(f, r.x, method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-7, "fatol": 1e-10})
    return r.x[:n], r.x[n:], r.fun


C = {"EUM": e["upper_80"], "Kumo": k["cvar_hi050"]}
print("\n2a. Upper 5% CVaR: regression backtest (calibrated: g = 0, d = 1) and information (mean FZ0 after recalibration)")
for nm, m in S.items():
    y = rt[m].to_numpy()
    fits = {}
    for c, cv in C.items():
        X = np.column_stack([np.ones(m.sum()), cv[m].to_numpy()])
        b, g, L = esr(X, y)
        fits[c] = L
        print(f"  {nm} {c:4}: ES = {g[0]:+7.2f} + {g[1]:.3f} x CVaR   (VaR = {b[0]:+.2f} + {b[1]:.3f} x CVaR)   FZ0 {L:.4f}")
    X = np.column_stack([np.ones(m.sum()), C["EUM"][m].to_numpy(), C["Kumo"][m].to_numpy()])
    b, g, L = esr(X, y)
    print(f"  {nm} both: ES = {g[0]:+7.2f} + {g[1]:.3f} x EUM + {g[2]:.3f} x Kumo   FZ0 {L:.4f}")
    # raw (unrecalibrated) FZ0 for Kumo with its own VaR
    print(f"  {nm} Kumo raw joint score with its own VaR (q950): FZ0 {fz0(k['q950'][m].to_numpy(), C['Kumo'][m].to_numpy(), y).mean():.4f}")
    # bootstrap the information difference: refit on day resamples (weights)
    days = pd.factorize(day[m])[0]
    nd = days.max() + 1
    diffs, gd = [], {c: [] for c in C}
    for _ in range(100):
        w = np.bincount(rng.integers(0, nd, nd), minlength=nd)[days].astype(float)
        Ls = []
        for c in ("EUM", "Kumo"):
            _, g_, L_ = esr(np.column_stack([np.ones(m.sum()), C[c][m].to_numpy()]), y, w)
            Ls.append(L_)
            gd[c].append(g_)
        diffs.append(Ls[0] - Ls[1])
    for c in C:
        G = np.array(gd[c])
        print(f"  {nm} {c:4} bootstrap 95%: intercept g {np.percentile(G[:, 0], 2.5):+.2f} .. {np.percentile(G[:, 0], 97.5):+.2f}, "
              f"slope d {np.percentile(G[:, 1], 2.5):.3f} .. {np.percentile(G[:, 1], 97.5):.3f}")
    print(f"  {nm} FZ0 EUM minus Kumo (positive = Kumo more informative): {fits['EUM'] - fits['Kumo']:+.4f} "
          f"(bootstrap 95% {np.percentile(diffs, 2.5):+.4f} .. {np.percentile(diffs, 97.5):+.4f})")


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l, bool)
    rk = pd.Series(s).rank().to_numpy()
    n1, n0 = l.sum(), (~l).sum()
    return (rk[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print("\n2b. Upper CVaR plain checks")
for nm, m in S.items():
    for c, cv in C.items():
        r_ = rt[m]
        print(f"  {nm} {c:4}: mean CVaR {cv[m].mean():6.1f} | RT above CVaR {(r_ > cv[m]).mean():.2%} | AUC RT>=150 "
              f"{auc(cv[m], r_ >= 150):.3f}, RT>=300 {auc(cv[m], r_ >= 300):.3f} | spike hours (>=150) mean CVaR "
              f"{cv[m][r_ >= 150].mean():.0f} vs RT {r_[r_ >= 150].mean():.0f}")
m = S["ALL"]
for c, cv in C.items():
    q = pd.qcut(cv[m], 10, labels=False, duplicates="drop")
    rows = []
    for b_, g_ in rt[m].groupby(q):
        top = g_.sort_values().tail(max(1, int(round(ALPHA * len(g_)))))
        rows.append(f"{cv[m][q == b_].mean():.0f}/{top.mean():.0f}")
    print(f"  ALL {c:4} by forecast decile, mean forecast / realised top-5% mean: " + " ".join(rows))

print("\n3. Lower 5% CVaR (plain checks)")
L_ = {"EUM": e["lower_80"], "Kumo": k["cvar_lo050"]}
for nm, mm in S.items():
    for c, cv in L_.items():
        r_ = rt[mm]
        print(f"  {nm} {c:4}: mean {cv[mm].mean():6.1f} | RT below it {(r_ < cv[mm]).mean():.2%}")
