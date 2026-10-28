"""Kumo and Timestral, RT and DA, and their blends, scored on RT (2026 test; 2025 for reference) and on DA.

  uv run python research/pjm_rt/blend_ts.py

Forecasts (q500 = median; all made with information at D-1 11:00 EPT, same inputs: baseline + 5 congestion PCs +
MISO / SPP DA forecast medians; walk-forward 2025-01-01..2026-10-08):
  k_rt   Kumo, RT target           outputs/pjm-fe/v0_congfc5_xmkt_rt26_daily  (context gap 24 h)
  k_da   Kumo, DA target           outputs/pjm-fe/v0_congfc5_xmkt26_daily
  t_rt   Timestral-209M, RT target outputs/pjm-fe/ts_congfc5_xmkt_rt26_daily  (history ends at D-2)
  t_da   Timestral-209M, DA target outputs/pjm-fe/ts_congfc5_xmkt26_daily
  k_bl = (k_rt + k_da)/2, t_bl = (t_rt + t_da)/2, kt_bl = (k_bl + t_bl)/2 (all four equally weighted)
  daprc  the DA price itself (clears ~13:30 EPT on D-1, after the cutoff)
Quantile blends average each quantile. Trading days = not Monday. Paired day-block bootstrap, 2,000 draws.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
QS = [("q050", .05), ("q100", .1), ("q250", .25), ("q500", .5), ("q750", .75), ("q900", .9), ("q950", .95)]
RUNS = {"k_rt": ("v0_congfc5_xmkt_rt26_daily", "val_walk_forward"), "k_da": ("v0_congfc5_xmkt26_daily", "val_walk_forward"),
        "t_rt": ("ts_congfc5_xmkt_rt26_daily", "val_timestral"), "t_da": ("ts_congfc5_xmkt26_daily", "val_timestral")}
BLENDS = {"k_bl": ("k_rt", "k_da"), "t_bl": ("t_rt", "t_da"), "kt_bl": ("k_rt", "k_da", "t_rt", "t_da"),
          "kt_rt": ("k_rt", "t_rt"), "kt_da": ("k_da", "t_da")}
raw = pd.read_parquet(ROOT / "data/pjm_raw.parquet")
raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts")
rng = np.random.default_rng(0)


def load(run, kind):
    d = pd.read_parquet(ROOT / "outputs/pjm-fe" / run / f"{kind}_predictions.parquet")
    return d.assign(ts=pd.to_datetime(d["ts"])).drop_duplicates("ts").set_index("ts")[[q for q, _ in QS]]


P = {k: load(*v) for k, v in RUNS.items()}
for k, parts in BLENDS.items():
    P[k] = sum(P[p] for p in parts) / len(parts)


def boot(g):
    bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
    return f"{g.mean():+.2f}±{np.std(bs):.2f}"


def rmse_gain(y, a, b, m, day):
    sa, sb = ((y - a) ** 2)[m].groupby(day[m]).sum().values, ((y - b) ** 2)[m].groupby(day[m]).sum().values
    n = m[m].groupby(day[m]).size().values
    f = lambda i: np.sqrt(sb[i].sum() / n[i].sum()) - np.sqrt(sa[i].sum() / n[i].sum())  # noqa: E731
    return f"{f(np.arange(len(n))):+.2f}±{np.std([f(rng.integers(0, len(n), len(n))) for _ in range(2000)]):.2f}"


def report(target, year, y0, y1, models, pairs, ref_name, ref):
    idx = pd.date_range(y0, pd.Timestamp(y1) - pd.Timedelta("1h"), freq="h")
    y = raw[f"{target}_energy_price"].reindex(idx)
    F = {k: P[k]["q500"].reindex(idx) for k in models}
    if ref is not None:
        F[ref_name] = ref.reindex(idx)
    day = pd.Series(idx.normalize(), idx)
    ok = y.notna() & pd.concat(F, axis=1).notna().all(axis=1)
    ok &= ok.groupby(day).transform("sum") >= 23
    m = ok & (idx.dayofweek != 0)
    print(f"\n== {target.upper()} target, {year} trading days: {day[m].nunique()} d, avg {y[m].mean():.1f}, max {y[m].max():.0f}")
    pin = {}
    for k, f in F.items():
        e = (y - f)[m]
        s = f"  {k:6} MAE {e.abs().mean():6.2f} RMSE {np.sqrt((e ** 2).mean()):6.2f} bias {e.mean():+6.2f}"
        if k in P:
            X = P[k].reindex(idx)
            pin[k] = sum(np.maximum(t * (y - X[q]), (t - 1) * (y - X[q])) for q, t in QS) / len(QS)
            s += f"  pinball7 {pin[k][m].mean():.2f}"
        print(s)
    for a, b in pairs:
        g = ((y - F[b]).abs() - (y - F[a]).abs())[m].groupby(day[m]).mean().values
        s = f"  {a:6} vs {b:6}: MAE {boot(g)} ({(g > 0).mean():.0%} days)  RMSE {rmse_gain(y, F[a], F[b], m, day)}"
        if a in pin and b in pin:
            s += f"  pinball7 {boot((pin[b] - pin[a])[m].groupby(day[m]).mean().values)}"
        print(s)
    sp = m & (y >= 150)
    print(f"  >=150 ({int(sp.sum())} h) / normal MAE: " + " ".join(f"{k} {(y - f).abs()[sp].mean():.0f}/{(y - f).abs()[m & ~sp].mean():.1f}" for k, f in F.items()))


YEARS = (("2026", "2026-01-01", "2026-10-09"), ("2025", "2025-01-01", "2026-01-01"))
prod = pd.read_parquet(ROOT / "data/pjm_extra/prod_flex_daep.parquet")
prod = prod.assign(ts=pd.to_datetime(prod["dt"]) + pd.to_timedelta(prod["hr"] - 1, unit="h")).drop_duplicates("ts").set_index("ts")["forecast"]
for yr in YEARS:
    report("rt", *yr, ["k_rt", "t_rt", "k_bl", "t_bl", "kt_bl"],
           [("t_bl", "daprc"), ("t_bl", "t_rt"), ("t_bl", "k_bl"), ("kt_bl", "k_bl"), ("kt_bl", "t_bl"), ("kt_bl", "k_rt"),
            ("kt_bl", "daprc"), ("k_bl", "daprc"), ("t_rt", "k_rt")], "daprc", raw["da_energy_price"])
for yr in YEARS:
    report("da", *yr, ["k_da", "t_da", "kt_da"], [("t_da", "k_da"), ("kt_da", "k_da"), ("kt_da", "prod"), ("k_da", "prod"), ("t_da", "prod")],
           "prod", prod)
