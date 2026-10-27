"""50/50 blend of Kumo RT and DA forecasts, scored on RT (2026 test, 2025 for reference).

  uv run python research/pjm_rt/blend26.py

  rt     v0_congfc5_xmkt_rt26_daily: RT target, baseline inputs + 5 congestion PCs + MISO / SPP DA forecast medians
         (--context-gap-hours 24)
  da     v0_congfc5_xmkt26_daily: same inputs, DA target (Artifact 7's new model), used as an RT forecast
  blend  0.5 rt + 0.5 da (medians; quantile-averaged for pinball)
  daprc  the DA price itself (clears ~13:30 EPT on D-1, after the 11:00 cutoff the Kumo forecasts use)
Trading days = not Monday. Paired day-block bootstrap (2,000 draws); +gain = first-named better.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
QS = [("q050", .05), ("q100", .1), ("q250", .25), ("q500", .5), ("q750", .75), ("q900", .9), ("q950", .95)]
raw = pd.read_parquet(ROOT / "data/pjm_raw.parquet")
raw = raw.assign(ts=pd.to_datetime(raw["time"])).drop_duplicates("ts").set_index("ts")


def load(r, idx):
    d = pd.read_parquet(ROOT / "outputs/pjm-fe" / r / "val_walk_forward_predictions.parquet")
    return d.assign(ts=pd.to_datetime(d["ts"])).drop_duplicates("ts").set_index("ts").reindex(idx)


rng = np.random.default_rng(0)


def mae_gain(rt, a, b, m, day):
    g = ((rt - b).abs() - (rt - a).abs())[m].groupby(day[m]).mean().values
    bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
    return f"{g.mean():+.2f}±{np.std(bs):.2f} ({(g > 0).mean():.0%})"


def rmse_gain(rt, a, b, m, day):
    sa = ((rt - a) ** 2)[m].groupby(day[m]).sum().values
    sb = ((rt - b) ** 2)[m].groupby(day[m]).sum().values
    n = m[m].groupby(day[m]).size().values
    f = lambda i: np.sqrt(sb[i].sum() / n[i].sum()) - np.sqrt(sa[i].sum() / n[i].sum())  # noqa: E731
    bs = [f(rng.integers(0, len(n), len(n))) for _ in range(2000)]
    return f"{f(np.arange(len(n))):+.2f}±{np.std(bs):.2f}"


for year, y0, y1 in (("2026", "2026-01-01", "2026-10-09"), ("2025", "2025-01-01", "2026-01-01")):
    idx = pd.date_range(y0, pd.Timestamp(y1) - pd.Timedelta("1h"), freq="h")
    rt, daprc = raw["rt_energy_price"].reindex(idx), raw["da_energy_price"].reindex(idx)
    R, D = load("v0_congfc5_xmkt_rt26_daily", idx), load("v0_congfc5_xmkt26_daily", idx)
    F = {"rt": R.q500, "da": D.q500, "blend": .5 * R.q500 + .5 * D.q500, "daprc": daprc}
    day = pd.Series(idx.normalize(), idx)
    ok = rt.notna() & R.q500.notna() & D.q500.notna() & daprc.notna()
    n_day = ok.groupby(day).transform("sum")
    ok &= n_day >= 23
    PB = {k: sum(np.maximum(t * (rt - X[q]), (t - 1) * (rt - X[q])) for q, t in QS) / len(QS)
          for k, X in (("rt", R), ("da", D), ("blend", pd.DataFrame({q: .5 * R[q] + .5 * D[q] for q, _ in QS})))}
    for mn, m in (("trading days", ok & (idx.dayofweek != 0)), ("all days", ok)):
        print(f"\n== {year} {mn}: {day[m].nunique()} d, {day[m].min().date()}..{day[m].max().date()}, RT avg {rt[m].mean():.1f} max {rt[m].max():.0f}")
        for k, f in F.items():
            e = (rt - f)[m]
            s = f"  {k:6} MAE {e.abs().mean():6.2f} RMSE {np.sqrt((e ** 2).mean()):6.2f} bias {e.mean():+6.2f}"
            if k in PB:
                s += f" pinball7 {PB[k][m].mean():.2f}"
            print(s)
        for a, b in (("blend", "rt"), ("blend", "da"), ("blend", "daprc"), ("rt", "daprc"), ("da", "daprc")):
            print(f"  {a} vs {b:5}: MAE {mae_gain(rt, F[a], F[b], m, day):22} RMSE {rmse_gain(rt, F[a], F[b], m, day)}")
        for a, b in (("blend", "rt"), ("blend", "da")):
            g = (PB[b] - PB[a])[m].groupby(day[m]).mean().values
            bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
            print(f"  pinball7 {a} vs {b}: {g.mean():+.3f}±{np.std(bs):.3f}")
        sp = m & (rt >= 150)
        print(f"  RT>=150 ({int(sp.sum())} h) / normal MAE: " + " ".join(f"{k} {(rt - f).abs()[sp].mean():.0f}/{(rt - f).abs()[m & ~sp].mean():.1f}" for k, f in F.items()))
        if year == "2026" and mn == "trading days":
            mo = pd.Series(idx.strftime("%b"), idx)
            for k in dict.fromkeys(mo[m]):
                mm = m & (mo == k)
                print(f"    {k} ({day[mm].nunique()} d, RT avg {rt[mm].mean():.0f}) MAE " + " ".join(f"{c} {(rt - f).abs()[mm].mean():.2f}" for c, f in F.items()))
            w = np.arange(0, 1.01, .1)
            print("    weight on RT, MAE: " + " ".join(f"{x:.1f}:{(rt - (x * R.q500 + (1 - x) * D.q500)).abs()[m].mean():.2f}" for x in w))
