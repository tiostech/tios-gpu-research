"""Score the MISO RT backpublish (m0_rt_bp24_daily, 2024-2025) on MISO RT energy, against the MISO DA price and
Kumo's MISO DA forecast (m0_original_bp_daily, from 2023-10) used as RT forecasts.

  uv run python research/miso_rt/score_rt.py

"Trading days" = not Mondays. Paired day-block bootstrap (2,000 draws) of daily MAE gains (positive = RT Kumo better).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def preds(run):
    d = pd.read_parquet(ROOT / "outputs/miso-fe" / run / "val_walk_forward_predictions.parquet")
    return d.assign(ts=pd.to_datetime(d["ts"])).drop_duplicates("ts").set_index("ts")


rt, da_k = preds("m0_rt_bp24_daily"), preds("m0_original_bp_daily")
e = pd.read_parquet(ROOT / "data/miso/miso_energy_prices.parquet")
e = e.assign(ts=pd.to_datetime(e["dt"]) + pd.to_timedelta(e["hr"] - 1, unit="h")).drop_duplicates("ts").set_index("ts")
P = pd.DataFrame({"y": e["rt_energy_price"].reindex(rt.index), "kumo_rt": rt["q500"], "kumo_rt_mean": rt["mean"],
                  "da_price": e["da_energy_price"].reindex(rt.index), "kumo_da": da_k["q500"].reindex(rt.index),
                  "lo": rt["q100"], "hi": rt["q900"], "q95": rt["q950"]})
assert (P["y"] - rt["y_true"]).abs().max() < 1e-3
P["blend"] = 0.5 * (P["kumo_rt"] + P["kumo_da"])
M = ["kumo_rt", "kumo_rt_mean", "blend", "kumo_da", "da_price"]
day = P.index.floor("D")


def paired(m, a, b, n=2000):
    g = ((P["y"] - P[b]).abs() - (P["y"] - P[a]).abs())[m].groupby(day[m]).mean()
    rng = np.random.default_rng(0)
    return g.mean(), np.std([g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n)])


for yr in (2024, 2025):
    for lab, extra in (("trading", P.index.dayofweek != 0), ("all", np.ones(len(P), bool))):
        m = (P.index.year == yr) & extra & P[M].notna().all(axis=1).to_numpy()
        y = P["y"][m]
        print(f"\n{yr} {lab}: {day[m].nunique()} days, RT mean {y.mean():.2f} max {y.max():.0f}, hours >= 150: {(y >= 150).sum()}")
        for c in M:
            err = (y - P[c][m])
            print(f"  {c:13} MAE {err.abs().mean():6.2f}  RMSE {np.sqrt((err ** 2).mean()):6.2f}  bias {err.mean():+6.2f}")
        print(f"  80% coverage {((y >= P['lo'][m]) & (y <= P['hi'][m])).mean():.3f}")
        for b in ("da_price", "kumo_da"):
            g, se = paired(m, "kumo_rt", b)
            print(f"  kumo_rt vs {b:8} {g:+.3f} +- {se:.3f}")
        g, se = paired(m, "blend", "da_price")
        print(f"  blend   vs da_price {g:+.3f} +- {se:.3f}")
        sp = y >= 150
        if sp.any():
            print(f"  spike hours: q95 >= 150 on {(P['q95'][m][sp] >= 150).mean():.0%} of them; DA >= 150 on "
                  f"{(P['da_price'][m][sp] >= 150).mean():.0%}")
