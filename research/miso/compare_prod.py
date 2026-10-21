"""Compare a MISO Kumo walk-forward run with the production forecast (flex.daep.20251101) on 2025.

  uv run python research/miso/compare_prod.py <kumo_predictions.parquet> [data/miso/prod_flex_daep.parquet]

Production rows come from miso_tios_forecasts (pulled via research/pjm/sql.R). ts = dt + hr - 1 h
(MISO fixed EST). HOLD = days whose day number is not a multiple of 3 (as for PJM).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
k = pd.read_parquet(sys.argv[1])
k = k.assign(ts=pd.to_datetime(k["ts"])).set_index("ts")
prod = pd.read_parquet(sys.argv[2] if len(sys.argv) > 2 else ROOT / "data/miso/prod_flex_daep.parquet")
prod["ts"] = pd.to_datetime(prod["dt"]) + pd.to_timedelta(prod["hr"] - 1, unit="h")
P = pd.DataFrame({"kumo": k["q500"], "prod": prod.set_index("ts")["forecast"].reindex(k.index)})
y = k["y_true"]
ok = y.notna() & P.notna().all(axis=1)
day = P.index.floor("D")
dn = (day - pd.Timestamp("2025-01-01")).days
masks = {"ALL": ok, "SCREEN": ok & (dn % 3 == 0), "HOLD": ok & (dn % 3 != 0)}
print(f"hours scored {ok.sum()} (prod missing {P['prod'].isna().sum()})")


def mae(p, m):
    return (y - p)[m].abs().mean()


def rmse(p, m):
    return np.sqrt(((y - p)[m] ** 2).mean())


def paired(p, ref, m, n=2000):
    g = ((y - ref).abs() - (y - p).abs())[m].groupby(day[m]).mean()
    rng = np.random.default_rng(0)
    bs = [g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n)]
    return f"{g.mean():+.3f} +- {np.std(bs):.3f} (win {(g > 0).mean():.2f})"


for name, m in masks.items():
    print(f"{name:6} kumo MAE {mae(P['kumo'], m):.3f} RMSE {rmse(P['kumo'], m):.2f} | "
          f"prod MAE {mae(P['prod'], m):.3f} RMSE {rmse(P['prod'], m):.2f} | kumo gain vs prod "
          f"{paired(P['kumo'], P['prod'], m)}")
print(f"error corr {np.corrcoef((y - P['kumo'])[ok], (y - P['prod'])[ok])[0, 1]:.3f}; "
      f"bias (y - fc): kumo {(y - P['kumo'])[ok].mean():+.2f}, prod {(y - P['prod'])[ok].mean():+.2f}")

ws = np.round(np.arange(0, 1.01, 0.05), 2)
w = min(ws, key=lambda w: mae(w * P["prod"] + (1 - w) * P["kumo"], masks["SCREEN"]))
for wt in sorted({w, 0.5}):
    b = wt * P["prod"] + (1 - wt) * P["kumo"]
    print(f"blend w_prod={wt:.2f}{'*' if wt == w else ''}: HOLD MAE {mae(b, masks['HOLD']):.3f} "
          f"RMSE {rmse(b, masks['HOLD']):.2f} | vs kumo {paired(b, P['kumo'], masks['HOLD'])} | "
          f"vs prod {paired(b, P['prod'], masks['HOLD'])}")

season = pd.Series(np.select([P.index.month.isin([12, 1, 2]), P.index.month.isin([3, 4, 5]),
                              P.index.month.isin([6, 7, 8])], ["DJF", "MAM", "JJA"], "SON"), P.index)
hb = pd.Series(pd.cut(P.index.hour, [-1, 5, 9, 15, 20, 23],
                      labels=["00-05", "06-09", "10-15", "16-20", "21-23"]), P.index)
E = P.sub(y, axis=0).abs()[ok]
for nm, key in (("season", season), ("hour block", hb)):
    print(f"\nMAE by {nm}:\n{E.groupby(key[ok], observed=True).mean().round(3).to_string()}")
