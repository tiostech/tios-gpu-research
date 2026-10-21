"""Compare OKGE load walk-forward runs with the Meteologica OKGE forecast on 2025.

  uv run python research/spp_load/compare.py <run_dir> [<run_dir> ...]

Meteologica = data/spp_load/meteologica_okge.parquet (single D-1 ~09:20 vintage). HOLD = days whose
day number is not a multiple of 3 (as in the price work); blend weights are picked on SCREEN days.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
met = pd.read_parquet(ROOT / "data/spp_load/meteologica_okge.parquet").set_index("ts")["meteologica"]
runs = {}
for d in map(Path, sys.argv[1:]):
    p = pd.read_parquet(d / "val_walk_forward_predictions.parquet")
    p = p.assign(ts=pd.to_datetime(p["ts"])).set_index("ts")
    runs[d.name] = p
first = next(iter(runs.values()))
y = first["y_true"]
P = pd.DataFrame({k: v["q500"] for k, v in runs.items()})
P["meteologica"] = met.reindex(P.index)
ok = y.notna() & P.notna().all(axis=1)
day = P.index.floor("D")
dn = (day - pd.Timestamp("2025-01-01")).days
masks = {"ALL": ok, "SCREEN": ok & (dn % 3 == 0), "HOLD": ok & (dn % 3 != 0)}


def stats(p, m):
    e = (y - p)[m]
    return f"MAE {e.abs().mean():6.1f} RMSE {np.sqrt((e**2).mean()):6.1f} MAPE {(e.abs() / y[m]).mean() * 100:4.2f}% bias {e.mean():+6.1f}"


def paired(p, ref, m, n=2000):
    g = ((y - ref).abs() - (y - p).abs())[m].groupby(day[m]).mean()
    rng = np.random.default_rng(0)
    bs = [g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n)]
    return f"{g.mean():+6.1f} +- {np.std(bs):4.1f} MW (win {(g > 0).mean():.2f})"


print(f"hours scored {ok.sum()}")
for name, m in masks.items():
    print(f"\n-- {name}")
    for c in P:
        extra = "" if c == "meteologica" else f" | gain vs Meteologica {paired(P[c], P['meteologica'], m)}"
        print(f"  {c:16} {stats(P[c], m)}{extra}")

print("\n-- blends with Meteologica (weight picked on SCREEN), HOLD")
ws = np.round(np.arange(0, 1.01, 0.05), 2)
for c in runs:
    w = min(ws, key=lambda w: (y - (w * P["meteologica"] + (1 - w) * P[c]))[masks["SCREEN"]].abs().mean())
    b = w * P["meteologica"] + (1 - w) * P[c]
    print(f"  {c:16} w_met={w:.2f}: {stats(b, masks['HOLD'])} | vs Meteologica {paired(b, P['meteologica'], masks['HOLD'])}"
          f" | vs Kumo {paired(b, P[c], masks['HOLD'])}")

season = pd.Series(np.select([P.index.month.isin([12, 1, 2]), P.index.month.isin([3, 4, 5]),
                              P.index.month.isin([6, 7, 8])], ["DJF", "MAM", "JJA"], "SON"), P.index)
hb = pd.Series(pd.cut(P.index.hour, [-1, 5, 9, 15, 20, 23],
                      labels=["00-05", "06-09", "10-15", "16-20", "21-23"]), P.index)
E = P.sub(y, axis=0).abs()[ok]
for nm, key in (("season", season), ("hour block", hb)):
    print(f"\nMAE by {nm}:\n{E.groupby(key[ok], observed=True).mean().round(1).to_string()}")
