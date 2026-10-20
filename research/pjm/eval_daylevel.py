"""Score the day-level models and re-level hourly forecasts with them.

  uv run python research/pjm/eval_daylevel.py <day_mean_run_dir> <day_peak_run_dir> [hourly runs...]

Hourly runs default to ensC_daily and e16_x9_daily under outputs/pjm-fe/s3_stage/. For each, the
hourly median is shifted so its day level moves toward the day model (lam picked on SCREEN days):
  A: all hours shifted by lam * (day-mean model - mean of hourly q500)
  B: hours 16-20 shifted by lam * (peak model - their mean hourly q500); other hours unchanged
  C: B for 16-20 plus the other 19 hours shifted so the day mean matches A
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
dmean_dir, dpeak_dir = Path(sys.argv[1]), Path(sys.argv[2])
hourly = sys.argv[3:] or ["ensC_daily", "e16_x9_daily"]


def load(path):
    d = pd.read_parquet(path)
    return d.assign(ts=pd.to_datetime(d["ts"])).set_index("ts")


Dm = load(dmean_dir / "val_walk_forward_predictions.parquet")
Dp = load(dpeak_dir / "val_walk_forward_predictions.parquet")


def paired(p, ref, y, m, day, n=2000):
    g = ((y - ref).abs() - (y - p).abs())[m].groupby(day[m]).mean()
    rng = np.random.default_rng(0)
    bs = [g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n)]
    return f"{g.mean():+.3f} +- {np.std(bs):.3f} win {(g > 0).mean():.2f}"


for run in hourly:
    H = load(ROOT / "outputs/pjm-fe/s3_stage" / run / "val_walk_forward_predictions.parquet")
    y, q = H["y_true"], H["q500"]
    day = H.index.floor("D")
    pk = H.index.hour.isin(range(16, 21))
    ok = y.notna()
    dn = (day - pd.Timestamp("2025-01-01")).days
    screen, hold = ok & (dn % 3 == 0), ok & (dn % 3 != 0)
    dscreen = (Dm.index - pd.Timestamp("2025-01-01")).days % 3 == 0

    # Day-level accuracy: day model vs the hourly model's implied level.
    hm = q.groupby(day).mean()
    hp = q[pk].groupby(day[pk]).mean()
    ym, yp = y.groupby(day).mean(), y[pk].groupby(day[pk]).mean()
    print(f"\n=== {run}")
    for tag, mdl, imp, act in (("day mean", Dm["q500"], hm, ym), ("peak 16-20", Dp["q500"], hp, yp)):
        h = ~dscreen
        print(f"  {tag:10} MAE  day model {(act - mdl).abs()[h].mean():.3f} | hourly-implied "
              f"{(act - imp).abs()[h].mean():.3f} | 50/50 {(act - (mdl + imp) / 2).abs()[h].mean():.3f}  (HOLD days)"
              f"  err corr {np.corrcoef((act - mdl), (act - imp))[0, 1]:.2f}")

    dA = (Dm["q500"] - hm).reindex(day).to_numpy()
    dB = (Dp["q500"] - hp).reindex(day).to_numpy()
    n_pk = pd.Series(pk, H.index).groupby(day).transform("sum").to_numpy()
    n_all = pd.Series(1, H.index).groupby(day).transform("sum").to_numpy()
    dC_off = (dA * n_all - dB * n_pk) / (n_all - n_pk)
    shifts = {"A": dA, "B": np.where(pk, dB, 0.0), "C": np.where(pk, dB, dC_off)}
    lams = np.round(np.arange(0, 1.01, 0.1), 1)
    base = f"MAE {(y - q)[hold].abs().mean():.3f} RMSE {np.sqrt(((y - q)[hold] ** 2).mean()):.2f}"
    print(f"  base HOLD {base}")
    for k, s in shifts.items():
        lam = min(lams, key=lambda l: (y - (q + l * s))[screen].abs().mean())
        for l in sorted({lam, 0.5, 1.0}):
            p = q + l * s
            e = (y - p)[hold]
            print(f"  {k} lam={l:.1f}{'*' if l == lam else ' '} HOLD MAE {e.abs().mean():.3f} "
                  f"RMSE {np.sqrt((e ** 2).mean()):.2f} | gain vs base {paired(p, q, y, hold, day)}")
