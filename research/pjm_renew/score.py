"""Score Kumo PJM wind / solar walk-forwards against Meteologica and ISO (same hours).

  uv run python research/pjm_renew/score.py <run_dir> <wind|solar> [--year 2025]

MAE / RMSE / nMAE (MAE / mean actual) / bias (actual minus forecast) of q500, Meteologica ALL and the ISO
forecast, on hours where all three and the actual exist; solar on daylight hours only (sun above the horizon at
any of the four centroids in make_features.CENTROIDS, a fixed rule). Paired day-block bootstrap of the daily
MAE gain over Meteologica and ISO. HOLD = days of 2025 whose day number is not a multiple of 3.
Quantiles of solar runs are clipped at 0 before scoring (rule fixed in the plan).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util  # noqa: E402

# research/pjm_renew/make_features.py imports research/spp_load/make_features.py, so load it under its own name
_spec = importlib.util.spec_from_file_location("renew_features", Path(__file__).resolve().parent / "make_features.py")
_mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mf)
CENTROIDS, sun = _mf.CENTROIDS, _mf.sun

ROOT = Path(__file__).resolve().parents[2]
Y0 = pd.Timestamp("2025-01-01")


def load_run(run):
    run = Path(run)
    f = next(run.glob("val_walk_forward*_predictions.parquet"))
    d = pd.read_parquet(f)
    return d.assign(ts=pd.to_datetime(d["ts"])).set_index("ts")


def daylight(idx):
    return pd.Series(np.column_stack([sun(idx, la, lo)[0] > 0 for la, lo in CENTROIDS.values()]).any(axis=1), idx)


def frame(run, target, daylight_only=True):
    """Hourly frame y / kumo (q500) / met / iso (+ quantiles) on scoreable hours."""
    k = load_run(run)
    b = pd.read_parquet(ROOT / "data/pjm_renew/bench.parquet").assign(ts=lambda d: pd.to_datetime(d["ts"])).set_index("ts")
    qs = [c for c in k.columns if c.startswith("q")]
    if target == "solar":
        k[qs] = k[qs].clip(lower=0)
    F = pd.DataFrame({"y": b[target].reindex(k.index), "kumo": k["q500"], "met": b[f"met_{target}"].reindex(k.index),
                      "iso": b[f"iso_{target}"].reindex(k.index)})
    F = F.join(k[qs].drop(columns="q500", errors="ignore"))
    ok = F[["y", "kumo", "met", "iso"]].notna().all(axis=1)
    if target == "solar" and daylight_only:
        ok &= daylight(F.index)
    return F[ok]


def stats(y, p):
    e = y - p
    return {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "nmae": float(e.abs().mean() / y.mean()), "bias": float(e.mean())}


def paired(F, new, ref, n=2000):
    day = F.index.floor("D")
    g = ((F["y"] - F[ref]).abs() - (F["y"] - F[new]).abs()).groupby(day).mean()
    rng = np.random.default_rng(0)
    bs = [g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n)]
    return {"gain": float(g.mean()), "se": float(np.std(bs)), "win": float((g > 0).mean()), "days": int(len(g))}


def scorecard(F):
    out = {c: stats(F["y"], F[c]) for c in ("met", "iso", "kumo")}
    out["kumo"] |= {"vs_met": paired(F, "kumo", "met"), "vs_iso": paired(F, "kumo", "iso")}
    out["hours"] = int(len(F))
    return out


def subsets(F):
    dn = (F.index.floor("D") - Y0).days
    y25 = (F.index >= Y0) & (F.index < Y0 + pd.DateOffset(years=1))
    return {"all": F[y25], "hold": F[y25 & (dn % 3 != 0)], "sel": F[y25 & (dn % 3 == 0)], "pre2025": F[F.index < Y0]}


def show(name, sc):
    m, i, k = sc["met"], sc["iso"], sc["kumo"]
    print(f"{name:8} h {sc['hours']:5} | MAE met {m['mae']:6.1f} iso {i['mae']:6.1f} kumo {k['mae']:6.1f} "
          f"({100 * (1 - k['mae'] / m['mae']):+.1f}% vs met) | RMSE {m['rmse']:.0f}/{i['rmse']:.0f}/{k['rmse']:.0f} | "
          f"nMAE {m['nmae']:.3f}/{i['nmae']:.3f}/{k['nmae']:.3f} | bias {m['bias']:+.0f}/{i['bias']:+.0f}/{k['bias']:+.0f} | "
          f"gain vs met {k['vs_met']['gain']:+.1f}+-{k['vs_met']['se']:.1f} win {k['vs_met']['win']:.2f}, "
          f"vs iso {k['vs_iso']['gain']:+.1f}+-{k['vs_iso']['se']:.1f} win {k['vs_iso']['win']:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("target", choices=["wind", "solar"])
    a = p.parse_args()
    F = frame(a.run, a.target)
    for nm, sub in subsets(F).items():
        if len(sub):
            show(nm, scorecard(sub))
    if "q100" in F and "q900" in F:
        print(f"80% band coverage: {((F['y'] >= F['q100']) & (F['y'] <= F['q900'])).mean():.3f}")
