"""Error breakdown of the best Kumo runs + blend with the production forecast (flex.daep.20261001).

  uv run python research/pjm/analyze_blend.py

Reads full-daily 2025 predictions from outputs/pjm-fe/s3_stage/<run>/ (synced from the box),
the production forecast from data/pjm_extra/prod_flex_daep.parquet (pulled via sql.R) and stress
features from data/pjm-fe/x9_v8_lean_extra/val.parquet. SCREEN = every 3rd day of 2025 (used to pick
blend weights), HOLD = the other 243 days.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["ensC_daily", "ensB_daily", "e16_x9_daily", "ens3_daily", "baseline_v0_daily"]


def load(run):
    d = pd.read_parquet(ROOT / "outputs/pjm-fe/s3_stage" / run / "val_walk_forward_predictions.parquet")
    return d.assign(ts=pd.to_datetime(d["ts"])).set_index("ts")


preds = {r: load(r) for r in RUNS}
y = preds["ensC_daily"]["y_true"]
P = pd.DataFrame({r: d["q500"] for r, d in preds.items()})
prod = pd.read_parquet(ROOT / "data/pjm_extra/prod_flex_daep.parquet")
prod["ts"] = pd.to_datetime(prod["dt"]) + pd.to_timedelta(prod["hr"] - 1, unit="h")
P["prod"] = prod.set_index("ts")["forecast"].reindex(P.index)
feat = pd.read_parquet(ROOT / "data/pjm-fe/x9_v8_lean_extra/val.parquet")
feat = feat.assign(ts=pd.to_datetime(feat["ts"])).set_index("ts").reindex(P.index)

ok = y.notna() & P.notna().all(axis=1)
day = P.index.floor("D")
dn = (day - pd.Timestamp("2025-01-01")).days
screen, hold = ok & (dn % 3 == 0), ok & (dn % 3 != 0)
print(f"rows ok {ok.sum()}, missing prod {P['prod'].isna().sum()}")


def mae(p, m):
    return (y - p)[m].abs().mean()


def rmse(p, m):
    return np.sqrt(((y - p)[m] ** 2).mean())


def paired(p, ref, m, n=2000):
    g = ((y - ref).abs() - (y - p).abs())[m].groupby(day[m]).mean()
    rng = np.random.default_rng(0)
    bs = [g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n)]
    return g.mean(), np.std(bs), (g > 0).mean()


# ---------------------------------------------------------------- #4 error breakdown
print("\n== Error breakdown (full 2025, MAE of q500) ==")
season = pd.Series(np.select([P.index.month.isin([12, 1, 2]), P.index.month.isin([3, 4, 5]),
                              P.index.month.isin([6, 7, 8])], ["DJF", "MAM", "JJA"], "SON"), P.index)
hb = pd.cut(P.index.hour, [-1, 5, 9, 15, 20, 23], labels=["00-05", "06-09", "10-15", "16-20", "21-23"])
stress = pd.qcut(feat["supply_cushion_ratio"].rank(method="first"), [0, .1, .5, .9, 1],
                 labels=["tight10%", "10-50", "50-90", "loose10%"])
E = (P.sub(y, axis=0)).abs()[ok]
for name, key in (("season", season), ("hour block", pd.Series(hb, P.index)), ("cushion", stress)):
    t = E.groupby(key[ok], observed=True)[["ensC_daily", "e16_x9_daily", "baseline_v0_daily", "prod"]].mean()
    t["share_abs_err_ensC"] = E.groupby(key[ok], observed=True)["ensC_daily"].sum() / E["ensC_daily"].sum()
    t["hours"] = E.groupby(key[ok], observed=True).size()
    print(f"\n-- by {name}\n{t.round(3).to_string()}")

err = (y - P["ensC_daily"])[ok]
dm = err.abs().groupby(day[ok]).mean().sort_values(ascending=False)
sse = (err**2).groupby(day[ok]).sum()
print(f"\nTop days by ensC daily MAE (share of total abs err / of SSE):")
for k in (3, 10, 30):
    top = dm.index[:k]
    print(f"  top {k:2d} days: {err.abs()[day[ok].isin(top)].sum() / err.abs().sum():.1%} of abs err, "
          f"{sse[top].sum() / sse.sum():.1%} of SSE")
tbl = pd.DataFrame({"ensC_MAE": dm.head(12),
                    "bias(y-ensC)": err.groupby(day[ok]).mean().reindex(dm.index[:12]),
                    "prod_MAE": (y - P["prod"]).abs()[ok].groupby(day[ok]).mean().reindex(dm.index[:12]),
                    "y_max": y[ok].groupby(day[ok]).max().reindex(dm.index[:12]),
                    "cushion_min": feat["supply_cushion_ratio"][ok].groupby(day[ok]).min().reindex(dm.index[:12])})
print(tbl.round(2).to_string())
print(f"Sign of daily bias on top-30 days: under-forecast {(err.groupby(day[ok]).mean()[dm.index[:30]] > 0).mean():.0%}")
print(f"Overall bias (y - q500): ensC {err.mean():+.2f}, prod {(y - P['prod'])[ok].mean():+.2f}")

# ---------------------------------------------------------------- #1 blend with production
print("\n== Production forecast vs Kumo (HOLD = 243 days) ==")
for c in P:
    print(f"  {c:18} HOLD MAE {mae(P[c], hold):.3f} RMSE {rmse(P[c], hold):.2f} | "
          f"ALL {mae(P[c], ok):.3f} / {rmse(P[c], ok):.2f}")
print("  error corr with prod:", {c: round(np.corrcoef((y - P[c])[ok], (y - P["prod"])[ok])[0, 1], 3)
                                 for c in RUNS})

ws = np.round(np.arange(0, 1.01, 0.05), 2)
print("\nBlend w*prod + (1-w)*kumo (w picked on SCREEN by MAE), HOLD vs kumo alone:")
for c in RUNS:
    w = min(ws, key=lambda w: mae(w * P["prod"] + (1 - w) * P[c], screen))
    for wt, tag in ((w, "picked"), (0.5, "50/50")):
        b = wt * P["prod"] + (1 - wt) * P[c]
        g, se, win = paired(b, P[c], hold)
        print(f"  {c:18} w={wt:.2f} ({tag}): HOLD MAE {mae(b, hold):.3f} RMSE {rmse(b, hold):.2f} | "
              f"gain {g:+.3f} +- {se:.3f} win {win:.2f}")
    curve = " ".join(f"{w:.1f}:{mae(w * P['prod'] + (1 - w) * P[c], hold):.3f}" for w in ws[::4])
    print(f"      HOLD MAE curve over w: {curve}")
