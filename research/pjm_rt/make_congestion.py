"""RT datasets with the top principal components of PJM hourly nodal congestion.

  Rscript research/pjm_rt/collect_congestion.R fc|act      (from ~/dev/tios-quant-shiny; caches data/pjm_rt/cong/)
  uv run python research/pjm_rt/make_congestion.py [--k 5] [--base v0_rt|v0]   (k != 5 -> v0_congfc<k>_rt;
                                                    --base v0 = DA target -> v0_congfc<k>)

Each hour's nodal congestion field (one value per pnode) is reduced to K principal-component scores.
  v0_congpf_rt  v0_rt + PCs of the ACTUAL RT congestion (pjm_lmp_combined.rt_congestion) on day D. Perfect-foresight
                diagnostic (leaks the future); built only if the act cache exists (not pulled for the 2026-10-26 study).
  v0_congfc_rt  v0_rt + PCs of the trader forecast rt_congestion_forecast_pure_shadow_based (shadow-price forecasts x
                shift factors; pjm_hourly_congestion_forecasts, trade_dt = operating day; generated the morning before and never
                revised afterwards, per the trading desk, so usable at the D-1 11:00 EPT cutoff). The pure-shadow column is
                null on every Monday through 2025 (other forecast columns are filled; fixed by Oct 2026), so Monday scores
                are missing (pivot drops all-null hours: not in the PCA fit either).
PCA is fitted on 2023-10-01 .. 2024-12-31 only (never on 2025, the scoring year; never on prices), on nodes present in
>= 95% of fit hours; a missing node-hour counts as 0 congestion. Values are centred, not scaled, so nodes with large
congestion swings weigh more. Scores are filled from 2023-10-01 to 2025-12-31 and missing before (as the other RT
diagnostics). Writes loadings and explained variance to outputs/pjm-rt/congestion/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/spp_load"))
from make_features import ts_of  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, default=5)
p.add_argument("--base", default="v0_rt", choices=["v0_rt", "v0"], help="v0_rt = RT target (default); v0 = DA target")
a = p.parse_args()
DATA = ROOT / "data"
A, FIT_END, B = pd.Timestamp("2023-10-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
OUT = ROOT / "outputs/pjm-rt/congestion"
OUT.mkdir(parents=True, exist_ok=True)


def field(kind, col):
    files = sorted((DATA / "pjm_rt/cong" / kind).glob("*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=["dt", "hr", "official_pnode_id", col]) for f in files], ignore_index=True)
    d["ts"] = ts_of(d)
    W = d.pivot_table(index="ts", columns="official_pnode_id", values=col, aggfunc="mean").astype("float32")
    return W[(W.index >= A) & (W.index < B)]


for name, kind, col in (("v0_congpf_rt", "act", "rt"), ("v0_congfc_rt", "fc", "fc")):
    if a.k != 5:
        name = name.replace("_rt", f"{a.k}_rt")      # e.g. v0_congfc10_rt; the 5-component sets keep their names
    if a.base == "v0":
        name = name.removesuffix("_rt")               # DA target: v0_congfc10 etc.
    if not any((DATA / "pjm_rt/cong" / kind).glob("*.parquet")):
        print(f"{name}: no cached {kind} data, skipped")
        continue
    W = field(kind, col)
    fit = W[W.index < FIT_END]
    keep = fit.columns[fit.notna().mean() >= 0.95]
    X = fit[keep].fillna(0.0).to_numpy(np.float64)
    mu = X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    V = Vt[: a.k].T                                               # nodes x K loadings
    Z = (W.reindex(columns=keep).fillna(0.0).to_numpy(np.float64) - mu) @ V
    sc = pd.DataFrame(Z, index=W.index, columns=[f"cong_pc{i + 1}" for i in range(a.k)])
    print(f"{name}: hours {len(W)} (fit {len(fit)}), nodes {W.shape[1]} -> kept {len(keep)}; explained variance "
          + " ".join(f"{v:.3f}" for v in ev[: a.k]) + f" (top {a.k}: {ev[: a.k].sum():.3f})")
    pd.DataFrame(V, index=keep, columns=sc.columns).assign(mean=mu).to_parquet(OUT / f"{name}_loadings.parquet")
    pd.Series(ev[:20], name="explained").to_csv(OUT / f"{name}_explained.csv")
    # what each component means: mean loading by transmission zone (pjm_official_pnodes.zone), largest first
    zf = DATA / "pjm_rt/cong/pnode_zones.parquet"
    if zf.exists():
        zone = pd.read_parquet(zf).set_index("official_pnode_id")["zone"].reindex(keep).fillna("?").str.split(",").str[0]
        Lz = pd.DataFrame(V, index=keep, columns=sc.columns).groupby(zone.to_numpy()).mean()
        Lz = Lz[Lz.index.isin(zone.value_counts()[lambda c: c >= 10].index)]
        Lz.to_csv(OUT / f"{name}_zone_loadings.csv")
        for c in sc:
            o = Lz[c].sort_values()
            print(f"  {c}: low " + ", ".join(f"{k} {v:+.3f}" for k, v in o.head(4).items())
                  + " | high " + ", ".join(f"{k} {v:+.3f}" for k, v in o.tail(4)[::-1].items())
                  + f" | score sd {sc[c].std():.0f}")
    dst = DATA / "pjm-fe" / name
    dst.mkdir(parents=True, exist_ok=True)
    for s_ in ("train", "val", "test"):
        df = pd.read_parquet(DATA / "pjm-fe" / a.base / f"{s_}.parquet")
        ts = pd.to_datetime(df["ts"])
        win = ((ts >= A) & (ts < B)).to_numpy()
        for c in sc:
            df[c] = np.where(win, sc[c].reindex(ts).to_numpy(), np.nan)
        df.to_parquet(dst / f"{s_}.parquet", index=False)
        if s_ != "test":
            print(f"  {name}/{s_}: rows {len(df)}, window rows {win.sum()}, missing in window "
                  f"{int(np.isnan(df['cong_pc1'].to_numpy()[win]).sum())}")
