"""Perfect-foresight RT datasets with same-hour RT reserve market results (leaks the future: a ceiling only).

  uv run python research/pjm_rt/make_reserves.py

Base v0_x5rt_25 (RT target, 5 congestion PCs + MISO / SPP DA forecasts; val = 2025). Features from
reserves_link.hourly() for the same hour (NaN before 2021-05; 30-min reserve from 2022-10):
  v0_x5rt_respf_mcp   prices: rmcp, sr_mad_mean, pr_mad_mean, sr_rto_mean, pr_rto_mean
  v0_x5rt_respf_mgn   physical margins only: tmr_rto_margin, pr_rto_margin, pr_mad_margin, sr_rto_margin, sr_mad_margin
Reserve MCPs are co-optimised with energy (and regulation's includes energy opportunity cost), so the price set is
close to the target itself; the margin set measures reserve slack.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from reserves_link import hourly  # noqa: E402

D = ROOT / "data/pjm-fe"
X = hourly()
SETS = {"v0_x5rt_respf_mcp": ["rmcp", "sr_mad_mean", "pr_mad_mean", "sr_rto_mean", "pr_rto_mean"],
        "v0_x5rt_respf_mgn": ["tmr_rto_margin", "pr_rto_margin", "pr_mad_margin", "sr_rto_margin", "sr_mad_margin"]}
for name, cols in SETS.items():
    (D / name).mkdir(exist_ok=True)
    for s in ("train", "val", "test"):
        df = pd.read_parquet(D / "v0_x5rt_25" / f"{s}.parquet")
        ts = pd.to_datetime(df["ts"])
        for c in cols:
            df[c] = X[c].reindex(ts).to_numpy()
        df.to_parquet(D / name / f"{s}.parquet", index=False)
        tail = df[ts >= "2023-11-01"]
        print(name, s, len(df), "nulls since 2023-11:", {c: int(tail[c].isna().sum()) for c in cols})
