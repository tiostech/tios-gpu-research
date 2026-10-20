import numpy as np, pandas as pd
O = "/scratch/outputs/pjm-fe/"
def q(name, mode="walk_forward"):
    d = pd.read_parquet(f"{O}{name}/val_{mode}_predictions.parquet")
    d["ts"] = pd.to_datetime(d["ts"]); return d.set_index("ts")
mem = {s: q(f"m_x9_f60_s{s}", "walk_forward_every3") for s in range(1, 9)}
idx = mem[1].index
y = mem[1]["y_true"]; ok = y.notna(); day = idx.floor("D")
def sc(p, m=ok):
    e = (y - p)[m]; return f"MAE {e.abs().mean():.3f} RMSE {np.sqrt((e**2).mean()):.2f}"
def daygain(p, ref):
    g = ((y - ref).abs() - (y - p).abs())[ok].groupby(day[ok]).mean()
    bs = [g.sample(len(g), replace=True, random_state=i).mean() for i in range(1000)]
    return f"gain {g.mean():+.3f} +- {np.std(bs):.3f} win {(g > 0).mean():.2f}"
x9 = q("e16_x9", "walk_forward_every3")["q500"].reindex(idx)
daily = {n: q(n)["q500"].reindex(idx) for n in
         ("v0_ctx20k_daily", "c1_ctx20k_daily", "e16_x9_daily", "e16_v8_knn40k_daily")}
ensC = pd.concat(daily.values(), axis=1).mean(axis=1)
for s, d in mem.items(): print(f"member s{s}: {sc(d['q500'])}")
M = pd.concat([d["q500"] for d in mem.values()], axis=1)
for k in (2, 4, 8): print(f"mtry mean of {k}: {sc(M.iloc[:, :k].mean(axis=1))}")
mtry = M.mean(axis=1)
print(f"e16_x9 (every3): {sc(x9)}")
print(f"ensC (screen days): {sc(ensC)}")
print("mtry8 vs e16_x9:", daygain(mtry, x9))
print("mtry8 vs ensC:", daygain(mtry, ensC))
C_swap = pd.concat([daily["v0_ctx20k_daily"], daily["c1_ctx20k_daily"], mtry, daily["e16_v8_knn40k_daily"]], axis=1).mean(axis=1)
C_add = pd.concat(list(daily.values()) + [mtry], axis=1).mean(axis=1)
print(f"ensC with x9 -> mtry8: {sc(C_swap)} |", daygain(C_swap, ensC))
print(f"ensC + mtry8 (5 members): {sc(C_add)} |", daygain(C_add, ensC))
print("member q500 corr w/ e16_x9 err:", round(np.corrcoef((y - mtry)[ok], (y - x9)[ok])[0, 1], 3),
      "| mean pairwise member err corr:",
      round(np.nanmean(np.corrcoef((y.values[:, None] - M.values)[ok.values].T)[np.triu_indices(8, 1)]), 3))
