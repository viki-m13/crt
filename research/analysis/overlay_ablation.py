"""
Overlay ablation: what does the 200ma_loose regime + vt18 cost across signals?

For each of: mom_12_1, mom_6_1, low_vol, mom_x_lowvol, 4T9wE blend
run the full 2x2 grid (regime on/off) x (vol_target on/off) and report
delta-Sharpe + delta-CAGR.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "research/analysis")

import pandas as pd
from diagnostics import (
    load_panel, load_daily, load_membership, run_with_score_fn,
    score_col, score_mom_x_lowvol, score_4t9we_blend,
    OOS_START, OOS_END,
)

print("Loading data ...")
panel = load_panel(); daily = load_daily(); mem = load_membership()
print("4T9wE blend cache build ...")
blend = score_4t9we_blend(panel, OOS_START, OOS_END)

signals = [
    ("mom_12_1",      score_col("mom_12_1")),
    ("mom_6_1",       score_col("mom_6_1")),
    ("low_vol",       score_col("vol_1y", sign=-1)),
    ("mom_x_lowvol",  score_mom_x_lowvol()),
    ("4T9wE_blend",   blend),
]
grid = [
    ("none",       False, False),
    ("regime",     True,  False),
    ("vt18",       False, True),
    ("regime_vt",  True,  True),
]

print(f"\n{'signal':<16} {'overlays':<11}  {'CAGR':>7}  {'Sharpe':>6}  {'MaxDD':>6}  {'AnnVol':>6}  {'N':>4}")
results = {}
for sname, sfn in signals:
    for gname, ureg, uvt in grid:
        df, m = run_with_score_fn(panel, daily, mem, OOS_START, OOS_END,
                                  sfn, top_k=30, use_regime=ureg, use_vol_target=uvt,
                                  weighting="invvol_cap5", label=f"{sname}_{gname}")
        if m:
            results[f"{sname}__{gname}"] = m
            print(f"{sname:<16} {gname:<11}  {m['cagr']:>6.1%}  {m['sharpe']:>6.2f}  "
                  f"{m['max_dd']:>5.1%}  {m['ann_vol']:>5.1%}  {m['n_months']:>4}")

json.dump(results, open("research/analysis/summary_overlay_ablation.json","w"),
          indent=2, default=str)

print("\nDelta from 'none' (the no-overlay baseline) per signal:")
for sname, _ in signals:
    base = results.get(f"{sname}__none", {})
    if not base: continue
    print(f"\n  {sname}:")
    for gname, _, _ in grid:
        m = results.get(f"{sname}__{gname}", {})
        if not m: continue
        dc = m["cagr"] - base["cagr"]
        ds = m["sharpe"] - base["sharpe"]
        print(f"    {gname:<11}  d_CAGR={dc:+.1%}  d_Sharpe={ds:+.2f}")
