#!/usr/bin/env python3
"""Study 21: joint optimization of the deployed ladder, honest fills.

Study 20 swept one lever at a time and found tenor dominant (30d 0.98 vs 83d
0.82), width mildly helpful (5% > 3%) and OTM mildly helpful (2% > 3%). Levers
interact — shorter tenor means more rungs, which is where the ladder's
diversification comes from, and that changes the optimal width. So sweep them
together, and annualize on the ACTUAL observation count of each configuration
rather than assuming a weekly calendar (the error that inflated study 19).
"""
import math, os, itertools
import numpy as np, pandas as pd
import study20_optimize as S20

def main():
    panel = S20.load_spy()
    rows = []
    tenors = [(15,45,"30d"),(45,75,"60d"),(60,110,"83d")]
    for (lo,hi,tlab), otm, wd in itertools.product(tenors,(0.02,0.03,0.05),(0.03,0.05,0.08,0.12)):
        d = S20.build(otm, wd, lo, hi, panel=panel)
        r = S20.ladder(d, rung_eq=0.03, cap=0.60, cadence="W")
        if r:
            rows.append((tlab,otm,wd,r['cagr'],r['sharpe'],r['dd'],r['n'],
                         (d.nat/d.w).mean()))
    df = pd.DataFrame(rows, columns=["tenor","otm","width","cagr","sharpe","dd","n","cred_w"])
    df = df.sort_values("sharpe", ascending=False)
    print("=== JOINT SWEEP (worst-side fills, weekly rungs, 3%/rung, 60% cap) ===")
    print(f"{'tenor':>6}{'otm':>6}{'width':>7}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}{'rungs':>7}{'cred/w':>8}")
    for r in df.itertuples(index=False):
        flag = "  <== deployed" if (r.tenor=="83d" and abs(r.otm-0.03)<1e-9 and abs(r.width-0.03)<1e-9) else ""
        print(f"{r.tenor:>6}{r.otm:>6.0%}{r.width:>7.0%}{r.cagr:>9.1%}{r.sharpe:>8.2f}"
              f"{r.dd:>9.1%}{r.n:>7}{r.cred_w:>8.4f}{flag}")
    df.to_csv("results_joint_sweep.csv", index=False)
    best = df.iloc[0]
    print(f"\nBEST: {best.tenor} / {best.otm:.0%} OTM / {best.width:.0%} wide -> "
          f"Sharpe {best.sharpe:.2f}, CAGR {best.cagr:.1%}, maxDD {best.dd:.1%}")
    dep = df[(df.tenor=="83d")&(np.isclose(df.otm,0.03))&(np.isclose(df.width,0.03))]
    if len(dep):
        b=dep.iloc[0]
        print(f"DEPLOYED: Sharpe {b.sharpe:.2f}, CAGR {b.cagr:.1%}, maxDD {b.dd:.1%}")
        print(f"IMPROVEMENT: {best.sharpe-b.sharpe:+.2f} Sharpe, {best.cagr-b.cagr:+.1%} CAGR, "
              f"{best.dd-b.dd:+.1%} drawdown")

if __name__ == "__main__":
    main()
