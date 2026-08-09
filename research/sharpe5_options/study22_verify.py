#!/usr/bin/env python3
"""Study 22: does the sweep's winner survive a dev/holdout split?

36 configurations were swept and winners picked, which is precisely the
trial-count trap this project has repeatedly caught. Before recommending a
parameter change to a live strategy, split it: fit nothing, just measure the
same configs on 2019-2024 and 2025-2026 separately. A real structural effect
holds in both; a swept artifact does not.
"""
import math
import numpy as np, pandas as pd
import study20_optimize as S20

DEV_END = pd.Timestamp("2024-12-31")

def split_stats(d, rung_eq=0.03, cap=0.60):
    out = {}
    for lab, seg in (("dev", d[d.date <= DEV_END]), ("hold", d[d.date > DEV_END]),
                     ("full", d)):
        r = S20.ladder(seg, rung_eq=rung_eq, cap=cap, cadence="W")
        out[lab] = r
    return out

def main():
    panel = S20.load_spy()
    configs = [
        ("DEPLOYED 83d/3%/3%", 0.03, 0.03, 60, 110),
        ("60d/5%/5%",          0.05, 0.05, 45, 75),
        ("60d/5%/3%",          0.05, 0.03, 45, 75),
        ("30d/5%/8%",          0.05, 0.08, 15, 45),
        ("60d/3%/8%",          0.03, 0.08, 45, 75),
        ("83d/5%/5%",          0.05, 0.05, 60, 110),
    ]
    print("=" * 88)
    print("STUDY 22 — dev/holdout verification of the swept winners")
    print("=" * 88)
    print(f"{'config':<22}{'dev Sh':>8}{'hold Sh':>9}{'full Sh':>9}"
          f"{'dev CAGR':>10}{'hold CAGR':>11}{'full DD':>9}")
    for name, otm, wd, lo, hi in configs:
        d = S20.build(otm, wd, lo, hi, panel=panel)
        if d is None or len(d) < 60:
            print(f"{name:<22} insufficient"); continue
        s = split_stats(d)
        if not all(s.values()):
            print(f"{name:<22} insufficient split"); continue
        print(f"{name:<22}{s['dev']['sharpe']:>8.2f}{s['hold']['sharpe']:>9.2f}"
              f"{s['full']['sharpe']:>9.2f}{s['dev']['cagr']:>10.1%}"
              f"{s['hold']['cagr']:>11.1%}{s['full']['dd']:>9.1%}")
    print("\nA config whose holdout Sharpe collapses relative to dev was fitted,")
    print("not discovered. The deployed config is the control.")

if __name__ == "__main__":
    main()
