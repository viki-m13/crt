"""Sweep candidate Floor signals on the cached scored frame and rank them
by realized downside.  Read-only over floor_scored.parquet -> fast.

We want the signal that minimizes how often a buy sits below its purchase
price (uw_frac / ever_below) WITHOUT collapsing forward return.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from floor_lib import build, HORIZONS

K = 10


def z(s):
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def signals(g):
    """name -> picked index (top-K safest) for one asof frame."""
    out = {}
    out["lowvol"] = g.nsmallest(K, "vol_3m_xs").index
    out["highmom"] = g.nlargest(K, "mom_12_1_xs").index
    out["quality5y"] = g.nlargest(K, "trend_health_5y_xs").index
    out["chr_exp_uw"] = g.nsmallest(K, "chr_exp_uw_frac_3m").index
    out["chr_p_below"] = g.nsmallest(K, "chr_p_below_end_3m").index
    out["chr_shallow"] = g.nlargest(K, "chr_trough_q30_3m").index   # least-negative trough
    out["chr_drift_lv"] = (z(g["chr_p50_end_3m"]) - z(g["vol_3m_xs"])).nlargest(K).index
    out["gbm_uw3"] = g.nsmallest(K, "gbm_uw_frac_3m").index
    out["gbm_uw1"] = g.nsmallest(K, "gbm_uw_frac_1m").index
    out["gbm_ever3"] = g.nsmallest(K, "gbm_ever_below_3m").index
    out["gbm_maxdd3"] = g.nlargest(K, "gbm_maxdd_3m").index         # least-negative dd
    gbm_blend = (g["gbm_uw_frac_3m"].rank() + g["gbm_uw_frac_1m"].rank()
                 + g["gbm_ever_below_3m"].rank() + (-g["gbm_maxdd_3m"]).rank())
    out["gbm_blend"] = gbm_blend.nsmallest(K).index
    # combos of HF-forecast + learned
    out["floor_blend"] = (g["gbm_uw_frac_3m"].rank()
                          + g["chr_exp_uw_frac_3m"].rank()).nsmallest(K).index
    comp = (-z(g["gbm_uw_frac_3m"]) - z(g["chr_exp_uw_frac_3m"])
            - z(g["vol_3m_xs"]) + z(g["chr_p50_end_3m"]))
    out["floor_comp"] = comp.nlargest(K).index
    # gbm-safe but require Chronos to forecast non-negative drift
    up = g[g["chr_p50_end_3m"] > 0]
    if len(up) >= K:
        out["floor_up_gbm"] = up.nsmallest(K, "gbm_uw_frac_3m").index
        out["floor_up_blend"] = (up["gbm_uw_frac_3m"].rank()
                                 + up["gbm_ever_below_3m"].rank()).nsmallest(K).index
    # --- refined transparent composites ---
    # FloorScore: learned drawdown + learned underwater + low vol + durable
    # uptrend + Chronos downside cushion (higher = safer)
    out["floor_final"] = (z(g["gbm_maxdd_3m"]) - z(g["gbm_uw_frac_3m"])
                          - z(g["vol_3m_xs"]) + 0.5 * z(g["trend_health_5y_xs"])
                          + 0.5 * z(g["chr_trough_q30_3m"])).nlargest(K).index
    # same WITHOUT any Chronos term, to isolate the HF model's marginal value
    out["floor_noChr"] = (z(g["gbm_maxdd_3m"]) - z(g["gbm_uw_frac_3m"])
                          - z(g["vol_3m_xs"]) + 0.5 * z(g["trend_health_5y_xs"])
                          ).nlargest(K).index
    # pure rank-mix of the robust ingredients
    out["floor_rankmix"] = ((-g["gbm_maxdd_3m"]).rank() + g["gbm_uw_frac_3m"].rank()
                            + g["vol_3m_xs"].rank()
                            + (-g["trend_health_5y_xs"]).rank()).nsmallest(K).index
    return out


def stats(rows, h):
    m = rows[~rows[f"censored_{h}"]]
    if len(m) == 0:
        return None
    return dict(
        n=len(m),
        uw=m[f"uw_frac_{h}"].mean(),
        ever=m[f"ever_below_{h}"].mean(),
        endb=m[f"end_below_{h}"].mean(),
        dd=m[f"maxdd_{h}"].mean(),
        ret=m[f"end_ret_{h}"].mean(),
        medret=m[f"end_ret_{h}"].median(),
        safe=(m[f"uw_frac_{h}"] < 0.15).mean(),
    )


def main():
    df = build()
    names = ["universe_avg", "lowvol", "highmom", "quality5y", "chr_exp_uw",
             "chr_p_below", "chr_shallow", "chr_drift_lv", "gbm_uw3", "gbm_uw1",
             "gbm_ever3", "gbm_maxdd3", "gbm_blend", "floor_blend", "floor_comp",
             "floor_up_gbm", "floor_up_blend", "floor_final", "floor_noChr",
             "floor_rankmix"]
    bucket = {n: [] for n in names}
    for t, g in df.groupby("asof"):
        if len(g) < K * 2:
            continue
        bucket["universe_avg"].append(g)
        for n, idx in signals(g).items():
            bucket[n].append(g.loc[idx])

    for h in ("1m", "3m", "12m"):
        print(f"\n=== {h} horizon (K={K}) — sorted by uw_frac (lower=safer) ===")
        print(f"{'signal':<15}{'n':>6}{'uw':>7}{'ever<':>7}{'end<':>7}"
              f"{'maxdd':>8}{'meanret':>8}{'medret':>8}{'safe%':>7}")
        rowstats = []
        for n in names:
            if not bucket[n]:
                continue
            s = stats(pd.concat(bucket[n]), h)
            if s:
                rowstats.append((n, s))
        for n, s in sorted(rowstats, key=lambda x: x[1]["uw"]):
            print(f"{n:<15}{s['n']:>6}{s['uw']:>7.3f}{s['ever']:>7.3f}{s['endb']:>7.3f}"
                  f"{s['dd']:>8.3f}{s['ret']:>8.3f}{s['medret']:>8.3f}{s['safe']*100:>6.1f}%")


if __name__ == "__main__":
    main()
