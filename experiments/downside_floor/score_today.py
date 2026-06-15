"""Print the current FloorScore ranking — the "safest buys" (lowest chance
of sitting below your purchase price) for the most recent asof available in
the cached PIT panel.

FloorScore = z(gbm_maxdd_3m) - z(gbm_uw_frac_3m) - z(vol_3m_xs)
             + 0.5 z(trend_health_5y_xs) + 0.5 z(chr_trough_q30_3m)
(higher = safer; see backtest_floor.py for the validated definition).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from floor_lib import build, HERE

TOPN = 20


def z(s):
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def main():
    df = build()
    asof = df["asof"].max()
    g = df[df["asof"] == asof].copy()
    g = g[g["gbm_maxdd_3m"].notna()]
    g["FloorScore"] = (z(g["gbm_maxdd_3m"]) - z(g["gbm_uw_frac_3m"]) - z(g["vol_3m_xs"])
                       + 0.5 * z(g["trend_health_5y_xs"]) + 0.5 * z(g["chr_trough_q30_3m"]))
    g = g.sort_values("FloorScore", ascending=False)
    cols = {
        "ticker": "ticker",
        "FloorScore": "FloorScore",
        "gbm_uw_frac_3m": "pred_uw_frac_3m",      # forecast fraction of days below buy
        "gbm_maxdd_3m": "pred_maxdd_3m",          # forecast worst dip from buy
        "chr_trough_q30_3m": "chr_trough_q30_3m",  # Chronos downside cushion
        "chr_p50_end_3m": "chr_drift_3m",          # Chronos median 3m drift
    }
    show = g[list(cols)].rename(columns=cols).head(TOPN).reset_index(drop=True)
    pd.set_option("display.width", 140)
    print(f"FloorScore — safest buys as of {pd.Timestamp(asof).date()} "
          f"(PIT S&P 500, {len(g)} names ranked)\n")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out = HERE / "floor_today.csv"
    show.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
