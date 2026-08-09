#!/usr/bin/env python3
"""Study 18: inventing a method from what actually survived.

Three things survived ~320 configurations, and they are structurally different
bets rather than variations of one:

  A. VOL-TARGETED EQUITY   time-series. Size SPY by 1/implied-vol. Sharpe 0.84
                           with near-zero out-of-sample decay (0.93 -> 0.85).
  B. SKEW -> STOCK         cross-sectional, market-neutral. IC t=+5.6, the only
                           book positive out of sample (+0.43 holdout).
  C. LONG-DATED WIDE       short-premium carry, defined risk. Sharpe 0.51,
     PUT SPREADS           positive in 7 of 8 years.

Combining weakly-correlated sleeves is the one lever never pulled here, and it
is the only one that raises IR without needing a better signal or more breadth:
for k uncorrelated sleeves of equal Sharpe, the combination scales as sqrt(k).

This also tests a genuinely NOVEL signal already sitting in the data and used
only once, to condition dispersion trades: IMPLIED CORRELATION, computed from
index IV against the component basket. It measures something VIX cannot — how
much co-movement is being priced. Low implied correlation means diversification
is cheap; high means the market is paying up for the one thing that fails in a
crash. As a risk signal for equity exposure it is untested here.

All sleeve returns are pre-computed elsewhere in this project and merely
combined here, so nothing is re-fitted. Weights are inverse-vol on DEV ONLY,
then frozen.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_END = pd.Timestamp("2024-12-31")


def perf(r: pd.Series, ppy: float, label: str, benchmark: float | None = None):
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        print(f"  {label:<40} insufficient")
        return None
    sh = r.mean() / r.std() * math.sqrt(ppy)
    eq = (1 + r).cumprod()
    yrs = len(r) / ppy
    cagr = float(eq.iloc[-1]) ** (1 / max(yrs, 1e-9)) - 1
    dd = float((eq / eq.cummax() - 1).min())
    rel = "" if benchmark is None else f" ({sh-benchmark:+.2f} vs SPY)"
    print(f"  {label:<40} Sharpe={sh:+.2f} CAGR={cagr:+7.1%} maxDD={dd:+7.1%}{rel}")
    return {"sharpe": sh, "cagr": cagr, "dd": dd, "r": r}


def build_sleeves():
    """Assemble the three surviving sleeves onto one weekly calendar."""
    f = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    f["date"] = pd.to_datetime(f.date)
    spy = f[f.act_symbol == "SPY"].sort_values("date").reset_index(drop=True)
    spy["ret"] = spy.spot.pct_change().shift(-1)

    # --- A: vol-targeted SPY on implied vol (past-only target)
    sig = spy.iv_front.replace(0, np.nan)
    tgt = sig.expanding(min_periods=40).median()
    expo = (tgt / sig).clip(0, 2.0).shift(1)
    A = pd.Series((expo * spy.ret).values, index=spy.date, name="A_voltgt")
    SPYr = pd.Series(spy.ret.values, index=spy.date, name="SPY")

    # --- B: skew -> stock cross-sectional long-short (h=8, 2bp costs)
    B = None
    p = os.path.join(HERE, "cache", "optsig_full.parquet")
    if os.path.exists(p):
        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d.date)
        d = d.dropna(subset=["skew25", "fwd8"])
        rows = []
        for dt, g in d.groupby("date"):
            if len(g) < 20:
                continue
            lo, hi = g.skew25.quantile(0.2), g.skew25.quantile(0.8)
            L, S = g[g.skew25 >= hi].fwd8, g[g.skew25 <= lo].fwd8
            if len(L) < 3 or len(S) < 3:
                continue
            rows.append((dt, 0.5 * (L.mean() - S.mean()) - 2.0 / 1e4))
        if rows:
            B = pd.Series(dict(rows)).sort_index()
            B.name = "B_skew_ls"
            B = B / 8.0        # h=8 return spread -> per-observation rate

    # --- C: long-dated wide put spreads (engine equity curve)
    C = None
    pc = os.path.join(HERE, "results", "eq_LD_wide.parquet")
    if os.path.exists(pc):
        eq = pd.read_parquet(pc)["equity"]
        C = eq.pct_change().dropna()
        C.name = "C_ldwide"
    return SPYr, A, B, C


def implied_correlation_signal():
    p = os.path.join(HERE, "cache", "dispersion.parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d.date)
    d = d[(d.rho_imp > 0) & (d.rho_imp < 1.5)].sort_values("date")
    return d.set_index("date").rho_imp


def main():
    SPYr, A, B, C = build_sleeves()
    ppy = 52.0

    def wk(s):
        return (1 + s.fillna(0)).resample("W-FRI").prod() - 1 if s is not None else None

    spy_w, A_w, B_w, C_w = map(wk, (SPYr, A, B, C))

    print("=" * 78)
    print("STUDY 18 — combining what survived")
    print("=" * 78)
    base = perf(spy_w, ppy, "BENCHMARK: SPY buy & hold")
    bsh = base["sharpe"]
    print()
    for nm, s in (("A vol-targeted SPY (implied vol)", A_w),
                  ("B skew->stock long-short", B_w),
                  ("C long-dated wide put spreads", C_w)):
        if s is not None:
            perf(s, ppy, nm, bsh)

    parts = {k: v for k, v in (("A", A_w), ("B", B_w), ("C", C_w)) if v is not None}
    R = pd.DataFrame(parts).dropna()
    print(f"\n--- sleeve correlation (weekly, n={len(R)}) ---")
    print(R.corr().round(3).to_string())

    dev = R[R.index <= DEV_END]
    w = (1 / dev.std()) / (1 / dev.std()).sum()
    print(f"\ninverse-vol weights fitted on DEV only: "
          f"{ {k: round(v,3) for k,v in w.items()} }")

    print("\n" + "=" * 78)
    print("COMBINED PORTFOLIO (weights frozen from dev)")
    print("=" * 78)
    port = (R * w).sum(axis=1)
    perf(port, ppy, "combined A+B+C", bsh)
    for lab, seg in (("  dev", port[port.index <= DEV_END]),
                     ("  holdout", port[port.index > DEV_END])):
        perf(seg, ppy, f"combined {lab.strip()}", bsh)
    # equal-risk pairs
    for pair in (("A", "B"), ("A", "C"), ("B", "C")):
        if all(k in R for k in pair):
            sub = R[list(pair)]
            ww = (1 / sub[sub.index <= DEV_END].std())
            ww = ww / ww.sum()
            perf((sub * ww).sum(axis=1), ppy, f"pair {'+'.join(pair)}", bsh)

    # ---- novel signal: implied correlation as an equity risk gate
    rho = implied_correlation_signal()
    if rho is not None:
        print("\n" + "=" * 78)
        print("NOVEL SIGNAL — implied correlation as an equity risk gate")
        print("  (index IV vs component basket: how much co-movement is priced)")
        print("=" * 78)
        rw = rho.resample("W-FRI").last().reindex(spy_w.index).ffill()
        q = rw.expanding(min_periods=40).quantile(0.7)   # past-only threshold
        med = rw.expanding(min_periods=40).median()
        print(f"  implied correlation: median={rho.median():.3f} "
              f"p90={rho.quantile(0.9):.3f}")
        for nm, mask in (("long only when rho < 70th pct", rw < q),
                         ("long only when rho < median", rw < med),
                         ("half size when rho > 70th pct",
                          None)):
            if mask is None:
                expo = pd.Series(np.where(rw > q, 0.5, 1.0), index=rw.index).shift(1)
            else:
                expo = mask.astype(float).shift(1)
            perf((expo * spy_w).dropna(), ppy, nm, bsh)
        # combine with vol targeting
        if A_w is not None:
            expo = pd.Series(np.where(rw > q, 0.5, 1.0), index=rw.index).shift(1)
            perf((expo * A_w).dropna(), ppy, "vol-target x rho gate", bsh)


if __name__ == "__main__":
    main()
