#!/usr/bin/env python3
"""Study 5: what Sharpe is actually reachable here? (information-theoretic ceiling)

Rather than only reporting that sleeves failed, derive the ceiling implied by
the measured edge. Grinold's fundamental law:

    IR ≈ IC · sqrt(BR)

where IC is the information coefficient of the signal against realized
structure returns and BR is the number of *independent* bets per year. The
honest version has to measure BR after correlation, because 40 short-vol
positions are nowhere near 40 independent bets.

Effective breadth uses the eigenvalue-based participation ratio of the
cross-sectional return correlation matrix:

    N_eff = (sum λ_i)^2 / sum λ_i^2

which equals N for independent assets and 1 for perfectly correlated ones.

Also computes the cost hurdle: the half-spread paid on entry, expressed in the
same units as the gross edge, so the two can be compared directly.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
LIQUID = {"SPY", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
          "AMD", "MU", "QCOM", "NFLX", "BA", "XOM", "JPM", "BAC", "C", "GM", "F",
          "INTC", "CSCO", "PYPL", "AVGO", "ORCL", "CRM", "ADBE", "TXN", "COST",
          "WMT", "DIS", "CAT", "GE", "PFE", "CVX", "WFC", "MS", "GS"}


def effective_breadth(ret_panel: pd.DataFrame) -> tuple[float, float]:
    """(N_eff, mean pairwise corr) from a date x symbol return panel."""
    R = ret_panel.dropna(axis=1, thresh=int(0.4 * len(ret_panel)))
    R = R.loc[:, R.std() > 0]
    if R.shape[1] < 3:
        return float(R.shape[1]), np.nan
    C = R.corr().values
    C = np.nan_to_num(C, nan=0.0)
    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 0]
    n_eff = float(lam.sum() ** 2 / (lam ** 2).sum())
    iu = np.triu_indices_from(C, k=1)
    return n_eff, float(np.nanmean(C[iu]))


def main():
    st = pd.read_parquet(os.path.join(HERE, "cache", "structures.parquet"))
    st["date"] = pd.to_datetime(st.date)
    st["ret"] = st.pnl / st.margin
    feats = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    feats["date"] = pd.to_datetime(feats.date)

    base = st[(st.structure == "short_straddle_dh") & st.act_symbol.isin(LIQUID)]
    d = base.merge(feats, on=["date", "act_symbol"], how="left", suffixes=("", "_f"))
    d["credit_yield"] = d.credit / d.margin

    print("=" * 68)
    print("STUDY 5 — information-theoretic ceiling on achievable Sharpe")
    print("=" * 68)

    # --- 1) measured IC of the best signal
    ics = []
    for _, g in d.groupby("date"):
        g = g[["credit_yield", "ret"]].dropna()
        if len(g) >= 12:
            ic = spearmanr(g.credit_yield, g.ret).statistic
            if np.isfinite(ic):
                ics.append(ic)
    ics = np.array(ics)
    ic_mean, ic_se = ics.mean(), ics.std() / math.sqrt(len(ics))
    print(f"\n1) Signal strength (credit_yield -> short straddle return, liquid tier)")
    print(f"   IC = {ic_mean:+.4f}  t = {ic_mean/ic_se:+.2f}  over {len(ics)} dates")

    # --- 2) effective breadth after correlation
    panel = base.pivot_table(index="date", columns="act_symbol", values="ret")
    n_eff, mean_corr = effective_breadth(panel)
    n_names = panel.shape[1]
    print(f"\n2) Breadth")
    print(f"   tradable names           : {n_names}")
    print(f"   mean pairwise corr       : {mean_corr:+.3f}")
    print(f"   EFFECTIVE independent    : {n_eff:.1f}  "
          f"({n_eff/n_names:.0%} of nominal)")

    obs_per_year = len(panel) / max((panel.index[-1] - panel.index[0]).days / 365.25, 1e-9)
    # holding period ~1 month => independent rounds per year ~12
    rounds = 12.0
    breadth_year = n_eff * rounds
    print(f"   observation dates/yr     : {obs_per_year:.0f}")
    print(f"   independent bets/yr (N_eff x 12 monthly rounds): {breadth_year:.0f}")

    # --- 3) implied IR ceiling, gross of costs
    ir = ic_mean * math.sqrt(breadth_year)
    print(f"\n3) Fundamental law ceiling (GROSS of costs)")
    print(f"   IR = IC x sqrt(BR) = {ic_mean:.4f} x sqrt({breadth_year:.0f}) = {ir:.2f}")
    need = (5.0 / ic_mean) ** 2
    print(f"   For IR = 5.0 you would need BR = (5/IC)^2 = {need:,.0f} independent")
    print(f"   bets per year — {need/breadth_year:,.0f}x the breadth this market offers.")
    print(f"   Equivalently, at this breadth you would need IC = "
          f"{5.0/math.sqrt(breadth_year):.3f} ({5.0/math.sqrt(breadth_year)/ic_mean:.0f}x measured).")

    # --- 4) the cost hurdle in the same units
    print(f"\n4) Cost hurdle")
    st2 = st[st.act_symbol.isin(LIQUID)]
    gross_edge = d.ret.mean()
    print(f"   mean realized return on margin per trade (net, worst-side): "
          f"{gross_edge:+.4f}")
    # half-spread as fraction of margin, measured from the chains
    ch = E.load_chain(E.available_dates()[len(E.available_dates()) // 2])
    sp = E.parity_spots(ch)
    hs = []
    for sym in LIQUID & set(sp):
        g = ch[(ch.act_symbol == sym) & (ch.dte.between(15, 50)) & (ch.bid > 0)]
        if g.empty:
            continue
        s = sp[sym]
        atm = g[(g.strike / s - 1).abs() < 0.02]
        if len(atm):
            # straddle: two legs, half-spread each, against 30% notional margin
            hs.append(float(atm.spread.median()) * 2 * 100.0 / (0.30 * s * 100.0))
    if hs:
        print(f"   median half-spread cost per straddle round trip, as fraction")
        print(f"   of margin: {np.median(hs)/2:.4f}  (full spread {np.median(hs):.4f})")
        print(f"   -> per-trade cost is {abs(np.median(hs)/2/gross_edge):.1f}x the "
              f"mean per-trade P&L" if gross_edge else "")

    print("\n" + "=" * 68)
    print("Interpretation: the signals are real and strongly significant, but")
    print("this market offers ~2 orders of magnitude too little independent")
    print("breadth to convert an IC of that size into a Sharpe of 5.")
    print("=" * 68)


if __name__ == "__main__":
    main()
