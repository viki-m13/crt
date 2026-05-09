"""
Fast vectorised sweep using a returns matrix.

Key idea: pre-build (asof, ticker) → next_month_return as a matrix. For any
strategy that picks a list of tickers per month, evaluate the equity curve in
a few numpy ops.

Run: python3 -m experiments.monthly_dca.v2.fast_sweep
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2"

EXCLUDE = ("SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD")


def classify_regime(spy: pd.Series) -> str:
    dd = spy.get("dd_from_52wh", 0.0)
    r21 = spy.get("ret_21d", 0.0)
    r6m = spy.get("mom_6_1", 0.0)
    streak = spy.get("max_below_200_streak", 0.0)
    dsma = spy.get("d_sma200", 0.0)
    mom12 = spy.get("mom_12_1", 0.0)
    mom3 = spy.get("mom_3", 0.0)
    rsi = spy.get("rsi_14", 50.0)

    if (r6m <= -0.10 and r21 <= -0.05) or (dd <= -0.15 and r21 <= -0.04 and rsi < 40):
        return "crash"
    if (streak >= 40 and -0.05 <= dsma <= 0.05 and r21 > 0) or (mom12 < -0.05 and mom3 > 0.05):
        return "recovery"
    if mom12 >= 0.15 and dsma > 0:
        return "bull"
    return "normal"


def classify_regime_alt(spy: pd.Series, mode: str = "default") -> str:
    """Alternative regime classifiers for sweeping."""
    dd = spy.get("dd_from_52wh", 0.0)
    r21 = spy.get("ret_21d", 0.0)
    r6m = spy.get("mom_6_1", 0.0)
    streak = spy.get("max_below_200_streak", 0.0)
    dsma = spy.get("d_sma200", 0.0)
    mom12 = spy.get("mom_12_1", 0.0)
    mom3 = spy.get("mom_3", 0.0)
    rsi = spy.get("rsi_14", 50.0)

    if mode == "default":
        if (r6m <= -0.10 and r21 <= -0.05) or (dd <= -0.15 and r21 <= -0.04 and rsi < 40):
            return "crash"
        if (streak >= 40 and -0.05 <= dsma <= 0.05 and r21 > 0) or (mom12 < -0.05 and mom3 > 0.05):
            return "recovery"
        if mom12 >= 0.15 and dsma > 0:
            return "bull"
        return "normal"

    if mode == "tight":
        # More aggressive crash gate
        if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
            return "crash"
        if (streak >= 40 and dsma > 0 and r21 > 0):
            return "recovery"
        if mom12 >= 0.10 and dsma > 0:
            return "bull"
        return "normal"

    if mode == "loose":
        # Only catastrophic crash
        if r6m <= -0.20 and r21 <= -0.10:
            return "crash"
        if (streak >= 40 and -0.05 <= dsma <= 0.05 and r21 > 0):
            return "recovery"
        if mom12 >= 0.15 and dsma > 0:
            return "bull"
        return "normal"

    if mode == "no_gate":
        return "normal"  # never goes to cash

    if mode == "trend_only":
        # SPY trend filter only
        if dsma <= -0.05:
            return "crash"
        if mom12 >= 0.15 and dsma > 0:
            return "bull"
        return "normal"

    return "normal"


def main():
    big = pd.read_parquet(OUT / "panel_cross_section_v3.parquet")
    big = big.reset_index()
    big["asof"] = pd.to_datetime(big["asof"])

    preds = pd.read_parquet(OUT / "ml_preds_v2.parquet")
    preds["asof"] = pd.to_datetime(preds["asof"])
    monthly_returns = pd.read_parquet(OUT / "monthly_returns_clean.parquet")

    # Build SPY features per asof for regime classification
    spy_rows = big[big["ticker"] == "SPY"].set_index("asof")

    # Build forward 1m return per (asof, ticker) — directly use the fwd_1m_ret in cross-section
    fwd_idx = big.set_index(["asof", "ticker"])["fwd_1m_ret"]

    # Trim to honest range
    YR_MIN, YR_MAX = 2003, 2024
    months = sorted(preds[(preds["asof"].dt.year >= YR_MIN) & (preds["asof"].dt.year <= YR_MAX)]["asof"].unique())

    # Build per-month sorted ranking of (ticker, pred, fwd_ret) for fast top-K lookups
    print("Building per-month ranked tables...")
    t0 = time.time()
    # Filter excluded tickers
    pe = preds[~preds["ticker"].isin(EXCLUDE)].copy()
    # Map (asof, ticker) -> fwd_1m_ret using fwd_idx
    # pe already has fwd_1m_ret column
    # Sort within each asof by pred desc
    pe_sorted = pe.sort_values(["asof", "pred"], ascending=[True, False])
    monthly_table = {}
    for d, gd in pe_sorted.groupby("asof"):
        if d.year < YR_MIN or d.year > YR_MAX:
            continue
        # Convert to numpy arrays for speed
        rets = gd["fwd_1m_ret"].values
        scores = gd["pred"].values
        tickers = gd["ticker"].values
        monthly_table[d] = (rets, scores, tickers)
    print(f"  Built {len(monthly_table)} months in {time.time()-t0:.1f}s")

    # Pre-classify regimes
    print("Classifying regimes...")
    regimes_by_mode = {}
    for mode in ("default", "tight", "loose", "no_gate", "trend_only"):
        regimes_by_mode[mode] = {}
        for d in monthly_table:
            try:
                spy_row = spy_rows.loc[d]
            except KeyError:
                regimes_by_mode[mode][d] = "normal"
                continue
            regimes_by_mode[mode][d] = classify_regime_alt(spy_row, mode=mode)

    def evaluate_strategy(score_col, K_normal, K_recovery, K_bull, conv, cash_crash, regime_mode, cost_bps=10.0):
        """Evaluate a single strategy variant."""
        cf = cost_bps / 10000.0
        equity = 1.0
        rets_m = []
        regimes = regimes_by_mode[regime_mode]

        if score_col != "pred":
            # Need to re-rank by alternative score
            # (We do this lazily; not common case)
            pass

        for d in months:
            if d not in monthly_table:
                rets_m.append(0.0)
                continue
            rets, scores, tickers = monthly_table[d]
            regime = regimes.get(d, "normal")
            if regime == "crash" and cash_crash:
                rets_m.append(0.0)
                continue
            if regime == "recovery":
                k = K_recovery
            elif regime == "bull":
                k = K_bull
            else:
                k = K_normal
            if len(rets) < k:
                rets_m.append(0.0)
                continue
            # Top-k (already sorted by pred desc)
            top_rets = rets[:k]
            top_scores = scores[:k]
            # Treat NaN forward returns as -1.0 (delist; honest)
            top_rets = np.where(np.isnan(top_rets), -1.0, top_rets)
            if conv:
                shifted = top_scores - top_scores.min() + 1e-6
                w = shifted / shifted.sum()
            else:
                w = np.ones(k) / k
            ret = float((top_rets * w).sum())
            equity *= (1 + ret) * (1 - cf)
            rets_m.append(ret)
        rets_m = np.array(rets_m)
        # CAGR
        n_months = len(rets_m)
        years = n_months / 12.0
        if years <= 0 or equity <= 0:
            return 0.0, 0.0, 0.0, 0.0
        cagr = equity ** (1.0 / years) - 1.0
        sharpe = rets_m.mean() / rets_m.std() * np.sqrt(12) if rets_m.std() > 0 else 0
        # Drawdown (on equity curve)
        eq = np.zeros(n_months)
        eq[0] = (1 + rets_m[0]) * (1 - cf) if rets_m[0] != 0 else 1.0
        for i in range(1, n_months):
            eq[i] = eq[i-1] * (1 + rets_m[i]) * (1 - cf if rets_m[i] != 0 else 1.0)
        roll_max = np.maximum.accumulate(eq)
        dd = ((eq / roll_max) - 1).min()
        return cagr, sharpe, dd, equity

    # Sweep grid
    K_normal_values = [3, 5, 7, 10, 15]
    K_recovery_values = [3, 5, 7]
    K_bull_values = [3, 5, 7, 10]
    conv_values = [True, False]
    cash_crash_values = [True, False]
    regime_modes = ["default", "tight", "loose", "no_gate", "trend_only"]

    print("Sweeping variants...")
    rows = []
    n = 0
    t0 = time.time()
    for kn, kr, kb, cv, cc, rm in itertools.product(
        K_normal_values, K_recovery_values, K_bull_values, conv_values, cash_crash_values, regime_modes
    ):
        cagr, sharpe, dd, eq = evaluate_strategy("pred", kn, kr, kb, cv, cc, rm)
        rows.append({
            "K_normal": kn, "K_recovery": kr, "K_bull": kb,
            "conv": cv, "cash_crash": cc, "regime_mode": rm,
            "CAGR_pct": round(cagr * 100, 2),
            "Sharpe": round(sharpe, 3),
            "MaxDD_pct": round(dd * 100, 2),
            "Final_eq": round(eq, 1),
        })
        n += 1
    elapsed = time.time() - t0
    print(f"  Tested {n} variants in {elapsed:.1f}s ({n/max(elapsed,0.001):.1f}/s)")
    df = pd.DataFrame(rows).sort_values("CAGR_pct", ascending=False)
    df.to_csv(OUT / "fast_sweep_results.csv", index=False)
    print("\n=== Top 30 by CAGR ===")
    print(df.head(30).to_string(index=False))
    print("\n=== Top 30 by Sharpe ===")
    print(df.sort_values("Sharpe", ascending=False).head(30).to_string(index=False))
    print("\n=== Best with MaxDD > -50% (more robust) ===")
    safer = df[df["MaxDD_pct"] > -50].sort_values("CAGR_pct", ascending=False)
    print(safer.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
