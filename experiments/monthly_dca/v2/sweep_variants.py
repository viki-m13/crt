"""
Sweep over strategy variants once predictions are computed.

This is FAST: it reads the pre-computed `ml_preds_v2.parquet` and tries many
combinations of:
- regime gate (which spy filter)
- top_k for normal / recovery / bull regimes
- conviction weighting on/off
- cash-in-crash on/off
- variants of the score (just 1m, 1m+3m, 1m+3m+6m, etc.)

Prints the top 30 by CAGR + Sharpe and saves to a CSV.

Run from the repo root:
    python3 -m experiments.monthly_dca.v2.sweep_variants
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.monthly_dca.v2.ml_strategy import (
    OUT, build_strategy_outputs, simulate_strategy, cagr,
    classify_regime, get_spy_regime, EXCLUDE,
)


def sharpe_annual(eq: pd.DataFrame) -> float:
    r = eq["ret_m"].values
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(12))


def main():
    big = pd.read_parquet(OUT / "panel_cross_section_v3.parquet")
    monthly_returns = pd.read_parquet(OUT / "monthly_returns_clean.parquet")
    preds = pd.read_parquet(OUT / "ml_preds_v2.parquet")
    preds["asof"] = pd.to_datetime(preds["asof"])

    # Sub-window evaluation: full 2003-2024 honestly walk-forward
    # Trim 2026 (incomplete) and pre-2003
    preds_eval = preds[(preds["asof"].dt.year >= 2003) & (preds["asof"].dt.year <= 2024)].copy()
    monthly_returns = monthly_returns.loc["2003-01-01":"2025-12-31"]

    # Variant grid
    score_variants = ["pred"]  # ensemble (avg of 1m/3m/6m); could also try "pred_1m" alone
    if "pred_1m" in preds_eval.columns:
        score_variants += ["pred_1m"]
    if "pred_3m" in preds_eval.columns:
        score_variants += ["pred_3m"]
    if "pred_6m" in preds_eval.columns:
        score_variants += ["pred_6m"]

    K_normal_values = [3, 5, 7, 10, 15]
    K_recovery_values = [3, 5, 7]
    K_bull_values = [5, 7, 10]
    conv_values = [True, False]
    cash_crash_values = [True, False]

    rows = []
    n = 0
    for sv, kn, kr, kb, cv, cc in itertools.product(
        score_variants, K_normal_values, K_recovery_values, K_bull_values, conv_values, cash_crash_values
    ):
        # Use chosen score column
        preds_use = preds_eval.copy()
        if sv != "pred":
            preds_use["pred"] = preds_use[sv]
        outs = build_strategy_outputs(
            preds_use, big,
            top_k_normal=kn, top_k_recovery=kr, top_k_bull=kb,
            use_conviction_weighting=cv, cash_in_crash=cc,
        )
        eq = simulate_strategy(outs, monthly_returns, cost_bps=10.0, starting_cash=1.0)
        if eq.empty:
            continue
        c = cagr(eq) * 100
        sh = sharpe_annual(eq)
        ret_total = eq["equity"].iloc[-1] / 1.0
        # Drawdown
        roll_max = eq["equity"].cummax()
        dd = (eq["equity"] / roll_max - 1).min() * 100
        # Worst calendar year
        eq["year"] = eq["date"].dt.year
        yr = eq.groupby("year")["ret_m"].apply(lambda x: ((1+x).prod() - 1) * 100)
        worst_yr = yr.min()
        best_yr = yr.max()
        n_neg_yrs = (yr < 0).sum()
        rows.append({
            "score": sv, "K_normal": kn, "K_recovery": kr, "K_bull": kb,
            "conv": cv, "cash_crash": cc,
            "CAGR_pct": round(c, 2), "Sharpe": round(sh, 3),
            "MaxDD_pct": round(dd, 2),
            "Final_equity": round(ret_total, 1),
            "Worst_year_pct": round(worst_yr, 1),
            "Best_year_pct": round(best_yr, 1),
            "N_neg_years": int(n_neg_yrs),
        })
        n += 1
        if n % 25 == 0:
            print(f"  Tested {n} variants...")

    df = pd.DataFrame(rows).sort_values("CAGR_pct", ascending=False)
    df.to_csv(OUT / "sweep_variants_results.csv", index=False)
    print(f"\nTotal variants tested: {len(df)}")
    print("\n=== Top 25 by CAGR ===")
    print(df.head(25).to_string(index=False))
    print("\n=== Top 25 by Sharpe ===")
    print(df.sort_values("Sharpe", ascending=False).head(25).to_string(index=False))
    print("\n=== Best with MaxDD > -50% ===")
    safer = df[df["MaxDD_pct"] > -50].sort_values("CAGR_pct", ascending=False)
    print(safer.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
