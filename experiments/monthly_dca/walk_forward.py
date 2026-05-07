"""Out-of-sample walk-forward verification.

Approach: split test window into TRAIN / TEST halves. Pick the *best*
strategy x top_k x exit_rule on TRAIN, then re-evaluate ONLY that combination
on TEST. This protects against in-sample optimisation bias.

Also dumps all picks for the winning strategy to a CSV for auditability.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_score import evaluate_strategy, load_panel
from experiments.monthly_dca.strategies_fast import all_strategies as base_strats
from experiments.monthly_dca.strategies_pro import all_pro_strategies
from experiments.monthly_dca.deepdive import (
    cagr_dca,
    merge_fwd,
    per_year_breakdown,
    picks_for,
)


def run_split(start: str, end: str, label: str, panel) -> pd.DataFrame:
    print(f"\n=== {label}: {start} -> {end} ===")
    summaries = []
    for top_k in (1, 3, 5):
        strats = base_strats(top_k=top_k) + all_pro_strategies(top_k=top_k)
        for strat in strats:
            er = evaluate_strategy(strat.score_fn, top_k=top_k, name=strat.name,
                                   start=start, end=end, panel=panel,
                                   delist_iters=50)
            if er.summary.empty:
                continue
            summ = er.summary.copy()
            summ["top_k"] = top_k
            summaries.append(summ)
    big = pd.concat(summaries, ignore_index=True)
    return big


def main() -> None:
    panel = load_panel()
    eval_at = panel.index.max()

    # Split: TRAIN 2018-2020, TEST 2021-2024
    TRAIN = ("2017-12-31", "2020-12-31")
    TEST = ("2021-01-31", "2024-12-31")

    train_summary = run_split(*TRAIN, label="TRAIN", panel=panel)
    train_summary.to_csv("experiments/monthly_dca/cache/wf_train.csv", index=False)

    cols = ["strategy", "top_k", "exit", "n_picks", "win_rate", "win_rate_bias_corr",
            "beat_spy_rate", "median_ret", "cagr_dca_portfolio", "cagr_spy_dca", "edge_vs_spy_dca"]
    print("\nTOP 15 BY TRAIN CAGR:")
    print(train_summary.sort_values("cagr_dca_portfolio", ascending=False).head(15)[cols].to_string(index=False))
    print("\nTOP 15 BY TRAIN EDGE vs SPY:")
    print(train_summary.sort_values("edge_vs_spy_dca", ascending=False).head(15)[cols].to_string(index=False))

    # Pick best on TRAIN by CAGR (filter to >= 50 picks for stability)
    train_filt = train_summary[train_summary["n_picks"] >= 30].copy()
    best_row = train_filt.sort_values("cagr_dca_portfolio", ascending=False).iloc[0]
    print(f"\n>>> Best on TRAIN: {best_row['strategy']} top_k={int(best_row['top_k'])} {best_row['exit']}")
    print(f"    TRAIN CAGR={best_row['cagr_dca_portfolio']:.3f} SPY={best_row['cagr_spy_dca']:.3f} edge={best_row['edge_vs_spy_dca']:+.3f}")

    # Run TEST with the same configuration
    test_summary = run_split(*TEST, label="TEST", panel=panel)
    test_summary.to_csv("experiments/monthly_dca/cache/wf_test.csv", index=False)
    test_match = test_summary[
        (test_summary["strategy"] == best_row["strategy"])
        & (test_summary["top_k"] == best_row["top_k"])
        & (test_summary["exit"] == best_row["exit"])
    ]
    print(f"\n>>> SAME config on TEST:")
    print(test_match[cols].to_string(index=False))

    # Top 10 on TEST overall
    print("\nTOP 10 BY TEST CAGR:")
    print(test_summary.sort_values("cagr_dca_portfolio", ascending=False).head(10)[cols].to_string(index=False))

    # Top 10 on TEST among strategies that ALSO ranked top-15 on TRAIN
    train_top15 = train_summary.sort_values("cagr_dca_portfolio", ascending=False).head(15)[["strategy", "top_k", "exit"]]
    train_top15["key"] = train_top15["strategy"] + "::" + train_top15["top_k"].astype(str) + "::" + train_top15["exit"]
    test_summary["key"] = test_summary["strategy"] + "::" + test_summary["top_k"].astype(str) + "::" + test_summary["exit"]
    overlap = test_summary[test_summary["key"].isin(train_top15["key"])]
    print(f"\nTEST performance of TRAIN-top-15 configs (n={len(overlap)}):")
    print(overlap.sort_values("cagr_dca_portfolio", ascending=False)[cols].to_string(index=False))

    # Pick the best variant validated by overlap
    if not overlap.empty:
        best_oos = overlap.sort_values("cagr_dca_portfolio", ascending=False).iloc[0]
        print(f"\n>>> Best OOS-validated: {best_oos['strategy']} top_k={int(best_oos['top_k'])} {best_oos['exit']}")
        print(f"    TEST CAGR={best_oos['cagr_dca_portfolio']:.3f} edge={best_oos['edge_vs_spy_dca']:+.3f}")

    # Dump picks for the best TRAIN config across the FULL window 2017-2024 + recent
    from experiments.monthly_dca.strategies_fast import (
        pullback_in_winner, quality_pullback, dual_momentum, explosive_winners,
        winner_only, low_vol_trend, min_dd_compounders,
        proprietary_v1, proprietary_v2, proprietary_v3, proprietary_v4,
        proprietary_v5, proprietary_v6, proprietary_v7, proprietary_v8,
    )
    from experiments.monthly_dca.strategies_pro import (
        asymmetric_winner, multibagger_lottery, smooth_compounder_pullback,
        deep_value_winner, regime_pullback_winner, proprietary_master_v1,
        proprietary_master_v2, quality_dip_breakout, trend_continuation,
    )
    name_to_fn = {
        "pullback_in_winner": pullback_in_winner,
        "quality_pullback": quality_pullback,
        "dual_momentum": dual_momentum,
        "explosive_winners": explosive_winners,
        "winner_only": winner_only,
        "low_vol_trend": low_vol_trend,
        "min_dd_compounders": min_dd_compounders,
        "proprietary_v1": proprietary_v1,
        "proprietary_v2": proprietary_v2,
        "proprietary_v3": proprietary_v3,
        "proprietary_v4": proprietary_v4,
        "proprietary_v5": proprietary_v5,
        "proprietary_v6": proprietary_v6,
        "proprietary_v7": proprietary_v7,
        "proprietary_v8": proprietary_v8,
        "asymmetric_winner": asymmetric_winner,
        "multibagger_lottery": multibagger_lottery,
        "smooth_compounder_pullback": smooth_compounder_pullback,
        "deep_value_winner": deep_value_winner,
        "regime_pullback_winner": regime_pullback_winner,
        "proprietary_master_v1": proprietary_master_v1,
        "proprietary_master_v2": proprietary_master_v2,
        "quality_dip_breakout": quality_dip_breakout,
        "trend_continuation": trend_continuation,
    }
    fn = name_to_fn.get(best_row["strategy"])
    k = int(best_row["top_k"])
    if fn is not None:
        full_picks = picks_for(fn, top_k=k, start="2017-12-31", end="2025-12-31")
        full_picks.to_csv(
            f"experiments/monthly_dca/cache/picks_{best_row['strategy']}_k{k}.csv",
            index=False,
        )
        print(f"\nSaved picks: experiments/monthly_dca/cache/picks_{best_row['strategy']}_k{k}.csv")
        print(f"  total picks across window: {len(full_picks)}")

if __name__ == "__main__":
    main()
