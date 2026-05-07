"""Run pro-tier strategies and dump full results."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.monthly_dca.fast_score import evaluate_strategy, load_panel
from experiments.monthly_dca.strategies_pro import all_pro_strategies
from experiments.monthly_dca.deepdive import (
    cagr_dca,
    merge_fwd,
    per_year_breakdown,
    picks_for,
)


def main() -> None:
    panel = load_panel()
    eval_at = panel.index.max()

    print("\n=== PRO STRATEGIES SWEEP ===")
    summaries = []
    for top_k in (1, 3, 5, 10):
        for strat in all_pro_strategies(top_k=top_k):
            er = evaluate_strategy(strat.score_fn, top_k=top_k, name=strat.name,
                                   start="2017-12-31", end="2024-12-31",
                                   panel=panel, delist_iters=100)
            if er.summary.empty:
                continue
            summ = er.summary.copy()
            summ["top_k"] = top_k
            summaries.append(summ)
        print(f"  done top_k={top_k}")

    if summaries:
        big = pd.concat(summaries, ignore_index=True)
        big.to_csv("experiments/monthly_dca/cache/sweep_pro.csv", index=False)
        cols = ["strategy", "top_k", "exit", "n_picks", "win_rate", "win_rate_bias_corr",
                "beat_spy_rate", "median_ret", "cagr_dca_portfolio", "cagr_spy_dca", "edge_vs_spy_dca"]
        print("\n=== TOP 25 BY DCA-PORTFOLIO CAGR ===")
        print(big.sort_values("cagr_dca_portfolio", ascending=False).head(25)[cols].to_string(index=False))

    # Deep dive on the very best:
    from experiments.monthly_dca.strategies_pro import (
        asymmetric_winner, multibagger_lottery, smooth_compounder_pullback,
        deep_value_winner, regime_pullback_winner, proprietary_master_v1,
        proprietary_master_v2, quality_dip_breakout, trend_continuation,
    )
    print("\n=== DEEP DIVE PRO STRATEGIES (k=1, hold_forever) ===")
    pro_fns = {
        "asymmetric_winner": asymmetric_winner,
        "multibagger_lottery": multibagger_lottery,
        "smooth_compounder_pullback": smooth_compounder_pullback,
        "regime_pullback_winner": regime_pullback_winner,
        "deep_value_winner": deep_value_winner,
        "proprietary_master_v1": proprietary_master_v1,
        "proprietary_master_v2": proprietary_master_v2,
        "quality_dip_breakout": quality_dip_breakout,
        "trend_continuation": trend_continuation,
    }
    for name, fn in pro_fns.items():
        for k in (1, 3):
            picks = picks_for(fn, top_k=k)
            if picks.empty:
                print(f"  {name} k={k}: NO PICKS")
                continue
            merged = merge_fwd(picks)
            stats = cagr_dca(merged, "ret__hold_forever", eval_at, panel, delist_iters=200)
            print(f"  {name} k={k} hold_forever:")
            print(f"    n_picks={stats.get('n')}  win={stats.get('win_rate'):.3f} (bias-corr={stats.get('win_rate_bias_corr_median'):.3f})")
            print(f"    CAGR={stats.get('cagr_dca'):.3f}  bias-corr median={stats.get('cagr_dca_bias_corr_median'):.3f}  vs_SPY={stats.get('cagr_spy_dca'):.3f}  edge={stats.get('edge'):+.3f}")
            yb = per_year_breakdown(merged, "ret__hold_forever", panel, eval_at)
            yb.to_csv(f"experiments/monthly_dca/cache/yb_{name}_k{k}.csv", index=False)
            print(yb.to_string(index=False))


if __name__ == "__main__":
    main()
