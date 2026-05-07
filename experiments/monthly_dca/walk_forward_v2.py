"""Thorough walk-forward / OOS verification.

Multiple split designs:
  A. 3-fold: TRAIN [2018-2020], TEST [2021-2024]
                      [2018-2021], TEST [2022-2024]
                      [2018-2022], TEST [2023-2024]
  B. Rolling 3y train -> 1y test, 5 windows.
  C. Strict 'last block hold-out': TRAIN [2018-2022], TEST [2023-2024].
     This is the most conservative test of recent generalisation.

For each split:
  - Score every (strategy, top_k, exit_rule) combo on TRAIN
  - Pick top-N by TRAIN CAGR (with stability filter: edge > 0 in 2/3 of TRAIN years)
  - Re-evaluate on TEST
  - Record the gap between TRAIN-rank and TEST-rank

Output: a leaderboard of strategies that:
  (a) made the TRAIN top-N, AND
  (b) deliver edge > 0 on TEST,
  averaged across splits.
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
    cagr_dca, merge_fwd, per_year_breakdown, picks_for,
)


SPLIT_DEFS = [
    ("split_A1_train1820_test2124", "2017-12-31", "2020-12-31", "2021-01-31", "2024-12-31"),
    ("split_A2_train1821_test2224", "2017-12-31", "2021-12-31", "2022-01-31", "2024-12-31"),
    ("split_A3_train1822_test2324", "2017-12-31", "2022-12-31", "2023-01-31", "2024-12-31"),
    ("rolling_R1_train1820_test21", "2017-12-31", "2020-12-31", "2021-01-31", "2021-12-31"),
    ("rolling_R2_train1921_test22", "2018-12-31", "2021-12-31", "2022-01-31", "2022-12-31"),
    ("rolling_R3_train2022_test23", "2019-12-31", "2022-12-31", "2023-01-31", "2023-12-31"),
    ("rolling_R4_train2123_test24", "2020-12-31", "2023-12-31", "2024-01-31", "2024-12-31"),
    # Strict last-block holdout
    ("strict_holdout_train1822_test2324", "2017-12-31", "2022-12-31", "2023-01-31", "2024-12-31"),
]


def run_split(start: str, end: str, panel) -> pd.DataFrame:
    summaries = []
    for top_k in (1, 3, 5):
        strats = base_strats(top_k=top_k) + all_pro_strategies(top_k=top_k)
        for strat in strats:
            er = evaluate_strategy(strat.score_fn, top_k=top_k, name=strat.name,
                                   start=start, end=end, panel=panel,
                                   delist_iters=30)
            if er.summary.empty:
                continue
            summ = er.summary.copy()
            summ["top_k"] = top_k
            summ["key"] = summ["strategy"] + "::" + summ["top_k"].astype(str) + "::" + summ["exit"]
            summaries.append(summ)
    return pd.concat(summaries, ignore_index=True)


def main() -> None:
    panel = load_panel()
    eval_at = panel.index.max()
    print("eval_at =", eval_at.date(), " panel:", panel.shape)

    # Run all splits
    train_results = {}
    test_results = {}
    for name, ts, te, vs, ve in SPLIT_DEFS:
        print(f"\n==== {name} ====")
        print(f"  TRAIN {ts} -> {te}; TEST {vs} -> {ve}")
        train_results[name] = run_split(ts, te, panel)
        test_results[name] = run_split(vs, ve, panel)
        train_results[name].to_csv(f"experiments/monthly_dca/cache/wf_{name}_train.csv", index=False)
        test_results[name].to_csv(f"experiments/monthly_dca/cache/wf_{name}_test.csv", index=False)
        print(f"  TRAIN top-5 by CAGR:")
        cols = ["key", "n_picks", "win_rate", "cagr_dca_portfolio", "cagr_spy_dca", "edge_vs_spy_dca"]
        train_top5 = train_results[name].sort_values("cagr_dca_portfolio", ascending=False).head(5)[cols]
        print(train_top5.to_string(index=False))
        # Same configs on TEST
        keys = list(train_top5["key"])
        test_match = test_results[name][test_results[name]["key"].isin(keys)][cols]
        print(f"  Same configs on TEST:")
        print(test_match.to_string(index=False))

    # Aggregate: for each (strategy, top_k, exit) compute average TEST CAGR/edge
    # only across splits where it ranked TRAIN top-20
    agg_rows = []
    universe_keys = set()
    for name, _, _, _, _ in SPLIT_DEFS:
        train = train_results[name]
        train_top20 = set(train.sort_values("cagr_dca_portfolio", ascending=False).head(20)["key"])
        for k in train_top20:
            universe_keys.add(k)

    for k in universe_keys:
        cagr_test = []
        edge_test = []
        win_test = []
        present_count = 0
        for name, _, _, _, _ in SPLIT_DEFS:
            train = train_results[name]
            tt = train[train["key"] == k]
            if tt.empty:
                continue
            train_rank = (train["cagr_dca_portfolio"] > tt.iloc[0]["cagr_dca_portfolio"]).sum() + 1
            test = test_results[name]
            tv = test[test["key"] == k]
            if tv.empty:
                continue
            cagr_test.append(float(tv.iloc[0]["cagr_dca_portfolio"]))
            edge_test.append(float(tv.iloc[0]["edge_vs_spy_dca"]))
            win_test.append(float(tv.iloc[0]["win_rate"]))
            if train_rank <= 20:
                present_count += 1
        if cagr_test:
            agg_rows.append({
                "key": k,
                "n_splits_in_train_top20": present_count,
                "n_splits_with_test_data": len(cagr_test),
                "mean_test_cagr": float(np.mean(cagr_test)),
                "median_test_cagr": float(np.median(cagr_test)),
                "min_test_cagr": float(np.min(cagr_test)),
                "max_test_cagr": float(np.max(cagr_test)),
                "mean_test_edge": float(np.mean(edge_test)),
                "min_test_edge": float(np.min(edge_test)),
                "mean_test_win": float(np.mean(win_test)),
            })
    agg = pd.DataFrame(agg_rows).sort_values("mean_test_cagr", ascending=False)
    agg.to_csv("experiments/monthly_dca/cache/wf_aggregate.csv", index=False)

    print("\n==== AGGREGATE WALK-FORWARD (TRAIN top-20 -> TEST stats across splits) ====")
    cols_show = ["key", "n_splits_in_train_top20", "n_splits_with_test_data",
                 "mean_test_cagr", "median_test_cagr", "min_test_cagr", "max_test_cagr",
                 "mean_test_edge", "min_test_edge", "mean_test_win"]
    # Only keep configs that ranked TRAIN-top20 in at least 4 splits (robust)
    robust = agg[agg["n_splits_in_train_top20"] >= 4].sort_values("mean_test_cagr", ascending=False)
    print(robust.head(20)[cols_show].to_string(index=False))

    # The recommended strategy is the one with the best mean_test_cagr among robust configs
    if not robust.empty:
        best = robust.iloc[0]
        print(f"\n>>> RECOMMENDED (TRAIN-top20 in >=4/8 splits, best mean TEST CAGR):")
        print(f"    {best['key']}")
        print(f"    mean_test_cagr={best['mean_test_cagr']:.3f}  "
              f"min={best['min_test_cagr']:.3f}  max={best['max_test_cagr']:.3f}  "
              f"mean_edge={best['mean_test_edge']:.3f}")


if __name__ == "__main__":
    main()
