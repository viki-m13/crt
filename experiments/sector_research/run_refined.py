#!/usr/bin/env python3
"""Test refined multifactor strategies."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.sector_research.engine import *
from experiments.sector_research.stock_strategy import build_stock_dfs
from experiments.sector_research.refined_strategy import run_refined_multifactor

print("Loading data...")
data = load_sector_data()
stock_close, stock_open, avail = build_stock_dfs(data)
spy_close = data[BENCHMARK]["Close"]
sector_close = build_close_df(data)
print(f"Stocks: {len(avail)}, dates: {stock_close.shape[0]}")

PERIODS = [("TRAIN", TRAIN_START, TRAIN_END),
           ("VALID", VALID_START, VALID_END),
           ("TEST", TEST_START, TEST_END)]

experiments = [
    # Weekly rebalance variants
    ("weekly_25_12pct", {"n_stocks": 25, "base_vol_target": 0.12, "rebalance_freq": "weekly"}),
    ("weekly_25_10pct", {"n_stocks": 25, "base_vol_target": 0.10, "rebalance_freq": "weekly"}),
    ("weekly_30_12pct", {"n_stocks": 30, "base_vol_target": 0.12, "rebalance_freq": "weekly"}),
    ("weekly_20_12pct", {"n_stocks": 20, "base_vol_target": 0.12, "rebalance_freq": "weekly"}),
    ("weekly_25_15pct", {"n_stocks": 25, "base_vol_target": 0.15, "rebalance_freq": "weekly"}),
    # With regime adaptation
    ("weekly_regime_25", {"n_stocks": 25, "base_vol_target": 0.12, "danger_vol_target": 0.04, "danger_threshold": 0.3, "rebalance_freq": "weekly"}),
    ("weekly_regime_tight", {"n_stocks": 25, "base_vol_target": 0.12, "danger_vol_target": 0.02, "danger_threshold": 0.25, "rebalance_freq": "weekly"}),
    # Monthly for comparison
    ("monthly_25_12pct", {"n_stocks": 25, "base_vol_target": 0.12, "rebalance_freq": "monthly"}),
    ("monthly_regime_25", {"n_stocks": 25, "base_vol_target": 0.12, "danger_vol_target": 0.04, "danger_threshold": 0.3, "rebalance_freq": "monthly"}),
    # High turnover penalty (reduce trading)
    ("weekly_turnover", {"n_stocks": 25, "base_vol_target": 0.12, "rebalance_freq": "weekly", "turnover_penalty": 0.1}),
    # Factor weight variants
    ("weekly_hi_mom", {"n_stocks": 25, "base_vol_target": 0.12, "rebalance_freq": "weekly",
                       "factor_weights": {"momentum": 0.45, "low_vol": 0.10, "quality": 0.15, "reversal": 0.10, "rel_strength": 0.15, "vol_compress": 0.05}}),
    ("weekly_hi_qual", {"n_stocks": 25, "base_vol_target": 0.12, "rebalance_freq": "weekly",
                        "factor_weights": {"momentum": 0.20, "low_vol": 0.20, "quality": 0.30, "reversal": 0.10, "rel_strength": 0.10, "vol_compress": 0.10}}),
    ("weekly_balanced", {"n_stocks": 25, "base_vol_target": 0.12, "rebalance_freq": "weekly",
                         "factor_weights": {"momentum": 0.25, "low_vol": 0.20, "quality": 0.25, "reversal": 0.10, "rel_strength": 0.10, "vol_compress": 0.10}}),
]

all_results = {}
for name, params in experiments:
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    try:
        weights = run_refined_multifactor(
            stock_close, stock_open, spy_close,
            sector_close_df=sector_close, params=params
        )
        results = {}
        for pname, start, end in PERIODS:
            rets, trades = backtest_allocation(weights, stock_close, stock_open, start, end)
            m = compute_metrics(rets)
            spy_m = spy_metrics(sector_close, start, end)
            n_pos = (weights.loc[start:end] > 0).sum(axis=1).mean()
            turnover = weights.loc[start:end].diff().abs().sum(axis=1).mean()
            print(f"  {pname}: Sharpe={m['sharpe']:6.3f} CAGR={m['cagr']:7.1%} MDD={m['max_dd']:7.1%} Vol={m['ann_vol']:5.1%} Pos={n_pos:.0f} TO={turnover:.3f} | SPY={spy_m['sharpe']:.3f}")
            results[pname] = m
        all_results[name] = results
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print(f"\n\n{'='*60}")
print("REFINED STRATEGY SUMMARY (sorted by min Sharpe)")
print(f"{'='*60}")
ranked = sorted(all_results.items(),
    key=lambda x: min(x[1].get(p,{}).get("sharpe",0) for p in ["TRAIN","VALID","TEST"]),
    reverse=True)
print(f"{'Name':<25} {'Train':>8} {'Valid':>8} {'Test':>8} {'MinSh':>8}")
for name, periods in ranked:
    t = periods.get("TRAIN",{}).get("sharpe",0)
    v = periods.get("VALID",{}).get("sharpe",0)
    te = periods.get("TEST",{}).get("sharpe",0)
    print(f"{name:<25} {t:>8.3f} {v:>8.3f} {te:>8.3f} {min(t,v,te):>8.3f}")
