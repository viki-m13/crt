#!/usr/bin/env python3
"""Test stock-level strategies."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.sector_research.engine import *
from experiments.sector_research.stock_strategy import (
    build_stock_dfs, run_stock_momentum, run_stock_multifactor,
    run_stock_sector_hybrid, STOCKS
)

print("Loading data...")
data = load_sector_data()

# Build stock-level DataFrames
stock_close, stock_open, avail_stocks = build_stock_dfs(data)
print(f"Stock universe: {len(avail_stocks)} stocks")
print(f"Stock close shape: {stock_close.shape}")

# SPY for gating
spy_close = data[BENCHMARK]["Close"]

# Sector close for hybrid
sector_close = build_close_df(data)

PERIODS = [("TRAIN", TRAIN_START, TRAIN_END),
           ("VALID", VALID_START, VALID_END),
           ("TEST", TEST_START, TEST_END)]

experiments = [
    ("stock_mom_20_10pct", run_stock_momentum, {"n_stocks": 20, "vol_target": 0.10}),
    ("stock_mom_20_8pct", run_stock_momentum, {"n_stocks": 20, "vol_target": 0.08}),
    ("stock_mom_10_8pct", run_stock_momentum, {"n_stocks": 10, "vol_target": 0.08}),
    ("stock_mom_30_10pct", run_stock_momentum, {"n_stocks": 30, "vol_target": 0.10}),
    ("stock_mom_no_gate", run_stock_momentum, {"n_stocks": 20, "vol_target": 0.10, "sma_gate": False}),
    ("stock_mom_no_skip", run_stock_momentum, {"n_stocks": 20, "vol_target": 0.10, "skip_recent": 0}),
    ("multifactor_15_8pct", run_stock_multifactor, {"n_stocks": 15, "vol_target": 0.08}),
    ("multifactor_20_10pct", run_stock_multifactor, {"n_stocks": 20, "vol_target": 0.10}),
    ("multifactor_no_gate", run_stock_multifactor, {"n_stocks": 20, "vol_target": 0.10, "sma_gate": False}),
    ("multifactor_hi_mom", run_stock_multifactor, {"n_stocks": 15, "vol_target": 0.08, "mom_weight": 0.7, "lowvol_weight": 0.15, "quality_weight": 0.15}),
]

all_results = {}
for name, fn, params in experiments:
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    try:
        if fn == run_stock_momentum:
            weights = fn(stock_close, stock_open, spy_close, params)
        elif fn == run_stock_multifactor:
            weights = fn(stock_close, stock_open, spy_close, data=data, params=params)
        else:
            weights = fn(stock_close, stock_open, spy_close, sector_close, data=data, params=params)

        results = {}
        for pname, start, end in PERIODS:
            rets, trades = backtest_allocation(weights, stock_close, stock_open, start, end)
            m = compute_metrics(rets)
            spy_m = spy_metrics(sector_close, start, end)
            n_pos = (weights.loc[start:end] > 0).sum(axis=1).mean()
            print(f"  {pname}: Sharpe={m['sharpe']:6.3f} CAGR={m['cagr']:7.1%} MDD={m['max_dd']:7.1%} Vol={m['ann_vol']:5.1%} AvgPos={n_pos:.1f} | SPY={spy_m['sharpe']:.3f}")
            results[pname] = m
        all_results[name] = results
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print(f"\n\n{'='*60}")
print("STOCK STRATEGY SUMMARY (sorted by min Sharpe)")
print(f"{'='*60}")
ranked = sorted(all_results.items(),
    key=lambda x: min(x[1].get("TRAIN",{}).get("sharpe",0),
                       x[1].get("VALID",{}).get("sharpe",0),
                       x[1].get("TEST",{}).get("sharpe",0)),
    reverse=True)
print(f"{'Name':<30} {'Train':>8} {'Valid':>8} {'Test':>8} {'MinSh':>8}")
for name, periods in ranked:
    t = periods.get("TRAIN",{}).get("sharpe",0)
    v = periods.get("VALID",{}).get("sharpe",0)
    te = periods.get("TEST",{}).get("sharpe",0)
    mn = min(t,v,te)
    print(f"{name:<30} {t:>8.3f} {v:>8.3f} {te:>8.3f} {mn:>8.3f}")
