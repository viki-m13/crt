#!/usr/bin/env python3
"""Test crash avoidance strategies."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.sector_research.engine import *
from experiments.sector_research.crash_avoidance import run_prism, run_prism_with_bonds

print("Loading data...")
data = load_sector_data()

# Extend close/open to include bonds
all_tickers = [BENCHMARK] + SECTOR_ETFS + ["TLT", "IEF", "HYG", "GLD"]
close_df = build_close_df(data, all_tickers)
open_df = build_open_df(data, all_tickers)
print(f"Close shape: {close_df.shape}, tickers: {list(close_df.columns)}")

PERIODS = [("TRAIN", TRAIN_START, TRAIN_END),
           ("VALID", VALID_START, VALID_END),
           ("TEST", TEST_START, TEST_END)]

experiments = [
    ("prism_default", run_prism, {}),
    ("prism_low_thr", run_prism, {"danger_threshold": 0.2}),
    ("prism_high_thr", run_prism, {"danger_threshold": 0.4}),
    ("prism_5pct", run_prism, {"vol_target": 0.05}),
    ("prism_8pct", run_prism, {"vol_target": 0.08}),
    ("prism_top2", run_prism, {"max_sectors": 2}),
    ("prism_bonds", run_prism_with_bonds, {}),
    ("prism_bonds_low", run_prism_with_bonds, {"danger_threshold": 0.2}),
    ("prism_bonds_high", run_prism_with_bonds, {"danger_threshold": 0.4}),
    ("prism_bonds_5pct", run_prism_with_bonds, {"vol_target": 0.05}),
    ("prism_bonds_8pct", run_prism_with_bonds, {"vol_target": 0.08}),
    ("prism_bonds_heavy", run_prism_with_bonds, {"bond_alloc": 0.8}),
]

all_results = {}
for name, fn, params in experiments:
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    try:
        result = fn(close_df, open_df, data=data, params=params)
        weights, danger = result
        results = {}
        for pname, start, end in PERIODS:
            rets, trades = backtest_allocation(weights, close_df, open_df, start, end)
            m = compute_metrics(rets)
            spy_m = spy_metrics(close_df, start, end)
            # Time in cash
            cash_pct = (weights.loc[start:end].sum(axis=1) == 0).mean()
            print(f"  {pname}: Sharpe={m['sharpe']:6.3f} CAGR={m['cagr']:7.1%} MDD={m['max_dd']:7.1%} Vol={m['ann_vol']:5.1%} Cash={cash_pct:5.1%} | SPY={spy_m['sharpe']:.3f}")
            results[pname] = m
        all_results[name] = results
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print(f"\n\n{'='*60}")
print("CRASH AVOIDANCE SUMMARY (sorted by min Sharpe across periods)")
print(f"{'='*60}")
ranked = sorted(all_results.items(),
    key=lambda x: min(x[1].get("TRAIN",{}).get("sharpe",0),
                       x[1].get("VALID",{}).get("sharpe",0),
                       x[1].get("TEST",{}).get("sharpe",0)),
    reverse=True)
print(f"{'Name':<25} {'Train':>8} {'Valid':>8} {'Test':>8} {'MinSh':>8}")
for name, periods in ranked:
    t = periods.get("TRAIN",{}).get("sharpe",0)
    v = periods.get("VALID",{}).get("sharpe",0)
    te = periods.get("TEST",{}).get("sharpe",0)
    mn = min(t,v,te)
    print(f"{name:<25} {t:>8.3f} {v:>8.3f} {te:>8.3f} {mn:>8.3f}")
