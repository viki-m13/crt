#!/usr/bin/env python3
"""Test composite strategies V1, V2, V3 with various parameters."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.sector_research.engine import *
from experiments.sector_research.composite_strategy import run_sdamc, run_sdamc_v2, run_sdamc_v3

print("Loading data...")
data = load_sector_data()
close_df = build_close_df(data)
open_df = build_open_df(data)

PERIODS = [("TRAIN", TRAIN_START, TRAIN_END),
           ("VALID", VALID_START, VALID_END),
           ("TEST", TEST_START, TEST_END)]

experiments = [
    ("SDAMC_v1_default", run_sdamc, {}),
    ("SDAMC_v1_tight", run_sdamc, {"spread_z_threshold": -1.0, "min_consensus": 3}),
    ("SDAMC_v1_loose", run_sdamc, {"spread_z_threshold": 0, "min_consensus": 2}),
    ("SDAMC_v2_default", run_sdamc_v2, {}),
    ("SDAMC_v2_1sector", run_sdamc_v2, {"max_sectors": 1, "vol_target": 0.10}),
    ("SDAMC_v2_relaxed", run_sdamc_v2, {"min_consensus": 2, "min_rel_strength": -0.05}),
    ("SSA_v3_default", run_sdamc_v3, {}),
    ("SSA_v3_tight", run_sdamc_v3, {"standout_threshold": 1.5, "max_sectors": 1}),
    ("SSA_v3_loose", run_sdamc_v3, {"standout_threshold": 0.5, "max_sectors": 3}),
    ("SSA_v3_fast", run_sdamc_v3, {"mom_lookback": 21, "standout_threshold": 1.0}),
    ("SSA_v3_slow", run_sdamc_v3, {"mom_lookback": 126, "standout_threshold": 1.0}),
]

all_results = {}
for name, fn, params in experiments:
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    try:
        weights = fn(close_df, open_df, params)
        results = {}
        for pname, start, end in PERIODS:
            rets, trades = backtest_allocation(weights, close_df, open_df, start, end)
            m = compute_metrics(rets)
            spy_m = spy_metrics(close_df, start, end)
            print(f"  {pname}: Sharpe={m['sharpe']:6.3f} CAGR={m['cagr']:7.1%} MDD={m['max_dd']:7.1%} Vol={m['ann_vol']:5.1%} TiM={m['time_in_market']:5.1%} | SPY={spy_m['sharpe']:.3f}")
            results[pname] = m
        all_results[name] = results
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print(f"\n\n{'='*60}")
print("COMPOSITE SUMMARY (sorted by avg Sharpe across all periods)")
print(f"{'='*60}")
ranked = sorted(all_results.items(),
    key=lambda x: (x[1].get("TRAIN",{}).get("sharpe",0) +
                    x[1].get("VALID",{}).get("sharpe",0) +
                    x[1].get("TEST",{}).get("sharpe",0)) / 3,
    reverse=True)
print(f"{'Name':<25} {'Train':>8} {'Valid':>8} {'Test':>8} {'Avg':>8}")
for name, periods in ranked:
    t = periods.get("TRAIN",{}).get("sharpe",0)
    v = periods.get("VALID",{}).get("sharpe",0)
    te = periods.get("TEST",{}).get("sharpe",0)
    avg = (t+v+te)/3
    print(f"{name:<25} {t:>8.3f} {v:>8.3f} {te:>8.3f} {avg:>8.3f}")
