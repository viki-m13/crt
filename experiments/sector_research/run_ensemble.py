#!/usr/bin/env python3
"""Test ensemble strategies with various parameters."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.sector_research.engine import *
from experiments.sector_research.ensemble_strategy import run_ensemble, run_ensemble_v2

print("Loading data...")
data = load_sector_data()
close_df = build_close_df(data)
open_df = build_open_df(data)

PERIODS = [("TRAIN", TRAIN_START, TRAIN_END),
           ("VALID", VALID_START, VALID_END),
           ("TEST", TEST_START, TEST_END)]

experiments = [
    ("ensemble_10pct_vol", run_ensemble, {"vol_target": 0.10}),
    ("ensemble_08pct_vol", run_ensemble, {"vol_target": 0.08}),
    ("ensemble_05pct_vol", run_ensemble, {"vol_target": 0.05}),
    ("ensemble_10_top2", run_ensemble, {"vol_target": 0.10, "max_sectors": 2}),
    ("ensemble_05_top2", run_ensemble, {"vol_target": 0.05, "max_sectors": 2}),
    ("ensemble_05_top1", run_ensemble, {"vol_target": 0.05, "max_sectors": 1}),
    ("ensemble_no_agree", run_ensemble, {"vol_target": 0.08, "agreement_bonus": False}),
    ("ensemble_high_ir", run_ensemble, {"vol_target": 0.08, "signal_weights": {"spread": 0.35, "vp": 0.10, "ir": 0.35, "consensus": 0.10, "momentum": 0.10}}),
    ("ensemble_high_mom", run_ensemble, {"vol_target": 0.08, "signal_weights": {"spread": 0.10, "vp": 0.15, "ir": 0.10, "consensus": 0.30, "momentum": 0.35}}),
    ("v2_default", run_ensemble_v2, {}),
    ("v2_05pct", run_ensemble_v2, {"vol_target": 0.05}),
    ("v2_10pct", run_ensemble_v2, {"vol_target": 0.10}),
    ("v2_top2", run_ensemble_v2, {"vol_target": 0.08, "max_sectors": 2}),
]

all_results = {}
for name, fn, params in experiments:
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    try:
        weights = fn(close_df, open_df, data=data, params=params)
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
print("ENSEMBLE SUMMARY")
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
