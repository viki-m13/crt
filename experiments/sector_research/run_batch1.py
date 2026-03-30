#!/usr/bin/env python3
"""Run signals 01-11 and print results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.sector_research.engine import *
from experiments.sector_research.signals import ALL_SIGNALS

print("Loading data...")
data = load_sector_data()
close_df = build_close_df(data)
open_df = build_open_df(data)

batch = {k: v for k, v in ALL_SIGNALS.items() if int(k.split("_")[0]) <= 11}

results = {}
for name, sig_fn in sorted(batch.items()):
    print(f"\n{'='*60}")
    print(f"Signal: {name}")
    print(f"{'='*60}")
    try:
        weights = sig_fn(close_df=close_df, open_df=open_df, data=data)
        for period_name, start, end in [("TRAIN", TRAIN_START, TRAIN_END),
                                         ("VALID", VALID_START, VALID_END),
                                         ("TEST", TEST_START, TEST_END)]:
            rets, trades = backtest_allocation(weights, close_df, open_df, start, end)
            m = compute_metrics(rets)
            spy_m = spy_metrics(close_df, start, end)
            print(f"  {period_name}: Sharpe={m['sharpe']:6.3f} CAGR={m['cagr']:7.1%} MDD={m['max_dd']:7.1%} | SPY: Sharpe={spy_m['sharpe']:6.3f} CAGR={spy_m['cagr']:7.1%}")
            results.setdefault(name, {})[period_name] = m
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print(f"\n\n{'='*60}")
print("BATCH 1 SUMMARY (sorted by TEST Sharpe)")
print(f"{'='*60}")
ranked = sorted(results.items(), key=lambda x: x[1].get("TEST", {}).get("sharpe", 0), reverse=True)
print(f"{'Signal':<30} {'Train':>8} {'Valid':>8} {'Test':>8}")
for name, periods in ranked:
    t = periods.get("TRAIN", {}).get("sharpe", 0)
    v = periods.get("VALID", {}).get("sharpe", 0)
    te = periods.get("TEST", {}).get("sharpe", 0)
    print(f"{name:<30} {t:>8.3f} {v:>8.3f} {te:>8.3f}")
