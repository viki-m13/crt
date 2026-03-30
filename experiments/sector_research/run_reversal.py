#!/usr/bin/env python3
"""Test reversal strategies with various holding periods and params."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.sector_research.reversal_engine import *

print("Loading data...")
data = load_data()
close_df, open_df, avail = build_dfs(data)
print(f"Stocks: {len(avail)}, dates: {close_df.shape[0]}")

PERIODS = [
    ("TRAIN", "2010-01-01", "2019-12-31"),
    ("VALID", "2020-04-01", "2022-12-31"),
    ("TEST", "2023-04-01", "2026-03-15"),
]

spy_close = data[BENCHMARK]["Close"]

experiments = [
    # Pullback in trend
    ("pit_5d_20pos", signal_pullback_in_trend, {"max_hold_days": 5, "n_positions": 20}),
    ("pit_3d_20pos", signal_pullback_in_trend, {"max_hold_days": 3, "n_positions": 20}),
    ("pit_5d_30pos", signal_pullback_in_trend, {"max_hold_days": 5, "n_positions": 30}),
    ("pit_10d_20pos", signal_pullback_in_trend, {"max_hold_days": 10, "n_positions": 20}),
    # Quality reversal
    ("qrev_5d_20", signal_reversal_quality, {"max_hold_days": 5, "n_positions": 20}),
    ("qrev_3d_20", signal_reversal_quality, {"max_hold_days": 3, "n_positions": 20}),
    ("qrev_5d_30", signal_reversal_quality, {"max_hold_days": 5, "n_positions": 30}),
    # Gap recovery
    ("gap_5d_20", signal_gap_recovery, {"max_hold_days": 5, "n_positions": 20}),
    ("gap_3d_20", signal_gap_recovery, {"max_hold_days": 3, "n_positions": 20}),
    # Oversold bounce
    ("osb_5d_20", signal_oversold_bounce, {"max_hold_days": 5, "n_positions": 20}),
    ("osb_10d_20", signal_oversold_bounce, {"max_hold_days": 10, "n_positions": 20}),
    # Combined
    ("comb_5d_20", signal_combined_reversal, {"max_hold_days": 5, "n_positions": 20}),
    ("comb_3d_20", signal_combined_reversal, {"max_hold_days": 3, "n_positions": 20}),
    ("comb_5d_30", signal_combined_reversal, {"max_hold_days": 5, "n_positions": 30}),
    ("comb_10d_20", signal_combined_reversal, {"max_hold_days": 10, "n_positions": 20}),
    ("comb_5d_15", signal_combined_reversal, {"max_hold_days": 5, "n_positions": 15}),
]

# SPY metrics
spy_metrics = {}
for pname, start, end in PERIODS:
    sr = spy_close.loc[start:end].pct_change().dropna()
    spy_metrics[pname] = compute_metrics(sr)

all_results = {}
for name, sig_fn, params in experiments:
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    try:
        results = {}
        for pname, start, end in PERIODS:
            rets = backtest_shorthold(close_df, open_df, sig_fn, start, end, params)
            m = compute_metrics(rets)
            sm = spy_metrics[pname]
            invested = (rets != 0).mean()
            print(f"  {pname}: Sharpe={m['sharpe']:6.3f} CAGR={m['cagr']:7.1%} MDD={m['max_dd']:7.1%} Vol={m['ann_vol']:5.1%} Invested={invested:4.0%} | SPY={sm['sharpe']:.3f}")
            results[pname] = m
        all_results[name] = results
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print(f"\n\n{'='*60}")
print("REVERSAL STRATEGY SUMMARY (sorted by min Sharpe)")
print(f"{'='*60}")
ranked = sorted(all_results.items(),
    key=lambda x: min(x[1].get(p,{}).get("sharpe",0) for p in ["TRAIN","VALID","TEST"]),
    reverse=True)
print(f"{'Name':<20} {'Train':>8} {'Valid':>8} {'Test':>8} {'MinSh':>8} {'AvgSh':>8}")
for name, periods in ranked:
    t = periods.get("TRAIN",{}).get("sharpe",0)
    v = periods.get("VALID",{}).get("sharpe",0)
    te = periods.get("TEST",{}).get("sharpe",0)
    print(f"{name:<20} {t:>8.3f} {v:>8.3f} {te:>8.3f} {min(t,v,te):>8.3f} {(t+v+te)/3:>8.3f}")
