#!/usr/bin/env python3
"""
Run only validation and test steps (data already downloaded).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.data_pipeline import load_or_download
from src.backtest import BacktestEngine, run_benchmark_comparison
from src.validation import (
    walk_forward_analysis,
    bootstrap_confidence_intervals,
    permutation_test,
    parameter_sensitivity,
)
from src.strategy import StrategyConfig
from src.experiment_loop import run_test_evaluation


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def main():
    print("Loading cached data...")
    data, _ = load_or_download()

    # Best config from experiment loop (experiment 12: SL=-0.07, TP=0.39, MaxHold=21)
    best_config = StrategyConfig()
    best_config.stop_loss = -0.07
    best_config.take_profit = 0.39
    best_config.max_hold_days = 21

    # === VALIDATION (2020-2022) ===
    print("\n[1/5] Validation backtest...")
    engine = BacktestEngine(data, config=best_config)
    valid_result = engine.run("2020-04-01", "2022-12-31", verbose=True)

    # Benchmark
    print("\n[2/5] Benchmark comparison...")
    benchmark = run_benchmark_comparison(data, valid_result)

    # === BOOTSTRAP CI ===
    print("\n[3/5] Bootstrap confidence intervals...")
    boot = bootstrap_confidence_intervals(valid_result, n_bootstrap=1000)

    # === PARAMETER SENSITIVITY ===
    print("\n[4/5] Parameter sensitivity (on training period)...")
    sensitivity = parameter_sensitivity(data, "2010-01-01", "2019-12-31", verbose=True)

    # === OUT-OF-SAMPLE TEST ===
    print("\n[5/5] Out-of-sample test (2023-2026)...")
    test_result = run_test_evaluation(data, config=best_config, verbose=True)
    test_benchmark = run_benchmark_comparison(data, test_result)

    # Save all results
    os.makedirs("results", exist_ok=True)

    validation_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "best_config": {
            "stop_loss": best_config.stop_loss,
            "take_profit": best_config.take_profit,
            "max_hold_days": best_config.max_hold_days,
            "mtmdi_zscore_entry": best_config.mtmdi_zscore_entry,
            "cacs_entry_threshold": best_config.cacs_entry_threshold,
            "mpr_threshold": best_config.mpr_threshold,
        },
        "validation_metrics": valid_result.to_dict(),
        "validation_benchmark": benchmark,
        "bootstrap": boot,
        "test_metrics": test_result.to_dict(),
        "test_benchmark": test_benchmark,
    }

    with open("results/validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2, default=make_serializable)

    test_metrics = test_result.to_dict()
    with open("results/test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=make_serializable)

    # Final summary
    vm = valid_result.to_dict()
    tm = test_metrics
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Period':<25} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>8}")
    print("-" * 65)
    print(f"{'Valid (2020-2022)':<25} {vm['sharpe']:>8.3f} "
          f"{vm['cagr']:>8.2%} {vm['max_drawdown']:>8.2%} "
          f"{vm['win_rate']:>8.2%} {vm['n_trades']:>8}")
    print(f"{'Test  (2023-2026)':<25} {tm['sharpe']:>8.3f} "
          f"{tm['cagr']:>8.2%} {tm['max_drawdown']:>8.2%} "
          f"{tm['win_rate']:>8.2%} {tm['n_trades']:>8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
