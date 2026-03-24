#!/usr/bin/env python3
"""
Crypto TMD-ARC Full Pipeline Runner
=====================================
Runs the complete experiment pipeline for crypto:
1. Download crypto data
2. Compute features
3. Run autonomous experiment loop
4. Validate best strategy
5. Final out-of-sample test

Usage:
    cd experiments/crypto/
    pip install -r ../requirements.txt
    python run_all.py
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_pipeline import load_or_download, split_data, TRADING_DAYS_PER_YEAR
from src.experiment_loop import (
    run_experiment_loop, run_final_evaluation, run_test_evaluation,
    ExperimentTracker,
)
from src.backtest import BacktestEngine, run_benchmark_comparison
from src.validation import full_validation_suite
from src.strategy import CryptoStrategyConfig


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def main():
    print("=" * 70)
    print("CRYPTO TMD-ARC: Temporal Momentum Dispersion — Crypto Edition")
    print("Autonomous Strategy Discovery Pipeline")
    print(f"Started: {datetime.datetime.now().isoformat()}")
    print("=" * 70)

    # === STEP 1: DATA ===
    print("\n[STEP 1/5] Loading crypto data...")
    data, removal_log = load_or_download()

    os.makedirs("results", exist_ok=True)
    with open("results/removal_log.json", "w") as f:
        json.dump(removal_log, f, indent=2)

    # === STEP 2: BASELINE ===
    print("\n[STEP 2/5] Running baseline strategy on training data...")
    baseline_config = CryptoStrategyConfig()
    engine = BacktestEngine(data, config=baseline_config)
    baseline_result = engine.run("2018-01-01", "2021-12-31", verbose=True)
    baseline_metrics = baseline_result.to_dict()

    with open("results/baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2, default=make_serializable)

    # === STEP 3: EXPERIMENT LOOP ===
    print("\n[STEP 3/5] Running autonomous experiment loop...")
    tracker = run_experiment_loop(data, n_experiments=24, verbose=True)

    best_cfg_dict = tracker.best_config()
    if best_cfg_dict:
        best_config = CryptoStrategyConfig()
        for k, v in best_cfg_dict.items():
            if hasattr(best_config, k):
                setattr(best_config, k, type(getattr(best_config, k))(v))
        print(f"\nBest config found: Sharpe {tracker.best_sharpe():.3f}")
    else:
        best_config = baseline_config

    # === STEP 4: VALIDATION ===
    print("\n[STEP 4/5] Running validation suite...")
    valid_result = run_final_evaluation(data, config=best_config, verbose=True)
    benchmark_results = run_benchmark_comparison(data, valid_result)

    validation_report = full_validation_suite(
        data, train_result=baseline_result, valid_result=valid_result
    )

    with open("results/validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2, default=make_serializable)

    # === STEP 5: OUT-OF-SAMPLE TEST ===
    print("\n[STEP 5/5] Out-of-sample test (2023-2026)...")
    test_result = run_test_evaluation(data, config=best_config, verbose=True)
    test_benchmark = run_benchmark_comparison(data, test_result)

    test_metrics = test_result.to_dict()
    test_metrics["benchmark"] = test_benchmark
    with open("results/test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=make_serializable)

    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("CRYPTO TMD-ARC FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Period':<25} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'WinRate':>8}")
    print("-" * 57)
    bm = baseline_metrics
    print(f"{'Train (2018-2021)':<25} {bm['sharpe']:>8.3f} "
          f"{bm['cagr']:>8.2%} {bm['max_drawdown']:>8.2%} "
          f"{bm['win_rate']:>8.2%}")
    vm = valid_result.to_dict()
    print(f"{'Valid (2022-2023)':<25} {vm['sharpe']:>8.3f} "
          f"{vm['cagr']:>8.2%} {vm['max_drawdown']:>8.2%} "
          f"{vm['win_rate']:>8.2%}")
    tm = test_metrics
    print(f"{'Test  (2023-2026)':<25} {tm['sharpe']:>8.3f} "
          f"{tm['cagr']:>8.2%} {tm['max_drawdown']:>8.2%} "
          f"{tm['win_rate']:>8.2%}")
    print("=" * 70)
    print(f"Completed: {datetime.datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
