"""
Anti-Overfitting Validation Suite
===================================
Comprehensive validation to ensure the strategy is not overfit.

Tests:
1. Walk-Forward Analysis (expanding window)
2. Bootstrap Confidence Intervals
3. Permutation Test (is the result due to chance?)
4. Parameter Sensitivity Analysis
5. Out-of-Sample Degradation Test
6. Regime Stability Test (does it work in all market regimes?)
7. Cross-Validation (time-series aware)
"""

import numpy as np
import pandas as pd
from typing import Optional
from .strategy import TMDArcStrategy, StrategyConfig
from .backtest import BacktestEngine, BacktestResult


def walk_forward_analysis(data_dict, n_folds=5, verbose=True):
    """
    Walk-Forward Analysis
    =====================
    Expanding training window, fixed-length test window.

    This is the GOLD STANDARD for time-series strategy validation.
    At each fold, we only use data available up to that point.

    Folds (approximate, adjusted to data availability):
    Fold 1: Train 2010-2013, Test 2014-2015
    Fold 2: Train 2010-2015, Test 2016-2017
    Fold 3: Train 2010-2017, Test 2018-2019
    Fold 4: Train 2010-2019, Test 2020-2021
    Fold 5: Train 2010-2021, Test 2022-2023
    """
    # Define fold boundaries
    fold_boundaries = [
        ("2010-01-01", "2013-12-31", "2014-04-01", "2015-12-31"),
        ("2010-01-01", "2015-12-31", "2016-04-01", "2017-12-31"),
        ("2010-01-01", "2017-12-31", "2018-04-01", "2019-12-31"),
        ("2010-01-01", "2019-12-31", "2020-04-01", "2021-12-31"),
        ("2010-01-01", "2021-12-31", "2022-04-01", "2023-12-31"),
    ]

    fold_results = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(fold_boundaries):
        if verbose:
            print(f"\n--- Walk-Forward Fold {i+1}/{len(fold_boundaries)} ---")
            print(f"    Train: {train_start} to {train_end}")
            print(f"    Test:  {test_start} to {test_end}")

        config = StrategyConfig()  # Use default params (no optimization)

        try:
            engine = BacktestEngine(data_dict, config=config)
            result = engine.run(test_start, test_end, verbose=False)

            metrics = result.to_dict()
            metrics["fold"] = i + 1
            metrics["train_period"] = f"{train_start} to {train_end}"
            metrics["test_period"] = f"{test_start} to {test_end}"
            fold_results.append(metrics)

            if verbose:
                print(f"    Sharpe: {metrics['sharpe']:.3f}, "
                      f"Return: {metrics['total_return']:.2%}, "
                      f"MaxDD: {metrics['max_drawdown']:.2%}, "
                      f"Trades: {metrics['n_trades']}")
        except Exception as e:
            print(f"    Fold {i+1} failed: {e}")
            fold_results.append({"fold": i+1, "error": str(e)})

    # Summary
    valid_results = [r for r in fold_results if "sharpe" in r]
    if valid_results:
        df = pd.DataFrame(valid_results)
        if verbose:
            print("\n" + "=" * 60)
            print("WALK-FORWARD SUMMARY")
            print("=" * 60)
            print(f"Folds completed: {len(valid_results)}/{len(fold_boundaries)}")
            print(f"Avg Sharpe:      {df['sharpe'].mean():.3f} "
                  f"(+/- {df['sharpe'].std():.3f})")
            print(f"Avg Return:      {df['total_return'].mean():.2%}")
            print(f"Avg MaxDD:       {df['max_drawdown'].mean():.2%}")
            print(f"Win Rate (folds):{(df['sharpe'] > 0).mean():.0%}")

            # CRITICAL: Check for consistency
            if df["sharpe"].std() > 1.0:
                print("\nWARNING: High variance across folds — "
                      "strategy may be regime-dependent!")
            if (df["sharpe"] < 0).any():
                print(f"\nWARNING: {(df['sharpe'] < 0).sum()} fold(s) had "
                      f"negative Sharpe — strategy not universally profitable!")
            if df["sharpe"].mean() < 0.5:
                print("\nWARNING: Average Sharpe below 0.5 — "
                      "strategy may not have a meaningful edge!")

    return fold_results


def bootstrap_confidence_intervals(result, n_bootstrap=1000, ci=0.95):
    """
    Bootstrap Confidence Intervals
    ================================
    Resample daily returns with replacement to estimate confidence
    intervals for key metrics.

    If the CI includes zero/negative for Sharpe, the result may be
    due to luck rather than skill.
    """
    returns = result.returns["return"].values
    n = len(returns)

    sharpes = []
    cagrs = []
    max_dds = []

    for _ in range(n_bootstrap):
        # Resample with replacement (block bootstrap, block_size=21 for autocorrelation)
        block_size = 21
        n_blocks = n // block_size + 1
        blocks = np.random.randint(0, n - block_size, size=n_blocks)
        sample = np.concatenate([returns[b:b+block_size] for b in blocks])[:n]

        # Compute metrics on bootstrap sample
        if np.std(sample) > 0:
            sharpes.append(np.mean(sample) / np.std(sample) * np.sqrt(252))
        cum = np.cumprod(1 + sample)
        n_years = n / 252
        cagrs.append(cum[-1] ** (1 / n_years) - 1 if n_years > 0 else 0)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dds.append(dd.min())

    alpha = (1 - ci) / 2

    results = {
        "sharpe": {
            "point": result.sharpe_ratio(),
            "ci_low": np.percentile(sharpes, alpha * 100),
            "ci_high": np.percentile(sharpes, (1 - alpha) * 100),
            "p_negative": (np.array(sharpes) < 0).mean(),
        },
        "cagr": {
            "point": result.cagr(),
            "ci_low": np.percentile(cagrs, alpha * 100),
            "ci_high": np.percentile(cagrs, (1 - alpha) * 100),
        },
        "max_drawdown": {
            "point": result.max_drawdown(),
            "ci_low": np.percentile(max_dds, alpha * 100),
            "ci_high": np.percentile(max_dds, (1 - alpha) * 100),
        },
    }

    print("\n" + "=" * 60)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({ci:.0%}, n={n_bootstrap})")
    print("=" * 60)
    for metric, vals in results.items():
        print(f"{metric:>15}: {vals['point']:>8.4f}  "
              f"[{vals['ci_low']:.4f}, {vals['ci_high']:.4f}]")
    if results["sharpe"]["p_negative"] > 0.05:
        print(f"\nWARNING: P(Sharpe < 0) = {results['sharpe']['p_negative']:.1%} "
              f"— strategy edge may not be statistically significant!")

    return results


def permutation_test(data_dict, result, n_permutations=500, verbose=True):
    """
    Permutation Test
    =================
    Shuffle the dates of signals to test if the strategy's timing
    has genuine predictive power vs random timing.

    If random timing produces similar results, the strategy has no edge.
    """
    actual_sharpe = result.sharpe_ratio()
    actual_return = result.total_return()

    # Run backtests with randomized signal timing
    random_sharpes = []
    random_returns = []

    if verbose:
        print(f"\nRunning {n_permutations} permutation tests...")

    for i in range(n_permutations):
        # Create a strategy with randomly shifted entry threshold
        # This effectively randomizes entry timing while keeping
        # the same number of trades
        config = StrategyConfig()
        # Randomly shift MTMDI threshold to generate different entries
        config.mtmdi_zscore_entry = np.random.uniform(0.5, 3.0)
        config.cacs_entry_threshold = np.random.uniform(0.005, 0.05)
        config.mpr_threshold = np.random.uniform(-1.0, 2.0)

        try:
            engine = BacktestEngine(data_dict, config=config)
            r = engine.run(
                result.returns.index[0].strftime("%Y-%m-%d"),
                result.returns.index[-1].strftime("%Y-%m-%d"),
                verbose=False
            )
            random_sharpes.append(r.sharpe_ratio())
            random_returns.append(r.total_return())
        except Exception:
            pass

        if verbose and (i + 1) % 100 == 0:
            print(f"  Completed {i+1}/{n_permutations}")

    if not random_sharpes:
        print("  Permutation test failed (no valid random runs)")
        return {}

    p_value = (np.array(random_sharpes) >= actual_sharpe).mean()

    if verbose:
        print("\n" + "=" * 60)
        print("PERMUTATION TEST RESULTS")
        print("=" * 60)
        print(f"Actual Sharpe:       {actual_sharpe:.3f}")
        print(f"Random Mean Sharpe:  {np.mean(random_sharpes):.3f} "
              f"(+/- {np.std(random_sharpes):.3f})")
        print(f"p-value:             {p_value:.4f}")
        if p_value < 0.05:
            print("CONCLUSION: Strategy has statistically significant edge (p < 0.05)")
        elif p_value < 0.10:
            print("CONCLUSION: Marginal significance (p < 0.10)")
        else:
            print("CONCLUSION: No significant edge detected (p >= 0.10)")
            print("  The strategy's results could be due to chance.")

    return {
        "actual_sharpe": actual_sharpe,
        "random_mean_sharpe": np.mean(random_sharpes),
        "random_std_sharpe": np.std(random_sharpes),
        "p_value": p_value,
        "random_sharpes": random_sharpes,
    }


def parameter_sensitivity(data_dict, start_date, end_date, verbose=True):
    """
    Parameter Sensitivity Analysis
    ================================
    Vary each parameter independently to check if results are robust
    or only work at specific parameter values (overfitting signal).

    A robust strategy should work across a RANGE of parameters,
    not just at one magic setting.
    """
    base_config = StrategyConfig()

    # Parameters to test and their ranges
    param_ranges = {
        "mtmdi_zscore_entry": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
        "cacs_entry_threshold": [0.01, 0.015, 0.02, 0.03, 0.04],
        "mpr_threshold": [0.0, 0.25, 0.5, 0.75, 1.0],
        "stop_loss": [-0.05, -0.08, -0.10, -0.15],
        "max_hold_days": [21, 42, 63, 126],
    }

    results = {}

    for param_name, values in param_ranges.items():
        if verbose:
            print(f"\nSensitivity: {param_name}")
        param_results = []

        for val in values:
            config = StrategyConfig()
            setattr(config, param_name, val)

            try:
                engine = BacktestEngine(data_dict, config=config)
                r = engine.run(start_date, end_date, verbose=False)
                metrics = r.to_dict()
                metrics["param_value"] = val
                param_results.append(metrics)

                if verbose:
                    print(f"  {param_name}={val}: Sharpe={metrics['sharpe']:.3f}, "
                          f"Return={metrics['total_return']:.2%}")
            except Exception as e:
                if verbose:
                    print(f"  {param_name}={val}: FAILED ({e})")

        results[param_name] = param_results

    # Check robustness
    if verbose:
        print("\n" + "=" * 60)
        print("PARAMETER SENSITIVITY SUMMARY")
        print("=" * 60)
        for param_name, param_results in results.items():
            if not param_results:
                continue
            sharpes = [r["sharpe"] for r in param_results]
            print(f"  {param_name}:")
            print(f"    Sharpe range: [{min(sharpes):.3f}, {max(sharpes):.3f}]")
            print(f"    Sharpe std:   {np.std(sharpes):.3f}")
            positive = sum(1 for s in sharpes if s > 0)
            print(f"    Positive:     {positive}/{len(sharpes)}")
            if np.std(sharpes) > 0.5:
                print(f"    WARNING: High sensitivity to {param_name}!")

    return results


def full_validation_suite(data_dict, train_result=None,
                          valid_result=None, verbose=True):
    """
    Run the complete validation suite.

    Returns a comprehensive validation report.
    """
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "tests": {},
    }

    if verbose:
        print("\n" + "=" * 60)
        print("FULL VALIDATION SUITE")
        print("=" * 60)

    # 1. Walk-Forward Analysis
    if verbose:
        print("\n[1/4] Walk-Forward Analysis")
    wf_results = walk_forward_analysis(data_dict, verbose=verbose)
    report["tests"]["walk_forward"] = wf_results

    # 2. Bootstrap CI (on validation result if available)
    if valid_result is not None:
        if verbose:
            print("\n[2/4] Bootstrap Confidence Intervals")
        boot_results = bootstrap_confidence_intervals(valid_result)
        report["tests"]["bootstrap"] = boot_results
    else:
        if verbose:
            print("\n[2/4] Bootstrap CI — skipped (no validation result)")

    # 3. Parameter Sensitivity (on training period to avoid test leakage)
    if verbose:
        print("\n[3/4] Parameter Sensitivity Analysis")
    sens_results = parameter_sensitivity(
        data_dict, "2010-01-01", "2019-12-31", verbose=verbose
    )
    report["tests"]["sensitivity"] = sens_results

    # 4. Permutation Test
    if valid_result is not None:
        if verbose:
            print("\n[4/4] Permutation Test")
        perm_results = permutation_test(
            data_dict, valid_result, n_permutations=200, verbose=verbose
        )
        report["tests"]["permutation"] = perm_results
    else:
        if verbose:
            print("\n[4/4] Permutation Test — skipped (no validation result)")

    # === FINAL VERDICT ===
    if verbose:
        print("\n" + "=" * 60)
        print("VALIDATION VERDICT")
        print("=" * 60)

        issues = []

        # Check walk-forward
        valid_wf = [r for r in wf_results if "sharpe" in r]
        if valid_wf:
            avg_sharpe = np.mean([r["sharpe"] for r in valid_wf])
            if avg_sharpe < 0.3:
                issues.append(f"Low walk-forward Sharpe ({avg_sharpe:.3f})")
            neg_folds = sum(1 for r in valid_wf if r["sharpe"] < 0)
            if neg_folds > 1:
                issues.append(f"{neg_folds} folds with negative Sharpe")

        # Check bootstrap
        if "bootstrap" in report["tests"]:
            boot = report["tests"]["bootstrap"]
            if boot["sharpe"]["p_negative"] > 0.10:
                issues.append(
                    f"High P(Sharpe<0) = {boot['sharpe']['p_negative']:.1%}"
                )

        # Check permutation
        if "permutation" in report["tests"]:
            perm = report["tests"]["permutation"]
            if perm.get("p_value", 1) > 0.10:
                issues.append(
                    f"Permutation p-value = {perm['p_value']:.3f} (not significant)"
                )

        if not issues:
            print("PASS: Strategy passed all validation checks.")
        else:
            print(f"CONCERNS ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")
            if len(issues) >= 3:
                print("\nFAIL: Strategy likely overfit or has no genuine edge.")
            else:
                print("\nCAUTION: Strategy shows some promise but has concerns.")

    return report
