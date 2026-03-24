"""
Anti-Overfitting Validation Suite for Crypto TMD-ARC
=====================================================
Same rigorous validation as stocks, adapted for crypto timeframes.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .strategy import CryptoTMDArcStrategy, CryptoStrategyConfig
from .backtest import BacktestEngine, BacktestResult, TRADING_DAYS_PER_YEAR


def walk_forward_analysis(data_dict, n_folds=4, verbose=True):
    """
    Walk-Forward Analysis for crypto.
    Expanding training window, fixed-length test window.

    Folds (crypto has shorter history):
    Fold 1: Train 2018-2019, Test 2020-H1
    Fold 2: Train 2018-2020, Test 2021-H1
    Fold 3: Train 2018-2021, Test 2022-H1
    Fold 4: Train 2018-2022, Test 2023
    """
    fold_boundaries = [
        ("2018-01-01", "2019-12-31", "2020-04-01", "2020-12-31"),
        ("2018-01-01", "2020-12-31", "2021-04-01", "2021-12-31"),
        ("2018-01-01", "2021-12-31", "2022-04-01", "2022-12-31"),
        ("2018-01-01", "2022-12-31", "2023-04-01", "2023-12-31"),
    ]

    fold_results = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(fold_boundaries):
        if verbose:
            print(f"\n--- Walk-Forward Fold {i+1}/{len(fold_boundaries)} ---")
            print(f"    Train: {train_start} to {train_end}")
            print(f"    Test:  {test_start} to {test_end}")

        config = CryptoStrategyConfig()

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
            fold_results.append({"fold": i + 1, "error": str(e)})

    valid_results = [r for r in fold_results if "sharpe" in r]
    if valid_results and verbose:
        df = pd.DataFrame(valid_results)
        print("\n" + "=" * 60)
        print("WALK-FORWARD SUMMARY")
        print("=" * 60)
        print(f"Folds completed: {len(valid_results)}/{len(fold_boundaries)}")
        print(f"Avg Sharpe:      {df['sharpe'].mean():.3f} "
              f"(+/- {df['sharpe'].std():.3f})")
        print(f"Avg Return:      {df['total_return'].mean():.2%}")
        print(f"Avg MaxDD:       {df['max_drawdown'].mean():.2%}")
        print(f"Win Rate (folds):{(df['sharpe'] > 0).mean():.0%}")

    return fold_results


def bootstrap_confidence_intervals(result, n_bootstrap=1000, ci=0.95):
    """Bootstrap CI adapted for crypto (365-day annualization)."""
    returns = result.returns["return"].values
    n = len(returns)

    sharpes = []
    cagrs = []
    max_dds = []

    for _ in range(n_bootstrap):
        block_size = 30  # 30-day blocks for crypto
        n_blocks = n // block_size + 1
        blocks = np.random.randint(0, max(1, n - block_size), size=n_blocks)
        sample = np.concatenate([returns[b:b + block_size] for b in blocks])[:n]

        if np.std(sample) > 0:
            sharpes.append(np.mean(sample) / np.std(sample) * np.sqrt(TRADING_DAYS_PER_YEAR))
        cum = np.cumprod(1 + sample)
        n_years = n / TRADING_DAYS_PER_YEAR
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

    if verbose := True:
        print("\n" + "=" * 60)
        print(f"BOOTSTRAP CONFIDENCE INTERVALS ({ci:.0%}, n={n_bootstrap})")
        print("=" * 60)
        for metric, vals in results.items():
            print(f"{metric:>15}: {vals['point']:>8.4f}  "
                  f"[{vals['ci_low']:.4f}, {vals['ci_high']:.4f}]")
        if results["sharpe"]["p_negative"] > 0.05:
            print(f"\nWARNING: P(Sharpe < 0) = {results['sharpe']['p_negative']:.1%}")

    return results


def permutation_test(data_dict, result, n_permutations=200, verbose=True):
    """Permutation test for crypto strategy."""
    actual_sharpe = result.sharpe_ratio()
    random_sharpes = []

    if verbose:
        print(f"\nRunning {n_permutations} permutation tests...")

    for i in range(n_permutations):
        config = CryptoStrategyConfig()
        config.mtmdi_zscore_entry = np.random.uniform(0.5, 3.0)
        config.cacs_entry_threshold = np.random.uniform(0.01, 0.08)
        config.mpr_threshold = np.random.uniform(-1.0, 2.0)

        try:
            engine = BacktestEngine(data_dict, config=config)
            r = engine.run(
                result.returns.index[0].strftime("%Y-%m-%d"),
                result.returns.index[-1].strftime("%Y-%m-%d"),
                verbose=False
            )
            random_sharpes.append(r.sharpe_ratio())
        except Exception:
            pass

        if verbose and (i + 1) % 50 == 0:
            print(f"  Completed {i+1}/{n_permutations}")

    if not random_sharpes:
        print("  Permutation test failed")
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

    return {
        "actual_sharpe": actual_sharpe,
        "random_mean_sharpe": np.mean(random_sharpes),
        "random_std_sharpe": np.std(random_sharpes),
        "p_value": p_value,
    }


def parameter_sensitivity(data_dict, start_date, end_date, verbose=True):
    """Parameter sensitivity analysis for crypto."""
    param_ranges = {
        "mtmdi_zscore_entry": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
        "cacs_entry_threshold": [0.01, 0.02, 0.03, 0.05, 0.07],
        "mpr_threshold": [0.0, 0.25, 0.5, 0.75, 1.0],
        "stop_loss": [-0.07, -0.10, -0.15, -0.20],
        "max_hold_days": [14, 21, 30, 45],
    }

    results = {}

    for param_name, values in param_ranges.items():
        if verbose:
            print(f"\nSensitivity: {param_name}")
        param_results = []

        for val in values:
            config = CryptoStrategyConfig()
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

    return results


def full_validation_suite(data_dict, train_result=None, valid_result=None, verbose=True):
    """Run the complete validation suite for crypto."""
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "asset_class": "crypto",
        "tests": {},
    }

    if verbose:
        print("\n" + "=" * 60)
        print("FULL CRYPTO VALIDATION SUITE")
        print("=" * 60)

    # 1. Walk-Forward
    if verbose:
        print("\n[1/4] Walk-Forward Analysis")
    wf_results = walk_forward_analysis(data_dict, verbose=verbose)
    report["tests"]["walk_forward"] = wf_results

    # 2. Bootstrap CI
    if valid_result is not None:
        if verbose:
            print("\n[2/4] Bootstrap Confidence Intervals")
        boot_results = bootstrap_confidence_intervals(valid_result)
        report["tests"]["bootstrap"] = boot_results

    # 3. Parameter Sensitivity
    if verbose:
        print("\n[3/4] Parameter Sensitivity Analysis")
    sens_results = parameter_sensitivity(
        data_dict, "2018-01-01", "2021-12-31", verbose=verbose
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

    return report
