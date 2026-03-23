"""
Autonomous Experiment Loop (AgentHub/Autoresearch-Inspired)
=============================================================
Modeled after Karpathy's autoresearch paradigm:
- Propose a strategy variant
- Backtest it
- If better → keep. If worse → discard.
- Log everything in results.tsv
- Repeat.

This is the "agent" that runs experiments autonomously,
iterating on the TMD-ARC strategy parameters and structure.

Key difference from autoresearch: instead of modifying train.py,
we modify StrategyConfig parameters and test structural variants.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from .strategy import StrategyConfig
from .backtest import BacktestEngine


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


class ExperimentTracker:
    """Tracks all experiments in a TSV log, mirroring autoresearch's results.tsv."""

    def __init__(self, results_dir=None):
        self.results_dir = results_dir or RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        self.tsv_path = os.path.join(self.results_dir, "results.tsv")
        self.experiments = []

        # Load existing results
        if os.path.exists(self.tsv_path):
            try:
                self.experiments = pd.read_csv(
                    self.tsv_path, sep="\t"
                ).to_dict("records")
            except Exception:
                pass

    def log(self, experiment_id, config_dict, metrics, status, description):
        """Log an experiment result."""
        record = {
            "id": experiment_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "sharpe": metrics.get("sharpe", 0),
            "cagr": metrics.get("cagr", 0),
            "max_dd": metrics.get("max_drawdown", 0),
            "n_trades": metrics.get("n_trades", 0),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "status": status,  # "keep", "discard", "crash"
            "description": description,
        }
        # Add config params
        for k, v in config_dict.items():
            record[f"cfg_{k}"] = v

        self.experiments.append(record)

        # Append to TSV
        df = pd.DataFrame([record])
        header = not os.path.exists(self.tsv_path) or os.path.getsize(self.tsv_path) == 0
        df.to_csv(self.tsv_path, sep="\t", mode="a", header=header, index=False)

        # Also save full JSON for detailed analysis
        json_path = os.path.join(
            self.results_dir, f"experiment_{experiment_id}.json"
        )
        with open(json_path, "w") as f:
            json.dump({**record, "config": config_dict, "metrics": metrics},
                      f, indent=2, default=str)

    def best_sharpe(self):
        """Return the best Sharpe achieved so far."""
        if not self.experiments:
            return -999
        kept = [e for e in self.experiments if e.get("status") == "keep"]
        if not kept:
            return -999
        return max(e.get("sharpe", -999) for e in kept)

    def best_config(self):
        """Return the config of the best experiment."""
        if not self.experiments:
            return None
        kept = [e for e in self.experiments if e.get("status") == "keep"]
        if not kept:
            return None
        best = max(kept, key=lambda e: e.get("sharpe", -999))
        return {k.replace("cfg_", ""): v for k, v in best.items()
                if k.startswith("cfg_")}

    def summary(self):
        """Print experiment summary."""
        if not self.experiments:
            print("No experiments logged yet.")
            return

        df = pd.DataFrame(self.experiments)
        print(f"\n{'='*60}")
        print(f"EXPERIMENT LOG: {len(df)} experiments")
        print(f"{'='*60}")
        kept = df[df["status"] == "keep"]
        discarded = df[df["status"] == "discard"]
        crashed = df[df["status"] == "crash"]
        print(f"  Kept: {len(kept)}, Discarded: {len(discarded)}, "
              f"Crashed: {len(crashed)}")
        if len(kept) > 0:
            print(f"  Best Sharpe:  {kept['sharpe'].max():.3f}")
            print(f"  Best CAGR:    {kept['cagr'].max():.2%}")
            best = kept.loc[kept["sharpe"].idxmax()]
            print(f"  Best run:     {best['id']} — {best['description']}")


def generate_variant(base_config=None, experiment_id=0):
    """
    Generate a strategy variant to test.

    Uses a mix of:
    1. Random perturbation of existing parameters
    2. Structured exploration of known good regions
    3. Novel structural changes (different entry logic, exits, etc.)
    """
    if base_config is None:
        base_config = StrategyConfig()

    config = StrategyConfig()

    # Experiment categories
    categories = [
        "mtmdi_threshold",
        "cascade_sensitivity",
        "momentum_filter",
        "risk_management",
        "position_sizing",
        "combined_conservative",
        "combined_aggressive",
        "wide_exploration",
    ]

    category = categories[experiment_id % len(categories)]

    if category == "mtmdi_threshold":
        # Explore MTMDI entry threshold
        config.mtmdi_zscore_entry = np.random.uniform(0.8, 2.5)
        config.mtmdi_zscore_exit = np.random.uniform(0.2, 0.8)
        desc = (f"MTMDI entry={config.mtmdi_zscore_entry:.2f}, "
                f"exit={config.mtmdi_zscore_exit:.2f}")

    elif category == "cascade_sensitivity":
        # Explore cascade gap threshold
        config.cacs_entry_threshold = np.random.uniform(0.005, 0.05)
        desc = f"Cascade threshold={config.cacs_entry_threshold:.3f}"

    elif category == "momentum_filter":
        # Explore MPR threshold
        config.mpr_threshold = np.random.uniform(-0.5, 1.5)
        desc = f"MPR threshold={config.mpr_threshold:.2f}"

    elif category == "risk_management":
        # Explore stop loss and take profit
        config.stop_loss = np.random.uniform(-0.15, -0.03)
        config.take_profit = np.random.uniform(0.10, 0.40)
        config.max_hold_days = np.random.choice([21, 42, 63, 126])
        desc = (f"SL={config.stop_loss:.2f}, TP={config.take_profit:.2f}, "
                f"MaxHold={config.max_hold_days}")

    elif category == "position_sizing":
        # Explore position sizing
        config.max_position_pct = np.random.uniform(0.02, 0.10)
        config.max_total_exposure = np.random.uniform(0.5, 1.0)
        config.vol_target = np.random.uniform(0.10, 0.25)
        desc = (f"MaxPos={config.max_position_pct:.2f}, "
                f"MaxExp={config.max_total_exposure:.2f}, "
                f"VolTgt={config.vol_target:.2f}")

    elif category == "combined_conservative":
        # Conservative combination
        config.mtmdi_zscore_entry = np.random.uniform(1.5, 2.5)
        config.cacs_entry_threshold = np.random.uniform(0.02, 0.04)
        config.mpr_threshold = np.random.uniform(0.5, 1.5)
        config.stop_loss = np.random.uniform(-0.06, -0.04)
        config.max_position_pct = 0.03
        config.max_total_exposure = 0.6
        desc = "Conservative combined"

    elif category == "combined_aggressive":
        # Aggressive combination
        config.mtmdi_zscore_entry = np.random.uniform(0.8, 1.5)
        config.cacs_entry_threshold = np.random.uniform(0.005, 0.02)
        config.mpr_threshold = np.random.uniform(-0.5, 0.5)
        config.stop_loss = np.random.uniform(-0.12, -0.08)
        config.take_profit = np.random.uniform(0.20, 0.40)
        config.max_position_pct = 0.08
        config.max_total_exposure = 0.9
        desc = "Aggressive combined"

    elif category == "wide_exploration":
        # Random exploration of full parameter space
        config.mtmdi_zscore_entry = np.random.uniform(0.5, 3.0)
        config.mtmdi_zscore_exit = np.random.uniform(0.1, 1.0)
        config.cacs_entry_threshold = np.random.uniform(0.003, 0.06)
        config.mpr_threshold = np.random.uniform(-1.0, 2.0)
        config.stop_loss = np.random.uniform(-0.20, -0.03)
        config.take_profit = np.random.uniform(0.08, 0.50)
        config.max_hold_days = np.random.choice([10, 21, 42, 63, 126, 252])
        config.max_position_pct = np.random.uniform(0.02, 0.10)
        config.vol_target = np.random.uniform(0.08, 0.30)
        desc = "Wide exploration"

    return config, category, desc


def run_experiment_loop(data_dict, n_experiments=24, verbose=True):
    """
    Run the autonomous experiment loop.

    Mirrors autoresearch's "LOOP FOREVER" approach:
    - Try a variant
    - If Sharpe improves on validation → keep
    - If not → discard
    - Log everything

    Returns the tracker with all results.
    """
    tracker = ExperimentTracker()
    best_sharpe = tracker.best_sharpe()

    # Use TRAINING period only for the experiment loop
    # Validation is reserved for final evaluation
    train_start = "2010-01-01"
    train_end = "2019-12-31"

    if verbose:
        print(f"\n{'='*60}")
        print("AUTONOMOUS EXPERIMENT LOOP")
        print(f"{'='*60}")
        print(f"Running {n_experiments} experiments on training data")
        print(f"Period: {train_start} to {train_end}")
        print(f"Current best Sharpe: {best_sharpe:.3f}")
        print()

    for i in range(n_experiments):
        exp_id = f"exp_{i+1:03d}"

        # Generate variant
        config, category, desc = generate_variant(experiment_id=i)
        config_dict = {
            k: getattr(config, k) for k in vars(config)
            if not k.startswith("_")
        }

        if verbose:
            print(f"\n--- Experiment {i+1}/{n_experiments}: {desc} ---")

        try:
            # Run backtest on training period
            engine = BacktestEngine(data_dict, config=config)
            result = engine.run(train_start, train_end, verbose=False)
            metrics = result.to_dict()

            # Decision: keep or discard?
            sharpe = metrics["sharpe"]
            n_trades = metrics["n_trades"]

            # Must have minimum trade count (prevents lucky few-trade results)
            if n_trades < 20:
                status = "discard"
                reason = f"Too few trades ({n_trades})"
            elif sharpe > best_sharpe:
                status = "keep"
                best_sharpe = sharpe
                reason = f"New best! Sharpe {sharpe:.3f} > {best_sharpe:.3f}"
            else:
                status = "discard"
                reason = f"Sharpe {sharpe:.3f} <= best {best_sharpe:.3f}"

            tracker.log(exp_id, config_dict, metrics, status, desc)

            if verbose:
                print(f"  Sharpe: {sharpe:.3f}, Return: {metrics['total_return']:.2%}, "
                      f"Trades: {n_trades}")
                print(f"  → {status.upper()}: {reason}")

        except Exception as e:
            tracker.log(exp_id, config_dict, {}, "crash", f"{desc} — {str(e)[:80]}")
            if verbose:
                print(f"  CRASH: {e}")

    tracker.summary()
    return tracker


def run_final_evaluation(data_dict, config=None, verbose=True):
    """
    Final evaluation on VALIDATION set (NOT test set).

    The test set is reserved for the absolute final check,
    to be run only once, ever.
    """
    if config is None:
        config = StrategyConfig()

    if verbose:
        print(f"\n{'='*60}")
        print("FINAL VALIDATION EVALUATION")
        print(f"{'='*60}")

    engine = BacktestEngine(data_dict, config=config)

    # Run on validation period
    valid_result = engine.run("2020-04-01", "2022-12-31", verbose=verbose)

    return valid_result


def run_test_evaluation(data_dict, config=None, verbose=True):
    """
    FINAL out-of-sample test evaluation.
    WARNING: Run this ONCE. Running multiple times invalidates the test.
    """
    if config is None:
        config = StrategyConfig()

    if verbose:
        print(f"\n{'='*60}")
        print("OUT-OF-SAMPLE TEST EVALUATION")
        print("WARNING: This should only be run ONCE!")
        print(f"{'='*60}")

    engine = BacktestEngine(data_dict, config=config)
    test_result = engine.run("2023-04-01", "2026-03-15", verbose=verbose)

    return test_result
