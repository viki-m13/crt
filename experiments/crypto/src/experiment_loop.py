"""
Autonomous Experiment Loop for Crypto TMD-ARC
===============================================
Following Karpathy's autoresearch paradigm:
- Fixed prepare.py (evaluation harness)
- Modifiable train.py (strategy parameters)
- Results tracked in results.tsv
- Agent modifies parameters, runs backtest, keeps if improved
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from .strategy import CryptoStrategyConfig, CryptoTMDArcStrategy
from .backtest import BacktestEngine, BacktestResult
from .data_pipeline import TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


class ExperimentTracker:
    """Track experiments in a TSV file (autoresearch-style)."""

    def __init__(self, results_dir=None):
        self.results_dir = results_dir or RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        self.tsv_path = os.path.join(self.results_dir, "results.tsv")
        self.experiments = []

        if os.path.exists(self.tsv_path):
            try:
                self.experiments = pd.read_csv(
                    self.tsv_path, sep="\t"
                ).to_dict("records")
            except Exception:
                pass

    def log(self, exp_id, metrics, config_dict, description, status="ok"):
        entry = {
            "id": exp_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "sharpe": round(metrics.get("sharpe", 0), 4),
            "cagr": round(metrics.get("cagr", 0), 4),
            "max_dd": round(metrics.get("max_drawdown", 0), 4),
            "n_trades": metrics.get("n_trades", 0),
            "win_rate": round(metrics.get("win_rate", 0), 4),
            "profit_factor": round(metrics.get("profit_factor", 0), 2),
            "status": status,
            "description": description,
        }
        for k, v in config_dict.items():
            entry[f"cfg_{k}"] = v

        self.experiments.append(entry)

        df = pd.DataFrame(self.experiments)
        df.to_csv(self.tsv_path, sep="\t", index=False)

        # Also save individual experiment JSON
        exp_path = os.path.join(self.results_dir, f"experiment_{exp_id:03d}.json")
        with open(exp_path, "w") as f:
            json.dump(entry, f, indent=2, default=str)

    def best_sharpe(self):
        if not self.experiments:
            return 0
        return max(e.get("sharpe", 0) for e in self.experiments if e.get("status") == "ok")

    def best_config(self):
        if not self.experiments:
            return None
        best = max(
            [e for e in self.experiments if e.get("status") == "ok"],
            key=lambda e: e.get("sharpe", 0),
            default=None
        )
        if best is None:
            return None
        return {k.replace("cfg_", ""): v for k, v in best.items() if k.startswith("cfg_")}


def run_experiment_loop(data_dict, n_experiments=24, verbose=True):
    """
    Run the autonomous experiment loop.
    Tests different parameter combinations and tracks results.
    """
    tracker = ExperimentTracker()

    # Experiment configurations to test
    experiments = [
        # Baseline
        {"desc": "Baseline defaults", "cfg": {}},
        # MTMDI threshold variations
        {"desc": "Lower MTMDI threshold (1.0)", "cfg": {"mtmdi_zscore_entry": 1.0}},
        {"desc": "Higher MTMDI threshold (2.0)", "cfg": {"mtmdi_zscore_entry": 2.0}},
        {"desc": "Very high MTMDI threshold (2.5)", "cfg": {"mtmdi_zscore_entry": 2.5}},
        # Cascade variations
        {"desc": "Lower cascade threshold (0.01)", "cfg": {"cacs_entry_threshold": 0.01}},
        {"desc": "Higher cascade threshold (0.05)", "cfg": {"cacs_entry_threshold": 0.05}},
        # Stop loss variations
        {"desc": "Tighter stop loss (-7%)", "cfg": {"stop_loss": -0.07}},
        {"desc": "Wider stop loss (-15%)", "cfg": {"stop_loss": -0.15}},
        {"desc": "Very wide stop loss (-20%)", "cfg": {"stop_loss": -0.20}},
        # Take profit variations
        {"desc": "Lower take profit (25%)", "cfg": {"take_profit": 0.25}},
        {"desc": "Higher take profit (50%)", "cfg": {"take_profit": 0.50}},
        {"desc": "Very high take profit (75%)", "cfg": {"take_profit": 0.75}},
        # Hold period variations
        {"desc": "Shorter hold (14 days)", "cfg": {"max_hold_days": 14}},
        {"desc": "Longer hold (30 days)", "cfg": {"max_hold_days": 30}},
        {"desc": "Long hold (45 days)", "cfg": {"max_hold_days": 45}},
        # Position sizing
        {"desc": "Smaller positions (5%)", "cfg": {"max_position_pct": 0.05}},
        {"desc": "Larger positions (12%)", "cfg": {"max_position_pct": 0.12}},
        # Combined experiments
        {"desc": "Aggressive: low thresh, wide stops",
         "cfg": {"mtmdi_zscore_entry": 1.0, "stop_loss": -0.15, "take_profit": 0.50}},
        {"desc": "Conservative: high thresh, tight stops",
         "cfg": {"mtmdi_zscore_entry": 2.0, "stop_loss": -0.07, "take_profit": 0.25}},
        {"desc": "Fast trading: short hold, tight exits",
         "cfg": {"max_hold_days": 14, "stop_loss": -0.07, "take_profit": 0.20}},
        {"desc": "Long swing: extended hold, wide exits",
         "cfg": {"max_hold_days": 45, "stop_loss": -0.20, "take_profit": 0.75}},
        {"desc": "High vol target (35%)",
         "cfg": {"vol_target": 0.35}},
        {"desc": "Low vol target (15%)",
         "cfg": {"vol_target": 0.15}},
        {"desc": "Best combo search 1",
         "cfg": {"mtmdi_zscore_entry": 1.25, "stop_loss": -0.12, "take_profit": 0.40, "max_hold_days": 25}},
    ]

    for i, exp in enumerate(experiments[:n_experiments]):
        exp_id = len(tracker.experiments) + 1
        if verbose:
            print(f"\n--- Experiment {exp_id}: {exp['desc']} ---")

        config = CryptoStrategyConfig()
        for k, v in exp["cfg"].items():
            setattr(config, k, v)

        config_dict = {
            "mtmdi_zscore_entry": config.mtmdi_zscore_entry,
            "cacs_entry_threshold": config.cacs_entry_threshold,
            "mpr_threshold": config.mpr_threshold,
            "stop_loss": config.stop_loss,
            "take_profit": config.take_profit,
            "max_hold_days": config.max_hold_days,
            "max_position_pct": config.max_position_pct,
            "vol_target": config.vol_target,
        }

        try:
            engine = BacktestEngine(data_dict, config=config)
            result = engine.run(TRAIN_START, TRAIN_END, verbose=False)
            metrics = result.to_dict()
            tracker.log(exp_id, metrics, config_dict, exp["desc"], "ok")

            if verbose:
                print(f"  Sharpe: {metrics['sharpe']:.3f}, "
                      f"CAGR: {metrics['cagr']:.2%}, "
                      f"MaxDD: {metrics['max_drawdown']:.2%}, "
                      f"Trades: {metrics['n_trades']}")
        except Exception as e:
            tracker.log(exp_id, {}, config_dict, exp["desc"], f"error: {e}")
            if verbose:
                print(f"  FAILED: {e}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT LOOP COMPLETE")
        print(f"Best Sharpe: {tracker.best_sharpe():.3f}")
        best_cfg = tracker.best_config()
        if best_cfg:
            print(f"Best Config: {best_cfg}")

    return tracker


def run_final_evaluation(data_dict, config=None, verbose=True):
    """Run on validation period."""
    if config is None:
        config = CryptoStrategyConfig()
    engine = BacktestEngine(data_dict, config=config)
    return engine.run(VALID_START, VALID_END, verbose=verbose)


def run_test_evaluation(data_dict, config=None, verbose=True):
    """Run on out-of-sample test period."""
    if config is None:
        config = CryptoStrategyConfig()
    engine = BacktestEngine(data_dict, config=config)
    return engine.run(TEST_START, TEST_END, verbose=verbose)
