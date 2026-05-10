"""Harness reproducibility test: must match published v3 numbers to 4 dp.

If this test ever fails, either the harness changed or the cached v3 results
moved -- both are critical to investigate before any new experiment.

Run: python3 tests/YLOka/test_harness_repro.py
"""
import sys
sys.path.insert(0, "/home/user/crt")

import numpy as np
import pandas as pd

from strategy.YLOka.harness import (
    StratConfig, simulate, metrics,
    load_panel, load_monthly_returns, load_spy_features,
)


def test_v3_baseline_full_window():
    panel = load_panel()
    mr = load_monthly_returns()
    spy = load_spy_features()
    cfg = StratConfig(name="v3_baseline_repro_test")
    eq = simulate(cfg, panel, mr, spy)
    met = metrics(eq)
    # Published v3 (from cache/v2/sp500_pit/v3_ml_3plus6_summary.json):
    # cagr_full 0.3977406189, sharpe 0.9553637478, max_dd -0.4982861929, n_cash_months 4
    assert met["n_months"] == 268, met["n_months"]
    assert np.isclose(met["cagr"], 0.39774062, atol=1e-4), met["cagr"]
    assert np.isclose(met["sharpe"], 0.95536375, atol=1e-4), met["sharpe"]
    assert np.isclose(met["max_dd"], -0.49828619, atol=1e-4), met["max_dd"]
    assert met["cash_months"] == 4, met["cash_months"]


if __name__ == "__main__":
    test_v3_baseline_full_window()
    print("PASS  test_v3_baseline_full_window  (39.77% / 0.955 / -49.83% / 4 cash months)")
