"""Run a sweep of YLOka Phase-2 experiments on the research window only.

Experiments:
  exp_00_baseline     v3 ml_3plus6, ew, K=3, h=6, tight gate (reproduction)
  exp_01_conv_lo      H2 conviction-spread sizing, lambda=0.5
  exp_02_conv_hi      H2 conviction-spread sizing, lambda=1.5
  exp_03_conv_floor   H2 + cash floor at q25 of universe score
  exp_04_accel        H3 acceleration overlay (filter by pred_1m strength)
  exp_05_donchian     H6 Donchian-130 breakout filter
  exp_06_softcash     H4 soft-cash continuum (no hard regime gate)
  exp_07_softcash_combo  H4 + H2 conv low

All runs are restricted to RESEARCH window (2003-09 -> 2024-04).
Holdout (2024-05+) is NOT touched here.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/home/user/crt")

import time
import pandas as pd

from strategy.YLOka.harness import (
    StratConfig, run_and_log, load_panel, load_monthly_returns,
    load_spy_features, load_prices, RESEARCH_END,
)


def run_all():
    print(">> loading inputs ...")
    panel = load_panel()
    mr = load_monthly_returns()
    spy = load_spy_features()
    prices = load_prices()
    print(f"   panel {panel.shape}, mr {mr.shape}, spy {spy.shape}, prices {prices.shape}")

    experiments = [
        StratConfig(name="exp_00_baseline"),

        # H2 conviction sizing
        StratConfig(name="exp_01_conv_lo", weighting="conv", conv_lambda=0.5),
        StratConfig(name="exp_02_conv_hi", weighting="conv", conv_lambda=1.5),
        StratConfig(name="exp_03_conv_floor",
                    weighting="conv", conv_lambda=0.5, cash_score_floor=0.25),

        # H3 acceleration overlay
        StratConfig(name="exp_04_accel", pick_filter="accel"),

        # H6 Donchian-130 filter
        StratConfig(name="exp_05_donchian", pick_filter="donchian130"),

        # H4 soft-cash
        StratConfig(name="exp_06_softcash", crash_gate=False, soft_cash=True,
                    cash_yield_apr=0.03),
        StratConfig(name="exp_07_softcash_combo", crash_gate=False, soft_cash=True,
                    cash_yield_apr=0.03, weighting="conv", conv_lambda=0.5),

        # Cash-yield-only: same as baseline but with 3% cash yield (sanity)
        StratConfig(name="exp_08_baseline_cy", cash_yield_apr=0.03),
    ]

    out_rows = []
    for cfg in experiments:
        t0 = time.time()
        try:
            met = run_and_log(cfg, panel, mr, spy, prices=prices, window="research")
        except Exception as e:
            print(f"  [{cfg.name}] FAILED: {e}")
            continue
        dt = time.time() - t0
        out_rows.append({"name": cfg.name, **met})
        print(f"  [{cfg.name}] CAGR={met['cagr']*100:6.2f}%  Sharpe={met['sharpe']:.3f}  "
              f"MaxDD={met['max_dd']*100:7.2f}%  cash_m={met['cash_months']}  ({dt:.1f}s)")

    df = pd.DataFrame(out_rows)
    print("\n=== summary ===")
    cols = ["name", "n_months", "cagr", "sharpe", "sortino", "max_dd", "cash_months"]
    print(df[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    run_all()
