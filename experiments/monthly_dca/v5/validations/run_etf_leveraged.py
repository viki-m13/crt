"""Leveraged-ETF momentum experiment.

Available leveraged ETFs (in prices_extended.parquet):
  - SPXL  : 3x S&P 500  (from 2008-11)
  - UPRO  : 3x S&P 500  (from 2009-06)
  - SSO   : 2x S&P 500  (from 2006-06)
  - TQQQ  : 3x Nasdaq    (from 2010-02)
  - TNA   : 3x Russell-2k (from 2008-11)
  - URTY  : 3x Russell-mid/small (from 2010-02)
  - SOXL  : 3x semis     (from 2010-03)
  - FAS   : 3x financials (from 2008-11)
  - YINN  : 3x China     (from 2009-12)
  - TMV   : -3x TLT (short long-bonds) — INVERSE

No TMF (3x long bonds) available. So no 3x bond complement to TLT.

Backtest window: leveraged universes start 2010-02-11 (when TQQQ/SOXL/URTY launch).
Apples-to-apples comparison restricts the baseline to the same window.

Tests:
  1. Pure leveraged universe (only 2x/3x ETFs)
  2. Mixed: 12 default ETFs + selected leveraged (TQQQ instead of XLK etc.)
  3. Best-leveraged-substitute: replace each ETF with its leveraged version when available
  4. WF on the leveraged subset where data permits
  5. Crisis behaviour: how does leveraged do in 2020 COVID drawdown
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import load_all
from experiments.monthly_dca.v5.validations.run_etf_only import (
    run_etf_only, metrics, DEFAULT_ASSETS,
)

RES = Path(__file__).resolve().parent / "results"

# Available leveraged ETFs
LEV_AVAILABLE = ["SPXL", "UPRO", "SSO", "TQQQ", "TNA", "URTY",
                  "SOXL", "FAS", "YINN", "TMV"]

# Approximate "leveraged equivalents" for the default ETFs
LEV_MAP = {
    "SPY": "UPRO",   # 3x S&P
    "QQQ": "TQQQ",   # 3x Nasdaq
    "IWM": "TNA",    # 3x Russell
    "XLK": "TQQQ",   # closest leveraged for tech
    "XLF": "FAS",    # 3x financials
    "XLE": None,     # no 3x energy in our panel
    "XLU": None,
    "XLV": None,
    "XLP": None,
    "XLY": None,
    "XLI": None,
    "XLB": None,
    "TLT": None,     # we have TMV (-3x) but not 3x long
    "EFA": None,
    "EEM": None,
}

UNIVERSES = {
    "pure_leveraged_5":     ["SPXL", "TQQQ", "TNA", "SOXL", "FAS"],
    "pure_leveraged_8":     ["SPXL", "TQQQ", "TNA", "URTY", "SOXL", "FAS", "YINN", "SSO"],
    "pure_leveraged_full":  LEV_AVAILABLE,
    "mixed_swap_3":         ["UPRO", "TQQQ", "FAS"] + ["XLE","XLU","XLV","XLP","XLY","XLI","XLB","TLT","EFA","EEM"],
    "mixed_swap_5":         ["UPRO", "TQQQ", "FAS", "TNA", "SOXL"] + ["XLE","XLU","XLV","XLP","XLY","XLI","XLB","TLT","EFA","EEM"],
    "mixed_lev_plus_TLT":   ["SPXL", "TQQQ", "TNA", "SOXL", "FAS", "TLT"],
    "mixed_lev_plus_safety": ["SPXL", "TQQQ", "TNA", "SOXL", "FAS", "TLT", "EFA", "XLP", "XLU"],
    "2x_only_SSO":          ["SSO", "TLT", "EFA", "EEM"] + ["XLE","XLF","XLK","XLU","XLV","XLP","XLY","XLI","XLB"],
    "REFERENCE_unleveraged": DEFAULT_ASSETS,
}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    mret = data.mret
    spy = mret["SPY"].copy(); spy.index = pd.to_datetime(spy.index)
    daily = pd.read_parquet(Path("experiments/monthly_dca/cache/prices_extended.parquet"))

    # Leveraged ETFs available from 2010-02-11
    START = "2010-03-31"
    END = "2026-04-30"
    asofs = pd.date_range(start=START, end=END, freq="ME")

    print(f"\n{'='*70}\n  LEVERAGED ETF EXPERIMENT (2010-2026, post-launch window)\n{'='*70}\n")
    print(f"Window: {START} → {END} ({len(asofs)} month-ends)")
    print(f"NOTE: Leveraged ETFs use daily-reset leverage; volatility decay applies.\n")

    rows = []
    for label, assets in UNIVERSES.items():
        avail = [a for a in assets if a in daily.columns]
        if len(avail) < 2:
            print(f"  {label}: SKIP (only {len(avail)} assets available)")
            continue
        tn = min(2, len(avail))
        for top_n in (2, 3):
            if top_n > len(avail): continue
            for lookback_m in (6, 12):
                lookback_d = lookback_m * 21
                sim = run_etf_only(daily, mret, list(asofs),
                                     assets=avail, top_n=top_n,
                                     lookback_d=lookback_d, cost_bps=10.0)
                m = metrics(sim["log"], spy)
                tag = f"{label}_top{top_n}_lb{lookback_m}m"
                print(f"  {tag:<50}  n={len(avail):>2}  CAGR {m['cagr']:>6.2f}%  edge {m['edge']:>+6.2f}pp  Sharpe {m['sharpe']:>4.2f}  MDD {m['mdd']:>7.2f}%  rotations {sim['n_rotations']}")
                rows.append({"variant": tag, "universe_label": label,
                              "n_assets": len(avail), "top_n": top_n,
                              "lookback_months": lookback_m, **m,
                              "n_rotations": sim["n_rotations"]})

    print(f"\n{'='*70}\n  Best leveraged variants (sorted by Sharpe)\n{'='*70}")
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(RES / "etf_leveraged_summary.csv", index=False)
    print(df[["variant","cagr","edge","sharpe","mdd","n_rotations"]].head(10).to_string(index=False))

    # Now run crisis windows on the best leveraged variant + the unleveraged baseline
    print(f"\n{'='*70}\n  Crisis-window stress: 2020 COVID + 2022 bear\n{'='*70}")
    best_variant_label = df.iloc[0]["universe_label"]
    best_assets = UNIVERSES[best_variant_label]
    crises = [
        ("2020 COVID crash (Feb-Apr)",   "2020-01-31", "2020-04-30"),
        ("2020 COVID full year",          "2020-01-31", "2020-12-31"),
        ("2022 bear",                      "2022-01-31", "2022-12-31"),
        ("2018 Q4 selloff",                "2018-09-30", "2018-12-31"),
        ("2023-2024 AI rally",             "2023-01-31", "2024-12-31"),
    ]
    crisis_rows = []
    for name, lo, hi in crises:
        sub_asofs = pd.date_range(start=lo, end=hi, freq="ME")
        # Default baseline
        sim_default = run_etf_only(daily, mret, list(sub_asofs), assets=DEFAULT_ASSETS, top_n=2, lookback_d=252)
        m_default = metrics(sim_default["log"], spy)
        # Best leveraged variant (top-2, 12m)
        avail_best = [a for a in best_assets if a in daily.columns]
        sim_lev = run_etf_only(daily, mret, list(sub_asofs), assets=avail_best, top_n=2, lookback_d=252)
        m_lev = metrics(sim_lev["log"], spy)
        # SPY only
        spy_sub = spy.loc[lo:hi]
        spy_eq = (1 + spy_sub.fillna(0)).cumprod()
        spy_cagr = (spy_eq.iloc[-1] ** (12 / len(spy_sub)) - 1) * 100 if len(spy_sub) else 0
        print(f"\n  {name} ({lo} → {hi})")
        print(f"    Unleveraged ETF-only:     CAGR {m_default['cagr']:+7.2f}%  MDD {m_default['mdd']:+7.2f}%")
        print(f"    Best leveraged ({best_variant_label}):  CAGR {m_lev['cagr']:+7.2f}%  MDD {m_lev['mdd']:+7.2f}%")
        print(f"    SPY:                       CAGR {spy_cagr:+7.2f}%")
        crisis_rows.append({"crisis": name,
                             "default_cagr": m_default["cagr"], "default_mdd": m_default["mdd"],
                             "leveraged_cagr": m_lev["cagr"], "leveraged_mdd": m_lev["mdd"],
                             "spy_cagr": spy_cagr})
    pd.DataFrame(crisis_rows).to_csv(RES / "etf_leveraged_crises.csv", index=False)


if __name__ == "__main__":
    main()
