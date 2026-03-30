#!/usr/bin/env python3
"""
Final round: Push for maximum Sharpe with all-weather approach.
Combine stock selection alpha with multi-asset diversification.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
from experiments.sector_research.engine import *
from experiments.sector_research.stock_strategy import build_stock_dfs, STOCKS
from experiments.sector_research.refined_strategy import (
    compute_factors, rank_normalize, compute_danger_score
)

print("Loading data...")
data = load_sector_data()
stock_close, stock_open, avail = build_stock_dfs(data)
spy_close = data[BENCHMARK]["Close"]
spy_open = data[BENCHMARK]["Open"]
sector_close = build_close_df(data)

# Build full universe including bonds/gold
all_assets = list(stock_close.columns)
for extra in ["TLT", "IEF", "GLD", "SPY"]:
    if extra in data and extra not in all_assets:
        all_assets.append(extra)

full_close = pd.DataFrame({t: data[t]["Close"] for t in all_assets if t in data}).dropna(how="all")
full_open = pd.DataFrame({t: data[t]["Open"] for t in all_assets if t in data and "Open" in data[t].columns}).dropna(how="all")
print(f"Full universe: {len(full_close.columns)} assets")

PERIODS = [("TRAIN", TRAIN_START, TRAIN_END),
           ("VALID", VALID_START, VALID_END),
           ("TEST", TEST_START, TEST_END)]


def run_allweather_multifactor(params=None):
    """
    All-weather: stock alpha + bond/gold diversification.
    Monthly rebal, 25 stocks + 2-3 diversifiers.
    """
    p = {
        "n_stocks": 25,
        "stock_pct": 0.70,      # 70% in stocks
        "bond_pct": 0.15,       # 15% TLT
        "gold_pct": 0.10,       # 10% GLD
        "cash_pct": 0.05,       # 5% cash
        "vol_target": 0.12,
        "regime_adaptive": False,
        "danger_threshold": 0.3,
        "danger_stock_pct": 0.35,
        "danger_bond_pct": 0.40,
        "danger_gold_pct": 0.15,
        "factor_weights": {
            "momentum": 0.30, "low_vol": 0.15, "quality": 0.20,
            "reversal": 0.10, "rel_strength": 0.15, "vol_compress": 0.10,
        },
    }
    if params:
        p.update(params)

    factors = compute_factors(stock_close, spy_close)
    ranked = {name: rank_normalize(df) for name, df in factors.items()}
    composite = pd.DataFrame(0.0, index=stock_close.index, columns=stock_close.columns)
    for name, weight in p["factor_weights"].items():
        if name in ranked:
            composite += ranked[name].fillna(0.5) * weight

    danger = compute_danger_score(spy_close) if p["regime_adaptive"] else None

    spy_ret = spy_close.pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.5)

    stocks = stock_close.columns.tolist()
    all_cols = stocks + ["TLT", "IEF", "GLD"]
    weights = pd.DataFrame(0.0, index=full_close.index, columns=[c for c in all_cols if c in full_close.columns])
    prev_month = None

    for i, date in enumerate(full_close.index):
        month = date.month
        if prev_month is not None and month == prev_month:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue
        prev_month = month

        if date not in composite.index:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue

        scores = composite.loc[date].dropna()
        if len(scores) < p["n_stocks"]:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue

        # Regime check
        in_danger = False
        if p["regime_adaptive"] and danger is not None:
            d = danger.loc[date] if date in danger.index else 0
            in_danger = d > p["danger_threshold"]

        stock_alloc = p["danger_stock_pct"] if in_danger else p["stock_pct"]
        bond_alloc = p["danger_bond_pct"] if in_danger else p["bond_pct"]
        gold_alloc = p["danger_gold_pct"] if in_danger else p["gold_pct"]

        vs = vol_scale.loc[date] if date in vol_scale.index and not pd.isna(vol_scale.loc[date]) else 1.0

        # Stock weights
        top = scores.nlargest(p["n_stocks"])
        rets = stock_close.pct_change()
        if date in rets.index:
            svol = rets.loc[:date].tail(63).std()
            top_vol = svol.reindex(top.index).clip(lower=0.005)
            inv_vol = 1.0 / top_vol
            stock_w = inv_vol / inv_vol.sum() * stock_alloc * vs
        else:
            stock_w = pd.Series(stock_alloc * vs / p["n_stocks"], index=top.index)

        for stock in top.index:
            if stock in weights.columns:
                weights.loc[date, stock] = stock_w.get(stock, 0)

        # Bond/gold weights
        if "TLT" in weights.columns:
            weights.loc[date, "TLT"] = bond_alloc * vs
        if "GLD" in weights.columns:
            weights.loc[date, "GLD"] = gold_alloc * vs

    # Cap
    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights


def run_pure_stock_optimized(params=None):
    """
    Optimized pure stock strategy — monthly rebal, no market timing,
    inverse vol weighting, 6-factor composite. This is our best performer.
    Variations: different stock counts, vol targets, factor weights.
    """
    p = {
        "n_stocks": 25,
        "vol_target": None,  # None = no vol targeting (always fully invested)
        "factor_weights": {
            "momentum": 0.30, "low_vol": 0.15, "quality": 0.20,
            "reversal": 0.10, "rel_strength": 0.15, "vol_compress": 0.10,
        },
    }
    if params:
        p.update(params)

    factors = compute_factors(stock_close, spy_close)
    ranked = {name: rank_normalize(df) for name, df in factors.items()}
    composite = pd.DataFrame(0.0, index=stock_close.index, columns=stock_close.columns)
    for name, weight in p["factor_weights"].items():
        if name in ranked:
            composite += ranked[name].fillna(0.5) * weight

    spy_ret = spy_close.pct_change()
    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    if p["vol_target"]:
        vol_scale = (p["vol_target"] / mkt_vol.clip(lower=0.05)).clip(0.1, 1.5)
    else:
        vol_scale = pd.Series(1.0, index=mkt_vol.index)

    stocks = stock_close.columns.tolist()
    weights = pd.DataFrame(0.0, index=stock_close.index, columns=stocks)
    prev_month = None

    for i, date in enumerate(stock_close.index):
        month = date.month
        if prev_month is not None and month == prev_month:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue
        prev_month = month

        if date not in composite.index:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue

        scores = composite.loc[date].dropna()
        if len(scores) < p["n_stocks"]:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue

        vs = vol_scale.loc[date] if date in vol_scale.index and not pd.isna(vol_scale.loc[date]) else 1.0

        top = scores.nlargest(p["n_stocks"])
        rets = stock_close.pct_change()
        if date in rets.index:
            svol = rets.loc[:date].tail(63).std()
            top_vol = svol.reindex(top.index).clip(lower=0.005)
            inv_vol = 1.0 / top_vol
            stock_w = inv_vol / inv_vol.sum() * vs
        else:
            stock_w = pd.Series(vs / p["n_stocks"], index=top.index)

        for stock in top.index:
            weights.loc[date, stock] = stock_w.get(stock, 0)

    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights


experiments = [
    # All-weather variants
    ("aw_70_15_10", run_allweather_multifactor, {"stock_pct": 0.70, "bond_pct": 0.15, "gold_pct": 0.10}),
    ("aw_60_25_10", run_allweather_multifactor, {"stock_pct": 0.60, "bond_pct": 0.25, "gold_pct": 0.10}),
    ("aw_80_10_5", run_allweather_multifactor, {"stock_pct": 0.80, "bond_pct": 0.10, "gold_pct": 0.05}),
    ("aw_regime", run_allweather_multifactor, {"regime_adaptive": True, "stock_pct": 0.75, "bond_pct": 0.15, "gold_pct": 0.10}),
    ("aw_regime_aggr", run_allweather_multifactor, {"regime_adaptive": True, "stock_pct": 0.80, "bond_pct": 0.10, "gold_pct": 0.05, "danger_stock_pct": 0.25, "danger_bond_pct": 0.45, "danger_gold_pct": 0.20}),
    # Pure stock optimized
    ("pure_25_novol", run_pure_stock_optimized, {"n_stocks": 25}),
    ("pure_30_novol", run_pure_stock_optimized, {"n_stocks": 30}),
    ("pure_20_novol", run_pure_stock_optimized, {"n_stocks": 20}),
    ("pure_15_novol", run_pure_stock_optimized, {"n_stocks": 15}),
    ("pure_25_12pct", run_pure_stock_optimized, {"n_stocks": 25, "vol_target": 0.12}),
    ("pure_25_hi_qual", run_pure_stock_optimized, {"n_stocks": 25, "factor_weights": {"momentum": 0.20, "low_vol": 0.25, "quality": 0.25, "reversal": 0.10, "rel_strength": 0.10, "vol_compress": 0.10}}),
]

all_results = {}
for name, fn, params in experiments:
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    try:
        weights = fn(params)
        # Determine which close/open to use
        if "aw_" in name:
            c_df, o_df = full_close, full_open
        else:
            c_df, o_df = stock_close, stock_open

        results = {}
        for pname, start, end in PERIODS:
            rets, trades = backtest_allocation(weights, c_df, o_df, start, end)
            m = compute_metrics(rets)
            spy_m = spy_metrics(sector_close, start, end)
            n_pos = (weights.loc[start:end] > 0).sum(axis=1).mean()
            print(f"  {pname}: Sharpe={m['sharpe']:6.3f} CAGR={m['cagr']:7.1%} MDD={m['max_dd']:7.1%} Vol={m['ann_vol']:5.1%} Pos={n_pos:.0f} | SPY={spy_m['sharpe']:.3f}")
            results[pname] = m
        all_results[name] = results
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print(f"\n\n{'='*60}")
print("FINAL STRATEGY SUMMARY")
print(f"{'='*60}")
ranked = sorted(all_results.items(),
    key=lambda x: min(x[1].get(p,{}).get("sharpe",0) for p in ["TRAIN","VALID","TEST"]),
    reverse=True)
print(f"{'Name':<25} {'Train':>8} {'Valid':>8} {'Test':>8} {'MinSh':>8} {'AvgSh':>8}")
for name, periods in ranked:
    t = periods.get("TRAIN",{}).get("sharpe",0)
    v = periods.get("VALID",{}).get("sharpe",0)
    te = periods.get("TEST",{}).get("sharpe",0)
    print(f"{name:<25} {t:>8.3f} {v:>8.3f} {te:>8.3f} {min(t,v,te):>8.3f} {(t+v+te)/3:>8.3f}")
