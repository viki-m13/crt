#!/usr/bin/env python3
"""
Ultimate attempt: PRISM + Dispersion Timing + Beta Targeting
=============================================================
Idea: The PRISM multifactor portfolio works best when cross-sectional
dispersion is high (stock-picking adds value). During low dispersion
(all stocks correlated), reduce exposure.

Also: target low portfolio beta to remove market risk.
Pure alpha should have much lower vol -> higher Sharpe.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
from experiments.sector_research.reversal_engine import load_data, build_dfs, compute_metrics, BENCHMARK, STOCKS
from experiments.sector_research.refined_strategy import compute_factors, rank_normalize

print("Loading data...")
data = load_data()
close_df, open_df, avail = build_dfs(data)
spy_close = data[BENCHMARK]["Close"]
spy_open = data[BENCHMARK]["Open"]
print(f"Stocks: {len(avail)}")

print("Computing factors...")
factors = compute_factors(close_df, spy_close)
ranked = {name: rank_normalize(df) for name, df in factors.items()}
FACTOR_WEIGHTS = {"momentum": 0.30, "low_vol": 0.15, "quality": 0.20,
                  "reversal": 0.10, "rel_strength": 0.15, "vol_compress": 0.10}
composite = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
for name, weight in FACTOR_WEIGHTS.items():
    if name in ranked:
        composite += ranked[name].fillna(0.5) * weight

# Pre-compute signals
rets = close_df.pct_change()
spy_ret = spy_close.pct_change()

# Cross-sectional dispersion (std of stock returns)
dispersion = rets.std(axis=1).rolling(21).mean()
disp_ma = dispersion.rolling(252).mean()
disp_z = (dispersion - disp_ma) / dispersion.rolling(252).std().clip(lower=1e-8)

# Stock betas
rolling_beta = {}
for stock in close_df.columns:
    cov = rets[stock].rolling(63).cov(spy_ret)
    var = spy_ret.rolling(63).var().clip(lower=1e-10)
    rolling_beta[stock] = cov / var
beta_df = pd.DataFrame(rolling_beta, index=close_df.index)

PERIODS = [
    ("TRAIN", "2010-01-01", "2019-12-31"),
    ("VALID", "2020-04-01", "2022-12-31"),
    ("TEST", "2023-04-01", "2026-03-15"),
]
spy_metrics = {}
for pname, start, end in PERIODS:
    sr = spy_close.loc[start:end].pct_change().dropna()
    spy_metrics[pname] = compute_metrics(sr)


def build_weights(n_stocks=25, target_beta=None, dispersion_gate=False,
                  disp_threshold=0.0, vol_target=None):
    """Build PRISM weights with optional beta targeting and dispersion gating."""
    stocks = close_df.columns.tolist()
    weights = pd.DataFrame(0.0, index=close_df.index, columns=stocks)
    prev_month = None

    mkt_vol = spy_ret.rolling(21).std() * np.sqrt(252)

    for i, date in enumerate(close_df.index):
        month = date.month
        if prev_month is not None and month == prev_month:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue
        prev_month = month

        # Dispersion gate
        if dispersion_gate:
            if date in disp_z.index and not pd.isna(disp_z.loc[date]):
                if disp_z.loc[date] < disp_threshold:
                    # Low dispersion: reduce or skip
                    if i > 0:
                        weights.iloc[i] = weights.iloc[i - 1] * 0.3  # reduce to 30%
                    continue

        if date not in composite.index:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue

        scores = composite.loc[date].dropna()
        if len(scores) < n_stocks:
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            continue

        top = scores.nlargest(n_stocks)

        # Inverse-vol weighting
        stock_rets = close_df.pct_change()
        svol = stock_rets.loc[:date].tail(63).std()
        top_vol = svol.reindex(top.index).fillna(0.2).clip(lower=0.005)
        inv_vol = 1.0 / top_vol
        w = inv_vol / inv_vol.sum()

        # Beta targeting: tilt toward low-beta stocks
        if target_beta is not None and date in beta_df.index:
            betas = beta_df.loc[date, top.index].clip(0.1, 3.0)
            # Adjust weights to hit target beta
            current_beta = (w * betas).sum()
            if current_beta > 0:
                beta_adj = target_beta / current_beta
                # Don't adjust too aggressively
                beta_adj = np.clip(beta_adj, 0.3, 1.5)
                w = w * beta_adj

        # Vol targeting
        if vol_target is not None:
            mv = mkt_vol.loc[date] if date in mkt_vol.index and not pd.isna(mkt_vol.loc[date]) else 0.15
            vs = min(vol_target / max(mv, 0.05), 1.5)
            w = w * vs

        weights.loc[date, top.index] = w.values

    # Cap
    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)

    return weights


def backtest_weights(weights, start, end):
    """Backtest with next-day-open execution."""
    dates = close_df.loc[start:end].index
    stocks = close_df.columns.tolist()
    slip = 5 / 10000
    daily_rets = []
    prev_w = pd.Series(0.0, index=stocks)

    for i, date in enumerate(dates):
        if i == 0:
            daily_rets.append(0.0)
            continue
        prev_date = dates[i - 1]
        target_w = weights.loc[prev_date].reindex(stocks, fill_value=0.0) if prev_date in weights.index else pd.Series(0.0, index=stocks)
        daily_ret = 0.0

        for t in stocks:
            if t not in close_df.columns or t not in open_df.columns:
                continue
            tc = close_df.loc[date, t] if not pd.isna(close_df.loc[date, t]) else 0
            to_ = open_df.loc[date, t] if not pd.isna(open_df.loc[date, t]) else 0
            pc = close_df.loc[prev_date, t] if not pd.isna(close_df.loc[prev_date, t]) else 0
            ow = prev_w.get(t, 0)
            nw = target_w.get(t, 0)

            if ow == nw and ow > 0 and pc > 0:
                daily_ret += ow * (tc / pc - 1)
            elif ow > 0 and nw > 0 and ow != nw:
                if pc > 0: daily_ret += nw * (tc / pc - 1)
                daily_ret -= abs(nw - ow) * slip
            elif ow == 0 and nw > 0:
                if to_ > 0: daily_ret += nw * (tc / to_ - 1)
                daily_ret -= nw * slip
            elif ow > 0 and nw == 0:
                if pc > 0: daily_ret += ow * (to_ / pc - 1)
                daily_ret -= ow * slip

        daily_rets.append(daily_ret)
        prev_w = target_w.copy()

    return pd.Series(daily_rets, index=dates)


experiments = [
    # PRISM baseline (no vol target)
    ("prism_baseline", {"n_stocks": 25}),
    # Beta targeting
    ("beta_0.5", {"n_stocks": 25, "target_beta": 0.5}),
    ("beta_0.7", {"n_stocks": 25, "target_beta": 0.7}),
    ("beta_0.3", {"n_stocks": 25, "target_beta": 0.3}),
    # Dispersion gating
    ("disp_gate_0", {"n_stocks": 25, "dispersion_gate": True, "disp_threshold": 0.0}),
    ("disp_gate_-0.5", {"n_stocks": 25, "dispersion_gate": True, "disp_threshold": -0.5}),
    ("disp_gate_0.5", {"n_stocks": 25, "dispersion_gate": True, "disp_threshold": 0.5}),
    # Beta + dispersion
    ("beta_disp", {"n_stocks": 25, "target_beta": 0.5, "dispersion_gate": True, "disp_threshold": 0.0}),
    # Vol targeting
    ("vol_8pct", {"n_stocks": 25, "vol_target": 0.08}),
    ("vol_10pct", {"n_stocks": 25, "vol_target": 0.10}),
    ("vol_12pct", {"n_stocks": 25, "vol_target": 0.12}),
    # Beta + vol target
    ("beta_vol_8", {"n_stocks": 25, "target_beta": 0.5, "vol_target": 0.08}),
    ("beta_vol_5", {"n_stocks": 25, "target_beta": 0.3, "vol_target": 0.05}),
    # More stocks
    ("prism_35", {"n_stocks": 35}),
    ("prism_50", {"n_stocks": 50}),
]

print(f"\nRunning {len(experiments)} experiments...\n")

all_results = {}
for name, params in experiments:
    print(f"{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    try:
        w = build_weights(**params)
        results = {}
        for pname, start, end in PERIODS:
            rets = backtest_weights(w, start, end)
            m = compute_metrics(rets)
            sm = spy_metrics[pname]
            inv = (w.loc[start:end].sum(axis=1) > 0.01).mean()
            port_beta = 0
            if pname in ["TRAIN", "VALID", "TEST"]:
                r = rets[rets != 0]
                sr = spy_close.loc[start:end].pct_change().reindex(rets.index)
                if len(r) > 100:
                    port_beta = r.cov(sr) / sr.var() if sr.var() > 0 else 0
            print(f"  {pname}: Sh={m['sharpe']:6.3f} CAGR={m['cagr']:7.1%} MDD={m['max_dd']:7.1%} Vol={m['ann_vol']:5.1%} β={port_beta:.2f} | SPY={sm['sharpe']:.3f}")
            results[pname] = m
        all_results[name] = results
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print(f"\n\n{'='*60}")
print("ULTIMATE SUMMARY (sorted by min Sharpe)")
print(f"{'='*60}")
ranked = sorted(all_results.items(),
    key=lambda x: min(x[1].get(p,{}).get("sharpe",-99) for p in ["TRAIN","VALID","TEST"]),
    reverse=True)
print(f"{'Name':<20} {'Train':>8} {'Valid':>8} {'Test':>8} {'MinSh':>8} {'AvgSh':>8}")
for name, periods in ranked:
    t = periods.get("TRAIN",{}).get("sharpe",0)
    v = periods.get("VALID",{}).get("sharpe",0)
    te = periods.get("TEST",{}).get("sharpe",0)
    print(f"{name:<20} {t:>8.3f} {v:>8.3f} {te:>8.3f} {min(t,v,te):>8.3f} {(t+v+te)/3:>8.3f}")
