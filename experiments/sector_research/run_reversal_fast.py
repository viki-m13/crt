#!/usr/bin/env python3
"""
Fast vectorized reversal backtests — no per-stock-per-day loops.
Uses rolling cohorts: each week, buy top N stocks by reversal score,
hold for H days, then exit. Overlapping cohorts for smooth returns.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
from experiments.sector_research.reversal_engine import load_data, build_dfs, compute_metrics, BENCHMARK

print("Loading data...")
data = load_data()
close_df, open_df, avail = build_dfs(data)
spy_close = data[BENCHMARK]["Close"]
print(f"Stocks: {len(avail)}")

rets_1d = close_df.pct_change()  # daily returns
rets_5d = close_df.pct_change(5)
rets_63d = close_df.pct_change(63)
rets_126d = close_df.pct_change(126)
sma50 = close_df.rolling(50).mean()
vol63 = rets_1d.rolling(63).std() * np.sqrt(252)

# RSI (vectorized)
delta = close_df.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rsi = 100 - 100 / (1 + gain / loss.clip(lower=1e-10))

# Quality: momentum persistence
pos_21d = (rets_1d.rolling(21).sum() > 0).rolling(126).mean()


def score_combined_vectorized(date):
    """Score all stocks on a given date using vectorized lookups."""
    if date not in close_df.index:
        return pd.Series(dtype=float)

    r5 = rets_5d.loc[date]
    r63 = rets_63d.loc[date]
    r126 = rets_126d.loc[date]
    r1 = rets_1d.loc[date]
    s50 = sma50.loc[date]
    v = vol63.loc[date]
    r = rsi.loc[date]
    price = close_df.loc[date]

    # Signals
    sig_pullback = (r5 < -0.02).astype(int)
    sig_drop = (r1 < -0.01).astype(int)
    sig_rsi = (r < 35).astype(int)
    sig_quality = (v < 0.35).astype(int)
    sig_count = sig_pullback + sig_drop + sig_rsi + sig_quality

    # Trend filter
    trend_ok = (r63 > 0) & (r126 > 0) & (price > s50 * 0.97)

    # Must have at least 2 signals + trend
    eligible = trend_ok & (sig_count >= 2)

    score = eligible.astype(float) * sig_count * (-r5).clip(lower=0) * (1 - v.clip(upper=0.5) / 0.5)
    return score[score > 0].dropna()


def backtest_cohort_strategy(start, end, hold_days=5, n_positions=20, rebal_every=5):
    """
    Cohort-based backtest: every `rebal_every` days, create a new cohort
    of `n_positions` stocks. Hold for `hold_days` days.
    Multiple cohorts overlap for smoother returns.

    Returns daily portfolio return series.
    """
    dates = close_df.loc[start:end].index
    slip = 5 / 10000

    # Pre-score all dates
    all_scores = {}
    for date in dates:
        scores = score_combined_vectorized(date)
        if len(scores) > 0:
            all_scores[date] = scores

    # Track active cohorts: list of {"stocks": [...], "start": date, "weights": [...]}
    cohorts = []
    daily_rets = []

    for i, date in enumerate(dates):
        if i == 0:
            daily_rets.append(0.0)
            continue

        prev_date = dates[i - 1]
        daily_ret = 0.0

        # 1. Remove expired cohorts & compute their exit return
        active = []
        for coh in cohorts:
            age = (date - coh["start"]).days
            if age > hold_days:
                # Exit at today's open
                for stock, w in zip(coh["stocks"], coh["weights"]):
                    if stock in close_df.columns and stock in open_df.columns:
                        pc = close_df.loc[prev_date, stock] if prev_date in close_df.index else 0
                        to_ = open_df.loc[date, stock] if date in open_df.index else 0
                        if pc > 0 and to_ > 0 and not pd.isna(pc) and not pd.isna(to_):
                            daily_ret += w * (to_ / pc - 1)
                        daily_ret -= w * slip
            else:
                active.append(coh)
        cohorts = active

        # 2. Compute returns for held cohorts (close-to-close)
        for coh in cohorts:
            for stock, w in zip(coh["stocks"], coh["weights"]):
                if stock in close_df.columns:
                    pc = close_df.loc[prev_date, stock] if prev_date in close_df.index else 0
                    tc = close_df.loc[date, stock] if date in close_df.index else 0
                    if pc > 0 and tc > 0 and not pd.isna(pc) and not pd.isna(tc):
                        daily_ret += w * (tc / pc - 1)

        # 3. Create new cohort if it's rebalance day
        if i % rebal_every == 0:
            # Signal from previous day (no look-ahead)
            sig_date = prev_date
            if sig_date in all_scores:
                scores = all_scores[sig_date]
                if len(scores) >= 3:
                    top = scores.nlargest(min(n_positions, len(scores)))
                    n_active_cohorts = len(cohorts) + 1
                    coh_weight = 1.0 / n_active_cohorts  # Share capital equally
                    stock_w = coh_weight / len(top)

                    new_stocks = top.index.tolist()
                    new_weights = [stock_w] * len(new_stocks)

                    # Recompute all cohort weights to keep total = 1.0
                    total_coh = len(cohorts) + 1
                    for c in cohorts:
                        c["weights"] = [1.0 / (total_coh * len(c["stocks"]))] * len(c["stocks"])

                    cohorts.append({"stocks": new_stocks, "weights": new_weights, "start": date})

                    # Entry cost
                    for stock in new_stocks:
                        if stock in open_df.columns and date in open_df.index:
                            daily_ret -= stock_w * slip

        daily_rets.append(daily_ret)

    return pd.Series(daily_rets, index=dates)


def backtest_simple_weekly(start, end, n_positions=20, hold_days=5):
    """
    Simpler approach: each Friday (or every 5 trading days),
    rebalance into top N reversal stocks. Non-overlapping cohorts.
    """
    dates = close_df.loc[start:end].index
    slip = 5 / 10000

    stocks = close_df.columns.tolist()
    weights = pd.DataFrame(0.0, index=dates, columns=stocks)
    prev_rebal = None
    days_since_rebal = 999

    for i, date in enumerate(dates):
        days_since_rebal += 1
        if days_since_rebal >= hold_days:
            # Rebalance
            if i > 0:
                sig_date = dates[i - 1]
                scores = score_combined_vectorized(sig_date)
                if len(scores) >= 3:
                    top = scores.nlargest(min(n_positions, len(scores)))
                    w = 1.0 / len(top)
                    weights.loc[date, top.index] = w
                    days_since_rebal = 0
                    continue
            weights.loc[date] = 0
        else:
            # Hold previous
            if i > 0:
                weights.iloc[i] = weights.iloc[i - 1]

    # Compute returns with execution model
    daily_rets = []
    prev_w = pd.Series(0.0, index=stocks)

    for i, date in enumerate(dates):
        if i == 0:
            daily_rets.append(0.0)
            continue

        prev_date = dates[i - 1]
        target_w = weights.loc[prev_date] if prev_date in weights.index else pd.Series(0.0, index=stocks)
        daily_ret = 0.0

        for t in stocks:
            if t not in close_df.columns or t not in open_df.columns:
                continue
            tc = close_df.loc[date, t] if date in close_df.index and not pd.isna(close_df.loc[date, t]) else 0
            to_ = open_df.loc[date, t] if date in open_df.index and not pd.isna(open_df.loc[date, t]) else 0
            pc = close_df.loc[prev_date, t] if prev_date in close_df.index and not pd.isna(close_df.loc[prev_date, t]) else 0
            ow = prev_w.get(t, 0.0)
            nw = target_w.get(t, 0.0)

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


PERIODS = [
    ("TRAIN", "2010-01-01", "2019-12-31"),
    ("VALID", "2020-04-01", "2022-12-31"),
    ("TEST", "2023-04-01", "2026-03-15"),
]

spy_metrics = {}
for pname, start, end in PERIODS:
    sr = spy_close.loc[start:end].pct_change().dropna()
    spy_metrics[pname] = compute_metrics(sr)

experiments = [
    ("simple_5d_20", "simple", {"hold_days": 5, "n_positions": 20}),
    ("simple_5d_30", "simple", {"hold_days": 5, "n_positions": 30}),
    ("simple_3d_20", "simple", {"hold_days": 3, "n_positions": 20}),
    ("simple_5d_15", "simple", {"hold_days": 5, "n_positions": 15}),
    ("simple_10d_20", "simple", {"hold_days": 10, "n_positions": 20}),
    ("simple_5d_40", "simple", {"hold_days": 5, "n_positions": 40}),
    ("cohort_5d_r3", "cohort", {"hold_days": 5, "n_positions": 15, "rebal_every": 3}),
    ("cohort_5d_r1", "cohort", {"hold_days": 5, "n_positions": 10, "rebal_every": 1}),
    ("cohort_3d_r1", "cohort", {"hold_days": 3, "n_positions": 10, "rebal_every": 1}),
    ("cohort_5d_r2", "cohort", {"hold_days": 5, "n_positions": 15, "rebal_every": 2}),
]

all_results = {}
for name, method, params in experiments:
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    try:
        results = {}
        for pname, start, end in PERIODS:
            if method == "simple":
                rets = backtest_simple_weekly(start, end, params["n_positions"], params["hold_days"])
            else:
                rets = backtest_cohort_strategy(start, end, params["hold_days"], params["n_positions"], params["rebal_every"])
            m = compute_metrics(rets)
            sm = spy_metrics[pname]
            invested = (rets != 0).mean()
            print(f"  {pname}: Sharpe={m['sharpe']:6.3f} CAGR={m['cagr']:7.1%} MDD={m['max_dd']:7.1%} Vol={m['ann_vol']:5.1%} Inv={invested:4.0%} | SPY={sm['sharpe']:.3f}")
            results[pname] = m
        all_results[name] = results
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print(f"\n\n{'='*60}")
print("FAST REVERSAL SUMMARY (sorted by min Sharpe)")
print(f"{'='*60}")
ranked = sorted(all_results.items(),
    key=lambda x: min(x[1].get(p,{}).get("sharpe",-99) for p in ["TRAIN","VALID","TEST"]),
    reverse=True)
print(f"{'Name':<20} {'Train':>8} {'Valid':>8} {'Test':>8} {'MinSh':>8}")
for name, periods in ranked:
    t = periods.get("TRAIN",{}).get("sharpe",0)
    v = periods.get("VALID",{}).get("sharpe",0)
    te = periods.get("TEST",{}).get("sharpe",0)
    print(f"{name:<20} {t:>8.3f} {v:>8.3f} {te:>8.3f} {min(t,v,te):>8.3f}")
