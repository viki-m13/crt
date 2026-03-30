#!/usr/bin/env python3
"""
Core backtest engine for sector ETF strategy research.
- Next-day-open execution (no look-ahead)
- Transaction costs
- Walk-forward validation
"""
import numpy as np
import pandas as pd

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
BENCHMARK = "SPY"
TX_COST_BPS = 5  # 5 bps per trade (one way)

# Data splits
TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
VALID_START, VALID_END = "2020-04-01", "2022-12-31"
TEST_START, TEST_END = "2023-04-01", "2026-03-15"


def load_sector_data():
    """Load cached data from prepare.py."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from prepare import load_data
    return load_data()


def build_close_df(data, tickers=None):
    """Build aligned close price DataFrame."""
    if tickers is None:
        tickers = [BENCHMARK] + SECTOR_ETFS
    frames = {}
    for t in tickers:
        if t in data:
            frames[t] = data[t]["Close"]
    return pd.DataFrame(frames).dropna(how="all")


def build_open_df(data, tickers=None):
    """Build aligned open price DataFrame."""
    if tickers is None:
        tickers = [BENCHMARK] + SECTOR_ETFS
    frames = {}
    for t in tickers:
        if t in data and "Open" in data[t].columns:
            frames[t] = data[t]["Open"]
    return pd.DataFrame(frames).dropna(how="all")


def backtest_allocation(
    weights_df,  # DataFrame: date x ticker, values are portfolio weights (0-1)
    close_df,
    open_df,
    start, end,
    tx_cost_bps=TX_COST_BPS,
    execution="next_open",  # "next_open" or "same_close"
):
    """
    Backtest a dynamic allocation strategy.

    weights_df: on date T, the target weights decided at T's close.
    execution="next_open": execute at T+1 open (realistic).

    Returns: daily_returns Series, trade_log list
    """
    dates = close_df.loc[start:end].index
    tickers = weights_df.columns.tolist()

    daily_rets = []
    trade_log = []
    prev_weights = pd.Series(0.0, index=tickers)
    slip = tx_cost_bps / 10000

    for i, date in enumerate(dates):
        if date not in weights_df.index:
            daily_rets.append(0.0)
            continue

        if execution == "next_open":
            # Signal from PREVIOUS day's close -> execute at today's open
            if i == 0:
                daily_rets.append(0.0)
                continue

            prev_date = dates[i - 1]
            if prev_date not in weights_df.index:
                daily_rets.append(0.0)
                continue

            target_w = weights_df.loc[prev_date].reindex(tickers, fill_value=0.0)

            # Compute daily return: for positions held from yesterday
            # Yesterday's weights earn today's close-to-close return
            # New positions earn open-to-close return (entered at open)
            # Exited positions earn prev_close-to-open return (exited at open)

            daily_ret = 0.0
            trades_today = 0

            for t in tickers:
                if t not in close_df.columns or t not in open_df.columns:
                    continue
                if date not in close_df.index or date not in open_df.index:
                    continue

                today_close = close_df.loc[date, t] if not pd.isna(close_df.loc[date, t]) else 0
                today_open = open_df.loc[date, t] if not pd.isna(open_df.loc[date, t]) else 0
                prev_close = close_df.loc[prev_date, t] if prev_date in close_df.index and not pd.isna(close_df.loc[prev_date, t]) else 0

                old_w = prev_weights.get(t, 0.0)
                new_w = target_w.get(t, 0.0)

                if old_w == new_w and old_w > 0 and prev_close > 0:
                    # Holding: full close-to-close return
                    daily_ret += old_w * (today_close / prev_close - 1)
                elif old_w > 0 and new_w > 0 and old_w != new_w:
                    # Rebalance: approximate as close-to-close on the continuing portion
                    # plus tx cost on the delta
                    if prev_close > 0:
                        daily_ret += new_w * (today_close / prev_close - 1)
                    delta = abs(new_w - old_w)
                    daily_ret -= delta * slip
                    trades_today += 1
                elif old_w == 0 and new_w > 0:
                    # New entry at open
                    if today_open > 0:
                        daily_ret += new_w * (today_close / today_open - 1)
                    daily_ret -= new_w * slip
                    trades_today += 1
                elif old_w > 0 and new_w == 0:
                    # Exit at open
                    if prev_close > 0:
                        daily_ret += old_w * (today_open / prev_close - 1)
                    daily_ret -= old_w * slip
                    trades_today += 1

            daily_rets.append(daily_ret)
            prev_weights = target_w.copy()

            if trades_today > 0:
                trade_log.append({
                    "date": date,
                    "n_trades": trades_today,
                    "weights": target_w[target_w > 0].to_dict(),
                })
        else:
            # Same-close execution (for comparison, known to be biased)
            target_w = weights_df.loc[date].reindex(tickers, fill_value=0.0)
            if i == 0:
                daily_rets.append(0.0)
                prev_weights = target_w.copy()
                continue
            prev_date = dates[i - 1]
            daily_ret = 0.0
            for t in tickers:
                if t not in close_df.columns:
                    continue
                w = target_w.get(t, 0.0)
                prev_c = close_df.loc[prev_date, t] if prev_date in close_df.index else 0
                curr_c = close_df.loc[date, t]
                if w > 0 and prev_c > 0 and not pd.isna(prev_c) and not pd.isna(curr_c):
                    daily_ret += w * (curr_c / prev_c - 1)
            # tx costs
            delta = (target_w - prev_weights).abs().sum()
            daily_ret -= delta * slip
            daily_rets.append(daily_ret)
            prev_weights = target_w.copy()

    return pd.Series(daily_rets, index=dates), trade_log


def compute_metrics(rets, rf_annual=0.02):
    """Compute strategy metrics from daily returns."""
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0, "calmar": 0,
                "ann_vol": 0, "total_ret": 0, "time_in_market": 0, "n_days": 0}

    excess = rets - rf_annual / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    n_years = len(rets) / 252
    cagr = (1 + total) ** (1 / max(n_years, 0.01)) - 1
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    downside = excess[excess < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    ann_vol = rets.std() * np.sqrt(252)
    time_in = (rets != 0).sum() / len(rets)

    return {
        "sharpe": round(float(sharpe), 3),
        "cagr": round(float(cagr), 4),
        "max_dd": round(float(max_dd), 4),
        "sortino": round(float(sortino), 3),
        "calmar": round(float(calmar), 3),
        "ann_vol": round(float(ann_vol), 4),
        "total_ret": round(float(total), 4),
        "time_in_market": round(float(time_in), 3),
        "n_days": len(rets),
    }


def spy_metrics(close_df, start, end, rf_annual=0.02):
    """SPY buy-and-hold metrics for comparison."""
    spy = close_df.loc[start:end, BENCHMARK].dropna()
    rets = spy.pct_change().dropna()
    return compute_metrics(rets, rf_annual)


def print_comparison(name, strat_metrics, spy_m):
    """Print strategy vs SPY comparison."""
    print(f"\n  {name}:")
    print(f"    {'':18} {'Strategy':>10} {'SPY B&H':>10}")
    print(f"    {'-'*38}")
    print(f"    {'Sharpe':<18} {strat_metrics['sharpe']:>10.3f} {spy_m['sharpe']:>10.3f}")
    print(f"    {'CAGR':<18} {strat_metrics['cagr']:>10.1%} {spy_m['cagr']:>10.1%}")
    print(f"    {'Max DD':<18} {strat_metrics['max_dd']:>10.1%} {spy_m['max_dd']:>10.1%}")
    print(f"    {'Sortino':<18} {strat_metrics['sortino']:>10.3f}")
    print(f"    {'Calmar':<18} {strat_metrics['calmar']:>10.3f}")
    print(f"    {'Ann Vol':<18} {strat_metrics['ann_vol']:>10.1%}")
    print(f"    {'Time in Market':<18} {strat_metrics['time_in_market']:>10.1%}")
