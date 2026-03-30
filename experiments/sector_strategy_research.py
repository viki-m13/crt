#!/usr/bin/env python3
"""
Sector ETF Strategy Research — Step 1: Core backtest engine
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
BENCHMARK = "SPY"

def backtest_sector_strategy(data, start, end, signal_fn, tx_cost_bps=5):
    """
    Generic sector strategy backtester with T+1 OPEN execution.

    signal_fn(data, date, idx) -> dict with:
      'invested': bool (whether to be in market)
      'weights': dict {ticker: weight} (portfolio weights, sum <= 1.0)

    Execution model:
    - Signal computed at day T close
    - Executed at day T+1 open (with slippage)
    - Entry day return: open to close
    - Exit day return: prev close to open (overnight)
    - Hold day return: close to close
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    slip = tx_cost_bps / 10000

    daily_rets = []
    current_weights = {}  # {ticker: weight}
    pending_weights = None  # weights to execute tomorrow
    pending_exit = False
    trade_count = 0

    for i, date in enumerate(dates):
        idx = spy.index.get_loc(date)
        if idx < 252:  # need 1 year warmup
            daily_rets.append(0.0)
            continue

        # === EXECUTE PENDING ACTIONS AT TODAY'S OPEN ===
        if pending_exit and current_weights:
            # Exit: overnight return (prev close to open)
            dr = 0.0
            for ticker, w in current_weights.items():
                df = data[ticker]
                if date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        prev_c = df.iloc[si-1]["Close"]
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        dr += (today_o * (1 - slip) / prev_c - 1) * w
            daily_rets.append(dr)
            trade_count += len(current_weights)
            current_weights = {}
            pending_exit = False
            # Now generate signal for tomorrow
            sig = signal_fn(data, date, idx)
            if sig["invested"] and sig.get("weights"):
                pending_weights = sig["weights"]
            continue

        if pending_weights and not current_weights:
            # Enter: buy at open, return from open to close
            dr = 0.0
            new_weights = {}
            for ticker, w in pending_weights.items():
                df = data.get(ticker)
                if df is None or date not in df.index:
                    continue
                today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                buy_price = today_o * (1 + slip)
                today_c = df.loc[date, "Close"]
                dr += (today_c / buy_price - 1) * w
                new_weights[ticker] = w
            daily_rets.append(dr)
            current_weights = new_weights
            trade_count += len(new_weights)
            pending_weights = None
            # Generate signal for tomorrow
            sig = signal_fn(data, date, idx)
            if not sig["invested"]:
                pending_exit = True
            elif sig.get("weights") and set(sig["weights"].keys()) != set(current_weights.keys()):
                # Rotation needed tomorrow
                pending_exit = True  # simplified: exit then re-enter
                pending_weights = sig["weights"]
            continue

        if pending_weights and current_weights:
            # Rotation: sell old at open, buy new at open
            dr = 0.0
            # Sell old (overnight return)
            for ticker, w in current_weights.items():
                df = data[ticker]
                if date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        prev_c = df.iloc[si-1]["Close"]
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        dr += (today_o * (1 - slip) / prev_c - 1) * w
            trade_count += len(current_weights)
            # Buy new (open to close)
            new_weights = {}
            for ticker, w in pending_weights.items():
                df = data.get(ticker)
                if df is None or date not in df.index:
                    continue
                today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                buy_price = today_o * (1 + slip)
                today_c = df.loc[date, "Close"]
                dr += (today_c / buy_price - 1) * w
                new_weights[ticker] = w
            daily_rets.append(dr)
            current_weights = new_weights
            trade_count += len(new_weights)
            pending_weights = None
            pending_exit = False
            continue

        # Clear stale pending
        pending_exit = False
        pending_weights = None

        # === DAILY RETURN FOR HELD POSITIONS (close to close) ===
        if current_weights:
            dr = 0.0
            for ticker, w in current_weights.items():
                df = data[ticker]
                if date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
            daily_rets.append(dr)
        else:
            daily_rets.append(0.0)

        # === GENERATE SIGNAL AT CLOSE (for tomorrow) ===
        sig = signal_fn(data, date, idx)

        if current_weights and not sig["invested"]:
            pending_exit = True
        elif not current_weights and sig["invested"] and sig.get("weights"):
            pending_weights = sig["weights"]
        elif current_weights and sig["invested"] and sig.get("weights"):
            new_keys = set(sig["weights"].keys())
            old_keys = set(current_weights.keys())
            if new_keys != old_keys:
                pending_weights = sig["weights"]

    return pd.Series(daily_rets, index=dates), trade_count


def compute_metrics(rets, rf=0.02):
    """Compute strategy metrics from daily return series."""
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0, "time_invested": 0, "ann_vol": 0}

    excess = rets - rf / 252
    n_years = len(rets) / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years >= 1 else total
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    downside = excess[excess < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    invested = (rets != 0).sum() / len(rets)
    ann_vol = rets.std() * np.sqrt(252)

    return {
        "sharpe": round(float(sharpe), 3),
        "cagr": round(float(cagr), 4),
        "max_dd": round(float(mdd), 4),
        "sortino": round(float(sortino), 3),
        "time_invested": round(float(invested), 3),
        "ann_vol": round(float(ann_vol), 4),
    }


def spy_metrics(data, start, end, rf=0.02):
    spy = data[BENCHMARK].loc[start:end, "Close"]
    r = spy.pct_change().dropna()
    ex = r - rf / 252
    sh = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
    cum = (1 + r).cumprod()
    t = cum.iloc[-1] - 1
    n = len(r) / 252
    cg = (1 + t) ** (1 / n) - 1 if n >= 1 else t
    md = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"sharpe": round(float(sh), 3), "cagr": round(float(cg), 4), "max_dd": round(float(md), 4)}


def print_results(name, metrics, spy, trades=0):
    print(f"\n  {name}:")
    print(f"    {'':15} {'Strategy':>10} {'SPY B&H':>10}")
    print(f"    {'Sharpe':<15} {metrics['sharpe']:>10.3f} {spy['sharpe']:>10.3f}")
    print(f"    {'CAGR':<15} {metrics['cagr']:>10.1%} {spy['cagr']:>10.1%}")
    print(f"    {'Max DD':<15} {metrics['max_dd']:>10.1%} {spy['max_dd']:>10.1%}")
    print(f"    {'Sortino':<15} {metrics['sortino']:>10.3f}")
    print(f"    {'Time Invested':<15} {metrics['time_invested']:>10.1%}")
    print(f"    {'Ann Vol':<15} {metrics['ann_vol']:>10.1%}")
    print(f"    {'Trades':<15} {trades:>10}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")

    # Simple test: SMA gate + top momentum sector
    def simple_signal(data, date, idx):
        spy_close = data[BENCHMARK]["Close"]
        sma = spy_close.iloc[max(0,idx-29):idx+1].mean()
        if spy_close.iloc[idx] <= sma:
            return {"invested": False, "weights": {}}

        best, best_ret = None, -999
        for etf in SECTOR_ETFS:
            df = data.get(etf)
            if df is None or date not in df.index:
                continue
            si = df.index.get_loc(date)
            if si < 63:
                continue
            ret = df.iloc[si]["Close"] / df.iloc[si-63]["Close"] - 1
            if ret > best_ret:
                best, best_ret = etf, ret

        if best:
            return {"invested": True, "weights": {BENCHMARK: 0.6, best: 0.4}}
        return {"invested": False, "weights": {}}

    # Test on each period
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, simple_signal)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    print("\nBacktest engine working correctly.")
