#!/usr/bin/env python3
"""
High-frequency equity reversal strategies targeting 3+ Sharpe.

Key math: Annual_Sharpe = per_trade_IR * sqrt(N_independent_trades)
If IR=0.15 and N=500: Sharpe = 0.15*sqrt(500) = 3.35

Approach: short-term REVERSAL (buy pullbacks in strong stocks)
- Stocks that drop 1-5 days but are in positive trends bounce back
- Many concurrent positions = many independent bets
- Weekly rebalancing to keep costs manageable
"""
import numpy as np
import pandas as pd

BENCHMARK = "SPY"
TX_COST_BPS = 5

STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM",
    "CSCO", "VZ", "ADBE", "CRM", "CMCSA", "PFE", "NFLX", "INTC",
    "ABT", "KO", "PEP", "TMO", "MRK", "ABBV", "COST", "AVGO", "ACN",
    "CVX", "LLY", "MCD", "WMT", "DHR", "TXN", "NEE", "BMY", "QCOM",
    "UNP", "HON", "LOW", "AMGN", "LIN", "RTX",
    "ORCL", "PM", "UPS", "CAT", "GS", "MS", "BLK", "ISRG", "MDT",
    "DE", "ADP", "GILD", "BKNG", "SYK", "MMM", "GE", "CB", "CI",
    "SO", "DUK", "MO", "CL", "ITW", "FIS", "USB", "SCHW", "PNC",
    "CME", "AON", "ICE", "NSC", "EMR", "APD", "SHW", "ETN", "ECL",
    "WM", "ROP", "LRCX", "KLAC", "AMAT", "MCHP", "SNPS", "CDNS",
    "FTNT", "PANW", "NOW", "WDAY",
]


def load_data():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from prepare import load_data as _load
    return _load()


def build_dfs(data, tickers=None):
    if tickers is None:
        tickers = [t for t in STOCKS if t in data]
    close = pd.DataFrame({t: data[t]["Close"] for t in tickers if t in data}).dropna(how="all")
    opn = pd.DataFrame({t: data[t]["Open"] for t in tickers if t in data and "Open" in data[t].columns}).dropna(how="all")
    return close, opn, [t for t in tickers if t in close.columns]


def compute_metrics(rets, rf=0.02):
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0, "ann_vol": 0,
                "calmar": 0, "total_ret": 0, "n_days": 0}
    excess = rets - rf / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    n_years = len(rets) / 252
    cagr = (1 + total) ** (1 / max(n_years, 0.01)) - 1
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    down = excess[excess < 0]
    sortino = excess.mean() / down.std() * np.sqrt(252) if len(down) > 0 and down.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    ann_vol = rets.std() * np.sqrt(252)
    return {
        "sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
        "max_dd": round(float(max_dd), 4), "sortino": round(float(sortino), 3),
        "calmar": round(float(calmar), 3), "ann_vol": round(float(ann_vol), 4),
        "total_ret": round(float(total), 4), "n_days": len(rets),
    }


def backtest_shorthold(close_df, open_df, signal_fn, start, end, params=None):
    """
    Generic backtest for short-hold strategies.

    signal_fn(close_df, open_df, date, params) -> dict of {stock: weight}
    Signal computed at day T close -> execute at day T+1 open.
    Position held until signal says to exit (or max_hold_days reached).

    Returns daily_returns Series.
    """
    p = {
        "max_hold_days": 5,
        "n_positions": 20,
        "equal_weight": True,
    }
    if params:
        p.update(params)

    dates = close_df.loc[start:end].index
    stocks = close_df.columns.tolist()
    slip = TX_COST_BPS / 10000

    # Track active positions: {stock: {"entry_date": date, "entry_price": px, "weight": w, "age": days}}
    positions = {}
    daily_rets = []

    for i, date in enumerate(dates):
        if i == 0:
            daily_rets.append(0.0)
            continue

        prev_date = dates[i - 1]
        daily_ret = 0.0

        # 1. Compute returns for existing positions (close-to-close for holds)
        exits = []
        for stock, pos in positions.items():
            if stock not in close_df.columns:
                continue
            pc = close_df.loc[prev_date, stock] if prev_date in close_df.index else 0
            tc = close_df.loc[date, stock] if date in close_df.index else 0
            to_ = open_df.loc[date, stock] if date in open_df.index and stock in open_df.columns else 0

            if pos["age"] >= p["max_hold_days"]:
                # EXIT at today's open (last day: prev_close -> today_open)
                if pc > 0 and to_ > 0:
                    daily_ret += pos["weight"] * (to_ / pc - 1)
                daily_ret -= pos["weight"] * slip
                exits.append(stock)
            else:
                # HOLD: close-to-close
                if pc > 0 and tc > 0:
                    daily_ret += pos["weight"] * (tc / pc - 1)
                pos["age"] += 1

        for stock in exits:
            del positions[stock]

        # 2. Generate new signals (from PREVIOUS day's close) -> execute at today's open
        new_targets = signal_fn(close_df, open_df, prev_date, p)
        if new_targets:
            # Filter out already-held stocks
            new_targets = {s: w for s, w in new_targets.items() if s not in positions}
            # Limit to available slots
            available_slots = p["n_positions"] - len(positions)
            if available_slots > 0:
                sorted_targets = sorted(new_targets.items(), key=lambda x: x[1], reverse=True)[:available_slots]
                if p["equal_weight"] and sorted_targets:
                    w = 1.0 / p["n_positions"]
                    for stock, _ in sorted_targets:
                        to_ = open_df.loc[date, stock] if date in open_df.index and stock in open_df.columns else 0
                        tc = close_df.loc[date, stock] if date in close_df.index else 0
                        if to_ > 0 and tc > 0:
                            # Entry: open-to-close return
                            daily_ret += w * (tc / to_ - 1)
                            daily_ret -= w * slip
                            positions[stock] = {"entry_date": date, "entry_price": to_, "weight": w, "age": 1}

        daily_rets.append(daily_ret)

    return pd.Series(daily_rets, index=dates)


# ============================================================
# SIGNAL FUNCTIONS
# ============================================================

def signal_pullback_in_trend(close_df, open_df, date, params):
    """
    Buy stocks that pulled back 5 days but are in positive 63-day trend.
    This is the classic "buy the dip in a winner" pattern.
    """
    idx = close_df.index.get_loc(date) if date in close_df.index else -1
    if idx < 126:
        return {}

    scores = {}
    for stock in close_df.columns:
        try:
            c = close_df[stock]
            # Short-term reversal: 5-day return (negative = pulled back)
            ret_5d = c.iloc[idx] / c.iloc[idx - 5] - 1
            # Long-term trend: 63-day return (positive = trending up)
            ret_63d = c.iloc[idx] / c.iloc[idx - 63] - 1
            # Quality: above 50-day SMA
            sma50 = c.iloc[max(0, idx-49):idx+1].mean()
            above_sma = c.iloc[idx] > sma50

            # ENTRY CRITERIA:
            # 1. Pulled back at least 2% in last 5 days
            # 2. Positive 63-day trend
            # 3. Above 50-day SMA (trend intact)
            if ret_5d < -0.02 and ret_63d > 0 and above_sma:
                # Score: magnitude of pullback (bigger dip = higher conviction)
                scores[stock] = -ret_5d  # positive score = bigger pullback
        except Exception:
            continue

    return scores


def signal_reversal_quality(close_df, open_df, date, params):
    """
    Buy quality stocks (low vol, persistent momentum) that had a bad week.
    The "quality reversal" is a higher-Sharpe variant of pure reversal.
    """
    idx = close_df.index.get_loc(date) if date in close_df.index else -1
    if idx < 126:
        return {}

    rets = close_df.pct_change()
    scores = {}
    for stock in close_df.columns:
        try:
            c = close_df[stock]
            r = rets[stock]

            ret_5d = c.iloc[idx] / c.iloc[idx - 5] - 1
            ret_63d = c.iloc[idx] / c.iloc[idx - 63] - 1

            # Quality: low volatility
            vol = r.iloc[max(0,idx-62):idx+1].std() * np.sqrt(252)
            # Quality: momentum persistence (% of 21d windows with positive ret)
            ret_21d = r.iloc[max(0,idx-125):idx+1].rolling(21).sum()
            persistence = (ret_21d > 0).mean()

            # ENTRY:
            # 1. 5-day pullback > 1.5%
            # 2. Positive long-term trend
            # 3. Low-to-moderate volatility (< 40% annualized)
            # 4. High momentum persistence (> 50%)
            if ret_5d < -0.015 and ret_63d > 0 and vol < 0.40 and persistence > 0.5:
                # Score: pullback magnitude * quality
                quality = (1 - vol/0.40) * persistence
                scores[stock] = (-ret_5d) * quality
        except Exception:
            continue

    return scores


def signal_gap_recovery(close_df, open_df, date, params):
    """
    Buy stocks that gapped down (open << prev close) but are in an uptrend.
    Gap-downs in trending stocks frequently fill within 1-5 days.
    """
    idx = close_df.index.get_loc(date) if date in close_df.index else -1
    if idx < 126:
        return {}

    scores = {}
    for stock in close_df.columns:
        try:
            if stock not in open_df.columns:
                continue
            c = close_df[stock]
            o = open_df[stock]

            # Gap: today's open vs yesterday's close
            prev_close = c.iloc[idx - 1] if idx > 0 else c.iloc[idx]
            today_close = c.iloc[idx]
            gap = (today_close / prev_close - 1) if prev_close > 0 else 0

            # Trend
            ret_63d = c.iloc[idx] / c.iloc[idx - 63] - 1
            sma50 = c.iloc[max(0, idx-49):idx+1].mean()

            # ENTRY: gap down > 1.5%, in uptrend
            if gap < -0.015 and ret_63d > 0 and c.iloc[idx] > sma50 * 0.95:
                scores[stock] = -gap
        except Exception:
            continue

    return scores


def signal_oversold_bounce(close_df, open_df, date, params):
    """
    RSI-based: buy stocks with RSI < 30 that are in positive 126-day trend.
    Classic oversold bounce in quality names.
    """
    idx = close_df.index.get_loc(date) if date in close_df.index else -1
    if idx < 126:
        return {}

    scores = {}
    for stock in close_df.columns:
        try:
            c = close_df[stock]
            delta = c.diff()
            window = delta.iloc[max(0,idx-13):idx+1]
            gain = window.clip(lower=0).mean()
            loss = (-window.clip(upper=0)).mean()
            if loss == 0:
                continue
            rs = gain / loss
            rsi = 100 - 100 / (1 + rs)

            ret_126d = c.iloc[idx] / c.iloc[idx - 126] - 1

            if rsi < 30 and ret_126d > 0:
                scores[stock] = (30 - rsi) / 30  # lower RSI = higher score
        except Exception:
            continue

    return scores


def signal_combined_reversal(close_df, open_df, date, params):
    """
    COMBINED: Multi-signal reversal with quality filter.
    Fires when MULTIPLE reversal signals agree on the same stock.
    Higher agreement = higher conviction = better IR per trade.
    """
    idx = close_df.index.get_loc(date) if date in close_df.index else -1
    if idx < 126:
        return {}

    rets = close_df.pct_change()
    scores = {}

    for stock in close_df.columns:
        try:
            c = close_df[stock]
            r = rets[stock]

            ret_1d = c.iloc[idx] / c.iloc[idx - 1] - 1 if idx > 0 else 0
            ret_5d = c.iloc[idx] / c.iloc[idx - 5] - 1
            ret_63d = c.iloc[idx] / c.iloc[idx - 63] - 1
            ret_126d = c.iloc[idx] / c.iloc[idx - 126] - 1

            sma50 = c.iloc[max(0, idx-49):idx+1].mean()
            vol = r.iloc[max(0,idx-62):idx+1].std() * np.sqrt(252)

            # RSI
            delta = c.diff()
            window = delta.iloc[max(0,idx-13):idx+1]
            gain = window.clip(lower=0).mean()
            loss = (-window.clip(upper=0)).mean()
            rsi = 100 - 100 / (1 + gain / max(loss, 1e-10))

            # Count reversal signals firing
            signals = 0
            if ret_5d < -0.02: signals += 1       # 5-day pullback
            if ret_1d < -0.01: signals += 1        # 1-day drop
            if rsi < 35: signals += 1              # RSI oversold
            if vol < 0.35: signals += 1            # quality (low vol)

            # REQUIRE: long-term trend + above SMA + at least 2 reversal signals
            if ret_63d > 0 and ret_126d > 0 and c.iloc[idx] > sma50 * 0.97 and signals >= 2:
                scores[stock] = signals * (-ret_5d) * (1 - vol/0.50)
        except Exception:
            continue

    return scores


ALL_SIGNALS = {
    "pullback_in_trend": signal_pullback_in_trend,
    "reversal_quality": signal_reversal_quality,
    "gap_recovery": signal_gap_recovery,
    "oversold_bounce": signal_oversold_bounce,
    "combined_reversal": signal_combined_reversal,
}
