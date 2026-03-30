#!/usr/bin/env python3
"""
Research: Stock-level strategies for higher alpha.
Individual stock momentum is much stronger than sector momentum.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END, UNIVERSE

BENCHMARK = "SPY"
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
SAFE_HAVENS = ["TLT", "GLD", "IEF"]
NON_STOCKS = set(SECTOR_ETFS + SAFE_HAVENS + ["SPY", "QQQ", "IWM", "DIA", "HYG", "SLV", "USO"])

def get_stock_universe(data):
    """Get individual stocks (exclude ETFs)."""
    return [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]

def backtest_stock_strategy(data, start, end, signal_fn, tx_cost_bps=5):
    """
    Backtest with T+1 OPEN execution. Same as sector engine but supports larger portfolios.
    signal_fn(data, date, idx) -> dict with:
      'weights': dict {ticker: weight}
    Always fully invested.
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    slip = tx_cost_bps / 10000

    daily_rets = []
    current_weights = {}
    pending_weights = None
    trade_count = 0

    for i, date in enumerate(dates):
        idx = spy.index.get_loc(date)
        if idx < 252:
            daily_rets.append(0.0)
            continue

        # === EXECUTE PENDING REBALANCE AT OPEN ===
        if pending_weights is not None:
            dr = 0.0
            # Close old positions (overnight return: prev close to open)
            for ticker, w in current_weights.items():
                if ticker not in pending_weights or abs(pending_weights.get(ticker, 0) - w) > 0.001:
                    df = data.get(ticker)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            prev_c = df.iloc[si-1]["Close"]
                            today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                            dr += (today_o * (1 - slip) / prev_c - 1) * w
                            trade_count += 1
                    # else: position lost, skip
                else:
                    # Position unchanged — close to close
                    df = data.get(ticker)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w

            # Open new positions (open to close)
            for ticker, w in pending_weights.items():
                if ticker not in current_weights or abs(current_weights.get(ticker, 0) - w) > 0.001:
                    df = data.get(ticker)
                    if df is not None and date in df.index:
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        buy_price = today_o * (1 + slip)
                        today_c = df.loc[date, "Close"]
                        if buy_price > 0:
                            dr += (today_c / buy_price - 1) * w
                            trade_count += 1

            daily_rets.append(dr)
            current_weights = pending_weights
            pending_weights = None

            # Generate signal for next day
            sig = signal_fn(data, date, idx)
            new_w = sig.get("weights", {})
            if set(new_w.keys()) != set(current_weights.keys()):
                pending_weights = new_w
            else:
                # Check if any weight changed significantly
                changed = any(abs(new_w.get(k, 0) - current_weights.get(k, 0)) > 0.01
                            for k in set(new_w.keys()) | set(current_weights.keys()))
                if changed:
                    pending_weights = new_w
            continue

        # === DAILY RETURN FOR HELD POSITIONS ===
        if current_weights:
            dr = 0.0
            for ticker, w in current_weights.items():
                df = data.get(ticker)
                if df is not None and date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
            daily_rets.append(dr)
        else:
            daily_rets.append(0.0)

        # === GENERATE SIGNAL AT CLOSE ===
        sig = signal_fn(data, date, idx)
        new_w = sig.get("weights", {})

        if not current_weights and new_w:
            pending_weights = new_w
        elif current_weights and new_w:
            if set(new_w.keys()) != set(current_weights.keys()):
                pending_weights = new_w
            else:
                changed = any(abs(new_w.get(k, 0) - current_weights.get(k, 0)) > 0.01
                            for k in set(new_w.keys()) | set(current_weights.keys()))
                if changed:
                    pending_weights = new_w

    return pd.Series(daily_rets, index=dates), trade_count


def compute_metrics(rets, rf=0.02):
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0, "ann_vol": 0, "time_invested": 0}
    excess = rets - rf / 252
    n_years = len(rets) / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years >= 1 else total
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    downside = excess[excess < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    invested = (rets != 0).sum() / len(rets)
    return {
        "sharpe": round(float(sharpe), 3),
        "cagr": round(float(cagr), 4),
        "max_dd": round(float(mdd), 4),
        "sortino": round(float(sortino), 3),
        "ann_vol": round(float(rets.std() * np.sqrt(252)), 4),
        "time_invested": round(float(invested), 3),
    }


def spy_metrics(data, start, end):
    spy = data[BENCHMARK].loc[start:end, "Close"]
    r = spy.pct_change().dropna()
    m = compute_metrics(r)
    return {"sharpe": m["sharpe"], "cagr": m["cagr"], "max_dd": m["max_dd"]}


def run_tests(data):
    stocks = get_stock_universe(data)
    print(f"Stock universe: {len(stocks)} stocks")

    # Precompute for all stocks
    closes = {}
    ret_d = {}
    vol63 = {}
    mom126 = {}  # 6-month momentum
    mom252 = {}  # 12-month momentum
    mom21 = {}   # 1-month (for skip)
    sma200 = {}

    for t in stocks + SAFE_HAVENS:
        if t not in data:
            continue
        df = data[t]
        closes[t] = df["Close"]
        ret_d[t] = df["Close"].pct_change()
        vol63[t] = ret_d[t].rolling(63, min_periods=21).std() * np.sqrt(252)
        mom126[t] = closes[t] / closes[t].shift(126) - 1
        mom252[t] = closes[t] / closes[t].shift(252) - 1
        mom21[t] = closes[t] / closes[t].shift(21) - 1
        sma200[t] = closes[t].rolling(200).mean()

    spy_close = data[BENCHMARK]["Close"]
    spy_sma100 = spy_close.rolling(100).mean()
    spy_sma200 = spy_close.rolling(200).mean()
    spy_ret = spy_close.pct_change()
    spy_vol21 = spy_ret.rolling(21).std() * np.sqrt(252)

    def monthly_wrap(fn):
        state = {"month": None, "sig": None}
        def wrapped(data, date, idx):
            m = date.month
            if state["month"] == m and state["sig"] is not None:
                return state["sig"]
            state["month"] = m
            state["sig"] = fn(data, date, idx)
            return state["sig"]
        return wrapped

    # =================================================================
    # Strategy S1: Top 15 stocks by 12m momentum (skip last month), inv-vol
    # =================================================================
    print("\n" + "="*60)
    print("S1: Top 15 Stocks by 12m-1m Momentum, Inv-Vol Weighted")
    print("="*60)

    def core_s1(data, date, idx):
        scored = []
        for t in stocks:
            if t not in mom252 or t not in mom21 or t not in vol63:
                continue
            if date not in mom252[t].index or date not in vol63[t].index or date not in mom21[t].index:
                continue
            m12 = mom252[t].loc[date]
            m1 = mom21[t].loc[date]
            v = vol63[t].loc[date]
            if pd.isna(m12) or pd.isna(m1) or pd.isna(v) or v <= 0:
                continue
            # 12m momentum minus last 1m (classic Jegadeesh-Titman)
            mom_score = m12 - m1
            if mom_score > 0:  # Only positive momentum
                scored.append((t, mom_score, 1.0 / v))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:15]
        if not top:
            return {"weights": {"IEF": 1.0}}
        total = sum(x[2] for x in top)
        weights = {t: iv / total for t, _, iv in top}
        return {"weights": weights}

    sig = monthly_wrap(core_s1)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = backtest_stock_strategy(data, s, e, sig)
        m = compute_metrics(r)
        spy = spy_metrics(data, s, e)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} | SPY: {spy['sharpe']:.3f} {spy['cagr']:.1%}")

    # =================================================================
    # Strategy S2: S1 + SMA200 gate + safe haven rotation
    # =================================================================
    print("\n" + "="*60)
    print("S2: Top 15 Momentum + SMA200 Gate + TLT/GLD hedge")
    print("="*60)

    def core_s2(data, date, idx):
        # Gate
        if date in spy_sma200.index:
            s = spy_sma200.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                # Bear: safe havens
                w = {}
                for h in SAFE_HAVENS:
                    if h in vol63 and date in vol63[h].index:
                        v = vol63[h].loc[date]
                        if not pd.isna(v) and v > 0:
                            w[h] = 1.0 / v
                if w:
                    total = sum(w.values())
                    return {"weights": {k: v/total for k, v in w.items()}}
                return {"weights": {"IEF": 1.0}}

        return core_s1(data, date, idx)

    sig = monthly_wrap(core_s2)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = backtest_stock_strategy(data, s, e, sig)
        m = compute_metrics(r)
        spy = spy_metrics(data, s, e)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Trades={t} | SPY: {spy['sharpe']:.3f} {spy['cagr']:.1%}")

    # =================================================================
    # Strategy S3: Multi-factor stock selection (momentum + quality + low vol)
    # =================================================================
    print("\n" + "="*60)
    print("S3: Multi-Factor (Momentum + Quality + Low Vol) + Gate")
    print("="*60)

    # Precompute quality: rolling Sharpe of each stock
    stock_quality = {}
    for t in stocks:
        if t in ret_d:
            r = ret_d[t]
            m63 = r.rolling(63, min_periods=42).mean() * 252
            s63 = r.rolling(63, min_periods=42).std() * np.sqrt(252)
            stock_quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)

    # Persistence: fraction of positive days
    stock_persist = {}
    for t in stocks:
        if t in ret_d:
            stock_persist[t] = ret_d[t].rolling(63, min_periods=42).apply(lambda x: (x > 0).mean(), raw=True)

    def core_s3(data, date, idx):
        # Gate
        bear = False
        if date in spy_sma100.index:
            s = spy_sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True

        if bear:
            w = {}
            for h in SAFE_HAVENS:
                if h in vol63 and date in vol63[h].index:
                    v = vol63[h].loc[date]
                    if not pd.isna(v) and v > 0:
                        # Check trend
                        if h in sma200 and date in sma200[h].index:
                            sm = sma200[h].loc[date]
                            p = closes[h].loc[date] if date in closes[h].index else 0
                            if not pd.isna(sm) and p > sm:
                                w[h] = 1.0 / v * 2.0  # Trending up: boost
                            else:
                                w[h] = 1.0 / v * 0.5  # Trending down: reduce
                        else:
                            w[h] = 1.0 / v
            if w:
                total = sum(w.values())
                return {"weights": {k: v/total for k, v in w.items()}}
            return {"weights": {"IEF": 1.0}}

        # Bull: multi-factor stock selection
        scored = []
        for t in stocks:
            if t not in mom252 or t not in stock_quality or t not in stock_persist or t not in vol63:
                continue
            if date not in mom252[t].index or date not in stock_quality[t].index:
                continue
            if date not in stock_persist[t].index or date not in vol63[t].index:
                continue

            m12 = mom252[t].loc[date]
            m1 = mom21[t].loc[date] if t in mom21 and date in mom21[t].index else 0
            q = stock_quality[t].loc[date]
            p = stock_persist[t].loc[date]
            v = vol63[t].loc[date]

            if pd.isna(m12) or pd.isna(q) or pd.isna(p) or pd.isna(v) or v <= 0:
                continue

            # Multi-factor composite score
            mom_score = m12 - m1  # 12m - 1m momentum (skip last month)
            quality_score = q  # Rolling Sharpe
            persist_score = p  # Fraction of positive days

            # Combined: momentum * quality * persistence / vol
            # All must be positive
            if mom_score > 0 and quality_score > 0 and persist_score > 0.5:
                composite = mom_score * quality_score * persist_score
                scored.append((t, composite, 1.0 / v))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:15]

        if not top:
            return {"weights": {"SPY": 1.0}}

        total = sum(x[2] for x in top)
        weights = {t: iv / total for t, _, iv in top}
        return {"weights": weights}

    sig = monthly_wrap(core_s3)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = backtest_stock_strategy(data, s, e, sig)
        m = compute_metrics(r)
        spy = spy_metrics(data, s, e)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Trades={t} | SPY: {spy['sharpe']:.3f} {spy['cagr']:.1%}")

    # =================================================================
    # Strategy S4: S3 + vol targeting (scale exposure to target 10% vol)
    # =================================================================
    print("\n" + "="*60)
    print("S4: S3 + Vol Targeting (10% target)")
    print("="*60)

    def core_s4(data, date, idx):
        base = core_s3(data, date, idx)
        weights = base.get("weights", {})
        if not weights:
            return base

        # Vol targeting: scale down if recent vol is high
        if date in spy_vol21.index:
            v = spy_vol21.loc[date]
            if not pd.isna(v) and v > 0:
                target = 0.10
                scale = min(target / v, 1.0)  # Can't lever up, only scale down
                if scale < 0.95:
                    # Reduce equity, add IEF
                    hedge = 1.0 - scale
                    new_w = {}
                    for t, w in weights.items():
                        new_w[t] = w * scale
                    new_w["IEF"] = new_w.get("IEF", 0) + hedge
                    return {"weights": new_w}

        return base

    sig = monthly_wrap(core_s4)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = backtest_stock_strategy(data, s, e, sig)
        m = compute_metrics(r)
        spy = spy_metrics(data, s, e)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Trades={t} | SPY: {spy['sharpe']:.3f} {spy['cagr']:.1%}")

    # =================================================================
    # Strategy S5: Top 10 stocks by composite + more concentrated
    # =================================================================
    print("\n" + "="*60)
    print("S5: Top 10 Concentrated Momentum-Quality + Gate")
    print("="*60)

    def core_s5(data, date, idx):
        bear = False
        if date in spy_sma100.index:
            s = spy_sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True

        if bear:
            return {"weights": {"TLT": 0.4, "GLD": 0.35, "IEF": 0.25}}

        scored = []
        for t in stocks:
            if t not in mom126 or t not in vol63:
                continue
            if date not in mom126[t].index or date not in vol63[t].index:
                continue
            m6 = mom126[t].loc[date]
            v = vol63[t].loc[date]
            if pd.isna(m6) or pd.isna(v) or v <= 0 or m6 <= 0:
                continue
            # Risk-adjusted momentum
            score = m6 / v
            scored.append((t, score, 1.0 / v))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:10]
        if not top:
            return {"weights": {"SPY": 1.0}}
        total = sum(x[2] for x in top)
        weights = {t: iv / total for t, _, iv in top}
        return {"weights": weights}

    sig = monthly_wrap(core_s5)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = backtest_stock_strategy(data, s, e, sig)
        m = compute_metrics(r)
        spy = spy_metrics(data, s, e)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Trades={t} | SPY: {spy['sharpe']:.3f} {spy['cagr']:.1%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
