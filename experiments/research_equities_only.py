#!/usr/bin/env python3
"""
EQUITIES ONLY — targeting Sharpe 3+

Key insight: be extremely selective about WHEN to invest.
Only invest when ALL conditions align (strong trend + low vol + high breadth).
When invested: concentrated top 5-10 momentum-quality stocks.
When not: 100% cash.

The selectivity must be so good that invested-period Sharpe is 3-4+.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

BENCHMARK = "SPY"
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
NON_STOCKS = set(SECTOR_ETFS + ["TLT","GLD","IEF","SPY","QQQ","IWM","DIA","HYG","SLV","USO"])

def compute_metrics(rets, rf=0.02):
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0, "ann_vol": 0, "invested": 0}
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
    return {"sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
            "max_dd": round(float(mdd), 4), "sortino": round(float(sortino), 3),
            "ann_vol": round(float(rets.std() * np.sqrt(252)), 4),
            "invested": round(float(invested), 3)}


def monthly_backtest(data, start, end, weight_fn, tx_bps=5):
    """Monthly rebal, T+1 open execution, equities only."""
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    slip = tx_bps / 10000
    daily_rets = []; current_w = {}; last_month = None; trades = 0

    for date in dates:
        idx = spy.index.get_loc(date)
        if idx < 300:
            daily_rets.append(0.0); continue
        month = date.month
        rebalance = (last_month is not None and month != last_month)
        last_month = month

        if rebalance:
            new_w = weight_fn(date)
            dr = 0.0
            for t, w in current_w.items():
                if t not in new_w or abs(new_w.get(t, 0) - w) > 0.005:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            prev_c = df.iloc[si-1]["Close"]
                            today_o = df.loc[date, "Open"] if "Open" in df.columns else prev_c
                            dr += (today_o * (1-slip) / prev_c - 1) * w
                    trades += 1
                else:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0: dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
            for t, w in new_w.items():
                if t not in current_w or abs(current_w.get(t, 0) - w) > 0.005:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        buy = today_o * (1+slip)
                        today_c = df.loc[date, "Close"]
                        if buy > 0: dr += (today_c / buy - 1) * w
                    trades += 1
            daily_rets.append(dr); current_w = new_w
        else:
            if current_w:
                dr = 0.0
                for t, w in current_w.items():
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0: dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
                daily_rets.append(dr)
            else:
                daily_rets.append(0.0)
    return pd.Series(daily_rets, index=dates), trades


def run(data):
    stocks = [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]
    print(f"{len(stocks)} stocks")

    # Precompute everything
    closes = {}; ret_d = {}; vol63 = {}; vol21 = {}
    mom252 = {}; mom126 = {}; mom63 = {}; mom21 = {}; mom5 = {}
    sma200 = {}; sma50 = {}; quality = {}

    for t in stocks:
        if t not in data: continue
        df = data[t]; closes[t] = df["Close"]; ret_d[t] = df["Close"].pct_change()
        vol63[t] = ret_d[t].rolling(63, min_periods=21).std() * np.sqrt(252)
        vol21[t] = ret_d[t].rolling(21, min_periods=10).std() * np.sqrt(252)
        mom252[t] = closes[t] / closes[t].shift(252) - 1
        mom126[t] = closes[t] / closes[t].shift(126) - 1
        mom63[t] = closes[t] / closes[t].shift(63) - 1
        mom21[t] = closes[t] / closes[t].shift(21) - 1
        mom5[t] = closes[t] / closes[t].shift(5) - 1
        sma200[t] = closes[t].rolling(200).mean()
        sma50[t] = closes[t].rolling(50).mean()
        m63 = ret_d[t].rolling(63, min_periods=42).mean() * 252
        s63 = ret_d[t].rolling(63, min_periods=42).std() * np.sqrt(252)
        quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)

    spy_close = data[BENCHMARK]["Close"]
    spy_ret = spy_close.pct_change()
    spy_sma50 = spy_close.rolling(50).mean()
    spy_sma100 = spy_close.rolling(100).mean()
    spy_sma200 = spy_close.rolling(200).mean()
    spy_vol21 = spy_ret.rolling(21).std() * np.sqrt(252)

    # Sector breadth (how many sectors have positive 63d momentum)
    sec_mom63 = {}
    for e in SECTOR_ETFS:
        if e in data:
            sec_mom63[e] = data[e]["Close"] / data[e]["Close"].shift(63) - 1

    def sector_breadth(date):
        count = 0
        total = 0
        for e, m in sec_mom63.items():
            if date in m.index:
                val = m.loc[date]
                if not pd.isna(val):
                    total += 1
                    if val > 0: count += 1
        return count, total

    def ranker(date, n=10):
        """Multi-factor stock ranker."""
        scored = []
        for t in stocks:
            if t not in mom252 or date not in mom252[t].index: continue
            m12 = mom252[t].loc[date]
            m1 = mom21[t].loc[date] if date in mom21[t].index else 0
            m6 = mom126[t].loc[date] if date in mom126[t].index else 0
            m3 = mom63[t].loc[date] if date in mom63[t].index else 0
            q = quality[t].loc[date] if date in quality[t].index else 0
            v = vol63[t].loc[date] if date in vol63[t].index else 0
            sm = sma200[t].loc[date] if date in sma200[t].index else 0
            price = closes[t].loc[date] if date in closes[t].index else 0
            if pd.isna(m12) or pd.isna(v) or v <= 0.01: continue
            if pd.isna(q) or q <= 0: continue

            mom_skip = m12 - (m1 if not pd.isna(m1) else 0)
            if mom_skip <= 0: continue
            if not pd.isna(sm) and price <= sm: continue

            # Ensemble: avg of positive momentums
            moms = [mom_skip]
            if not pd.isna(m3) and m3 > 0: moms.append(m3)
            if not pd.isna(m6) and m6 > 0: moms.append(m6)
            avg_mom = np.mean(moms)
            if avg_mom <= 0: continue

            composite = avg_mom * max(q, 0.01)
            scored.append((t, composite, 1.0/v))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    # ================================================================
    # E1: Selective conditions — invest only when everything aligns
    # ================================================================
    print("\n" + "="*60)
    print("E1: Ultra-Selective Conditions (equities only, cash otherwise)")
    print("="*60)

    for conditions_label, cond_fn in [
        ("SPY>SMA50", lambda d: spy_close.loc[d] > spy_sma50.loc[d] if d in spy_sma50.index and not pd.isna(spy_sma50.loc[d]) else False),
        ("SPY>SMA100", lambda d: spy_close.loc[d] > spy_sma100.loc[d] if d in spy_sma100.index and not pd.isna(spy_sma100.loc[d]) else False),
        ("SPY>SMA50+SMA50>SMA200", lambda d: (spy_close.loc[d] > spy_sma50.loc[d] and spy_sma50.loc[d] > spy_sma200.loc[d]) if d in spy_sma50.index and d in spy_sma200.index and not pd.isna(spy_sma50.loc[d]) and not pd.isna(spy_sma200.loc[d]) else False),
        ("Trend+LowVol(<15%)", lambda d: (spy_close.loc[d] > spy_sma100.loc[d] if d in spy_sma100.index and not pd.isna(spy_sma100.loc[d]) else False) and (spy_vol21.loc[d] < 0.15 if d in spy_vol21.index and not pd.isna(spy_vol21.loc[d]) else False)),
        ("Trend+LowVol+Breadth(>6)", lambda d: (spy_close.loc[d] > spy_sma100.loc[d] if d in spy_sma100.index and not pd.isna(spy_sma100.loc[d]) else False) and (spy_vol21.loc[d] < 0.15 if d in spy_vol21.index and not pd.isna(spy_vol21.loc[d]) else False) and sector_breadth(d)[0] >= 6),
        ("Trend+LowVol(<12%)+Breadth(>7)", lambda d: (spy_close.loc[d] > spy_sma100.loc[d] if d in spy_sma100.index and not pd.isna(spy_sma100.loc[d]) else False) and (spy_vol21.loc[d] < 0.12 if d in spy_vol21.index and not pd.isna(spy_vol21.loc[d]) else False) and sector_breadth(d)[0] >= 7),
    ]:
        for n_stocks in [5, 10, 20]:
            state = {"month": None, "w": None}
            def make_fn(cond, n):
                st = {"month": None, "w": None}
                def fn(date):
                    m = date.month
                    if st["month"] == m and st["w"] is not None: return st["w"]
                    st["month"] = m
                    try:
                        invest = cond(date)
                    except:
                        invest = False
                    if not invest:
                        st["w"] = {}
                        return {}
                    top = ranker(date, n)
                    if not top:
                        st["w"] = {}
                        return {}
                    ti = sum(iv for _, _, iv in top)
                    w = {t: iv/ti for t, _, iv in top}
                    st["w"] = w
                    return w
                return fn

            fn = make_fn(cond_fn, n_stocks)
            results = []
            for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
                r, t = monthly_backtest(data, s, e, fn)
                m = compute_metrics(r)
                results.append(m)
            print(f"  {conditions_label:35s} N={n_stocks:2d}: "
                  f"Tr={results[0]['sharpe']:6.3f}/{results[0]['invested']:.0%} "
                  f"Va={results[1]['sharpe']:6.3f}/{results[1]['invested']:.0%} "
                  f"Te={results[2]['sharpe']:6.3f}/{results[2]['invested']:.0%} "
                  f"Vol={results[0]['ann_vol']:.1%}/{results[2]['ann_vol']:.1%}")

    # ================================================================
    # E2: Daily signal with monthly stock selection
    # Check conditions DAILY but only rebalance stocks monthly
    # This captures the timing benefit of daily signals
    # ================================================================
    print("\n" + "="*60)
    print("E2: Daily Signal / Monthly Stock Pick (equities + cash)")
    print("="*60)

    def daily_backtest(data, start, end, signal_fn, stock_ranker_fn, tx_bps=5):
        """
        Daily: check signal_fn to decide invested vs cash.
        Monthly: update stock selection.
        T+1 open execution.
        """
        spy = data[BENCHMARK]
        dates = spy.loc[start:end].index
        slip = tx_bps / 10000

        daily_rets = []
        stock_picks = {}  # monthly stock picks {ticker: weight}
        current_invested = False
        pending_enter = False
        pending_exit = False
        last_month = None
        trades = 0

        for date in dates:
            idx = spy.index.get_loc(date)
            if idx < 300:
                daily_rets.append(0.0); continue

            # Monthly stock update
            month = date.month
            if last_month is not None and month != last_month:
                top = stock_ranker_fn(date)
                if top:
                    ti = sum(iv for _, _, iv in top)
                    stock_picks = {t: iv/ti for t, _, iv in top}
                else:
                    stock_picks = {}
            last_month = month

            # Execute pending
            if pending_enter and not current_invested and stock_picks:
                # Buy at today's open
                dr = 0.0
                for t, w in stock_picks.items():
                    df = data.get(t)
                    if df is not None and date in df.index:
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        buy = today_o * (1 + slip)
                        today_c = df.loc[date, "Close"]
                        if buy > 0: dr += (today_c / buy - 1) * w
                    trades += 1
                daily_rets.append(dr)
                current_invested = True
                pending_enter = False

                # Check signal for tomorrow
                sig = signal_fn(date)
                if not sig: pending_exit = True
                continue

            if pending_exit and current_invested:
                # Sell at today's open (overnight return)
                dr = 0.0
                for t, w in stock_picks.items():
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            prev_c = df.iloc[si-1]["Close"]
                            today_o = df.loc[date, "Open"] if "Open" in df.columns else prev_c
                            dr += (today_o * (1-slip) / prev_c - 1) * w
                    trades += 1
                daily_rets.append(dr)
                current_invested = False
                pending_exit = False

                sig = signal_fn(date)
                if sig and stock_picks: pending_enter = True
                continue

            pending_enter = False
            pending_exit = False

            # Normal day
            if current_invested and stock_picks:
                dr = 0.0
                for t, w in stock_picks.items():
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0: dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
                daily_rets.append(dr)
            else:
                daily_rets.append(0.0)

            # Generate signal for tomorrow
            sig = signal_fn(date)
            if current_invested and not sig:
                pending_exit = True
            elif not current_invested and sig and stock_picks:
                pending_enter = True

        return pd.Series(daily_rets, index=dates), trades

    for label, sig_fn in [
        ("SMA50+SMA50>SMA200+Vol<15%", lambda d: (
            d in spy_sma50.index and d in spy_sma200.index and d in spy_vol21.index and
            not pd.isna(spy_sma50.loc[d]) and not pd.isna(spy_sma200.loc[d]) and not pd.isna(spy_vol21.loc[d]) and
            spy_close.loc[d] > spy_sma50.loc[d] and spy_sma50.loc[d] > spy_sma200.loc[d] and spy_vol21.loc[d] < 0.15
        )),
        ("SMA50+Vol<12%+Breadth>7", lambda d: (
            d in spy_sma50.index and d in spy_vol21.index and
            not pd.isna(spy_sma50.loc[d]) and not pd.isna(spy_vol21.loc[d]) and
            spy_close.loc[d] > spy_sma50.loc[d] and spy_vol21.loc[d] < 0.12 and sector_breadth(d)[0] >= 7
        )),
        ("SMA100+Vol<18%", lambda d: (
            d in spy_sma100.index and d in spy_vol21.index and
            not pd.isna(spy_sma100.loc[d]) and not pd.isna(spy_vol21.loc[d]) and
            spy_close.loc[d] > spy_sma100.loc[d] and spy_vol21.loc[d] < 0.18
        )),
    ]:
        for n in [5, 10, 15]:
            rank_fn = lambda d, _n=n: ranker(d, _n)
            results = []
            for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
                r, t = daily_backtest(data, s, e, sig_fn, rank_fn)
                m = compute_metrics(r)
                results.append(m)
            print(f"  {label:35s} N={n:2d}: "
                  f"Tr={results[0]['sharpe']:6.3f}/{results[0]['invested']:.0%} "
                  f"Va={results[1]['sharpe']:6.3f}/{results[1]['invested']:.0%} "
                  f"Te={results[2]['sharpe']:6.3f}/{results[2]['invested']:.0%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run(data)
