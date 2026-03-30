#!/usr/bin/env python3
"""
Hybrid: stocks + bonds + gold combined for maximum Sharpe.
Key insight: stock alpha + bond/gold diversification = high risk-adjusted return.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

BENCHMARK = "SPY"
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
SAFE_HAVENS = ["TLT", "GLD", "IEF"]
NON_STOCKS = set(SECTOR_ETFS + SAFE_HAVENS + ["SPY", "QQQ", "IWM", "DIA", "HYG", "SLV", "USO"])

def get_stocks(data):
    return [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]

def compute_metrics(rets, rf=0.02):
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0, "ann_vol": 0}
    excess = rets - rf / 252
    n_years = len(rets) / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years >= 1 else total
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    downside = excess[excess < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    return {"sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
            "max_dd": round(float(mdd), 4), "sortino": round(float(sortino), 3),
            "ann_vol": round(float(rets.std() * np.sqrt(252)), 4)}

def simple_backtest(data, start, end, weight_fn, tx_bps=5):
    """
    Simple monthly-rebalance backtest with T+1 open execution.
    weight_fn(date) -> dict {ticker: weight}
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    slip = tx_bps / 10000

    daily_rets = []
    current_w = {}
    last_month = None
    trades = 0

    for date in dates:
        idx = spy.index.get_loc(date)
        if idx < 252:
            daily_rets.append(0.0)
            continue

        # Check if rebalance needed (new month)
        month = date.month
        rebalance = (last_month is not None and month != last_month)
        last_month = month

        if rebalance:
            # Compute new weights (signal from PREVIOUS day's close, execute at today's open)
            new_w = weight_fn(date)

            # P&L from rebalancing
            dr = 0.0
            # Existing positions: overnight return then sell
            for t, w in current_w.items():
                if t not in new_w or abs(new_w.get(t, 0) - w) > 0.005:
                    # Position changing: overnight return (prev close to open)
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            prev_c = df.iloc[si-1]["Close"]
                            today_o = df.loc[date, "Open"] if "Open" in df.columns else prev_c
                            dr += (today_o * (1 - slip) / prev_c - 1) * w
                    trades += 1
                else:
                    # Position unchanged: close to close
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w

            # New positions: buy at open, return is open to close
            for t, w in new_w.items():
                if t not in current_w or abs(current_w.get(t, 0) - w) > 0.005:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        buy = today_o * (1 + slip)
                        today_c = df.loc[date, "Close"]
                        if buy > 0:
                            dr += (today_c / buy - 1) * w
                    trades += 1

            daily_rets.append(dr)
            current_w = new_w
        else:
            # Normal day: close to close return
            if current_w:
                dr = 0.0
                for t, w in current_w.items():
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
                daily_rets.append(dr)
            else:
                daily_rets.append(0.0)

    return pd.Series(daily_rets, index=dates), trades


def run_tests(data):
    stocks = get_stocks(data)
    print(f"Stock universe: {len(stocks)} stocks")

    # Precompute
    closes = {}; ret_d = {}; vol63 = {}; mom252 = {}; mom21 = {}; mom126 = {}; sma200 = {}
    quality = {}; persistence = {}

    all_tickers = stocks + SAFE_HAVENS
    for t in all_tickers:
        if t not in data: continue
        df = data[t]
        closes[t] = df["Close"]
        ret_d[t] = df["Close"].pct_change()
        vol63[t] = ret_d[t].rolling(63, min_periods=21).std() * np.sqrt(252)
        mom252[t] = closes[t] / closes[t].shift(252) - 1
        mom126[t] = closes[t] / closes[t].shift(126) - 1
        mom21[t] = closes[t] / closes[t].shift(21) - 1
        sma200[t] = closes[t].rolling(200).mean()
        # Quality = rolling 63d Sharpe
        m63 = ret_d[t].rolling(63, min_periods=42).mean() * 252
        s63 = ret_d[t].rolling(63, min_periods=42).std() * np.sqrt(252)
        quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)
        # Persistence
        persistence[t] = ret_d[t].rolling(63, min_periods=42).apply(lambda x: (x > 0).mean(), raw=True)

    spy_close = data[BENCHMARK]["Close"]
    spy_sma100 = spy_close.rolling(100).mean()

    def stock_ranker(date, n=15):
        """Rank stocks by multi-factor score, return top N with inv-vol weights."""
        scored = []
        for t in stocks:
            if t not in mom252 or t not in quality or t not in persistence or t not in vol63:
                continue
            if date not in mom252[t].index: continue
            m12 = mom252[t].loc[date]
            m1 = mom21[t].loc[date] if t in mom21 and date in mom21[t].index else 0
            q = quality[t].loc[date] if date in quality[t].index else 0
            p = persistence[t].loc[date] if date in persistence[t].index else 0
            v = vol63[t].loc[date] if date in vol63[t].index else 0
            sm = sma200[t].loc[date] if t in sma200 and date in sma200[t].index else 0
            price = closes[t].loc[date] if date in closes[t].index else 0

            if pd.isna(m12) or pd.isna(q) or pd.isna(v) or v <= 0.01:
                continue
            if pd.isna(p): p = 0.5

            # Filters (all must pass)
            mom_score = m12 - (m1 if not pd.isna(m1) else 0)
            if mom_score <= 0: continue
            if q <= 0: continue
            if not pd.isna(sm) and price <= sm: continue  # Must be above SMA200

            composite = mom_score * max(q, 0.01) * max(p, 0.4)
            scored.append((t, composite, 1.0 / v))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    # =================================================================
    # H1: 70% stocks + 30% bonds/gold (fixed split)
    # =================================================================
    print("\n" + "="*60)
    print("H1: 70% Top 15 Stocks + 30% TLT/GLD/IEF Risk Parity")
    print("="*60)

    def weights_h1(date):
        eq_pct = 0.70
        hedge_pct = 0.30

        # Stock picks
        top = stock_ranker(date, 15)
        weights = {}
        if top:
            total_iv = sum(iv for _, _, iv in top)
            for t, _, iv in top:
                weights[t] = (iv / total_iv) * eq_pct
        else:
            weights["SPY"] = eq_pct

        # Safe haven allocation
        hw = {}
        for h in SAFE_HAVENS:
            if h in vol63 and date in vol63[h].index:
                v = vol63[h].loc[date]
                if not pd.isna(v) and v > 0:
                    hw[h] = 1.0 / v
        if hw:
            ht = sum(hw.values())
            for h, iv in hw.items():
                weights[h] = (iv / ht) * hedge_pct
        else:
            weights["IEF"] = hedge_pct
        return weights

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = simple_backtest(data, s, e, weights_h1)
        m = compute_metrics(r)
        spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
        sm = compute_metrics(spy_r)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Sortino={m['sortino']:.3f} Trades={t}")
        print(f"          SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%}")

    # =================================================================
    # H2: Same but with SMA100 regime — shift to more hedge in bear
    # =================================================================
    print("\n" + "="*60)
    print("H2: Regime-Adaptive: 80% stocks (bull) / 20% stocks (bear) + hedge")
    print("="*60)

    def weights_h2(date):
        bear = False
        if date in spy_sma100.index:
            s = spy_sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True

        eq_pct = 0.20 if bear else 0.80
        hedge_pct = 1.0 - eq_pct

        top = stock_ranker(date, 15 if not bear else 5)
        weights = {}
        if top:
            total_iv = sum(iv for _, _, iv in top)
            for t, _, iv in top:
                weights[t] = (iv / total_iv) * eq_pct
        else:
            weights["SPY"] = eq_pct

        hw = {}
        for h in SAFE_HAVENS:
            if h in vol63 and date in vol63[h].index:
                v = vol63[h].loc[date]
                if not pd.isna(v) and v > 0:
                    # In bear, check haven trend too
                    if bear and h in sma200 and date in sma200[h].index:
                        sm = sma200[h].loc[date]
                        p = closes[h].loc[date] if date in closes[h].index else 0
                        if not pd.isna(sm) and p > sm:
                            hw[h] = 1.0 / v * 2.0
                        else:
                            hw[h] = 1.0 / v * 0.3
                    else:
                        hw[h] = 1.0 / v
        if hw:
            ht = sum(hw.values())
            for h, iv in hw.items():
                weights[h] = (iv / ht) * hedge_pct
        else:
            weights["IEF"] = hedge_pct
        return weights

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = simple_backtest(data, s, e, weights_h2)
        m = compute_metrics(r)
        spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
        sm = compute_metrics(spy_r)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Sortino={m['sortino']:.3f} Trades={t}")
        print(f"          SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%}")

    # =================================================================
    # H3: Pure stock momentum with dynamic safe haven split
    # Split determined by how many stocks pass all filters
    # =================================================================
    print("\n" + "="*60)
    print("H3: Dynamic Split — more stocks pass = more equity")
    print("="*60)

    def weights_h3(date):
        all_passing = stock_ranker(date, 100)  # Get all passing stocks
        n_pass = len(all_passing)

        # Dynamic equity allocation: more passing stocks = stronger market
        # Scale: 0 stocks = 10% equity, 50+ stocks = 90% equity
        eq_pct = min(0.90, max(0.10, 0.10 + (n_pass / 50) * 0.80))
        hedge_pct = 1.0 - eq_pct

        # Pick top 15 (or fewer)
        top = all_passing[:15]
        weights = {}
        if top:
            total_iv = sum(iv for _, _, iv in top)
            for t, _, iv in top:
                weights[t] = (iv / total_iv) * eq_pct
        else:
            eq_pct = 0
            hedge_pct = 1.0

        hw = {}
        for h in SAFE_HAVENS:
            if h in vol63 and date in vol63[h].index:
                v = vol63[h].loc[date]
                if not pd.isna(v) and v > 0:
                    hw[h] = 1.0 / v
        if hw:
            ht = sum(hw.values())
            for h, iv in hw.items():
                weights[h] = (iv / ht) * hedge_pct
        else:
            weights["IEF"] = hedge_pct
        return weights

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = simple_backtest(data, s, e, weights_h3)
        m = compute_metrics(r)
        spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
        sm = compute_metrics(spy_r)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Sortino={m['sortino']:.3f} Trades={t}")
        print(f"          SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%}")

    # =================================================================
    # H4: Vary number of stocks (5, 10, 15, 20, 30) with fixed 30% hedge
    # =================================================================
    print("\n" + "="*60)
    print("H4: Varying concentration (N stocks, 70/30 split)")
    print("="*60)
    for n_stocks in [5, 10, 15, 20, 30]:
        def make_fn(n):
            def fn(date):
                top = stock_ranker(date, n)
                weights = {}
                if top:
                    total_iv = sum(iv for _, _, iv in top)
                    for t, _, iv in top:
                        weights[t] = (iv / total_iv) * 0.70
                else:
                    weights["SPY"] = 0.70
                hw = {}
                for h in SAFE_HAVENS:
                    if h in vol63 and date in vol63[h].index:
                        v = vol63[h].loc[date]
                        if not pd.isna(v) and v > 0: hw[h] = 1.0 / v
                if hw:
                    ht = sum(hw.values())
                    for h, iv in hw.items(): weights[h] = (iv / ht) * 0.30
                else:
                    weights["IEF"] = 0.30
                return weights
            return fn

        fn = make_fn(n_stocks)
        results = []
        for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
            r, t = simple_backtest(data, s, e, fn)
            m = compute_metrics(r)
            results.append(m)
        print(f"  N={n_stocks:2d}: Train Sharpe={results[0]['sharpe']:.3f} Valid={results[1]['sharpe']:.3f} Test={results[2]['sharpe']:.3f} | "
              f"MaxDD: {results[0]['max_dd']:.1%}/{results[1]['max_dd']:.1%}/{results[2]['max_dd']:.1%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
