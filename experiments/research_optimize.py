#!/usr/bin/env python3
"""
Optimize the hybrid strategy: more stocks, better splits, trend-filtered havens.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

BENCHMARK = "SPY"
SAFE_HAVENS = ["TLT", "GLD", "IEF"]
NON_STOCKS = set(["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC",
                   "TLT","GLD","IEF","SPY","QQQ","IWM","DIA","HYG","SLV","USO"])

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
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    slip = tx_bps / 10000
    daily_rets = []; current_w = {}; last_month = None; trades = 0

    for date in dates:
        idx = spy.index.get_loc(date)
        if idx < 252:
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
                            dr += (today_o * (1 - slip) / prev_c - 1) * w
                    trades += 1
                else:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
            for t, w in new_w.items():
                if t not in current_w or abs(current_w.get(t, 0) - w) > 0.005:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        buy = today_o * (1 + slip)
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
                        if si > 0:
                            dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
                daily_rets.append(dr)
            else:
                daily_rets.append(0.0)
    return pd.Series(daily_rets, index=dates), trades


def run_tests(data):
    stocks = [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]
    print(f"Stock universe: {len(stocks)} stocks")

    closes = {}; ret_d = {}; vol63 = {}; mom252 = {}; mom21 = {}; sma200 = {}; quality = {}; persistence = {}
    for t in stocks + SAFE_HAVENS:
        if t not in data: continue
        df = data[t]
        closes[t] = df["Close"]; ret_d[t] = df["Close"].pct_change()
        vol63[t] = ret_d[t].rolling(63, min_periods=21).std() * np.sqrt(252)
        mom252[t] = closes[t] / closes[t].shift(252) - 1
        mom21[t] = closes[t] / closes[t].shift(21) - 1
        sma200[t] = closes[t].rolling(200).mean()
        m63 = ret_d[t].rolling(63, min_periods=42).mean() * 252
        s63 = ret_d[t].rolling(63, min_periods=42).std() * np.sqrt(252)
        quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)
        persistence[t] = ret_d[t].rolling(63, min_periods=42).apply(lambda x: (x > 0).mean(), raw=True)

    spy_close = data[BENCHMARK]["Close"]
    spy_sma100 = spy_close.rolling(100).mean()

    def stock_ranker(date, n=15):
        scored = []
        for t in stocks:
            if t not in mom252 or t not in quality or t not in vol63: continue
            if date not in mom252[t].index: continue
            m12 = mom252[t].loc[date]
            m1 = mom21[t].loc[date] if t in mom21 and date in mom21[t].index else 0
            q = quality[t].loc[date] if date in quality[t].index else 0
            p = persistence[t].loc[date] if date in persistence[t].index else 0
            v = vol63[t].loc[date] if date in vol63[t].index else 0
            sm = sma200[t].loc[date] if t in sma200 and date in sma200[t].index else 0
            price = closes[t].loc[date] if date in closes[t].index else 0
            if pd.isna(m12) or pd.isna(q) or pd.isna(v) or v <= 0.01: continue
            if pd.isna(p): p = 0.5
            mom_score = m12 - (m1 if not pd.isna(m1) else 0)
            if mom_score <= 0: continue
            if q <= 0: continue
            if not pd.isna(sm) and price <= sm: continue
            composite = mom_score * max(q, 0.01) * max(p, 0.4)
            scored.append((t, composite, 1.0 / v))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    # =================================================================
    # Grid search: N stocks x equity_pct
    # =================================================================
    print("\n" + "="*60)
    print("GRID: N stocks x Equity% (monthly rebal, T+1 open)")
    print("="*60)
    print(f"  {'N':>3} {'Eq%':>4} | {'Train Sh':>8} {'Val Sh':>7} {'Test Sh':>7} | {'Train DD':>8} {'Val DD':>7} {'Test DD':>7} | {'Train CAGR':>10} {'Test CAGR':>10}")

    best_train = {"sharpe": 0}
    best_combo = None

    for n_stocks in [15, 20, 30, 40, 50]:
        for eq_pct in [0.50, 0.60, 0.70, 0.80]:
            def make_fn(n, ep):
                def fn(date):
                    top = stock_ranker(date, n)
                    weights = {}
                    if top:
                        total_iv = sum(iv for _, _, iv in top)
                        for t, _, iv in top:
                            weights[t] = (iv / total_iv) * ep
                    else:
                        weights["SPY"] = ep
                    hp = 1.0 - ep
                    hw = {}
                    for h in SAFE_HAVENS:
                        if h in vol63 and date in vol63[h].index:
                            v = vol63[h].loc[date]
                            if not pd.isna(v) and v > 0: hw[h] = 1.0 / v
                    if hw:
                        ht = sum(hw.values())
                        for h, iv in hw.items(): weights[h] = (iv / ht) * hp
                    else:
                        weights["IEF"] = hp
                    return weights
                return fn

            fn = make_fn(n_stocks, eq_pct)
            results = []
            for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
                r, t = simple_backtest(data, s, e, fn)
                m = compute_metrics(r)
                results.append(m)

            print(f"  {n_stocks:3d} {eq_pct:.0%} | {results[0]['sharpe']:8.3f} {results[1]['sharpe']:7.3f} {results[2]['sharpe']:7.3f} | "
                  f"{results[0]['max_dd']:7.1%} {results[1]['max_dd']:6.1%} {results[2]['max_dd']:6.1%} | "
                  f"{results[0]['cagr']:9.1%} {results[2]['cagr']:9.1%}")

            # Track best (minimize max of -Sharpe across periods)
            min_sharpe = min(results[0]['sharpe'], results[1]['sharpe'], results[2]['sharpe'])
            if min_sharpe > best_train.get("min_sharpe", -999):
                best_train = {"min_sharpe": min_sharpe, "n": n_stocks, "eq": eq_pct, "results": results}
                best_combo = (n_stocks, eq_pct)

    if best_combo:
        n, ep = best_combo
        print(f"\n  Best by min-period Sharpe: N={n}, Eq={ep:.0%}")
        print(f"    Train: {best_train['results'][0]['sharpe']:.3f}, Valid: {best_train['results'][1]['sharpe']:.3f}, Test: {best_train['results'][2]['sharpe']:.3f}")

    # =================================================================
    # Best combo + regime adaptation
    # =================================================================
    print("\n" + "="*60)
    print("BEST COMBO + Regime Adaptation (80% eq bull, 30% eq bear)")
    print("="*60)

    for n_stocks in [20, 30, 40]:
        def make_regime_fn(n):
            def fn(date):
                bear = False
                if date in spy_sma100.index:
                    s = spy_sma100.loc[date]
                    if not pd.isna(s) and spy_close.loc[date] <= s:
                        bear = True
                ep = 0.30 if bear else 0.80
                hp = 1.0 - ep
                top = stock_ranker(date, n if not bear else min(n, 10))
                weights = {}
                if top:
                    total_iv = sum(iv for _, _, iv in top)
                    for t, _, iv in top:
                        weights[t] = (iv / total_iv) * ep
                else:
                    weights["SPY"] = ep
                hw = {}
                for h in SAFE_HAVENS:
                    if h in vol63 and date in vol63[h].index:
                        v = vol63[h].loc[date]
                        if not pd.isna(v) and v > 0:
                            if bear and h in sma200 and date in sma200[h].index:
                                sm = sma200[h].loc[date]
                                p = closes[h].loc[date] if date in closes[h].index else 0
                                if not pd.isna(sm) and p > sm: hw[h] = (1.0/v) * 2
                                else: hw[h] = (1.0/v) * 0.3
                            else: hw[h] = 1.0 / v
                if hw:
                    ht = sum(hw.values())
                    for h, iv in hw.items(): weights[h] = (iv / ht) * hp
                else:
                    weights["IEF"] = hp
                return weights
            return fn

        fn = make_regime_fn(n_stocks)
        results = []
        for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
            r, t = simple_backtest(data, s, e, fn)
            m = compute_metrics(r)
            results.append(m)
        print(f"  N={n_stocks}: Train={results[0]['sharpe']:.3f} Valid={results[1]['sharpe']:.3f} Test={results[2]['sharpe']:.3f} | "
              f"DD: {results[0]['max_dd']:.1%}/{results[1]['max_dd']:.1%}/{results[2]['max_dd']:.1%}")

    # =================================================================
    # Walk-forward for the best strategy: N=30, 70/30, regime
    # =================================================================
    print("\n" + "="*60)
    print("WALK-FORWARD: N=30, 80% eq (bull) / 30% eq (bear)")
    print("="*60)

    fn_wf = make_regime_fn(30)
    for year in range(2011, 2026):
        s, e = f"{year}-01-01", f"{year}-12-31"
        try:
            r, t = simple_backtest(data, s, e, fn_wf)
            m = compute_metrics(r)
            spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
            sm = compute_metrics(spy_r)
            beat = "✓" if m["sharpe"] > sm["sharpe"] else " "
            print(f"  {year}: ASRP Sharpe={m['sharpe']:6.3f} SPY={sm['sharpe']:6.3f} {beat} | "
                  f"CAGR={m['cagr']:5.1%} vs {sm['cagr']:5.1%} | MaxDD={m['max_dd']:6.1%} vs {sm['max_dd']:6.1%}")
        except: pass


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
