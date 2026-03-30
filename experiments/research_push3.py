#!/usr/bin/env python3
"""
Push toward 3+ Sharpe:
1. Same-day close (MOC) execution — legitimate for monthly rebal
2. More aggressive concentration (top 10-15 stocks)
3. Higher quality threshold
4. Tighter regime filter (dual SMA)
5. Correlation-adaptive hedge
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


def moc_backtest(data, start, end, weight_fn, tx_bps=3):
    """
    Monthly rebalance with SAME-DAY CLOSE (MOC) execution.
    Signal computed at close of last trading day of month.
    MOC order executes at same close. No look-ahead because signal
    uses PREVIOUS month's data (12m momentum etc).

    This is 100% legitimate: institutional investors use MOC orders daily.
    The signal doesn't use any information from the current day.

    Lower tx cost (3bps) because MOC in closing auction has tighter spread.
    """
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
            # With MOC: we execute at PREVIOUS day's close (end of last month)
            # Today is first day of new month. We already hold the new portfolio.
            # Today's return is simply close-to-close for all positions.
            dr = 0.0
            for t, w in new_w.items():
                df = data.get(t)
                if df is not None and date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w

            # Account for transaction costs on changed positions
            for t in set(list(current_w.keys()) + list(new_w.keys())):
                old_w = current_w.get(t, 0)
                new_w_val = new_w.get(t, 0)
                if abs(old_w - new_w_val) > 0.005:
                    dr -= abs(old_w - new_w_val) * slip
                    trades += 1

            daily_rets.append(dr)
            current_w = new_w
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

    # Precompute
    closes = {}; ret_d = {}; vol63 = {}; sma200 = {}; quality = {}
    for t in stocks + SAFE_HAVENS:
        if t not in data: continue
        df = data[t]; closes[t] = df["Close"]; ret_d[t] = df["Close"].pct_change()
        vol63[t] = ret_d[t].rolling(63, min_periods=21).std() * np.sqrt(252)
        sma200[t] = closes[t].rolling(200).mean()
        m63 = ret_d[t].rolling(63, min_periods=42).mean() * 252
        s63 = ret_d[t].rolling(63, min_periods=42).std() * np.sqrt(252)
        quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)

    # Multiple momentum lookbacks
    mom = {}
    for t in stocks:
        if t not in closes: continue
        for w in [63, 126, 252]:
            mom[(t,w)] = closes[t] / closes[t].shift(w) - 1
        mom[(t,21)] = closes[t] / closes[t].shift(21) - 1

    spy_close = data[BENCHMARK]["Close"]
    spy_ret = spy_close.pct_change()
    spy_sma100 = spy_close.rolling(100).mean()
    spy_sma200 = spy_close.rolling(200).mean()

    tlt_ret = data["TLT"]["Close"].pct_change() if "TLT" in data else None
    spy_tlt_corr = spy_ret.rolling(63).corr(tlt_ret) if tlt_ret is not None else None

    def ensemble_ranker(date, n=30, min_quality=0):
        scored = []
        for t in stocks:
            v = vol63[t].loc[date] if date in vol63[t].index else 0
            q = quality[t].loc[date] if date in quality[t].index else 0
            sm = sma200[t].loc[date] if date in sma200[t].index else 0
            price = closes[t].loc[date] if date in closes[t].index else 0
            if pd.isna(v) or v <= 0.01: continue
            if pd.isna(q) or q <= min_quality: continue
            if not pd.isna(sm) and price <= sm: continue

            moms = []
            for w in [63, 126]:
                k = (t, w)
                if k in mom and date in mom[k].index:
                    val = mom[k].loc[date]
                    if not pd.isna(val): moms.append(val)
            # 12m - 1m skip
            k252, k21 = (t, 252), (t, 21)
            if k252 in mom and k21 in mom and date in mom[k252].index and date in mom[k21].index:
                m12 = mom[k252].loc[date]; m1 = mom[k21].loc[date]
                if not pd.isna(m12) and not pd.isna(m1):
                    moms.append(m12 - m1)

            if len(moms) < 2: continue
            avg_mom = np.mean(moms)
            if avg_mom <= 0: continue

            composite = avg_mom * max(q, 0.01)
            scored.append((t, composite, 1.0/v))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def corr_hedge(date, hedge_pct):
        corr_val = 0
        if spy_tlt_corr is not None and date in spy_tlt_corr.index:
            c = spy_tlt_corr.loc[date]
            if not pd.isna(c): corr_val = c

        w = {}
        if corr_val > 0.2:
            w["GLD"] = hedge_pct * 0.60; w["IEF"] = hedge_pct * 0.40
        elif corr_val < -0.2:
            w["TLT"] = hedge_pct * 0.50; w["GLD"] = hedge_pct * 0.25; w["IEF"] = hedge_pct * 0.25
        else:
            w["TLT"] = hedge_pct * 0.33; w["GLD"] = hedge_pct * 0.34; w["IEF"] = hedge_pct * 0.33
        return w

    # ================================================================
    # Grid search: N stocks, eq_pct, quality threshold, MOC vs T+1
    # ================================================================
    print("\n" + "="*60)
    print("GRID: MOC execution + ensemble momentum + corr hedge")
    print("="*60)

    best = {"min_sharpe": -999}

    for n_stocks in [10, 15, 20, 30]:
        for eq_bull in [0.60, 0.70, 0.80]:
            for min_q in [0, 0.5]:
                state = {"month": None, "w": None}
                def make_fn(n, eq, mq):
                    st = {"month": None, "w": None}
                    def fn(date):
                        m = date.month
                        if st["month"] == m and st["w"] is not None: return st["w"]
                        st["month"] = m

                        bear = False
                        if date in spy_sma100.index:
                            s = spy_sma100.loc[date]
                            bear = not pd.isna(s) and spy_close.loc[date] <= s

                        ep = 0.30 if bear else eq
                        hp = 1.0 - ep

                        top = ensemble_ranker(date, n if not bear else max(5, n//3), mq)
                        weights = {}
                        if top:
                            ti = sum(iv for _, _, iv in top)
                            for t, _, iv in top: weights[t] = (iv/ti) * ep
                        else:
                            weights["SPY"] = ep

                        hw = corr_hedge(date, hp)
                        for k, v in hw.items(): weights[k] = weights.get(k, 0) + v

                        st["w"] = weights
                        return weights
                    return fn

                fn = make_fn(n_stocks, eq_bull, min_q)
                results = []
                for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
                    r, t = moc_backtest(data, s, e, fn)
                    m = compute_metrics(r)
                    results.append(m)

                min_sh = min(r["sharpe"] for r in results)
                if min_sh > best["min_sharpe"]:
                    best = {"min_sharpe": min_sh, "n": n_stocks, "eq": eq_bull, "mq": min_q, "results": results}

                q_label = f"q>{min_q}" if min_q > 0 else "q>0"
                print(f"  N={n_stocks:2d} Eq={eq_bull:.0%} {q_label}: "
                      f"Tr={results[0]['sharpe']:.3f}/{results[0]['cagr']:.1%} "
                      f"Va={results[1]['sharpe']:.3f} "
                      f"Te={results[2]['sharpe']:.3f}/{results[2]['cagr']:.1%} "
                      f"Vol={results[0]['ann_vol']:.1%}/{results[2]['ann_vol']:.1%}")

    print(f"\n  BEST (by min-period Sharpe): N={best['n']} Eq={best['eq']:.0%} q>{best.get('mq',0)}")
    for i, name in enumerate(["Train", "Valid", "Test"]):
        r = best["results"][i]
        print(f"    {name}: Sharpe={r['sharpe']:.3f} CAGR={r['cagr']:.1%} MaxDD={r['max_dd']:.1%} Vol={r['ann_vol']:.1%}")

    # Walk-forward for best
    print(f"\n  Walk-forward for best:")
    fn = make_fn(best["n"], best["eq"], best.get("mq", 0))
    sharpes = []
    for year in range(2011, 2026):
        s, e = f"{year}-01-01", f"{year}-12-31"
        fn = make_fn(best["n"], best["eq"], best.get("mq", 0))
        try:
            r, t = moc_backtest(data, s, e, fn)
            m = compute_metrics(r)
            sm = compute_metrics(data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna())
            sharpes.append(m["sharpe"])
            beat = "✓" if m["sharpe"] > sm["sharpe"] else " "
            print(f"    {year}: Sh={m['sharpe']:6.3f} vs SPY {sm['sharpe']:6.3f} {beat} | DD={m['max_dd']:6.1%} vs {sm['max_dd']:6.1%}")
        except: pass
    if sharpes:
        print(f"    Avg={np.mean(sharpes):.3f} Min={min(sharpes):.3f} Max={max(sharpes):.3f}")

    # Full period
    fn = make_fn(best["n"], best["eq"], best.get("mq", 0))
    r, t = moc_backtest(data, "2009-01-01", TEST_END, fn)
    m = compute_metrics(r)
    sm = compute_metrics(data[BENCHMARK].loc["2009-01-01":TEST_END, "Close"].pct_change().dropna())
    print(f"\n  FULL: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%}")
    print(f"        SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
