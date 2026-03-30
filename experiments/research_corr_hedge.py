#!/usr/bin/env python3
"""
Dynamic correlation-based hedging:
When stock-bond correlation is POSITIVE (like 2022), bonds aren't a hedge.
Shift to gold or reduce exposure.

Also: test ADAPTIVE momentum lookback and ensemble of lookbacks.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
from research_hybrid import simple_backtest, compute_metrics

BENCHMARK = "SPY"
SAFE_HAVENS = ["TLT", "GLD", "IEF"]
NON_STOCKS = set(["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC",
                   "TLT","GLD","IEF","SPY","QQQ","IWM","DIA","HYG","SLV","USO"])

def run_tests(data):
    stocks = [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]
    print(f"Stock universe: {len(stocks)} stocks")

    # Precompute
    closes = {}; ret_d = {}; vol63 = {}; sma200 = {}; quality = {}
    mom_windows = {}
    for t in stocks + SAFE_HAVENS:
        if t not in data: continue
        df = data[t]; closes[t] = df["Close"]; ret_d[t] = df["Close"].pct_change()
        vol63[t] = ret_d[t].rolling(63, min_periods=21).std() * np.sqrt(252)
        sma200[t] = closes[t].rolling(200).mean()
        m63 = ret_d[t].rolling(63, min_periods=42).mean() * 252
        s63 = ret_d[t].rolling(63, min_periods=42).std() * np.sqrt(252)
        quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)
        # Multiple momentum windows
        for w in [63, 126, 252]:
            mom_windows[(t, w)] = closes[t] / closes[t].shift(w) - 1
        mom_windows[(t, "skip")] = (closes[t] / closes[t].shift(252) - 1) - (closes[t] / closes[t].shift(21) - 1)

    spy_close = data[BENCHMARK]["Close"]
    spy_ret = spy_close.pct_change()
    spy_sma100 = spy_close.rolling(100).mean()

    # Stock-bond correlation
    tlt_ret = data["TLT"]["Close"].pct_change() if "TLT" in data else None
    gld_ret = data["GLD"]["Close"].pct_change() if "GLD" in data else None
    spy_tlt_corr = spy_ret.rolling(63).corr(tlt_ret) if tlt_ret is not None else None
    spy_gld_corr = spy_ret.rolling(63).corr(gld_ret) if gld_ret is not None else None

    def stock_ranker(date, n=30):
        scored = []
        for t in stocks:
            key = (t, "skip")
            if key not in mom_windows or date not in mom_windows[key].index: continue
            ms = mom_windows[key].loc[date]
            q = quality[t].loc[date] if date in quality[t].index else 0
            v = vol63[t].loc[date] if date in vol63[t].index else 0
            sm = sma200[t].loc[date] if date in sma200[t].index else 0
            price = closes[t].loc[date] if date in closes[t].index else 0
            if pd.isna(ms) or pd.isna(v) or v <= 0.01: continue
            if pd.isna(q): q = 0
            if ms <= 0 or q <= 0: continue
            if not pd.isna(sm) and price <= sm: continue
            scored.append((t, ms * max(q, 0.01), 1.0/v))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    # ================================================================
    # CH1: Correlation-adaptive hedge allocation
    # When SPY-TLT corr > 0: shift bonds to gold
    # ================================================================
    print("\n" + "="*60)
    print("CH1: Correlation-Adaptive Hedging")
    print("    SPY-TLT corr > 0 → shift to GLD; < 0 → keep TLT")
    print("="*60)

    state1 = {"month": None, "w": None}
    def ch1(date):
        m = date.month
        if state1["month"] == m and state1["w"] is not None: return state1["w"]
        state1["month"] = m

        bear = False
        if date in spy_sma100.index:
            s = spy_sma100.loc[date]
            bear = not pd.isna(s) and spy_close.loc[date] <= s

        eq_pct = 0.30 if bear else 0.80
        hedge_pct = 1.0 - eq_pct

        top = stock_ranker(date, 30 if not bear else 10)
        weights = {}
        if top:
            ti = sum(iv for _, _, iv in top)
            for t, _, iv in top: weights[t] = (iv/ti) * eq_pct
        else:
            weights["SPY"] = eq_pct

        # Correlation-adaptive hedge
        corr_val = 0
        if spy_tlt_corr is not None and date in spy_tlt_corr.index:
            c = spy_tlt_corr.loc[date]
            if not pd.isna(c): corr_val = c

        if corr_val > 0.2:
            # Positive correlation: TLT is NOT a hedge → shift to GLD + IEF
            weights["GLD"] = hedge_pct * 0.60
            weights["IEF"] = hedge_pct * 0.40
        elif corr_val < -0.2:
            # Strong negative corr: TLT IS a great hedge
            weights["TLT"] = hedge_pct * 0.50
            weights["GLD"] = hedge_pct * 0.25
            weights["IEF"] = hedge_pct * 0.25
        else:
            # Neutral: balanced
            weights["TLT"] = hedge_pct * 0.33
            weights["GLD"] = hedge_pct * 0.34
            weights["IEF"] = hedge_pct * 0.33

        state1["w"] = weights
        return weights

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        state1 = {"month": None, "w": None}
        r, t = simple_backtest(data, s, e, ch1)
        m = compute_metrics(r)
        spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
        sm = compute_metrics(spy_r)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Sortino={m['sortino']:.3f}")
        print(f"        SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%}")

    # ================================================================
    # CH2: Ensemble momentum (average of 63d, 126d, 252d-21d)
    # ================================================================
    print("\n" + "="*60)
    print("CH2: Ensemble Momentum + Correlation Hedge")
    print("="*60)

    state2 = {"month": None, "w": None}
    def ch2(date):
        m = date.month
        if state2["month"] == m and state2["w"] is not None: return state2["w"]
        state2["month"] = m

        bear = False
        if date in spy_sma100.index:
            s = spy_sma100.loc[date]
            bear = not pd.isna(s) and spy_close.loc[date] <= s

        eq_pct = 0.30 if bear else 0.80
        hedge_pct = 1.0 - eq_pct

        # Ensemble momentum: average z-score across 3 lookbacks
        scored = []
        for t in stocks:
            v = vol63[t].loc[date] if date in vol63[t].index else 0
            q = quality[t].loc[date] if date in quality[t].index else 0
            sm = sma200[t].loc[date] if date in sma200[t].index else 0
            price = closes[t].loc[date] if date in closes[t].index else 0
            if pd.isna(v) or v <= 0.01: continue
            if not pd.isna(sm) and price <= sm: continue
            if pd.isna(q) or q <= 0: continue

            # Get momentum at each lookback
            moms = []
            for w in [63, 126]:
                key = (t, w)
                if key in mom_windows and date in mom_windows[key].index:
                    val = mom_windows[key].loc[date]
                    if not pd.isna(val): moms.append(val)
            key_skip = (t, "skip")
            if key_skip in mom_windows and date in mom_windows[key_skip].index:
                val = mom_windows[key_skip].loc[date]
                if not pd.isna(val): moms.append(val)

            if len(moms) < 2: continue
            avg_mom = np.mean(moms)
            if avg_mom <= 0: continue

            composite = avg_mom * max(q, 0.01)
            scored.append((t, composite, 1.0/v))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:30 if not bear else 10]

        weights = {}
        if top:
            ti = sum(iv for _, _, iv in top)
            for t, _, iv in top: weights[t] = (iv/ti) * eq_pct
        else:
            weights["SPY"] = eq_pct

        # Correlation-adaptive hedge (same as CH1)
        corr_val = 0
        if spy_tlt_corr is not None and date in spy_tlt_corr.index:
            c = spy_tlt_corr.loc[date]
            if not pd.isna(c): corr_val = c

        if corr_val > 0.2:
            weights["GLD"] = hedge_pct * 0.60
            weights["IEF"] = hedge_pct * 0.40
        elif corr_val < -0.2:
            weights["TLT"] = hedge_pct * 0.50
            weights["GLD"] = hedge_pct * 0.25
            weights["IEF"] = hedge_pct * 0.25
        else:
            weights["TLT"] = hedge_pct * 0.33
            weights["GLD"] = hedge_pct * 0.34
            weights["IEF"] = hedge_pct * 0.33

        state2["w"] = weights
        return weights

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        state2 = {"month": None, "w": None}
        r, t = simple_backtest(data, s, e, ch2)
        m = compute_metrics(r)
        spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
        sm = compute_metrics(spy_r)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Sortino={m['sortino']:.3f}")
        print(f"        SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%}")

    # Walk-forward for CH2
    print("\n  Walk-forward CH2:")
    sharpes = []
    for year in range(2011, 2026):
        s, e = f"{year}-01-01", f"{year}-12-31"
        state2 = {"month": None, "w": None}
        try:
            r, t = simple_backtest(data, s, e, ch2)
            m = compute_metrics(r)
            spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
            sm = compute_metrics(spy_r)
            beat = "✓" if m["sharpe"] > sm["sharpe"] else " "
            sharpes.append(m["sharpe"])
            print(f"    {year}: Sh={m['sharpe']:6.3f} vs SPY {sm['sharpe']:6.3f} {beat} | "
                  f"CAGR={m['cagr']:5.1%} MaxDD={m['max_dd']:6.1%} vs {sm['max_dd']:6.1%}")
        except: pass
    if sharpes:
        print(f"    Avg: {np.mean(sharpes):.3f} | Min: {min(sharpes):.3f} | Max: {max(sharpes):.3f}")

    # Full period
    state2 = {"month": None, "w": None}
    r, t = simple_backtest(data, "2009-01-01", TEST_END, ch2)
    m = compute_metrics(r)
    sm = compute_metrics(data[BENCHMARK].loc["2009-01-01":TEST_END, "Close"].pct_change().dropna())
    print(f"\n  FULL: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%}")
    print(f"        SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%} MaxDD={sm['max_dd']:.1%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
