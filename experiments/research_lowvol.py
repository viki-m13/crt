#!/usr/bin/env python3
"""
Low-vol momentum: one of the strongest anomalies in finance.
Select low-volatility stocks with positive momentum.
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


def run_tests(data):
    stocks = [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]
    print(f"Stock universe: {len(stocks)} stocks")

    closes = {}; ret_d = {}; vol63 = {}; vol252 = {}; mom252 = {}; mom21 = {}; sma200 = {}
    quality = {}
    for t in stocks + SAFE_HAVENS:
        if t not in data: continue
        df = data[t]; closes[t] = df["Close"]; ret_d[t] = df["Close"].pct_change()
        vol63[t] = ret_d[t].rolling(63, min_periods=21).std() * np.sqrt(252)
        vol252[t] = ret_d[t].rolling(252, min_periods=126).std() * np.sqrt(252)
        mom252[t] = closes[t] / closes[t].shift(252) - 1
        mom21[t] = closes[t] / closes[t].shift(21) - 1
        sma200[t] = closes[t].rolling(200).mean()
        m63 = ret_d[t].rolling(63, min_periods=42).mean() * 252
        s63 = ret_d[t].rolling(63, min_periods=42).std() * np.sqrt(252)
        quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)

    spy_close = data[BENCHMARK]["Close"]
    spy_sma100 = spy_close.rolling(100).mean()

    # =================================================================
    # L1: Low-Vol Momentum — pick stocks in bottom HALF of vol that have
    #     positive momentum. Weight equally (since already low vol).
    # =================================================================
    print("\n" + "="*60)
    print("L1: Low-Vol Momentum (bottom 50% vol + positive 12m mom)")
    print("="*60)

    def core_l1(date, n=20, eq_pct=0.70):
        # Get vol for all stocks
        all_vols = []
        for t in stocks:
            if t in vol252 and date in vol252[t].index:
                v = vol252[t].loc[date]
                if not pd.isna(v) and v > 0:
                    all_vols.append((t, v))

        if len(all_vols) < 10:
            return {"SPY": eq_pct, "TLT": 0.15, "GLD": 0.15}

        # Bottom 50% by vol
        all_vols.sort(key=lambda x: x[1])
        low_vol_cutoff = len(all_vols) // 2
        low_vol_stocks = set(t for t, _ in all_vols[:low_vol_cutoff])

        # Among low-vol, pick those with best momentum
        scored = []
        for t in low_vol_stocks:
            if t not in mom252 or date not in mom252[t].index: continue
            m12 = mom252[t].loc[date]
            m1 = mom21[t].loc[date] if t in mom21 and date in mom21[t].index else 0
            sm = sma200[t].loc[date] if t in sma200 and date in sma200[t].index else 0
            price = closes[t].loc[date] if date in closes[t].index else 0

            if pd.isna(m12): continue
            mom_score = m12 - (m1 if not pd.isna(m1) else 0)
            if mom_score <= 0: continue
            if not pd.isna(sm) and price <= sm: continue

            v = vol63[t].loc[date] if t in vol63 and date in vol63[t].index else 1
            if pd.isna(v) or v <= 0: continue
            scored.append((t, mom_score, 1.0 / v))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:n]

        weights = {}
        if top:
            total_iv = sum(iv for _, _, iv in top)
            for t, _, iv in top:
                weights[t] = (iv / total_iv) * eq_pct
        else:
            weights["SPY"] = eq_pct

        hp = 1.0 - eq_pct
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

    for n in [10, 15, 20, 30]:
        for eq in [0.60, 0.70, 0.80]:
            fn = lambda date, _n=n, _e=eq: core_l1(date, _n, _e)
            results = []
            for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
                r, t = simple_backtest(data, s, e, fn)
                m = compute_metrics(r)
                results.append(m)
            print(f"  N={n:2d} Eq={eq:.0%}: Train={results[0]['sharpe']:.3f} Valid={results[1]['sharpe']:.3f} Test={results[2]['sharpe']:.3f} | "
                  f"Vol={results[0]['ann_vol']:.1%}/{results[2]['ann_vol']:.1%} | DD={results[0]['max_dd']:.1%}/{results[2]['max_dd']:.1%}")

    # =================================================================
    # L2: Low-Vol Momentum + Regime
    # =================================================================
    print("\n" + "="*60)
    print("L2: Low-Vol Momentum + Regime Adaptation")
    print("="*60)

    def core_l2(date, n=20):
        bear = False
        if date in spy_sma100.index:
            s = spy_sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True

        eq_pct = 0.30 if bear else 0.70
        return core_l1(date, n if not bear else min(n, 10), eq_pct)

    for n in [15, 20, 30]:
        fn = lambda date, _n=n: core_l2(date, _n)
        results = []
        for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
            r, t = simple_backtest(data, s, e, fn)
            m = compute_metrics(r)
            results.append(m)
        print(f"  N={n:2d}: Train={results[0]['sharpe']:.3f}/{results[0]['cagr']:.1%} Valid={results[1]['sharpe']:.3f}/{results[1]['cagr']:.1%} Test={results[2]['sharpe']:.3f}/{results[2]['cagr']:.1%} | "
              f"DD={results[0]['max_dd']:.1%}/{results[1]['max_dd']:.1%}/{results[2]['max_dd']:.1%}")

    # =================================================================
    # L3: Ultra-quality: quality > 1.5 + low vol + momentum
    # =================================================================
    print("\n" + "="*60)
    print("L3: Ultra-Quality Filter (quality > 1.0 + low vol + momentum)")
    print("="*60)

    def core_l3(date, eq_pct=0.70):
        scored = []
        for t in stocks:
            if t not in vol252 or t not in mom252 or t not in quality: continue
            if date not in vol252[t].index or date not in quality[t].index: continue

            v252 = vol252[t].loc[date]
            m12 = mom252[t].loc[date]
            m1 = mom21[t].loc[date] if t in mom21 and date in mom21[t].index else 0
            q = quality[t].loc[date]
            v = vol63[t].loc[date] if t in vol63 and date in vol63[t].index else 1
            sm = sma200[t].loc[date] if t in sma200 and date in sma200[t].index else 0
            price = closes[t].loc[date] if date in closes[t].index else 0

            if pd.isna(m12) or pd.isna(q) or pd.isna(v252) or pd.isna(v): continue
            if v <= 0 or v252 <= 0: continue

            # Ultra-quality filters
            if q <= 1.0: continue  # Must have rolling Sharpe > 1.0
            if v252 > 0.30: continue  # Must have below-average vol
            mom_score = m12 - (m1 if not pd.isna(m1) else 0)
            if mom_score <= 0: continue
            if not pd.isna(sm) and price <= sm: continue

            composite = q * mom_score
            scored.append((t, composite, 1.0 / v))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:20]

        weights = {}
        if top:
            total_iv = sum(iv for _, _, iv in top)
            for t, _, iv in top:
                weights[t] = (iv / total_iv) * eq_pct
        else:
            weights["SPY"] = eq_pct

        hp = 1.0 - eq_pct
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

    for eq in [0.50, 0.60, 0.70, 0.80]:
        fn = lambda date, _e=eq: core_l3(date, _e)
        results = []
        for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
            r, t = simple_backtest(data, s, e, fn)
            m = compute_metrics(r)
            results.append(m)
        print(f"  Eq={eq:.0%}: Train={results[0]['sharpe']:.3f}/{results[0]['cagr']:.1%} Valid={results[1]['sharpe']:.3f}/{results[1]['cagr']:.1%} Test={results[2]['sharpe']:.3f}/{results[2]['cagr']:.1%} | "
              f"Vol={results[0]['ann_vol']:.1%}/{results[2]['ann_vol']:.1%}")

    # =================================================================
    # L4: Ultra-quality + Regime
    # =================================================================
    print("\n" + "="*60)
    print("L4: Ultra-Quality + Regime")
    print("="*60)

    def core_l4(date):
        bear = False
        if date in spy_sma100.index:
            s = spy_sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True
        eq = 0.30 if bear else 0.70
        return core_l3(date, eq)

    fn = core_l4
    results = []
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = simple_backtest(data, s, e, fn)
        m = compute_metrics(r)
        spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
        sm = compute_metrics(spy_r)
        results.append(m)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Sortino={m['sortino']:.3f}")
        print(f"          SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%}")

    # Walk-forward
    print("\n  Walk-forward:")
    for year in range(2011, 2026):
        s, e = f"{year}-01-01", f"{year}-12-31"
        try:
            r, t = simple_backtest(data, s, e, fn)
            m = compute_metrics(r)
            spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
            sm = compute_metrics(spy_r)
            beat = "✓" if m["sharpe"] > sm["sharpe"] else " "
            print(f"    {year}: Sharpe={m['sharpe']:6.3f} SPY={sm['sharpe']:6.3f} {beat} | "
                  f"CAGR={m['cagr']:5.1%} | MaxDD={m['max_dd']:6.1%} vs {sm['max_dd']:6.1%}")
        except: pass


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
