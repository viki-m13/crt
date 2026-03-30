#!/usr/bin/env python3
"""
Weekly rebalancing with multiple alpha signals:
1. Buy-the-dip-in-uptrend (reversal + momentum)
2. Concentrated top picks (5-10 stocks)
3. Bond/gold carry hedge
4. Adaptive vol scaling

Also test: combining L/S alpha with long-only momentum
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


def weekly_backtest(data, start, end, weight_fn, tx_bps=5):
    """Weekly rebalance on Mondays, T+1 open execution."""
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    slip = tx_bps / 10000
    daily_rets = []; current_w = {}; last_week = None; trades = 0

    for date in dates:
        idx = spy.index.get_loc(date)
        if idx < 300:
            daily_rets.append(0.0); continue

        # Rebalance on Monday (or first day after weekend)
        week = date.isocalendar()[1]
        rebalance = (last_week is not None and week != last_week and date.weekday() <= 1)
        last_week = week

        if rebalance:
            new_w = weight_fn(date)
            dr = 0.0
            # Sell changed positions (overnight)
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
            # Buy new positions (open to close)
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

    # Precompute
    closes = {}; ret_d = {}; vol63 = {}; mom252 = {}; mom21 = {}; mom5 = {}
    sma200 = {}; quality = {}
    for t in stocks + SAFE_HAVENS:
        if t not in data: continue
        df = data[t]; closes[t] = df["Close"]; ret_d[t] = df["Close"].pct_change()
        vol63[t] = ret_d[t].rolling(63, min_periods=21).std() * np.sqrt(252)
        mom252[t] = closes[t] / closes[t].shift(252) - 1
        mom21[t] = closes[t] / closes[t].shift(21) - 1
        mom5[t] = closes[t] / closes[t].shift(5) - 1
        sma200[t] = closes[t].rolling(200).mean()
        m63 = ret_d[t].rolling(63, min_periods=42).mean() * 252
        s63 = ret_d[t].rolling(63, min_periods=42).std() * np.sqrt(252)
        quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)

    spy_close = data[BENCHMARK]["Close"]
    spy_sma100 = spy_close.rolling(100).mean()
    spy_ret = spy_close.pct_change()
    spy_vol21 = spy_ret.rolling(21).std() * np.sqrt(252)

    # ================================================================
    # W1: Buy-the-dip — stocks with strong 12m mom but negative 5d return
    # Weekly rebalance, concentrated (top 5-10)
    # ================================================================
    print("\n" + "="*60)
    print("W1: Buy-the-Dip in Uptrend (weekly, concentrated)")
    print("="*60)

    def dip_ranker(date, n=10, require_dip=True):
        """Rank stocks: must have 12m momentum > 0, quality > 0, above SMA200.
        If require_dip: prefer stocks with negative 5d return (dip buying).
        """
        scored = []
        for t in stocks:
            if t not in mom252 or date not in mom252[t].index: continue
            m12 = mom252[t].loc[date]
            m1 = mom21[t].loc[date] if t in mom21 and date in mom21[t].index else 0
            m5 = mom5[t].loc[date] if t in mom5 and date in mom5[t].index else 0
            q = quality[t].loc[date] if date in quality[t].index else 0
            v = vol63[t].loc[date] if date in vol63[t].index else 0
            sm = sma200[t].loc[date] if date in sma200[t].index else 0
            price = closes[t].loc[date] if date in closes[t].index else 0

            if pd.isna(m12) or pd.isna(v) or v <= 0.01: continue
            if pd.isna(q): q = 0

            mom_score = m12 - (m1 if not pd.isna(m1) else 0)
            if mom_score <= 0: continue
            if q <= 0: continue
            if not pd.isna(sm) and price <= sm: continue

            # Dip bonus: stocks that dipped in last 5 days get score boost
            dip_bonus = 1.0
            if not pd.isna(m5):
                if m5 < -0.02:  # More than 2% dip
                    dip_bonus = 2.0
                elif m5 < 0:
                    dip_bonus = 1.5

            composite = mom_score * max(q, 0.01) * dip_bonus
            scored.append((t, composite, 1.0 / v))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    for n_stocks in [5, 10, 15]:
        for eq_pct in [0.50, 0.60, 0.70]:
            def make_fn(n, ep):
                def fn(date):
                    bear = False
                    if date in spy_sma100.index:
                        s = spy_sma100.loc[date]
                        bear = not pd.isna(s) and spy_close.loc[date] <= s

                    eff_eq = ep * (0.3 if bear else 1.0)
                    eff_hedge = 1.0 - eff_eq

                    top = dip_ranker(date, n if not bear else max(3, n//3))
                    weights = {}
                    if top:
                        total_iv = sum(iv for _, _, iv in top)
                        for t, _, iv in top:
                            weights[t] = (iv / total_iv) * eff_eq
                    else:
                        weights["SPY"] = eff_eq

                    hw = {}
                    for h in SAFE_HAVENS:
                        if h in vol63 and date in vol63[h].index:
                            v = vol63[h].loc[date]
                            if not pd.isna(v) and v > 0:
                                if bear and h in sma200 and date in sma200[h].index:
                                    sm = sma200[h].loc[date]
                                    p = closes[h].loc[date] if date in closes[h].index else 0
                                    hw[h] = (1/v) * (2 if (not pd.isna(sm) and p > sm) else 0.3)
                                else:
                                    hw[h] = 1.0 / v
                    if hw:
                        ht = sum(hw.values())
                        for h, iv in hw.items(): weights[h] = (iv / ht) * eff_hedge
                    else:
                        weights["IEF"] = eff_hedge
                    return weights
                return fn

            fn = make_fn(n_stocks, eq_pct)
            results = []
            for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
                r, t = weekly_backtest(data, s, e, fn)
                m = compute_metrics(r)
                results.append((m, t))

            print(f"  N={n_stocks:2d} Eq={eq_pct:.0%}: "
                  f"Train={results[0][0]['sharpe']:.3f}/{results[0][0]['ann_vol']:.1%} "
                  f"Val={results[1][0]['sharpe']:.3f} "
                  f"Test={results[2][0]['sharpe']:.3f}/{results[2][0]['ann_vol']:.1%} "
                  f"DD={results[0][0]['max_dd']:.1%}/{results[2][0]['max_dd']:.1%} "
                  f"Tr={results[0][1]+results[2][1]}")

    # ================================================================
    # W2: Multi-alpha: momentum + reversal + quality combined
    # Monthly rebal (same as before) but with vol targeting overlay
    # ================================================================
    print("\n" + "="*60)
    print("W2: Multi-Alpha Monthly + Vol Targeting to 8%")
    print("="*60)

    def make_voltarget_fn(target_vol=0.08):
        """Monthly rebal with vol targeting."""
        state = {"month": None, "weights": None}

        def fn(date):
            month = date.month
            if state["month"] == month and state["weights"] is not None:
                return state["weights"]
            state["month"] = month

            bear = False
            if date in spy_sma100.index:
                s = spy_sma100.loc[date]
                bear = not pd.isna(s) and spy_close.loc[date] <= s

            eq_pct = 0.30 if bear else 0.80
            hedge_pct = 1.0 - eq_pct

            # Vol targeting: scale equity exposure
            if date in spy_vol21.index:
                v = spy_vol21.loc[date]
                if not pd.isna(v) and v > 0:
                    scale = min(target_vol / v, 1.0)
                    eq_pct *= scale
                    hedge_pct = 1.0 - eq_pct

            # Stock selection (same as before)
            scored = []
            for t in stocks:
                if t not in mom252 or date not in mom252[t].index: continue
                m12 = mom252[t].loc[date]
                m1 = mom21[t].loc[date] if t in mom21 and date in mom21[t].index else 0
                q = quality[t].loc[date] if date in quality[t].index else 0
                v = vol63[t].loc[date] if date in vol63[t].index else 0
                sm = sma200[t].loc[date] if date in sma200[t].index else 0
                price = closes[t].loc[date] if date in closes[t].index else 0
                if pd.isna(m12) or pd.isna(v) or v <= 0.01: continue
                if pd.isna(q): q = 0
                mom_score = m12 - (m1 if not pd.isna(m1) else 0)
                if mom_score <= 0 or q <= 0: continue
                if not pd.isna(sm) and price <= sm: continue
                p = 0.5  # default persistence
                composite = mom_score * max(q, 0.01) * max(p, 0.4)
                scored.append((t, composite, 1.0 / v))

            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:30]

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
                        if bear and h in sma200 and date in sma200[h].index:
                            sm = sma200[h].loc[date]
                            p = closes[h].loc[date] if date in closes[h].index else 0
                            hw[h] = (1/v) * (2 if (not pd.isna(sm) and p > sm) else 0.3)
                        else:
                            hw[h] = 1.0 / v
            if hw:
                ht = sum(hw.values())
                for h, iv in hw.items(): weights[h] = (iv / ht) * hedge_pct
            else:
                weights["IEF"] = hedge_pct

            state["weights"] = weights
            return weights
        return fn

    from research_hybrid import simple_backtest

    for tv in [0.05, 0.08, 0.10, 0.15, 1.0]:
        fn = make_voltarget_fn(tv)
        results = []
        for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
            r, t = simple_backtest(data, s, e, fn)
            m = compute_metrics(r)
            results.append(m)
        label = f"VT={tv:.0%}" if tv < 1 else "NoVT"
        print(f"  {label:>6}: Train={results[0]['sharpe']:.3f}/{results[0]['ann_vol']:.1%}/{results[0]['cagr']:.1%} "
              f"Val={results[1]['sharpe']:.3f} "
              f"Test={results[2]['sharpe']:.3f}/{results[2]['ann_vol']:.1%}/{results[2]['cagr']:.1%}")

    # ================================================================
    # W3: Alpha stacking — momentum + L/S sector overlay + bonds
    # 50% long-only stocks + 20% L/S sector alpha + 30% bonds
    # ================================================================
    print("\n" + "="*60)
    print("W3: Alpha Stacking: Long stocks + L/S sector + bonds")
    print("="*60)

    SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
    sec_mom = {e: data[e]["Close"] / data[e]["Close"].shift(63) - 1 for e in SECTOR_ETFS if e in data}
    sec_rets = {e: data[e]["Close"].pct_change() for e in SECTOR_ETFS if e in data}

    def alpha_stack(date):
        bear = False
        if date in spy_sma100.index:
            s = spy_sma100.loc[date]
            bear = not pd.isna(s) and spy_close.loc[date] <= s

        weights = {}

        # Layer 1: Long-only stocks (50% bull / 15% bear)
        eq_pct = 0.15 if bear else 0.50
        scored = []
        for t in stocks:
            if t not in mom252 or date not in mom252[t].index: continue
            m12 = mom252[t].loc[date]
            m1 = mom21[t].loc[date] if t in mom21 and date in mom21[t].index else 0
            q = quality[t].loc[date] if date in quality[t].index else 0
            v = vol63[t].loc[date] if date in vol63[t].index else 0
            sm = sma200[t].loc[date] if date in sma200[t].index else 0
            price = closes[t].loc[date] if date in closes[t].index else 0
            if pd.isna(m12) or pd.isna(v) or v <= 0.01: continue
            mom_score = m12 - (m1 if not pd.isna(m1) else 0)
            if mom_score <= 0: continue
            if not pd.isna(q) and q <= 0: continue
            if not pd.isna(sm) and price <= sm: continue
            scored.append((t, mom_score * max(q if not pd.isna(q) else 0.01, 0.01), 1.0/v))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:20]
        if top:
            ti = sum(iv for _, _, iv in top)
            for t, _, iv in top: weights[t] = (iv/ti) * eq_pct

        # Layer 2: L/S sector momentum (20% bull / 5% bear)
        # This is ADDITIVE — adds L/S alpha on top
        ls_pct = 0.05 if bear else 0.20
        sec_scored = []
        for e in SECTOR_ETFS:
            if e in sec_mom and date in sec_mom[e].index:
                m = sec_mom[e].loc[date]
                if not pd.isna(m): sec_scored.append((e, m))
        sec_scored.sort(key=lambda x: x[1], reverse=True)
        if len(sec_scored) >= 6:
            # Long top 3 sectors
            for e, _ in sec_scored[:3]:
                weights[e] = weights.get(e, 0) + ls_pct / 3
            # Short bottom 3 (negative weight = we subtract their return)
            for e, _ in sec_scored[-3:]:
                weights[f"SHORT_{e}"] = ls_pct / 3

        # Layer 3: Bonds/gold (remaining)
        used = eq_pct + ls_pct  # Only count the long side of L/S
        hedge_pct = max(0, 1.0 - used)
        hw = {}
        for h in SAFE_HAVENS:
            if h in vol63 and date in vol63[h].index:
                v = vol63[h].loc[date]
                if not pd.isna(v) and v > 0: hw[h] = 1.0/v
        if hw and hedge_pct > 0:
            ht = sum(hw.values())
            for h, iv in hw.items(): weights[h] = weights.get(h, 0) + (iv/ht) * hedge_pct

        return weights

    # Custom backtest that handles SHORT_ positions
    def alpha_stack_backtest(data, start, end):
        spy = data[BENCHMARK]
        dates = spy.loc[start:end].index
        daily_rets = []; current_w = {}; last_month = None; trades = 0
        slip = 5 / 10000

        for date in dates:
            idx = spy.index.get_loc(date)
            if idx < 300:
                daily_rets.append(0.0); continue
            month = date.month
            rebalance = (last_month is not None and month != last_month)
            last_month = month

            if rebalance:
                new_w = alpha_stack(date)
                dr = 0.0
                # Close all, reopen (simplified)
                for t, w in current_w.items():
                    real_t = t.replace("SHORT_", "")
                    is_short = t.startswith("SHORT_")
                    df = data.get(real_t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            ret = df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1
                            if is_short:
                                dr -= ret * w  # Short position
                            else:
                                dr += ret * w  # Long position
                    trades += 1
                daily_rets.append(dr); current_w = new_w
            else:
                dr = 0.0
                for t, w in current_w.items():
                    real_t = t.replace("SHORT_", "")
                    is_short = t.startswith("SHORT_")
                    df = data.get(real_t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            ret = df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1
                            if is_short:
                                dr -= ret * w
                            else:
                                dr += ret * w
                daily_rets.append(dr)
        return pd.Series(daily_rets, index=dates), trades

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = alpha_stack_backtest(data, s, e)
        m = compute_metrics(r)
        spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
        sm = compute_metrics(spy_r)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Sortino={m['sortino']:.3f}")
        print(f"        SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
