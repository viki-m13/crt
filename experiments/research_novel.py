#!/usr/bin/env python3
"""
Step 6: Novel strategies.
1. Sector Quality Score (rolling Sharpe of each sector)
2. Vol targeting overlay
3. Asymmetric hedging (progressive, not binary)
4. Momentum persistence scoring
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
from sector_strategy_research import (
    backtest_sector_strategy, compute_metrics, spy_metrics,
    print_results, SECTOR_ETFS, BENCHMARK
)

def run_tests(data):
    available_sectors = [e for e in SECTOR_ETFS if e in data]

    spy_close = data[BENCHMARK]["Close"]
    spy_ret = spy_close.pct_change()
    sma100 = spy_close.rolling(100).mean()
    sma200 = spy_close.rolling(200).mean()
    spy_vol21 = spy_ret.rolling(21).std() * np.sqrt(252)

    # Precompute sector quality scores (rolling Sharpe)
    sec_close = {e: data[e]["Close"] for e in available_sectors}
    sec_ret = {e: data[e]["Close"].pct_change() for e in available_sectors}
    sec_vol63 = {e: sec_ret[e].rolling(63, min_periods=21).std() * np.sqrt(252) for e in available_sectors}

    # Sector quality = rolling 63-day Sharpe ratio
    sec_quality = {}
    for etf in available_sectors:
        r = sec_ret[etf]
        mean63 = r.rolling(63, min_periods=42).mean() * 252
        std63 = r.rolling(63, min_periods=42).std() * np.sqrt(252)
        sec_quality[etf] = (mean63 - 0.02) / std63.clip(lower=0.01)

    # Sector momentum persistence: fraction of positive days
    sec_persistence = {}
    for etf in available_sectors:
        r = sec_ret[etf]
        sec_persistence[etf] = r.rolling(63, min_periods=42).apply(lambda x: (x > 0).mean(), raw=True)

    # Composite sector score: quality * persistence / vol
    sec_composite = {}
    for etf in available_sectors:
        q = sec_quality[etf]
        p = sec_persistence[etf]
        v = sec_vol63[etf]
        sec_composite[etf] = q * p / v.clip(lower=0.01)

    # Average sector quality (market breadth quality)
    all_quality = pd.DataFrame(sec_quality)
    avg_quality = all_quality.mean(axis=1)
    quality_pctile = avg_quality.rolling(252, min_periods=126).rank(pct=True)

    # SPY quality
    spy_mean63 = spy_ret.rolling(63, min_periods=42).mean() * 252
    spy_std63 = spy_ret.rolling(63, min_periods=42).std() * np.sqrt(252)
    spy_quality = (spy_mean63 - 0.02) / spy_std63.clip(lower=0.01)

    DEFENSIVE = ["XLP", "XLU", "XLV"]

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

    def inv_vol_weights(date, sectors):
        weights = {}
        for etf in sectors:
            if etf in sec_vol63 and date in sec_vol63[etf].index:
                vol = sec_vol63[etf].loc[date]
                if not pd.isna(vol) and vol > 0.01:
                    weights[etf] = 1.0 / vol
        if weights:
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        return weights

    # =================================================================
    # Strategy J: Quality-Based Sector Selection
    # =================================================================
    print("\n" + "="*60)
    print("J: Quality-Based Sector Selection (rolling Sharpe)")
    print("="*60)

    def core_j(data, date, idx):
        # Select sectors with positive quality (positive rolling Sharpe)
        good_sectors = []
        for etf in available_sectors:
            if etf in sec_quality and date in sec_quality[etf].index:
                q = sec_quality[etf].loc[date]
                if not pd.isna(q) and q > 0:
                    good_sectors.append(etf)

        if len(good_sectors) < 3:
            # Few good sectors → defensive
            return {"invested": True, "weights": {"TLT": 0.4, "GLD": 0.3, "XLP": 0.15, "XLV": 0.15}}

        # Weight good sectors by inverse vol
        w = inv_vol_weights(date, good_sectors)
        return {"invested": True, "weights": w}

    sig_j = monthly_wrap(core_j)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_j)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy K: Quality + SMA Gate + Progressive Hedging
    # =================================================================
    print("\n" + "="*60)
    print("K: Quality + SMA100 + Progressive Hedge")
    print("="*60)

    def core_k(data, date, idx):
        # Base: SMA100 gate
        bear = False
        if date in sma100.index:
            s = sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True

        if bear:
            # Bear: defensive + heavy hedge
            return {"invested": True, "weights": {"TLT": 0.4, "GLD": 0.3, "XLP": 0.15, "XLV": 0.15}}

        # Bull: select by quality
        good = []
        for etf in available_sectors:
            if etf in sec_quality and date in sec_quality[etf].index:
                q = sec_quality[etf].loc[date]
                if not pd.isna(q) and q > 0.5:  # quality threshold
                    good.append((etf, q))

        if len(good) < 2:
            # Low quality → moderate hedge
            def_w = inv_vol_weights(date, [e for e in DEFENSIVE if e in available_sectors])
            weights = {}
            for etf, w in def_w.items():
                weights[etf] = w * 0.5
            weights["TLT"] = 0.3
            weights["GLD"] = 0.2
            return {"invested": True, "weights": weights}

        # High quality → full equity
        good_sectors = [etf for etf, _ in good]
        w = inv_vol_weights(date, good_sectors)
        return {"invested": True, "weights": w}

    sig_k = monthly_wrap(core_k)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_k)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy L: Vol Targeting + Strategy H
    # =================================================================
    print("\n" + "="*60)
    print("L: Vol-Targeted Defensive/Offensive Rotation")
    print("="*60)

    # Compute running portfolio vol for Strategy H
    def core_l(data, date, idx):
        # Strategy H logic
        bear = False
        if date in sma100.index:
            s = sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True

        if bear:
            def_w = inv_vol_weights(date, [e for e in DEFENSIVE if e in available_sectors])
            weights = {}
            for etf, w in def_w.items():
                weights[etf] = w * 0.4
            weights["TLT"] = 0.35
            weights["GLD"] = 0.25
            return {"invested": True, "weights": weights}

        # Bull: all sectors inv-vol
        w = inv_vol_weights(date, available_sectors)

        # Vol targeting: reduce exposure if portfolio vol is high
        if date in spy_vol21.index:
            v = spy_vol21.loc[date]
            if not pd.isna(v):
                target = 0.10
                scale = min(target / max(v, 0.01), 1.0)
                if scale < 0.9:
                    # Reduce equity, add hedge
                    hedge = 1.0 - scale
                    weights = {}
                    for etf, wt in w.items():
                        weights[etf] = wt * scale
                    weights["TLT"] = weights.get("TLT", 0) + hedge * 0.5
                    weights["GLD"] = weights.get("GLD", 0) + hedge * 0.5
                    return {"invested": True, "weights": weights}

        return {"invested": True, "weights": w}

    sig_l = monthly_wrap(core_l)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_l)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy M: Composite Score (quality + persistence + inv-vol)
    # =================================================================
    print("\n" + "="*60)
    print("M: Composite Score + Adaptive Hedge")
    print("="*60)

    def core_m(data, date, idx):
        # Market regime
        bear = False
        if date in sma100.index:
            s = sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True

        if bear:
            return {"invested": True, "weights": {"TLT": 0.4, "GLD": 0.3, "XLP": 0.15, "XLV": 0.15}}

        # Composite score for each sector
        scored = []
        for etf in available_sectors:
            if etf in sec_composite and date in sec_composite[etf].index:
                cs = sec_composite[etf].loc[date]
                if not pd.isna(cs):
                    scored.append((etf, cs))

        if not scored:
            return {"invested": True, "weights": {BENCHMARK: 1.0}}

        scored.sort(key=lambda x: x[1], reverse=True)

        # Top half with positive score get allocated
        n_good = max(3, len(scored) // 2)
        top = [(etf, s) for etf, s in scored[:n_good] if s > 0]

        if len(top) < 2:
            # No high-quality sectors → moderate hedge
            return {"invested": True, "weights": {"TLT": 0.3, "GLD": 0.2, "XLP": 0.25, "XLV": 0.25}}

        # Weight by inverse vol
        w = inv_vol_weights(date, [etf for etf, _ in top])
        return {"invested": True, "weights": w}

    sig_m = monthly_wrap(core_m)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_m)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy N: Weekly rebal + quality + trend
    # =================================================================
    print("\n" + "="*60)
    print("N: Weekly Quality + Trend + Progressive Hedge")
    print("="*60)

    def weekly_wrap(fn):
        state = {"week": None, "sig": None}
        def wrapped(data, date, idx):
            if date.weekday() != 0 and state["sig"] is not None:
                return state["sig"]
            state["sig"] = fn(data, date, idx)
            return state["sig"]
        return wrapped

    def core_n(data, date, idx):
        # Trend check
        bear = False
        if date in sma100.index:
            s = sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True

        if bear:
            return {"invested": True, "weights": {"TLT": 0.4, "GLD": 0.3, "XLP": 0.15, "XLV": 0.15}}

        # Quality-weighted allocation
        good = []
        for etf in available_sectors:
            if etf in sec_quality and date in sec_quality[etf].index:
                q = sec_quality[etf].loc[date]
                vol = sec_vol63[etf].loc[date] if date in sec_vol63[etf].index else 1
                if not pd.isna(q) and not pd.isna(vol) and vol > 0 and q > 0:
                    good.append((etf, q / vol))

        if len(good) < 2:
            return {"invested": True, "weights": {"TLT": 0.3, "GLD": 0.2, "XLP": 0.25, "XLV": 0.25}}

        good.sort(key=lambda x: x[1], reverse=True)
        top = good[:5]
        total = sum(s for _, s in top)
        weights = {etf: s / total for etf, s in top}
        return {"invested": True, "weights": weights}

    sig_n = weekly_wrap(core_n)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_n)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy O: Oracle (perfect monthly foresight) — upper bound
    # =================================================================
    print("\n" + "="*60)
    print("O: ORACLE — perfect 1-month foresight (upper bound)")
    print("="*60)

    def oracle(data, date, idx):
        # Look ahead 21 trading days — perfect foresight
        best_etf, best_ret = None, -999
        for etf in available_sectors:
            df = data[etf]
            if date in df.index:
                si = df.index.get_loc(date)
                if si + 21 < len(df):
                    future_ret = df.iloc[si+21]["Close"] / df.iloc[si]["Close"] - 1
                    if future_ret > best_ret:
                        best_etf, best_ret = etf, future_ret
        if best_etf and best_ret > 0:
            return {"invested": True, "weights": {best_etf: 1.0}}
        return {"invested": True, "weights": {"TLT": 1.0}}

    sig_o = monthly_wrap(oracle)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_o)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(f"{name} (ORACLE - CHEATING)", m, spy, trades)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
