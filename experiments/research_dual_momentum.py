#!/usr/bin/env python3
"""
Step 8: Individual sector trend filters + dual momentum.
Novel: apply trend filter to EACH sector individually, not just SPY.
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
    ALL_ASSETS = available_sectors + ["TLT", "GLD", "IEF"]
    all_assets = [a for a in ALL_ASSETS if a in data]

    # Precompute
    closes = {t: data[t]["Close"] for t in all_assets}
    rets = {t: data[t]["Close"].pct_change() for t in all_assets}
    vol63 = {t: rets[t].rolling(63, min_periods=21).std() * np.sqrt(252) for t in all_assets}
    mom63 = {t: closes[t] / closes[t].shift(63) - 1 for t in all_assets}
    mom126 = {t: closes[t] / closes[t].shift(126) - 1 for t in all_assets}
    mom252 = {t: closes[t] / closes[t].shift(252) - 1 for t in all_assets}

    # Individual SMAs
    sma100 = {t: closes[t].rolling(100).mean() for t in all_assets}
    sma200 = {t: closes[t].rolling(200).mean() for t in all_assets}

    spy_close = data[BENCHMARK]["Close"]
    spy_sma100 = spy_close.rolling(100).mean()

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
    # Strategy U: Individual Sector SMA Filters + Inv-Vol
    # =================================================================
    print("\n" + "="*60)
    print("U: Individual Sector SMA100 Filters + Inv-Vol")
    print("  Only invest in sectors above their own SMA100")
    print("="*60)

    def core_u(data, date, idx):
        # For each sector: only include if above its own SMA100
        bullish = []
        for etf in available_sectors:
            if etf not in sma100 or date not in sma100[etf].index:
                continue
            s = sma100[etf].loc[date]
            p = closes[etf].loc[date] if date in closes[etf].index else 0
            if not pd.isna(s) and p > s:
                bullish.append(etf)

        if len(bullish) < 2:
            # Most sectors bearish → defensive
            return {"invested": True, "weights": {"TLT": 0.40, "GLD": 0.35, "IEF": 0.25}}

        # Inv-vol weight the bullish sectors
        weights = {}
        for etf in bullish:
            if date in vol63[etf].index:
                v = vol63[etf].loc[date]
                if not pd.isna(v) and v > 0:
                    weights[etf] = 1.0 / v

        if not weights:
            return {"invested": True, "weights": {"TLT": 0.40, "GLD": 0.35, "IEF": 0.25}}

        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        return {"invested": True, "weights": weights}

    sig_u = monthly_wrap(core_u)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets_s, trades = backtest_sector_strategy(data, s, e, sig_u)
        m = compute_metrics(rets_s)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy V: Dual Momentum — absolute + relative
    # =================================================================
    print("\n" + "="*60)
    print("V: Dual Momentum (absolute 252d > 0 + relative top 3)")
    print("="*60)

    def core_v(data, date, idx):
        # Absolute: only sectors with positive 252d momentum
        positive = []
        for etf in available_sectors:
            if etf in mom252 and date in mom252[etf].index:
                m = mom252[etf].loc[date]
                if not pd.isna(m) and m > 0:
                    positive.append((etf, m))

        if len(positive) < 2:
            # Few positive sectors → safe haven
            return {"invested": True, "weights": {"TLT": 0.40, "GLD": 0.35, "IEF": 0.25}}

        # Relative: top 3 by momentum, inv-vol weighted
        positive.sort(key=lambda x: x[1], reverse=True)
        top = positive[:5]
        weights = {}
        for etf, _ in top:
            if date in vol63[etf].index:
                v = vol63[etf].loc[date]
                if not pd.isna(v) and v > 0:
                    weights[etf] = 1.0 / v

        if not weights:
            return {"invested": True, "weights": {"TLT": 0.40, "GLD": 0.35, "IEF": 0.25}}

        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        return {"invested": True, "weights": weights}

    sig_v = monthly_wrap(core_v)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets_s, trades = backtest_sector_strategy(data, s, e, sig_v)
        m = compute_metrics(rets_s)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy W: Individual SMA + Safe Haven Allocation
    # =================================================================
    print("\n" + "="*60)
    print("W: Individual SMA100 + proportional safe haven fill")
    print("  Bearish sectors' allocation goes to TLT/GLD")
    print("="*60)

    def core_w(data, date, idx):
        weights = {}
        total_inv_vol = 0
        bullish_inv_vol = 0

        # Compute inv-vol for all sectors
        sector_iv = {}
        for etf in available_sectors:
            if date in vol63[etf].index and date in sma100[etf].index:
                v = vol63[etf].loc[date]
                s = sma100[etf].loc[date]
                p = closes[etf].loc[date] if date in closes[etf].index else 0
                if not pd.isna(v) and v > 0 and not pd.isna(s):
                    iv = 1.0 / v
                    total_inv_vol += iv
                    if p > s:  # bullish
                        sector_iv[etf] = iv
                        bullish_inv_vol += iv

        if total_inv_vol == 0:
            return {"invested": True, "weights": {"TLT": 0.40, "GLD": 0.35, "IEF": 0.25}}

        # Bullish sectors get their share
        for etf, iv in sector_iv.items():
            weights[etf] = iv / total_inv_vol

        # Bearish sectors' share goes to safe haven
        bear_share = 1.0 - sum(weights.values())
        if bear_share > 0:
            weights["TLT"] = bear_share * 0.45
            weights["GLD"] = bear_share * 0.35
            weights["IEF"] = bear_share * 0.20

        return {"invested": True, "weights": weights}

    sig_w = monthly_wrap(core_w)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets_s, trades = backtest_sector_strategy(data, s, e, sig_w)
        m = compute_metrics(rets_s)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy X: Combined — individual SMA + dual momentum + risk parity
    # =================================================================
    print("\n" + "="*60)
    print("X: Individual SMA + Dual Momentum + Risk Parity Fill")
    print("="*60)

    def core_x(data, date, idx):
        # Step 1: Individual SMA filter
        bullish = []
        bearish = []
        for etf in available_sectors:
            if date not in sma100[etf].index or date not in vol63[etf].index:
                continue
            s = sma100[etf].loc[date]
            p = closes[etf].loc[date] if date in closes[etf].index else 0
            v = vol63[etf].loc[date]
            if pd.isna(s) or pd.isna(v) or v <= 0:
                continue
            if p > s:
                # Step 2: Among bullish, also check absolute momentum (63d > 0)
                if etf in mom63 and date in mom63[etf].index:
                    m = mom63[etf].loc[date]
                    if not pd.isna(m) and m > 0:
                        bullish.append((etf, 1.0 / v))
                    else:
                        bearish.append((etf, 1.0 / v))
                else:
                    bullish.append((etf, 1.0 / v))
            else:
                bearish.append((etf, 1.0 / v))

        total_iv = sum(iv for _, iv in bullish) + sum(iv for _, iv in bearish)
        if total_iv == 0:
            return {"invested": True, "weights": {"TLT": 0.40, "GLD": 0.35, "IEF": 0.25}}

        # Bullish sectors get their inv-vol share
        weights = {}
        for etf, iv in bullish:
            weights[etf] = iv / total_iv

        # Bearish share goes to risk-parity safe haven
        bear_share = sum(iv for _, iv in bearish) / total_iv
        if bear_share > 0:
            # Risk parity among safe haven assets
            haven_w = {}
            for t in ["TLT", "GLD", "IEF"]:
                if t in vol63 and date in vol63[t].index:
                    v = vol63[t].loc[date]
                    if not pd.isna(v) and v > 0:
                        haven_w[t] = 1.0 / v
            if haven_w:
                ht = sum(haven_w.values())
                for t, w in haven_w.items():
                    weights[t] = (w / ht) * bear_share
            else:
                weights["TLT"] = bear_share * 0.5
                weights["GLD"] = bear_share * 0.5

        return {"invested": True, "weights": weights}

    sig_x = monthly_wrap(core_x)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets_s, trades = backtest_sector_strategy(data, s, e, sig_x)
        m = compute_metrics(rets_s)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy Y: Same as X but with SMA200 (slower, fewer whipsaws)
    # =================================================================
    print("\n" + "="*60)
    print("Y: Same as X but SMA200 for individual sector filters")
    print("="*60)

    def core_y(data, date, idx):
        bullish = []
        bearish = []
        for etf in available_sectors:
            if date not in sma200[etf].index or date not in vol63[etf].index:
                continue
            s = sma200[etf].loc[date]
            p = closes[etf].loc[date] if date in closes[etf].index else 0
            v = vol63[etf].loc[date]
            if pd.isna(s) or pd.isna(v) or v <= 0:
                continue
            if p > s:
                if etf in mom63 and date in mom63[etf].index:
                    m = mom63[etf].loc[date]
                    if not pd.isna(m) and m > 0:
                        bullish.append((etf, 1.0 / v))
                    else:
                        bearish.append((etf, 1.0 / v))
                else:
                    bullish.append((etf, 1.0 / v))
            else:
                bearish.append((etf, 1.0 / v))

        total_iv = sum(iv for _, iv in bullish) + sum(iv for _, iv in bearish)
        if total_iv == 0:
            return {"invested": True, "weights": {"TLT": 0.40, "GLD": 0.35, "IEF": 0.25}}

        weights = {}
        for etf, iv in bullish:
            weights[etf] = iv / total_iv

        bear_share = sum(iv for _, iv in bearish) / total_iv
        if bear_share > 0:
            haven_w = {}
            for t in ["TLT", "GLD", "IEF"]:
                if t in vol63 and date in vol63[t].index:
                    v = vol63[t].loc[date]
                    if not pd.isna(v) and v > 0:
                        haven_w[t] = 1.0 / v
            if haven_w:
                ht = sum(haven_w.values())
                for t, w in haven_w.items():
                    weights[t] = (w / ht) * bear_share
            else:
                weights["TLT"] = bear_share * 0.5
                weights["GLD"] = bear_share * 0.5

        return {"invested": True, "weights": weights}

    sig_y = monthly_wrap(core_y)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets_s, trades = backtest_sector_strategy(data, s, e, sig_y)
        m = compute_metrics(rets_s)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # Full period test for best strategies
    print("\n" + "="*60)
    print("FULL PERIOD (2009-2026) for best strategies")
    print("="*60)
    for label, sig_fn in [("X: SMA100+DualMom+RP", monthly_wrap(core_x)),
                           ("Y: SMA200+DualMom+RP", monthly_wrap(core_y)),
                           ("W: SMA100+PropFill", monthly_wrap(core_w))]:
        rets_s, trades = backtest_sector_strategy(data, "2009-01-01", TEST_END, sig_fn)
        m = compute_metrics(rets_s)
        spy = spy_metrics(data, "2009-01-01", TEST_END)
        print_results(f"FULL {label}", m, spy, trades)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
