#!/usr/bin/env python3
"""
Step 9: Final strategy candidates — absolute trend on all assets, ensembles.
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
    SAFE_HAVENS = ["TLT", "GLD", "IEF"]
    ALL_TRADEABLE = available_sectors + [h for h in SAFE_HAVENS if h in data]

    closes = {t: data[t]["Close"] for t in ALL_TRADEABLE}
    rets_d = {t: data[t]["Close"].pct_change() for t in ALL_TRADEABLE}
    vol63 = {t: rets_d[t].rolling(63, min_periods=21).std() * np.sqrt(252) for t in ALL_TRADEABLE}
    sma100 = {t: closes[t].rolling(100).mean() for t in ALL_TRADEABLE}
    sma200 = {t: closes[t].rolling(200).mean() for t in ALL_TRADEABLE}
    mom63 = {t: closes[t] / closes[t].shift(63) - 1 for t in ALL_TRADEABLE}

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
    # Strategy Z1: Absolute Trend on ALL Assets
    # Only invest in assets above SMA100, weight by inv-vol
    # =================================================================
    print("\n" + "="*60)
    print("Z1: Absolute Trend ALL Assets (SMA100 filter, inv-vol weight)")
    print("="*60)

    def core_z1(data, date, idx):
        weights = {}
        for t in ALL_TRADEABLE:
            if date not in sma100[t].index or date not in vol63[t].index:
                continue
            s = sma100[t].loc[date]
            p = closes[t].loc[date] if date in closes[t].index else 0
            v = vol63[t].loc[date]
            if pd.isna(s) or pd.isna(v) or v <= 0:
                continue
            if p > s:
                weights[t] = 1.0 / v

        if not weights:
            return {"invested": False, "weights": {}}

        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        return {"invested": True, "weights": weights}

    sig = monthly_wrap(core_z1)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = backtest_sector_strategy(data, s, e, sig)
        m = compute_metrics(r)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, t)

    # =================================================================
    # Strategy Z2: Regime-Adaptive All-Asset Trend
    # =================================================================
    print("\n" + "="*60)
    print("Z2: All-Asset Trend + Defensive Sectors as Fallback")
    print("="*60)

    DEFENSIVE = ["XLP", "XLU", "XLV"]

    def core_z2(data, date, idx):
        # Trend-following on all assets
        bullish = {}
        for t in ALL_TRADEABLE:
            if date not in sma100[t].index or date not in vol63[t].index:
                continue
            s = sma100[t].loc[date]
            p = closes[t].loc[date] if date in closes[t].index else 0
            v = vol63[t].loc[date]
            if pd.isna(s) or pd.isna(v) or v <= 0:
                continue
            if p > s:
                bullish[t] = 1.0 / v

        if not bullish:
            # Nothing in uptrend — go to lowest vol defensive sectors
            w = {}
            for t in DEFENSIVE + ["IEF"]:
                if t in vol63 and date in vol63[t].index:
                    v = vol63[t].loc[date]
                    if not pd.isna(v) and v > 0:
                        w[t] = 1.0 / v
            if w:
                total = sum(w.values())
                return {"invested": True, "weights": {k: v/total for k, v in w.items()}}
            return {"invested": False, "weights": {}}

        total = sum(bullish.values())
        weights = {k: v/total for k, v in bullish.items()}
        return {"invested": True, "weights": weights}

    sig = monthly_wrap(core_z2)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = backtest_sector_strategy(data, s, e, sig)
        m = compute_metrics(r)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, t)

    # =================================================================
    # Strategy Z3: Risk Parity base + SMA100 Rotation Enhancement
    # Core insight: ALWAYS in risk parity, but tilt sector allocation
    # based on individual sector trends
    # =================================================================
    print("\n" + "="*60)
    print("Z3: Risk Parity Base + Individual Sector Trend Tilt")
    print("="*60)

    def core_z3(data, date, idx):
        # Base: risk parity across all
        all_w = {}
        for t in ALL_TRADEABLE:
            if date in vol63[t].index:
                v = vol63[t].loc[date]
                if not pd.isna(v) and v > 0.01:
                    all_w[t] = 1.0 / v

        if not all_w:
            return {"invested": True, "weights": {"IEF": 1.0}}

        total = sum(all_w.values())
        base_weights = {k: v/total for k, v in all_w.items()}

        # Tilt: sectors above SMA100 get 2x weight; below get 0.5x
        # Safe havens get inverse tilt (2x when more sectors are bearish)
        n_bearish = 0
        n_total = 0
        tilted = {}
        for t in ALL_TRADEABLE:
            w = base_weights.get(t, 0)
            if t in available_sectors:
                n_total += 1
                if date in sma100[t].index:
                    s = sma100[t].loc[date]
                    p = closes[t].loc[date] if date in closes[t].index else 0
                    if not pd.isna(s) and p > s:
                        tilted[t] = w * 1.5  # Bullish sector: boost
                    else:
                        tilted[t] = w * 0.3  # Bearish sector: reduce
                        n_bearish += 1
                else:
                    tilted[t] = w
            else:
                tilted[t] = w  # Keep base for now

        # Boost safe havens when many sectors are bearish
        if n_total > 0:
            bear_ratio = n_bearish / n_total
            for t in SAFE_HAVENS:
                if t in tilted:
                    tilted[t] = tilted[t] * (1.0 + bear_ratio * 2.0)

        # Renormalize
        total = sum(tilted.values())
        if total > 0:
            tilted = {k: v/total for k, v in tilted.items()}

        return {"invested": True, "weights": tilted}

    sig = monthly_wrap(core_z3)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = backtest_sector_strategy(data, s, e, sig)
        m = compute_metrics(r)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, t)

    # =================================================================
    # Strategy Z4: ENSEMBLE — average Z3 + Q + T weights
    # Simulate by running each monthly and averaging
    # =================================================================
    print("\n" + "="*60)
    print("Z4: Ensemble of Z3 + Risk Parity (Q-style) + Defensive H")
    print("="*60)

    # Q-style: pure risk parity
    def core_q(data, date, idx):
        w = {}
        for t in ALL_TRADEABLE:
            if date in vol63[t].index:
                v = vol63[t].loc[date]
                if not pd.isna(v) and v > 0.01:
                    w[t] = 1.0 / v
        if w:
            total = sum(w.values())
            w = {k: v/total for k, v in w.items()}
        return {"invested": True, "weights": w}

    # H-style: defensive/offensive rotation
    def core_h(data, date, idx):
        bear = False
        if date in spy_sma100.index:
            s = spy_sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True
        if bear:
            w = {}
            for t in DEFENSIVE:
                if t in vol63 and date in vol63[t].index:
                    v = vol63[t].loc[date]
                    if not pd.isna(v) and v > 0:
                        w[t] = 1.0 / v
            if w:
                total = sum(w.values())
                w = {k: v/total * 0.35 for k, v in w.items()}
            w["TLT"] = 0.35
            w["GLD"] = 0.30
            return {"invested": True, "weights": w}
        # Bull
        w = {}
        for t in available_sectors:
            if date in vol63[t].index:
                v = vol63[t].loc[date]
                if not pd.isna(v) and v > 0:
                    w[t] = 1.0 / v
        if w:
            total = sum(w.values())
            w = {k: v/total for k, v in w.items()}
        return {"invested": True, "weights": w}

    def core_z4(data, date, idx):
        sigs = [core_z3(data, date, idx), core_q(data, date, idx), core_h(data, date, idx)]
        # Average weights
        combined = {}
        for sig in sigs:
            for t, w in sig.get("weights", {}).items():
                combined[t] = combined.get(t, 0) + w / len(sigs)
        # Renormalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}
        return {"invested": True, "weights": combined}

    sig = monthly_wrap(core_z4)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r, t = backtest_sector_strategy(data, s, e, sig)
        m = compute_metrics(r)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, t)

    # Full period
    print("\n  FULL period:")
    for label, fn in [("Z3", core_z3), ("Z4", core_z4), ("Q", core_q)]:
        sig = monthly_wrap(fn)
        r, t = backtest_sector_strategy(data, "2009-01-01", TEST_END, sig)
        m = compute_metrics(r)
        spy = spy_metrics(data, "2009-01-01", TEST_END)
        print(f"    {label}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} | SPY: {spy['sharpe']:.3f} {spy['cagr']:.1%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
