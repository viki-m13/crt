#!/usr/bin/env python3
"""
Step 7: Multi-asset risk parity + minimum variance approaches.
Try mixing sectors with TLT/GLD for lower vol with good returns.
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
    spy_vol21 = spy_ret.rolling(21).std() * np.sqrt(252)

    sec_ret = {e: data[e]["Close"].pct_change() for e in available_sectors}
    sec_vol63 = {e: sec_ret[e].rolling(63, min_periods=21).std() * np.sqrt(252) for e in available_sectors}

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
    # Strategy P: Fixed allocation - 50% sectors(invvol) + 25% TLT + 25% GLD
    # =================================================================
    print("\n" + "="*60)
    print("P: Fixed 50% inv-vol sectors + 25% TLT + 25% GLD")
    print("="*60)

    def core_p(data, date, idx):
        weights = {"TLT": 0.25, "GLD": 0.25}
        sec_w = {}
        for etf in available_sectors:
            if etf in sec_vol63 and date in sec_vol63[etf].index:
                vol = sec_vol63[etf].loc[date]
                if not pd.isna(vol) and vol > 0:
                    sec_w[etf] = 1.0 / vol
        if sec_w:
            total = sum(sec_w.values())
            for etf, w in sec_w.items():
                weights[etf] = (w / total) * 0.50
        return {"invested": True, "weights": weights}

    sig_p = monthly_wrap(core_p)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_p)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy Q: Dynamic risk parity (1/vol) across sectors + TLT + GLD
    # =================================================================
    print("\n" + "="*60)
    print("Q: Dynamic Risk Parity (all assets weighted by 1/vol)")
    print("="*60)

    tlt_ret = data["TLT"]["Close"].pct_change() if "TLT" in data else None
    gld_ret = data["GLD"]["Close"].pct_change() if "GLD" in data else None
    ief_ret = data["IEF"]["Close"].pct_change() if "IEF" in data else None

    tlt_vol63 = tlt_ret.rolling(63, min_periods=21).std() * np.sqrt(252) if tlt_ret is not None else None
    gld_vol63 = gld_ret.rolling(63, min_periods=21).std() * np.sqrt(252) if gld_ret is not None else None
    ief_vol63 = ief_ret.rolling(63, min_periods=21).std() * np.sqrt(252) if ief_ret is not None else None

    def core_q(data, date, idx):
        all_weights = {}
        # Sectors
        for etf in available_sectors:
            if etf in sec_vol63 and date in sec_vol63[etf].index:
                vol = sec_vol63[etf].loc[date]
                if not pd.isna(vol) and vol > 0.01:
                    all_weights[etf] = 1.0 / vol
        # TLT
        if tlt_vol63 is not None and date in tlt_vol63.index:
            v = tlt_vol63.loc[date]
            if not pd.isna(v) and v > 0.01:
                all_weights["TLT"] = 1.0 / v
        # GLD
        if gld_vol63 is not None and date in gld_vol63.index:
            v = gld_vol63.loc[date]
            if not pd.isna(v) and v > 0.01:
                all_weights["GLD"] = 1.0 / v
        # IEF
        if ief_vol63 is not None and date in ief_vol63.index:
            v = ief_vol63.loc[date]
            if not pd.isna(v) and v > 0.01:
                all_weights["IEF"] = 1.0 / v

        if all_weights:
            total = sum(all_weights.values())
            all_weights = {k: v/total for k, v in all_weights.items()}
        return {"invested": True, "weights": all_weights}

    sig_q = monthly_wrap(core_q)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_q)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy R: Risk parity + SMA100 tilt (more equity when bullish)
    # =================================================================
    print("\n" + "="*60)
    print("R: Risk Parity + SMA100 Tilt (shift to equity in bull)")
    print("="*60)

    def core_r(data, date, idx):
        # Base: risk parity across everything
        all_weights = {}
        for etf in available_sectors:
            if etf in sec_vol63 and date in sec_vol63[etf].index:
                vol = sec_vol63[etf].loc[date]
                if not pd.isna(vol) and vol > 0.01:
                    all_weights[etf] = 1.0 / vol
        if tlt_vol63 is not None and date in tlt_vol63.index:
            v = tlt_vol63.loc[date]
            if not pd.isna(v) and v > 0.01:
                all_weights["TLT"] = 1.0 / v
        if gld_vol63 is not None and date in gld_vol63.index:
            v = gld_vol63.loc[date]
            if not pd.isna(v) and v > 0.01:
                all_weights["GLD"] = 1.0 / v

        if not all_weights:
            return {"invested": True, "weights": {BENCHMARK: 1.0}}

        total = sum(all_weights.values())
        all_weights = {k: v/total for k, v in all_weights.items()}

        # SMA100 tilt: in bull, double equity weights; in bear, double hedge
        bull = True
        if date in sma100.index:
            s = sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bull = False

        tilted = {}
        for k, w in all_weights.items():
            if k in available_sectors:
                tilted[k] = w * (1.5 if bull else 0.5)
            else:
                tilted[k] = w * (0.7 if bull else 1.5)

        total = sum(tilted.values())
        tilted = {k: v/total for k, v in tilted.items()}
        return {"invested": True, "weights": tilted}

    sig_r = monthly_wrap(core_r)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_r)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy S: Regime-adaptive risk parity with quality tilt
    # =================================================================
    print("\n" + "="*60)
    print("S: Regime-Adaptive Risk Parity + Quality Tilt")
    print("="*60)

    # Sector quality
    sec_quality = {}
    for etf in available_sectors:
        r = sec_ret[etf]
        m63 = r.rolling(63, min_periods=42).mean() * 252
        s63 = r.rolling(63, min_periods=42).std() * np.sqrt(252)
        sec_quality[etf] = (m63 - 0.02) / s63.clip(lower=0.01)

    DEFENSIVE = ["XLP", "XLU", "XLV"]
    OFFENSIVE = [e for e in available_sectors if e not in DEFENSIVE]

    def core_s(data, date, idx):
        # Regime
        bull = True
        if date in sma100.index:
            s = sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bull = False

        # Vol regime
        high_vol = False
        if date in spy_vol21.index:
            v = spy_vol21.loc[date]
            if not pd.isna(v) and v > 0.18:
                high_vol = True

        if not bull:
            # Bear: defensive sectors + heavy bonds/gold
            weights = {}
            for etf in DEFENSIVE:
                if etf in available_sectors and etf in sec_vol63 and date in sec_vol63[etf].index:
                    vol = sec_vol63[etf].loc[date]
                    if not pd.isna(vol) and vol > 0:
                        weights[etf] = 1.0 / vol
            if weights:
                total = sum(weights.values())
                weights = {k: v/total * 0.30 for k, v in weights.items()}
            weights["TLT"] = 0.40
            weights["GLD"] = 0.30
            return {"invested": True, "weights": weights}

        if high_vol:
            # High vol bull: broader allocation, more hedge
            weights = {}
            for etf in available_sectors:
                if etf in sec_vol63 and date in sec_vol63[etf].index:
                    vol = sec_vol63[etf].loc[date]
                    if not pd.isna(vol) and vol > 0:
                        weights[etf] = 1.0 / vol
            if weights:
                total = sum(weights.values())
                weights = {k: v/total * 0.60 for k, v in weights.items()}
            weights["TLT"] = 0.25
            weights["GLD"] = 0.15
            return {"invested": True, "weights": weights}

        # Low vol bull: quality-tilted sectors
        quality_sectors = []
        for etf in available_sectors:
            if etf in sec_quality and date in sec_quality[etf].index:
                q = sec_quality[etf].loc[date]
                if not pd.isna(q) and q > 0:
                    quality_sectors.append(etf)

        if not quality_sectors:
            quality_sectors = available_sectors

        weights = {}
        for etf in quality_sectors:
            if etf in sec_vol63 and date in sec_vol63[etf].index:
                vol = sec_vol63[etf].loc[date]
                if not pd.isna(vol) and vol > 0:
                    weights[etf] = 1.0 / vol
        if weights:
            total = sum(weights.values())
            weights = {k: v/total * 0.80 for k, v in weights.items()}
        weights["TLT"] = 0.12
        weights["GLD"] = 0.08
        return {"invested": True, "weights": weights}

    sig_s = monthly_wrap(core_s)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_s)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy T: Best of H + risk parity base
    # =================================================================
    print("\n" + "="*60)
    print("T: Strategy H (defensive/offensive) + risk parity base")
    print("="*60)

    def core_t(data, date, idx):
        bear = False
        if date in sma100.index:
            s = sma100.loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                bear = True

        if bear:
            # Bear: 30% defensive sectors + 40% TLT + 30% GLD
            weights = {}
            for etf in DEFENSIVE:
                if etf in available_sectors and etf in sec_vol63 and date in sec_vol63[etf].index:
                    vol = sec_vol63[etf].loc[date]
                    if not pd.isna(vol) and vol > 0:
                        weights[etf] = 1.0 / vol
            if weights:
                total = sum(weights.values())
                weights = {k: v/total * 0.30 for k, v in weights.items()}
            weights["TLT"] = 0.40
            weights["GLD"] = 0.30
            return {"invested": True, "weights": weights}

        # Bull: 85% all sectors inv-vol + 10% TLT + 5% GLD
        weights = {}
        for etf in available_sectors:
            if etf in sec_vol63 and date in sec_vol63[etf].index:
                vol = sec_vol63[etf].loc[date]
                if not pd.isna(vol) and vol > 0:
                    weights[etf] = 1.0 / vol
        if weights:
            total = sum(weights.values())
            weights = {k: v/total * 0.85 for k, v in weights.items()}
        weights["TLT"] = 0.10
        weights["GLD"] = 0.05
        return {"invested": True, "weights": weights}

    sig_t = monthly_wrap(core_t)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_t)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Print summary table
    # =================================================================
    print("\n" + "="*60)
    print("ALSO: Pure asset benchmarks")
    print("="*60)

    # TLT buy and hold
    def tlt_only(data, date, idx):
        return {"invested": True, "weights": {"TLT": 1.0}}
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, _ = backtest_sector_strategy(data, s, e, tlt_only)
        m = compute_metrics(rets)
        print(f"  TLT {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%}")

    # GLD buy and hold
    def gld_only(data, date, idx):
        return {"invested": True, "weights": {"GLD": 1.0}}
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, _ = backtest_sector_strategy(data, s, e, gld_only)
        m = compute_metrics(rets)
        print(f"  GLD {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
