#!/usr/bin/env python3
"""
Step 4: Refined strategies with monthly rebalancing + hedging.
Key insight: use TLT/GLD as hedge instead of cash during risk-off.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
from sector_strategy_research import (
    backtest_sector_strategy, compute_metrics, spy_metrics,
    print_results, SECTOR_ETFS, BENCHMARK
)


def make_monthly_rebal_signal(core_signal_fn):
    """Wrapper: only allow weight changes on first trading day of month."""
    last_month = [None]
    last_weights = [None]

    def wrapped(data, date, idx):
        month = date.month
        if last_month[0] is not None and month == last_month[0] and last_weights[0] is not None:
            # Same month — return last weights
            return last_weights[0]
        last_month[0] = month
        sig = core_signal_fn(data, date, idx)
        last_weights[0] = sig
        return sig

    return wrapped


def make_weekly_rebal_signal(core_signal_fn):
    """Wrapper: only allow weight changes on Mondays."""
    last_weights = [None]

    def wrapped(data, date, idx):
        # Rebalance on Monday or first trading day of week
        if date.weekday() != 0 and last_weights[0] is not None:
            return last_weights[0]
        sig = core_signal_fn(data, date, idx)
        last_weights[0] = sig
        return sig

    return wrapped


def run_tests(data):
    # Precompute
    sector_closes = {}
    sector_rets = {}
    rolling_vols = {}
    rolling_mom = {}

    available_sectors = [e for e in SECTOR_ETFS if e in data]
    for etf in available_sectors:
        sector_closes[etf] = data[etf]["Close"]
        sector_rets[etf] = data[etf]["Close"].pct_change()
        rolling_vols[etf] = sector_rets[etf].rolling(63, min_periods=21).std() * np.sqrt(252)
        rolling_mom[etf] = sector_closes[etf] / sector_closes[etf].shift(63) - 1

    spy_close = data[BENCHMARK]["Close"]
    spy_sma200 = spy_close.rolling(200).mean()
    spy_sma100 = spy_close.rolling(100).mean()
    spy_ret = spy_close.pct_change()
    spy_vol21 = spy_ret.rolling(21).std() * np.sqrt(252)

    # ============================================================
    # Strategy A: Monthly inv-vol top 5 sectors (always invested)
    # ============================================================
    print("\n" + "="*60)
    print("A: Monthly Inv-Vol Top 5 Sectors (always invested)")
    print("="*60)

    def core_a(data, date, idx):
        scored = []
        for etf in available_sectors:
            if date in rolling_mom[etf].index and date in rolling_vols[etf].index:
                mom = rolling_mom[etf].loc[date]
                vol = rolling_vols[etf].loc[date]
                if not pd.isna(mom) and not pd.isna(vol) and vol > 0:
                    scored.append((etf, mom / vol, 1.0 / vol))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:5]
        if not top:
            return {"invested": True, "weights": {BENCHMARK: 1.0}}
        total = sum(x[2] for x in top)
        weights = {etf: inv_v / total for etf, _, inv_v in top}
        return {"invested": True, "weights": weights}

    sig_a = make_monthly_rebal_signal(core_a)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_a)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # ============================================================
    # Strategy B: Monthly rebal with TLT/GLD hedge in risk-off
    # ============================================================
    print("\n" + "="*60)
    print("B: Monthly Sectors (risk-on) vs TLT+GLD (risk-off)")
    print("="*60)

    def core_b(data, date, idx):
        if date not in spy_sma200.index:
            return {"invested": True, "weights": {"TLT": 0.6, "GLD": 0.4}}
        sma = spy_sma200.loc[date]
        price = spy_close.loc[date]
        if pd.isna(sma) or price <= sma:
            # Risk-off: bonds + gold
            return {"invested": True, "weights": {"TLT": 0.6, "GLD": 0.4}}

        # Risk-on: top 5 sectors by risk-adj momentum, inv-vol weighted
        scored = []
        for etf in available_sectors:
            if date in rolling_mom[etf].index and date in rolling_vols[etf].index:
                mom = rolling_mom[etf].loc[date]
                vol = rolling_vols[etf].loc[date]
                if not pd.isna(mom) and not pd.isna(vol) and vol > 0:
                    scored.append((etf, mom / vol, 1.0 / vol))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:5]
        if not top:
            return {"invested": True, "weights": {BENCHMARK: 1.0}}
        total = sum(x[2] for x in top)
        weights = {etf: inv_v / total for etf, _, inv_v in top}
        return {"invested": True, "weights": weights}

    sig_b = make_monthly_rebal_signal(core_b)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_b)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # ============================================================
    # Strategy C: All sectors inv-vol weighted + TLT hedge (risk-off)
    # ============================================================
    print("\n" + "="*60)
    print("C: All Sectors Inv-Vol (risk-on) + TLT+GLD (risk-off), Monthly")
    print("="*60)

    def core_c(data, date, idx):
        if date not in spy_sma200.index:
            return {"invested": True, "weights": {"TLT": 0.6, "GLD": 0.4}}
        sma = spy_sma200.loc[date]
        price = spy_close.loc[date]
        if pd.isna(sma) or price <= sma:
            return {"invested": True, "weights": {"TLT": 0.6, "GLD": 0.4}}

        weights = {}
        for etf in available_sectors:
            if date in rolling_vols[etf].index:
                vol = rolling_vols[etf].loc[date]
                if not pd.isna(vol) and vol > 0:
                    weights[etf] = 1.0 / vol
        if not weights:
            return {"invested": True, "weights": {BENCHMARK: 1.0}}
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        return {"invested": True, "weights": weights}

    sig_c = make_monthly_rebal_signal(core_c)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_c)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # ============================================================
    # Strategy D: Dual regime — all inv-vol sectors + partial TLT/GLD hedge
    # ============================================================
    print("\n" + "="*60)
    print("D: Blended — sectors + partial TLT/GLD based on vol regime")
    print("="*60)

    def core_d(data, date, idx):
        # Sector weights (always some sector exposure)
        sec_weights = {}
        for etf in available_sectors:
            if date in rolling_vols[etf].index:
                vol = rolling_vols[etf].loc[date]
                if not pd.isna(vol) and vol > 0:
                    sec_weights[etf] = 1.0 / vol
        if sec_weights:
            total = sum(sec_weights.values())
            sec_weights = {k: v/total for k, v in sec_weights.items()}

        # Determine equity/hedge split based on regime
        equity_pct = 1.0
        if date in spy_sma200.index and date in spy_vol21.index:
            sma = spy_sma200.loc[date]
            price = spy_close.loc[date]
            vol = spy_vol21.loc[date]
            if not pd.isna(sma) and not pd.isna(vol):
                if price <= sma:
                    equity_pct = 0.3  # 30% sectors, 70% hedge
                elif vol > 0.20:
                    equity_pct = 0.5  # 50/50
                else:
                    equity_pct = 1.0  # all sectors

        hedge_pct = 1.0 - equity_pct
        weights = {}
        for etf, w in sec_weights.items():
            weights[etf] = w * equity_pct
        if hedge_pct > 0:
            weights["TLT"] = hedge_pct * 0.6
            weights["GLD"] = hedge_pct * 0.4
        return {"invested": True, "weights": weights}

    sig_d = make_monthly_rebal_signal(core_d)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_d)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # ============================================================
    # Strategy E: Same as D but SMA100
    # ============================================================
    print("\n" + "="*60)
    print("E: Blended with SMA100 regime + monthly rebal")
    print("="*60)

    def core_e(data, date, idx):
        sec_weights = {}
        for etf in available_sectors:
            if date in rolling_vols[etf].index:
                vol = rolling_vols[etf].loc[date]
                if not pd.isna(vol) and vol > 0:
                    sec_weights[etf] = 1.0 / vol
        if sec_weights:
            total = sum(sec_weights.values())
            sec_weights = {k: v/total for k, v in sec_weights.items()}

        equity_pct = 1.0
        if date in spy_sma100.index:
            sma = spy_sma100.loc[date]
            price = spy_close.loc[date]
            if not pd.isna(sma):
                if price <= sma:
                    equity_pct = 0.2
                else:
                    equity_pct = 1.0

        hedge_pct = 1.0 - equity_pct
        weights = {}
        for etf, w in sec_weights.items():
            weights[etf] = w * equity_pct
        if hedge_pct > 0:
            weights["TLT"] = hedge_pct * 0.5
            weights["GLD"] = hedge_pct * 0.5
        return {"invested": True, "weights": weights}

    sig_e = make_monthly_rebal_signal(core_e)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_e)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
