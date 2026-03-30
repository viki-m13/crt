#!/usr/bin/env python3
"""
Step 3: Test different portfolio construction approaches.
Key insight: to get high Sharpe, need LOW VOLATILITY, not just high returns.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
from sector_strategy_research import (
    backtest_sector_strategy, compute_metrics, spy_metrics,
    print_results, SECTOR_ETFS, BENCHMARK
)


def test_allocation_approaches(data):
    """Test various allocation methods."""

    # Precompute sector data
    sector_closes = {}
    sector_rets = {}
    for etf in SECTOR_ETFS:
        if etf in data:
            sector_closes[etf] = data[etf]["Close"]
            sector_rets[etf] = data[etf]["Close"].pct_change()

    available_sectors = list(sector_closes.keys())

    # Approach 1: Equal weight all sectors (always invested)
    print("\n" + "="*60)
    print("APPROACH 1: Equal Weight All Sectors (always invested)")
    print("="*60)

    def equal_weight(data, date, idx):
        w = 1.0 / len(available_sectors)
        weights = {}
        for etf in available_sectors:
            if etf in data and date in data[etf].index:
                weights[etf] = w
        if weights:
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        return {"invested": True, "weights": weights}

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, equal_weight)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # Approach 2: Inverse volatility (risk parity)
    print("\n" + "="*60)
    print("APPROACH 2: Inverse Volatility Weighted (always invested)")
    print("="*60)

    # Precompute rolling vols
    rolling_vols = {}
    for etf in available_sectors:
        rolling_vols[etf] = sector_rets[etf].rolling(63, min_periods=21).std() * np.sqrt(252)

    def inv_vol_weight(data, date, idx):
        weights = {}
        for etf in available_sectors:
            if etf not in rolling_vols or date not in rolling_vols[etf].index:
                continue
            vol = rolling_vols[etf].loc[date]
            if pd.isna(vol) or vol <= 0:
                continue
            weights[etf] = 1.0 / vol

        if not weights:
            return {"invested": False, "weights": {}}
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        return {"invested": True, "weights": weights}

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, inv_vol_weight)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # Approach 3: Top 3 momentum (always invested)
    print("\n" + "="*60)
    print("APPROACH 3: Top 3 Momentum (63d, always invested)")
    print("="*60)

    rolling_mom = {}
    for etf in available_sectors:
        rolling_mom[etf] = sector_closes[etf] / sector_closes[etf].shift(63) - 1

    def top3_momentum(data, date, idx):
        scored = []
        for etf in available_sectors:
            if etf in rolling_mom and date in rolling_mom[etf].index:
                val = rolling_mom[etf].loc[date]
                if not pd.isna(val):
                    scored.append((etf, val))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:3]
        if not top:
            return {"invested": False, "weights": {}}
        w = 1.0 / len(top)
        return {"invested": True, "weights": {etf: w for etf, _ in top}}

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, top3_momentum)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # Approach 4: Top 3 risk-adjusted momentum (mom/vol)
    print("\n" + "="*60)
    print("APPROACH 4: Top 3 Risk-Adjusted Momentum (always invested)")
    print("="*60)

    def top3_risk_adj(data, date, idx):
        scored = []
        for etf in available_sectors:
            if etf in rolling_mom and etf in rolling_vols:
                if date in rolling_mom[etf].index and date in rolling_vols[etf].index:
                    mom = rolling_mom[etf].loc[date]
                    vol = rolling_vols[etf].loc[date]
                    if not pd.isna(mom) and not pd.isna(vol) and vol > 0:
                        scored.append((etf, mom / vol))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:3]
        if not top:
            return {"invested": False, "weights": {}}
        w = 1.0 / len(top)
        return {"invested": True, "weights": {etf: w for etf, _ in top}}

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, top3_risk_adj)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # Approach 5: Inv vol weighted, momentum-tilted (top 5 only)
    print("\n" + "="*60)
    print("APPROACH 5: Inv Vol + Momentum Tilt (top 5)")
    print("="*60)

    def inv_vol_mom_tilt(data, date, idx):
        scored = []
        for etf in available_sectors:
            if etf in rolling_mom and etf in rolling_vols:
                if date in rolling_mom[etf].index and date in rolling_vols[etf].index:
                    mom = rolling_mom[etf].loc[date]
                    vol = rolling_vols[etf].loc[date]
                    if not pd.isna(mom) and not pd.isna(vol) and vol > 0:
                        scored.append((etf, mom / vol, 1.0 / vol))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:5]
        if not top:
            return {"invested": False, "weights": {}}
        # Weight by inverse vol among top 5
        total = sum(x[2] for x in top)
        weights = {etf: inv_v / total for etf, _, inv_v in top}
        return {"invested": True, "weights": weights}

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, inv_vol_mom_tilt)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # Approach 6: SMA200 gate + inv vol momentum tilt
    print("\n" + "="*60)
    print("APPROACH 6: SMA200 Gate + Inv Vol Momentum Tilt (top 5)")
    print("="*60)

    spy_close = data[BENCHMARK]["Close"]
    spy_sma200 = spy_close.rolling(200).mean()

    def gated_inv_vol_mom(data, date, idx):
        if date not in spy_sma200.index:
            return {"invested": False, "weights": {}}
        sma = spy_sma200.loc[date]
        price = spy_close.loc[date]
        if pd.isna(sma) or price <= sma:
            return {"invested": False, "weights": {}}
        return inv_vol_mom_tilt(data, date, idx)

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, gated_inv_vol_mom)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    test_allocation_approaches(data)
