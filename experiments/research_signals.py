#!/usr/bin/env python3
"""
Step 2: Test individual signal components for alpha.
Focus on signals that work with T+1 open execution.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
from sector_strategy_research import (
    backtest_sector_strategy, compute_metrics, spy_metrics,
    print_results, SECTOR_ETFS, BENCHMARK
)


def precompute_signals(data):
    """Precompute all signals to avoid recomputation in backtest loop."""
    signals = {}

    spy = data[BENCHMARK]["Close"]

    # SMA signals for SPY
    for period in [20, 50, 100, 200]:
        signals[f"spy_sma{period}"] = spy.rolling(period).mean()

    # SPY trend: price > SMA20 AND SMA20 > SMA50
    signals["spy_trend_strong"] = (spy > signals["spy_sma20"]) & (signals["spy_sma20"] > signals["spy_sma50"])

    # SPY realized vol
    spy_ret = spy.pct_change()
    for w in [10, 21, 63]:
        signals[f"spy_vol_{w}d"] = spy_ret.rolling(w).std() * np.sqrt(252)

    # Sector momentum (63d, 21d returns)
    for etf in SECTOR_ETFS:
        df = data.get(etf)
        if df is None:
            continue
        c = df["Close"]
        for lookback in [21, 42, 63, 126]:
            signals[f"{etf}_mom_{lookback}d"] = c / c.shift(lookback) - 1

        # Sector vol
        r = c.pct_change()
        signals[f"{etf}_vol_21d"] = r.rolling(21).std() * np.sqrt(252)
        signals[f"{etf}_vol_63d"] = r.rolling(63).std() * np.sqrt(252)

        # Risk-adjusted momentum (momentum / vol)
        for lb in [21, 63]:
            vol = signals[f"{etf}_vol_21d"]
            mom = signals[f"{etf}_mom_{lb}d"]
            signals[f"{etf}_risk_adj_mom_{lb}d"] = mom / vol.clip(lower=0.01)

    # Sector breadth: how many sectors have positive momentum
    for lookback in [21, 63]:
        breadth_data = []
        for etf in SECTOR_ETFS:
            key = f"{etf}_mom_{lookback}d"
            if key in signals:
                breadth_data.append(signals[key] > 0)
        if breadth_data:
            signals[f"breadth_{lookback}d"] = pd.concat(breadth_data, axis=1).sum(axis=1)

    # Cross-sector correlation (average pairwise correlation over 21 days)
    sector_rets = {}
    for etf in SECTOR_ETFS:
        df = data.get(etf)
        if df is not None:
            sector_rets[etf] = df["Close"].pct_change()

    if len(sector_rets) >= 5:
        sr_df = pd.DataFrame(sector_rets).dropna()
        # Rolling average correlation
        for w in [21, 63]:
            corr_series = sr_df.rolling(w).corr()
            # Average of upper triangle
            avg_corr = []
            dates = sr_df.index[w-1:]
            for d in dates:
                try:
                    c = corr_series.loc[d]
                    if isinstance(c, pd.DataFrame):
                        mask = np.triu(np.ones(c.shape), k=1).astype(bool)
                        vals = c.values[mask]
                        avg_corr.append(np.nanmean(vals))
                    else:
                        avg_corr.append(np.nan)
                except:
                    avg_corr.append(np.nan)
            signals[f"sector_avg_corr_{w}d"] = pd.Series(avg_corr, index=dates)

    # Sector dispersion (cross-sectional std of returns)
    for lookback in [5, 21]:
        disp_data = []
        for etf in SECTOR_ETFS:
            key = f"{etf}_mom_{21}d" if lookback == 21 else f"{etf}_mom_{21}d"
            df = data.get(etf)
            if df is not None:
                disp_data.append(df["Close"].pct_change(lookback))
        if disp_data:
            disp_df = pd.concat(disp_data, axis=1)
            signals[f"sector_dispersion_{lookback}d"] = disp_df.std(axis=1)

    # SPY drawdown from 252-day high
    spy_high = spy.rolling(252, min_periods=21).max()
    signals["spy_drawdown"] = (spy - spy_high) / spy_high

    # TLT momentum (flight to safety signal)
    if "TLT" in data:
        tlt = data["TLT"]["Close"]
        signals["tlt_mom_21d"] = tlt / tlt.shift(21) - 1
        signals["tlt_mom_63d"] = tlt / tlt.shift(63) - 1

    return signals


def test_signal_1_trend_gate(data, signals):
    """Signal 1: Multi-timeframe trend confirmation."""
    print("\n" + "="*60)
    print("SIGNAL 1: SPY Multi-Timeframe Trend Gate")
    print("="*60)

    spy_close = data[BENCHMARK]["Close"]
    sma20 = signals["spy_sma20"]
    sma50 = signals["spy_sma50"]
    sma200 = signals["spy_sma200"]

    def signal_fn(data, date, idx):
        if date not in sma20.index or date not in sma50.index:
            return {"invested": False, "weights": {}}

        price = spy_close.loc[date] if date in spy_close.index else 0
        s20 = sma20.loc[date] if date in sma20.index else 0
        s50 = sma50.loc[date] if date in sma50.index else 0

        # Invest in SPY only when price > SMA20 and SMA20 > SMA50
        if price > s20 and s20 > s50:
            return {"invested": True, "weights": {BENCHMARK: 1.0}}
        return {"invested": False, "weights": {}}

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, signal_fn)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)


def test_signal_2_breadth(data, signals):
    """Signal 2: Sector breadth filter."""
    print("\n" + "="*60)
    print("SIGNAL 2: Sector Breadth + Top Momentum")
    print("="*60)

    breadth = signals["breadth_63d"]

    def signal_fn(data, date, idx):
        if date not in breadth.index:
            return {"invested": False, "weights": {}}

        b = breadth.loc[date] if date in breadth.index else 0
        if pd.isna(b) or b < 7:  # need 7+ of 11 sectors positive
            return {"invested": False, "weights": {}}

        # Pick top 3 sectors by 63d momentum
        scored = []
        for etf in SECTOR_ETFS:
            key = f"{etf}_mom_63d"
            if key in signals and date in signals[key].index:
                val = signals[key].loc[date]
                if not pd.isna(val):
                    scored.append((etf, val))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:3]
        if not top:
            return {"invested": False, "weights": {}}

        w = 1.0 / len(top)
        weights = {etf: w for etf, _ in top}
        return {"invested": True, "weights": weights}

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, signal_fn)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)


def test_signal_3_vol_regime(data, signals):
    """Signal 3: Low volatility regime + trend."""
    print("\n" + "="*60)
    print("SIGNAL 3: Low Vol Regime + Trend + Top Sector")
    print("="*60)

    spy_close = data[BENCHMARK]["Close"]
    sma50 = signals["spy_sma50"]
    vol21 = signals["spy_vol_21d"]

    def signal_fn(data, date, idx):
        if date not in sma50.index or date not in vol21.index:
            return {"invested": False, "weights": {}}

        price = spy_close.loc[date]
        s50 = sma50.loc[date]
        v = vol21.loc[date]

        # Invest when: trend up AND vol < 20%
        if pd.isna(s50) or pd.isna(v):
            return {"invested": False, "weights": {}}
        if price <= s50 or v > 0.20:
            return {"invested": False, "weights": {}}

        # Top sector by risk-adjusted momentum
        best, best_score = None, -999
        for etf in SECTOR_ETFS:
            key = f"{etf}_risk_adj_mom_63d"
            if key in signals and date in signals[key].index:
                val = signals[key].loc[date]
                if not pd.isna(val) and val > best_score:
                    best, best_score = etf, val
        if best:
            return {"invested": True, "weights": {BENCHMARK: 0.5, best: 0.5}}
        return {"invested": False, "weights": {}}

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, signal_fn)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)


def test_signal_4_combined(data, signals):
    """Signal 4: All signals combined — the full strategy."""
    print("\n" + "="*60)
    print("SIGNAL 4: Combined (Trend + Breadth + Low Vol + Momentum)")
    print("="*60)

    spy_close = data[BENCHMARK]["Close"]
    sma20 = signals["spy_sma20"]
    sma50 = signals["spy_sma50"]
    vol21 = signals["spy_vol_21d"]
    breadth = signals["breadth_63d"]

    def signal_fn(data, date, idx):
        # Gate 1: Trend
        if date not in sma20.index or date not in sma50.index:
            return {"invested": False, "weights": {}}
        price = spy_close.loc[date]
        s20 = sma20.loc[date]
        s50 = sma50.loc[date]
        if pd.isna(s20) or pd.isna(s50):
            return {"invested": False, "weights": {}}
        if not (price > s20 and s20 > s50):
            return {"invested": False, "weights": {}}

        # Gate 2: Low vol
        if date in vol21.index:
            v = vol21.loc[date]
            if not pd.isna(v) and v > 0.22:
                return {"invested": False, "weights": {}}

        # Gate 3: Breadth
        if date in breadth.index:
            b = breadth.loc[date]
            if not pd.isna(b) and b < 6:
                return {"invested": False, "weights": {}}

        # Select top 3 sectors by risk-adj momentum
        scored = []
        for etf in SECTOR_ETFS:
            key = f"{etf}_risk_adj_mom_63d"
            if key in signals and date in signals[key].index:
                val = signals[key].loc[date]
                if not pd.isna(val) and val > 0:
                    scored.append((etf, val))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:3]
        if not top:
            return {"invested": False, "weights": {}}

        w = 1.0 / len(top)
        weights = {etf: w for etf, _ in top}
        return {"invested": True, "weights": weights}

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, signal_fn)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")

    print("\nPrecomputing signals...")
    signals = precompute_signals(data)
    print(f"Computed {len(signals)} signals")

    test_signal_1_trend_gate(data, signals)
    test_signal_2_breadth(data, signals)
    test_signal_3_vol_regime(data, signals)
    test_signal_4_combined(data, signals)
