#!/usr/bin/env python3
"""
Step 5: Advanced signals — cross-asset, reversal timing, adaptive regimes.
Test multiple approaches quickly.
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

    # Precompute everything
    spy_close = data[BENCHMARK]["Close"]
    spy_ret = spy_close.pct_change()

    # SMAs
    sma = {}
    for p in [50, 100, 150, 200]:
        sma[p] = spy_close.rolling(p).mean()

    # Sector data
    sec_close = {e: data[e]["Close"] for e in available_sectors}
    sec_ret = {e: data[e]["Close"].pct_change() for e in available_sectors}
    sec_vol63 = {e: sec_ret[e].rolling(63, min_periods=21).std() * np.sqrt(252) for e in available_sectors}
    sec_mom63 = {e: sec_close[e] / sec_close[e].shift(63) - 1 for e in available_sectors}
    sec_mom21 = {e: sec_close[e] / sec_close[e].shift(21) - 1 for e in available_sectors}
    sec_mom5 = {e: sec_close[e] / sec_close[e].shift(5) - 1 for e in available_sectors}

    # Cross-asset
    tlt_close = data["TLT"]["Close"] if "TLT" in data else None
    gld_close = data["GLD"]["Close"] if "GLD" in data else None
    ief_close = data["IEF"]["Close"] if "IEF" in data else None
    hyg_close = data["HYG"]["Close"] if "HYG" in data else None

    tlt_mom21 = (tlt_close / tlt_close.shift(21) - 1) if tlt_close is not None else None
    gld_mom21 = (gld_close / gld_close.shift(21) - 1) if gld_close is not None else None

    # Credit spread proxy: HYG relative to IEF
    credit_signal = None
    if hyg_close is not None and ief_close is not None:
        ratio = hyg_close / ief_close
        credit_signal = ratio / ratio.rolling(63).mean() - 1

    # SPY vol
    spy_vol21 = spy_ret.rolling(21).std() * np.sqrt(252)

    # === Helper: monthly rebalance wrapper ===
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

    def inv_vol_weights(date, sectors=None):
        """Compute inverse-vol weights for given sectors."""
        secs = sectors or available_sectors
        weights = {}
        for etf in secs:
            if etf in sec_vol63 and date in sec_vol63[etf].index:
                vol = sec_vol63[etf].loc[date]
                if not pd.isna(vol) and vol > 0.01:
                    weights[etf] = 1.0 / vol
        if weights:
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        return weights

    def top_n_by_momentum(date, n=5, lookback="63d"):
        """Get top N sectors by momentum."""
        mom = sec_mom63 if lookback == "63d" else sec_mom21
        scored = []
        for etf in available_sectors:
            if etf in mom and date in mom[etf].index:
                val = mom[etf].loc[date]
                if not pd.isna(val):
                    scored.append((etf, val))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [etf for etf, _ in scored[:n]]

    # =================================================================
    # Strategy F: Reversal Timing — buy momentum sectors after dips
    # =================================================================
    print("\n" + "="*60)
    print("F: Momentum sectors + buy-the-dip timing")
    print("="*60)

    def core_f(data, date, idx):
        # Gate: SPY > SMA100
        if date in sma[100].index:
            s = sma[100].loc[date]
            if pd.isna(s) or spy_close.loc[date] <= s:
                return {"invested": True, "weights": {"TLT": 0.5, "GLD": 0.5}}

        # Get top 5 by 63d momentum
        top5 = top_n_by_momentum(date, 5, "63d")
        if not top5:
            return {"invested": True, "weights": {BENCHMARK: 1.0}}

        # Among top 5, overweight those with negative 5d return (dip)
        weights = {}
        for etf in top5:
            base_w = 1.0
            if etf in sec_mom5 and date in sec_mom5[etf].index:
                r5 = sec_mom5[etf].loc[date]
                if not pd.isna(r5) and r5 < -0.01:  # had a dip
                    base_w = 2.0  # double weight for dip sectors
            if etf in sec_vol63 and date in sec_vol63[etf].index:
                vol = sec_vol63[etf].loc[date]
                if not pd.isna(vol) and vol > 0:
                    weights[etf] = base_w / vol

        if weights:
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            return {"invested": True, "weights": weights}
        return {"invested": True, "weights": {BENCHMARK: 1.0}}

    sig_f = monthly_wrap(core_f)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_f)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy G: Credit-aware sector rotation
    # =================================================================
    print("\n" + "="*60)
    print("G: Credit-aware regime + sector rotation")
    print("="*60)

    def core_g(data, date, idx):
        # Regime: SPY trend + credit health
        risk_on = True
        equity_pct = 1.0

        # SMA check
        if date in sma[100].index:
            s = sma[100].loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                equity_pct = 0.2

        # Credit check: if credit is deteriorating, reduce exposure
        if credit_signal is not None and date in credit_signal.index:
            cs = credit_signal.loc[date]
            if not pd.isna(cs) and cs < -0.02:  # credit widening
                equity_pct = min(equity_pct, 0.5)

        # Sector allocation
        sec_w = inv_vol_weights(date)
        weights = {}
        for etf, w in sec_w.items():
            weights[etf] = w * equity_pct
        hedge = 1.0 - equity_pct
        if hedge > 0:
            weights["TLT"] = hedge * 0.5
            weights["GLD"] = hedge * 0.5
        return {"invested": True, "weights": weights}

    sig_g = monthly_wrap(core_g)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_g)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy H: Defensive sectors in bear + all sectors in bull
    # =================================================================
    print("\n" + "="*60)
    print("H: Defensive/offensive sector rotation by regime")
    print("="*60)

    DEFENSIVE = ["XLP", "XLU", "XLV"]  # staples, utilities, healthcare
    OFFENSIVE = ["XLK", "XLY", "XLI", "XLF", "XLE", "XLB", "XLC"]

    def core_h(data, date, idx):
        if date in sma[100].index:
            s = sma[100].loc[date]
            if not pd.isna(s) and spy_close.loc[date] <= s:
                # Bear: defensive sectors + TLT + GLD
                def_w = inv_vol_weights(date, [e for e in DEFENSIVE if e in available_sectors])
                weights = {}
                for etf, w in def_w.items():
                    weights[etf] = w * 0.4  # 40% defensive sectors
                weights["TLT"] = 0.35
                weights["GLD"] = 0.25
                return {"invested": True, "weights": weights}

        # Bull: all sectors inv-vol weighted
        weights = inv_vol_weights(date)
        return {"invested": True, "weights": weights}

    sig_h = monthly_wrap(core_h)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_h)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)

    # =================================================================
    # Strategy I: FULL COMBO — all signals
    # =================================================================
    print("\n" + "="*60)
    print("I: Full Combo — trend + credit + vol + defensive/offensive")
    print("="*60)

    def core_i(data, date, idx):
        # Score regime 0-100
        regime_score = 50  # neutral

        # Trend: SPY > SMA100 (+25), > SMA200 (+15)
        for p, bonus in [(100, 25), (200, 15)]:
            if date in sma[p].index:
                s = sma[p].loc[date]
                if not pd.isna(s):
                    if spy_close.loc[date] > s:
                        regime_score += bonus
                    else:
                        regime_score -= bonus

        # Vol: low vol is bullish
        if date in spy_vol21.index:
            v = spy_vol21.loc[date]
            if not pd.isna(v):
                if v < 0.12:
                    regime_score += 10
                elif v > 0.25:
                    regime_score -= 15

        # Credit
        if credit_signal is not None and date in credit_signal.index:
            cs = credit_signal.loc[date]
            if not pd.isna(cs):
                if cs > 0.01:
                    regime_score += 10  # credit improving
                elif cs < -0.02:
                    regime_score -= 15  # credit deteriorating

        # Map score to allocation
        if regime_score >= 80:
            # Strong bull: aggressive sectors
            top5 = top_n_by_momentum(date, 5)
            if top5:
                w = inv_vol_weights(date, top5)
                return {"invested": True, "weights": w}
        elif regime_score >= 50:
            # Mild bull: all sectors inv-vol
            w = inv_vol_weights(date)
            return {"invested": True, "weights": w}
        elif regime_score >= 30:
            # Cautious: 60% defensive sectors, 40% TLT/GLD
            def_sectors = [e for e in DEFENSIVE if e in available_sectors]
            w = inv_vol_weights(date, def_sectors)
            weights = {}
            for etf, wt in w.items():
                weights[etf] = wt * 0.6
            weights["TLT"] = 0.25
            weights["GLD"] = 0.15
            return {"invested": True, "weights": weights}
        else:
            # Bear: mostly TLT/GLD
            return {"invested": True, "weights": {"TLT": 0.5, "GLD": 0.3, "XLP": 0.1, "XLV": 0.1}}

    sig_i = monthly_wrap(core_i)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        rets, trades = backtest_sector_strategy(data, s, e, sig_i)
        m = compute_metrics(rets)
        spy = spy_metrics(data, s, e)
        print_results(name, m, spy, trades)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
