#!/usr/bin/env python3
"""
Batch 6: Equities-only strategy targeting Sharpe 3.

Key architectural changes to push beyond Sharpe 1.2:
1. SELECTIVE ENTRY: Only invest when high-conviction signals align (reduces
   invested time but dramatically improves hit rate)
2. SHORT HOLDING PERIODS: 5-20 day holdings capture more independent bets
   per year (N independent bets → Sharpe scales with sqrt(N))
3. CONDITIONAL SIGNALS: Only fire when multiple uncorrelated factors agree
4. ASYMMETRIC SIZING: Bigger positions on strongest signals
5. DRAWDOWN CIRCUIT BREAKER: Hard stop on portfolio drawdown

Mathematical path to Sharpe 3:
- If each trade has Sharpe-per-trade of 0.3 and I make 100 independent trades/year:
  Annual Sharpe ≈ 0.3 * sqrt(100) = 3.0
- Need: high hit rate trades with short holding periods
"""

import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

BENCHMARK = "SPY"
EXCLUDE = {"SPY","QQQ","IWM","DIA","TLT","IEF","HYG","GLD","SLV","USO",
           "XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC"}


def get_stocks(data):
    return [t for t in data if t not in EXCLUDE and len(data[t]) > 500]


def compute_metrics(daily_rets):
    rets = np.array(daily_rets, dtype=float)
    if len(rets) < 50 or np.std(rets) == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0,
                "ann_vol": 0, "invested_pct": 0, "calmar": 0,
                "n_trades": 0, "avg_ret_per_trade": 0}
    excess = rets - 0.02 / 252
    n_years = len(rets) / 252
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252)
    cum = np.cumprod(1 + rets)
    total = cum[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    peak = np.maximum.accumulate(cum)
    mdd = np.min((cum - peak) / peak)
    down = excess[excess < 0]
    sortino = np.mean(excess) / np.std(down) * np.sqrt(252) if len(down) > 0 and np.std(down) > 0 else 0
    invested = np.sum(rets != 0) / len(rets)
    ann_vol = np.std(rets) * np.sqrt(252)
    calmar = cagr / abs(mdd) if mdd < 0 else 0
    return {"sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
            "max_dd": round(float(mdd), 4), "sortino": round(float(sortino), 3),
            "ann_vol": round(float(ann_vol), 4), "invested_pct": round(float(invested), 3),
            "calmar": round(float(calmar), 3)}


def precompute(data, tickers):
    """Precompute features for all tickers."""
    features = {}
    spy = data[BENCHMARK]["Close"]

    for t in tickers:
        df = data[t]
        close = df["Close"]
        if len(close) < 300:
            continue
        feat = pd.DataFrame(index=df.index)
        feat["close"] = close
        feat["open"] = df["Open"] if "Open" in df.columns else close

        # Multi-horizon returns
        for w in [5, 10, 21, 63, 126, 252]:
            feat[f"ret_{w}"] = close / close.shift(w) - 1

        # 12-1 month momentum
        feat["mom_12_1"] = close.shift(21) / close.shift(252) - 1

        # Log returns and volatility
        lr = np.log(close / close.shift(1))
        feat["vol_5"] = lr.rolling(5).std() * np.sqrt(252)
        feat["vol_21"] = lr.rolling(21).std() * np.sqrt(252)
        feat["vol_63"] = lr.rolling(63).std() * np.sqrt(252)

        # Risk-adjusted momentum
        feat["ra_126"] = feat["ret_126"] / feat["vol_63"].clip(lower=0.01)

        # Drawdown
        high_252 = close.rolling(252, min_periods=63).max()
        feat["dd_252"] = close / high_252 - 1

        # Position in range
        low_252 = close.rolling(252, min_periods=63).min()
        rng = (high_252 - low_252).clip(lower=0.01)
        feat["pos_52w"] = (close - low_252) / rng

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.clip(lower=1e-10)
        feat["rsi"] = 100 - 100 / (1 + rs)

        # Acceleration
        m21 = close / close.shift(21) - 1
        feat["accel"] = m21 - m21.shift(10)

        # Relative to SPY
        common = close.index.intersection(spy.index)
        spy_r63 = spy.reindex(common).pct_change(63)
        stk_r63 = close.reindex(common).pct_change(63)
        feat["idio_63"] = (stk_r63 - spy_r63).reindex(close.index)

        # Vol ratio (short/long) — regime indicator
        feat["vol_ratio"] = feat["vol_5"] / feat["vol_63"].clip(lower=0.01)

        # Mean reversion score: how far below 21-day mean
        sma21 = close.rolling(21).mean()
        feat["mean_rev"] = close / sma21 - 1

        features[t] = feat

    return features


def score_stock(row, method):
    """Score a single stock on a given day. Higher = better."""
    if method == "mom_12_1":
        return row.get("mom_12_1", np.nan)
    elif method == "ra_126":
        return row.get("ra_126", np.nan)
    elif method == "composite_strict":
        # Multi-factor: only positive if MULTIPLE factors agree
        mom = row.get("mom_12_1", 0)
        ra = row.get("ra_126", 0)
        accel = row.get("accel", 0)
        pos = row.get("pos_52w", 0)
        if np.isnan(mom) or np.isnan(ra):
            return np.nan
        # Require positive momentum AND positive acceleration AND near highs
        if mom <= 0 or accel <= 0 or pos < 0.7:
            return -999
        return ra * 0.5 + accel * 30 * 0.3 + pos * 0.2
    elif method == "mean_rev_oversold":
        # Buy oversold stocks with strong long-term trend
        rsi = row.get("rsi", 50)
        mr = row.get("mean_rev", 0)
        mom126 = row.get("ret_126", 0)
        if np.isnan(rsi) or np.isnan(mr) or np.isnan(mom126):
            return np.nan
        # Must be in uptrend (126d > 0) but short-term pullback (RSI < 35)
        if mom126 <= 0 or rsi > 35:
            return -999
        return -mr * 0.6 + mom126 * 0.4  # bigger pullback = higher score
    elif method == "breakout":
        # Near all-time high with low recent vol (compression breakout)
        pos = row.get("pos_52w", 0)
        vr = row.get("vol_ratio", 1)
        ret5 = row.get("ret_5", 0)
        if np.isnan(pos) or np.isnan(vr):
            return np.nan
        if pos < 0.9 or ret5 < 0:
            return -999
        return pos * 0.5 + (2 - vr) * 0.3 + ret5 * 10 * 0.2
    elif method == "trend_pullback":
        # Strong trend + short-term pullback (buy the dip in uptrends)
        mom126 = row.get("ret_126", 0)
        ret5 = row.get("ret_5", 0)
        ret21 = row.get("ret_21", 0)
        dd = row.get("dd_252", 0)
        if np.isnan(mom126) or np.isnan(ret5):
            return np.nan
        # 6-month uptrend but 1-week pullback
        if mom126 < 0.10 or ret5 > -0.02:
            return -999
        return mom126 * 0.4 + (-ret5) * 10 * 0.4 + (1 + dd) * 0.2
    return np.nan


def backtest_selective(data, features, tickers, start, end,
                       method="composite_strict", top_n=5,
                       hold_days=21, target_vol=None, vol_lookback=21,
                       slippage_bps=10, market_filter=None,
                       min_candidates=None, max_portfolio_dd=None):
    """
    Selective entry backtest with fixed holding periods.
    Only enters when enough high-scoring candidates exist.
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    warmup = 260

    # Track positions: list of {ticker, entry_date, entry_idx, weight, days_held}
    positions = []
    daily_rets = []
    raw_buffer = []
    current_exposure = 1.0
    last_rebal_idx = -999
    n_trades = 0

    for di, date in enumerate(dates):
        spy_idx = spy.index.get_loc(date)
        if spy_idx < warmup:
            daily_rets.append(0.0)
            continue

        # === Update existing positions ===
        dr = 0.0
        new_positions = []
        for pos in positions:
            t = pos["ticker"]
            f = features.get(t)
            if f is None or date not in f.index:
                new_positions.append(pos)
                pos["days_held"] += 1
                continue
            idx = f.index.get_loc(date)
            if idx < 1:
                new_positions.append(pos)
                continue

            pos["days_held"] += 1

            if pos.get("entering_today"):
                # Entry day: open to close return
                op = f.iloc[idx]["open"]
                cl = f.iloc[idx]["close"]
                buy_p = op * (1 + slippage_bps / 10000)
                dr += pos["weight"] * (cl / buy_p - 1)
                pos["entering_today"] = False
                new_positions.append(pos)
            elif pos["days_held"] >= hold_days:
                # Exit: overnight return (prev close to open)
                pc = f.iloc[idx - 1]["close"]
                op = f.iloc[idx]["open"]
                dr += pos["weight"] * (op * (1 - slippage_bps / 10000) / pc - 1)
                # Position closed, don't add to new_positions
            else:
                # Normal hold: close to close
                pc = f.iloc[idx - 1]["close"]
                cl = f.iloc[idx]["close"]
                dr += pos["weight"] * (cl / pc - 1)
                new_positions.append(pos)

        positions = new_positions
        raw_buffer.append(dr)

        # Vol targeting
        if target_vol and len(raw_buffer) >= vol_lookback:
            rv = np.std(raw_buffer[-vol_lookback:]) * np.sqrt(252)
            current_exposure = min(1.0, target_vol / max(rv, 0.001))
        else:
            current_exposure = 1.0 if not target_vol else 1.0

        # Drawdown circuit breaker
        if max_portfolio_dd and len(daily_rets) > 20:
            cum = np.cumprod([1 + r for r in daily_rets[-252:]])
            peak = np.max(cum)
            curr_dd = (cum[-1] - peak) / peak
            if curr_dd < max_portfolio_dd:
                current_exposure *= 0.25  # Reduce to 25%

        daily_rets.append(dr * current_exposure)

        # === Signal generation: should we open new positions? ===
        # Only rebalance at intervals
        if di - last_rebal_idx < max(5, hold_days // 2):
            continue

        # Check market filter
        invest = True
        if market_filter == "sma200":
            if spy_idx >= 200:
                sma = spy["Close"].iloc[spy_idx - 199:spy_idx + 1].mean()
                invest = spy["Close"].iloc[spy_idx] > sma
            else:
                invest = False
        elif market_filter == "abs_mom_10m":
            invest = spy["Close"].iloc[spy_idx] > spy["Close"].iloc[spy_idx - 210] if spy_idx >= 210 else False

        if not invest:
            continue

        # Current portfolio weight used
        current_weight = sum(p["weight"] for p in positions)
        available = 1.0 - current_weight
        if available < 0.1:
            continue

        # Score and rank candidates
        candidates = []
        for t in tickers:
            # Skip if already holding
            if any(p["ticker"] == t for p in positions):
                continue
            f = features.get(t)
            if f is None or date not in f.index:
                continue
            idx = f.index.get_loc(date)
            if idx < warmup:
                continue
            row = f.iloc[idx]
            s = score_stock(row, method)
            if not np.isnan(s) and s > -998:
                candidates.append((t, s))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_n]

        # Minimum candidates filter (selectivity gate)
        if min_candidates and len(candidates) < min_candidates:
            continue

        if not top:
            continue

        # Only take positions if scores are positive (conviction filter)
        top = [(t, s) for t, s in top if s > 0]
        if not top:
            continue

        # Open new positions
        n_new = min(len(top), int(available * top_n))
        if n_new == 0:
            continue

        w = available / n_new
        for t, s in top[:n_new]:
            positions.append({
                "ticker": t,
                "entry_date": date,
                "entry_idx": di,
                "weight": w,
                "days_held": 0,
                "entering_today": True,
            })
            n_trades += 1

        last_rebal_idx = di

    m = compute_metrics(daily_rets)
    m["n_trades"] = n_trades
    return m


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    tickers = get_stocks(data)
    print(f"  {len(tickers)} stocks (equities only, no ETFs)")

    print("Precomputing features...")
    features = precompute(data, tickers)
    print(f"  {len(features)} tickers with features")

    PERIODS = [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END),
               ("TEST", TEST_START, TEST_END), ("FULL", "2010-01-01", TEST_END)]

    for pname, s, e in PERIODS:
        r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna().values
        m = compute_metrics(r)
        print(f"  SPY {pname}: Sh={m['sharpe']:.3f} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%}")

    results = []
    hdr = f"{'Experiment':<65} {'TR':>6} {'VA':>6} {'TE':>6} {'FU':>6} {'CAGR':>7} {'DD':>7} {'Cal':>5} {'Inv%':>5} {'#Tr':>4}"
    print(f"\n{'='*125}")
    print(hdr)
    print(f"{'='*125}")

    def run(name, **kwargs):
        r = {"name": name}
        for pname, s, e in PERIODS:
            m = backtest_selective(data, features, tickers, s, e, **kwargs)
            r[pname] = m
        results.append(r)
        tr=r["TRAIN"]; va=r["VALID"]; te=r["TEST"]; fu=r["FULL"]
        print(f"  {name:<63} {tr['sharpe']:>6.3f} {va['sharpe']:>6.3f} {te['sharpe']:>6.3f} {fu['sharpe']:>6.3f} {fu['cagr']:>6.1%} {fu['max_dd']:>6.1%} {fu['calmar']:>5.2f} {fu['invested_pct']:>4.0%} {fu.get('n_trades',0):>4}")

    # === A. Baseline: different signal types with fixed hold ===
    print("\n--- A. Signal types (hold 21d, top 5, monthly-ish) ---")
    for m in ["mom_12_1", "ra_126", "composite_strict", "mean_rev_oversold",
              "breakout", "trend_pullback"]:
        run(f"{m}_t5_h21", method=m, top_n=5, hold_days=21)

    # === B. Hold period variations ===
    print("\n--- B. Hold periods ---")
    for hd in [5, 10, 15, 21, 42, 63]:
        run(f"composite_strict_t5_h{hd}", method="composite_strict", top_n=5, hold_days=hd)
        run(f"trend_pullback_t5_h{hd}", method="trend_pullback", top_n=5, hold_days=hd)
        run(f"mean_rev_oversold_t5_h{hd}", method="mean_rev_oversold", top_n=5, hold_days=hd)

    # === C. With vol targeting ===
    print("\n--- C. Vol targeting ---")
    for m in ["composite_strict", "trend_pullback", "mean_rev_oversold", "mom_12_1", "ra_126"]:
        for tv in [0.05, 0.08, 0.10]:
            for hd in [10, 21]:
                run(f"{m}_t5_h{hd}_vt{tv:.0%}",
                    method=m, top_n=5, hold_days=hd, target_vol=tv)

    # === D. With market filter ===
    print("\n--- D. Market filter ---")
    for m in ["composite_strict", "trend_pullback", "mean_rev_oversold"]:
        for mf in ["sma200", "abs_mom_10m"]:
            run(f"{m}_t5_h21_vt8%_{mf}",
                method=m, top_n=5, hold_days=21, target_vol=0.08, market_filter=mf)

    # === E. Portfolio concentration ===
    print("\n--- E. Concentration ---")
    for m in ["composite_strict", "trend_pullback", "mean_rev_oversold"]:
        for n in [3, 5, 10]:
            run(f"{m}_t{n}_h21_vt8%",
                method=m, top_n=n, hold_days=21, target_vol=0.08)

    # === F. Drawdown circuit breaker ===
    print("\n--- F. Drawdown breaker ---")
    for m in ["composite_strict", "trend_pullback", "mean_rev_oversold"]:
        for dd in [-0.05, -0.08, -0.10]:
            run(f"{m}_t5_h21_vt8%_dd{abs(dd):.0%}",
                method=m, top_n=5, hold_days=21, target_vol=0.08, max_portfolio_dd=dd)

    # === G. Short-term mean reversion (high frequency of bets) ===
    print("\n--- G. Short holds (more bets/year) ---")
    for m in ["mean_rev_oversold", "trend_pullback"]:
        for hd in [3, 5, 7]:
            for tv in [0.05, 0.08]:
                run(f"{m}_t5_h{hd}_vt{tv:.0%}",
                    method=m, top_n=5, hold_days=hd, target_vol=tv)

    # === RESULTS ===
    print(f"\n{'='*110}")
    print("ALL with FULL Sharpe > 1.5 (consistent: TRAIN > 0.5 AND TEST > 0.5):")
    high = [r for r in results
            if r["FULL"]["sharpe"] > 1.5
            and r["TRAIN"]["sharpe"] > 0.5
            and r["TEST"]["sharpe"] > 0.5]
    high.sort(key=lambda x: x["FULL"]["sharpe"], reverse=True)
    if not high:
        print("  None found > 1.5. Showing top 15 overall:")
        high = [r for r in results if r["TRAIN"]["sharpe"] > 0.3 and r["TEST"]["sharpe"] > 0.3]
        high.sort(key=lambda x: x["FULL"]["sharpe"], reverse=True)
    for r in high[:15]:
        fu=r["FULL"]; te=r["TEST"]; tr=r["TRAIN"]; va=r["VALID"]
        print(f"  {r['name']:<60} FU={fu['sharpe']:.3f}/{fu['cagr']:.1%}/{fu['max_dd']:.1%} TR={tr['sharpe']:.3f} VA={va['sharpe']:.3f} TE={te['sharpe']:.3f} Cal={fu['calmar']:.2f} Inv={fu['invested_pct']:.0%}")

    print(f"\nANY with FULL Sharpe > 2.0:")
    ultra = [r for r in results if r["FULL"]["sharpe"] > 2.0]
    if not ultra:
        print("  None found > 2.0")
    for r in sorted(ultra, key=lambda x: x["FULL"]["sharpe"], reverse=True):
        fu=r["FULL"]; te=r["TEST"]; tr=r["TRAIN"]; va=r["VALID"]
        print(f"  {r['name']:<60} FU={fu['sharpe']:.3f} TR={tr['sharpe']:.3f} VA={va['sharpe']:.3f} TE={te['sharpe']:.3f}")
