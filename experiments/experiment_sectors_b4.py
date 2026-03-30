#!/usr/bin/env python3
"""
Batch 4: Fundamentally different approaches to achieve Sharpe 3+.
Expands to full stock universe (100+ tickers), not just sector ETFs.
All signals next-day open execution with slippage.
"""

import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

BENCHMARK = "SPY"
SECTOR_ETFS = ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC"]
# Exclude benchmarks and bond/commodity ETFs from the tradeable universe
EXCLUDE = {"SPY","QQQ","IWM","DIA","TLT","IEF","HYG","GLD","SLV","USO"}


def get_tradeable(data):
    return [t for t in data if t not in EXCLUDE and len(data[t]) > 500]


def compute_metrics(daily_rets):
    rets = np.array(daily_rets, dtype=float)
    if len(rets) < 50 or np.std(rets) == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0,
                "ann_vol": 0, "time_invested": 0, "calmar": 0}
    excess = rets - 0.02 / 252
    n_years = len(rets) / 252
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252)
    cum = np.cumprod(1 + rets)
    total = cum[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = np.min(dd)
    down = excess[excess < 0]
    sortino = np.mean(excess) / np.std(down) * np.sqrt(252) if len(down) > 0 and np.std(down) > 0 else 0
    invested = np.sum(rets != 0) / len(rets)
    ann_vol = np.std(rets) * np.sqrt(252)
    calmar = cagr / abs(mdd) if mdd < 0 else 0
    return {"sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
            "max_dd": round(float(mdd), 4), "sortino": round(float(sortino), 3),
            "ann_vol": round(float(ann_vol), 4), "time_invested": round(float(invested), 3),
            "calmar": round(float(calmar), 3)}


def precompute_features(data, tickers):
    """Precompute all features for all tickers. Returns dict of DataFrames."""
    features = {}
    spy_close = data[BENCHMARK]["Close"] if BENCHMARK in data else None

    for ticker in tickers:
        df = data[ticker]
        close = df["Close"]
        n = len(close)
        if n < 300:
            continue

        feat = pd.DataFrame(index=df.index)

        # Returns at multiple horizons
        for w in [5, 10, 21, 63, 126, 252]:
            feat[f"ret_{w}"] = close / close.shift(w) - 1

        # 12-1 month momentum (skip most recent month)
        feat["mom_12_1"] = close.shift(21) / close.shift(252) - 1

        # Acceleration: change in 21d momentum over 10 days
        m21 = close / close.shift(21) - 1
        feat["accel_21_10"] = m21 - m21.shift(10)

        # Acceleration: change in 5d momentum over 5 days
        m5 = close / close.shift(5) - 1
        feat["accel_5_5"] = m5 - m5.shift(5)

        # Volatility
        log_ret = np.log(close / close.shift(1))
        feat["vol_21"] = log_ret.rolling(21).std() * np.sqrt(252)
        feat["vol_63"] = log_ret.rolling(63).std() * np.sqrt(252)

        # Risk-adjusted momentum
        for w in [63, 126]:
            vol = log_ret.rolling(w).std() * np.sqrt(252)
            feat[f"risk_adj_{w}"] = feat[f"ret_{w}"] / vol.clip(lower=0.01)

        # Drawdown from 252-day high
        high_252 = close.rolling(252, min_periods=63).max()
        feat["dd_252"] = close / high_252 - 1

        # Position in 52-week range
        low_252 = close.rolling(252, min_periods=63).min()
        rng = (high_252 - low_252).clip(lower=0.01)
        feat["pos_52w"] = (close - low_252) / rng

        # RSI (14-day)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.clip(lower=1e-10)
        feat["rsi_14"] = 100 - (100 / (1 + rs))

        # Relative to SPY (idiosyncratic momentum)
        if spy_close is not None:
            common = close.index.intersection(spy_close.index)
            spy_ret_63 = spy_close.reindex(common).pct_change(63)
            stock_ret_63 = close.reindex(common).pct_change(63)
            feat["idio_mom_63"] = (stock_ret_63 - spy_ret_63).reindex(close.index)

        # Open prices for next-day execution
        if "Open" in df.columns:
            feat["open"] = df["Open"]
        else:
            feat["open"] = close
        feat["close"] = close

        features[ticker] = feat

    return features


def rank_universe(features, date, tickers, method="composite", top_n=10):
    """Rank tickers by signal. Returns [(ticker, score), ...] top_n."""
    scores = []
    for t in tickers:
        f = features.get(t)
        if f is None or date not in f.index:
            continue
        idx = f.index.get_loc(date)
        if idx < 260:
            continue
        row = f.iloc[idx]

        if method == "mom_63":
            s = row.get("ret_63", np.nan)
        elif method == "mom_126":
            s = row.get("ret_126", np.nan)
        elif method == "mom_12_1":
            s = row.get("mom_12_1", np.nan)
        elif method == "accel_21":
            s = row.get("accel_21_10", np.nan)
        elif method == "risk_adj_126":
            s = row.get("risk_adj_126", np.nan)
        elif method == "risk_adj_63":
            s = row.get("risk_adj_63", np.nan)
        elif method == "idio_mom_63":
            s = row.get("idio_mom_63", np.nan)
        elif method == "inv_vol":
            v = row.get("vol_21", np.nan)
            s = -v if not np.isnan(v) else np.nan
        elif method == "composite":
            # Multi-factor composite
            mom = row.get("ret_63", 0) if not np.isnan(row.get("ret_63", np.nan)) else 0
            mom126 = row.get("ret_126", 0) if not np.isnan(row.get("ret_126", np.nan)) else 0
            accel = row.get("accel_21_10", 0) if not np.isnan(row.get("accel_21_10", np.nan)) else 0
            vol = row.get("vol_21", 0.15) if not np.isnan(row.get("vol_21", np.nan)) else 0.15
            ra = mom / max(vol, 0.01)
            # Combine: momentum + acceleration + quality
            s = ra * 0.4 + accel * 20 * 0.3 + mom126 / max(vol, 0.01) * 0.3
        elif method == "quality_momentum":
            # Quality: low drawdown + high RSI + strong momentum
            dd = row.get("dd_252", -0.5) if not np.isnan(row.get("dd_252", np.nan)) else -0.5
            rsi = row.get("rsi_14", 50) if not np.isnan(row.get("rsi_14", np.nan)) else 50
            mom = row.get("ret_63", 0) if not np.isnan(row.get("ret_63", np.nan)) else 0
            pos = row.get("pos_52w", 0.5) if not np.isnan(row.get("pos_52w", np.nan)) else 0.5
            # High quality = near highs + positive momentum + not overbought
            s = (1 + dd) * 0.3 + mom * 0.4 + (pos if pos > 0.7 else 0) * 0.3
        elif method == "reversal_5d":
            # Short-term reversal: buy 5-day losers
            s = -(row.get("ret_5", 0) if not np.isnan(row.get("ret_5", np.nan)) else 0)
        elif method == "accel_quality":
            # Acceleration + quality combo
            accel = row.get("accel_21_10", 0) if not np.isnan(row.get("accel_21_10", np.nan)) else 0
            ra126 = row.get("risk_adj_126", 0) if not np.isnan(row.get("risk_adj_126", np.nan)) else 0
            dd = row.get("dd_252", -0.5) if not np.isnan(row.get("dd_252", np.nan)) else -0.5
            # Only consider stocks with positive acceleration AND strong long-term trend
            if accel <= 0 or ra126 <= 0:
                s = -999
            else:
                s = accel * 50 * 0.5 + ra126 * 0.3 + (1 + dd) * 0.2
        elif method == "fresh_highs":
            # Near 52-week high with acceleration
            pos = row.get("pos_52w", 0) if not np.isnan(row.get("pos_52w", np.nan)) else 0
            accel = row.get("accel_21_10", 0) if not np.isnan(row.get("accel_21_10", np.nan)) else 0
            if pos < 0.8:
                s = -999
            else:
                s = pos * 0.5 + accel * 30 * 0.5
        else:
            s = row.get("ret_63", np.nan)

        if not np.isnan(s) and s > -998:
            scores.append((t, float(s)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def backtest_nextopen(data, features, tickers, start, end,
                      selector="composite", top_n=10,
                      rebal_freq="monthly", target_vol=None,
                      vol_lookback=21, slippage_bps=10,
                      universe_filter=None):
    """
    Backtest with strict next-day-open execution.
    Signal at T close → execute at T+1 open.
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    warmup = 260

    if universe_filter == "sectors_only":
        trade_universe = [t for t in tickers if t in SECTOR_ETFS]
    elif universe_filter == "stocks_only":
        trade_universe = [t for t in tickers if t not in SECTOR_ETFS]
    else:
        trade_universe = tickers

    weights = {}  # {ticker: weight}
    pending = None  # pending allocation
    daily_rets = []
    raw_rets_buffer = []
    current_exposure = 1.0
    prev_month = None
    prev_week = None

    for date in dates:
        spy_idx = spy.index.get_loc(date)
        if spy_idx < warmup:
            daily_rets.append(0.0)
            continue

        # === EXECUTE PENDING at today's open ===
        if pending is not None:
            old_w = dict(weights)
            new_w = dict(pending)
            dr = 0.0

            all_tickers = set(list(old_w.keys()) + list(new_w.keys()))
            for t in all_tickers:
                f = features.get(t)
                if f is None or date not in f.index:
                    continue
                idx = f.index.get_loc(date)
                if idx < 1:
                    continue

                op = f.iloc[idx]["open"]
                cl = f.iloc[idx]["close"]
                pc = f.iloc[idx - 1]["close"]

                ow = old_w.get(t, 0)
                nw = new_w.get(t, 0)

                if ow > 0 and nw == 0:
                    # Selling: overnight return (prev close to open)
                    dr += ow * (op * (1 - slippage_bps / 10000) / pc - 1)
                elif ow == 0 and nw > 0:
                    # Buying: intraday return (open to close)
                    buy_p = op * (1 + slippage_bps / 10000)
                    dr += nw * (cl / buy_p - 1)
                elif ow > 0 and nw > 0:
                    # Keeping: full close-to-close + slippage on weight change
                    dr += nw * (cl / pc - 1)
                    dr -= abs(nw - ow) * slippage_bps / 10000

            weights = new_w

            # Apply vol targeting
            raw_rets_buffer.append(dr)
            if target_vol and len(raw_rets_buffer) >= vol_lookback:
                rv = np.std(raw_rets_buffer[-vol_lookback:]) * np.sqrt(252)
                current_exposure = min(1.0, target_vol / max(rv, 0.001))
            else:
                current_exposure = 1.0

            daily_rets.append(dr * current_exposure)
            pending = None
            prev_month = date.month
            prev_week = date.isocalendar()[1]
            continue

        # === Normal day return ===
        dr = 0.0
        if weights:
            for t, w in weights.items():
                f = features.get(t)
                if f is None or date not in f.index:
                    continue
                idx = f.index.get_loc(date)
                if idx < 1:
                    continue
                dr += w * (f.iloc[idx]["close"] / f.iloc[idx - 1]["close"] - 1)

        raw_rets_buffer.append(dr)

        # Vol targeting
        if target_vol and len(raw_rets_buffer) >= vol_lookback and weights:
            rv = np.std(raw_rets_buffer[-vol_lookback:]) * np.sqrt(252)
            current_exposure = min(1.0, target_vol / max(rv, 0.001))

        daily_rets.append(dr * current_exposure)

        # === Generate signal at close ===
        new_month = prev_month is None or date.month != prev_month
        new_week = prev_week is None or date.isocalendar()[1] != prev_week

        do_rebal = False
        if rebal_freq == "monthly" and new_month:
            do_rebal = True
        elif rebal_freq == "biweekly" and new_week and date.isocalendar()[1] % 2 == 0:
            do_rebal = True
        elif rebal_freq == "weekly" and new_week:
            do_rebal = True
        elif not weights:
            do_rebal = True  # Initial entry

        if do_rebal:
            ranked = rank_universe(features, date, trade_universe, method=selector, top_n=top_n)
            if ranked:
                w = 1.0 / len(ranked)
                new_alloc = {t: w for t, _ in ranked}
                if set(new_alloc.keys()) != set(weights.keys()):
                    pending = new_alloc
                else:
                    weights = new_alloc  # Same tickers, just reweight

        prev_month = date.month
        prev_week = date.isocalendar()[1]

    return compute_metrics(daily_rets)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    tickers = get_tradeable(data)
    print(f"  {len(data)} total, {len(tickers)} tradeable ({len([t for t in tickers if t in SECTOR_ETFS])} sectors, {len([t for t in tickers if t not in SECTOR_ETFS])} stocks)")

    print("\nPrecomputing features...")
    features = precompute_features(data, tickers)
    print(f"  Features computed for {len(features)} tickers")

    PERIODS = [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END),
               ("TEST", TEST_START, TEST_END), ("FULL", "2010-01-01", TEST_END)]

    # SPY baseline
    for pname, s, e in PERIODS:
        r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna().values
        m = compute_metrics(r)
        print(f"  SPY {pname}: Sh={m['sharpe']:.3f} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%}")

    print(f"\n{'='*160}")
    hdr = f"{'Experiment':<60} {'TR Sh':>6} {'TR CAGR':>8} {'VA Sh':>6} {'VA CAGR':>8} {'TE Sh':>6} {'TE CAGR':>8} {'FU Sh':>6} {'FU CAGR':>8} {'FU DD':>7} {'Vol':>5} {'TIM':>4} {'Calm':>5}"
    print(hdr)
    print(f"{'='*160}")

    results = []

    def run(name, **kwargs):
        r = {"name": name}
        for pname, s, e in PERIODS:
            m = backtest_nextopen(data, features, tickers, s, e, **kwargs)
            r[pname] = m
        results.append(r)
        tr=r["TRAIN"]; va=r["VALID"]; te=r["TEST"]; fu=r["FULL"]
        print(f"  {name:<58} {tr['sharpe']:>6.3f} {tr['cagr']:>7.1%} {va['sharpe']:>6.3f} {va['cagr']:>7.1%} {te['sharpe']:>6.3f} {te['cagr']:>7.1%} {fu['sharpe']:>6.3f} {fu['cagr']:>7.1%} {fu['max_dd']:>6.1%} {fu['ann_vol']:>5.1%} {fu['time_invested']:>3.0%} {fu['calmar']:>5.2f}")

    # === A. STOCK UNIVERSE — various selectors ===
    print("\n--- A. Stock universe, monthly rebalance, next-open execution ---")
    for sel in ["mom_63", "mom_126", "mom_12_1", "accel_21", "risk_adj_126",
                "risk_adj_63", "composite", "quality_momentum", "idio_mom_63",
                "accel_quality", "fresh_highs", "reversal_5d"]:
        for n in [5, 10]:
            run(f"stocks_{sel}_top{n}_monthly",
                selector=sel, top_n=n, rebal_freq="monthly",
                universe_filter="stocks_only")

    # === B. WITH VOL TARGETING ===
    print("\n--- B. Stocks + vol targeting ---")
    for sel in ["composite", "risk_adj_126", "accel_quality", "quality_momentum",
                "fresh_highs", "mom_12_1", "idio_mom_63"]:
        for tv in [0.08, 0.10, 0.12, 0.15]:
            run(f"stocks_{sel}_t10_vt{tv:.0%}_monthly",
                selector=sel, top_n=10, rebal_freq="monthly",
                target_vol=tv, universe_filter="stocks_only")

    # === C. WEEKLY REBALANCING ===
    print("\n--- C. Weekly rebalancing ---")
    for sel in ["composite", "risk_adj_126", "accel_21", "accel_quality",
                "quality_momentum", "reversal_5d"]:
        run(f"stocks_{sel}_t10_weekly",
            selector=sel, top_n=10, rebal_freq="weekly",
            universe_filter="stocks_only")
        run(f"stocks_{sel}_t10_vt10%_weekly",
            selector=sel, top_n=10, rebal_freq="weekly",
            target_vol=0.10, universe_filter="stocks_only")

    # === D. BIWEEKLY ===
    print("\n--- D. Biweekly rebalancing ---")
    for sel in ["composite", "risk_adj_126", "accel_quality"]:
        run(f"stocks_{sel}_t10_vt10%_biweekly",
            selector=sel, top_n=10, rebal_freq="biweekly",
            target_vol=0.10, universe_filter="stocks_only")

    # === E. FULL UNIVERSE (stocks + sectors) ===
    print("\n--- E. Full universe (stocks + sectors) ---")
    for sel in ["composite", "risk_adj_126", "accel_quality", "quality_momentum"]:
        run(f"all_{sel}_t10_vt10%_monthly",
            selector=sel, top_n=10, rebal_freq="monthly",
            target_vol=0.10)

    # === F. CONCENTRATION (top 3-5) ===
    print("\n--- F. Higher concentration ---")
    for sel in ["composite", "risk_adj_126", "accel_quality", "quality_momentum"]:
        for n in [3, 5]:
            run(f"stocks_{sel}_t{n}_vt10%_monthly",
                selector=sel, top_n=n, rebal_freq="monthly",
                target_vol=0.10, universe_filter="stocks_only")

    # === G. SECTORS ONLY (baseline comparison) ===
    print("\n--- G. Sectors only (baseline) ---")
    for sel in ["composite", "risk_adj_126", "accel_21"]:
        run(f"sectors_{sel}_t3_vt15%_monthly",
            selector=sel, top_n=3, rebal_freq="monthly",
            target_vol=0.15, universe_filter="sectors_only")

    # === RESULTS ===
    print(f"\n{'='*100}")
    print("TOP 20 by FULL Sharpe (must have TRAIN > 0.3 AND TEST > 0.3):")
    consistent = [r for r in results
                  if r.get("TRAIN", {}).get("sharpe", 0) > 0.3
                  and r.get("TEST", {}).get("sharpe", 0) > 0.3]
    consistent.sort(key=lambda x: x.get("FULL", {}).get("sharpe", 0), reverse=True)
    for i, r in enumerate(consistent[:20]):
        fu = r["FULL"]; te = r["TEST"]; tr = r["TRAIN"]; va = r["VALID"]
        print(f"  {i+1:>2}. {r['name']:<55} FU={fu['sharpe']:.3f}/{fu['cagr']:.1%}/{fu['max_dd']:.1%} TR={tr['sharpe']:.3f} VA={va['sharpe']:.3f} TE={te['sharpe']:.3f} Cal={fu['calmar']:.2f}")

    print(f"\nTOP 10 by FULL Calmar:")
    consistent.sort(key=lambda x: x.get("FULL", {}).get("calmar", 0), reverse=True)
    for i, r in enumerate(consistent[:10]):
        fu = r["FULL"]; te = r["TEST"]; tr = r["TRAIN"]
        print(f"  {i+1:>2}. {r['name']:<55} Cal={fu['calmar']:.2f} Sh={fu['sharpe']:.3f} CAGR={fu['cagr']:.1%} DD={fu['max_dd']:.1%}")

    print(f"\nANY strategy with FULL Sharpe > 1.0:")
    high_sh = [r for r in results if r.get("FULL", {}).get("sharpe", 0) > 1.0]
    if not high_sh:
        print("  None found.")
    for r in sorted(high_sh, key=lambda x: x["FULL"]["sharpe"], reverse=True):
        fu = r["FULL"]; te = r["TEST"]; tr = r["TRAIN"]; va = r["VALID"]
        print(f"  {r['name']:<55} FU={fu['sharpe']:.3f}/{fu['cagr']:.1%}/{fu['max_dd']:.1%} TR={tr['sharpe']:.3f} VA={va['sharpe']:.3f} TE={te['sharpe']:.3f}")
