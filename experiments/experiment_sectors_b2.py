#!/usr/bin/env python3
"""
Batch 2: Creative/novel sector ETF strategy experiments.
Tests volatility targeting, acceleration momentum, defensive rotation,
sector dispersion timing, same-day execution, and multi-filter approaches.
"""

import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

CORE = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
ALL = CORE + ["XLRE", "XLC"]
DEFENSIVE = ["XLP", "XLU", "XLV"]
CYCLICAL = ["XLK", "XLY", "XLF", "XLI", "XLB", "XLE"]
BENCHMARK = "SPY"


def compute_metrics(daily_rets):
    rets = pd.Series(daily_rets)
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0, "time_in_market": 0, "ann_vol": 0}
    excess = rets - 0.02 / 252
    n_years = len(rets) / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    downside = excess[excess < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    invested = (rets != 0).sum() / len(rets)
    ann_vol = rets.std() * np.sqrt(252)
    return {"sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
            "max_dd": round(float(mdd), 4), "sortino": round(float(sortino), 3),
            "time_in_market": round(float(invested), 3), "ann_vol": round(float(ann_vol), 4)}


def get_mom(df, idx, lookback):
    if idx < lookback:
        return np.nan
    return df.iloc[idx]["Close"] / df.iloc[idx - lookback]["Close"] - 1


def get_open(df, si):
    if "Open" in df.columns:
        op = df.iloc[si]["Open"]
        if not np.isnan(op) and op > 0:
            return op
    return df.iloc[si]["Close"]


def rank_sectors(data, date, method="mom_63"):
    """Rank sectors. Returns [(etf, score), ...]."""
    scores = []
    for etf in ALL:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)

        if method == "mom_21":
            s = get_mom(df, idx, 21)
        elif method == "mom_63":
            s = get_mom(df, idx, 63)
        elif method == "mom_126":
            s = get_mom(df, idx, 126)
        elif method == "blended":
            m21 = get_mom(df, idx, 21)
            m63 = get_mom(df, idx, 63)
            m126 = get_mom(df, idx, 126)
            vals = [(m, w) for m, w in [(m21, 0.25), (m63, 0.5), (m126, 0.25)] if not np.isnan(m)]
            s = sum(m * w for m, w in vals) / sum(w for _, w in vals) if vals else np.nan
        elif method == "risk_adj_126":
            m = get_mom(df, idx, 126)
            rets = df["Close"].iloc[max(0, idx - 126):idx + 1].pct_change().dropna()
            vol = rets.std() * np.sqrt(252) if len(rets) > 5 else 0.15
            s = m / max(vol, 0.01) if not np.isnan(m) else np.nan
        elif method == "accel_63":
            # Momentum acceleration: change in 63d mom over last 21 days
            m_now = get_mom(df, idx, 63)
            m_prev = get_mom(df, idx - 21, 63) if idx >= 84 else np.nan
            s = (m_now - m_prev) if not np.isnan(m_now) and not np.isnan(m_prev) else np.nan
        elif method == "accel_21":
            m_now = get_mom(df, idx, 21)
            m_prev = get_mom(df, idx - 10, 21) if idx >= 31 else np.nan
            s = (m_now - m_prev) if not np.isnan(m_now) and not np.isnan(m_prev) else np.nan
        elif method == "inv_vol":
            rets = df["Close"].iloc[max(0, idx - 63):idx + 1].pct_change().dropna()
            vol = rets.std() * np.sqrt(252) if len(rets) > 5 else 0.15
            s = 1.0 / max(vol, 0.01)
        else:
            s = get_mom(df, idx, 63)

        if not np.isnan(s):
            scores.append((etf, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ============================================================
# STRATEGY: Volatility-targeted sector momentum
# ============================================================

def backtest_voltarget(data, start, end, top_n=3, selector="mom_63",
                       target_vol=0.10, vol_lookback=21, rebalance="monthly",
                       timing="sma200", slippage_bps=5, exec_mode="next_open"):
    """
    Sector momentum with volatility targeting.
    Scales total exposure to maintain target portfolio volatility.
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    warmup = 260

    weights = {}  # {etf: weight} (pre-scaling)
    in_market = False
    prev_month = None
    pending = None
    daily_rets = []
    exposure_history = []

    # Track portfolio returns for vol estimation
    port_rets_buffer = []

    for date in dates:
        spy_idx = spy.index.get_loc(date)
        if spy_idx < warmup:
            daily_rets.append(0.0)
            continue

        dr = 0.0

        # Execute pending
        if pending is not None:
            if len(pending) == 0 and in_market:
                # EXIT
                if exec_mode == "next_open":
                    for etf, w in weights.items():
                        df = data[etf]
                        if date in df.index:
                            si = df.index.get_loc(date)
                            if si > 0:
                                op = get_open(df, si)
                                pc = df.iloc[si - 1]["Close"]
                                dr += w * (op * (1 - slippage_bps / 10000) / pc - 1)
                weights = {}
                in_market = False
            elif len(pending) > 0 and not in_market:
                # ENTER
                if exec_mode == "next_open":
                    for etf, w in pending.items():
                        df = data[etf]
                        if date in df.index:
                            si = df.index.get_loc(date)
                            op = get_open(df, si)
                            cl = df.iloc[si]["Close"]
                            bp = op * (1 + slippage_bps / 10000)
                            dr += w * (cl / bp - 1)
                else:
                    pass  # same-day: no partial return on entry day
                weights = dict(pending)
                in_market = True
            elif len(pending) > 0 and in_market:
                # REBALANCE
                old_w = dict(weights)
                new_w = dict(pending)
                if exec_mode == "next_open":
                    for etf in set(list(old_w.keys()) + list(new_w.keys())):
                        df = data.get(etf)
                        if df is None or date not in df.index:
                            continue
                        si = df.index.get_loc(date)
                        if si == 0:
                            continue
                        op = get_open(df, si)
                        cl = df.iloc[si]["Close"]
                        pc = df.iloc[si - 1]["Close"]
                        ow = old_w.get(etf, 0)
                        nw = new_w.get(etf, 0)
                        if ow > 0 and nw == 0:
                            dr += ow * (op * (1 - slippage_bps / 10000) / pc - 1)
                        elif ow == 0 and nw > 0:
                            dr += nw * (cl / (op * (1 + slippage_bps / 10000)) - 1)
                        elif ow > 0 and nw > 0:
                            dr += nw * (cl / pc - 1)
                            dr -= abs(nw - ow) * slippage_bps / 10000
                weights = dict(pending)

            pending = None
            daily_rets.append(dr)
            port_rets_buffer.append(dr)
            prev_month = date.month
            continue

        # Normal day return (with vol-targeted exposure)
        raw_dr = 0.0
        if in_market and weights:
            for etf, w in weights.items():
                df = data[etf]
                if date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        raw_dr += w * (df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1)

        # Apply vol targeting
        if len(port_rets_buffer) >= vol_lookback and in_market:
            recent_vol = np.std(port_rets_buffer[-vol_lookback:]) * np.sqrt(252)
            if recent_vol > 0.001:
                exposure = min(1.0, target_vol / recent_vol)
            else:
                exposure = 1.0
        else:
            exposure = 1.0

        dr = raw_dr * exposure
        daily_rets.append(dr)
        port_rets_buffer.append(raw_dr)  # track unscaled for vol estimate
        exposure_history.append(exposure)

        # Signal generation
        should_invest = True
        if timing == "sma200":
            if spy_idx >= 200:
                sma = spy["Close"].iloc[spy_idx - 199:spy_idx + 1].mean()
                should_invest = spy["Close"].iloc[spy_idx] > sma
            else:
                should_invest = False
        elif timing == "abs_mom_12m":
            should_invest = spy["Close"].iloc[spy_idx] > spy["Close"].iloc[spy_idx - 252] if spy_idx >= 252 else False
        elif timing == "none":
            should_invest = True
        elif timing == "defensive_rotation":
            # When defensives outperform cyclicals over 21d, go to cash
            def_mom = np.nanmean([get_mom(data[e], data[e].index.get_loc(date), 21)
                                  for e in DEFENSIVE if e in data and date in data[e].index])
            cyc_mom = np.nanmean([get_mom(data[e], data[e].index.get_loc(date), 21)
                                  for e in CYCLICAL if e in data and date in data[e].index])
            should_invest = not (def_mom > cyc_mom and def_mom > 0)
        elif timing == "dispersion":
            # High dispersion = momentum works, low = doesn't
            moms = []
            for etf in CORE:
                df = data.get(etf)
                if df is not None and date in df.index:
                    m = get_mom(df, df.index.get_loc(date), 63)
                    if not np.isnan(m):
                        moms.append(m)
            if len(moms) >= 5:
                disp = np.std(moms)
                # Only invest when dispersion is above median
                # Use simple threshold (historical median is ~0.05-0.10)
                should_invest = disp > 0.05
            else:
                should_invest = False
        elif timing == "sma200_and_low_vol":
            # Only invest when SPY > SMA200 AND portfolio vol is low
            if spy_idx >= 200:
                sma = spy["Close"].iloc[spy_idx - 199:spy_idx + 1].mean()
                above_sma = spy["Close"].iloc[spy_idx] > sma
            else:
                above_sma = False
            spy_rets = spy["Close"].iloc[max(0, spy_idx - 20):spy_idx + 1].pct_change().dropna()
            spy_vol = spy_rets.std() * np.sqrt(252) if len(spy_rets) > 5 else 0.15
            low_vol = spy_vol < 0.20
            should_invest = above_sma and low_vol

        if not should_invest and in_market:
            pending = {}
        elif should_invest and not in_market:
            ranked = rank_sectors(data, date, selector)
            top = [(e, m) for e, m in ranked[:top_n] if m > 0]
            if not top and ranked:
                top = ranked[:1]
            if top:
                w = 1.0 / len(top)
                pending = {e: w for e, _ in top}
        elif should_invest and in_market:
            new_month = prev_month is None or date.month != prev_month
            if new_month and rebalance == "monthly":
                ranked = rank_sectors(data, date, selector)
                top = [(e, m) for e, m in ranked[:top_n] if m > 0]
                if top:
                    w = 1.0 / len(top)
                    new_alloc = {e: w for e, _ in top}
                    if set(new_alloc.keys()) != set(weights.keys()):
                        pending = new_alloc
                    else:
                        weights = new_alloc

        prev_month = date.month

    return compute_metrics(daily_rets)


# ============================================================
# STRATEGY: Same-day close execution (simpler)
# ============================================================

def backtest_sameday(data, start, end, top_n=3, selector="mom_63",
                     timing="sma200", slippage_bps=5):
    """
    Same-day close execution: signal AND execute at close.
    More aggressive but allowed per user specification.
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    warmup = 260

    weights = {}
    in_market = False
    prev_month = None
    daily_rets = []

    for date in dates:
        spy_idx = spy.index.get_loc(date)
        if spy_idx < warmup:
            daily_rets.append(0.0)
            continue

        dr = 0.0

        # Normal day return for current holdings
        if in_market and weights:
            for etf, w in weights.items():
                df = data[etf]
                if date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        dr += w * (df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1)
        daily_rets.append(dr)

        # Signal at close — execute immediately (same day)
        should_invest = True
        if timing == "sma200":
            sma = spy["Close"].iloc[max(0, spy_idx - 199):spy_idx + 1].mean() if spy_idx >= 200 else spy["Close"].iloc[spy_idx]
            should_invest = spy["Close"].iloc[spy_idx] > sma
        elif timing == "abs_mom_12m":
            should_invest = spy["Close"].iloc[spy_idx] > spy["Close"].iloc[spy_idx - 252] if spy_idx >= 252 else False
        elif timing == "none":
            should_invest = True

        if not should_invest and in_market:
            # Charge slippage on exit
            daily_rets[-1] -= sum(weights.values()) * slippage_bps / 10000
            weights = {}
            in_market = False
        elif should_invest and not in_market:
            ranked = rank_sectors(data, date, selector)
            top = [(e, m) for e, m in ranked[:top_n] if m > 0]
            if not top and ranked:
                top = ranked[:1]
            if top:
                w = 1.0 / len(top)
                weights = {e: w for e, _ in top}
                in_market = True
                daily_rets[-1] -= sum(weights.values()) * slippage_bps / 10000
        elif should_invest and in_market:
            new_month = prev_month is None or date.month != prev_month
            if new_month:
                ranked = rank_sectors(data, date, selector)
                top = [(e, m) for e, m in ranked[:top_n] if m > 0]
                if top:
                    w = 1.0 / len(top)
                    new_alloc = {e: w for e, _ in top}
                    if set(new_alloc.keys()) != set(weights.keys()):
                        # Charge slippage on changed portion
                        changed = set(new_alloc.keys()) ^ set(weights.keys())
                        cost = len(changed) / max(1, len(new_alloc)) * slippage_bps / 10000
                        daily_rets[-1] -= cost
                        weights = new_alloc
                    else:
                        weights = new_alloc

        prev_month = date.month

    return compute_metrics(daily_rets)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} tickers\n")

    # SPY baseline
    for pname, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END), ("FULL", "2010-01-01", TEST_END)]:
        spy_slice = data[BENCHMARK].loc[s:e, "Close"]
        r = spy_slice.pct_change().dropna()
        ex = r - 0.02 / 252
        sh = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
        cum = (1 + r).cumprod()
        t = cum.iloc[-1] - 1; n = len(r) / 252
        cg = (1 + t) ** (1 / n) - 1 if n > 0 else 0
        pk = cum.cummax(); md = ((cum - pk) / pk).min()
        print(f"  SPY {pname}: Sharpe {sh:.3f} | CAGR {cg:.1%} | MaxDD {md:.1%}")

    PERIODS = [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END), ("FULL", "2010-01-01", TEST_END)]

    print(f"\n{'='*130}")
    print(f"{'Experiment':<50} {'TRAIN Sh':>9} {'TRAIN CAGR':>11} {'TEST Sh':>8} {'TEST CAGR':>10} {'FULL Sh':>8} {'FULL CAGR':>10} {'FULL DD':>8} {'Vol':>5} {'TIM%':>5}")
    print(f"{'='*130}")

    experiments = []

    def run_and_print(name, fn, **kwargs):
        results = {"name": name}
        for pname, s, e in PERIODS:
            m = fn(data, s, e, **kwargs)
            results[pname] = m
        experiments.append(results)
        tr = results["TRAIN"]; te = results["TEST"]; fu = results["FULL"]
        print(f"  {name:<48} {tr['sharpe']:>9.3f} {tr['cagr']:>10.1%} {te['sharpe']:>8.3f} {te['cagr']:>10.1%} {fu['sharpe']:>8.3f} {fu['cagr']:>10.1%} {fu['max_dd']:>7.1%} {fu['ann_vol']:>5.1%} {fu['time_in_market']:>4.0%}")

    # === VOL-TARGETED STRATEGIES ===
    print("\n--- Vol-Targeted (next-open execution) ---")
    for tv in [0.05, 0.08, 0.10, 0.12, 0.15]:
        for sel in ["mom_63", "blended", "risk_adj_126"]:
            name = f"voltgt_{tv:.0%}_{sel}_sma200_top3"
            run_and_print(name, backtest_voltarget, top_n=3, selector=sel,
                         target_vol=tv, timing="sma200")

    print("\n--- Vol-Targeted with different timing ---")
    for tim in ["none", "abs_mom_12m", "defensive_rotation", "dispersion", "sma200_and_low_vol"]:
        name = f"voltgt_10%_blended_{tim}_top3"
        run_and_print(name, backtest_voltarget, top_n=3, selector="blended",
                     target_vol=0.10, timing=tim)

    print("\n--- Vol-Targeted top-N variations ---")
    for n in [1, 2, 3, 5]:
        name = f"voltgt_10%_blended_sma200_top{n}"
        run_and_print(name, backtest_voltarget, top_n=n, selector="blended",
                     target_vol=0.10, timing="sma200")

    # === ACCELERATION MOMENTUM ===
    print("\n--- Acceleration Momentum ---")
    for sel in ["accel_63", "accel_21"]:
        for tim in ["sma200", "none", "abs_mom_12m"]:
            name = f"voltgt_10%_{sel}_{tim}_top3"
            run_and_print(name, backtest_voltarget, top_n=3, selector=sel,
                         target_vol=0.10, timing=tim)

    # === SAME-DAY CLOSE EXECUTION ===
    print("\n--- Same-Day Close Execution ---")
    for sel in ["mom_63", "blended", "risk_adj_126", "mom_21"]:
        for tim in ["sma200", "none", "abs_mom_12m"]:
            for n in [1, 3]:
                name = f"sameday_{sel}_{tim}_top{n}"
                run_and_print(name, backtest_sameday, top_n=n, selector=sel,
                             timing=tim)

    # === SAME-DAY + VOL TARGET ===
    print("\n--- Same-Day Vol-Targeted ---")
    # Can't easily combine since sameday doesn't have vol targeting built in
    # But we can test the key combos via voltarget with different params

    # === INVERSE VOL WEIGHTING (risk parity lite) ===
    print("\n--- Inverse Volatility Selection ---")
    for tim in ["sma200", "none", "abs_mom_12m"]:
        name = f"voltgt_10%_inv_vol_{tim}_top3"
        run_and_print(name, backtest_voltarget, top_n=3, selector="inv_vol",
                     target_vol=0.10, timing=tim)

    # Print top results
    print(f"\n{'='*80}")
    print("TOP 15 by FULL Sharpe:")
    print(f"{'='*80}")
    sorted_exp = sorted(experiments, key=lambda x: x.get("FULL", {}).get("sharpe", 0), reverse=True)
    for i, r in enumerate(sorted_exp[:15]):
        fu = r["FULL"]
        te = r["TEST"]
        tr = r["TRAIN"]
        print(f"  {i+1:>2}. {r['name']:<48} FULL Sh={fu['sharpe']:.3f} CAGR={fu['cagr']:.1%} DD={fu['max_dd']:.1%} | TEST Sh={te['sharpe']:.3f} | TRAIN Sh={tr['sharpe']:.3f}")

    print(f"\nTOP 10 by TEST Sharpe (out-of-sample):")
    sorted_test = sorted(experiments, key=lambda x: x.get("TEST", {}).get("sharpe", 0), reverse=True)
    for i, r in enumerate(sorted_test[:10]):
        fu = r["FULL"]
        te = r["TEST"]
        tr = r["TRAIN"]
        print(f"  {i+1:>2}. {r['name']:<48} TEST Sh={te['sharpe']:.3f} CAGR={te['cagr']:.1%} | FULL Sh={fu['sharpe']:.3f} | TRAIN Sh={tr['sharpe']:.3f}")
