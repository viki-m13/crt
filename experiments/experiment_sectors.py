#!/usr/bin/env python3
"""
Comprehensive sector ETF strategy experiments.
Tests 30+ variants to find what works, what doesn't, and why.
"""

import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

CORE = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
ALL = CORE + ["XLRE", "XLC"]
BENCHMARK = "SPY"


def backtest(data, start, end, timing_fn, select_fn, top_n=3,
             rebalance="monthly", slippage_bps=5, exec_mode="next_open"):
    """
    Generic sector backtest engine.

    timing_fn(data, date, spy_idx) -> bool  (should we be invested?)
    select_fn(data, date) -> list of (etf, score) sorted desc
    exec_mode: "next_open" or "same_close"
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    warmup = 260

    weights = {}
    in_market = False
    prev_month = None
    pending = None
    daily_rets = []
    n_entries = 0
    n_exits = 0
    n_rotations = 0

    for date in dates:
        spy_idx = spy.index.get_loc(date)
        if spy_idx < warmup:
            daily_rets.append(0.0)
            continue

        dr = 0.0

        # === Execute pending (next-open mode only) ===
        if exec_mode == "next_open" and pending is not None:
            if len(pending) == 0 and in_market:
                for etf, w in weights.items():
                    df = data[etf]
                    if date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            op = df.iloc[si]["Open"] if "Open" in df.columns else df.iloc[si]["Close"]
                            pc = df.iloc[si - 1]["Close"]
                            dr += w * (op * (1 - slippage_bps / 10000) / pc - 1)
                weights = {}
                in_market = False
                n_exits += 1
            elif len(pending) > 0 and not in_market:
                for etf, w in pending.items():
                    df = data[etf]
                    if date in df.index:
                        si = df.index.get_loc(date)
                        op = df.iloc[si]["Open"] if "Open" in df.columns else df.iloc[si]["Close"]
                        cl = df.iloc[si]["Close"]
                        buy_p = op * (1 + slippage_bps / 10000)
                        dr += w * (cl / buy_p - 1)
                weights = dict(pending)
                in_market = True
                n_entries += 1
            elif len(pending) > 0 and in_market:
                old_w = dict(weights)
                new_w = dict(pending)
                for etf in set(list(old_w.keys()) + list(new_w.keys())):
                    df = data.get(etf)
                    if df is None or date not in df.index:
                        continue
                    si = df.index.get_loc(date)
                    if si == 0:
                        continue
                    op = df.iloc[si]["Open"] if "Open" in df.columns else df.iloc[si]["Close"]
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
                n_rotations += 1

            pending = None
            daily_rets.append(dr)
            prev_month = date.month
            continue

        # === Normal day return ===
        if in_market and weights:
            for etf, w in weights.items():
                df = data[etf]
                if date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        dr += w * (df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1)
        daily_rets.append(dr)

        # === Signal generation at close ===
        should_invest = timing_fn(data, date, spy_idx)

        if not should_invest and in_market:
            if exec_mode == "next_open":
                pending = {}
            else:
                weights = {}
                in_market = False
                n_exits += 1

        elif should_invest and not in_market:
            ranked = select_fn(data, date)
            top = [(e, m) for e, m in ranked[:top_n] if m > 0]
            if not top and ranked:
                top = ranked[:1]
            if top:
                w = 1.0 / len(top)
                alloc = {e: w for e, _ in top}
                if exec_mode == "next_open":
                    pending = alloc
                else:
                    weights = alloc
                    in_market = True
                    n_entries += 1

        elif should_invest and in_market:
            new_month = prev_month is None or date.month != prev_month
            do_rebal = new_month if rebalance == "monthly" else False
            if rebalance == "quarterly" and new_month:
                do_rebal = date.month in [1, 4, 7, 10]
            if do_rebal:
                ranked = select_fn(data, date)
                top = [(e, m) for e, m in ranked[:top_n] if m > 0]
                if top:
                    w = 1.0 / len(top)
                    new_alloc = {e: w for e, _ in top}
                    if set(new_alloc.keys()) != set(weights.keys()):
                        if exec_mode == "next_open":
                            pending = new_alloc
                        else:
                            weights = new_alloc
                            n_rotations += 1
                    else:
                        weights = new_alloc

        prev_month = date.month

    return compute_metrics(daily_rets, n_entries, n_exits, n_rotations)


def compute_metrics(daily_rets, n_entries=0, n_exits=0, n_rotations=0):
    rets = pd.Series(daily_rets)
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0,
                "n_entries": 0, "n_exits": 0, "n_rotations": 0, "time_in_market": 0}
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
    return {
        "sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
        "max_dd": round(float(mdd), 4), "sortino": round(float(sortino), 3),
        "n_entries": n_entries, "n_exits": n_exits, "n_rotations": n_rotations,
        "time_in_market": round(float(invested), 3),
    }


# ============================================================
# TIMING FUNCTIONS
# ============================================================

def _sma(data, date, spy_idx, period):
    spy = data[BENCHMARK]
    if spy_idx < period:
        return False
    sma = spy["Close"].iloc[spy_idx - period + 1:spy_idx + 1].mean()
    return spy["Close"].iloc[spy_idx] > sma

def _sma_slope(data, date, spy_idx, period, slope_days=10):
    spy = data[BENCHMARK]
    if spy_idx < period + slope_days:
        return False
    sma_now = spy["Close"].iloc[spy_idx - period + 1:spy_idx + 1].mean()
    sma_prev = spy["Close"].iloc[spy_idx - period + 1 - slope_days:spy_idx + 1 - slope_days].mean()
    above = spy["Close"].iloc[spy_idx] > sma_now
    rising = sma_now > sma_prev
    return above and rising

def _abs_momentum(data, date, spy_idx, lookback):
    spy = data[BENCHMARK]
    if spy_idx < lookback:
        return False
    return spy["Close"].iloc[spy_idx] > spy["Close"].iloc[spy_idx - lookback]

def _breadth(data, date, spy_idx, lookback, threshold_pct=0.5):
    pos = 0
    total = 0
    for etf in CORE:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < lookback:
            continue
        total += 1
        if df.iloc[idx]["Close"] > df.iloc[idx - lookback]["Close"]:
            pos += 1
    return pos > total * threshold_pct if total > 0 else False

def _sector_above_sma(data, date, spy_idx, sma_period, threshold_pct=0.5):
    """Breadth using SMA crosses instead of momentum."""
    pos = 0
    total = 0
    for etf in CORE:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < sma_period:
            continue
        total += 1
        sma = df["Close"].iloc[idx - sma_period + 1:idx + 1].mean()
        if df.iloc[idx]["Close"] > sma:
            pos += 1
    return pos > total * threshold_pct if total > 0 else False

def _relative_strength_filter(data, date, spy_idx, lookback, min_outperformers=1):
    """Only invest if at least N sectors outperform SPY."""
    spy = data[BENCHMARK]
    if spy_idx < lookback:
        return False
    spy_ret = spy["Close"].iloc[spy_idx] / spy["Close"].iloc[spy_idx - lookback] - 1
    outperformers = 0
    for etf in ALL:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < lookback:
            continue
        ret = df.iloc[idx]["Close"] / df.iloc[idx - lookback]["Close"] - 1
        if ret > spy_ret:
            outperformers += 1
    return outperformers >= min_outperformers

# Pre-built timing functions
TIMING = {
    "none": lambda d, dt, i: True,
    "sma50": lambda d, dt, i: _sma(d, dt, i, 50),
    "sma100": lambda d, dt, i: _sma(d, dt, i, 100),
    "sma200": lambda d, dt, i: _sma(d, dt, i, 200),
    "sma200_slope": lambda d, dt, i: _sma_slope(d, dt, i, 200),
    "abs_mom_6m": lambda d, dt, i: _abs_momentum(d, dt, i, 126),
    "abs_mom_10m": lambda d, dt, i: _abs_momentum(d, dt, i, 210),
    "abs_mom_12m": lambda d, dt, i: _abs_momentum(d, dt, i, 252),
    "breadth_63": lambda d, dt, i: _breadth(d, dt, i, 63),
    "breadth_126": lambda d, dt, i: _breadth(d, dt, i, 126),
    "breadth_252": lambda d, dt, i: _breadth(d, dt, i, 252),
    "breadth_63_67pct": lambda d, dt, i: _breadth(d, dt, i, 63, 0.67),
    "breadth_sma200": lambda d, dt, i: _sector_above_sma(d, dt, i, 200),
    "breadth_sma200_67pct": lambda d, dt, i: _sector_above_sma(d, dt, i, 200, 0.67),
    "dual_sma200_absmom": lambda d, dt, i: _sma(d, dt, i, 200) and _abs_momentum(d, dt, i, 210),
    "dual_sma200_breadth": lambda d, dt, i: _sma(d, dt, i, 200) and _breadth(d, dt, i, 63),
    "triple_confirm": lambda d, dt, i: _sma(d, dt, i, 200) and _sma_slope(d, dt, i, 200) and _breadth(d, dt, i, 63),
    "rel_strength_3": lambda d, dt, i: _relative_strength_filter(d, dt, i, 63, 3),
    "rel_strength_5": lambda d, dt, i: _relative_strength_filter(d, dt, i, 63, 5),
}


# ============================================================
# SELECTION FUNCTIONS
# ============================================================

def _mom_selector(data, date, lookback):
    scores = []
    for etf in ALL:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < lookback:
            continue
        ret = df.iloc[idx]["Close"] / df.iloc[idx - lookback]["Close"] - 1
        scores.append((etf, ret))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _blended_mom_selector(data, date, windows=[21, 63, 126], blend=[0.25, 0.5, 0.25]):
    scores = []
    for etf in ALL:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        total = 0
        total_w = 0
        for w, b in zip(windows, blend):
            if idx >= w:
                ret = df.iloc[idx]["Close"] / df.iloc[idx - w]["Close"] - 1
                total += ret * b
                total_w += b
        if total_w > 0:
            scores.append((etf, total / total_w))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _risk_adj_selector(data, date, lookback=63):
    scores = []
    for etf in ALL:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < lookback:
            continue
        ret = df.iloc[idx]["Close"] / df.iloc[idx - lookback]["Close"] - 1
        rets = df["Close"].iloc[max(0, idx - lookback):idx + 1].pct_change().dropna()
        vol = rets.std() * np.sqrt(252) if len(rets) > 5 else 0.15
        scores.append((etf, ret / max(vol, 0.01)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _skip_month_selector(data, date, total=252, skip=21):
    """Classic 12-1 month momentum: skip most recent month."""
    scores = []
    for etf in ALL:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < total:
            continue
        ret = df.iloc[idx - skip]["Close"] / df.iloc[idx - total]["Close"] - 1
        scores.append((etf, ret))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _relative_to_spy(data, date, lookback=63):
    """Rank by return relative to SPY."""
    spy = data.get(BENCHMARK)
    if spy is None or date not in spy.index:
        return []
    spy_idx = spy.index.get_loc(date)
    if spy_idx < lookback:
        return []
    spy_ret = spy.iloc[spy_idx]["Close"] / spy.iloc[spy_idx - lookback]["Close"] - 1
    scores = []
    for etf in ALL:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < lookback:
            continue
        ret = df.iloc[idx]["Close"] / df.iloc[idx - lookback]["Close"] - 1
        scores.append((etf, ret - spy_ret))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _inverse_vol_selector(data, date, lookback=63):
    """Equal-weight but rank by inverse volatility (risk parity lite)."""
    scores = []
    for etf in ALL:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < lookback:
            continue
        rets = df["Close"].iloc[max(0, idx - lookback):idx + 1].pct_change().dropna()
        vol = rets.std() * np.sqrt(252) if len(rets) > 5 else 0.15
        scores.append((etf, 1.0 / max(vol, 0.01)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _anti_momentum(data, date, lookback=63):
    """Mean reversion: buy the WORST performers."""
    scores = []
    for etf in ALL:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < lookback:
            continue
        ret = df.iloc[idx]["Close"] / df.iloc[idx - lookback]["Close"] - 1
        scores.append((etf, -ret))  # Negative = buy losers
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

SELECTORS = {
    "mom_21": lambda d, dt: _mom_selector(d, dt, 21),
    "mom_63": lambda d, dt: _mom_selector(d, dt, 63),
    "mom_126": lambda d, dt: _mom_selector(d, dt, 126),
    "mom_252": lambda d, dt: _mom_selector(d, dt, 252),
    "blended_21_63_126": lambda d, dt: _blended_mom_selector(d, dt),
    "risk_adj_63": lambda d, dt: _risk_adj_selector(d, dt, 63),
    "risk_adj_126": lambda d, dt: _risk_adj_selector(d, dt, 126),
    "skip_month_252": lambda d, dt: _skip_month_selector(d, dt, 252, 21),
    "skip_month_126": lambda d, dt: _skip_month_selector(d, dt, 126, 21),
    "relative_spy_63": lambda d, dt: _relative_to_spy(d, dt, 63),
    "relative_spy_126": lambda d, dt: _relative_to_spy(d, dt, 126),
    "inv_vol_63": lambda d, dt: _inverse_vol_selector(d, dt, 63),
    "anti_mom_63": lambda d, dt: _anti_momentum(d, dt, 63),
    "anti_mom_126": lambda d, dt: _anti_momentum(d, dt, 126),
}


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(data, name, timing_key, selector_key, top_n=3,
                   rebalance="monthly", exec_mode="next_open",
                   periods=None):
    """Run one experiment across all periods."""
    if periods is None:
        periods = [
            ("TRAIN", TRAIN_START, TRAIN_END),
            ("TEST", TEST_START, TEST_END),
            ("FULL", "2010-01-01", TEST_END),
        ]

    timing_fn = TIMING[timing_key]
    select_fn = SELECTORS[selector_key]

    results = {"name": name, "timing": timing_key, "selector": selector_key,
               "top_n": top_n, "rebalance": rebalance, "exec": exec_mode}

    for pname, s, e in periods:
        m = backtest(data, s, e, timing_fn, select_fn, top_n=top_n,
                     rebalance=rebalance, exec_mode=exec_mode)
        results[pname] = m

    return results


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded\n")

    # SPY baseline
    spy = data[BENCHMARK]
    for pname, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("TEST", TEST_START, TEST_END), ("FULL", "2010-01-01", TEST_END)]:
        spy_slice = spy.loc[s:e, "Close"]
        r = spy_slice.pct_change().dropna()
        ex = r - 0.02 / 252
        sh = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
        cum = (1 + r).cumprod()
        t = cum.iloc[-1] - 1
        n = len(r) / 252
        cg = (1 + t) ** (1 / n) - 1 if n > 0 else 0
        pk = cum.cummax()
        md = ((cum - pk) / pk).min()
        print(f"  SPY B&H {pname}: Sharpe {sh:.3f} | CAGR {cg:.1%} | MaxDD {md:.1%}")

    print(f"\n{'='*120}")
    print(f"{'Experiment':<45} {'TRAIN Sharpe':>12} {'TRAIN CAGR':>11} {'TEST Sharpe':>12} {'TEST CAGR':>10} {'FULL Sharpe':>12} {'FULL CAGR':>10} {'FULL DD':>8} {'TIM%':>5}")
    print(f"{'='*120}")

    experiments = []

    # Import experiment list
    EXPERIMENT_LIST = [
        # === BATCH 1: Timing signals (with top-3 momentum as baseline selector) ===
        ("no_timing_top3_mom63", "none", "mom_63", 3),
        ("sma50_top3_mom63", "sma50", "mom_63", 3),
        ("sma100_top3_mom63", "sma100", "mom_63", 3),
        ("sma200_top3_mom63", "sma200", "mom_63", 3),
        ("sma200slope_top3_mom63", "sma200_slope", "mom_63", 3),
        ("absmom6m_top3_mom63", "abs_mom_6m", "mom_63", 3),
        ("absmom10m_top3_mom63", "abs_mom_10m", "mom_63", 3),
        ("absmom12m_top3_mom63", "abs_mom_12m", "mom_63", 3),
        ("breadth63_top3_mom63", "breadth_63", "mom_63", 3),
        ("breadth126_top3_mom63", "breadth_126", "mom_63", 3),
        ("breadth252_top3_mom63", "breadth_252", "mom_63", 3),
        ("breadth63_67pct_top3_mom63", "breadth_63_67pct", "mom_63", 3),
        ("breadthSMA200_top3_mom63", "breadth_sma200", "mom_63", 3),
        ("breadthSMA200_67_top3_mom63", "breadth_sma200_67pct", "mom_63", 3),
        ("dual_sma200_absmom", "dual_sma200_absmom", "mom_63", 3),
        ("dual_sma200_breadth", "dual_sma200_breadth", "mom_63", 3),
        ("triple_confirm", "triple_confirm", "mom_63", 3),
        ("relstr3_top3_mom63", "rel_strength_3", "mom_63", 3),
        ("relstr5_top3_mom63", "rel_strength_5", "mom_63", 3),

        # === BATCH 2: Sector selection (with sma200 as good timing baseline) ===
        ("sma200_top1_mom63", "sma200", "mom_63", 1),
        ("sma200_top2_mom63", "sma200", "mom_63", 2),
        ("sma200_top5_mom63", "sma200", "mom_63", 5),
        ("sma200_allcore_mom63", "sma200", "mom_63", 9),
        ("sma200_top3_mom21", "sma200", "mom_21", 3),
        ("sma200_top3_mom126", "sma200", "mom_126", 3),
        ("sma200_top3_mom252", "sma200", "mom_252", 3),
        ("sma200_top3_blended", "sma200", "blended_21_63_126", 3),
        ("sma200_top3_riskadj63", "sma200", "risk_adj_63", 3),
        ("sma200_top3_riskadj126", "sma200", "risk_adj_126", 3),
        ("sma200_top3_skipmonth252", "sma200", "skip_month_252", 3),
        ("sma200_top3_skipmonth126", "sma200", "skip_month_126", 3),
        ("sma200_top3_relspy63", "sma200", "relative_spy_63", 3),
        ("sma200_top3_relspy126", "sma200", "relative_spy_126", 3),
        ("sma200_top3_invvol63", "sma200", "inv_vol_63", 3),

        # === BATCH 3: Novel/non-traditional ideas ===
        ("sma200_top3_antimom63", "sma200", "anti_mom_63", 3),
        ("sma200_top3_antimom126", "sma200", "anti_mom_126", 3),
        ("sma200_top1_blended", "sma200", "blended_21_63_126", 1),
        ("sma200_top2_blended", "sma200", "blended_21_63_126", 2),
        ("sma200slope_top3_blended", "sma200_slope", "blended_21_63_126", 3),
        ("sma200slope_top1_blended", "sma200_slope", "blended_21_63_126", 1),
        ("absmom10m_top3_blended", "abs_mom_10m", "blended_21_63_126", 3),
        ("absmom10m_top1_blended", "abs_mom_10m", "blended_21_63_126", 1),
        ("triple_top1_blended", "triple_confirm", "blended_21_63_126", 1),
        ("triple_top3_blended", "triple_confirm", "blended_21_63_126", 3),
        # Quarterly rebalance variants
        ("sma200_top3_mom63_quarterly", "sma200", "mom_63", 3),
    ]

    for item in EXPERIMENT_LIST:
        name, timing, selector, top_n = item[:4]
        rebal = "quarterly" if name.endswith("_quarterly") else "monthly"
        r = run_experiment(data, name, timing, selector, top_n=top_n, rebalance=rebal)
        experiments.append(r)

        tr = r.get("TRAIN", {})
        te = r.get("TEST", {})
        fu = r.get("FULL", {})
        print(f"  {name:<43} {tr.get('sharpe',0):>12.3f} {tr.get('cagr',0):>10.1%} {te.get('sharpe',0):>12.3f} {te.get('cagr',0):>10.1%} {fu.get('sharpe',0):>12.3f} {fu.get('cagr',0):>10.1%} {fu.get('max_dd',0):>7.1%} {fu.get('time_in_market',0):>5.0%}")

    # Sort by FULL Sharpe
    print(f"\n{'='*80}")
    print("TOP 10 by FULL Sharpe:")
    print(f"{'='*80}")
    sorted_exp = sorted(experiments, key=lambda x: x.get("FULL", {}).get("sharpe", 0), reverse=True)
    for i, r in enumerate(sorted_exp[:10]):
        fu = r["FULL"]
        te = r["TEST"]
        print(f"  {i+1}. {r['name']:<40} FULL: Sh={fu['sharpe']:.3f} CAGR={fu['cagr']:.1%} DD={fu['max_dd']:.1%} | TEST: Sh={te.get('sharpe',0):.3f} CAGR={te.get('cagr',0):.1%}")

    # Save results
    with open(os.path.join(os.path.dirname(__file__), "results", "sector_experiments.json"), "w") as f:
        json.dump(experiments, f, indent=2, default=str)
    print(f"\nResults saved to results/sector_experiments.json")
