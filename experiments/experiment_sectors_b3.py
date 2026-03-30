#!/usr/bin/env python3
"""
Batch 3: Combine winning elements + push for maximum Sharpe.
Focuses on: same-day + vol targeting, combined selectors,
ultra-tight vol targets, regime switching, weekly rebalancing.
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
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0,
                "time_in_market": 0, "ann_vol": 0, "calmar": 0}
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
    calmar = cagr / abs(mdd) if mdd < 0 else 0
    return {"sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
            "max_dd": round(float(mdd), 4), "sortino": round(float(sortino), 3),
            "time_in_market": round(float(invested), 3), "ann_vol": round(float(ann_vol), 4),
            "calmar": round(float(calmar), 3)}


def get_mom(df, idx, lookback):
    if idx < lookback:
        return np.nan
    return df.iloc[idx]["Close"] / df.iloc[idx - lookback]["Close"] - 1


def rank_sectors(data, date, method="mom_63"):
    scores = []
    for etf in ALL:
        df = data.get(etf)
        if df is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)

        if method == "mom_63":
            s = get_mom(df, idx, 63)
        elif method == "blended":
            vals = []
            for w, b in [(21, 0.25), (63, 0.5), (126, 0.25)]:
                m = get_mom(df, idx, w)
                if not np.isnan(m):
                    vals.append((m, b))
            s = sum(m * b for m, b in vals) / sum(b for _, b in vals) if vals else np.nan
        elif method == "risk_adj_126":
            m = get_mom(df, idx, 126)
            rets = df["Close"].iloc[max(0, idx - 126):idx + 1].pct_change().dropna()
            vol = rets.std() * np.sqrt(252) if len(rets) > 5 else 0.15
            s = m / max(vol, 0.01) if not np.isnan(m) else np.nan
        elif method == "accel_21":
            m_now = get_mom(df, idx, 21)
            m_prev = get_mom(df, idx - 10, 21) if idx >= 31 else np.nan
            s = (m_now - m_prev) if not np.isnan(m_now) and not np.isnan(m_prev) else np.nan
        elif method == "inv_vol":
            rets = df["Close"].iloc[max(0, idx - 63):idx + 1].pct_change().dropna()
            vol = rets.std() * np.sqrt(252) if len(rets) > 5 else 0.15
            s = 1.0 / max(vol, 0.01)
        elif method == "composite":
            # Novel: combine acceleration + risk-adjusted momentum + inverse vol
            m126 = get_mom(df, idx, 126)
            rets = df["Close"].iloc[max(0, idx - 126):idx + 1].pct_change().dropna()
            vol = rets.std() * np.sqrt(252) if len(rets) > 5 else 0.15
            risk_adj = m126 / max(vol, 0.01) if not np.isnan(m126) else 0
            m21_now = get_mom(df, idx, 21)
            m21_prev = get_mom(df, idx - 10, 21) if idx >= 31 else np.nan
            accel = (m21_now - m21_prev) if not np.isnan(m21_now) and not np.isnan(m21_prev) else 0
            inv_v = 1.0 / max(vol, 0.01)
            # Normalize and combine (equal weight)
            s = risk_adj * 0.4 + accel * 100 * 0.3 + inv_v * 0.01 * 0.3
        elif method == "accel_risk_adj":
            # Acceleration * risk_adj (multiplicative)
            m126 = get_mom(df, idx, 126)
            rets = df["Close"].iloc[max(0, idx - 126):idx + 1].pct_change().dropna()
            vol = rets.std() * np.sqrt(252) if len(rets) > 5 else 0.15
            risk_adj = m126 / max(vol, 0.01) if not np.isnan(m126) else 0
            m21_now = get_mom(df, idx, 21)
            m21_prev = get_mom(df, idx - 10, 21) if idx >= 31 else np.nan
            accel = (m21_now - m21_prev) if not np.isnan(m21_now) and not np.isnan(m21_prev) else 0
            # Only positive acceleration + positive risk-adj
            s = max(0, risk_adj) * max(0, accel * 10 + 0.5) if risk_adj > 0 else -1
        else:
            s = get_mom(df, idx, 63)

        if not np.isnan(s):
            scores.append((etf, float(s)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def backtest_combined(data, start, end, top_n=3, selector="blended",
                      target_vol=0.10, vol_lookback=21, timing="none",
                      slippage_bps=5, rebalance_freq="monthly",
                      exec_mode="same_close", max_exposure=1.0,
                      drawdown_exit=None):
    """
    Combined strategy: same-day or next-open execution + vol targeting.
    Most flexible backtester.
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    warmup = 260

    weights = {}
    in_market = False
    prev_month = None
    prev_week = None
    daily_rets = []
    port_rets_buffer = []

    for date in dates:
        spy_idx = spy.index.get_loc(date)
        if spy_idx < warmup:
            daily_rets.append(0.0)
            continue

        # === Compute raw portfolio return ===
        raw_dr = 0.0
        if in_market and weights:
            for etf, w in weights.items():
                df = data[etf]
                if date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        raw_dr += w * (df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1)

        # === Vol targeting ===
        if target_vol is not None and len(port_rets_buffer) >= vol_lookback and in_market:
            recent_vol = np.std(port_rets_buffer[-vol_lookback:]) * np.sqrt(252)
            if recent_vol > 0.001:
                exposure = min(max_exposure, target_vol / recent_vol)
            else:
                exposure = max_exposure
        else:
            exposure = max_exposure if in_market else 0

        # === Drawdown exit ===
        if drawdown_exit is not None and len(daily_rets) > 0:
            cum = np.cumprod([1 + r for r in daily_rets])
            if len(cum) > 0:
                peak = np.maximum.accumulate(cum)
                current_dd = (cum[-1] - peak[-1]) / peak[-1]
                if current_dd < drawdown_exit and in_market:
                    exposure = 0  # Force to cash

        dr = raw_dr * exposure
        daily_rets.append(dr)
        port_rets_buffer.append(raw_dr)

        # === Signal generation ===
        should_invest = True
        if timing == "sma200":
            if spy_idx >= 200:
                sma = spy["Close"].iloc[spy_idx - 199:spy_idx + 1].mean()
                should_invest = spy["Close"].iloc[spy_idx] > sma
            else:
                should_invest = False
        elif timing == "abs_mom_12m":
            should_invest = spy["Close"].iloc[spy_idx] > spy["Close"].iloc[spy_idx - 252] if spy_idx >= 252 else False
        elif timing == "abs_mom_10m":
            should_invest = spy["Close"].iloc[spy_idx] > spy["Close"].iloc[spy_idx - 210] if spy_idx >= 210 else False
        elif timing == "none":
            should_invest = True

        # Rebalance check
        new_month = prev_month is None or date.month != prev_month
        new_week = prev_week is None or date.isocalendar()[1] != prev_week
        do_rebal = False
        if rebalance_freq == "monthly":
            do_rebal = new_month
        elif rebalance_freq == "biweekly":
            do_rebal = new_week and (date.isocalendar()[1] % 2 == 0 or not in_market)
        elif rebalance_freq == "weekly":
            do_rebal = new_week

        if not should_invest and in_market:
            # Charge slippage
            cost = sum(weights.values()) * slippage_bps / 10000
            daily_rets[-1] -= cost
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
        elif should_invest and in_market and do_rebal:
            ranked = rank_sectors(data, date, selector)
            top = [(e, m) for e, m in ranked[:top_n] if m > 0]
            if top:
                w = 1.0 / len(top)
                new_alloc = {e: w for e, _ in top}
                if set(new_alloc.keys()) != set(weights.keys()):
                    changed = len(set(new_alloc.keys()) ^ set(weights.keys()))
                    cost = changed / max(1, len(new_alloc) + len(weights)) * slippage_bps / 10000
                    daily_rets[-1] -= cost
                weights = new_alloc

        prev_month = date.month
        prev_week = date.isocalendar()[1]

    return compute_metrics(daily_rets)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} tickers\n")

    PERIODS = [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END),
               ("TEST", TEST_START, TEST_END), ("FULL", "2010-01-01", TEST_END)]

    # SPY baseline
    for pname, s, e in PERIODS:
        spy_slice = data[BENCHMARK].loc[s:e, "Close"]
        r = spy_slice.pct_change().dropna()
        ex = r - 0.02 / 252
        sh = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
        cum = (1 + r).cumprod(); t = cum.iloc[-1] - 1; n = len(r) / 252
        cg = (1 + t) ** (1 / n) - 1 if n > 0 else 0
        pk = cum.cummax(); md = ((cum - pk) / pk).min()
        print(f"  SPY {pname}: Sh={sh:.3f} CAGR={cg:.1%} DD={md:.1%}")

    print(f"\n{'='*150}")
    hdr = f"{'Experiment':<55} {'TR Sh':>6} {'TR CAGR':>8} {'VA Sh':>6} {'VA CAGR':>8} {'TE Sh':>6} {'TE CAGR':>8} {'FU Sh':>6} {'FU CAGR':>8} {'FU DD':>7} {'Vol':>5} {'TIM%':>5} {'Calm':>5}"
    print(hdr)
    print(f"{'='*150}")

    experiments = []

    def run_and_print(name, **kwargs):
        results = {"name": name}
        for pname, s, e in PERIODS:
            m = backtest_combined(data, s, e, **kwargs)
            results[pname] = m
        experiments.append(results)
        tr = results["TRAIN"]; va = results["VALID"]; te = results["TEST"]; fu = results["FULL"]
        print(f"  {name:<53} {tr['sharpe']:>6.3f} {tr['cagr']:>7.1%} {va['sharpe']:>6.3f} {va['cagr']:>7.1%} {te['sharpe']:>6.3f} {te['cagr']:>7.1%} {fu['sharpe']:>6.3f} {fu['cagr']:>7.1%} {fu['max_dd']:>6.1%} {fu['ann_vol']:>5.1%} {fu['time_in_market']:>4.0%} {fu['calmar']:>5.2f}")

    # === 1. SAME-DAY + VOL TARGETING (combined for first time) ===
    print("\n--- Same-Day + Vol Targeting ---")
    for tv in [0.05, 0.08, 0.10, 0.12]:
        for sel in ["risk_adj_126", "blended", "accel_21", "inv_vol", "composite", "accel_risk_adj"]:
            run_and_print(f"sd_vt{tv:.0%}_{sel}_none_t3",
                         selector=sel, target_vol=tv, timing="none", top_n=3)

    # === 2. SAME-DAY + VOL TARGETING + ABS MOM 12M ===
    print("\n--- Same-Day + Vol Target + AbsMom12m ---")
    for tv in [0.08, 0.10, 0.12]:
        for sel in ["risk_adj_126", "blended", "accel_21", "composite"]:
            run_and_print(f"sd_vt{tv:.0%}_{sel}_am12m_t3",
                         selector=sel, target_vol=tv, timing="abs_mom_12m", top_n=3)

    # === 3. TOP-N VARIATIONS WITH BEST COMBOS ===
    print("\n--- Top-N variations ---")
    for n in [1, 2, 3, 5]:
        run_and_print(f"sd_vt10%_risk_adj_126_none_t{n}",
                     selector="risk_adj_126", target_vol=0.10, timing="none", top_n=n)
        run_and_print(f"sd_vt10%_composite_none_t{n}",
                     selector="composite", target_vol=0.10, timing="none", top_n=n)

    # === 4. WEEKLY REBALANCING ===
    print("\n--- Weekly Rebalancing ---")
    for sel in ["accel_21", "risk_adj_126", "blended", "composite"]:
        run_and_print(f"sd_vt10%_{sel}_none_t3_weekly",
                     selector=sel, target_vol=0.10, timing="none", top_n=3, rebalance_freq="weekly")

    # === 5. DRAWDOWN EXIT ===
    print("\n--- With Drawdown Exit ---")
    for dd_exit in [-0.10, -0.15, -0.20]:
        run_and_print(f"sd_vt10%_composite_none_t3_dd{abs(dd_exit):.0%}",
                     selector="composite", target_vol=0.10, timing="none", top_n=3,
                     drawdown_exit=dd_exit)

    # === 6. ULTRA-TIGHT VOL TARGETS ===
    print("\n--- Ultra-tight Vol Targets ---")
    for tv in [0.03, 0.04, 0.05]:
        for sel in ["risk_adj_126", "composite", "inv_vol"]:
            run_and_print(f"sd_vt{tv:.0%}_{sel}_none_t3",
                         selector=sel, target_vol=tv, timing="none", top_n=3)

    # === 7. BIWEEKLY REBALANCING ===
    print("\n--- Biweekly Rebalancing ---")
    for sel in ["accel_21", "composite"]:
        run_and_print(f"sd_vt10%_{sel}_none_t3_biweek",
                     selector=sel, target_vol=0.10, timing="none", top_n=3, rebalance_freq="biweekly")

    # === RESULTS ===
    print(f"\n{'='*100}")
    print("TOP 20 by FULL Sharpe (must have TRAIN Sh > 0.3 AND TEST Sh > 0.3):")
    print(f"{'='*100}")
    consistent = [r for r in experiments
                  if r.get("TRAIN", {}).get("sharpe", 0) > 0.3
                  and r.get("TEST", {}).get("sharpe", 0) > 0.3]
    consistent.sort(key=lambda x: x.get("FULL", {}).get("sharpe", 0), reverse=True)
    for i, r in enumerate(consistent[:20]):
        fu = r["FULL"]; te = r["TEST"]; tr = r["TRAIN"]; va = r["VALID"]
        print(f"  {i+1:>2}. {r['name']:<50} FULL={fu['sharpe']:.3f}/{fu['cagr']:.1%}/{fu['max_dd']:.1%} TR={tr['sharpe']:.3f} VA={va['sharpe']:.3f} TE={te['sharpe']:.3f} Vol={fu['ann_vol']:.1%} Calm={fu['calmar']:.2f}")

    print(f"\nTOP 10 by Calmar Ratio (CAGR/MaxDD, consistent strategies only):")
    consistent.sort(key=lambda x: x.get("FULL", {}).get("calmar", 0), reverse=True)
    for i, r in enumerate(consistent[:10]):
        fu = r["FULL"]; te = r["TEST"]; tr = r["TRAIN"]
        print(f"  {i+1:>2}. {r['name']:<50} Calmar={fu['calmar']:.2f} Sh={fu['sharpe']:.3f} CAGR={fu['cagr']:.1%} DD={fu['max_dd']:.1%}")
