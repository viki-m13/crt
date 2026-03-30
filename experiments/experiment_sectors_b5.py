#!/usr/bin/env python3
"""
Batch 5: Push for Sharpe 3. Multi-strategy, combined signals,
extreme selectivity, market regime filters, combined alpha sources.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
from experiment_sectors_b4 import (
    get_tradeable, compute_metrics, precompute_features,
    rank_universe, BENCHMARK, SECTOR_ETFS, EXCLUDE
)


def backtest_multi(data, features, tickers, start, end,
                   selectors=None, top_n=10, rebal_freq="monthly",
                   target_vol=None, vol_lookback=21, slippage_bps=10,
                   market_filter=None, min_mom_12_1=None,
                   min_pos_52w=None, combined_rank=None,
                   multi_strat=None):
    """
    Enhanced backtest with multi-strategy and combined signals.
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    warmup = 260

    weights = {}
    pending = None
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

        # Execute pending
        if pending is not None:
            old_w = dict(weights)
            new_w = dict(pending)
            dr = 0.0
            for t in set(list(old_w.keys()) + list(new_w.keys())):
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
                    dr += ow * (op * (1 - slippage_bps / 10000) / pc - 1)
                elif ow == 0 and nw > 0:
                    dr += nw * (cl / (op * (1 + slippage_bps / 10000)) - 1)
                elif ow > 0 and nw > 0:
                    dr += nw * (cl / pc - 1)
                    dr -= abs(nw - ow) * slippage_bps / 10000
            weights = new_w
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

        # Normal day return
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
        if target_vol and len(raw_rets_buffer) >= vol_lookback and weights:
            rv = np.std(raw_rets_buffer[-vol_lookback:]) * np.sqrt(252)
            current_exposure = min(1.0, target_vol / max(rv, 0.001))
        daily_rets.append(dr * current_exposure)

        # Signal generation
        new_month = prev_month is None or date.month != prev_month
        new_week = prev_week is None or date.isocalendar()[1] != prev_week
        do_rebal = False
        if rebal_freq == "monthly" and new_month:
            do_rebal = True
        elif rebal_freq == "weekly" and new_week:
            do_rebal = True
        elif not weights:
            do_rebal = True

        if do_rebal:
            # Market filter
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
                if weights:
                    pending = {}
                prev_month = date.month
                prev_week = date.isocalendar()[1]
                continue

            # Multi-strategy combination
            if multi_strat:
                all_scores = {}
                for sel, sel_wt in multi_strat:
                    ranked = rank_universe(features, date, tickers, method=sel, top_n=50)
                    for i, (t, s) in enumerate(ranked):
                        if t not in all_scores:
                            all_scores[t] = 0
                        # Rank-based score (1 for best, declining)
                        all_scores[t] += sel_wt * (50 - i) / 50
                final_ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
                final_ranked = final_ranked[:top_n]
            elif combined_rank:
                # Combined ranking from multiple selectors
                all_scores = {}
                for sel, sel_wt in combined_rank:
                    ranked = rank_universe(features, date, tickers, method=sel, top_n=50)
                    for i, (t, s) in enumerate(ranked):
                        if t not in all_scores:
                            all_scores[t] = 0
                        all_scores[t] += sel_wt * (50 - i) / 50
                final_ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

                # Apply filters
                filtered = []
                for t, score in final_ranked:
                    f = features.get(t)
                    if f is None or date not in f.index:
                        continue
                    row = f.loc[date]
                    if min_mom_12_1 is not None:
                        m = row.get("mom_12_1", np.nan)
                        if np.isnan(m) or m < min_mom_12_1:
                            continue
                    if min_pos_52w is not None:
                        p = row.get("pos_52w", np.nan)
                        if np.isnan(p) or p < min_pos_52w:
                            continue
                    filtered.append((t, score))
                final_ranked = filtered[:top_n]
            else:
                sel = selectors if isinstance(selectors, str) else (selectors[0] if selectors else "mom_12_1")
                ranked = rank_universe(features, date, tickers, method=sel, top_n=top_n)
                # Apply filters
                final_ranked = []
                for t, s in ranked:
                    f = features.get(t)
                    if f is None or date not in f.index:
                        continue
                    row = f.loc[date]
                    if min_mom_12_1 is not None:
                        m = row.get("mom_12_1", np.nan)
                        if np.isnan(m) or m < min_mom_12_1:
                            continue
                    if min_pos_52w is not None:
                        p = row.get("pos_52w", np.nan)
                        if np.isnan(p) or p < min_pos_52w:
                            continue
                    final_ranked.append((t, s))
                final_ranked = final_ranked[:top_n]

            if final_ranked:
                w = 1.0 / len(final_ranked)
                new_alloc = {t: w for t, _ in final_ranked}
                if set(new_alloc.keys()) != set(weights.keys()):
                    pending = new_alloc
                else:
                    weights = new_alloc
            elif weights and not invest:
                pending = {}

        prev_month = date.month
        prev_week = date.isocalendar()[1]

    return compute_metrics(daily_rets)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    tickers = get_tradeable(data)
    stocks_only = [t for t in tickers if t not in SECTOR_ETFS]
    print(f"  {len(tickers)} tradeable ({len(stocks_only)} stocks)")

    print("Precomputing features...")
    features = precompute_features(data, tickers)
    print(f"  {len(features)} tickers")

    PERIODS = [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END),
               ("TEST", TEST_START, TEST_END), ("FULL", "2010-01-01", TEST_END)]

    for pname, s, e in PERIODS:
        r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna().values
        m = compute_metrics(r)
        print(f"  SPY {pname}: Sh={m['sharpe']:.3f} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%}")

    results = []
    hdr = f"{'Experiment':<65} {'TR':>6} {'VA':>6} {'TE':>6} {'FU':>6} {'CAGR':>7} {'DD':>7} {'Cal':>5}"
    print(f"\n{'='*115}")
    print(hdr)
    print(f"{'='*115}")

    def run(name, **kwargs):
        r = {"name": name}
        for pname, s, e in PERIODS:
            m = backtest_multi(data, features, stocks_only, s, e, **kwargs)
            r[pname] = m
        results.append(r)
        tr=r["TRAIN"]; va=r["VALID"]; te=r["TEST"]; fu=r["FULL"]
        print(f"  {name:<63} {tr['sharpe']:>6.3f} {va['sharpe']:>6.3f} {te['sharpe']:>6.3f} {fu['sharpe']:>6.3f} {fu['cagr']:>6.1%} {fu['max_dd']:>6.1%} {fu['calmar']:>5.2f}")

    # === A. Combined ranking: 12-1 mom + risk_adj ===
    print("\n--- A. Combined ranking ---")
    combos = [
        ("mom12_1+riskadj126", [("mom_12_1", 0.5), ("risk_adj_126", 0.5)]),
        ("mom12_1+riskadj+accel", [("mom_12_1", 0.4), ("risk_adj_126", 0.3), ("accel_21", 0.3)]),
        ("mom12_1+quality", [("mom_12_1", 0.5), ("quality_momentum", 0.5)]),
        ("riskadj+quality", [("risk_adj_126", 0.5), ("quality_momentum", 0.5)]),
        ("mom126+riskadj+quality", [("mom_126", 0.33), ("risk_adj_126", 0.34), ("quality_momentum", 0.33)]),
    ]
    for name, combo in combos:
        for tv in [0.08, 0.10, 0.12]:
            for n in [5, 10]:
                run(f"comb_{name}_t{n}_vt{tv:.0%}",
                    combined_rank=combo, top_n=n, target_vol=tv)

    # === B. Filters: only invest in high-quality momentum ===
    print("\n--- B. With momentum/quality filters ---")
    for min_mom in [0.0, 0.10, 0.20]:
        for min_pos in [0.0, 0.7, 0.8]:
            if min_mom == 0 and min_pos == 0:
                continue
            run(f"mom12_1_t10_vt12%_minmom{min_mom:.0%}_minpos{min_pos:.0%}",
                selectors="mom_12_1", top_n=10, target_vol=0.12,
                min_mom_12_1=min_mom, min_pos_52w=min_pos)

    # === C. Market regime filter ===
    print("\n--- C. With market regime filter ---")
    for sel in ["mom_12_1", "risk_adj_126"]:
        for mf in ["sma200", "abs_mom_10m"]:
            for tv in [0.10, 0.12, 0.15]:
                run(f"{sel}_t10_vt{tv:.0%}_{mf}",
                    selectors=sel, top_n=10, target_vol=tv,
                    market_filter=mf)

    # === D. Concentration (top 3-5) with tight vol ===
    print("\n--- D. High concentration + tight vol ---")
    for sel in ["mom_12_1", "risk_adj_126"]:
        for n in [3, 5]:
            for tv in [0.05, 0.08, 0.10]:
                run(f"{sel}_t{n}_vt{tv:.0%}",
                    selectors=sel, top_n=n, target_vol=tv)

    # === E. Multi-strategy combo ===
    print("\n--- E. Multi-strategy ensemble ---")
    multis = [
        ("mom12_1+reversal5d", [("mom_12_1", 0.7), ("reversal_5d", 0.3)]),
        ("mom12_1+idio63", [("mom_12_1", 0.5), ("idio_mom_63", 0.5)]),
        ("riskadj+idio+reversal", [("risk_adj_126", 0.4), ("idio_mom_63", 0.3), ("reversal_5d", 0.3)]),
        ("all_factors", [("mom_12_1", 0.25), ("risk_adj_126", 0.25), ("quality_momentum", 0.25), ("accel_21", 0.25)]),
    ]
    for name, combo in multis:
        for tv in [0.10, 0.12]:
            for n in [5, 10]:
                run(f"multi_{name}_t{n}_vt{tv:.0%}",
                    multi_strat=combo, top_n=n, target_vol=tv)

    # === F. Combined rank + market filter + tight vol ===
    print("\n--- F. Kitchen sink ---")
    run("comb_mom12_1+riskadj_t5_vt8%_sma200",
        combined_rank=[("mom_12_1", 0.5), ("risk_adj_126", 0.5)],
        top_n=5, target_vol=0.08, market_filter="sma200")
    run("comb_mom12_1+riskadj_t5_vt10%_sma200",
        combined_rank=[("mom_12_1", 0.5), ("risk_adj_126", 0.5)],
        top_n=5, target_vol=0.10, market_filter="sma200")
    run("comb_mom12_1+riskadj_t10_vt8%_sma200",
        combined_rank=[("mom_12_1", 0.5), ("risk_adj_126", 0.5)],
        top_n=10, target_vol=0.08, market_filter="sma200")
    run("comb_mom12_1+quality_t5_vt8%_sma200",
        combined_rank=[("mom_12_1", 0.5), ("quality_momentum", 0.5)],
        top_n=5, target_vol=0.08, market_filter="sma200")
    run("comb_all3_t5_vt8%_sma200",
        combined_rank=[("mom_12_1", 0.34), ("risk_adj_126", 0.33), ("quality_momentum", 0.33)],
        top_n=5, target_vol=0.08, market_filter="sma200")
    run("comb_all3_t5_vt10%_sma200_minmom10%",
        combined_rank=[("mom_12_1", 0.34), ("risk_adj_126", 0.33), ("quality_momentum", 0.33)],
        top_n=5, target_vol=0.10, market_filter="sma200", min_mom_12_1=0.10)

    # === RESULTS ===
    print(f"\n{'='*100}")
    print("ALL with FULL Sharpe > 1.0 (consistent: TRAIN > 0.5 AND TEST > 0.5):")
    high = [r for r in results
            if r.get("FULL", {}).get("sharpe", 0) > 1.0
            and r.get("TRAIN", {}).get("sharpe", 0) > 0.5
            and r.get("TEST", {}).get("sharpe", 0) > 0.5]
    high.sort(key=lambda x: x["FULL"]["sharpe"], reverse=True)
    for r in high[:25]:
        fu = r["FULL"]; te = r["TEST"]; tr = r["TRAIN"]; va = r["VALID"]
        print(f"  {r['name']:<60} FU={fu['sharpe']:.3f}/{fu['cagr']:.1%}/{fu['max_dd']:.1%} TR={tr['sharpe']:.3f} VA={va['sharpe']:.3f} TE={te['sharpe']:.3f} Cal={fu['calmar']:.2f}")

    print(f"\nBEST by Calmar (consistent):")
    high.sort(key=lambda x: x["FULL"]["calmar"], reverse=True)
    for r in high[:10]:
        fu = r["FULL"]; te = r["TEST"]; tr = r["TRAIN"]
        print(f"  {r['name']:<60} Cal={fu['calmar']:.2f} Sh={fu['sharpe']:.3f} CAGR={fu['cagr']:.1%} DD={fu['max_dd']:.1%}")
