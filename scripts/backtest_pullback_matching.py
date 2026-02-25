#!/usr/bin/env python3
"""
Backtest v2: Same-stock vs Cross-stock pooled analog matching at scale.

30+ diverse tickers spanning:
  - Big tech, mid-cap growth, recent IPOs (short history)
  - Cyclicals, defensives, financials, energy, biotech
  - Varying history lengths to test pooling benefit for thin-data stocks

Compares three strategies (all k=250):
  A) Same-stock (current production approach)
  B) Pooled cross-stock (analogs from any stock in the pool)
  C) Hybrid (same-stock primary; pooled fallback when same-stock n < threshold)

Key question: does pooling help exponentially for stocks with short histories
where same-stock analog data is thin?

Run:
    python scripts/backtest_pullback_matching.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from daily_scan import (
    download_ohlcv_period,
    compute_core_features,
    compute_idiosyncratic_features,
    compute_market_regime,
    compute_bottom_confirmation,
    compute_washout_meter,
    forward_returns,
    build_feature_matrix,
    make_regime_bucket,
    select_analogs_regime_balanced,
    summarize,
    robust_z,
    safe_float,
    LB_LT, LB_ST, BETA_LB,
    ANALOG_K, ANALOG_MIN, ANALOG_MIN_SEP_DAYS,
    MIN_HISTORY_BARS,
    BENCH,
)

# =========================
# CONFIG
# =========================
# Diverse universe: mix of history lengths, sectors, volatility profiles
TEST_TICKERS = [
    # Long history (20+ years) — big blue chips
    "AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "WMT", "KO", "HD", "UNH",
    # Medium history (10-20 years) — growth / tech / cyclicals
    "NVDA", "AMZN", "GOOGL", "META", "TSLA", "CRM", "NFLX", "COST", "AMD", "AVGO",
    # Shorter history (5-10 years) — newer listings / IPOs / spinoffs
    "UBER", "SNOW", "CRWD", "DDOG", "ZS", "NET", "ABNB", "ARM", "DASH", "PLTR",
    # More sectors: energy, biotech, industrials, REITs
    "CVX", "LLY", "CAT", "BA", "GS", "MS", "ABBV", "MRK", "AMT",
]

K = 250                     # analog pool size — same for all methods
SAMPLE_EVERY = 63           # sample quarterly to avoid autocorrelation
MIN_EVAL_BARS = LB_LT + BETA_LB + LB_ST + 252 + 252  # need features + 1Y forward
HYBRID_FALLBACK_N = 150     # if same-stock finds < this many analogs, use pooled

FEATURE_COLS = [
    "dd_lt", "pos_lt", "dd_st", "pos_st",
    "atr_pct", "volu_z", "gap", "trend_st",
    "idio_dd_lt", "idio_pos_lt", "idio_dd_st", "idio_pos_st",
    "mkt_trend", "mkt_vol", "mkt_dd", "mkt_atr_pct",
]
ZWIN = max(63, LB_ST)


# =========================
# Pooled KNN
# =========================
def knn_analogs_pooled(pooled_X, pooled_y, now_vec, now_idx, now_ticker, k,
                       min_sep_days=10, exclude_self=False):
    """Cross-stock KNN: find nearest neighbors across all stocks.

    Args:
        exclude_self: if True, exclude analogs from the same ticker
                      (tests pure cross-stock signal)
    """
    all_dists = []
    all_tickers = []
    all_timestamps = []

    for ticker, X_t in pooled_X.items():
        if exclude_self and ticker == now_ticker:
            continue
        y_t = pooled_y[ticker]
        valid_mask = X_t.notna().all(axis=1) & y_t.notna() & (X_t.index < now_idx)
        valid_idx = X_t.index[valid_mask]
        if len(valid_idx) == 0:
            continue

        Xc = X_t.loc[valid_idx].values.astype(float)
        dists = np.sqrt(((Xc - now_vec) ** 2).sum(axis=1))

        all_dists.extend(dists)
        all_tickers.extend([ticker] * len(dists))
        all_timestamps.extend(valid_idx)

    if not all_dists:
        return [], []

    order = np.argsort(all_dists)

    chosen = []
    chosen_tickers = []
    last_per_ticker = {}
    for idx in order:
        if len(chosen) >= k:
            break
        ticker = all_tickers[idx]
        t = all_timestamps[idx]
        if ticker in last_per_ticker:
            if abs((t - last_per_ticker[ticker]).days) < min_sep_days:
                continue
        chosen.append((ticker, t))
        chosen_tickers.append(ticker)
        last_per_ticker[ticker] = t

    return chosen, chosen_tickers


def compute_pooled_winrate(pooled_X, pooled_y, now_vec, now_idx, now_ticker, k,
                           exclude_self=False):
    """Get 1Y win-rate from pooled cross-stock analogs."""
    analogs, tickers_used = knn_analogs_pooled(
        pooled_X, pooled_y, now_vec, now_idx, now_ticker, k,
        exclude_self=exclude_self)
    if len(analogs) < ANALOG_MIN:
        return np.nan, 0, np.nan, []
    fwd_vals = []
    for ticker, t in analogs:
        val = pooled_y[ticker].get(t, np.nan)
        if np.isfinite(val):
            fwd_vals.append(val)
    if len(fwd_vals) < ANALOG_MIN:
        return np.nan, 0, np.nan, []
    fwd_vals = np.array(fwd_vals)
    return float(np.mean(fwd_vals > 0)), len(fwd_vals), float(np.median(fwd_vals)), tickers_used


def compute_same_stock_winrate(X, y, regimes, now_idx, k):
    """Get 1Y win-rate from same-stock analogs."""
    analog_idx = select_analogs_regime_balanced(
        X, y, regimes, now_idx, k=k, min_sep_days=ANALOG_MIN_SEP_DAYS)
    if len(analog_idx) < ANALOG_MIN:
        return np.nan, len(analog_idx), np.nan
    vals = y.loc[analog_idx].dropna().values.astype(float)
    vals = np.where(np.isnan(vals), -1.0, vals)
    if len(vals) < ANALOG_MIN:
        return np.nan, len(vals), np.nan
    return float(np.mean(vals > 0)), len(vals), float(np.median(vals))


# =========================
# Main backtest
# =========================
def run_backtest():
    all_tickers = sorted(set(TEST_TICKERS + [BENCH]))
    print(f"[BACKTEST v2] Downloading data for {len(all_tickers)} tickers ...")
    print(f"  Tickers: {all_tickers}")

    data = download_ohlcv_period(all_tickers, period="max", interval="1d", chunk_size=40)
    O, H, L, C, V, A = (
        data["Open"], data["High"], data["Low"],
        data["Close"], data["Volume"], data["AdjClose"],
    )
    PX = A if (not A.empty and BENCH in A.columns) else C

    # SPY / market regime
    spy_px = PX[BENCH].dropna()
    spy_h = H[BENCH].reindex(spy_px.index).dropna()
    spy_l = L[BENCH].reindex(spy_px.index).dropna()
    common = spy_px.index.intersection(spy_h.index).intersection(spy_l.index)
    spy_px = spy_px.reindex(common)
    spy_h = spy_h.reindex(common)
    spy_l = spy_l.reindex(common)
    mkt = compute_market_regime(spy_px, spy_h, spy_l)

    print(f"[DATA] SPY: {len(spy_px)} bars, {spy_px.index[0].date()} to {spy_px.index[-1].date()}")

    # Build features for each ticker
    ticker_data = {}
    pooled_X = {}
    pooled_y = {}
    ticker_history_len = {}  # {ticker: n_bars} for grouping analysis

    for i, t in enumerate(sorted(TEST_TICKERS)):
        if t not in C.columns or t == BENCH:
            print(f"  [{i+1:>2}/{len(TEST_TICKERS)}] SKIP {t} — not in data")
            continue

        df = pd.DataFrame({
            "open": O[t], "high": H[t], "low": L[t],
            "close": C[t], "volume": V[t], "px": PX[t],
        }).dropna(subset=["open", "high", "low", "close", "volume", "px"])

        if len(df) < MIN_HISTORY_BARS:
            print(f"  [{i+1:>2}/{len(TEST_TICKERS)}] SKIP {t} — {len(df)} bars (need {MIN_HISTORY_BARS})")
            continue

        feat = compute_core_features(df[["open", "high", "low", "close", "volume"]])
        feat["px"] = df["px"]

        idx = feat.index.intersection(spy_px.index).intersection(mkt.index)
        if len(idx) < MIN_HISTORY_BARS:
            print(f"  [{i+1:>2}/{len(TEST_TICKERS)}] SKIP {t} — {len(idx)} bars after SPY align")
            continue
        feat = feat.reindex(idx)
        spy_aligned = spy_px.reindex(idx)

        idio = compute_idiosyncratic_features(feat["px"], spy_aligned)
        feat = feat.join(idio, how="left")
        feat["bottom_confirm"] = compute_bottom_confirmation(feat["dd_lt"], feat["pos_lt"])
        feat = feat.join(mkt.reindex(idx), how="left")
        feat = feat.join(forward_returns(feat["px"]), how="left")
        feat["washout_meter"] = compute_washout_meter(feat)

        X = build_feature_matrix(feat, FEATURE_COLS, zwin=ZWIN)
        regimes = feat.apply(make_regime_bucket, axis=1)
        y_1Y = feat.get("fwd_1Y", pd.Series(dtype=float))

        ticker_data[t] = {"feat": feat, "X": X, "regimes": regimes, "y_1Y": y_1Y}
        pooled_X[t] = X
        pooled_y[t] = y_1Y
        ticker_history_len[t] = len(feat)

        yrs = len(feat) / 252
        print(f"  [{i+1:>2}/{len(TEST_TICKERS)}] OK   {t:<6} {len(feat):>5} bars ({yrs:.1f}Y), "
              f"{feat.index[0].date()} to {feat.index[-1].date()}")

    n_ok = len(ticker_data)
    if n_ok < 5:
        print(f"[ERROR] Only {n_ok} tickers processed. Aborting.")
        return

    print(f"\n[POOL] {n_ok} tickers in pool, total bars: {sum(ticker_history_len.values()):,}")

    # Classify tickers by history length
    short_thr = 2520   # ~10 years
    med_thr = 5040     # ~20 years
    short_tickers = [t for t, n in ticker_history_len.items() if n < short_thr]
    med_tickers = [t for t, n in ticker_history_len.items() if short_thr <= n < med_thr]
    long_tickers = [t for t, n in ticker_history_len.items() if n >= med_thr]
    print(f"  Short (<10Y): {short_tickers}")
    print(f"  Medium (10-20Y): {med_tickers}")
    print(f"  Long (20Y+): {long_tickers}")

    # =========================
    # Run evaluation
    # =========================
    print(f"\n{'='*110}")
    print(f"RUNNING BACKTEST — k={K}, sampling every {SAMPLE_EVERY} bars")
    print(f"{'='*110}")

    results = []
    t0 = time.time()

    for ti, (t, td) in enumerate(ticker_data.items()):
        feat = td["feat"]
        X = td["X"]
        regimes = td["regimes"]
        y_1Y = td["y_1Y"]

        ok = X.notna().all(axis=1) & y_1Y.notna() & feat["bottom_confirm"].notna()
        ok_idx = X.index[ok]
        if len(ok_idx) == 0:
            continue
        min_start = ok_idx[0] + pd.Timedelta(days=MIN_EVAL_BARS)
        eval_idx = ok_idx[ok_idx >= min_start]
        eval_idx = eval_idx[::SAMPLE_EVERY]

        n_hist = ticker_history_len[t]
        hist_group = "short" if n_hist < short_thr else ("medium" if n_hist < med_thr else "long")

        print(f"\n  [{ti+1}/{n_ok}] {t} ({hist_group}, {len(eval_idx)} eval pts) ...", end="", flush=True)

        for i, now_idx in enumerate(eval_idx):
            actual_1y = float(y_1Y.loc[now_idx])
            actual_positive = 1 if actual_1y > 0 else 0
            washout = safe_float(feat.loc[now_idx, "washout_meter"])

            now_vec = X.loc[now_idx].values.astype(float)
            if not np.all(np.isfinite(now_vec)):
                continue

            # How many same-stock data points exist before this date?
            n_same_avail = int((X.index < now_idx).sum())

            # Method A: Same-stock, k=250
            win_a, n_a, med_a = compute_same_stock_winrate(X, y_1Y, regimes, now_idx, k=K)

            # Method B: Pooled cross-stock, k=250
            win_b, n_b, med_b, tickers_b = compute_pooled_winrate(
                pooled_X, pooled_y, now_vec, now_idx, t, k=K, exclude_self=False)

            # Method B2: Pooled EXCLUDING self (pure cross-stock signal)
            win_b2, n_b2, med_b2, tickers_b2 = compute_pooled_winrate(
                pooled_X, pooled_y, now_vec, now_idx, t, k=K, exclude_self=True)

            # Method C: Hybrid — same-stock primary, pooled fallback
            if n_a >= HYBRID_FALLBACK_N and np.isfinite(win_a):
                win_c, n_c, med_c = win_a, n_a, med_a
                hybrid_used = "same"
            else:
                win_c, n_c, med_c = win_b, n_b, med_b
                hybrid_used = "pooled"

            # Count unique tickers in pooled analogs
            unique_tickers_b = len(set(tickers_b)) if tickers_b else 0
            # What fraction of pooled analogs come from the same stock?
            self_frac_b = (sum(1 for tk in tickers_b if tk == t) / len(tickers_b)
                           if tickers_b else 0)

            results.append({
                "ticker": t,
                "date": now_idx,
                "hist_group": hist_group,
                "n_hist_bars": n_hist,
                "n_same_avail": n_same_avail,
                "actual_1y": actual_1y,
                "actual_pos": actual_positive,
                "washout": washout if np.isfinite(washout) else 0,
                # Method A: same-stock
                "win_a": win_a, "n_a": n_a, "med_a": med_a,
                # Method B: pooled (incl self)
                "win_b": win_b, "n_b": n_b, "med_b": med_b,
                "n_tickers_b": unique_tickers_b,
                "self_frac_b": self_frac_b,
                # Method B2: pooled (excl self)
                "win_b2": win_b2, "n_b2": n_b2, "med_b2": med_b2,
                # Method C: hybrid
                "win_c": win_c, "n_c": n_c, "med_c": med_c,
                "hybrid_used": hybrid_used,
            })

        print(f" done ({time.time() - t0:.0f}s elapsed)")

    df = pd.DataFrame(results)
    print(f"\n[DATA] Total evaluation points: {len(df)}")

    if len(df) < 50:
        print("[ERROR] Not enough evaluation points for meaningful analysis")
        return

    # =========================
    # Analysis
    # =========================
    methods = {
        "A: Same-stock k=250 (current)":  ("win_a", "n_a", "med_a"),
        "B: Pooled k=250 (incl self)":    ("win_b", "n_b", "med_b"),
        "B2: Pooled k=250 (excl self)":   ("win_b2", "n_b2", "med_b2"),
        "C: Hybrid (same→pooled fallback)":("win_c", "n_c", "med_c"),
    }

    # --- 1. Overall ---
    print(f"\n{'='*110}")
    print("OVERALL RESULTS")
    print(f"{'='*110}")
    print(f"\n{'Method':<40} {'N_valid':>8} {'Pred Win':>10} {'Act Hit':>10} {'Brier':>8} {'Med Ret':>10}")
    print("-" * 95)

    for label, (win_col, n_col, med_col) in methods.items():
        valid = df[df[win_col].notna()]
        if len(valid) == 0:
            continue
        pred_win = valid[win_col].mean()
        act_hit = valid["actual_pos"].mean()
        brier = ((valid[win_col] - valid["actual_pos"]) ** 2).mean()
        med_ret = valid["actual_1y"].median()
        print(f"{label:<40} {len(valid):>8} {pred_win:>9.1%} {act_hit:>9.1%} {brier:>7.4f} {med_ret:>+9.1%}")

    # --- 2. THE KEY QUESTION: breakdown by history length ---
    print(f"\n{'='*110}")
    print("KEY QUESTION: Does pooling help for stocks with shorter history?")
    print(f"{'='*110}")

    for group, group_label in [("short", "SHORT HISTORY (<10Y)"),
                                ("medium", "MEDIUM HISTORY (10-20Y)"),
                                ("long", "LONG HISTORY (20Y+)")]:
        g_df = df[df["hist_group"] == group]
        if len(g_df) < 10:
            print(f"\n  {group_label}: <10 eval points, skipping")
            continue

        tickers_in_group = sorted(g_df["ticker"].unique())
        print(f"\n  {group_label} — {len(g_df)} eval points across {tickers_in_group}")
        print(f"  {'Method':<40} {'N':>6} {'Pred Win':>10} {'Act Hit':>10} {'Brier':>8} {'Med Ret':>10}")
        print(f"  {'-'*85}")

        for label, (win_col, n_col, med_col) in methods.items():
            valid = g_df[g_df[win_col].notna()]
            if len(valid) < 5:
                continue
            pred_win = valid[win_col].mean()
            act_hit = valid["actual_pos"].mean()
            brier = ((valid[win_col] - valid["actual_pos"]) ** 2).mean()
            med_ret = valid["actual_1y"].median()
            print(f"  {label:<40} {len(valid):>6} {pred_win:>9.1%} {act_hit:>9.1%} {brier:>7.4f} {med_ret:>+9.1%}")

    # --- 3. Breakdown by same-stock data availability at eval time ---
    print(f"\n{'='*110}")
    print("SAME-STOCK DATA AVAILABILITY: How well do methods perform when same-stock data is thin?")
    print(f"{'='*110}")

    avail_bins = [(0, 500, "<500 same-stock bars"),
                  (500, 1000, "500-1000 bars"),
                  (1000, 2000, "1000-2000 bars"),
                  (2000, 5000, "2000-5000 bars"),
                  (5000, 99999, "5000+ bars")]

    for lo, hi, blabel in avail_bins:
        sub = df[(df["n_same_avail"] >= lo) & (df["n_same_avail"] < hi)]
        if len(sub) < 10:
            continue
        print(f"\n  {blabel} (N={len(sub)}, tickers: {sorted(sub['ticker'].unique())})")
        print(f"  {'Method':<40} {'N_valid':>8} {'Act Hit':>10} {'Brier':>8} {'Med Ret':>10}")
        print(f"  {'-'*75}")
        for label, (win_col, n_col, med_col) in methods.items():
            valid = sub[sub[win_col].notna()]
            if len(valid) < 5:
                continue
            act_hit = valid["actual_pos"].mean()
            brier = ((valid[win_col] - valid["actual_pos"]) ** 2).mean()
            med_ret = valid["actual_1y"].median()
            print(f"  {label:<40} {len(valid):>8} {act_hit:>9.1%} {brier:>7.4f} {med_ret:>+9.1%}")

    # --- 4. Calibration quintiles (overall) ---
    print(f"\n{'='*110}")
    print("CALIBRATION: Predicted win-rate quintile vs actual hit-rate")
    print(f"{'='*110}")

    for label, (win_col, n_col, med_col) in methods.items():
        valid = df[df[win_col].notna()].copy()
        if len(valid) < 50:
            continue
        print(f"\n  {label}")
        try:
            valid["q"] = pd.qcut(valid[win_col], 5,
                                 labels=["Q1_Low", "Q2", "Q3", "Q4", "Q5_High"],
                                 duplicates="drop")
        except ValueError:
            valid["q"] = pd.qcut(valid[win_col].rank(method="first"), 5,
                                 labels=["Q1_Low", "Q2", "Q3", "Q4", "Q5_High"])

        print(f"  {'Quintile':<12} {'N':>5} {'Pred Win':>10} {'Act Hit':>10} {'Med Ret':>10} {'Avg Ret':>10}")
        print(f"  {'-'*60}")
        for q in ["Q1_Low", "Q2", "Q3", "Q4", "Q5_High"]:
            sub = valid[valid["q"] == q]
            if len(sub) == 0:
                continue
            pw = sub[win_col].mean()
            ah = sub["actual_pos"].mean()
            mr = sub["actual_1y"].median()
            ar = sub["actual_1y"].mean()
            print(f"  {q:<12} {len(sub):>5} {pw:>9.1%} {ah:>9.1%} {mr:>+9.1%} {ar:>+9.1%}")

    # --- 5. High-conviction signals ---
    print(f"\n{'='*110}")
    print("HIGH-CONVICTION: predicted win > threshold")
    print(f"{'='*110}")

    thresholds = [0.60, 0.65, 0.70, 0.75]
    print(f"\n{'Method':<40} {'Thr':>5} {'N':>6} {'Act Hit':>10} {'Med Ret':>10} {'Avg Ret':>10} {'P10':>10}")
    print("-" * 100)

    for label, (win_col, n_col, med_col) in methods.items():
        valid = df[df[win_col].notna()]
        for thr in thresholds:
            sub = valid[valid[win_col] >= thr]
            if len(sub) < 10:
                continue
            ah = sub["actual_pos"].mean()
            mr = sub["actual_1y"].median()
            ar = sub["actual_1y"].mean()
            p10 = np.quantile(sub["actual_1y"].values, 0.10)
            print(f"{label:<40} {thr:>4.0%} {len(sub):>6} {ah:>9.1%} {mr:>+9.1%} {ar:>+9.1%} {p10:>+9.1%}")

    # --- 6. Pullback-gated ---
    print(f"\n{'='*110}")
    print("PULLBACK-GATED: washout > 20 AND predicted win > 65%")
    print(f"{'='*110}")

    print(f"\n{'Method':<40} {'N':>6} {'Act Hit':>10} {'Med Ret':>10} {'Avg Ret':>10} {'P10':>10}")
    print("-" * 90)

    for label, (win_col, n_col, med_col) in methods.items():
        valid = df[df[win_col].notna()]
        sub = valid[(valid[win_col] >= 0.65) & (valid["washout"] >= 20)]
        if len(sub) < 5:
            print(f"{label:<40} {'<5':>6}")
            continue
        ah = sub["actual_pos"].mean()
        mr = sub["actual_1y"].median()
        ar = sub["actual_1y"].mean()
        p10 = np.quantile(sub["actual_1y"].values, 0.10)
        print(f"{label:<40} {len(sub):>6} {ah:>9.1%} {mr:>+9.1%} {ar:>+9.1%} {p10:>+9.1%}")

    # --- 7. Pullback-gated by history group ---
    print(f"\n{'='*110}")
    print("PULLBACK-GATED by HISTORY LENGTH")
    print("(washout > 20 AND predicted win > 65%)")
    print(f"{'='*110}")

    for group in ["short", "medium", "long"]:
        g_df = df[df["hist_group"] == group]
        if len(g_df) < 5:
            continue
        print(f"\n  {group.upper()} history:")
        print(f"  {'Method':<40} {'N':>6} {'Act Hit':>10} {'Med Ret':>10}")
        print(f"  {'-'*70}")
        for label, (win_col, n_col, med_col) in methods.items():
            valid = g_df[g_df[win_col].notna()]
            sub = valid[(valid[win_col] >= 0.65) & (valid["washout"] >= 20)]
            if len(sub) < 3:
                print(f"  {label:<40} {'<3':>6}")
                continue
            ah = sub["actual_pos"].mean()
            mr = sub["actual_1y"].median()
            print(f"  {label:<40} {len(sub):>6} {ah:>9.1%} {mr:>+9.1%}")

    # --- 8. Pooled diversity ---
    print(f"\n{'='*110}")
    print("POOLED ANALOG DIVERSITY")
    print(f"{'='*110}")

    valid_b = df[df["n_tickers_b"] > 0]
    if len(valid_b) > 0:
        print(f"\n  Pooled k=250 (incl self):")
        print(f"    Mean unique tickers per analog set: {valid_b['n_tickers_b'].mean():.1f}")
        print(f"    Median fraction from SAME stock:    {valid_b['self_frac_b'].median():.1%}")
        for group in ["short", "medium", "long"]:
            g = valid_b[valid_b["hist_group"] == group]
            if len(g) > 0:
                print(f"    {group:>8}: same-stock frac = {g['self_frac_b'].median():.1%}, "
                      f"unique tickers = {g['n_tickers_b'].median():.0f}")

    # --- 9. Hybrid analysis ---
    print(f"\n{'='*110}")
    print(f"HYBRID METHOD: fallback threshold = {HYBRID_FALLBACK_N} analogs")
    print(f"{'='*110}")

    hybrid_df = df[df["win_c"].notna()]
    if len(hybrid_df) > 0:
        n_same = (hybrid_df["hybrid_used"] == "same").sum()
        n_pool = (hybrid_df["hybrid_used"] == "pooled").sum()
        print(f"  Used same-stock: {n_same} ({100*n_same/len(hybrid_df):.1f}%)")
        print(f"  Used pooled:     {n_pool} ({100*n_pool/len(hybrid_df):.1f}%)")

        for used in ["same", "pooled"]:
            sub = hybrid_df[hybrid_df["hybrid_used"] == used]
            if len(sub) < 10:
                continue
            act_hit = sub["actual_pos"].mean()
            brier = ((sub["win_c"] - sub["actual_pos"]) ** 2).mean()
            print(f"    When using {used:>6}: N={len(sub)}, Hit={act_hit:.1%}, Brier={brier:.4f}")

    # --- 10. Per-stock breakdown ---
    print(f"\n{'='*110}")
    print("PER-STOCK: Act hit-rate when predicted win > 60%")
    print(f"{'='*110}")

    for t in sorted(df["ticker"].unique()):
        t_df = df[df["ticker"] == t]
        hist_g = t_df["hist_group"].iloc[0] if len(t_df) > 0 else "?"
        n_bars = ticker_history_len.get(t, 0)
        print(f"\n  {t} ({hist_g}, {n_bars} bars, {len(t_df)} evals)")
        print(f"  {'Method':<40} {'N>60%':>6} {'Act Hit':>10} {'Med Ret':>10}")
        print(f"  {'-'*70}")
        for label, (win_col, n_col, med_col) in methods.items():
            valid = t_df[t_df[win_col].notna()]
            sub = valid[valid[win_col] >= 0.60]
            if len(sub) < 3:
                print(f"  {label:<40} {'<3':>6}")
                continue
            ah = sub["actual_pos"].mean()
            mr = sub["actual_1y"].median()
            print(f"  {label:<40} {len(sub):>6} {ah:>9.1%} {mr:>+9.1%}")

    # --- 11. Summary ---
    print(f"\n{'='*110}")
    print("SUMMARY")
    print(f"{'='*110}")

    baseline_hit = float(df["actual_pos"].mean())
    baseline_med = float(df["actual_1y"].median())
    print(f"\n  Baseline (all eval points): Hit={baseline_hit:.1%}, Med 1Y={baseline_med:+.1%}")

    for label, (win_col, n_col, med_col) in methods.items():
        valid = df[df[win_col].notna()]
        if len(valid) < 20:
            continue
        brier = float(((valid[win_col] - valid["actual_pos"]) ** 2).mean())

        hc = valid[valid[win_col] >= 0.65]
        hc_hit = float(hc["actual_pos"].mean()) if len(hc) >= 10 else np.nan
        hc_med = float(hc["actual_1y"].median()) if len(hc) >= 10 else np.nan

        gate = valid[(valid[win_col] >= 0.65) & (valid["washout"] >= 20)]
        gate_hit = float(gate["actual_pos"].mean()) if len(gate) >= 5 else np.nan
        gate_med = float(gate["actual_1y"].median()) if len(gate) >= 5 else np.nan

        print(f"\n  {label}")
        print(f"    Brier: {brier:.4f}")
        print(f"    High-conviction (win>65%): N={len(hc)}, Hit={hc_hit:.1%}, Med={hc_med:+.1%}" if np.isfinite(hc_hit) else "    High-conviction: insufficient data")
        print(f"    Pullback-gated:            N={len(gate)}, Hit={gate_hit:.1%}, Med={gate_med:+.1%}" if np.isfinite(gate_hit) else "    Pullback-gated: insufficient data")

    print(f"\n  Elapsed: {time.time() - t0:.0f}s")
    print(f"{'='*110}")


if __name__ == "__main__":
    run_backtest()
