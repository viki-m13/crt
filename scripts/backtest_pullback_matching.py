#!/usr/bin/env python3
"""
Backtest: Same-stock analog matching vs Cross-stock pooled matching.

Compares four strategies:
  A) Current approach — same-stock KNN (k=250)
  B) Tighter same-stock — same-stock KNN (k=100, closer matches only)
  C) Pooled cross-stock — KNN across ALL stocks (k=250)
  D) Pooled + tighter — KNN across ALL stocks (k=100)

For each historical evaluation point (sampled every ~63 trading days to avoid
autocorrelation), we compute the predicted 1Y win-rate from each method's
analogs, then check whether the actual 1Y forward return was positive.

Metrics:
  - Calibration: predicted win-rate vs actual hit-rate
  - Hit-rate in top-quintile predictions (the actionable signal)
  - Median actual return when model says "strong buy" (predicted win > 70%)
  - Brier score (lower = better calibrated probability)

Run:
    python scripts/backtest_pullback_matching.py
"""

import sys, os
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
# Diverse test stocks: big-tech, cyclical, defensive, growth, value
TEST_TICKERS = ["AAPL", "JPM", "JNJ", "NVDA", "XOM"]
SAMPLE_EVERY = 63          # sample quarterly to keep it quick
MIN_EVAL_BARS = LB_LT + BETA_LB + LB_ST + 252 + 252  # need 1Y forward
FEATURE_COLS = [
    "dd_lt", "pos_lt", "dd_st", "pos_st",
    "atr_pct", "volu_z", "gap", "trend_st",
    "idio_dd_lt", "idio_pos_lt", "idio_dd_st", "idio_pos_st",
    "mkt_trend", "mkt_vol", "mkt_dd", "mkt_atr_pct",
]
ZWIN = max(63, LB_ST)


# =========================
# KNN analog selection (simplified — no regime balancing for pooled, to keep it clean)
# =========================
def knn_analogs_same_stock(X, y, regimes, now_idx, k):
    """Current approach: same-stock KNN with regime balancing."""
    return select_analogs_regime_balanced(X, y, regimes, now_idx, k=k, min_sep_days=ANALOG_MIN_SEP_DAYS)


def knn_analogs_pooled(pooled_X, pooled_y, now_vec, now_idx, k, min_sep_days=10):
    """Cross-stock KNN: find nearest neighbors across all stocks.
    pooled_X: dict of {ticker: DataFrame} of z-scored features
    pooled_y: dict of {ticker: Series} of forward returns
    now_vec: feature vector for the current point (1D array)
    now_idx: timestamp of current point (to ensure no look-ahead)
    """
    candidates = []  # (distance, ticker, timestamp)

    for ticker, X_t in pooled_X.items():
        y_t = pooled_y[ticker]
        # Only use points before now_idx with valid features and forward returns
        valid_mask = X_t.notna().all(axis=1) & y_t.notna() & (X_t.index < now_idx)
        valid_idx = X_t.index[valid_mask]
        if len(valid_idx) == 0:
            continue

        Xc = X_t.loc[valid_idx].values.astype(float)
        dists = np.sqrt(((Xc - now_vec) ** 2).sum(axis=1))

        for i, (t, d) in enumerate(zip(valid_idx, dists)):
            candidates.append((d, ticker, t))

    if not candidates:
        return []

    # Sort by distance
    candidates.sort(key=lambda x: x[0])

    # Select top-k with minimum separation (within same stock)
    chosen = []
    last_per_ticker = {}
    for dist, ticker, t in candidates:
        if len(chosen) >= k:
            break
        # Enforce min_sep_days within same ticker
        if ticker in last_per_ticker:
            last_t = last_per_ticker[ticker]
            if abs((t - last_t).days) < min_sep_days:
                continue
        chosen.append((ticker, t, dist))
        last_per_ticker[ticker] = t

    return chosen


def compute_pooled_winrate(pooled_X, pooled_y, now_vec, now_idx, k):
    """Get 1Y win-rate from pooled cross-stock analogs."""
    analogs = knn_analogs_pooled(pooled_X, pooled_y, now_vec, now_idx, k)
    if len(analogs) < ANALOG_MIN:
        return np.nan, 0, np.nan, []
    fwd_vals = []
    tickers_used = []
    for ticker, t, dist in analogs:
        val = pooled_y[ticker].get(t, np.nan)
        if np.isfinite(val):
            fwd_vals.append(val)
            tickers_used.append(ticker)
    if len(fwd_vals) < ANALOG_MIN:
        return np.nan, 0, np.nan, []
    fwd_vals = np.array(fwd_vals)
    win_rate = float(np.mean(fwd_vals > 0))
    median_ret = float(np.median(fwd_vals))
    return win_rate, len(fwd_vals), median_ret, tickers_used


def compute_same_stock_winrate(X, y, regimes, now_idx, k):
    """Get 1Y win-rate from same-stock analogs."""
    analog_idx = knn_analogs_same_stock(X, y, regimes, now_idx, k)
    if len(analog_idx) < ANALOG_MIN:
        return np.nan, 0, np.nan
    vals = y.loc[analog_idx].dropna().values.astype(float)
    vals = np.where(np.isnan(vals), -1.0, vals)
    if len(vals) < ANALOG_MIN:
        return np.nan, 0, np.nan
    win_rate = float(np.mean(vals > 0))
    median_ret = float(np.median(vals))
    return win_rate, len(vals), median_ret


# =========================
# Main backtest
# =========================
def run_backtest():
    all_tickers = sorted(set(TEST_TICKERS + [BENCH]))
    print(f"[BACKTEST] Downloading data for {all_tickers} ...")

    data = download_ohlcv_period(all_tickers, period="max", interval="1d", chunk_size=len(all_tickers))
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

    print(f"[BACKTEST] SPY data: {len(spy_px)} bars, {spy_px.index[0].date()} to {spy_px.index[-1].date()}")

    # Build features for each ticker
    ticker_data = {}  # {ticker: {feat, X, regimes, y_1Y}}
    pooled_X = {}     # {ticker: X DataFrame}
    pooled_y = {}     # {ticker: fwd_1Y Series}

    for t in TEST_TICKERS:
        if t not in C.columns:
            print(f"  [SKIP] {t} not in data")
            continue

        df = pd.DataFrame({
            "open": O[t], "high": H[t], "low": L[t],
            "close": C[t], "volume": V[t], "px": PX[t],
        }).dropna(subset=["open", "high", "low", "close", "volume", "px"])

        if len(df) < MIN_HISTORY_BARS:
            print(f"  [SKIP] {t} only {len(df)} bars (need {MIN_HISTORY_BARS})")
            continue

        feat = compute_core_features(df[["open", "high", "low", "close", "volume"]])
        feat["px"] = df["px"]

        idx = feat.index.intersection(spy_px.index).intersection(mkt.index)
        if len(idx) < MIN_HISTORY_BARS:
            print(f"  [SKIP] {t} only {len(idx)} bars after SPY align")
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

        ticker_data[t] = {
            "feat": feat, "X": X, "regimes": regimes, "y_1Y": y_1Y,
        }
        pooled_X[t] = X
        pooled_y[t] = y_1Y

        print(f"  [OK] {t}: {len(feat)} bars, {feat.index[0].date()} to {feat.index[-1].date()}, "
              f"fwd_1Y non-null: {y_1Y.notna().sum()}")

    if len(ticker_data) < 3:
        print("[ERROR] Not enough tickers processed. Aborting.")
        return

    # =========================
    # Run evaluation
    # =========================
    print(f"\n{'='*110}")
    print("RUNNING BACKTEST — sampling every {SAMPLE_EVERY} bars")
    print(f"{'='*110}")

    results = []  # list of dicts

    for t, td in ticker_data.items():
        feat = td["feat"]
        X = td["X"]
        regimes = td["regimes"]
        y_1Y = td["y_1Y"]

        # Valid evaluation points: need features + forward return + enough history
        ok = X.notna().all(axis=1) & y_1Y.notna() & feat["bottom_confirm"].notna()
        ok_idx = X.index[ok]
        # Need enough history before each point
        min_start = ok_idx[0] + pd.Timedelta(days=MIN_EVAL_BARS)
        eval_idx = ok_idx[ok_idx >= min_start]
        # Sample
        eval_idx = eval_idx[::SAMPLE_EVERY]

        print(f"\n  [{t}] {len(eval_idx)} evaluation points")

        for i, now_idx in enumerate(eval_idx):
            actual_1y = float(y_1Y.loc[now_idx])
            actual_positive = 1 if actual_1y > 0 else 0
            washout = safe_float(feat.loc[now_idx, "washout_meter"])

            now_vec = X.loc[now_idx].values.astype(float)
            if not np.all(np.isfinite(now_vec)):
                continue

            # Method A: Same-stock, k=250
            win_a, n_a, med_a = compute_same_stock_winrate(X, y_1Y, regimes, now_idx, k=250)

            # Method B: Same-stock, k=100 (tighter)
            win_b, n_b, med_b = compute_same_stock_winrate(X, y_1Y, regimes, now_idx, k=100)

            # Method C: Pooled, k=250
            win_c, n_c, med_c, tickers_c = compute_pooled_winrate(pooled_X, pooled_y, now_vec, now_idx, k=250)

            # Method D: Pooled, k=100 (tighter)
            win_d, n_d, med_d, tickers_d = compute_pooled_winrate(pooled_X, pooled_y, now_vec, now_idx, k=100)

            results.append({
                "ticker": t,
                "date": now_idx,
                "actual_1y": actual_1y,
                "actual_pos": actual_positive,
                "washout": washout if np.isfinite(washout) else 0,
                # Method A
                "win_a": win_a, "n_a": n_a, "med_a": med_a,
                # Method B
                "win_b": win_b, "n_b": n_b, "med_b": med_b,
                # Method C
                "win_c": win_c, "n_c": n_c, "med_c": med_c,
                "n_tickers_c": len(set(tickers_c)) if tickers_c else 0,
                # Method D
                "win_d": win_d, "n_d": n_d, "med_d": med_d,
                "n_tickers_d": len(set(tickers_d)) if tickers_d else 0,
            })

            if (i + 1) % 10 == 0:
                print(f"    ... {i+1}/{len(eval_idx)}")

    df = pd.DataFrame(results)
    print(f"\n[DATA] Total evaluation points: {len(df)}")

    if len(df) < 50:
        print("[ERROR] Not enough evaluation points for meaningful analysis")
        return

    # =========================
    # Analysis
    # =========================
    print(f"\n{'='*110}")
    print("RESULTS: SAME-STOCK vs POOLED ANALOG MATCHING")
    print(f"{'='*110}")

    methods = {
        "A: Same-stock k=250 (current)": ("win_a", "n_a", "med_a"),
        "B: Same-stock k=100 (tighter)": ("win_b", "n_b", "med_b"),
        "C: Pooled k=250":               ("win_c", "n_c", "med_c"),
        "D: Pooled k=100 (tighter)":     ("win_d", "n_d", "med_d"),
    }

    # --- 1. Overall metrics ---
    print(f"\n{'Method':<35} {'N_valid':>8} {'Pred Win':>10} {'Act Hit':>10} {'Brier':>8} {'Med Ret':>10}")
    print("-" * 90)

    for label, (win_col, n_col, med_col) in methods.items():
        valid = df[df[win_col].notna()]
        if len(valid) == 0:
            print(f"{label:<35} {'N/A':>8}")
            continue
        pred_win = valid[win_col].mean()
        act_hit = valid["actual_pos"].mean()
        brier = ((valid[win_col] - valid["actual_pos"]) ** 2).mean()
        med_ret = valid["actual_1y"].median()
        print(f"{label:<35} {len(valid):>8} {pred_win:>9.1%} {act_hit:>9.1%} {brier:>7.4f} {med_ret:>+9.1%}")

    # --- 2. Calibration by predicted-win quintile ---
    print(f"\n{'='*110}")
    print("CALIBRATION: Predicted win-rate quintile vs actual hit-rate")
    print("(Good model: predicted and actual should track each other)")
    print(f"{'='*110}")

    for label, (win_col, n_col, med_col) in methods.items():
        valid = df[df[win_col].notna()].copy()
        if len(valid) < 50:
            continue

        print(f"\n  {label}")
        try:
            valid["q"] = pd.qcut(valid[win_col], 5, labels=["Q1_Low", "Q2", "Q3", "Q4", "Q5_High"], duplicates="drop")
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

    # --- 3. High-conviction signals (pred win > 65%) ---
    print(f"\n{'='*110}")
    print("HIGH-CONVICTION SIGNALS: When model predicts win > 65%")
    print("(This is what actually matters for stock picks)")
    print(f"{'='*110}")

    thresholds = [0.60, 0.65, 0.70, 0.75]
    print(f"\n{'Method':<35} {'Thr':>5} {'N':>6} {'Act Hit':>10} {'Med Ret':>10} {'Avg Ret':>10} {'P10 Ret':>10}")
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
            print(f"{label:<35} {thr:>4.0%} {len(sub):>6} {ah:>9.1%} {mr:>+9.1%} {ar:>+9.1%} {p10:>+9.1%}")

    # --- 4. Pullback-gated signals (washout > 20 AND pred win > 65%) ---
    print(f"\n{'='*110}")
    print("PULLBACK-GATED: washout > 20 AND predicted win > 65%")
    print("(Closest to actual production usage)")
    print(f"{'='*110}")

    print(f"\n{'Method':<35} {'N':>6} {'Act Hit':>10} {'Med Ret':>10} {'Avg Ret':>10} {'P10 Ret':>10}")
    print("-" * 90)

    for label, (win_col, n_col, med_col) in methods.items():
        valid = df[df[win_col].notna()]
        sub = valid[(valid[win_col] >= 0.65) & (valid["washout"] >= 20)]
        if len(sub) < 5:
            print(f"{label:<35} {'<5 obs':>6}")
            continue
        ah = sub["actual_pos"].mean()
        mr = sub["actual_1y"].median()
        ar = sub["actual_1y"].mean()
        p10 = np.quantile(sub["actual_1y"].values, 0.10)
        print(f"{label:<35} {len(sub):>6} {ah:>9.1%} {mr:>+9.1%} {ar:>+9.1%} {p10:>+9.1%}")

    # --- 5. Per-stock breakdown ---
    print(f"\n{'='*110}")
    print("PER-STOCK BREAKDOWN (Act Hit-rate when predicted win > 60%)")
    print(f"{'='*110}")

    for t in sorted(df["ticker"].unique()):
        t_df = df[df["ticker"] == t]
        print(f"\n  {t} ({len(t_df)} eval points)")
        print(f"  {'Method':<35} {'N>60%':>6} {'Act Hit':>10} {'Med Ret':>10}")
        print(f"  {'-'*65}")
        for label, (win_col, n_col, med_col) in methods.items():
            valid = t_df[t_df[win_col].notna()]
            sub = valid[valid[win_col] >= 0.60]
            if len(sub) < 3:
                print(f"  {label:<35} {'<3':>6}")
                continue
            ah = sub["actual_pos"].mean()
            mr = sub["actual_1y"].median()
            print(f"  {label:<35} {len(sub):>6} {ah:>9.1%} {mr:>+9.1%}")

    # --- 6. Pooled diversity analysis ---
    print(f"\n{'='*110}")
    print("POOLED METHOD: Analog diversity (how many different tickers contribute?)")
    print(f"{'='*110}")

    for col, label in [("n_tickers_c", "Pooled k=250"), ("n_tickers_d", "Pooled k=100")]:
        valid = df[df[col] > 0]
        if len(valid) == 0:
            continue
        print(f"\n  {label}:")
        print(f"    Mean tickers per analog set: {valid[col].mean():.1f}")
        print(f"    Median: {valid[col].median():.0f}")
        print(f"    Min: {valid[col].min():.0f}, Max: {valid[col].max():.0f}")

    # --- 7. Baseline comparison ---
    print(f"\n{'='*110}")
    print("BASELINE: Buy-and-hold (any random day)")
    print(f"{'='*110}")

    all_actual = df["actual_1y"].dropna()
    print(f"  All eval points: N={len(all_actual)}, Hit={float(np.mean(all_actual > 0)):.1%}, "
          f"Med={float(np.median(all_actual)):.1%}, Avg={float(np.mean(all_actual)):.1%}")

    # --- 8. Summary verdict ---
    print(f"\n{'='*110}")
    print("SUMMARY & RECOMMENDATION")
    print(f"{'='*110}")

    # Compute summary stats for comparison
    summary = {}
    for label, (win_col, n_col, med_col) in methods.items():
        valid = df[df[win_col].notna()]
        if len(valid) < 20:
            continue
        brier = float(((valid[win_col] - valid["actual_pos"]) ** 2).mean())
        high_conv = valid[valid[win_col] >= 0.65]
        hc_hit = float(high_conv["actual_pos"].mean()) if len(high_conv) >= 10 else np.nan
        hc_med = float(high_conv["actual_1y"].median()) if len(high_conv) >= 10 else np.nan
        summary[label] = {"brier": brier, "hc_hit": hc_hit, "hc_med": hc_med, "n": len(valid)}

    if summary:
        best_brier = min(summary.items(), key=lambda x: x[1]["brier"])
        best_hc_hit = max(summary.items(), key=lambda x: x[1]["hc_hit"] if np.isfinite(x[1]["hc_hit"]) else -1)
        best_hc_med = max(summary.items(), key=lambda x: x[1]["hc_med"] if np.isfinite(x[1]["hc_med"]) else -999)

        print(f"\n  Best calibrated (lowest Brier): {best_brier[0]} (Brier={best_brier[1]['brier']:.4f})")
        print(f"  Best high-conviction hit-rate:  {best_hc_hit[0]} (Hit={best_hc_hit[1]['hc_hit']:.1%})")
        print(f"  Best high-conviction returns:   {best_hc_med[0]} (Med 1Y={best_hc_med[1]['hc_med']:+.1%})")

    print(f"\n{'='*110}")


if __name__ == "__main__":
    run_backtest()
