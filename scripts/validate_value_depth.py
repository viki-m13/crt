#!/usr/bin/env python3
"""
Validate that the value_depth scoring system identifies stocks that genuinely
rebound with superior returns — not just DCA gains.

Methodology:
  For each stock with ≥5 years of history, compute value_depth at every
  historical point.  Split into quintiles (Q1=cheapest … Q5=most expensive).
  Measure ACTUAL forward 1Y & 3Y returns for each quintile.

  If the model works, Q1 (high value_depth) should show:
    1. Higher hit rate (% positive) than Q5 and the baseline
    2. Higher median returns than Q5 and SPY
    3. Meaningful alpha (not just broad market gains)

Run:
    python scripts/validate_value_depth.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from daily_scan import (
    compute_core_features,
    compute_idiosyncratic_features,
    compute_washout_meter,
    compute_value_depth,
    compute_bottom_confirmation,
    forward_returns,
    LB_LT, LB_ST, HORIZONS_DAYS,
    fetch_ishares_holdings_tickers,
    ISHARES_HOLDINGS_URL,
)

# =========================
# Config
# =========================
MIN_BARS = 1260          # need ~5Y of history
SAMPLE_EVERY = 21        # sample monthly (avoid autocorrelation)
SPY_TICKER = "SPY"
QUINTILE_LABELS = ["Q1_Deep_Value", "Q2_Value", "Q3_Neutral", "Q4_Expensive", "Q5_Rich"]

# =========================
# Data fetching (reuse yfinance)
# =========================
def fetch_price_data(tickers: list) -> dict:
    """Fetch 10Y daily OHLCV for all tickers + SPY."""
    import yfinance as yf
    all_tickers = list(set(tickers + [SPY_TICKER]))
    print(f"[FETCH] Downloading {len(all_tickers)} tickers (10Y) …")
    raw = yf.download(all_tickers, period="10y", interval="1d", group_by="ticker", threads=True, progress=True)
    return raw

def extract_series(raw, ticker: str, field: str) -> pd.Series:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            return raw[(ticker, field)].dropna()
        return raw[field].dropna()
    except (KeyError, TypeError):
        return pd.Series(dtype=float)


# =========================
# Main validation
# =========================
def run_validation():
    # Fetch universe
    tickers = fetch_ishares_holdings_tickers(ISHARES_HOLDINGS_URL)
    if not tickers:
        print("[ERROR] Could not fetch ticker universe")
        return

    # Cap for feasibility
    tickers = tickers[:200]
    raw = fetch_price_data(tickers)

    spy_close = extract_series(raw, SPY_TICKER, "Adj Close")
    if spy_close.empty:
        spy_close = extract_series(raw, SPY_TICKER, "Close")
    if spy_close.empty:
        print("[ERROR] Could not get SPY data")
        return

    spy_close.index = pd.to_datetime(spy_close.index, utc=True)
    spy_high = extract_series(raw, SPY_TICKER, "High")
    spy_high.index = pd.to_datetime(spy_high.index, utc=True)
    spy_low = extract_series(raw, SPY_TICKER, "Low")
    spy_low.index = pd.to_datetime(spy_low.index, utc=True)

    # Compute SPY forward returns for benchmark
    spy_fwd = pd.DataFrame(index=spy_close.index)
    for name, days in HORIZONS_DAYS.items():
        spy_fwd[f"spy_fwd_{name}"] = (spy_close.shift(-days) / spy_close - 1.0).replace([np.inf, -np.inf], np.nan)

    # Collect all observations: (value_depth, washout, fwd_1Y, fwd_3Y, spy_fwd_1Y, spy_fwd_3Y)
    all_obs = []

    for i, t in enumerate(tickers):
        if t == SPY_TICKER:
            continue

        px = extract_series(raw, t, "Adj Close")
        if px.empty:
            px = extract_series(raw, t, "Close")
        high = extract_series(raw, t, "High")
        low = extract_series(raw, t, "Low")
        vol = extract_series(raw, t, "Volume")
        opn = extract_series(raw, t, "Open")

        if len(px) < MIN_BARS:
            continue

        # Build features
        df = pd.DataFrame({
            "open": opn, "high": high, "low": low, "close": px, "volume": vol, "px": px,
        }).dropna()
        df.index = pd.to_datetime(df.index, utc=True)

        if len(df) < MIN_BARS:
            continue

        feat = compute_core_features(df[["open", "high", "low", "close", "volume"]])
        feat["px"] = df["px"]

        # Align with SPY
        idx = feat.index.intersection(spy_close.index)
        if len(idx) < MIN_BARS:
            continue
        feat = feat.reindex(idx)
        spy_aligned = spy_close.reindex(idx)

        # Idiosyncratic features
        idio = compute_idiosyncratic_features(feat["px"], spy_aligned)
        feat = feat.join(idio, how="left")
        feat["bottom_confirm"] = compute_bottom_confirmation(feat["dd_lt"], feat["pos_lt"])

        # Value depth + washout
        feat["washout_meter"] = compute_washout_meter(feat)
        feat["value_depth"] = compute_value_depth(feat)

        # Forward returns
        fwd = forward_returns(feat["px"])
        feat = feat.join(fwd, how="left")

        # Sample monthly to avoid autocorrelation
        valid = feat.dropna(subset=["value_depth", "fwd_1Y"])
        if len(valid) < 100:
            continue

        sampled = valid.iloc[::SAMPLE_EVERY]

        for ts, row in sampled.iterrows():
            spy_row = spy_fwd.loc[ts] if ts in spy_fwd.index else None
            obs = {
                "ticker": t,
                "date": ts,
                "value_depth": float(row["value_depth"]),
                "washout": float(row["washout_meter"]) if np.isfinite(row["washout_meter"]) else 0,
                "fwd_1Y": float(row["fwd_1Y"]),
                "fwd_3Y": float(row.get("fwd_3Y", np.nan)),
                "spy_fwd_1Y": float(spy_row[f"spy_fwd_1Y"]) if spy_row is not None and f"spy_fwd_1Y" in spy_row else np.nan,
                "spy_fwd_3Y": float(spy_row[f"spy_fwd_3Y"]) if spy_row is not None and f"spy_fwd_3Y" in spy_row else np.nan,
            }
            all_obs.append(obs)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(tickers)}] processed {t}, total obs={len(all_obs)}")

    print(f"\n[DATA] Total observations: {len(all_obs)}")

    if len(all_obs) < 500:
        print("[ERROR] Not enough observations for meaningful analysis")
        return

    df_obs = pd.DataFrame(all_obs)

    # ========== ANALYSIS ==========
    print("\n" + "=" * 100)
    print("VALIDATION: VALUE DEPTH SCORING — DO HIGH-SCORING STOCKS ACTUALLY REBOUND?")
    print("=" * 100)

    # --- Quintile analysis ---
    try:
        df_obs["quintile"] = pd.qcut(df_obs["value_depth"], 5, labels=QUINTILE_LABELS, duplicates="drop")
    except ValueError:
        # If too many duplicate edges, use rank-based quintiles
        df_obs["quintile"] = pd.qcut(df_obs["value_depth"].rank(method="first"), 5, labels=QUINTILE_LABELS)

    print(f"\n{'Quintile':<20} {'N':>6} {'Avg VD':>8} {'Hit 1Y':>8} {'Med 1Y':>9} {'Avg 1Y':>9} {'Hit 3Y':>8} {'Med 3Y':>9} {'Avg 3Y':>9}")
    print("-" * 100)

    for q in QUINTILE_LABELS:
        sub = df_obs[df_obs["quintile"] == q]
        if len(sub) == 0:
            continue
        n = len(sub)
        avg_vd = sub["value_depth"].mean()
        hit_1y = (sub["fwd_1Y"] > 0).mean() * 100
        med_1y = sub["fwd_1Y"].median() * 100
        avg_1y = sub["fwd_1Y"].mean() * 100
        sub_3y = sub.dropna(subset=["fwd_3Y"])
        hit_3y = (sub_3y["fwd_3Y"] > 0).mean() * 100 if len(sub_3y) > 0 else np.nan
        med_3y = sub_3y["fwd_3Y"].median() * 100 if len(sub_3y) > 0 else np.nan
        avg_3y = sub_3y["fwd_3Y"].mean() * 100 if len(sub_3y) > 0 else np.nan
        print(f"{q:<20} {n:>6} {avg_vd:>7.1f} {hit_1y:>7.1f}% {med_1y:>+8.1f}% {avg_1y:>+8.1f}% {hit_3y:>7.1f}% {med_3y:>+8.1f}% {avg_3y:>+8.1f}%")

    # --- SPY comparison (alpha) ---
    print(f"\n{'':.<100}")
    print("VS SPY (Alpha = stock return − SPY return over same period)")
    print(f"{'':.<100}")

    valid_spy = df_obs.dropna(subset=["spy_fwd_1Y"])
    valid_spy["alpha_1Y"] = valid_spy["fwd_1Y"] - valid_spy["spy_fwd_1Y"]

    valid_spy_3y = df_obs.dropna(subset=["spy_fwd_3Y", "fwd_3Y"])
    valid_spy_3y["alpha_3Y"] = valid_spy_3y["fwd_3Y"] - valid_spy_3y["spy_fwd_3Y"]

    print(f"\n{'Quintile':<20} {'N':>6} {'Med Alpha 1Y':>14} {'Avg Alpha 1Y':>14} {'Beat SPY 1Y':>13} {'Med Alpha 3Y':>14} {'Beat SPY 3Y':>13}")
    print("-" * 100)

    for q in QUINTILE_LABELS:
        sub1 = valid_spy[valid_spy["quintile"] == q]
        sub3 = valid_spy_3y[valid_spy_3y["quintile"] == q]
        if len(sub1) == 0:
            continue
        n = len(sub1)
        med_a1 = sub1["alpha_1Y"].median() * 100
        avg_a1 = sub1["alpha_1Y"].mean() * 100
        beat1 = (sub1["alpha_1Y"] > 0).mean() * 100
        med_a3 = sub3["alpha_3Y"].median() * 100 if len(sub3) > 0 else np.nan
        beat3 = (sub3["alpha_3Y"] > 0).mean() * 100 if len(sub3) > 0 else np.nan
        print(f"{q:<20} {n:>6} {med_a1:>+13.1f}% {avg_a1:>+13.1f}% {beat1:>12.1f}% {med_a3:>+13.1f}% {beat3:>12.1f}%")

    # --- High value depth threshold analysis ---
    print(f"\n{'':.<100}")
    print("HIGH VALUE DEPTH THRESHOLD ANALYSIS (absolute thresholds)")
    print(f"{'':.<100}")

    thresholds = [60, 50, 40, 30, 20]
    print(f"\n{'Threshold':<12} {'N':>6} {'Hit 1Y':>8} {'Med 1Y':>9} {'Med Alpha 1Y':>14} {'Beat SPY 1Y':>13} {'Hit 3Y':>8} {'Med 3Y':>9}")
    print("-" * 100)

    baseline_1y = df_obs["fwd_1Y"].median() * 100

    for thr in thresholds:
        sub = df_obs[df_obs["value_depth"] >= thr]
        if len(sub) < 50:
            continue
        n = len(sub)
        hit_1y = (sub["fwd_1Y"] > 0).mean() * 100
        med_1y = sub["fwd_1Y"].median() * 100

        sub_spy = valid_spy[valid_spy["value_depth"] >= thr]
        med_a1 = sub_spy["alpha_1Y"].median() * 100 if len(sub_spy) > 0 else np.nan
        beat1 = (sub_spy["alpha_1Y"] > 0).mean() * 100 if len(sub_spy) > 0 else np.nan

        sub_3y = sub.dropna(subset=["fwd_3Y"])
        hit_3y = (sub_3y["fwd_3Y"] > 0).mean() * 100 if len(sub_3y) > 0 else np.nan
        med_3y = sub_3y["fwd_3Y"].median() * 100 if len(sub_3y) > 0 else np.nan

        print(f"VD ≥ {thr:<5} {n:>6} {hit_1y:>7.1f}% {med_1y:>+8.1f}% {med_a1:>+13.1f}% {beat1:>12.1f}% {hit_3y:>7.1f}% {med_3y:>+8.1f}%")

    print(f"{'ALL (baseline)':<12} {len(df_obs):>6} {(df_obs['fwd_1Y'] > 0).mean()*100:>7.1f}% {baseline_1y:>+8.1f}%")

    # --- Washout-only comparison (does value_depth beat simple washout?) ---
    print(f"\n{'':.<100}")
    print("VALUE DEPTH vs WASHOUT-ONLY (Q1 comparison)")
    print(f"{'':.<100}")

    try:
        df_obs["wash_quintile"] = pd.qcut(df_obs["washout"], 5, labels=False, duplicates="drop")
    except ValueError:
        df_obs["wash_quintile"] = pd.qcut(df_obs["washout"].rank(method="first"), 5, labels=False)
    q1_vd = df_obs[df_obs["quintile"] == "Q1_Deep_Value"]
    q1_wash = df_obs[df_obs["wash_quintile"] == 4]  # highest washout quintile

    for label, sub in [("Value Depth Q1", q1_vd), ("Washout Q5 (highest)", q1_wash)]:
        if len(sub) == 0:
            continue
        hit_1y = (sub["fwd_1Y"] > 0).mean() * 100
        med_1y = sub["fwd_1Y"].median() * 100
        sub_spy = sub.merge(valid_spy[["alpha_1Y"]], left_index=True, right_index=True, how="inner", suffixes=("", "_spy"))
        alpha_col = "alpha_1Y_spy" if "alpha_1Y_spy" in sub_spy.columns else "alpha_1Y"
        if alpha_col in sub_spy.columns:
            med_a = sub_spy[alpha_col].median() * 100
        else:
            med_a = np.nan
        print(f"  {label:<30} N={len(sub):>5}  Hit 1Y={hit_1y:.1f}%  Med 1Y={med_1y:+.1f}%  Med Alpha 1Y={med_a:+.1f}%")

    # --- Washout gate analysis (the actual production approach) ---
    print(f"\n{'':.<100}")
    print("WASHOUT GATE ANALYSIS (production approach: quality × win_rate × gate)")
    print("Does filtering to only stocks IN a pullback (washout > threshold) improve outcomes?")
    print(f"{'':.<100}")

    wash_thresholds = [5, 10, 20, 30, 40]
    print(f"\n{'Washout Gate':<14} {'N':>6} {'Hit 1Y':>8} {'Med 1Y':>9} {'Avg 1Y':>9} {'Hit 3Y':>8} {'Med 3Y':>9}")
    print("-" * 80)

    for thr in wash_thresholds:
        sub = df_obs[df_obs["washout"] >= thr]
        if len(sub) < 50:
            continue
        n = len(sub)
        hit_1y = (sub["fwd_1Y"] > 0).mean() * 100
        med_1y = sub["fwd_1Y"].median() * 100
        avg_1y = sub["fwd_1Y"].mean() * 100
        sub_3y = sub.dropna(subset=["fwd_3Y"])
        hit_3y = (sub_3y["fwd_3Y"] > 0).mean() * 100 if len(sub_3y) > 0 else np.nan
        med_3y = sub_3y["fwd_3Y"].median() * 100 if len(sub_3y) > 0 else np.nan
        print(f"wash ≥ {thr:<6} {n:>6} {hit_1y:>7.1f}% {med_1y:>+8.1f}% {avg_1y:>+8.1f}% {hit_3y:>7.1f}% {med_3y:>+8.1f}%")

    no_wash = df_obs[df_obs["washout"] < 5]
    if len(no_wash) > 0:
        hit_1y = (no_wash["fwd_1Y"] > 0).mean() * 100
        med_1y = no_wash["fwd_1Y"].median() * 100
        avg_1y = no_wash["fwd_1Y"].mean() * 100
        no_3y = no_wash.dropna(subset=["fwd_3Y"])
        hit_3y = (no_3y["fwd_3Y"] > 0).mean() * 100 if len(no_3y) > 0 else np.nan
        med_3y = no_3y["fwd_3Y"].median() * 100 if len(no_3y) > 0 else np.nan
        print(f"wash < 5       {len(no_wash):>6} {hit_1y:>7.1f}% {med_1y:>+8.1f}% {avg_1y:>+8.1f}% {hit_3y:>7.1f}% {med_3y:>+8.1f}%")

    print(f"{'ALL':<14} {len(df_obs):>6} {(df_obs['fwd_1Y'] > 0).mean()*100:>7.1f}% {df_obs['fwd_1Y'].median()*100:>+8.1f}% {df_obs['fwd_1Y'].mean()*100:>+8.1f}%")

    # --- Washout quintile analysis ---
    print(f"\n{'':.<100}")
    print("WASHOUT QUINTILE ANALYSIS")
    print(f"{'':.<100}")

    wash_labels = ["W1_No_Pullback", "W2_Mild", "W3_Moderate", "W4_Significant", "W5_Deep_Pullback"]
    try:
        df_obs["wash_q"] = pd.qcut(df_obs["washout"].rank(method="first"), 5, labels=wash_labels)
    except ValueError:
        df_obs["wash_q"] = "unknown"

    print(f"\n{'Quintile':<20} {'N':>6} {'Avg Wash':>10} {'Hit 1Y':>8} {'Med 1Y':>9} {'Hit 3Y':>8} {'Med 3Y':>9}")
    print("-" * 80)

    for q in wash_labels:
        sub = df_obs[df_obs["wash_q"] == q]
        if len(sub) == 0:
            continue
        n = len(sub)
        avg_w = sub["washout"].mean()
        hit_1y = (sub["fwd_1Y"] > 0).mean() * 100
        med_1y = sub["fwd_1Y"].median() * 100
        sub_3y = sub.dropna(subset=["fwd_3Y"])
        hit_3y = (sub_3y["fwd_3Y"] > 0).mean() * 100 if len(sub_3y) > 0 else np.nan
        med_3y = sub_3y["fwd_3Y"].median() * 100 if len(sub_3y) > 0 else np.nan
        print(f"{q:<20} {n:>6} {avg_w:>9.1f} {hit_1y:>7.1f}% {med_1y:>+8.1f}% {hit_3y:>7.1f}% {med_3y:>+8.1f}%")

    # --- Summary verdict ---
    print(f"\n{'=' * 100}")
    q1 = df_obs[df_obs["quintile"] == "Q1_Deep_Value"]
    q5 = df_obs[df_obs["quintile"] == "Q5_Rich"]
    q1_med = q1["fwd_1Y"].median() * 100 if len(q1) > 0 else 0
    q5_med = q5["fwd_1Y"].median() * 100 if len(q5) > 0 else 0
    spread = q1_med - q5_med

    w5 = df_obs[df_obs["wash_q"] == "W5_Deep_Pullback"] if "wash_q" in df_obs.columns else pd.DataFrame()
    w1 = df_obs[df_obs["wash_q"] == "W1_No_Pullback"] if "wash_q" in df_obs.columns else pd.DataFrame()
    w5_med = w5["fwd_1Y"].median() * 100 if len(w5) > 0 else 0
    w1_med = w1["fwd_1Y"].median() * 100 if len(w1) > 0 else 0
    wash_spread = w5_med - w1_med

    print(f"SUMMARY:")
    print(f"  Value Depth: Q1 (deep) med 1Y = {q1_med:+.1f}%, Q5 (rich) med 1Y = {q5_med:+.1f}%, spread = {spread:+.1f}%")
    print(f"  Washout:     W5 (deep) med 1Y = {w5_med:+.1f}%, W1 (none) med 1Y = {w1_med:+.1f}%, spread = {wash_spread:+.1f}%")
    print()
    print("  Key insight: The pullback gate (washout) serves as a FILTER to focus on")
    print("  recovery setups. The analog engine provides the predictive power (win rates")
    print("  from matched historical situations). Quality score filters value traps.")
    print("  The gated Opp Score = quality × analog_win_rate × pullback_gate.")
    print("=" * 100)


if __name__ == "__main__":
    run_validation()
