#!/usr/bin/env python3
"""
Backtest: Quality Score weight in the Opportunity Score formula.

PRODUCTION formula (adopted Feb 2026):
    OppScore = Quality × 1Y_Win_Rate × Pullback_Gate(washout)
    WITH quality ≥ 50 gate (stocks below 50 get no Opp Score)

This gate was adopted based on backtest results showing:
    - P5 during pullbacks: -20% (vs -42% without the gate)
    - Hit rate maintained: 83%
    - Only 17% of signals dropped — the dangerous ones

We also compare against alternative formulas for reference:
  A) OLD (no gate): Quality × Win × Gate
  B) Q-squared: (Q/100)² × Win × Gate
  C) Q^1.5: moderate emphasis
  D) Gate≥60: stricter gate
  E) No quality: baseline
  F) PRODUCTION: Quality × Win × Gate with quality≥50 gate

Run:
    python scripts/backtest_quality_weight.py
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
    trend_quality,
    recovery_track_record,
    selling_deceleration,
    pullback_gate,
    safe_float,
    robust_z,
    LB_LT, LB_ST, BETA_LB,
    ANALOG_K, ANALOG_MIN, ANALOG_MIN_SEP_DAYS,
    MIN_HISTORY_BARS,
    BENCH,
    HORIZON_WEIGHTS,
)

# =========================
# CONFIG
# =========================
TEST_TICKERS = [
    # Reliable growers / recoverers
    "AAPL", "MSFT", "JPM", "JNJ", "HD", "UNH", "COST", "GOOGL", "AMZN",
    # High-growth / volatile
    "NVDA", "META", "TSLA", "CRM", "NFLX", "AMD", "AVGO",
    # Newer / shorter history
    "UBER", "CRWD", "DDOG", "ZS", "NET", "PLTR",
    # Cyclicals / value traps / mixed recovery
    "XOM", "CVX", "BA", "GS", "CAT",
    # Pharma / biotech (mixed)
    "LLY", "ABBV", "MRK",
    # REITs / other
    "AMT", "PG", "WMT", "KO",
]

K = 250
SAMPLE_EVERY = 42           # ~bimonthly (denser than quarterly for more data)
MIN_EVAL_BARS = LB_LT + BETA_LB + LB_ST + 252 + 252

FEATURE_COLS = [
    "dd_lt", "pos_lt", "dd_st", "pos_st",
    "atr_pct", "volu_z", "gap", "trend_st",
    "idio_dd_lt", "idio_pos_lt", "idio_dd_st", "idio_pos_st",
    "mkt_trend", "mkt_vol", "mkt_dd", "mkt_atr_pct",
]
ZWIN = max(63, LB_ST)


# =========================
# Opp Score Formula Variants
# =========================
def opp_A_old_no_gate(quality: float, win_1y: float, washout: float) -> float:
    """OLD formula (no quality gate): Quality × Win × Gate"""
    return quality * win_1y * pullback_gate(washout)


def opp_B_quality_sq(quality: float, win_1y: float, washout: float) -> float:
    """Quality squared: (Q/100)² × 100 × Win × Gate — punishes low quality hard"""
    q_norm = np.clip(quality / 100.0, 0.0, 1.0)
    return (q_norm ** 2) * 100 * win_1y * pullback_gate(washout)


def opp_C_quality_1p5(quality: float, win_1y: float, washout: float) -> float:
    """Quality^1.5: moderate emphasis"""
    q_norm = np.clip(quality / 100.0, 0.0, 1.0)
    return (q_norm ** 1.5) * 100 * win_1y * pullback_gate(washout)


def opp_D_quality_gate60(quality: float, win_1y: float, washout: float) -> float:
    """Stricter gate: drops to NaN if quality < 60"""
    if quality < 60:
        return np.nan
    return quality * win_1y * pullback_gate(washout)


def opp_E_no_quality(quality: float, win_1y: float, washout: float) -> float:
    """No quality at all: just Win × Gate × 100"""
    return 100 * win_1y * pullback_gate(washout)


def opp_F_production(quality: float, win_1y: float, washout: float) -> float:
    """PRODUCTION formula: Q×W×G with quality≥50 gate (adopted Feb 2026)"""
    if quality < 50:
        return np.nan
    return quality * win_1y * pullback_gate(washout)


FORMULAS = {
    "A: Old (Q×W×G, no gate)":       opp_A_old_no_gate,
    "B: Q²×W×G (sq emphasis)":       opp_B_quality_sq,
    "C: Q^1.5×W×G (moderate)":       opp_C_quality_1p5,
    "D: Q×W×G + gate≥60":            opp_D_quality_gate60,
    "E: No quality (W×G only)":       opp_E_no_quality,
    "F: PRODUCTION (Q×W×G gate≥50)":  opp_F_production,
}


# =========================
# Quality computation (no look-ahead)
# =========================
def compute_quality_at(px_up_to: pd.Series) -> float:
    """Compute quality score using only data up to a point."""
    parts = {}
    weights = {"trend": 0.45, "recovery": 0.35, "momentum": 0.20}

    tq = trend_quality(px_up_to)
    if np.isfinite(tq):
        parts["trend"] = tq

    rtr = recovery_track_record(px_up_to)
    rr = rtr.get("recovery_rate", np.nan)
    if np.isfinite(rr):
        parts["recovery"] = rr * 100

    sd = selling_deceleration(px_up_to)
    if np.isfinite(sd):
        parts["momentum"] = sd

    if not parts:
        return np.nan

    total_w = sum(weights[k] for k in parts)
    return sum(parts[k] * weights[k] for k in parts) / total_w


# =========================
# Main backtest
# =========================
def run_backtest():
    all_tickers = sorted(set(TEST_TICKERS + [BENCH]))
    print(f"[BACKTEST: Quality in Opp Score] Downloading data for {len(all_tickers)} tickers ...")

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
    spy_px, spy_h, spy_l = spy_px.reindex(common), spy_h.reindex(common), spy_l.reindex(common)
    mkt = compute_market_regime(spy_px, spy_h, spy_l)

    print(f"[DATA] SPY: {len(spy_px)} bars, {spy_px.index[0].date()} to {spy_px.index[-1].date()}")

    # Build features for each ticker
    ticker_data = {}

    for i, t in enumerate(sorted(TEST_TICKERS)):
        if t not in C.columns or t == BENCH:
            print(f"  [{i+1:>2}/{len(TEST_TICKERS)}] SKIP {t} — not in data")
            continue

        df = pd.DataFrame({
            "open": O[t], "high": H[t], "low": L[t],
            "close": C[t], "volume": V[t], "px": PX[t],
        }).dropna(subset=["open", "high", "low", "close", "volume", "px"])

        if len(df) < MIN_HISTORY_BARS:
            print(f"  [{i+1:>2}/{len(TEST_TICKERS)}] SKIP {t} — {len(df)} bars")
            continue

        feat = compute_core_features(df[["open", "high", "low", "close", "volume"]])
        feat["px"] = df["px"]

        idx = feat.index.intersection(spy_px.index).intersection(mkt.index)
        if len(idx) < MIN_HISTORY_BARS:
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
        yrs = len(feat) / 252
        print(f"  [{i+1:>2}/{len(TEST_TICKERS)}] OK   {t:<6} {len(feat):>5} bars ({yrs:.1f}Y)")

    n_ok = len(ticker_data)
    if n_ok < 5:
        print(f"[ERROR] Only {n_ok} tickers. Aborting.")
        return

    print(f"\n[POOL] {n_ok} tickers ready")

    # =========================
    # Run evaluation
    # =========================
    print(f"\n{'='*120}")
    print(f"RUNNING BACKTEST — sampling every {SAMPLE_EVERY} bars, k={K}")
    print(f"{'='*120}")

    results = []
    t0 = time.time()

    for ti, (t, td) in enumerate(ticker_data.items()):
        feat = td["feat"]
        X = td["X"]
        regimes = td["regimes"]
        y_1Y = td["y_1Y"]

        ok = X.notna().all(axis=1) & y_1Y.notna()
        ok_idx = X.index[ok]
        if len(ok_idx) == 0:
            continue
        min_start = ok_idx[0] + pd.Timedelta(days=MIN_EVAL_BARS)
        eval_idx = ok_idx[ok_idx >= min_start]
        eval_idx = eval_idx[::SAMPLE_EVERY]

        print(f"\n  [{ti+1}/{n_ok}] {t} ({len(eval_idx)} eval pts) ...", end="", flush=True)

        for now_idx in eval_idx:
            actual_1y = float(y_1Y.loc[now_idx])
            actual_positive = 1 if actual_1y > 0 else 0
            washout = safe_float(feat.loc[now_idx, "washout_meter"])
            if not np.isfinite(washout):
                washout = 0.0

            # Quality: no look-ahead
            px_up_to = feat.loc[:now_idx, "px"].dropna()
            quality = compute_quality_at(px_up_to)
            if not np.isfinite(quality):
                continue  # skip if quality can't be computed

            # 1Y analog win-rate
            y = feat.get("fwd_1Y", None)
            if y is None:
                continue
            analog_idx = select_analogs_regime_balanced(
                X, y, regimes, now_idx, k=K, min_sep_days=ANALOG_MIN_SEP_DAYS)
            if len(analog_idx) < ANALOG_MIN:
                continue
            vals = y.loc[analog_idx].dropna().values.astype(float)
            vals = np.where(np.isnan(vals), -1.0, vals)
            if len(vals) < ANALOG_MIN:
                continue
            win_1y = float(np.mean(vals > 0))
            med_analog = float(np.median(vals))
            p10_analog = float(np.quantile(vals, 0.10))

            # Compute opp score for each formula variant
            row = {
                "ticker": t,
                "date": now_idx,
                "actual_1y": actual_1y,
                "actual_pos": actual_positive,
                "washout": washout,
                "quality": quality,
                "win_1y": win_1y,
                "med_analog": med_analog,
                "p10_analog": p10_analog,
                "gate": pullback_gate(washout),
            }

            for fname, func in FORMULAS.items():
                row[f"opp_{fname}"] = func(quality, win_1y, washout)

            results.append(row)

        print(f" done ({time.time() - t0:.0f}s elapsed)")

    df = pd.DataFrame(results)
    print(f"\n[DATA] Total evaluation points: {len(df)}")
    print(f"  With washout ≥ 15: {len(df[df['washout'] >= 15])}")
    print(f"  With washout ≥ 30: {len(df[df['washout'] >= 30])}")

    if len(df) < 50:
        print("[ERROR] Not enough evaluation points")
        return

    opp_cols = {fname: f"opp_{fname}" for fname in FORMULAS}

    # ==========================================================================
    # 1. RAW QUALITY SIGNAL — does quality alone predict recovery?
    # ==========================================================================
    print(f"\n{'='*120}")
    print("1. RAW QUALITY SIGNAL: Does quality alone predict actual 1Y recovery?")
    print(f"{'='*120}")

    q_bins = [(0, 40, "Low (0-40)"), (40, 55, "Med-Low (40-55)"),
              (55, 70, "Med (55-70)"), (70, 85, "Med-High (70-85)"),
              (85, 101, "High (85-100)")]

    print(f"\n  ALL eval points:")
    print(f"  {'Quality':<18} {'N':>6} {'Hit':>8} {'Med1Y':>10} {'Avg1Y':>10} {'P10':>10} {'P5':>10}")
    print(f"  {'-'*75}")
    for lo, hi, label in q_bins:
        sub = df[(df["quality"] >= lo) & (df["quality"] < hi)]
        if len(sub) < 10:
            print(f"  {label:<18} {len(sub):>6}   (too few)")
            continue
        ah = sub["actual_pos"].mean()
        mr = sub["actual_1y"].median()
        ar = sub["actual_1y"].mean()
        p10 = np.quantile(sub["actual_1y"].values, 0.10)
        p5 = np.quantile(sub["actual_1y"].values, 0.05)
        print(f"  {label:<18} {len(sub):>6} {ah:>7.1%} {mr:>+9.1%} {ar:>+9.1%} {p10:>+9.1%} {p5:>+9.1%}")

    # Pullback-only
    pb = df[df["washout"] >= 15]
    if len(pb) >= 50:
        print(f"\n  PULLBACK ONLY (washout ≥ 15, N={len(pb)}):")
        print(f"  {'Quality':<18} {'N':>6} {'Hit':>8} {'Med1Y':>10} {'Avg1Y':>10} {'P10':>10} {'P5':>10}")
        print(f"  {'-'*75}")
        for lo, hi, label in q_bins:
            sub = pb[(pb["quality"] >= lo) & (pb["quality"] < hi)]
            if len(sub) < 5:
                print(f"  {label:<18} {len(sub):>6}   (too few)")
                continue
            ah = sub["actual_pos"].mean()
            mr = sub["actual_1y"].median()
            ar = sub["actual_1y"].mean()
            p10 = np.quantile(sub["actual_1y"].values, 0.10)
            p5 = np.quantile(sub["actual_1y"].values, 0.05)
            print(f"  {label:<18} {len(sub):>6} {ah:>7.1%} {mr:>+9.1%} {ar:>+9.1%} {p10:>+9.1%} {p5:>+9.1%}")

    # ==========================================================================
    # 2. QUALITY × WASHOUT INTERACTION
    # ==========================================================================
    print(f"\n{'='*120}")
    print("2. QUALITY × WASHOUT INTERACTION: Does quality matter more during pullbacks?")
    print(f"{'='*120}")

    for wash_label, wash_lo, wash_hi in [("No pullback (0-10)", 0, 10),
                                          ("Mild (10-25)", 10, 25),
                                          ("Moderate (25-45)", 25, 45),
                                          ("Deep (45+)", 45, 101)]:
        w_sub = df[(df["washout"] >= wash_lo) & (df["washout"] < wash_hi)]
        if len(w_sub) < 20:
            continue
        print(f"\n  {wash_label} (N={len(w_sub)})")
        print(f"  {'Quality':<18} {'N':>6} {'Hit':>8} {'Med1Y':>10} {'P10':>10}")
        print(f"  {'-'*55}")
        for lo, hi, qlabel in q_bins:
            sub = w_sub[(w_sub["quality"] >= lo) & (w_sub["quality"] < hi)]
            if len(sub) < 5:
                continue
            ah = sub["actual_pos"].mean()
            mr = sub["actual_1y"].median()
            p10 = np.quantile(sub["actual_1y"].values, 0.10)
            print(f"  {qlabel:<18} {len(sub):>6} {ah:>7.1%} {mr:>+9.1%} {p10:>+9.1%}")

    # ==========================================================================
    # 3. OPP SCORE QUINTILE CALIBRATION
    # ==========================================================================
    print(f"\n{'='*120}")
    print("3. OPP SCORE QUINTILE CALIBRATION: Does higher Opp Score → better outcomes?")
    print(f"{'='*120}")

    for fname, col in opp_cols.items():
        valid = df[df[col].notna()].copy()
        if len(valid) < 50:
            continue
        print(f"\n  {fname} (N={len(valid)})")
        try:
            valid["q"] = pd.qcut(valid[col], 5,
                                 labels=["Q1_Low", "Q2", "Q3", "Q4", "Q5_High"],
                                 duplicates="drop")
        except ValueError:
            valid["q"] = pd.qcut(valid[col].rank(method="first"), 5,
                                 labels=["Q1_Low", "Q2", "Q3", "Q4", "Q5_High"])

        print(f"  {'Quintile':<12} {'N':>5} {'Med Opp':>10} {'Hit':>8} {'Med1Y':>10} {'Avg1Y':>10} {'P10':>10}")
        print(f"  {'-'*70}")
        for q in ["Q1_Low", "Q2", "Q3", "Q4", "Q5_High"]:
            sub = valid[valid["q"] == q]
            if len(sub) == 0:
                continue
            mopp = sub[col].median()
            ah = sub["actual_pos"].mean()
            mr = sub["actual_1y"].median()
            ar = sub["actual_1y"].mean()
            p10 = np.quantile(sub["actual_1y"].values, 0.10)
            print(f"  {q:<12} {len(sub):>5} {mopp:>9.1f} {ah:>7.1%} {mr:>+9.1%} {ar:>+9.1%} {p10:>+9.1%}")

    # ==========================================================================
    # 4. HEAD-TO-HEAD: Top-N signals by each formula → actual outcomes
    # ==========================================================================
    print(f"\n{'='*120}")
    print("4. HEAD-TO-HEAD: Top 10%/20% signals by each formula → actual outcomes")
    print(f"   (Which formula's top picks produce the best real returns?)")
    print(f"{'='*120}")

    for pct_label, pct in [("Top 10%", 0.90), ("Top 20%", 0.80), ("Top 30%", 0.70)]:
        print(f"\n  {pct_label}:")
        print(f"  {'Formula':<34} {'N':>6} {'Hit':>8} {'Med1Y':>10} {'Avg1Y':>10} {'P10':>10} {'P5':>10}")
        print(f"  {'-'*85}")
        for fname, col in opp_cols.items():
            valid = df[df[col].notna()]
            if len(valid) < 20:
                continue
            thr = valid[col].quantile(pct)
            sub = valid[valid[col] >= thr]
            if len(sub) < 10:
                continue
            ah = sub["actual_pos"].mean()
            mr = sub["actual_1y"].median()
            ar = sub["actual_1y"].mean()
            p10 = np.quantile(sub["actual_1y"].values, 0.10)
            p5 = np.quantile(sub["actual_1y"].values, 0.05)
            print(f"  {fname:<34} {len(sub):>6} {ah:>7.1%} {mr:>+9.1%} {ar:>+9.1%} {p10:>+9.1%} {p5:>+9.1%}")

    # ==========================================================================
    # 5. PULLBACK SIGNALS ONLY: formulas compared during real pullbacks
    # ==========================================================================
    print(f"\n{'='*120}")
    print("5. PULLBACK SIGNALS: Formulas compared at washout ≥ 20")
    print(f"{'='*120}")

    pb20 = df[df["washout"] >= 20]
    if len(pb20) >= 30:
        for pct_label, pct in [("Top 25%", 0.75), ("Top 50%", 0.50)]:
            print(f"\n  {pct_label} of pullback signals (wash≥20, N_total={len(pb20)}):")
            print(f"  {'Formula':<34} {'N':>6} {'Hit':>8} {'Med1Y':>10} {'Avg1Y':>10} {'P10':>10} {'P5':>10}")
            print(f"  {'-'*85}")
            for fname, col in opp_cols.items():
                valid = pb20[pb20[col].notna()]
                if len(valid) < 10:
                    continue
                thr = valid[col].quantile(pct)
                sub = valid[valid[col] >= thr]
                if len(sub) < 5:
                    continue
                ah = sub["actual_pos"].mean()
                mr = sub["actual_1y"].median()
                ar = sub["actual_1y"].mean()
                p10 = np.quantile(sub["actual_1y"].values, 0.10)
                p5 = np.quantile(sub["actual_1y"].values, 0.05)
                print(f"  {fname:<34} {len(sub):>6} {ah:>7.1%} {mr:>+9.1%} {ar:>+9.1%} {p10:>+9.1%} {p5:>+9.1%}")

    # ==========================================================================
    # 6. THE KEY COMPARISON: High-quality pullback vs low-quality pullback
    # ==========================================================================
    print(f"\n{'='*120}")
    print("6. THE KEY TEST: High-quality pullbacks vs low-quality pullbacks")
    print(f"   (Same washout conditions, different quality — does quality predict recovery?)")
    print(f"{'='*120}")

    for wash_min in [15, 25, 40]:
        pb_sub = df[df["washout"] >= wash_min]
        if len(pb_sub) < 20:
            continue
        med_q = pb_sub["quality"].median()
        hi_q = pb_sub[pb_sub["quality"] >= med_q]
        lo_q = pb_sub[pb_sub["quality"] < med_q]

        print(f"\n  Washout ≥ {wash_min} (N={len(pb_sub)}, median quality={med_q:.1f})")
        for label, sub in [("HIGH quality (above median)", hi_q), ("LOW quality (below median)", lo_q)]:
            if len(sub) < 5:
                continue
            ah = sub["actual_pos"].mean()
            mr = sub["actual_1y"].median()
            ar = sub["actual_1y"].mean()
            p10 = np.quantile(sub["actual_1y"].values, 0.10)
            p5 = np.quantile(sub["actual_1y"].values, 0.05)
            worst = sub["actual_1y"].min()
            print(f"    {label:<35} N={len(sub):>4}  Hit={ah:.1%}  Med={mr:+.1%}  Avg={ar:+.1%}  P10={p10:+.1%}  P5={p5:+.1%}  Worst={worst:+.1%}")

    # ==========================================================================
    # 7. OVERALL FORMULA RANKING: Brier score + discrimination
    # ==========================================================================
    print(f"\n{'='*120}")
    print("7. OVERALL FORMULA RANKING")
    print(f"{'='*120}")

    print(f"\n  {'Formula':<34} {'N':>6} {'Q5-Q1 Hit':>12} {'Q5-Q1 Med':>12} {'Q5 P10':>10}")
    print(f"  {'-'*80}")

    for fname, col in opp_cols.items():
        valid = df[df[col].notna()].copy()
        if len(valid) < 50:
            continue
        try:
            valid["q"] = pd.qcut(valid[col], 5,
                                 labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                                 duplicates="drop")
        except ValueError:
            valid["q"] = pd.qcut(valid[col].rank(method="first"), 5,
                                 labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

        q1 = valid[valid["q"] == "Q1"]
        q5 = valid[valid["q"] == "Q5"]
        if len(q1) < 10 or len(q5) < 10:
            continue
        hit_spread = q5["actual_pos"].mean() - q1["actual_pos"].mean()
        med_spread = q5["actual_1y"].median() - q1["actual_1y"].median()
        q5_p10 = np.quantile(q5["actual_1y"].values, 0.10)
        print(f"  {fname:<34} {len(valid):>6} {hit_spread:>+11.1%} {med_spread:>+11.1%} {q5_p10:>+9.1%}")

    # ==========================================================================
    # 8. Per-ticker comparison
    # ==========================================================================
    print(f"\n{'='*120}")
    print("8. PER-TICKER: Quality vs actual outcomes")
    print(f"{'='*120}")

    print(f"\n{'Ticker':<8} {'N':>5} {'Q':>5} {'Wash':>6} {'Hit':>8} {'Med1Y':>10} {'P10':>10}")
    print("-" * 60)

    for t in sorted(df["ticker"].unique()):
        t_df = df[df["ticker"] == t]
        if len(t_df) < 5:
            continue
        aq = t_df["quality"].mean()
        aw = t_df["washout"].mean()
        ah = t_df["actual_pos"].mean()
        mr = t_df["actual_1y"].median()
        p10 = np.quantile(t_df["actual_1y"].values, 0.10)
        print(f"{t:<8} {len(t_df):>5} {aq:>5.0f} {aw:>5.1f} {ah:>7.1%} {mr:>+9.1%} {p10:>+9.1%}")

    print(f"\n  Elapsed: {time.time() - t0:.0f}s")
    print(f"{'='*120}")


if __name__ == "__main__":
    run_backtest()
