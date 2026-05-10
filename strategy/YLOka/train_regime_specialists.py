"""Session 5 — Regime-specific specialist GBMs.

Hypothesis: a single GBM trained on all-regime data learns the AVERAGE
predictive pattern. Different regimes (bull/normal/recovery) reward
different signals. Train a specialist GBM per regime; route at test time.

Implementation:
  1. Compute the regime label per (asof) using the same `tight` SPY-based
     classifier that v3 uses for its crash gate. We get bull/normal/
     recovery (skip crash since the production strategy goes to cash).
  2. For each regime r, walk-forward train a HistGradientBoostingRegressor
     using ONLY training rows where regime(asof) == r. 22 train windows
     (Jan refit, expanding window, 7-month embargo).
  3. Predict for ALL test asofs (not just same-regime ones) so we have
     a full set of pred_specialist_<regime>_3m and _6m, and ensemble
     them at backtest time.

Output: data/YLOka/ml_preds_specialist_<regime>.parquet with
  [asof, ticker, pred_3m_<regime>, pred_6m_<regime>]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/user/crt")

CACHE = Path("/home/user/crt/experiments/monthly_dca/cache")
FEATURES_DIR = CACHE / "features"
DATA = Path("/home/user/crt/data/YLOka")

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
            "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
            "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}

FEATS = [
    "pullback_1y", "pullback_3y", "pullback_5y", "pullback_all",
    "range_pos_1y", "trend_health_5y",
    "mom_12_1", "mom_6_1", "mom_3", "mom_3y", "mom_2y", "mom_5y",
    "vol_1y", "vol_12m", "vol_3m", "vol_6m", "vol_contraction",
    "ret_5d", "ret_21d", "accel",
    "d_sma200", "d_sma50", "sma50_above_200",
    "rsi_14", "frac_above_50dma_1y", "sharpe_1y", "sharpe_12m",
    "dd_from_52wh", "below_52wh", "new_52wh",
    "best_month_24m", "worst_month_24m", "tail_ratio_24m", "multibagger_ratio_24m",
    "trend_r2_12m", "max_below_200_streak",
    "rs_3m_spy", "rs_6m_spy", "rs_12m_spy", "excess_5y_logret",
    "mom_accel", "mom_consistency_12m", "dist_from_low_1y", "near_52wh_60d",
    "bb_width_pct", "bb_width_contraction", "drawdown_age_days",
    "log_price", "trend_slope_252", "beta_2y",
    "mean_ret_12m", "vol_expansion_24m",
    # Add the runner-pattern features (production v3 uses all of these)
    "idio_mom_12_1", "fip_score", "tight_consolidation_60", "breakout_strength_60",
    "min_dd_60d", "rsi_zone_score", "acceleration_2y", "mom_per_unit_vol_12",
    "crt_3m", "crt_6m",
]


def regime_tight(s: dict) -> str:
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    if pd.isna(r21):
        return "normal"
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def compute_regime_per_asof() -> pd.Series:
    """Return Series indexed by asof with regime label."""
    rows = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        s = {
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
        }
        rows.append({"asof": d, "regime": regime_tight(s)})
    return pd.DataFrame(rows).set_index("asof")["regime"]


def load_features_long_with_regime():
    print("Loading features...")
    t0 = time.time()
    files = sorted(FEATURES_DIR.glob("*.parquet"))
    rows = []
    for f in files:
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f).reset_index()
        keep = ["ticker"] + [c for c in FEATS if c in df.columns]
        df = df[keep].copy()
        df["asof"] = d
        rows.append(df)
    big = pd.concat(rows, ignore_index=True)
    big = big[~big["ticker"].isin(EXCLUDE)]
    print(f"  {big.shape}, {time.time()-t0:.1f}s")
    return big


def winsorize_xs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = df.groupby("asof")[c].transform(lambda s: s.rank(pct=True) * 2 - 1)
    return df


def train_specialist(regime: str, target_horizon_months: int,
                      feats: pd.DataFrame, regime_labels: pd.Series,
                      embargo_months: int = 7,
                      first_test_year: int = 2003,
                      last_test_year: int = 2026) -> pd.DataFrame:
    """Train a regime-specialist walk-forward GBM. Returns predictions for
    ALL test asofs (not just same-regime asofs) so we can ensemble at
    backtest time."""
    print(f"\n=== Specialist regime={regime} target={target_horizon_months}m ===")
    t0 = time.time()

    # Build forward returns from monthly returns
    mr = pd.read_parquet(CACHE / "v2" / "monthly_returns_clean.parquet")
    fwd_rows = []
    asofs = list(mr.index)
    n = target_horizon_months
    for i, t in enumerate(asofs):
        if i + n >= len(asofs): continue
        sub = mr.iloc[i+1:i+1+n]
        prod = (1 + sub).prod() - 1
        for tk, v in prod.items():
            fwd_rows.append({"asof": t, "ticker": tk, "fwd_target": float(v)})
    fwd_df = pd.DataFrame(fwd_rows)

    # Merge features + target + regime
    df = feats.merge(fwd_df, on=["asof", "ticker"], how="inner")
    df = df.merge(regime_labels.to_frame("regime"), on="asof", how="left")
    df["target_rank"] = df.groupby("asof")["fwd_target"].rank(pct=True)
    df = df.dropna(subset=["target_rank"])

    # Rank-transform features per asof
    feat_cols = [c for c in FEATS if c in df.columns]
    df = winsorize_xs(df.copy(), feat_cols)

    preds_chunks = []
    for test_year in range(first_test_year, last_test_year + 1):
        cutoff = pd.Timestamp(f"{test_year-1}-12-31") - pd.DateOffset(months=embargo_months)
        train_df = df[(df["asof"] <= cutoff) & (df["regime"] == regime)]
        test_df = df[(df["asof"] >= pd.Timestamp(f"{test_year}-01-01")) &
                      (df["asof"] <= pd.Timestamp(f"{test_year}-12-31"))]
        if len(train_df) < 500 or len(test_df) < 100:
            continue

        X_train = train_df[feat_cols].values
        y_train = train_df["target_rank"].values
        valid_train = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        if len(X_train) < 500:
            continue

        X_test = test_df[feat_cols].values
        valid_test = ~np.isnan(X_test).any(axis=1)

        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.05,
            l2_regularization=0.1, random_state=42,
        )
        model.fit(X_train, y_train)
        pred = np.full(len(test_df), np.nan)
        pred[valid_test] = model.predict(X_test[valid_test])

        out = test_df[["asof", "ticker"]].copy()
        out[f"pred_{target_horizon_months}m_{regime}"] = pred
        preds_chunks.append(out)
        print(f"  test {test_year}: train cutoff {cutoff.date()} "
              f"(train_n={len(train_df):>6} regime-only={len(X_train):>6}) -> "
              f"{len(test_df):>6} preds (valid={valid_test.sum()})")

    if not preds_chunks:
        print(f"  no preds (regime {regime} too sparse)")
        return pd.DataFrame()
    preds = pd.concat(preds_chunks, ignore_index=True).dropna()
    out_path = DATA / f"ml_preds_{target_horizon_months}m_{regime}.parquet"
    preds.to_parquet(out_path)
    print(f"  wrote {out_path}: {preds.shape} ({time.time()-t0:.1f}s)")
    return preds


if __name__ == "__main__":
    print("Computing regime labels...")
    regime_labels = compute_regime_per_asof()
    print(f"regime distribution: {regime_labels.value_counts().to_dict()}")
    regime_labels.to_frame("regime").to_parquet(DATA / "regime_labels.parquet")

    feats = load_features_long_with_regime()

    for regime in ["bull", "normal", "recovery"]:
        for h in [3, 6]:
            train_specialist(regime, h, feats, regime_labels)
