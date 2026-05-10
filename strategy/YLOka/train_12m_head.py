"""H1 — train a walk-forward GBM head predicting cross-sectional rank of
12-month forward returns. Mirrors the production v3 pipeline:
  - same feature set (subset of 79 cached features)
  - same model class (HistGradientBoostingRegressor)
  - rank-transformed features (regime-free)
  - cross-sectional rank label
  - January refit, expanding window, 13-month embargo (since 12m fwd label
    of training rows must end before test month -> train cutoff = test_year-1
    Dec means label of last train row ends Dec next year, which is still
    earlier than test_year Jan; 13m gap is the safe embargo).

Output: data/YLOka/ml_preds_12m.parquet with columns
  [asof, ticker, pred_12m]

Also produces a classifier head predicting P(top-quintile) over the same
horizon.

Output: data/YLOka/ml_preds_12m_cls.parquet with [asof, ticker, pred_12m_cls]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

CACHE = Path("/home/user/crt/experiments/monthly_dca/cache")
FEATURES_DIR = CACHE / "features"
OUT = Path("/home/user/crt/data/YLOka")
OUT.mkdir(parents=True, exist_ok=True)

# Match production feature set (from v2/ml_strategy.py / ml_strategy.py).
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
]

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
            "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
            "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}


def load_features_long(verbose: bool = False) -> pd.DataFrame:
    """Stack all month-end feature parquets into one long-format frame."""
    files = sorted(FEATURES_DIR.glob("*.parquet"))
    rows = []
    for f in files:
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        # df has ticker as index, features as columns
        df = df.reset_index()
        df["asof"] = d
        rows.append(df)
        if verbose and len(rows) % 50 == 0:
            print(f"  loaded {len(rows)}/{len(files)}")
    big = pd.concat(rows, ignore_index=True)
    big = big[~big["ticker"].isin(EXCLUDE)]
    return big


def cross_sectional_rank(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-asof rank-transform a column to [0, 1] within each month."""
    return df.groupby("asof")[col].transform(lambda s: s.rank(pct=True))


def winsorize_xs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-asof winsorize features at 1/99 percentile and rank-transform to [-1, 1]."""
    for c in cols:
        if c not in df.columns:
            continue
        # Per-asof rank in [0, 1], shift to [-1, 1]
        df[c] = df.groupby("asof")[c].transform(lambda s: s.rank(pct=True) * 2 - 1)
    return df


def train_one_head(target_col: str, embargo_months: int = 13,
                   first_train_year: int = 1999,
                   last_test_year: int = 2026,
                   classifier: bool = False) -> pd.DataFrame:
    """Walk-forward train (Jan refit, expanding window) and return predictions."""
    print(f"\n=== Training {'classifier' if classifier else 'regressor'} head: target={target_col} embargo={embargo_months}m ===")
    t0 = time.time()
    feats = load_features_long(verbose=False)
    print(f"  features long: {feats.shape}, {time.time()-t0:.1f}s")

    fwd = pd.read_parquet(CACHE / "fwd_returns.parquet")
    fwd = fwd[~fwd["ticker"].isin(EXCLUDE)]
    fwd = fwd[["asof", "ticker", target_col]].rename(columns={target_col: "target_raw"})

    # Merge
    df = feats.merge(fwd, on=["asof", "ticker"], how="inner")
    print(f"  merged: {df.shape}")

    # Build label: cross-sectional rank of forward return per asof
    df["target_rank"] = cross_sectional_rank(df, "target_raw")
    if classifier:
        # Top quintile = label 1, else 0
        df["target"] = (df["target_rank"] >= 0.80).astype(int)
    else:
        df["target"] = df["target_rank"]
    df = df.dropna(subset=["target"])

    # Rank-transform features per-asof to [-1, 1] (regime-free)
    feat_cols = [c for c in FEATS if c in df.columns]
    print(f"  using {len(feat_cols)} features")
    df = winsorize_xs(df.copy(), feat_cols)

    preds_chunks = []
    asofs = sorted(df["asof"].unique())
    asofs = pd.DatetimeIndex(asofs)
    for test_year in range(first_train_year + 4, last_test_year + 1):
        # Train cutoff: last asof in calendar year (test_year - 1)
        train_cutoff_target_year = test_year - (1 + embargo_months // 12)
        # Embargo: training rows asof <= dec(test_year - 1 - embargo_months/12)
        cutoff = pd.Timestamp(f"{test_year-1}-12-31") - pd.DateOffset(months=embargo_months)
        train_mask = df["asof"] <= cutoff
        # Test window: all asofs in test_year
        test_mask = (df["asof"] >= pd.Timestamp(f"{test_year}-01-01")) & \
                     (df["asof"] <= pd.Timestamp(f"{test_year}-12-31"))
        train_df = df[train_mask]
        test_df = df[test_mask]
        if len(train_df) < 1000 or len(test_df) < 100:
            continue

        X_train = train_df[feat_cols].values
        y_train = train_df["target"].values
        # Clean NaN rows in features
        valid_train = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        X_test = test_df[feat_cols].values
        valid_test = ~np.isnan(X_test).any(axis=1)

        if classifier:
            model = HistGradientBoostingClassifier(
                max_iter=200, max_depth=4, learning_rate=0.05,
                l2_regularization=0.1, random_state=42,
            )
            model.fit(X_train, y_train)
            pred = np.full(len(test_df), np.nan)
            pred[valid_test] = model.predict_proba(X_test[valid_test])[:, 1]
        else:
            model = HistGradientBoostingRegressor(
                max_iter=200, max_depth=4, learning_rate=0.05,
                l2_regularization=0.1, random_state=42,
            )
            model.fit(X_train, y_train)
            pred = np.full(len(test_df), np.nan)
            pred[valid_test] = model.predict(X_test[valid_test])

        out_col = "pred_12m_cls" if classifier else "pred_12m"
        out = test_df[["asof", "ticker"]].copy()
        out[out_col] = pred
        preds_chunks.append(out)
        print(f"  test {test_year}: train cutoff {cutoff.date()} ({len(train_df):>6} rows) -> "
              f"{len(test_df):>6} preds (valid={valid_test.sum()})")

    preds = pd.concat(preds_chunks, ignore_index=True)
    preds = preds.dropna()
    out_path = OUT / ("ml_preds_12m_cls.parquet" if classifier else "ml_preds_12m.parquet")
    preds.to_parquet(out_path)
    print(f"\nWrote {out_path}: {preds.shape} ({time.time()-t0:.1f}s total)")
    return preds


if __name__ == "__main__":
    # Train both heads
    preds_reg = train_one_head("ret__fixed_1y", embargo_months=13, classifier=False)
    preds_cls = train_one_head("ret__fixed_1y", embargo_months=13, classifier=True)
    print("\n=== summary ===")
    print(f"  pred_12m: {len(preds_reg)} preds, asof range {preds_reg['asof'].min().date()} -> {preds_reg['asof'].max().date()}")
    print(f"  pred_12m_cls: {len(preds_cls)} preds, asof range {preds_cls['asof'].min().date()} -> {preds_cls['asof'].max().date()}")
