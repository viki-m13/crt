"""ML-based stock-picking strategy: gradient-boosted regression on features.

Trains on historical (features at month-end) -> 3y forward return,
predicts per (asof, ticker) and ranks. Uses purged time-aware CV to avoid
leakage.

Picks: at each month-end, predict 3y return for every eligible ticker; pick top-K.

We train ONCE per major historical block, walk-forward to next block. This
gives an honest picks log over 1997-2024.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from experiments.monthly_dca.fast_score import BENCH_EXCLUDED, load_features_long, load_fwd, load_panel


# Features used by the ML model. Excludes 'price' (raw level not predictive) and
# 'recovery_rate' (NaN for many).
ML_FEATURES = [
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


def _winsorize(x: pd.Series, lo: float = 0.005, hi: float = 0.995) -> pd.Series:
    a = x.quantile(lo); b = x.quantile(hi)
    return x.clip(a, b)


def build_ml_dataset(asof_min: pd.Timestamp = pd.Timestamp("2002-01-01"),
                      asof_max: pd.Timestamp = pd.Timestamp("2099-01-01"),
                      target: str = "ret__fixed_3y") -> pd.DataFrame:
    feats = load_features_long().reset_index()
    fwd = load_fwd().reset_index()
    m = feats.merge(fwd[["asof", "ticker", target]], on=["asof", "ticker"], how="inner")
    m = m[(m["asof"] >= asof_min) & (m["asof"] <= asof_max)]
    m = m[~m["ticker"].isin(BENCH_EXCLUDED)]
    return m


def train_block(train_df: pd.DataFrame, target_col: str) -> GradientBoostingRegressor:
    cols = [c for c in ML_FEATURES if c in train_df.columns]
    X = train_df[cols].copy()
    # winsorize each
    for c in cols:
        X[c] = _winsorize(X[c])
    y = train_df[target_col].astype(float)
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]; y = y[valid]
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X.values, y.values)
    return model


def predict_block(model: GradientBoostingRegressor, df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ML_FEATURES if c in df.columns]
    X = df[cols].copy()
    for c in cols:
        X[c] = _winsorize(X[c])
    valid = X.notna().all(axis=1)
    pred = pd.Series(np.nan, index=df.index)
    if valid.any():
        pred.loc[valid] = model.predict(X[valid].values)
    return pred


def walk_forward_ml(
    target: str = "ret__fixed_3y",
    train_blocks: list[tuple[str, str]] | None = None,
    test_blocks: list[tuple[str, str]] | None = None,
    feats_long: pd.DataFrame = None,
    fwd: pd.DataFrame = None,
) -> pd.DataFrame:
    """Walk-forward training. Default: 5y train -> 3y test, then re-train.

    Returns a DataFrame of (asof, ticker, ml_score) for predictions in the
    test windows.
    """
    if train_blocks is None:
        # Walk-forward: train on rows that have 3y forward returns (asof <= eval - 3y).
        # Use anchored expansion: each test window uses ALL prior data (asof <= test_start - 3y)
        # for training. 3y embargo ensures no target leakage.
        train_blocks = [
            ("1997-01-01", "1999-12-31"),  # need 3y forward, so train asof <= 1999-12-31
            ("1997-01-01", "2002-12-31"),
            ("1997-01-01", "2005-12-31"),
            ("1997-01-01", "2008-12-31"),
            ("1997-01-01", "2011-12-31"),
            ("1997-01-01", "2014-12-31"),
            ("1997-01-01", "2017-12-31"),
            ("1997-01-01", "2020-12-31"),
        ]
        test_blocks = [
            ("2003-01-01", "2005-12-31"),  # 3y embargo from train end
            ("2006-01-01", "2008-12-31"),
            ("2009-01-01", "2011-12-31"),
            ("2012-01-01", "2014-12-31"),
            ("2015-01-01", "2017-12-31"),
            ("2018-01-01", "2020-12-31"),
            ("2021-01-01", "2023-12-31"),
            ("2024-01-01", "2024-12-31"),  # most recent (only 1y, but useful)
        ]
    if feats_long is None:
        feats_long = load_features_long()
    if fwd is None:
        fwd = load_fwd()

    feats = feats_long.reset_index()
    fwd_r = fwd.reset_index()

    df_full = feats.merge(fwd_r[["asof", "ticker", target]], on=["asof", "ticker"], how="inner")
    df_full = df_full[~df_full["ticker"].isin(BENCH_EXCLUDED)]
    print(f"  full ML dataset: {len(df_full)} rows")

    preds_chunks = []
    for (tr_start, tr_end), (te_start, te_end) in zip(train_blocks, test_blocks):
        train_df = df_full[(df_full["asof"] >= pd.Timestamp(tr_start)) &
                            (df_full["asof"] <= pd.Timestamp(tr_end))].copy()
        test_df = df_full[(df_full["asof"] >= pd.Timestamp(te_start)) &
                           (df_full["asof"] <= pd.Timestamp(te_end))].copy()
        # purge: ensure target horizon doesn't bleed into test window
        # (fixed_3y target uses 3y forward, so train should end at least 3y before test start)
        train_df = train_df[train_df["asof"] <= pd.Timestamp(tr_end)]
        purge_cutoff = pd.Timestamp(tr_end)
        if train_df.empty or test_df.empty:
            continue
        print(f"  train [{tr_start} -> {purge_cutoff.date()}] ({len(train_df)}) -> test [{te_start} -> {te_end}] ({len(test_df)})")
        try:
            model = train_block(train_df, target)
        except Exception as e:
            print(f"    skip: {e}")
            continue
        pred = predict_block(model, test_df)
        out = test_df[["asof", "ticker"]].copy()
        out["ml_score"] = pred.values
        out = out.dropna(subset=["ml_score"])
        preds_chunks.append(out)

    if not preds_chunks:
        return pd.DataFrame(columns=["asof", "ticker", "ml_score"])
    return pd.concat(preds_chunks, ignore_index=True)


def main():
    print("Loading...")
    feats_long = load_features_long()
    fwd = load_fwd()
    panel = load_panel()
    print("Walk-forward training...")
    preds = walk_forward_ml(target="ret__fixed_3y", feats_long=feats_long, fwd=fwd)
    out_path = Path("experiments/monthly_dca/cache") / "ml_predictions.parquet"
    preds.to_parquet(out_path, compression="zstd")
    print(f"Wrote {out_path}: {preds.shape}")
    print(preds.head())


if __name__ == "__main__":
    main()
