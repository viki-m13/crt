#!/usr/bin/env python3
"""
stock_pred_data.py — Data preparation for stock prediction model
================================================================
Prepares labeled dataset for predicting which stocks will gain 10%+ in 30 days.

ANTI-LEAKAGE GUARANTEES:
- Labels (forward 30-day return) are NEVER used as features
- All features use strictly backward-looking windows
- Walk-forward splits ensure train < valid < test chronologically
- 63-day buffer between all splits to prevent information bleed
- Features are z-scored using ONLY the training window statistics
"""

import numpy as np
import pandas as pd
from prepare import (
    load_data, compute_features, UNIVERSE,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    TEST_START, TEST_END,
)

# Target: stock goes up >= 10% in next 30 trading days
TARGET_RETURN = 0.10
TARGET_HORIZON = 30  # trading days forward

# Walk-forward configuration
# Each fold: train on N years, validate on next period, slide forward
WALK_FORWARD_TRAIN_DAYS = 504   # ~2 years of training data per fold
WALK_FORWARD_STEP_DAYS = 63     # slide forward ~3 months per fold
WALK_FORWARD_BUFFER_DAYS = 63   # gap between train and valid to prevent leakage


def build_dataset():
    """
    Build the full labeled dataset from raw price data.
    Returns: DataFrame with features, labels, and metadata columns.
    """
    print("Loading market data...")
    raw_data = load_data()

    if "SPY" not in raw_data:
        raise ValueError("SPY data required for market features")

    spy_close = raw_data["SPY"]["Close"]

    all_rows = []

    for ticker in UNIVERSE:
        if ticker not in raw_data or ticker == "SPY":
            continue

        df = raw_data[ticker]
        if len(df) < 500:
            continue

        close = df["Close"]
        volume = df.get("Volume", None)

        # Compute backward-looking features (no leakage)
        feat = compute_features(close, volume=volume, market_close=spy_close)

        # Compute forward return for LABELING ONLY
        fwd_ret = close.shift(-TARGET_HORIZON) / close - 1.0
        feat["fwd_return_30d"] = fwd_ret.reindex(feat.index)
        feat["label"] = (feat["fwd_return_30d"] >= TARGET_RETURN).astype(int)

        # Add metadata
        feat["ticker"] = ticker
        feat["close"] = close.reindex(feat.index)
        feat["date"] = feat.index

        all_rows.append(feat)

    dataset = pd.concat(all_rows, ignore_index=True)
    dataset = dataset.dropna(subset=["label"])

    print(f"Dataset: {len(dataset)} rows, {dataset['label'].sum()} positive "
          f"({dataset['label'].mean():.1%} hit rate)")

    return dataset


def get_feature_columns(dataset):
    """Return list of feature column names (excludes labels, metadata)."""
    exclude = {"fwd_return_30d", "label", "ticker", "close", "date"}
    return [c for c in dataset.columns if c not in exclude]


def get_walk_forward_splits(dataset):
    """
    Generate walk-forward train/validation splits.

    Each fold:
    - Train: [start, start + TRAIN_DAYS)
    - Buffer: BUFFER_DAYS gap (no data used)
    - Valid: [start + TRAIN_DAYS + BUFFER, start + TRAIN_DAYS + BUFFER + STEP)

    Yields: (fold_id, train_mask, valid_mask)

    ANTI-LEAKAGE: strict chronological ordering with buffer gaps.
    """
    dates = pd.to_datetime(dataset["date"])
    unique_dates = sorted(dates.unique())

    # Only use train+valid period for walk-forward CV
    cutoff_valid_end = pd.Timestamp(VALID_END)
    cutoff_train_start = pd.Timestamp(TRAIN_START)

    available_dates = [d for d in unique_dates
                       if cutoff_train_start <= d <= cutoff_valid_end]

    if len(available_dates) < WALK_FORWARD_TRAIN_DAYS + WALK_FORWARD_BUFFER_DAYS + WALK_FORWARD_STEP_DAYS:
        raise ValueError("Not enough data for walk-forward validation")

    fold_id = 0
    start_idx = 0

    while True:
        train_end_idx = start_idx + WALK_FORWARD_TRAIN_DAYS
        valid_start_idx = train_end_idx + WALK_FORWARD_BUFFER_DAYS
        valid_end_idx = valid_start_idx + WALK_FORWARD_STEP_DAYS

        if valid_end_idx > len(available_dates):
            break

        train_start_date = available_dates[start_idx]
        train_end_date = available_dates[train_end_idx - 1]
        valid_start_date = available_dates[valid_start_idx]
        valid_end_date = available_dates[valid_end_idx - 1]

        train_mask = (dates >= train_start_date) & (dates <= train_end_date)
        valid_mask = (dates >= valid_start_date) & (dates <= valid_end_date)

        n_train = train_mask.sum()
        n_valid = valid_mask.sum()
        pos_rate_train = dataset.loc[train_mask, "label"].mean()
        pos_rate_valid = dataset.loc[valid_mask, "label"].mean()

        print(f"  Fold {fold_id}: train {train_start_date.date()}..{train_end_date.date()} "
              f"({n_train} rows, {pos_rate_train:.1%} pos) | "
              f"valid {valid_start_date.date()}..{valid_end_date.date()} "
              f"({n_valid} rows, {pos_rate_valid:.1%} pos)")

        yield fold_id, train_mask.values, valid_mask.values

        fold_id += 1
        start_idx += WALK_FORWARD_STEP_DAYS

    print(f"Total walk-forward folds: {fold_id}")


def get_test_split(dataset):
    """Return train (all pre-test) and test masks for final evaluation."""
    dates = pd.to_datetime(dataset["date"])
    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)

    # Train on everything before test (with buffer)
    train_cutoff = test_start - pd.Timedelta(days=90)  # 63 trading day buffer
    train_mask = dates <= train_cutoff
    test_mask = (dates >= test_start) & (dates <= test_end)

    print(f"Final test split: train {train_mask.sum()} rows, test {test_mask.sum()} rows")
    return train_mask.values, test_mask.values


def normalize_features(X_train, X_valid, feature_cols):
    """
    Z-score normalize features using ONLY training statistics.
    This prevents leakage from validation/test into training.
    """
    means = X_train[feature_cols].mean()
    stds = X_train[feature_cols].std().clip(lower=1e-8)

    X_train_norm = X_train.copy()
    X_valid_norm = X_valid.copy()

    X_train_norm[feature_cols] = (X_train[feature_cols] - means) / stds
    X_valid_norm[feature_cols] = (X_valid[feature_cols] - means) / stds

    # Clip extreme values
    X_train_norm[feature_cols] = X_train_norm[feature_cols].clip(-5, 5)
    X_valid_norm[feature_cols] = X_valid_norm[feature_cols].clip(-5, 5)

    return X_train_norm, X_valid_norm, means, stds


if __name__ == "__main__":
    print("Building dataset...")
    ds = build_dataset()
    print(f"\nFeature columns: {get_feature_columns(ds)}")
    print(f"\nWalk-forward splits:")
    for fold_id, train_mask, valid_mask in get_walk_forward_splits(ds):
        pass
    print(f"\nTest split:")
    get_test_split(ds)
