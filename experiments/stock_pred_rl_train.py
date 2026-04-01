#!/usr/bin/env python3
"""
stock_pred_rl_train.py — Neural Network Scorer for Stock Selection
===================================================================
Neural network trained with combined objectives:
1. Binary cross-entropy: classify stocks gaining 10%+ in 30 days
2. ListNet ranking loss: rank stocks so top-K have highest returns

This is equivalent to a learned policy: the network scores each stock
and we select top-K daily — the REINFORCE-style ranking loss ensures
the top picks are the most profitable.

Walk-forward validated, no leakage.

ANTI-LEAKAGE:
- Walk-forward: train on past, evaluate on future
- Features normalized using ONLY training statistics
- Forward returns used ONLY for loss computation during training
"""

import numpy as np
import pandas as pd
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from stock_pred_data import (
    build_dataset, get_feature_columns, get_walk_forward_splits,
    get_test_split, normalize_features,
    WALK_FORWARD_TRAIN_DAYS, WALK_FORWARD_BUFFER_DAYS, WALK_FORWARD_STEP_DAYS,
)
from prepare import TRAIN_START, VALID_END


class StockScorer(nn.Module):
    """Neural network that scores stocks for selection."""

    def __init__(self, n_features, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_scorer_batch(model, X_train, y_train, returns_train,
                       n_epochs=15, lr=1e-3, batch_size=2048):
    """
    Train the scorer using batched BCE + return-weighted loss.

    Much faster than per-day iteration — processes all data in mini-batches.
    The ranking signal comes from weighting the BCE loss by forward returns:
    - Stocks with high returns get higher weight when they're positive
    - This teaches the model to prioritize the best stocks, not just any positive
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    ret_t = torch.FloatTensor(returns_train)

    dataset = TensorDataset(X_t, y_t, ret_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for X_batch, y_batch, ret_batch in loader:
            scores = model(X_batch)

            # Weighted BCE: upweight high-return positives
            # This creates a ranking signal — not just classify, but rank by quality
            weights = torch.ones_like(y_batch)
            # Positive samples with high returns get extra weight
            pos_mask = y_batch > 0.5
            weights[pos_mask] = 1.0 + ret_batch[pos_mask].clamp(0, 0.5) * 4.0
            # Negative samples that lost money get extra weight (avoid losers)
            neg_mask = (y_batch < 0.5) & (ret_batch < -0.05)
            weights[neg_mask] = 1.5

            bce = nn.functional.binary_cross_entropy(scores, y_batch,
                                                      weight=weights, reduction="mean")

            # Return-maximizing loss: encourage high scores for high-return stocks
            return_loss = -(scores * ret_batch.clamp(-0.3, 0.5)).mean()

            loss = bce + 0.3 * return_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

    return total_loss / max(n_batches, 1)


def evaluate_scorer(model, eval_data, feature_cols, top_k=3):
    """Evaluate the scorer on a dataset."""
    model.eval()
    eval_data = eval_data.copy()
    eval_data["_date_str"] = pd.to_datetime(eval_data["date"]).dt.strftime("%Y-%m-%d")
    dates = sorted(eval_data["_date_str"].unique())

    all_picks = []

    X_all = np.nan_to_num(eval_data[feature_cols].values.astype(np.float32),
                           nan=0.0, posinf=5.0, neginf=-5.0).clip(-5, 5)

    with torch.no_grad():
        all_scores = model(torch.FloatTensor(X_all)).numpy()

    eval_data["_score"] = all_scores

    for date_str in dates:
        daily = eval_data[eval_data["_date_str"] == date_str]
        if len(daily) < top_k:
            continue

        top = daily.nlargest(top_k, "_score")
        for _, row in top.iterrows():
            all_picks.append({
                "date": date_str,
                "ticker": row["ticker"],
                "score": float(row["_score"]),
                "fwd_return": row["fwd_return_30d"],
                "label": row["label"],
            })

    if not all_picks:
        return {"n_picks": 0, "hit_rate": 0, "avg_return": 0, "median_return": 0}, []

    picks_df = pd.DataFrame(all_picks)
    metrics = {
        "n_picks": len(picks_df),
        "hit_rate": float(picks_df["label"].mean()),
        "avg_return": float(picks_df["fwd_return"].mean()),
        "median_return": float(picks_df["fwd_return"].median()),
    }

    return metrics, all_picks


def train_rl_walk_forward(dataset, feature_cols):
    """Train neural scorer with walk-forward validation."""
    results = []
    print("\n=== Walk-Forward Neural Scorer Training ===\n")

    dates = pd.to_datetime(dataset["date"])
    unique_dates = sorted(dates.unique())
    cutoff_valid_end = pd.Timestamp(VALID_END)
    cutoff_train_start = pd.Timestamp(TRAIN_START)
    available_dates = [d for d in unique_dates
                       if cutoff_train_start <= d <= cutoff_valid_end]

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

        train_data = dataset[train_mask.values].copy()
        valid_data = dataset[valid_mask.values].copy()

        train_data, valid_data, means, stds = normalize_features(
            train_data, valid_data, feature_cols
        )

        # Prepare training arrays
        X_train = np.nan_to_num(train_data[feature_cols].values.astype(np.float32),
                                nan=0.0, posinf=5.0, neginf=-5.0).clip(-5, 5)
        y_train = train_data["label"].values.astype(np.float32)
        ret_train = train_data["fwd_return_30d"].values.astype(np.float32)

        model = StockScorer(len(feature_cols))
        train_scorer_batch(model, X_train, y_train, ret_train, n_epochs=12)

        eval_metrics, picks = evaluate_scorer(model, valid_data, feature_cols, top_k=3)
        eval_metrics["fold"] = fold_id

        print(f"  Fold {fold_id}: "
              f"train {train_start_date.date()}..{train_end_date.date()} | "
              f"valid {valid_start_date.date()}..{valid_end_date.date()} | "
              f"picks={eval_metrics['n_picks']}, "
              f"hit={eval_metrics['hit_rate']:.1%}, "
              f"ret={eval_metrics['avg_return']:.2%}")

        results.append({
            "fold_id": fold_id,
            "metrics": eval_metrics,
            "model_state": model.state_dict(),
            "norm_means": means,
            "norm_stds": stds,
        })

        fold_id += 1
        start_idx += WALK_FORWARD_STEP_DAYS

    if results:
        avg_hit = np.mean([r["metrics"]["hit_rate"] for r in results])
        avg_ret = np.mean([r["metrics"]["avg_return"] for r in results])
        print(f"\n  Aggregate: hit_rate={avg_hit:.1%}, avg_return={avg_ret:.2%}")

    return results


def train_final_rl_model(dataset, feature_cols):
    """Train final neural scorer on all pre-test data, evaluate on test."""
    print("\n=== Final Neural Scorer (Train on pre-test, evaluate on test) ===\n")

    train_mask, test_mask = get_test_split(dataset)
    train_data = dataset.iloc[train_mask].copy()
    test_data = dataset.iloc[test_mask].copy()

    train_data, test_data, means, stds = normalize_features(
        train_data, test_data, feature_cols
    )

    X_train = np.nan_to_num(train_data[feature_cols].values.astype(np.float32),
                            nan=0.0, posinf=5.0, neginf=-5.0).clip(-5, 5)
    y_train = train_data["label"].values.astype(np.float32)
    ret_train = train_data["fwd_return_30d"].values.astype(np.float32)

    model = StockScorer(len(feature_cols), hidden_dim=128)
    train_scorer_batch(model, X_train, y_train, ret_train, n_epochs=20)

    test_metrics, test_picks = evaluate_scorer(model, test_data, feature_cols, top_k=3)

    print(f"  Test Picks: {test_metrics['n_picks']}")
    print(f"  Test Hit Rate: {test_metrics['hit_rate']:.1%}")
    print(f"  Test Avg Return: {test_metrics['avg_return']:.2%}")
    print(f"  Test Median Return: {test_metrics['median_return']:.2%}")

    return model, test_metrics, means, stds


if __name__ == "__main__":
    dataset = build_dataset()
    feature_cols = get_feature_columns(dataset)

    wf_results = train_rl_walk_forward(dataset, feature_cols)
    model, test_metrics, means, stds = train_final_rl_model(dataset, feature_cols)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(results_dir, "rl_scorer_model.pt"))
    with open(os.path.join(results_dir, "rl_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2, default=str)

    print("\nDone. Neural scorer model and metrics saved to results/")
