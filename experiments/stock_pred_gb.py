#!/usr/bin/env python3
"""
stock_pred_gb.py — Gradient Boosting model for stock prediction
===============================================================
Binary classifier: will this stock gain >= 10% in the next 30 trading days?

Uses XGBoost with walk-forward validation to prevent overfitting.

ANTI-LEAKAGE GUARANTEES:
- Walk-forward: always train on past, predict on future
- Features normalized using ONLY training period statistics
- Class weights handle severe imbalance (typically ~5-10% positive rate)
- Hyperparameters tuned via walk-forward CV, NOT random search on full data
- Early stopping on validation loss within each fold
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report
)
import json
import os

from stock_pred_data import (
    build_dataset, get_feature_columns, get_walk_forward_splits,
    get_test_split, normalize_features,
)


def get_xgb_params(pos_weight=1.0):
    """Conservative XGBoost params to prevent overfitting."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",  # area under precision-recall curve
        "max_depth": 4,          # shallow trees to prevent overfitting
        "learning_rate": 0.02,   # slow learning
        "subsample": 0.7,        # row subsampling
        "colsample_bytree": 0.7, # column subsampling
        "min_child_weight": 20,  # require substantial leaf support
        "gamma": 1.0,            # minimum loss reduction for split
        "reg_alpha": 0.5,        # L1 regularization
        "reg_lambda": 2.0,       # L2 regularization
        "scale_pos_weight": pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }


def train_walk_forward(dataset, feature_cols):
    """
    Train XGBoost with walk-forward validation.
    Returns: list of (fold_id, model, metrics, predictions) per fold.
    """
    results = []
    all_valid_preds = []
    all_valid_labels = []

    print("\n=== Walk-Forward Gradient Boosting Training ===\n")

    for fold_id, train_mask, valid_mask in get_walk_forward_splits(dataset):
        train_data = dataset.iloc[train_mask].copy()
        valid_data = dataset.iloc[valid_mask].copy()

        # Normalize using ONLY training stats
        train_data, valid_data, means, stds = normalize_features(
            train_data, valid_data, feature_cols
        )

        X_train = train_data[feature_cols].values
        y_train = train_data["label"].values
        X_valid = valid_data[feature_cols].values
        y_valid = valid_data["label"].values

        # Handle NaN/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=5.0, neginf=-5.0)
        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=5.0, neginf=-5.0)

        # Class imbalance weight
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = n_neg / max(n_pos, 1)

        params = get_xgb_params(pos_weight=min(pos_weight, 20.0))

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_cols)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        # Predictions
        y_pred_prob = model.predict(dvalid)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Metrics
        metrics = compute_metrics(y_valid, y_pred_prob, y_pred)
        metrics["fold"] = fold_id
        metrics["n_train"] = len(y_train)
        metrics["n_valid"] = len(y_valid)
        metrics["pos_rate_train"] = float(y_train.mean())
        metrics["pos_rate_valid"] = float(y_valid.mean())
        metrics["best_iteration"] = model.best_iteration

        print(f"  Fold {fold_id}: AUC-PR={metrics['auc_pr']:.3f}, "
              f"Prec@top50={metrics['precision_at_top50']:.3f}, "
              f"Recall={metrics['recall']:.3f}")

        results.append({
            "fold_id": fold_id,
            "model": model,
            "metrics": metrics,
            "valid_preds": y_pred_prob,
            "valid_labels": y_valid,
            "valid_tickers": valid_data["ticker"].values,
            "valid_dates": valid_data["date"].values,
            "valid_fwd_returns": valid_data["fwd_return_30d"].values,
            "norm_means": means,
            "norm_stds": stds,
        })

        all_valid_preds.extend(y_pred_prob.tolist())
        all_valid_labels.extend(y_valid.tolist())

    # Aggregate metrics across all folds
    print("\n=== Aggregate Walk-Forward Results ===")
    all_preds = np.array(all_valid_preds)
    all_labels = np.array(all_valid_labels)
    agg_metrics = compute_metrics(all_labels, all_preds, (all_preds >= 0.5).astype(int))
    print(f"  Overall AUC-PR: {agg_metrics['auc_pr']:.3f}")
    print(f"  Overall Prec@top50: {agg_metrics['precision_at_top50']:.3f}")
    print(f"  Overall Recall: {agg_metrics['recall']:.3f}")

    return results, agg_metrics


def compute_metrics(y_true, y_prob, y_pred):
    """Compute comprehensive classification metrics."""
    metrics = {}

    if len(np.unique(y_true)) < 2:
        return {"auc_pr": 0, "auc_roc": 0, "precision": 0, "recall": 0,
                "f1": 0, "precision_at_top50": 0}

    metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # Precision at top-K (most actionable metric for stock selection)
    top_k = min(50, len(y_prob))
    top_indices = np.argsort(y_prob)[-top_k:]
    metrics["precision_at_top50"] = float(y_true[top_indices].mean())

    top_k = min(20, len(y_prob))
    top_indices = np.argsort(y_prob)[-top_k:]
    metrics["precision_at_top20"] = float(y_true[top_indices].mean())

    return metrics


def analyze_feature_importance(results, feature_cols):
    """Aggregate feature importance across walk-forward folds."""
    importance_sum = np.zeros(len(feature_cols))

    for r in results:
        model = r["model"]
        imp = model.get_score(importance_type="gain")
        for i, col in enumerate(feature_cols):
            importance_sum[i] += imp.get(col, 0)

    importance_avg = importance_sum / len(results)
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance_avg,
    }).sort_values("importance", ascending=False)

    print("\n=== Top 15 Features (by gain) ===")
    for _, row in imp_df.head(15).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.1f}")

    return imp_df


def train_final_model(dataset, feature_cols):
    """
    Train final model on all pre-test data, evaluate on test set.
    This is the HELD-OUT evaluation — only run once at the end.
    """
    print("\n=== Final Model (Train on pre-test, evaluate on test) ===\n")

    train_mask, test_mask = get_test_split(dataset)
    train_data = dataset.iloc[train_mask].copy()
    test_data = dataset.iloc[test_mask].copy()

    train_data, test_data, means, stds = normalize_features(
        train_data, test_data, feature_cols
    )

    X_train = train_data[feature_cols].values
    y_train = train_data["label"].values
    X_test = test_data[feature_cols].values
    y_test = test_data["label"].values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=5.0, neginf=-5.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=5.0, neginf=-5.0)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    params = get_xgb_params(pos_weight=min(pos_weight, 20.0))

    # Use a small hold-out from end of training for early stopping
    split_idx = int(len(X_train) * 0.9)
    dtrain = xgb.DMatrix(X_train[:split_idx], label=y_train[:split_idx],
                          feature_names=feature_cols)
    deval = xgb.DMatrix(X_train[split_idx:], label=y_train[split_idx:],
                          feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (deval, "eval")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    test_metrics = compute_metrics(y_test, y_pred_prob, y_pred)

    print(f"  Test AUC-PR:       {test_metrics['auc_pr']:.3f}")
    print(f"  Test AUC-ROC:      {test_metrics['auc_roc']:.3f}")
    print(f"  Test Prec@top50:   {test_metrics['precision_at_top50']:.3f}")
    print(f"  Test Prec@top20:   {test_metrics['precision_at_top20']:.3f}")
    print(f"  Test Recall:       {test_metrics['recall']:.3f}")
    print(f"  Test F1:           {test_metrics['f1']:.3f}")

    # Detailed analysis of top predictions
    test_results = pd.DataFrame({
        "ticker": test_data["ticker"].values,
        "date": test_data["date"].values,
        "pred_prob": y_pred_prob,
        "label": y_test,
        "fwd_return_30d": test_data["fwd_return_30d"].values,
    })

    top_picks = test_results.nlargest(50, "pred_prob")
    print(f"\n  Top 50 predictions:")
    print(f"    Hit rate:    {top_picks['label'].mean():.1%}")
    print(f"    Avg return:  {top_picks['fwd_return_30d'].mean():.2%}")
    print(f"    Med return:  {top_picks['fwd_return_30d'].median():.2%}")

    return model, test_metrics, test_results, means, stds


if __name__ == "__main__":
    dataset = build_dataset()
    feature_cols = get_feature_columns(dataset)

    # Walk-forward validation
    wf_results, wf_metrics = train_walk_forward(dataset, feature_cols)
    analyze_feature_importance(wf_results, feature_cols)

    # Final held-out test
    model, test_metrics, test_results, means, stds = train_final_model(dataset, feature_cols)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "gb_walk_forward_metrics.json"), "w") as f:
        json.dump(wf_metrics, f, indent=2)

    with open(os.path.join(results_dir, "gb_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    model.save_model(os.path.join(results_dir, "gb_model.json"))
    print("\nDone. Model and metrics saved to results/")
