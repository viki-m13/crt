#!/usr/bin/env python3
"""
stock_pred_ensemble.py — Ensemble combining GB + Neural Scorer predictions
===========================================================================
Combines gradient boosting probability scores with neural scorer scores
for a final stock selection decision.

The ensemble uses a calibrated blend:
1. GB provides P(stock gains 10%+ in 30d)
2. Neural scorer provides learned buy scores
3. Final score = weighted combination, stocks ranked daily

ANTI-LEAKAGE: Ensemble weights determined on validation folds, never test.
"""

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score

from stock_pred_data import (
    build_dataset, get_feature_columns, get_walk_forward_splits,
    get_test_split, normalize_features,
)
from stock_pred_rl_train import StockScorer


def get_gb_predictions(model, X, feature_cols):
    """Get gradient boosting probability predictions."""
    X_clean = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
    dmat = xgb.DMatrix(X_clean, feature_names=feature_cols)
    return model.predict(dmat)


def get_neural_scores(scorer_model, X_np):
    """Get neural scorer predictions."""
    X_clean = np.nan_to_num(X_np, nan=0.0, posinf=5.0, neginf=-5.0).clip(-5, 5)
    with torch.no_grad():
        scores = scorer_model(torch.FloatTensor(X_clean)).numpy()
    return scores


class EnsemblePredictor:
    """Ensemble model combining GB and neural scorer predictions."""

    def __init__(self, gb_model, neural_model, feature_cols, blend_alpha=0.6,
                 norm_means=None, norm_stds=None):
        self.gb_model = gb_model
        self.neural_model = neural_model
        self.feature_cols = feature_cols
        self.blend_alpha = blend_alpha
        self.norm_means = norm_means
        self.norm_stds = norm_stds

    def predict(self, dataset):
        """
        Score all stock-day observations.
        Returns: (ensemble_scores, gb_scores, neural_scores)
        """
        data = dataset.copy()

        if self.norm_means is not None and self.norm_stds is not None:
            data[self.feature_cols] = (
                (data[self.feature_cols] - self.norm_means) / self.norm_stds
            )
            data[self.feature_cols] = data[self.feature_cols].clip(-5, 5)

        X = data[self.feature_cols].values

        gb_scores = get_gb_predictions(self.gb_model, X, self.feature_cols)

        self.neural_model.eval()
        neural_scores = get_neural_scores(self.neural_model, X)

        ensemble_scores = (
            self.blend_alpha * gb_scores +
            (1 - self.blend_alpha) * neural_scores
        )

        return ensemble_scores, gb_scores, neural_scores

    def select_daily_stocks(self, dataset, top_k=3, min_score=0.3):
        """
        For each day, select top-K stocks with score above threshold.
        Returns: DataFrame with daily picks.
        """
        data = dataset.copy()
        ensemble_scores, gb_scores, neural_scores = self.predict(data)

        data["ensemble_score"] = ensemble_scores
        data["gb_score"] = gb_scores
        data["neural_score"] = neural_scores

        picks = []
        data["_date_str"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")

        for date_str in sorted(data["_date_str"].unique()):
            daily = data[data["_date_str"] == date_str]
            daily = daily[daily["ensemble_score"] >= min_score]

            if len(daily) == 0:
                continue

            top = daily.nlargest(top_k, "ensemble_score")
            for _, row in top.iterrows():
                picks.append({
                    "date": date_str,
                    "ticker": row["ticker"],
                    "ensemble_score": row["ensemble_score"],
                    "gb_score": row["gb_score"],
                    "neural_score": row["neural_score"],
                    "fwd_return_30d": row.get("fwd_return_30d", np.nan),
                    "label": row.get("label", np.nan),
                    "close": row.get("close", np.nan),
                })

        picks_df = pd.DataFrame(picks)
        return picks_df
