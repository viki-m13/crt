"""
Purged Walk-Forward Cross-Validation for CCRL
===============================================
PROPRIETARY / PATENTABLE

Implements a rigorous validation framework specifically designed for
stock prediction with a 30-day forward target. Standard cross-validation
fails catastrophically for financial time series because:

1. Temporal autocorrelation → train/test contamination
2. Overlapping labels → same information in train and test
3. Market regimes → performance varies across regimes

Our framework addresses ALL three issues:

PURGED WALK-FORWARD CV (Novel Combination):
=============================================

1. PURGING: When constructing train/test splits, we REMOVE all training
   samples whose labels overlap with the test period. For a 30-day
   forward label, we purge the last 30 days of training data before
   each test fold. This prevents any label leakage.

2. EMBARGO: Additional buffer period AFTER purging to account for
   serial correlation in features. Even after label purging, features
   computed from recent data may contain information about the test
   period through autocorrelation.

3. EXPANDING WINDOW: Training window grows over time (never shrinks),
   simulating real-world deployment where you always have more data.

4. REGIME-STRATIFIED EVALUATION: Results are broken down by market
   regime to verify the model works in ALL conditions, not just one.

5. COMBINATORIAL PURGED CV (CPCV): In addition to walk-forward,
   we implement combinatorial purged CV which generates MORE test paths
   from the same data, giving tighter confidence intervals.

Patent claim: Combined purged-embargo walk-forward cross-validation
with regime-stratified evaluation for financial time series prediction,
specifically adapted for asymmetric large-return event detection.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss,
)


@dataclass
class PurgedCVConfig:
    """Configuration for purged walk-forward CV."""

    # Label horizon (must match target_horizon in CCRLConfig)
    label_horizon: int = 30  # trading days

    # Purge buffer: remove this many days from end of training
    # Must be >= label_horizon to prevent label leakage
    purge_days: int = 30

    # Embargo buffer: additional days removed AFTER purge
    # Accounts for feature autocorrelation
    embargo_days: int = 10

    # Walk-forward settings
    min_train_days: int = 504      # Minimum 2 years of training data
    test_window_days: int = 126    # 6 months test window per fold
    step_days: int = 63            # Advance 3 months between folds

    # Regime detection
    vol_regime_threshold_high: float = 0.20  # Annualized vol > 20% = high vol
    vol_regime_threshold_low: float = 0.12   # Annualized vol < 12% = low vol
    trend_lookback: int = 252                # 1 year for trend detection


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation Engine.

    Generates train/test splits with proper purging and embargo
    to prevent ALL forms of data leakage in financial time series.
    """

    def __init__(self, config: Optional[PurgedCVConfig] = None):
        self.config = config or PurgedCVConfig()

    def split(self, dates, y=None):
        """
        Generate purged walk-forward splits.

        Parameters:
        - dates: pd.DatetimeIndex of all available dates
        - y: optional labels (not used for splitting, but can validate)

        Yields: (train_indices, test_indices, fold_info) tuples
        """
        cfg = self.config
        n = len(dates)
        total_buffer = cfg.purge_days + cfg.embargo_days

        fold = 0
        train_end = cfg.min_train_days

        while train_end + total_buffer + cfg.test_window_days <= n:
            # Training indices: [0, train_end - purge_days)
            # We purge the last purge_days of training to prevent label overlap
            train_idx = np.arange(0, train_end - cfg.purge_days)

            # Test indices: [train_end + embargo_days, train_end + embargo_days + test_window)
            test_start = train_end + cfg.embargo_days
            test_end = min(test_start + cfg.test_window_days, n)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) < cfg.min_train_days // 2:
                train_end += cfg.step_days
                continue

            if len(test_idx) < cfg.test_window_days // 4:
                break

            fold_info = {
                "fold": fold,
                "train_start": dates[train_idx[0]],
                "train_end": dates[train_idx[-1]],
                "purge_start": dates[train_end - cfg.purge_days] if train_end - cfg.purge_days < n else None,
                "test_start": dates[test_idx[0]],
                "test_end": dates[test_idx[-1]],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "gap_days": total_buffer,
            }

            yield train_idx, test_idx, fold_info

            train_end += cfg.step_days
            fold += 1

    def detect_regime(self, close, dates):
        """
        Detect market regime for each date.

        Returns: pd.Series with regime labels ("bull", "bear", "high_vol", "low_vol")
        """
        cfg = self.config
        ret = close.pct_change()
        vol = ret.rolling(21).std() * np.sqrt(252)
        trend = close.pct_change(cfg.trend_lookback)

        regimes = pd.Series("normal", index=dates)

        for i, date in enumerate(dates):
            if date not in vol.index or date not in trend.index:
                continue
            v = vol.get(date, np.nan)
            t = trend.get(date, np.nan)

            if np.isnan(v) or np.isnan(t):
                continue

            if v > cfg.vol_regime_threshold_high:
                regimes.iloc[i] = "high_vol"
            elif v < cfg.vol_regime_threshold_low:
                regimes.iloc[i] = "low_vol"
            elif t > 0.10:
                regimes.iloc[i] = "bull"
            elif t < -0.10:
                regimes.iloc[i] = "bear"

        return regimes


def evaluate_fold(y_true, y_pred_proba, threshold=0.5):
    """
    Evaluate a single CV fold with metrics relevant to 10%+ prediction.

    Returns dict of metrics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {}

    # Classification metrics
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["brier"] = brier_score_loss(y_true, y_pred_proba)
    else:
        metrics["auc"] = 0.5
        metrics["brier"] = 0.25

    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # Custom metrics for our use case
    n_predicted_positive = y_pred.sum()
    n_actual_positive = y_true.sum()
    metrics["n_predicted_positive"] = int(n_predicted_positive)
    metrics["n_actual_positive"] = int(n_actual_positive)
    metrics["positive_rate"] = float(y_true.mean())

    # Hit rate at different thresholds
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_t = y_pred_proba >= t
        if pred_t.sum() > 0:
            metrics[f"precision_at_{t}"] = float(y_true[pred_t].mean())
            metrics[f"n_picks_at_{t}"] = int(pred_t.sum())
        else:
            metrics[f"precision_at_{t}"] = 0.0
            metrics[f"n_picks_at_{t}"] = 0

    return metrics


def run_purged_walkforward_cv(X, y, dates, feature_names=None,
                               market_close=None, config=None,
                               verbose=True):
    """
    Run the full purged walk-forward cross-validation.

    Parameters:
    - X: np.ndarray (n_samples, n_features)
    - y: np.ndarray binary labels
    - dates: pd.DatetimeIndex
    - feature_names: list of feature name strings
    - market_close: pd.Series for regime detection
    - config: PurgedCVConfig
    - verbose: print progress

    Returns:
    - results: dict with fold-by-fold and aggregate metrics
    """
    from .rl_stock_selector import EnsemblePredictionLayer, CCRLConfig

    if config is None:
        config = PurgedCVConfig()

    cv = PurgedWalkForwardCV(config)
    ccrl_config = CCRLConfig()

    all_fold_metrics = []
    all_predictions = []
    regime_metrics = {}

    # Detect regimes if market data available
    regimes = None
    if market_close is not None:
        regimes = cv.detect_regime(market_close, dates)

    if verbose:
        print("=" * 70)
        print("PURGED WALK-FORWARD CROSS-VALIDATION")
        print("=" * 70)
        print(f"  Samples: {len(y)}")
        print(f"  Positive rate: {y.mean():.3f}")
        print(f"  Purge: {config.purge_days} days")
        print(f"  Embargo: {config.embargo_days} days")
        print(f"  Total gap: {config.purge_days + config.embargo_days} days")
        print()

    for train_idx, test_idx, fold_info in cv.split(dates, y):
        fold = fold_info["fold"]

        if verbose:
            print(f"--- Fold {fold} ---")
            print(f"  Train: {fold_info['train_start'].date()} to "
                  f"{fold_info['train_end'].date()} ({fold_info['n_train']} samples)")
            print(f"  Test:  {fold_info['test_start'].date()} to "
                  f"{fold_info['test_end'].date()} ({fold_info['n_test']} samples)")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Check minimum samples
        if len(y_train) < config.min_train_days // 2:
            if verbose:
                print(f"  SKIPPED: insufficient training data")
            continue

        if y_train.sum() < 10:
            if verbose:
                print(f"  SKIPPED: too few positive examples ({y_train.sum()})")
            continue

        # Train ensemble
        ensemble = EnsemblePredictionLayer(ccrl_config)
        try:
            ensemble.fit(X_train, y_train, feature_names)
        except Exception as e:
            if verbose:
                print(f"  TRAINING FAILED: {e}")
            continue

        # Predict on test
        try:
            probas, mean_proba, std_proba = ensemble.predict_proba(X_test)
        except Exception as e:
            if verbose:
                print(f"  PREDICTION FAILED: {e}")
            continue

        # Evaluate
        fold_metrics = evaluate_fold(y_test, mean_proba)
        fold_metrics.update(fold_info)
        fold_metrics["train_positive_rate"] = float(y_train.mean())
        fold_metrics["test_positive_rate"] = float(y_test.mean())

        # Regime-specific evaluation
        if regimes is not None:
            test_regimes = regimes.iloc[test_idx].values
            for regime in ["bull", "bear", "high_vol", "low_vol", "normal"]:
                regime_mask = test_regimes == regime
                if regime_mask.sum() >= 10:
                    regime_y = y_test[regime_mask]
                    regime_p = mean_proba[regime_mask]
                    if len(np.unique(regime_y)) > 1:
                        fold_metrics[f"auc_{regime}"] = roc_auc_score(regime_y, regime_p)
                    fold_metrics[f"n_{regime}"] = int(regime_mask.sum())
                    fold_metrics[f"precision_{regime}"] = float(
                        precision_score(regime_y, (regime_p >= 0.5).astype(int),
                                        zero_division=0)
                    )

        all_fold_metrics.append(fold_metrics)

        # Store predictions for aggregate analysis
        for i, idx in enumerate(test_idx):
            all_predictions.append({
                "date": dates[idx],
                "y_true": y_test[i],
                "y_pred_proba": mean_proba[i],
                "std_proba": std_proba[i],
                "fold": fold,
                "regime": regimes.iloc[idx] if regimes is not None else "unknown",
            })

        if verbose:
            print(f"  AUC: {fold_metrics.get('auc', 0):.3f}, "
                  f"Precision@0.5: {fold_metrics.get('precision_at_0.5', 0):.3f}, "
                  f"Recall: {fold_metrics.get('recall', 0):.3f}")
            print(f"  Picks@0.5: {fold_metrics.get('n_picks_at_0.5', 0)}, "
                  f"Precision@0.6: {fold_metrics.get('precision_at_0.6', 0):.3f}")

    # Aggregate results
    if not all_fold_metrics:
        print("  NO VALID FOLDS — check data availability")
        return {"folds": [], "aggregate": {}, "predictions": []}

    folds_df = pd.DataFrame(all_fold_metrics)
    preds_df = pd.DataFrame(all_predictions)

    aggregate = {
        "n_folds": len(all_fold_metrics),
        "mean_auc": float(folds_df["auc"].mean()),
        "std_auc": float(folds_df["auc"].std()),
        "mean_precision": float(folds_df["precision"].mean()),
        "mean_recall": float(folds_df["recall"].mean()),
        "mean_f1": float(folds_df["f1"].mean()),
        "mean_brier": float(folds_df["brier"].mean()),
    }

    # Precision at various thresholds (aggregate across all folds)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        key = f"precision_at_{t}"
        if key in folds_df.columns:
            aggregate[f"mean_{key}"] = float(folds_df[key].mean())

    # Regime-specific aggregate
    for regime in ["bull", "bear", "high_vol", "low_vol", "normal"]:
        key = f"auc_{regime}"
        if key in folds_df.columns:
            vals = folds_df[key].dropna()
            if len(vals) > 0:
                aggregate[f"mean_auc_{regime}"] = float(vals.mean())

    if verbose:
        print()
        print("=" * 70)
        print("AGGREGATE RESULTS")
        print("=" * 70)
        print(f"  Folds completed:     {aggregate['n_folds']}")
        print(f"  Mean AUC:            {aggregate['mean_auc']:.3f} "
              f"(+/- {aggregate['std_auc']:.3f})")
        print(f"  Mean Precision:      {aggregate['mean_precision']:.3f}")
        print(f"  Mean Recall:         {aggregate['mean_recall']:.3f}")
        print(f"  Mean F1:             {aggregate['mean_f1']:.3f}")
        print(f"  Mean Brier Score:    {aggregate['mean_brier']:.3f}")
        print()

        # Regime breakdown
        print("  Regime Breakdown:")
        for regime in ["bull", "bear", "high_vol", "low_vol", "normal"]:
            key = f"mean_auc_{regime}"
            if key in aggregate:
                print(f"    {regime:>10}: AUC = {aggregate[key]:.3f}")

        # Overfitting check
        print()
        print("  OVERFITTING DIAGNOSTICS:")
        if aggregate["std_auc"] > 0.10:
            print("    WARNING: High AUC variance across folds "
                  f"(std={aggregate['std_auc']:.3f})")
        else:
            print(f"    OK: AUC variance is acceptable (std={aggregate['std_auc']:.3f})")

        # Check for degradation over time
        if len(folds_df) >= 3:
            first_half_auc = folds_df.iloc[:len(folds_df)//2]["auc"].mean()
            second_half_auc = folds_df.iloc[len(folds_df)//2:]["auc"].mean()
            degradation = first_half_auc - second_half_auc
            if degradation > 0.05:
                print(f"    WARNING: Performance degrades over time "
                      f"(early AUC={first_half_auc:.3f}, late AUC={second_half_auc:.3f})")
            else:
                print(f"    OK: No significant degradation "
                      f"(early={first_half_auc:.3f}, late={second_half_auc:.3f})")

        # Check regime stability
        regime_aucs = []
        for regime in ["bull", "bear", "high_vol", "low_vol"]:
            key = f"mean_auc_{regime}"
            if key in aggregate:
                regime_aucs.append(aggregate[key])
        if len(regime_aucs) >= 2:
            regime_spread = max(regime_aucs) - min(regime_aucs)
            if regime_spread > 0.15:
                print(f"    WARNING: Large regime sensitivity "
                      f"(AUC spread={regime_spread:.3f})")
            else:
                print(f"    OK: Regime-robust (AUC spread={regime_spread:.3f})")

    results = {
        "folds": all_fold_metrics,
        "aggregate": aggregate,
        "predictions": all_predictions,
    }

    return results


def run_combinatorial_purged_cv(X, y, dates, feature_names=None,
                                 n_groups=6, n_test_groups=2,
                                 config=None, verbose=True):
    """
    Combinatorial Purged Cross-Validation (CPCV)
    ==============================================
    NOVEL IMPLEMENTATION

    Standard walk-forward gives few test paths. CPCV generates C(n,k)
    combinations of test groups, each properly purged, giving many
    more independent estimates of out-of-sample performance.

    Reference: de Prado, "Advances in Financial ML" Ch. 12
    Enhanced with our embargo and regime-stratified evaluation.

    Parameters:
    - n_groups: number of time groups to divide data into
    - n_test_groups: number of groups used for testing in each combination
    """
    from .rl_stock_selector import EnsemblePredictionLayer, CCRLConfig
    from itertools import combinations

    if config is None:
        config = PurgedCVConfig()

    ccrl_config = CCRLConfig()

    n = len(dates)
    group_size = n // n_groups
    total_buffer = config.purge_days + config.embargo_days

    # Create group boundaries
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n
        groups.append((start, end))

    # Generate all combinations of test groups
    combos = list(combinations(range(n_groups), n_test_groups))

    if verbose:
        print("=" * 70)
        print("COMBINATORIAL PURGED CROSS-VALIDATION")
        print("=" * 70)
        print(f"  Groups: {n_groups}, Test groups per combo: {n_test_groups}")
        print(f"  Total combinations: {len(combos)}")
        print(f"  Group size: ~{group_size} samples")
        print()

    all_fold_metrics = []

    for combo_idx, test_groups in enumerate(combos):
        # Test indices: union of selected groups
        test_idx = np.concatenate([
            np.arange(groups[g][0], groups[g][1]) for g in test_groups
        ])

        # Training indices: all other groups, with purging
        train_groups = [g for g in range(n_groups) if g not in test_groups]
        train_idx_list = []
        for g in train_groups:
            g_start, g_end = groups[g]
            # Check if this group is adjacent to a test group
            purge_end = g_end
            for tg in test_groups:
                tg_start = groups[tg][0]
                # If this train group immediately precedes a test group, purge
                if g_end <= tg_start and tg_start - g_end < total_buffer:
                    purge_end = max(g_start, g_end - total_buffer)
                    break
            train_idx_list.append(np.arange(g_start, purge_end))

        if not train_idx_list:
            continue
        train_idx = np.concatenate(train_idx_list)

        if len(train_idx) < config.min_train_days // 2:
            continue

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if y_train.sum() < 10:
            continue

        # Train and predict
        ensemble = EnsemblePredictionLayer(ccrl_config)
        try:
            ensemble.fit(X_train, y_train, feature_names)
            _, mean_proba, _ = ensemble.predict_proba(X_test)
            fold_metrics = evaluate_fold(y_test, mean_proba)
            fold_metrics["combo"] = test_groups
            fold_metrics["n_train"] = len(train_idx)
            fold_metrics["n_test"] = len(test_idx)
            all_fold_metrics.append(fold_metrics)

            if verbose and combo_idx % 3 == 0:
                print(f"  Combo {combo_idx+1}/{len(combos)}: "
                      f"AUC={fold_metrics.get('auc', 0):.3f}")
        except Exception as e:
            if verbose:
                print(f"  Combo {combo_idx+1} failed: {e}")

    if not all_fold_metrics:
        return {"folds": [], "aggregate": {}}

    folds_df = pd.DataFrame(all_fold_metrics)
    aggregate = {
        "n_combos": len(all_fold_metrics),
        "mean_auc": float(folds_df["auc"].mean()),
        "std_auc": float(folds_df["auc"].std()),
        "mean_precision": float(folds_df["precision"].mean()),
        "mean_recall": float(folds_df["recall"].mean()),
        "ci95_auc_low": float(folds_df["auc"].quantile(0.025)),
        "ci95_auc_high": float(folds_df["auc"].quantile(0.975)),
    }

    if verbose:
        print()
        print(f"  CPCV Results ({len(all_fold_metrics)} combinations):")
        print(f"    Mean AUC:      {aggregate['mean_auc']:.3f} "
              f"(+/- {aggregate['std_auc']:.3f})")
        print(f"    95% CI AUC:    [{aggregate['ci95_auc_low']:.3f}, "
              f"{aggregate['ci95_auc_high']:.3f}]")
        print(f"    Mean Precision: {aggregate['mean_precision']:.3f}")

    return {"folds": all_fold_metrics, "aggregate": aggregate}
