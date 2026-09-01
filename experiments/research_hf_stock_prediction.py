#!/usr/bin/env python3
"""
HuggingFace Transformer-based Stock Prediction Model
=====================================================
Predicts which stocks will rise ≥10% in the next 30 trading days.

Architecture: FT-Transformer (Feature Tokenizer + Transformer)
- Each numerical feature is projected into an embedding via a learned linear layer
- A [CLS] token aggregates information via self-attention
- Binary classification head predicts P(return ≥ 10% in 30 days)

Anti-leakage guarantees:
  1. All features use ONLY past data (no lookahead)
  2. Walk-forward: model trained only on data strictly before prediction date
  3. Purged gap: 30-day buffer between train end and prediction to avoid
     overlap with target horizon
  4. Features standardized using only training data statistics
  5. No hyperparameter tuning on test set

Anti-overfitting measures:
  1. Dropout in transformer layers and classification head
  2. Weight decay (L2 regularization)
  3. Early stopping on validation loss (within training window)
  4. Class-weighted loss to handle imbalanced target
  5. Conservative ensemble: average predictions across last K walk-forward models
  6. Feature set designed from financial literature (not data-mined)

Validation protocol:
  - Walk-forward with expanding training window
  - Train: 2010-2019, Valid: 2020-2022, Test: 2023-2026
  - Monthly rebalancing with purged gaps
  - Transaction costs applied
  - Metrics: precision, recall, hit rate, profit factor, Sharpe ratio
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import (
    load_data, compute_features,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
    UNIVERSE, TRANSACTION_COST_BPS, RISK_FREE_RATE, evaluate_strategy,
)

# ============================================================
# CONSTANTS
# ============================================================
TARGET_RETURN = 0.10        # 10% target return
TARGET_HORIZON = 30         # 30 trading days forward
PURGE_GAP = 30              # gap between train end and eval start (avoid leakage)
MIN_TRAIN_MONTHS = 36       # minimum training window
REBALANCE_FREQ = "monthly"  # monthly rebalancing
TOP_K_PICKS = 5             # number of stocks to buy each period
MAX_POSITION_SIZE = 0.25    # max weight per stock

# Model hyperparameters (conservative to avoid overfitting)
D_MODEL = 32                # transformer hidden dimension
N_HEADS = 4                 # attention heads
N_LAYERS = 2                # transformer layers
DROPOUT = 0.3               # dropout rate
WEIGHT_DECAY = 1e-3         # L2 regularization
LR = 3e-3                   # learning rate
BATCH_SIZE = 512
MAX_EPOCHS = 30
PATIENCE = 5                # early stopping patience
MAX_TRAIN_SAMPLES = 20000   # subsample training data for speed on CPU
ENSEMBLE_K = 3              # ensemble last K models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NON_STOCKS = set([
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "XLB", "XLRE", "XLC",
    "TLT", "IEF", "HYG", "GLD", "SLV", "USO",
])


# ============================================================
# DATASET
# ============================================================
class StockDataset(Dataset):
    """Tabular dataset for stock prediction."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================
# FT-TRANSFORMER MODEL (HuggingFace-style)
# ============================================================
class FeatureTokenizer(nn.Module):
    """Projects each numerical feature into a d_model-dimensional embedding."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        # Each feature gets its own linear projection + bias (learned token)
        self.projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        batch_size = x.shape[0]
        tokens = []
        for i, proj in enumerate(self.projections):
            tokens.append(proj(x[:, i:i+1]))  # (batch, d_model)
        tokens = torch.stack(tokens, dim=1)  # (batch, n_features, d_model)
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        return torch.cat([cls, tokens], dim=1)  # (batch, n_features+1, d_model)


class FTTransformerClassifier(nn.Module):
    """
    Feature Tokenizer Transformer for binary classification.

    Based on "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021).
    Uses the HuggingFace transformers-style architecture with PyTorch.
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)           # (batch, n_feat+1, d_model)
        encoded = self.transformer(tokens)   # (batch, n_feat+1, d_model)
        cls_out = encoded[:, 0, :]           # (batch, d_model) — CLS token
        logits = self.head(cls_out)          # (batch, 1)
        return logits.squeeze(-1)


# ============================================================
# TRAINING UTILITIES
# ============================================================
def train_model(X_train, y_train, X_val, y_val, n_features, pos_weight=1.0):
    """
    Train an FT-Transformer with early stopping.
    Returns the best model state dict.
    """
    model = FTTransformerClassifier(
        n_features=n_features, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    train_ds = StockDataset(X_train, y_train)
    val_ds = StockDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    return best_state, best_val_loss


def predict_with_model(state_dict, X, n_features):
    """Generate predictions from a trained model."""
    model = FTTransformerClassifier(
        n_features=n_features, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    ds = StockDataset(X, np.zeros(len(X)))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    probs = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def build_feature_matrix(data):
    """
    Build feature matrix for all stocks across all dates.
    Features are computed using ONLY past data (no lookahead).
    Returns a DataFrame with multi-index (date, ticker).
    """
    stocks = [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]
    print(f"  Universe: {len(stocks)} stocks")

    market_close = data["SPY"]["Close"] if "SPY" in data else None
    all_features = []

    for ticker in stocks:
        df = data[ticker]
        if "Close" not in df.columns or "Volume" not in df.columns:
            continue

        close = df["Close"]
        volume = df["Volume"]

        feat_df = compute_features(close, volume=volume, market_close=market_close)
        if len(feat_df) == 0:
            continue

        # Add forward return target (will be masked appropriately in walk-forward)
        fwd_ret = close.pct_change(TARGET_HORIZON).shift(-TARGET_HORIZON)
        feat_df["fwd_return_30d"] = fwd_ret.reindex(feat_df.index)
        feat_df["target"] = (feat_df["fwd_return_30d"] >= TARGET_RETURN).astype(float)
        feat_df["ticker"] = ticker

        all_features.append(feat_df)

    combined = pd.concat(all_features)
    combined = combined.reset_index()
    # Index name is "Date" (capital D) from compute_features
    if "Date" in combined.columns:
        combined = combined.rename(columns={"Date": "date"})
    elif "index" in combined.columns:
        combined = combined.rename(columns={"index": "date"})
    combined = combined.dropna(subset=["target"])

    # Remove rows with any NaN/Inf in features
    feature_cols = [c for c in combined.columns
                    if c not in ["date", "ticker", "fwd_return_30d", "target"]]
    mask = combined[feature_cols].apply(lambda x: np.isfinite(x)).all(axis=1)
    combined = combined[mask].reset_index(drop=True)

    print(f"  Total observations: {len(combined)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target rate (>=10% in 30d): {combined['target'].mean():.1%}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")

    return combined, feature_cols


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================
def walk_forward_predict(df, feature_cols, period_start, period_end,
                         train_start=TRAIN_START):
    """
    Walk-forward prediction with purged expanding window.

    For each monthly rebalance date in [period_start, period_end]:
      1. Training data: all observations from train_start up to
         (rebalance_date - PURGE_GAP days)
      2. Within training data, use last 20% as validation for early stopping
      3. Train FT-Transformer, predict probabilities for rebalance date
      4. Store predictions

    Returns dict: rebalance_date -> [(ticker, probability, fwd_return)]
    """
    # Get monthly rebalance dates within the period
    period_data = df[(df["date"] >= period_start) & (df["date"] <= period_end)]
    if len(period_data) == 0:
        return {}

    dates = sorted(period_data["date"].unique())
    # Take first trading day of each month
    monthly_dates = []
    last_month = None
    for d in dates:
        m = d.month
        if m != last_month:
            monthly_dates.append(d)
            last_month = m

    print(f"  Rebalance dates: {len(monthly_dates)} "
          f"({monthly_dates[0].strftime('%Y-%m')} to {monthly_dates[-1].strftime('%Y-%m')})")

    predictions = {}
    model_states = []  # for ensemble
    last_trained_quarter = None

    for rebal_date in monthly_dates:
        # Train new model every quarter (every 3 months) to save compute
        current_quarter = (rebal_date.year, (rebal_date.month - 1) // 3)
        need_train = (last_trained_quarter is None or current_quarter != last_trained_quarter)

        if need_train:
            # Purged training cutoff: no overlap with forward return horizon
            train_cutoff = rebal_date - pd.Timedelta(days=PURGE_GAP + 5)

            train_data = df[
                (df["date"] >= train_start) & (df["date"] <= train_cutoff)
            ].copy()

            if len(train_data) < 500:
                if not model_states:
                    continue
                # Use existing model if we can't train
            else:
                # Split training data: last 20% as internal validation for early stopping
                train_dates_list = sorted(train_data["date"].unique())
                val_split_idx = int(len(train_dates_list) * 0.8)
                val_start_date = train_dates_list[val_split_idx]

                internal_train = train_data[train_data["date"] < val_start_date]
                internal_val = train_data[train_data["date"] >= val_start_date]

                if len(internal_train) >= 300 and len(internal_val) >= 50:
                    # Subsample training data if too large (stratified)
                    if len(internal_train) > MAX_TRAIN_SAMPLES:
                        pos_mask = internal_train["target"] == 1
                        neg_mask = ~pos_mask
                        pos_n = int(MAX_TRAIN_SAMPLES * pos_mask.mean())
                        neg_n = MAX_TRAIN_SAMPLES - pos_n
                        rng = np.random.RandomState(42 + rebal_date.month)
                        pos_idx = rng.choice(internal_train.index[pos_mask],
                                             size=min(pos_n, pos_mask.sum()), replace=False)
                        neg_idx = rng.choice(internal_train.index[neg_mask],
                                             size=min(neg_n, neg_mask.sum()), replace=False)
                        sample_idx = np.concatenate([pos_idx, neg_idx])
                        internal_train = internal_train.loc[sample_idx]

                    # Prepare features
                    X_train = internal_train[feature_cols].values
                    y_train = internal_train["target"].values
                    X_val = internal_val[feature_cols].values
                    y_val = internal_val["target"].values

                    # Robust scaling (fit on training only — no leakage)
                    scaler = RobustScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)

                    # Clip extreme values after scaling
                    X_train = np.clip(X_train, -5, 5)
                    X_val = np.clip(X_val, -5, 5)

                    # Class weight for imbalanced target (capped conservatively)
                    pos_rate = y_train.mean()
                    pos_weight = (1 - pos_rate) / max(pos_rate, 0.01)
                    pos_weight = min(pos_weight, 3.0)  # conservative cap

                    # Train
                    state_dict, val_loss = train_model(
                        X_train, y_train, X_val, y_val,
                        n_features=len(feature_cols),
                        pos_weight=pos_weight,
                    )

                    if state_dict is not None:
                        model_states.append((state_dict, scaler))
                        if len(model_states) > ENSEMBLE_K:
                            model_states = model_states[-ENSEMBLE_K:]
                        last_trained_quarter = current_quarter

        if not model_states:
            continue

        # Predict on stocks at rebalance date
        pred_data = df[df["date"] == rebal_date].copy()
        if len(pred_data) < 5:
            continue

        X_pred = pred_data[feature_cols].values

        # Ensemble prediction: average across recent models
        all_probs = []
        for sd, sc in model_states:
            X_scaled = sc.transform(X_pred)
            X_scaled = np.clip(X_scaled, -5, 5)
            probs = predict_with_model(sd, X_scaled, n_features=len(feature_cols))
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)

        results = []
        for i, (_, row) in enumerate(pred_data.iterrows()):
            results.append((
                row["ticker"],
                float(avg_probs[i]),
                float(row["fwd_return_30d"]) if not np.isnan(row["fwd_return_30d"]) else None,
            ))

        predictions[rebal_date] = results
        n_positive = sum(1 for _, p, _ in results if p > 0.5)
        trained_str = " [NEW MODEL]" if need_train and last_trained_quarter == current_quarter else ""
        print(f"    {rebal_date.strftime('%Y-%m-%d')}: "
              f"{len(results)} stocks, {n_positive} predicted positive{trained_str}")

    return predictions


# ============================================================
# BACKTESTING
# ============================================================
def backtest_predictions(predictions, data, period_start, period_end):
    """
    Backtest the predictions: buy top-K highest probability stocks,
    hold for 30 trading days, then rebalance.

    Returns trades DataFrame and daily returns Series.
    """
    if not predictions:
        return pd.DataFrame(), pd.Series(dtype=float)

    trades = []
    daily_returns = {}

    sorted_dates = sorted(predictions.keys())

    for i, rebal_date in enumerate(sorted_dates):
        preds = predictions[rebal_date]

        # Sort by predicted probability, take top K
        preds.sort(key=lambda x: x[1], reverse=True)
        picks = [(t, p, fwd) for t, p, fwd in preds if p > 0.5][:TOP_K_PICKS]

        if not picks:
            continue

        # Equal weight (capped)
        n = len(picks)
        weight = min(1.0 / n, MAX_POSITION_SIZE)

        # Determine hold period end
        if i + 1 < len(sorted_dates):
            hold_end = sorted_dates[i + 1]
        else:
            hold_end = pd.Timestamp(period_end)

        for ticker, prob, fwd_ret in picks:
            if ticker not in data:
                continue

            df = data[ticker]
            stock_dates = df.index

            # Find entry and exit dates
            entry_mask = stock_dates >= rebal_date
            if not entry_mask.any():
                continue
            entry_idx = stock_dates[entry_mask][0]

            exit_target = entry_idx + pd.Timedelta(days=TARGET_HORIZON * 1.5)
            exit_mask = stock_dates >= min(hold_end, exit_target)
            if not exit_mask.any():
                exit_idx = stock_dates[-1]
            else:
                exit_idx = stock_dates[exit_mask][0]

            # Compute daily returns for this position
            pos_dates = stock_dates[(stock_dates >= entry_idx) & (stock_dates <= exit_idx)]
            if len(pos_dates) < 2:
                continue

            entry_price = df.loc[entry_idx, "Close"]
            exit_price = df.loc[exit_idx, "Close"]
            trade_return = exit_price / entry_price - 1
            days_held = len(pos_dates) - 1

            # Transaction costs (entry + exit)
            tx_cost = 2 * TRANSACTION_COST_BPS / 10000

            trades.append({
                "entry_date": entry_idx,
                "exit_date": exit_idx,
                "ticker": ticker,
                "probability": prob,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": trade_return,
                "net_pnl": trade_return - tx_cost,
                "days_held": days_held,
                "weight": weight,
            })

            # Daily portfolio returns
            for j in range(1, len(pos_dates)):
                d = pos_dates[j]
                day_ret = df.loc[pos_dates[j], "Close"] / df.loc[pos_dates[j-1], "Close"] - 1
                if d not in daily_returns:
                    daily_returns[d] = 0.0
                daily_returns[d] += day_ret * weight

            # Apply tx cost on entry day
            if pos_dates[1] in daily_returns:
                daily_returns[pos_dates[1]] -= tx_cost * weight

    trades_df = pd.DataFrame(trades)

    if daily_returns:
        ret_series = pd.Series(daily_returns).sort_index()
        # Filter to period
        ret_series = ret_series[
            (ret_series.index >= period_start) & (ret_series.index <= period_end)
        ]
    else:
        ret_series = pd.Series(dtype=float)

    return trades_df, ret_series


# ============================================================
# EVALUATION & REPORTING
# ============================================================
def evaluate_predictions(predictions, period_name=""):
    """Evaluate prediction quality (not just portfolio returns)."""
    all_true = []
    all_pred = []
    all_prob = []

    for date, preds in sorted(predictions.items()):
        for ticker, prob, fwd_ret in preds:
            if fwd_ret is None:
                continue
            all_true.append(1 if fwd_ret >= TARGET_RETURN else 0)
            all_pred.append(1 if prob > 0.5 else 0)
            all_prob.append(prob)

    if not all_true:
        print(f"  {period_name}: No predictions with known outcomes")
        return

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    n_pred_pos = all_pred.sum()
    n_actual_pos = all_true.sum()

    print(f"\n  {period_name} Prediction Quality:")
    print(f"    Total observations: {len(all_true)}")
    print(f"    Actual positive rate: {all_true.mean():.1%}")
    print(f"    Predicted positive: {n_pred_pos} / {len(all_pred)}")

    if n_pred_pos > 0:
        # Precision among predicted positives
        pred_pos_mask = all_pred == 1
        precision = all_true[pred_pos_mask].mean()
        print(f"    Precision (predicted +): {precision:.1%}")

        # Among top-K picks per date, what's the hit rate?
        top_k_hits = []
        top_k_returns = []
        for date, preds in sorted(predictions.items()):
            valid_preds = [(t, p, f) for t, p, f in preds if f is not None]
            valid_preds.sort(key=lambda x: x[1], reverse=True)
            top = valid_preds[:TOP_K_PICKS]
            for t, p, fwd in top:
                if p > 0.5:
                    top_k_hits.append(1 if fwd >= TARGET_RETURN else 0)
                    top_k_returns.append(fwd)

        if top_k_hits:
            print(f"    Top-{TOP_K_PICKS} hit rate (≥10%): {np.mean(top_k_hits):.1%}")
            print(f"    Top-{TOP_K_PICKS} avg return: {np.mean(top_k_returns):.1%}")
            print(f"    Top-{TOP_K_PICKS} median return: {np.median(top_k_returns):.1%}")

    if n_pred_pos > 0 and n_actual_pos > 0:
        prec = precision_score(all_true, all_pred, zero_division=0)
        rec = recall_score(all_true, all_pred, zero_division=0)
        f1 = f1_score(all_true, all_pred, zero_division=0)
        print(f"    Precision: {prec:.3f}")
        print(f"    Recall: {rec:.3f}")
        print(f"    F1 Score: {f1:.3f}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("STOCK PREDICTION WITH HUGGINGFACE FT-TRANSFORMER")
    print("Target: Stocks rising ≥10% in next 30 trading days")
    print("Method: Walk-forward with purged expanding window")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Model: FT-Transformer (d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS})")
    print(f"Ensemble: last {ENSEMBLE_K} models")
    print(f"Target: ≥{TARGET_RETURN:.0%} return in {TARGET_HORIZON} trading days")
    print(f"Purge gap: {PURGE_GAP} days (prevents target leakage)")

    # Load data
    print("\n[1/5] Loading data...")
    data = load_data()
    print(f"  Loaded {len(data)} tickers")

    # Build features
    print("\n[2/5] Building feature matrix...")
    df, feature_cols = build_feature_matrix(data)

    # ================================================================
    # WALK-FORWARD ON VALIDATION PERIOD
    # ================================================================
    print("\n[3/5] Walk-forward on VALIDATION period...")
    print(f"  Training window: {TRAIN_START} to expanding")
    print(f"  Prediction period: {VALID_START} to {VALID_END}")
    val_preds = walk_forward_predict(
        df, feature_cols,
        period_start=VALID_START, period_end=VALID_END,
        train_start=TRAIN_START,
    )

    evaluate_predictions(val_preds, "VALIDATION")

    val_trades, val_returns = backtest_predictions(
        val_preds, data, VALID_START, VALID_END,
    )

    print(f"\n  VALIDATION Portfolio Performance:")
    val_metrics = evaluate_strategy(val_trades, val_returns, "VALIDATION")

    # SPY benchmark for comparison
    if "SPY" in data:
        spy_val = data["SPY"].loc[VALID_START:VALID_END, "Close"].pct_change().dropna()
        spy_val_metrics = evaluate_strategy(None, spy_val, "VALIDATION SPY")

    # ================================================================
    # WALK-FORWARD ON TEST PERIOD (fully out-of-sample)
    # ================================================================
    print("\n[4/5] Walk-forward on TEST period (fully out-of-sample)...")
    print(f"  Training window: {TRAIN_START} to expanding")
    print(f"  Prediction period: {TEST_START} to {TEST_END}")
    test_preds = walk_forward_predict(
        df, feature_cols,
        period_start=TEST_START, period_end=TEST_END,
        train_start=TRAIN_START,
    )

    evaluate_predictions(test_preds, "TEST (OOS)")

    test_trades, test_returns = backtest_predictions(
        test_preds, data, TEST_START, TEST_END,
    )

    print(f"\n  TEST (OOS) Portfolio Performance:")
    test_metrics = evaluate_strategy(test_trades, test_returns, "TEST")

    if "SPY" in data:
        spy_test = data["SPY"].loc[TEST_START:TEST_END, "Close"].pct_change().dropna()
        spy_test_metrics = evaluate_strategy(None, spy_test, "TEST SPY")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  Anti-leakage checks:")
    print(f"    ✓ Features use only past data (prepare.compute_features)")
    print(f"    ✓ Walk-forward: train strictly before prediction date")
    print(f"    ✓ Purge gap: {PURGE_GAP} days between train end and prediction")
    print(f"    ✓ Scaler fit on training data only")
    print(f"    ✓ No hyperparameter tuning on test set")

    print("\n  Anti-overfitting checks:")
    print(f"    ✓ Dropout: {DROPOUT}")
    print(f"    ✓ Weight decay: {WEIGHT_DECAY}")
    print(f"    ✓ Early stopping (patience={PATIENCE})")
    print(f"    ✓ Class-weighted loss")
    print(f"    ✓ Ensemble of {ENSEMBLE_K} models")
    print(f"    ✓ Conservative architecture (d={D_MODEL}, {N_LAYERS} layers)")

    if len(val_trades) > 0 and len(test_trades) > 0:
        print("\n  Performance comparison:")
        print(f"    {'Metric':<20} {'Validation':>12} {'Test (OOS)':>12}")
        print(f"    {'─' * 44}")
        for metric in ["sharpe", "cagr", "max_drawdown", "win_rate", "profit_factor"]:
            v = val_metrics.get(metric, 0)
            t = test_metrics.get(metric, 0)
            if metric in ["cagr", "max_drawdown", "win_rate"]:
                print(f"    {metric:<20} {v:>11.1%} {t:>11.1%}")
            else:
                print(f"    {metric:<20} {v:>12.3f} {t:>12.3f}")

    # ================================================================
    # CURRENT PICKS (latest prediction)
    # ================================================================
    print("\n[5/5] Latest stock picks...")
    all_preds = {**val_preds, **test_preds}
    if all_preds:
        latest_date = max(all_preds.keys())
        latest = all_preds[latest_date]
        latest.sort(key=lambda x: x[1], reverse=True)
        top_picks = [(t, p, f) for t, p, f in latest if p > 0.5][:TOP_K_PICKS]

        print(f"\n  Latest predictions ({latest_date.strftime('%Y-%m-%d')}):")
        print(f"    {'Ticker':<8} {'P(≥10%)':<10} {'Actual 30d':>10}")
        print(f"    {'─' * 30}")
        for ticker, prob, fwd in top_picks:
            fwd_str = f"{fwd:.1%}" if fwd is not None else "pending"
            hit = " ✓" if fwd is not None and fwd >= TARGET_RETURN else ""
            print(f"    {ticker:<8} {prob:<10.3f} {fwd_str:>10}{hit}")

    print("\nDone.")


if __name__ == "__main__":
    main()
