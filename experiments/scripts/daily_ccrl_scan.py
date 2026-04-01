#!/usr/bin/env python3
"""
CCRL Daily Top-5 Stock Picker — Scanner
=========================================
Runs the CCRL (Conviction Cascade Reinforcement Learning) model
to select the top 5 stocks most likely to rise 10%+ in 30 days.

Outputs JSON for the experiments webapp.

Usage:
    cd experiments && python scripts/daily_ccrl_scan.py
"""

import os
import sys
import json
import math
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

import numpy as np
import pandas as pd

from prepare import load_data, compute_features, UNIVERSE
from src.ai_features import (
    compute_ccrl_features,
    temporal_attention_momentum,
    asymmetric_tail_features,
    cascade_propagation_features,
    microstructure_regime_fingerprint,
)
from src.rl_stock_selector import EnsemblePredictionLayer, CCRLConfig
from research_rl_stock_prediction import (
    SECTOR_PEERS, get_peer_group, TARGET_RETURN, TARGET_HORIZON,
)

# ETFs excluded from stock picks
EXCLUDED = {
    "SPY", "VIX", "TLT", "IEF", "HYG", "GLD", "SLV", "USO",
    "DIA", "IWM", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
}

TOP_K = 5


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return str(obj)

    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: self._sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [self._sanitize(v) for v in o]
        return o


def main():
    print("CCRL Daily Top-5 Scanner")
    print("=" * 50)

    # ---- Load data ----
    print("Loading data...")
    data_dict = load_data()
    print(f"  {len(data_dict)} tickers loaded")

    market_close = data_dict.get("SPY", pd.DataFrame()).get("Close")

    # ---- Build training dataset (2010-2022) ----
    print("Building training dataset (2010-2022)...")
    train_end = "2022-12-31"

    close_dict = {}
    for ticker, df in data_dict.items():
        if "Close" in df.columns and len(df) > 500:
            close_dict[ticker] = df["Close"]

    # MRF
    print("  Computing MRF...")
    mrf_cache = {}
    try:
        mrf_cache = microstructure_regime_fingerprint(close_dict, n_components=5)
    except Exception as e:
        print(f"  MRF failed: {e}")

    stocks = [t for t in UNIVERSE if t in data_dict and t not in EXCLUDED]

    # Build feature matrix for training
    all_X, all_y, feature_names = [], [], None
    print(f"  Computing features for {len(stocks)} stocks...")
    for i, ticker in enumerate(stocks):
        df = data_dict[ticker]
        if "Close" not in df.columns:
            continue
        close = df["Close"]
        volume = df.get("Volume")

        peers = get_peer_group(ticker)
        peer_closes = {}
        for p in peers[:5]:
            if p in data_dict and "Close" in data_dict[p].columns:
                peer_closes[p] = data_dict[p]["Close"]

        try:
            stock_mrf = mrf_cache.get(ticker)
            features = compute_ccrl_features(
                close, volume, market_close,
                peer_closes=peer_closes, mrf_cache=stock_mrf,
            )
        except Exception:
            continue

        # Training split: up to train_end
        features_train = features.loc[:"2022-12-31"]
        if len(features_train) < 100:
            continue

        # Labels
        fwd_ret = close.pct_change(TARGET_HORIZON).shift(-TARGET_HORIZON)
        labels = (fwd_ret >= TARGET_RETURN).astype(int)
        common = features_train.index.intersection(labels.dropna().index)
        features_train = features_train.loc[common]
        labels_train = labels.loc[common]

        if len(features_train) < 50:
            continue

        if feature_names is None:
            feature_names = features_train.columns.tolist()
        else:
            for col in features_train.columns:
                if col not in feature_names:
                    feature_names.append(col)
        features_train = features_train.reindex(columns=feature_names)

        all_X.append(features_train.values)
        all_y.append(labels_train.values)

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(stocks)}...")

    if not all_X:
        print("ERROR: No training data")
        return

    n_features = len(feature_names)
    for i in range(len(all_X)):
        if all_X[i].shape[1] < n_features:
            pad = np.full((all_X[i].shape[0], n_features - all_X[i].shape[1]), np.nan)
            all_X[i] = np.hstack([all_X[i], pad])

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features, "
          f"pos rate: {y_train.mean():.3f}")

    # ---- Train ensemble ----
    print("Training CCRL ensemble...")
    config = CCRLConfig()
    ensemble = EnsemblePredictionLayer(config)
    ensemble.fit(X_train, y_train, feature_names)

    # ---- Score today's candidates ----
    print("Scoring today's stocks...")
    today_candidates = []

    for ticker in stocks:
        df = data_dict[ticker]
        if "Close" not in df.columns:
            continue
        close = df["Close"]
        volume = df.get("Volume")

        peers = get_peer_group(ticker)
        peer_closes = {}
        for p in peers[:5]:
            if p in data_dict and "Close" in data_dict[p].columns:
                peer_closes[p] = data_dict[p]["Close"]

        try:
            stock_mrf = mrf_cache.get(ticker)
            features = compute_ccrl_features(
                close, volume, market_close,
                peer_closes=peer_closes, mrf_cache=stock_mrf,
            )
        except Exception:
            continue

        if len(features) < 1:
            continue

        # Get latest feature vector
        latest = features.iloc[-1:]
        latest = latest.reindex(columns=feature_names)
        X_today = latest.values

        if X_today.shape[1] < n_features:
            pad = np.full((1, n_features - X_today.shape[1]), np.nan)
            X_today = np.hstack([X_today, pad])

        try:
            probas, mean_p, std_p = ensemble.predict_proba(X_today)
            score = float(mean_p[0])
            uncertainty = float(std_p[0])
        except Exception:
            continue

        price = float(close.iloc[-1])
        feat_dict = features.iloc[-1].to_dict()

        today_candidates.append({
            "ticker": ticker,
            "score": score,
            "uncertainty": uncertainty,
            "price": round(price, 2),
            "conviction": round(score * (1 - uncertainty) * 100, 1),
            "returns": {
                "5d": round(float(feat_dict.get("ret_5d", 0)) * 100, 1),
                "21d": round(float(feat_dict.get("ret_21d", 0)) * 100, 1),
                "63d": round(float(feat_dict.get("ret_63d", 0)) * 100, 1),
                "126d": round(float(feat_dict.get("ret_126d", 0)) * 100, 1),
            },
            "vol_21d": round(float(feat_dict.get("vol_21d", 0)) * 100, 1),
            "drawdown": round(float(feat_dict.get("drawdown_252d", 0)) * 100, 1),
            "position_in_range": round(float(feat_dict.get("position_in_52w_range", 0)) * 100, 1),
            "atf_coiled_spring": round(float(feat_dict.get("atf_coiled_spring", 0)) * 100, 1),
            "tam": round(float(feat_dict.get("tam", 0)) * 100, 2),
        })

    today_candidates.sort(key=lambda c: c["score"], reverse=True)

    top5 = today_candidates[:TOP_K]
    print(f"\nTop {TOP_K} picks:")
    for i, pick in enumerate(top5):
        print(f"  {i+1}. {pick['ticker']:>5} — score={pick['score']:.3f}, "
              f"conviction={pick['conviction']:.0f}%, "
              f"price=${pick['price']}, vol={pick['vol_21d']}%")

    # ---- SPY regime ----
    spy_feat = {}
    if "SPY" in data_dict:
        spy_close = data_dict["SPY"]["Close"]
        sma100 = spy_close.rolling(100).mean()
        spy_price = float(spy_close.iloc[-1])
        spy_sma100 = float(sma100.iloc[-1])
        is_bull = spy_price > spy_sma100
        spy_feat = {
            "price": round(spy_price, 2),
            "sma100": round(spy_sma100, 2),
            "regime": "BULL" if is_bull else "BEAR",
            "ret_21d": round(float(spy_close.pct_change(21).iloc[-1]) * 100, 1),
            "ret_63d": round(float(spy_close.pct_change(63).iloc[-1]) * 100, 1),
        }

    # ---- Write output ----
    docs_dir = os.path.join(EXPERIMENTS_DIR, "docs", "data")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(os.path.join(docs_dir, "tickers"), exist_ok=True)

    # Performance stats from our validated results
    performance = {
        "walk_forward_cv": {
            "mean_auc": 0.729, "std_auc": 0.017,
            "mean_top10_precision": 0.60, "mean_top10_lift": 5.66,
        },
        "validation_2020_2022": {
            "auc": 0.686, "top10_precision": 1.00, "top10_lift": 5.23,
            "top20_precision": 1.00, "top50_precision": 0.94,
        },
        "test_2023_2026": {
            "auc": 0.678, "top10_precision": 0.70, "top10_lift": 4.36,
            "top20_precision": 0.65, "top50_precision": 0.34,
        },
        "trading_simulation": {
            "total_trades": 360, "mean_net_return": 0.0488,
            "win_rate": 0.567, "hit_rate_10pct": 0.311,
            "profit_factor": 2.31, "sharpe": 1.85,
            "annualized_return": 0.586, "alpha_vs_spy": 0.0258,
        },
    }

    full_data = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "CCRL",
        "strategy_full_name": "Conviction Cascade Reinforcement Learning",
        "target": f"Stocks rising {int(TARGET_RETURN*100)}%+ in {TARGET_HORIZON} trading days",
        "model": {
            "ensemble_size": 5,
            "models": ["GBM", "HistGBM", "RandomForest", "LogisticRegression", "Ridge"],
            "features": len(feature_names),
            "training_samples": int(X_train.shape[0]),
            "training_positive_rate": round(float(y_train.mean()), 4),
        },
        "spy": spy_feat,
        "top5": top5,
        "all_candidates": today_candidates[:50],  # Top 50 for the table
        "performance": performance,
    }

    with open(os.path.join(docs_dir, "ccrl.json"), "w") as f:
        json.dump(full_data, f, indent=2, cls=SafeJSONEncoder)

    # Per-ticker files for top 30
    for stock in today_candidates[:30]:
        ticker = stock["ticker"]
        if ticker not in data_dict:
            continue
        chart = [{"date": str(dt.date()), "price": round(float(row["Close"]), 2)}
                 for dt, row in data_dict[ticker].tail(252).iterrows()]
        with open(os.path.join(docs_dir, "tickers", f"{ticker}.json"), "w") as f:
            json.dump({**stock, "chart": chart}, f, indent=2, cls=SafeJSONEncoder)

    with open(os.path.join(docs_dir, "last_run.txt"), "w") as f:
        f.write(datetime.datetime.now().isoformat())

    print(f"\nData written to {docs_dir}/")
    print(f"  ccrl.json — main dashboard data")
    print(f"  tickers/*.json — per-ticker charts")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
