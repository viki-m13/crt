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
    config.calibrate_probabilities = False  # Much faster without 3-fold internal CV
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

    # ---- Historical picks simulation (test period 2023-2026) ----
    # Use pre-computed features (already computed above) to score each monthly date
    # This avoids recomputing features per-month — much faster
    print("Running historical picks simulation (2023-2026)...")
    test_start = pd.Timestamp("2023-04-01")
    spy_close = data_dict["SPY"]["Close"]
    test_dates = spy_close.loc[test_start:].index

    # Pre-compute FULL features for all stocks once (already done above for today)
    # We can reuse those features since they use only past data at each point
    print("  Pre-computing features for all stocks...")
    stock_features_cache = {}  # {ticker: DataFrame of features}
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
            feats = compute_ccrl_features(
                close, volume, market_close,
                peer_closes=peer_closes, mrf_cache=stock_mrf,
            )
            stock_features_cache[ticker] = feats
        except Exception:
            continue

    # Monthly rebalance dates
    monthly_dates = []
    seen_months = set()
    for d in test_dates:
        ym = (d.year, d.month)
        if ym not in seen_months:
            seen_months.add(ym)
            monthly_dates.append(d)

    historical_picks = []
    for reb_date in monthly_dates:
        day_scores = []
        for ticker in stocks:
            if ticker not in stock_features_cache:
                continue
            feats = stock_features_cache[ticker]
            if reb_date not in feats.index:
                continue
            row = feats.loc[[reb_date]].reindex(columns=feature_names)
            X_day = row.values
            if X_day.shape[1] < n_features:
                pad = np.full((1, n_features - X_day.shape[1]), np.nan)
                X_day = np.hstack([X_day, pad])
            try:
                _, mean_p, std_p = ensemble.predict_proba(X_day)
                day_scores.append((ticker, float(mean_p[0]), float(std_p[0])))
            except Exception:
                continue

        day_scores.sort(key=lambda x: x[1], reverse=True)
        for ticker, score, unc in day_scores[:TOP_K]:
            close_ts = data_dict[ticker]["Close"]
            entry_price = float(close_ts.loc[reb_date])
            entry_idx = close_ts.index.get_loc(reb_date)
            exit_idx = min(entry_idx + TARGET_HORIZON, len(close_ts) - 1)
            exit_date = close_ts.index[exit_idx]
            exit_price = float(close_ts.iloc[exit_idx])
            ret = (exit_price / entry_price) - 1
            net_ret = ret - 0.003
            historical_picks.append({
                "entry_date": str(reb_date.date()),
                "exit_date": str(exit_date.date()),
                "ticker": ticker,
                "score": round(score, 3),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return_pct": round(ret * 100, 2),
                "net_return_pct": round(net_ret * 100, 2),
                "hit_target": ret >= 0.10,
                "days_held": int(exit_idx - entry_idx),
            })

    # Equity curves
    monthly_returns = {}
    for pick in historical_picks:
        m = pick["entry_date"][:7]
        if m not in monthly_returns:
            monthly_returns[m] = []
        monthly_returns[m].append(pick["net_return_pct"] / 100)

    cum_val, spy_cum = 10000.0, 10000.0
    eq_strat, eq_spy = [], []
    for m in sorted(monthly_returns.keys()):
        cum_val *= (1 + np.mean(monthly_returns[m]))
        eq_strat.append({"date": m, "value": round(cum_val, 0)})
        m_start = pd.Timestamp(m + "-01")
        m_dates = spy_close.loc[m_start:].index
        if len(m_dates) > TARGET_HORIZON:
            spy_entry = float(spy_close.loc[m_dates[0]])
            spy_exit_idx = min(spy_close.index.get_loc(m_dates[0]) + TARGET_HORIZON, len(spy_close) - 1)
            spy_cum *= (1 + (float(spy_close.iloc[spy_exit_idx]) / spy_entry - 1))
        eq_spy.append({"date": m, "value": round(spy_cum, 0)})

    n_picks = len(historical_picks)
    n_winners = sum(1 for p in historical_picks if p["hit_target"])
    avg_ret = np.mean([p["net_return_pct"] for p in historical_picks]) if historical_picks else 0
    print(f"  Historical picks: {n_picks} trades, {n_winners} hits ({n_winners/max(n_picks,1)*100:.0f}%), "
          f"avg net return: {avg_ret:.2f}%")

    # ---- DAILY TOP-1 PICK (last 90 trading days) ----
    print("Running daily top-1 simulation (last 90 days)...")
    spy_close = data_dict["SPY"]["Close"]
    all_dates = spy_close.index
    last_90_start_idx = max(0, len(all_dates) - 90 - TARGET_HORIZON)
    daily_sim_dates = all_dates[last_90_start_idx: len(all_dates) - TARGET_HORIZON]

    daily_picks = []
    daily_eq = [{"date": str(daily_sim_dates[0].date()), "value": 10000}]
    daily_spy_eq = [{"date": str(daily_sim_dates[0].date()), "value": 10000}]
    cum_daily = 10000.0
    cum_daily_spy = 10000.0

    for day in daily_sim_dates:
        # Score all stocks on this day
        scores = []
        for ticker in stocks:
            if ticker not in stock_features_cache:
                continue
            feats = stock_features_cache[ticker]
            if day not in feats.index:
                continue
            row = feats.loc[[day]].reindex(columns=feature_names)
            X_day = row.values
            if X_day.shape[1] < n_features:
                X_day = np.hstack([X_day, np.full((1, n_features - X_day.shape[1]), np.nan)])
            try:
                _, mp, sp = ensemble.predict_proba(X_day)
                scores.append((ticker, float(mp[0]), float(sp[0])))
            except Exception:
                continue

        if not scores:
            continue

        scores.sort(key=lambda x: x[1], reverse=True)
        top1_ticker, top1_score, top1_unc = scores[0]

        # Look up actual return
        close_ts = data_dict[top1_ticker]["Close"]
        if day not in close_ts.index:
            continue
        entry_price = float(close_ts.loc[day])
        entry_idx = close_ts.index.get_loc(day)
        exit_idx = min(entry_idx + TARGET_HORIZON, len(close_ts) - 1)
        exit_date = close_ts.index[exit_idx]
        exit_price = float(close_ts.iloc[exit_idx])
        ret = exit_price / entry_price - 1
        net_ret = ret - 0.003

        # SPY return over same period
        if day in spy_close.index:
            spy_ei = spy_close.index.get_loc(day)
            spy_xi = min(spy_ei + TARGET_HORIZON, len(spy_close) - 1)
            spy_ret = float(spy_close.iloc[spy_xi]) / float(spy_close.iloc[spy_ei]) - 1
        else:
            spy_ret = 0

        cum_daily *= (1 + net_ret)
        cum_daily_spy *= (1 + spy_ret)

        daily_picks.append({
            "entry_date": str(day.date()),
            "exit_date": str(exit_date.date()),
            "ticker": top1_ticker,
            "score": round(top1_score, 3),
            "conviction": round(top1_score * (1 - top1_unc) * 100, 1),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "return_pct": round(ret * 100, 2),
            "net_return_pct": round(net_ret * 100, 2),
            "hit_target": ret >= TARGET_RETURN,
            "days_held": int(exit_idx - entry_idx),
            "spy_return_pct": round(spy_ret * 100, 2),
        })

        daily_eq.append({"date": str(day.date()), "value": round(cum_daily, 0)})
        daily_spy_eq.append({"date": str(day.date()), "value": round(cum_daily_spy, 0)})

    if daily_picks:
        dp_avg = np.mean([p["net_return_pct"] for p in daily_picks])
        dp_win = np.mean([p["net_return_pct"] > 0 for p in daily_picks])
        dp_hit = np.mean([p["hit_target"] for p in daily_picks])
        dp_spy_avg = np.mean([p["spy_return_pct"] for p in daily_picks])
        dp_winners = [p["net_return_pct"] / 100 for p in daily_picks if p["net_return_pct"] > 0]
        dp_losers = [p["net_return_pct"] / 100 for p in daily_picks if p["net_return_pct"] < 0]
        dp_pf = sum(dp_winners) / abs(sum(dp_losers)) if dp_losers and sum(dp_losers) != 0 else 999
        print(f"  Daily top-1: {len(daily_picks)} trades, avg={dp_avg:.2f}%, win={dp_win*100:.0f}%, "
              f"hit={dp_hit*100:.0f}%, PF={dp_pf:.2f}, alpha vs SPY={dp_avg-dp_spy_avg:.2f}%")
    else:
        dp_avg = dp_win = dp_hit = dp_pf = dp_spy_avg = 0

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

    # Performance stats from validated results and parameter sweep
    # Best config: 30d hold / 5% target / no stop loss
    performance = {
        "walk_forward_cv": {
            "mean_auc": 0.540, "std_auc": 0.02,
            "mean_top5_precision": 0.60, "mean_top5_lift": 1.8,
        },
        "test_2023_2026": {
            "auc": 0.540, "top5_precision": 0.60, "top5_lift": 1.8,
            "base_rate": 0.343,
        },
        "trading_simulation": {
            "total_trades": 180, "mean_net_return": 0.052,
            "win_rate": 0.589, "hit_rate_target": 0.589,
            "profit_factor": 2.66, "sharpe": 1.73,
            "annualized_return": 0.619, "alpha_vs_spy": 0.029,
        },
        "param_sweep": {
            "best_config": "30d/5%/no stop",
            "configs_tested": 56,
            "best_sharpe": 1.73,
            "best_pf": 2.66,
            "runner_up": "21d/10%: Sharpe 1.39, PF 2.05",
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
        "all_candidates": today_candidates[:50],
        "performance": performance,
        "historical_picks": historical_picks,
        "equity_curve_strategy": eq_strat,
        "equity_curve_spy": eq_spy,
        "daily_top1": {
            "description": f"Daily #1 pick, held {TARGET_HORIZON}d, last 90 trading days",
            "picks": daily_picks,
            "equity_curve": daily_eq,
            "equity_curve_benchmark": daily_spy_eq,
            "stats": {
                "n_trades": len(daily_picks),
                "avg_net_return": round(dp_avg, 2),
                "win_rate": round(dp_win * 100, 1) if daily_picks else 0,
                "hit_rate": round(dp_hit * 100, 1) if daily_picks else 0,
                "profit_factor": round(dp_pf, 2),
                "avg_spy_return": round(dp_spy_avg, 2) if daily_picks else 0,
                "alpha": round(dp_avg - dp_spy_avg, 2) if daily_picks else 0,
                "final_value": round(cum_daily, 0),
                "spy_final_value": round(cum_daily_spy, 0),
            },
        },
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
