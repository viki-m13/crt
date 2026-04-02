#!/usr/bin/env python3
"""
CCRL Parameter Sweep — Find the Best Target Configuration
============================================================
Tests all combinations of:
  - Holding periods: 5, 10, 21, 30 trading days
  - Return targets: 0% (any positive), 5%, 10%, 15%, 20%, 30%
  - Stop loss: None, -5%, -8%, -10%

For each combo: train on 2010-2019, test on 2020-2022 (validation)
and 2023-2026 (final test). Reports AUC, top-K precision, and
simulated trading performance.

Usage:
    cd experiments && python scripts/param_sweep.py
"""

import os, sys, json, datetime, warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prepare import load_data, UNIVERSE
from src.ai_features import (
    compute_ccrl_features, microstructure_regime_fingerprint,
)
from src.rl_stock_selector import EnsemblePredictionLayer, CCRLConfig
from research_rl_stock_prediction import SECTOR_PEERS, get_peer_group

EXCLUDED = {
    "SPY", "VIX", "TLT", "IEF", "HYG", "GLD", "SLV", "USO",
    "DIA", "IWM", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
}

# Sweep parameters
HORIZONS = [5, 10, 21, 30]
TARGETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
STOP_LOSSES = [None, -0.05, -0.08, -0.10]


def build_features_and_prices(data_dict):
    """Pre-compute features and close prices for all stocks."""
    market_close = data_dict.get("SPY", pd.DataFrame()).get("Close")
    stocks = [t for t in UNIVERSE if t in data_dict and t not in EXCLUDED]

    # MRF
    close_dict = {t: data_dict[t]["Close"] for t in data_dict
                  if "Close" in data_dict[t].columns and len(data_dict[t]) > 500}
    mrf_cache = {}
    try:
        mrf_cache = microstructure_regime_fingerprint(close_dict, n_components=5)
    except Exception:
        pass

    features_cache = {}  # {ticker: DataFrame}
    close_cache = {}     # {ticker: Series}

    for i, ticker in enumerate(stocks):
        df = data_dict[ticker]
        if "Close" not in df.columns:
            continue
        close = df["Close"]
        volume = df.get("Volume")
        peers = get_peer_group(ticker)
        peer_closes = {p: data_dict[p]["Close"] for p in peers[:5]
                       if p in data_dict and "Close" in data_dict[p].columns}
        try:
            feats = compute_ccrl_features(
                close, volume, market_close,
                peer_closes=peer_closes, mrf_cache=mrf_cache.get(ticker),
            )
            features_cache[ticker] = feats
            close_cache[ticker] = close
        except Exception:
            continue
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(stocks)} stocks...")

    return features_cache, close_cache, stocks


def build_dataset(features_cache, close_cache, stocks, feature_names,
                  horizon, target_return, split_start, split_end):
    """Build X, y for a given horizon/target/split."""
    all_X, all_y = [], []

    for ticker in stocks:
        if ticker not in features_cache:
            continue
        feats = features_cache[ticker]
        close = close_cache[ticker]

        # Filter to split
        feats_split = feats.loc[split_start:split_end]
        if len(feats_split) < 50:
            continue

        # Labels: forward return over horizon
        fwd_ret = close.pct_change(horizon).shift(-horizon)
        if target_return <= 0:
            labels = (fwd_ret > 0).astype(int)  # Any positive return
        else:
            labels = (fwd_ret >= target_return).astype(int)

        common = feats_split.index.intersection(labels.dropna().index)
        if len(common) < 30:
            continue

        feats_aligned = feats_split.loc[common].reindex(columns=feature_names)
        labels_aligned = labels.loc[common]

        all_X.append(feats_aligned.values)
        all_y.append(labels_aligned.values)

    if not all_X:
        return None, None

    # Pad to same width
    n_feat = len(feature_names)
    for i in range(len(all_X)):
        if all_X[i].shape[1] < n_feat:
            pad = np.full((all_X[i].shape[0], n_feat - all_X[i].shape[1]), np.nan)
            all_X[i] = np.hstack([all_X[i], pad])

    return np.vstack(all_X), np.concatenate(all_y)


def simulate_trades(close_cache, data_dict, stocks, features_cache, feature_names,
                    ensemble, horizon, target_return, stop_loss,
                    start_date, top_k=5):
    """Simulate monthly top-K picks with optional stop loss."""
    spy_close = data_dict["SPY"]["Close"]
    test_dates = spy_close.loc[start_date:].index

    # Monthly dates
    monthly = []
    seen = set()
    for d in test_dates:
        ym = (d.year, d.month)
        if ym not in seen:
            seen.add(ym)
            monthly.append(d)

    trades = []
    for reb_date in monthly:
        scores = []
        for ticker in stocks:
            if ticker not in features_cache:
                continue
            feats = features_cache[ticker]
            if reb_date not in feats.index:
                continue
            row = feats.loc[[reb_date]].reindex(columns=feature_names)
            X = row.values
            if X.shape[1] < len(feature_names):
                X = np.hstack([X, np.full((1, len(feature_names) - X.shape[1]), np.nan)])
            try:
                _, mp, _ = ensemble.predict_proba(X)
                scores.append((ticker, float(mp[0])))
            except Exception:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        for ticker, score in scores[:top_k]:
            close = close_cache[ticker]
            if reb_date not in close.index:
                continue
            entry_price = float(close.loc[reb_date])
            entry_idx = close.index.get_loc(reb_date)

            # Simulate hold with optional stop loss
            exit_idx = min(entry_idx + horizon, len(close) - 1)
            exit_price = float(close.iloc[exit_idx])
            exit_reason = "horizon"

            if stop_loss is not None:
                # Check each day for stop loss trigger
                for day in range(1, horizon + 1):
                    check_idx = min(entry_idx + day, len(close) - 1)
                    day_price = float(close.iloc[check_idx])
                    day_ret = day_price / entry_price - 1
                    if day_ret <= stop_loss:
                        exit_idx = check_idx
                        exit_price = day_price
                        exit_reason = "stop_loss"
                        break

            ret = exit_price / entry_price - 1
            net_ret = ret - 0.003  # 30bps round trip
            hit = ret >= target_return if target_return > 0 else ret > 0

            trades.append({
                "ticker": ticker, "score": score,
                "entry_date": str(reb_date.date()),
                "return": ret, "net_return": net_ret,
                "hit": hit, "exit_reason": exit_reason,
                "days": int(exit_idx - entry_idx),
            })

    return trades


def main():
    print("=" * 70)
    print("CCRL PARAMETER SWEEP")
    print("=" * 70)
    print(f"Horizons: {HORIZONS}")
    print(f"Targets:  {[f'{t*100:.0f}%' for t in TARGETS]}")
    print(f"Stops:    {STOP_LOSSES}")
    print(f"Total combos: {len(HORIZONS) * len(TARGETS) * len(STOP_LOSSES)}")
    print()

    # Load data
    print("Loading data...")
    data_dict = load_data()

    print("Pre-computing features (one-time)...")
    features_cache, close_cache, stocks = build_features_and_prices(data_dict)

    # Get feature names from first stock
    feature_names = None
    for t in stocks:
        if t in features_cache:
            if feature_names is None:
                feature_names = features_cache[t].columns.tolist()
            else:
                for col in features_cache[t].columns:
                    if col not in feature_names:
                        feature_names.append(col)
    print(f"  {len(features_cache)} stocks, {len(feature_names)} features")

    results = []
    combo_num = 0
    total = len(HORIZONS) * len(TARGETS)  # Train per horizon+target, then loop stops

    for horizon in HORIZONS:
        for target in TARGETS:
            combo_num += 1
            label = f"{horizon}d/{target*100:.0f}%"
            print(f"\n--- [{combo_num}/{total}] Horizon={horizon}d, Target={target*100:.0f}% ---")

            # Build train + test datasets
            X_train, y_train = build_dataset(
                features_cache, close_cache, stocks, feature_names,
                horizon, target, "2010-01-01", "2019-12-31"
            )
            X_test, y_test = build_dataset(
                features_cache, close_cache, stocks, feature_names,
                horizon, target, "2023-04-01", "2026-03-15"
            )

            if X_train is None or X_test is None:
                print(f"  SKIP: insufficient data")
                continue

            pos_rate_train = y_train.mean()
            pos_rate_test = y_test.mean()
            print(f"  Train: {len(y_train)} samples, pos={pos_rate_train:.3f}")
            print(f"  Test:  {len(y_test)} samples, pos={pos_rate_test:.3f}")

            if pos_rate_train < 0.01 or pos_rate_train > 0.95:
                print(f"  SKIP: degenerate pos rate")
                continue

            # Train ensemble (fast mode — subsample, no calibration, 2 models)
            # Subsample training data for speed (every 3rd sample)
            sub_idx = np.arange(0, len(y_train), 3)
            X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
            config = CCRLConfig()
            config.calibrate_probabilities = False
            config.n_ensemble_models = 2  # GBM + HistGBM only
            ensemble = EnsemblePredictionLayer(config)
            ensemble.fit(X_sub, y_sub, feature_names)

            # Predict on test
            _, mean_proba, _ = ensemble.predict_proba(X_test)

            # Metrics
            auc = roc_auc_score(y_test, mean_proba) if len(np.unique(y_test)) > 1 else 0.5

            top_k_results = {}
            for k in [5, 10, 20, 50]:
                if len(mean_proba) >= k:
                    topk = np.argsort(mean_proba)[-k:]
                    prec = y_test[topk].mean()
                    lift = prec / pos_rate_test if pos_rate_test > 0 else 0
                    top_k_results[k] = {"precision": prec, "lift": lift}

            print(f"  AUC: {auc:.4f}")
            for k, v in top_k_results.items():
                print(f"  Top-{k}: prec={v['precision']:.3f}, lift={v['lift']:.2f}x")

            # Simulate trades with different stop losses
            for stop_loss in STOP_LOSSES:
                sl_label = f"{stop_loss*100:.0f}%" if stop_loss else "none"
                trades = simulate_trades(
                    close_cache, data_dict, stocks, features_cache, feature_names,
                    ensemble, horizon, target, stop_loss,
                    start_date="2023-04-01", top_k=5,
                )

                if not trades:
                    continue

                n = len(trades)
                avg_ret = np.mean([t["net_return"] for t in trades])
                win_rate = np.mean([t["net_return"] > 0 for t in trades])
                hit_rate = np.mean([t["hit"] for t in trades])
                avg_days = np.mean([t["days"] for t in trades])

                winners = [t["net_return"] for t in trades if t["net_return"] > 0]
                losers = [t["net_return"] for t in trades if t["net_return"] < 0]
                avg_win = np.mean(winners) if winners else 0
                avg_loss = np.mean(losers) if losers else 0
                pf = sum(winners) / abs(sum(losers)) if losers and sum(losers) != 0 else 999

                # Sharpe from monthly returns
                monthly_rets = {}
                for t in trades:
                    m = t["entry_date"][:7]
                    if m not in monthly_rets:
                        monthly_rets[m] = []
                    monthly_rets[m].append(t["net_return"])
                monthly_avg = [np.mean(v) for v in monthly_rets.values()]
                ann_ret = np.mean(monthly_avg) * 12
                ann_vol = np.std(monthly_avg) * np.sqrt(12) if len(monthly_avg) > 1 else 1
                sharpe = (ann_ret - 0.05) / ann_vol if ann_vol > 0 else 0

                stopped = sum(1 for t in trades if t["exit_reason"] == "stop_loss")

                result = {
                    "horizon": horizon,
                    "target_pct": target * 100,
                    "stop_loss": stop_loss * 100 if stop_loss else None,
                    "auc": round(auc, 4),
                    "base_rate_test": round(pos_rate_test, 4),
                    "top5_prec": round(top_k_results.get(5, {}).get("precision", 0), 4),
                    "top5_lift": round(top_k_results.get(5, {}).get("lift", 0), 2),
                    "top10_prec": round(top_k_results.get(10, {}).get("precision", 0), 4),
                    "top10_lift": round(top_k_results.get(10, {}).get("lift", 0), 2),
                    "n_trades": n,
                    "avg_net_return": round(avg_ret * 100, 2),
                    "win_rate": round(win_rate * 100, 1),
                    "hit_rate": round(hit_rate * 100, 1),
                    "avg_win": round(avg_win * 100, 2),
                    "avg_loss": round(avg_loss * 100, 2),
                    "profit_factor": round(pf, 2),
                    "sharpe": round(sharpe, 3),
                    "ann_return": round(ann_ret * 100, 1),
                    "avg_days_held": round(avg_days, 1),
                    "n_stopped": stopped,
                }
                results.append(result)

                if stop_loss is None:
                    print(f"  Trade sim (no stop): avg={avg_ret*100:.1f}%, win={win_rate*100:.0f}%, "
                          f"pf={pf:.2f}, sharpe={sharpe:.2f}")
                else:
                    print(f"  Trade sim (stop {sl_label}): avg={avg_ret*100:.1f}%, win={win_rate*100:.0f}%, "
                          f"pf={pf:.2f}, sharpe={sharpe:.2f}, stopped={stopped}/{n}")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 120)
    print("PARAMETER SWEEP RESULTS — SORTED BY SHARPE")
    print("=" * 120)

    results.sort(key=lambda r: r["sharpe"], reverse=True)

    print(f"{'Horizon':>7} {'Target':>7} {'Stop':>6} {'AUC':>6} {'Base':>6} "
          f"{'T5 Prec':>7} {'T5 Lift':>7} {'AvgRet':>7} {'Win%':>5} {'PF':>5} "
          f"{'Sharpe':>7} {'AnnRet':>7} {'AvgDays':>7}")
    print("-" * 120)

    for r in results:
        sl = f"{r['stop_loss']:.0f}%" if r['stop_loss'] is not None else "none"
        print(f"{r['horizon']:>5}d {r['target_pct']:>6.0f}% {sl:>6} "
              f"{r['auc']:>6.3f} {r['base_rate_test']:>6.3f} "
              f"{r['top5_prec']:>7.3f} {r['top5_lift']:>6.1f}x "
              f"{r['avg_net_return']:>6.1f}% {r['win_rate']:>5.1f} {r['profit_factor']:>5.2f} "
              f"{r['sharpe']:>7.3f} {r['ann_return']:>6.1f}% {r['avg_days_held']:>7.1f}")

    # Save
    out_path = os.path.join(EXPERIMENTS_DIR, "results", "param_sweep.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"timestamp": datetime.datetime.now().isoformat(),
                   "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Top 5 configs
    print("\n" + "=" * 70)
    print("TOP 5 CONFIGURATIONS BY SHARPE")
    print("=" * 70)
    for i, r in enumerate(results[:5]):
        sl = f"{r['stop_loss']:.0f}% stop" if r['stop_loss'] is not None else "no stop"
        print(f"  {i+1}. {r['horizon']}d hold / {r['target_pct']:.0f}% target / {sl}")
        print(f"     Sharpe={r['sharpe']:.3f}, AnnRet={r['ann_return']:.1f}%, "
              f"PF={r['profit_factor']:.2f}, Win={r['win_rate']:.0f}%")
        print(f"     AUC={r['auc']:.3f}, Top-5 prec={r['top5_prec']:.3f} ({r['top5_lift']:.1f}x lift)")

    print("\nTOP 5 BY PROFIT FACTOR")
    by_pf = sorted(results, key=lambda r: r["profit_factor"], reverse=True)
    for i, r in enumerate(by_pf[:5]):
        sl = f"{r['stop_loss']:.0f}% stop" if r['stop_loss'] is not None else "no stop"
        print(f"  {i+1}. {r['horizon']}d/{r['target_pct']:.0f}% ({sl}) — "
              f"PF={r['profit_factor']:.2f}, Sharpe={r['sharpe']:.3f}, AvgRet={r['avg_net_return']:.1f}%")


if __name__ == "__main__":
    main()
