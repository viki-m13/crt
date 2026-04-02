#!/usr/bin/env python3
"""
Crypto CCRL — Parameter Sweep + Daily Scan
============================================
1) Runs parameter sweep (horizons x targets) to find best crypto config
2) Trains final model with best config
3) Scores today's crypto + generates historical picks + equity curve
4) Outputs ccrl_crypto.json for the webapp

Usage:
    cd experiments && python scripts/crypto_ccrl_sweep_and_scan.py
"""

import os, sys, json, math, datetime, warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
CRYPTO_DIR = os.path.join(EXPERIMENTS_DIR, "crypto")

# Must set path BEFORE any imports from our packages
os.chdir(EXPERIMENTS_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crypto.prepare import (
    load_data as crypto_load_data, download_data as crypto_download_data,
    compute_features as crypto_compute_features,
    UNIVERSE as CRYPTO_UNIVERSE,
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    TRANSACTION_COST_BPS,
)
from src.rl_stock_selector import EnsemblePredictionLayer, CCRLConfig
from src.ai_features import (
    temporal_attention_momentum,
    asymmetric_tail_features,
)

# Benchmark = BTC
BENCHMARK = "BTC-USD"
# Exclude stablecoins and benchmark from picks
EXCLUDED = {"BTC-USD"}  # BTC is the benchmark, not a pick target

TOP_K = 5

# Sweep parameters
HORIZONS = [5, 10, 14, 21, 30]
TARGETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


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


def compute_crypto_features(close, volume, btc_close):
    """Compute CCRL-style features for one crypto asset."""
    features = []

    # TAM (temporal attention momentum)
    try:
        tam = temporal_attention_momentum(close, windows=[7, 14, 30, 60, 90, 180])
        features.append(tam)
    except Exception:
        pass

    # ATF (asymmetric tail features)
    try:
        atf = asymmetric_tail_features(close, lookback=365)
        features.append(atf)
    except Exception:
        pass

    # Classic crypto features from prepare.py
    try:
        classic = crypto_compute_features(close, volume, btc_close)
        features.append(classic)
    except Exception:
        pass

    if not features:
        return pd.DataFrame()

    result = pd.concat(features, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]

    core = ["vol_30d"] if "vol_30d" in result.columns else []
    if "tam" in result.columns:
        core.append("tam")
    if core:
        result = result.dropna(subset=core)

    return result


def build_dataset(features_cache, close_cache, tickers, feature_names,
                  horizon, target_return, split_start, split_end):
    """Build X, y for a given horizon/target/split."""
    all_X, all_y = [], []
    for ticker in tickers:
        if ticker not in features_cache or ticker in EXCLUDED:
            continue
        feats = features_cache[ticker]
        close = close_cache[ticker]
        feats_split = feats.loc[split_start:split_end]
        if len(feats_split) < 30:
            continue
        fwd_ret = close.pct_change(horizon).shift(-horizon)
        labels = (fwd_ret > 0).astype(int) if target_return <= 0 else (fwd_ret >= target_return).astype(int)
        common = feats_split.index.intersection(labels.dropna().index)
        if len(common) < 20:
            continue
        fa = feats_split.loc[common].reindex(columns=feature_names)
        all_X.append(fa.values)
        all_y.append(labels.loc[common].values)
    if not all_X:
        return None, None
    n_feat = len(feature_names)
    for i in range(len(all_X)):
        if all_X[i].shape[1] < n_feat:
            all_X[i] = np.hstack([all_X[i], np.full((all_X[i].shape[0], n_feat - all_X[i].shape[1]), np.nan)])
    return np.vstack(all_X), np.concatenate(all_y)


def main():
    print("=" * 70)
    print("CRYPTO CCRL — Parameter Sweep + Daily Scan")
    print("=" * 70)

    # Load data
    print("Loading crypto data...")
    data_dict = crypto_load_data()
    if len(data_dict) < 20:
        print("Downloading crypto data...")
        data_dict = crypto_download_data()
    print(f"  {len(data_dict)} tickers loaded")

    btc_close = data_dict.get("BTC-USD", pd.DataFrame()).get("Close")
    tickers = [t for t in CRYPTO_UNIVERSE if t in data_dict and t not in EXCLUDED]

    # Pre-compute features
    print("Pre-computing features...")
    features_cache, close_cache = {}, {}
    for i, ticker in enumerate(tickers):
        df = data_dict[ticker]
        if "Close" not in df.columns:
            continue
        try:
            feats = compute_crypto_features(df["Close"], df.get("Volume"), btc_close)
            if len(feats) > 50:
                features_cache[ticker] = feats
                close_cache[ticker] = df["Close"]
        except Exception:
            continue
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(tickers)}...")

    # Feature names
    feature_names = None
    for t in tickers:
        if t in features_cache:
            if feature_names is None:
                feature_names = features_cache[t].columns.tolist()
            else:
                for col in features_cache[t].columns:
                    if col not in feature_names:
                        feature_names.append(col)
    print(f"  {len(features_cache)} cryptos, {len(feature_names)} features")

    # ============================================================
    # PARAMETER SWEEP
    # ============================================================
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP")
    print("=" * 70)

    sweep_results = []
    combo = 0
    total = len(HORIZONS) * len(TARGETS)

    for horizon in HORIZONS:
        for target in TARGETS:
            combo += 1
            print(f"\n--- [{combo}/{total}] {horizon}d / {target*100:.0f}% ---")

            X_train, y_train = build_dataset(
                features_cache, close_cache, tickers, feature_names,
                horizon, target, TRAIN_START, TRAIN_END
            )
            X_test, y_test = build_dataset(
                features_cache, close_cache, tickers, feature_names,
                horizon, target, TEST_START, TEST_END
            )
            if X_train is None or X_test is None:
                print("  SKIP: insufficient data")
                continue

            pos_train, pos_test = y_train.mean(), y_test.mean()
            print(f"  Train: {len(y_train)}, pos={pos_train:.3f} | Test: {len(y_test)}, pos={pos_test:.3f}")
            if pos_train < 0.01 or pos_train > 0.95:
                print("  SKIP: degenerate")
                continue

            # Subsample for speed
            sub = np.arange(0, len(y_train), 3)
            config = CCRLConfig()
            config.calibrate_probabilities = False
            config.n_ensemble_models = 2
            ensemble = EnsemblePredictionLayer(config)
            ensemble.fit(X_train[sub], y_train[sub], feature_names)

            _, mean_p, _ = ensemble.predict_proba(X_test)
            auc = roc_auc_score(y_test, mean_p) if len(np.unique(y_test)) > 1 else 0.5

            # Top-K
            top5_prec, top5_lift = 0, 0
            if len(mean_p) >= 5:
                topk = np.argsort(mean_p)[-5:]
                top5_prec = y_test[topk].mean()
                top5_lift = top5_prec / pos_test if pos_test > 0 else 0

            # Trading sim (monthly top-5, hold for horizon)
            btc_close_ts = data_dict["BTC-USD"]["Close"]
            test_dates = btc_close_ts.loc[TEST_START:].index
            monthly = []
            seen = set()
            for d in test_dates:
                ym = (d.year, d.month)
                if ym not in seen:
                    seen.add(ym)
                    monthly.append(d)

            trades = []
            for reb in monthly:
                scores = []
                for t in tickers:
                    if t not in features_cache or t in EXCLUDED:
                        continue
                    f = features_cache[t]
                    if reb not in f.index:
                        continue
                    row = f.loc[[reb]].reindex(columns=feature_names)
                    X = row.values
                    if X.shape[1] < len(feature_names):
                        X = np.hstack([X, np.full((1, len(feature_names) - X.shape[1]), np.nan)])
                    try:
                        _, mp, _ = ensemble.predict_proba(X)
                        scores.append((t, float(mp[0])))
                    except Exception:
                        continue
                scores.sort(key=lambda x: x[1], reverse=True)
                for t, sc in scores[:TOP_K]:
                    cl = close_cache[t]
                    if reb not in cl.index:
                        continue
                    ep = float(cl.loc[reb])
                    ei = cl.index.get_loc(reb)
                    xi = min(ei + horizon, len(cl) - 1)
                    xp = float(cl.iloc[xi])
                    ret = xp / ep - 1
                    trades.append({"ticker": t, "score": sc, "entry_date": str(reb.date()),
                                   "exit_date": str(cl.index[xi].date()),
                                   "entry_price": round(ep, 4), "exit_price": round(xp, 4),
                                   "return_pct": round(ret * 100, 2),
                                   "net_return_pct": round((ret - TRANSACTION_COST_BPS * 2 / 10000) * 100, 2),
                                   "hit_target": ret >= target if target > 0 else ret > 0,
                                   "days_held": int(xi - ei)})

            if not trades:
                continue

            avg_ret = np.mean([t["net_return_pct"] / 100 for t in trades])
            win_rate = np.mean([t["net_return_pct"] > 0 for t in trades])
            winners = [t["net_return_pct"] / 100 for t in trades if t["net_return_pct"] > 0]
            losers = [t["net_return_pct"] / 100 for t in trades if t["net_return_pct"] < 0]
            pf = sum(winners) / abs(sum(losers)) if losers and sum(losers) != 0 else 999

            # Sharpe
            mr = {}
            for t in trades:
                m = t["entry_date"][:7]
                if m not in mr:
                    mr[m] = []
                mr[m].append(t["net_return_pct"] / 100)
            monthly_avg = [np.mean(v) for v in mr.values()]
            ann_ret = np.mean(monthly_avg) * 12
            ann_vol = np.std(monthly_avg) * np.sqrt(12) if len(monthly_avg) > 1 else 1
            sharpe = (ann_ret - 0.05) / ann_vol if ann_vol > 0 else 0

            r = {
                "horizon": horizon, "target_pct": target * 100,
                "auc": round(auc, 4), "base_rate": round(pos_test, 4),
                "top5_prec": round(top5_prec, 4), "top5_lift": round(top5_lift, 2),
                "n_trades": len(trades), "avg_net_ret": round(avg_ret * 100, 2),
                "win_rate": round(win_rate * 100, 1), "pf": round(pf, 2),
                "sharpe": round(sharpe, 3), "ann_ret": round(ann_ret * 100, 1),
            }
            sweep_results.append(r)
            print(f"  AUC={auc:.3f}, T5={top5_prec:.3f}({top5_lift:.1f}x), "
                  f"avg={avg_ret*100:.1f}%, win={win_rate*100:.0f}%, PF={pf:.2f}, Sh={sharpe:.2f}")

    sweep_results.sort(key=lambda r: r["sharpe"], reverse=True)

    print("\n" + "=" * 100)
    print("SWEEP RESULTS — SORTED BY SHARPE")
    print("=" * 100)
    print(f"{'Horiz':>6} {'Target':>7} {'AUC':>6} {'Base':>6} {'T5Prec':>7} {'T5Lift':>7} "
          f"{'AvgRet':>7} {'Win%':>5} {'PF':>5} {'Sharpe':>7} {'AnnRet':>7}")
    for r in sweep_results:
        print(f"{r['horizon']:>4}d {r['target_pct']:>6.0f}% {r['auc']:>6.3f} {r['base_rate']:>6.3f} "
              f"{r['top5_prec']:>7.3f} {r['top5_lift']:>6.1f}x {r['avg_net_ret']:>6.1f}% "
              f"{r['win_rate']:>5.1f} {r['pf']:>5.2f} {r['sharpe']:>7.3f} {r['ann_ret']:>6.1f}%")

    best = sweep_results[0] if sweep_results else None
    if best:
        print(f"\nBEST: {best['horizon']}d / {best['target_pct']:.0f}% — "
              f"Sharpe={best['sharpe']:.3f}, PF={best['pf']:.2f}, AnnRet={best['ann_ret']:.1f}%")

    # Save sweep
    sweep_path = os.path.join(EXPERIMENTS_DIR, "results", "crypto_param_sweep.json")
    os.makedirs(os.path.dirname(sweep_path), exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump({"timestamp": datetime.datetime.now().isoformat(), "results": sweep_results}, f, indent=2)

    if not best:
        print("No valid configs found!")
        return

    # ============================================================
    # FINAL SCAN WITH BEST CONFIG
    # ============================================================
    best_horizon = best["horizon"]
    best_target = best["target_pct"] / 100

    print(f"\n{'=' * 70}")
    print(f"FINAL SCAN — {best_horizon}d / {best['target_pct']:.0f}% target")
    print(f"{'=' * 70}")

    # Train full ensemble on best config
    print("Training full ensemble...")
    X_full, y_full = build_dataset(
        features_cache, close_cache, tickers, feature_names,
        best_horizon, best_target, TRAIN_START, TRAIN_END
    )
    config = CCRLConfig()
    final_ensemble = EnsemblePredictionLayer(config)
    final_ensemble.fit(X_full, y_full, feature_names)

    # Score today
    print("Scoring today's cryptos...")
    today_candidates = []
    for ticker in tickers:
        if ticker in EXCLUDED or ticker not in features_cache:
            continue
        feats = features_cache[ticker]
        if len(feats) < 1:
            continue
        latest = feats.iloc[-1:]
        latest = latest.reindex(columns=feature_names)
        X = latest.values
        if X.shape[1] < len(feature_names):
            X = np.hstack([X, np.full((1, len(feature_names) - X.shape[1]), np.nan)])
        try:
            _, mp, sp = final_ensemble.predict_proba(X)
            score = float(mp[0])
            unc = float(sp[0])
        except Exception:
            continue
        close = close_cache[ticker]
        price = float(close.iloc[-1])
        fd = feats.iloc[-1].to_dict()
        today_candidates.append({
            "ticker": ticker.replace("-USD", ""),
            "ticker_full": ticker,
            "score": score, "uncertainty": unc,
            "price": round(price, 4) if price < 1 else round(price, 2),
            "conviction": round(score * (1 - unc) * 100, 1),
            "returns": {
                "7d": round(float(fd.get("ret_7d", 0)) * 100, 1),
                "14d": round(float(fd.get("ret_14d", 0)) * 100, 1),
                "30d": round(float(fd.get("ret_30d", 0)) * 100, 1),
                "90d": round(float(fd.get("ret_90d", 0)) * 100, 1),
            },
            "vol_30d": round(float(fd.get("vol_30d", 0)) * 100, 1) if "vol_30d" in fd else 0,
            "drawdown": round(float(fd.get("drawdown_365d", 0)) * 100, 1) if "drawdown_365d" in fd else 0,
        })

    today_candidates.sort(key=lambda c: c["score"], reverse=True)
    top5 = today_candidates[:TOP_K]

    print(f"\nTop {TOP_K} crypto picks:")
    for i, p in enumerate(top5):
        print(f"  {i+1}. {p['ticker']:>6} — score={p['score']:.3f}, conv={p['conviction']:.0f}%, ${p['price']}")

    # Historical picks
    print("Running historical picks...")
    btc_close_ts = data_dict["BTC-USD"]["Close"]
    test_dates = btc_close_ts.loc[TEST_START:].index
    monthly = []
    seen = set()
    for d in test_dates:
        ym = (d.year, d.month)
        if ym not in seen:
            seen.add(ym)
            monthly.append(d)

    hist_picks = []
    for reb in monthly:
        scores = []
        for t in tickers:
            if t in EXCLUDED or t not in features_cache:
                continue
            f = features_cache[t]
            if reb not in f.index:
                continue
            row = f.loc[[reb]].reindex(columns=feature_names)
            X = row.values
            if X.shape[1] < len(feature_names):
                X = np.hstack([X, np.full((1, len(feature_names) - X.shape[1]), np.nan)])
            try:
                _, mp, _ = final_ensemble.predict_proba(X)
                scores.append((t, float(mp[0])))
            except Exception:
                continue
        scores.sort(key=lambda x: x[1], reverse=True)
        for t, sc in scores[:TOP_K]:
            cl = close_cache[t]
            if reb not in cl.index:
                continue
            ep = float(cl.loc[reb])
            ei = cl.index.get_loc(reb)
            xi = min(ei + best_horizon, len(cl) - 1)
            xp = float(cl.iloc[xi])
            ret = xp / ep - 1
            net_ret = ret - TRANSACTION_COST_BPS * 2 / 10000
            hist_picks.append({
                "entry_date": str(reb.date()),
                "exit_date": str(cl.index[xi].date()),
                "ticker": t.replace("-USD", ""),
                "ticker_full": t,
                "score": round(sc, 3),
                "entry_price": round(ep, 4) if ep < 1 else round(ep, 2),
                "exit_price": round(xp, 4) if xp < 1 else round(xp, 2),
                "return_pct": round(ret * 100, 2),
                "net_return_pct": round(net_ret * 100, 2),
                "hit_target": ret >= best_target if best_target > 0 else ret > 0,
                "days_held": int(xi - ei),
            })

    # Equity curves
    mr = {}
    for p in hist_picks:
        m = p["entry_date"][:7]
        if m not in mr:
            mr[m] = []
        mr[m].append(p["net_return_pct"] / 100)
    cum, btc_cum = 10000.0, 10000.0
    eq_strat, eq_btc = [], []
    for m in sorted(mr.keys()):
        cum *= (1 + np.mean(mr[m]))
        eq_strat.append({"date": m, "value": round(cum, 0)})
        ms = pd.Timestamp(m + "-01")
        md = btc_close_ts.loc[ms:].index
        if len(md) > best_horizon:
            be = float(btc_close_ts.loc[md[0]])
            bxi = min(btc_close_ts.index.get_loc(md[0]) + best_horizon, len(btc_close_ts) - 1)
            btc_cum *= (1 + (float(btc_close_ts.iloc[bxi]) / be - 1))
        eq_btc.append({"date": m, "value": round(btc_cum, 0)})

    n_picks = len(hist_picks)
    n_hits = sum(1 for p in hist_picks if p["hit_target"])
    avg_r = np.mean([p["net_return_pct"] for p in hist_picks]) if hist_picks else 0
    print(f"  Historical: {n_picks} trades, {n_hits} hits ({n_hits/max(n_picks,1)*100:.0f}%), avg net: {avg_r:.2f}%")

    # ---- DAILY TOP-1 PICK (last 90 days) ----
    print("Running daily top-1 simulation (last 90 days)...")
    all_dates = btc_close_ts.index
    d90_start = max(0, len(all_dates) - 90 - best_horizon)
    daily_sim_dates = all_dates[d90_start: len(all_dates) - best_horizon]
    tc = TRANSACTION_COST_BPS * 2 / 10000

    daily_picks = []
    cum_d, cum_d_btc = 10000.0, 10000.0
    daily_eq, daily_btc_eq = [], []

    for day in daily_sim_dates:
        scores = []
        for t in tickers:
            if t in EXCLUDED or t not in features_cache:
                continue
            f = features_cache[t]
            if day not in f.index:
                continue
            row = f.loc[[day]].reindex(columns=feature_names)
            X = row.values
            if X.shape[1] < len(feature_names):
                X = np.hstack([X, np.full((1, len(feature_names) - X.shape[1]), np.nan)])
            try:
                _, mp, sp = final_ensemble.predict_proba(X)
                scores.append((t, float(mp[0]), float(sp[0])))
            except Exception:
                continue
        if not scores:
            continue
        scores.sort(key=lambda x: x[1], reverse=True)
        t1, t1_sc, t1_unc = scores[0]
        cl = close_cache.get(t1)
        if cl is None or day not in cl.index:
            continue
        ep = float(cl.loc[day])
        ei = cl.index.get_loc(day)
        xi = min(ei + best_horizon, len(cl) - 1)
        xp = float(cl.iloc[xi])
        ret = xp / ep - 1
        net_ret = ret - tc
        # BTC return
        if day in btc_close_ts.index:
            bei = btc_close_ts.index.get_loc(day)
            bxi = min(bei + best_horizon, len(btc_close_ts) - 1)
            btc_ret = float(btc_close_ts.iloc[bxi]) / float(btc_close_ts.iloc[bei]) - 1
        else:
            btc_ret = 0
        cum_d *= (1 + net_ret)
        cum_d_btc *= (1 + btc_ret)
        daily_picks.append({
            "entry_date": str(day.date()),
            "exit_date": str(cl.index[xi].date()),
            "ticker": t1.replace("-USD", ""),
            "ticker_full": t1,
            "score": round(t1_sc, 3),
            "conviction": round(t1_sc * (1 - t1_unc) * 100, 1),
            "entry_price": round(ep, 4) if ep < 1 else round(ep, 2),
            "exit_price": round(xp, 4) if xp < 1 else round(xp, 2),
            "return_pct": round(ret * 100, 2),
            "net_return_pct": round(net_ret * 100, 2),
            "hit_target": ret >= best_target if best_target > 0 else ret > 0,
            "days_held": int(xi - ei),
            "benchmark_return_pct": round(btc_ret * 100, 2),
        })
        daily_eq.append({"date": str(day.date()), "value": round(cum_d, 0)})
        daily_btc_eq.append({"date": str(day.date()), "value": round(cum_d_btc, 0)})

    if daily_picks:
        dp_avg = np.mean([p["net_return_pct"] for p in daily_picks])
        dp_win = np.mean([p["net_return_pct"] > 0 for p in daily_picks])
        dp_hit = np.mean([p["hit_target"] for p in daily_picks])
        dp_btc = np.mean([p["benchmark_return_pct"] for p in daily_picks])
        dp_w = [p["net_return_pct"] / 100 for p in daily_picks if p["net_return_pct"] > 0]
        dp_l = [p["net_return_pct"] / 100 for p in daily_picks if p["net_return_pct"] < 0]
        dp_pf = sum(dp_w) / abs(sum(dp_l)) if dp_l and sum(dp_l) != 0 else 999
        print(f"  Daily top-1: {len(daily_picks)} trades, avg={dp_avg:.2f}%, win={dp_win*100:.0f}%, PF={dp_pf:.2f}")
    else:
        dp_avg = dp_win = dp_hit = dp_pf = dp_btc = 0

    # BTC regime
    btc_price = float(btc_close_ts.iloc[-1])
    btc_sma100 = float(btc_close_ts.rolling(100).mean().iloc[-1])

    # Assemble output
    performance = {
        "param_sweep": {
            "configs_tested": len(sweep_results),
            "best_config": f"{best_horizon}d/{best['target_pct']:.0f}%",
            "best_sharpe": best["sharpe"],
            "best_pf": best["pf"],
            "best_ann_ret": best["ann_ret"],
        },
        "test_period": {
            "auc": best["auc"],
            "top5_precision": best["top5_prec"],
            "top5_lift": best["top5_lift"],
        },
        "trading_simulation": {
            "total_trades": n_picks,
            "mean_net_return": round(avg_r / 100, 4),
            "win_rate": round(np.mean([p["net_return_pct"] > 0 for p in hist_picks]), 3) if hist_picks else 0,
            "hit_rate": round(n_hits / max(n_picks, 1), 3),
            "profit_factor": best["pf"],
            "sharpe": best["sharpe"],
            "annualized_return": round(best["ann_ret"] / 100, 3),
        },
    }

    full_data = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "CCRL-Crypto",
        "strategy_full_name": "Conviction Cascade RL — Crypto",
        "target": f"Crypto rising {best['target_pct']:.0f}%+ in {best_horizon} days",
        "best_config": {"horizon": best_horizon, "target_pct": best["target_pct"]},
        "model": {
            "ensemble_size": 5,
            "features": len(feature_names),
            "training_samples": int(len(y_full)),
            "training_positive_rate": round(float(y_full.mean()), 4),
        },
        "btc": {
            "price": round(btc_price, 2),
            "sma100": round(btc_sma100, 2),
            "regime": "BULL" if btc_price > btc_sma100 else "BEAR",
        },
        "top5": top5,
        "all_candidates": today_candidates[:50],
        "performance": performance,
        "historical_picks": hist_picks,
        "equity_curve_strategy": eq_strat,
        "equity_curve_btc": eq_btc,
        "sweep_results": sweep_results[:10],
        "daily_top1": {
            "description": f"Daily #1 pick, held {best_horizon}d, last 90 trading days",
            "picks": daily_picks,
            "equity_curve": daily_eq,
            "equity_curve_benchmark": daily_btc_eq,
            "stats": {
                "n_trades": len(daily_picks),
                "avg_net_return": round(dp_avg, 2),
                "win_rate": round(dp_win * 100, 1) if daily_picks else 0,
                "hit_rate": round(dp_hit * 100, 1) if daily_picks else 0,
                "profit_factor": round(dp_pf, 2),
                "avg_benchmark_return": round(dp_btc, 2) if daily_picks else 0,
                "alpha": round(dp_avg - dp_btc, 2) if daily_picks else 0,
                "final_value": round(cum_d, 0),
                "benchmark_final_value": round(cum_d_btc, 0),
            },
        },
    }

    # Write output
    docs_dir = os.path.join(EXPERIMENTS_DIR, "docs", "data")
    os.makedirs(docs_dir, exist_ok=True)
    with open(os.path.join(docs_dir, "ccrl_crypto.json"), "w") as f:
        json.dump(full_data, f, indent=2, cls=SafeJSONEncoder)

    # Per-ticker chart files
    tickers_dir = os.path.join(docs_dir, "tickers")
    os.makedirs(tickers_dir, exist_ok=True)
    for cand in today_candidates[:30]:
        tf = cand["ticker_full"]
        if tf not in data_dict:
            continue
        chart = [{"date": str(dt.date()), "price": round(float(row["Close"]), 4 if float(row["Close"]) < 1 else 2)}
                 for dt, row in data_dict[tf].tail(365).iterrows()]
        fname = tf.replace("-", "_") + ".json"
        with open(os.path.join(tickers_dir, fname), "w") as f:
            json.dump({**cand, "chart": chart}, f, indent=2, cls=SafeJSONEncoder)

    print(f"\nData written to {docs_dir}/ccrl_crypto.json")


if __name__ == "__main__":
    main()
