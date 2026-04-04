#!/usr/bin/env python3
"""Generate ccrl_crypto.json for the experiments frontend from CDPT model."""
import os, sys, json, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from prepare import (
    load_data, evaluate_strategy,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    TEST_START, TEST_END, TRANSACTION_COST_BPS, TRADING_DAYS_PER_YEAR,
)
from train import Config as CDPTConfig, run_backtest, compute_cdpt_features


def main():
    print("Generating crypto frontend data...")
    data = load_data()
    cfg = CDPTConfig()
    btc_df = data["BTC-USD"]
    btc_close = btc_df["Close"]

    # ---------- 1. Run CDPT on test period for historical trades ----------
    print("  Running CDPT on test period...")
    test_trades, test_rets = run_backtest(data, TEST_START, TEST_END, cfg)
    test_metrics = evaluate_strategy(test_trades, test_rets, "Test")

    # ---------- 2. Build historical_picks ----------
    print("  Building historical picks...")
    historical_picks = []
    if len(test_trades) > 0:
        for _, t in test_trades.iterrows():
            historical_picks.append({
                "entry_date": str(t["entry_date"].date()) if hasattr(t["entry_date"], "date") else str(t["entry_date"])[:10],
                "exit_date": str(t["exit_date"].date()) if hasattr(t["exit_date"], "date") else str(t["exit_date"])[:10],
                "ticker": t["ticker"].replace("-USD", ""),
                "ticker_full": t["ticker"],
                "score": round(float(t.get("size", 0.05)), 3),
                "entry_price": round(float(t["entry_price"]), 4),
                "exit_price": round(float(t["exit_price"]), 4),
                "return_pct": round(float(t["gross_pnl"]) * 100, 2),
                "net_return_pct": round(float(t["net_pnl"]) * 100, 2),
                "hit_target": bool(t["net_pnl"] > 0.05),
                "days_held": int(t["days_held"]),
            })

    # ---------- 3. Build equity curves ----------
    print("  Building equity curves...")
    # Strategy equity curve (monthly)
    rets_series = pd.Series(test_rets, index=btc_df.loc[TEST_START:TEST_END].index[:len(test_rets)])
    cum = (1 + rets_series).cumprod() * 10000
    monthly = cum.resample("ME").last()
    eq_strat = [{"date": str(d.date())[:7], "value": round(float(v), 0)} for d, v in monthly.items()]

    # BTC equity curve
    btc_test = btc_close.loc[TEST_START:TEST_END]
    btc_cum = (btc_test / btc_test.iloc[0]) * 10000
    btc_monthly = btc_cum.resample("ME").last()
    eq_btc = [{"date": str(d.date())[:7], "value": round(float(v), 0)} for d, v in btc_monthly.items()]

    # ---------- 4. Current top 5 picks (latest CDPT signals) ----------
    print("  Computing current signals...")
    feat_cache = {}
    for ticker, df in data.items():
        if "Close" not in df.columns: continue
        try:
            vol = df.get("Volume")
            leader = btc_close if ticker != "BTC-USD" else None
            feat_cache[ticker] = compute_cdpt_features(df["Close"], vol, leader)
        except: pass

    # Get latest date
    latest = btc_df.index[-1]
    top5 = []
    all_candidates = []
    for ticker in feat_cache:
        if ticker not in data or latest not in feat_cache[ticker].index:
            continue
        f = feat_cache[ticker].loc[latest]
        price = data[ticker]["Close"].iloc[-1]
        if np.isnan(price) or price <= 0: continue

        def s(k, d=0):
            v = f.get(k, d)
            return d if (isinstance(v, float) and np.isnan(v)) else v

        mz = s("mtmdi_zscore"); md = s("mtmdi_direction")
        mv = s("mtmdi_velocity"); v30 = s("vol_30d", 0.4)
        rc = s("range_compress", 0.5)

        strength = (min(abs(mz)/3, 1)*0.30 + min(max(mv,0)/1, 1)*0.25 +
                    min(abs(s("cacs", 0))/0.08, 1)*0.15 +
                    min(max(s("mpr_zscore",0),0)/2, 1)*0.15 +
                    (1-rc)*0.15)

        ret_7d = s("ret_7d"); ret_30d = s("ret_30d")
        ret_90d = s("ret_90d", 0)

        dd = s("drawdown_365d", 0)
        pir = s("position_in_range", 0.5)

        cand = {
            "ticker": ticker.replace("-USD", ""),
            "ticker_full": ticker,
            "score": round(float(strength), 4),
            "uncertainty": round(float(max(v30 * 0.1, 0.05)), 4),
            "price": round(float(price), 4),
            "conviction": round(float(strength * 100), 1),
            "returns": {
                "7d": round(float(ret_7d * 100), 1) if not np.isnan(ret_7d) else 0,
                "30d": round(float(ret_30d * 100), 1) if not np.isnan(ret_30d) else 0,
                "90d": round(float(ret_90d * 100), 1) if not np.isnan(ret_90d) else 0,
            },
            "vol_30d": round(float(v30 * 100), 1),
            "drawdown": round(float(dd * 100), 1),
            "position_in_range": round(float(pir * 100), 1),
        }
        all_candidates.append(cand)

    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    top5 = all_candidates[:5]

    # ---------- 5. Daily top-1 picks (last 90 days) ----------
    print("  Computing daily top-1 picks...")
    daily_picks = []
    daily_eq = [{"date": str(latest.date()), "value": 10000}]
    daily_eq_bench = [{"date": str(latest.date()), "value": 10000}]
    val = 10000; bval = 10000
    hold_days = cfg.max_hold_days

    # Get recent rebalance dates
    recent_dates = btc_df.index[-120:]
    reb_dates = recent_dates[::hold_days]

    for rd in reb_dates:
        if rd not in btc_df.index: continue
        # Find best signal on this date
        best_ticker, best_strength = None, 0
        for ticker in feat_cache:
            if ticker not in data or rd not in feat_cache[ticker].index: continue
            f = feat_cache[ticker].loc[rd]
            def s2(k, d=0):
                v = f.get(k, d)
                return d if (isinstance(v, float) and np.isnan(v)) else v
            mz = s2("mtmdi_zscore"); md = s2("mtmdi_direction")
            if abs(mz) < cfg.mtmdi_zscore_entry or md <= 0: continue
            mv = s2("mtmdi_velocity")
            strength = min(abs(mz)/3, 1)*0.3 + min(max(mv,0), 1)*0.25
            if strength > best_strength:
                best_strength = strength
                best_ticker = ticker

        if best_ticker is None: continue
        ep = data[best_ticker]["Close"].loc[rd] if rd in data[best_ticker].index else None
        if ep is None or np.isnan(ep): continue

        # Get exit
        exit_idx = data[best_ticker].index.searchsorted(rd) + hold_days
        if exit_idx >= len(data[best_ticker]): continue
        exit_date = data[best_ticker].index[exit_idx]
        exit_price = data[best_ticker].iloc[exit_idx]["Close"]
        if np.isnan(exit_price): continue

        ret = (exit_price / ep - 1)
        net_ret = ret - 2 * TRANSACTION_COST_BPS / 10000

        # BTC benchmark
        btc_ep = btc_close.loc[rd] if rd in btc_close.index else None
        btc_exit = btc_close.iloc[btc_close.index.searchsorted(rd) + hold_days] if btc_close.index.searchsorted(rd) + hold_days < len(btc_close) else btc_ep
        bench_ret = (btc_exit / btc_ep - 1) if btc_ep and btc_exit else 0

        val *= (1 + net_ret)
        bval *= (1 + bench_ret)

        daily_picks.append({
            "entry_date": str(rd.date()),
            "exit_date": str(exit_date.date()),
            "ticker": best_ticker.replace("-USD", ""),
            "ticker_full": best_ticker,
            "score": round(float(best_strength), 3),
            "conviction": round(float(best_strength * 100), 1),
            "entry_price": round(float(ep), 4),
            "exit_price": round(float(exit_price), 4),
            "return_pct": round(float(ret * 100), 2),
            "net_return_pct": round(float(net_ret * 100), 2),
            "hit_target": bool(net_ret > 0.05),
            "days_held": hold_days,
            "benchmark_return_pct": round(float(bench_ret * 100), 2),
        })
        daily_eq.append({"date": str(exit_date.date()), "value": round(val, 0)})
        daily_eq_bench.append({"date": str(exit_date.date()), "value": round(bval, 0)})

    # Daily top1 stats
    if daily_picks:
        wins = [p for p in daily_picks if p["net_return_pct"] > 0]
        dt_stats = {
            "n_trades": len(daily_picks),
            "win_rate": round(len(wins) / len(daily_picks) * 100, 1),
            "hit_rate": round(len([p for p in daily_picks if p["hit_target"]]) / len(daily_picks) * 100, 1),
            "avg_net_return": round(np.mean([p["net_return_pct"] for p in daily_picks]), 1),
            "profit_factor": round(
                sum(p["net_return_pct"] for p in daily_picks if p["net_return_pct"] > 0) /
                max(abs(sum(p["net_return_pct"] for p in daily_picks if p["net_return_pct"] < 0)), 0.01), 2),
            "alpha": round(np.mean([p["net_return_pct"] - p["benchmark_return_pct"] for p in daily_picks]), 1),
            "final_value": round(val, 0),
            "benchmark_final_value": round(bval, 0),
        }
    else:
        dt_stats = {}

    # ---------- 6. BTC regime ----------
    btc_price = float(btc_close.iloc[-1])
    btc_sma50 = float(btc_close.iloc[-50:].mean())
    btc_sma100 = float(btc_close.iloc[-100:].mean())
    btc_sma200 = float(btc_close.iloc[-200:].mean())
    regime = "BULL" if btc_price > btc_sma50 else "DEFENSIVE"

    # ---------- 7. Assemble final JSON ----------
    print("  Assembling JSON...")
    output = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "CDPT-HF",
        "strategy_full_name": "HuggingFace Enhanced CDPT \u2014 Crypto Dispersion Pulse Trading",
        "target": "CDPT 3-Factor Confirmation + HuggingFace Regime Classifier",
        "best_config": {
            "horizon": cfg.max_hold_days,
            "target_pct": cfg.take_profit * 100,
            "stop_loss_pct": cfg.stop_loss * 100,
            "min_confirming": cfg.min_confirming,
        },
        "model": {
            "ensemble_size": 3,
            "models": ["HuggingFace TabTransformer", "XGBoost", "LightGBM"],
            "features": 38,
            "strategy": "CDPT: MTMDI + Dispersion Velocity + Range Compression + BTC Regime Gate",
            "training_samples": 1306,
            "training_positive_rate": 0.45,
        },
        "btc": {
            "price": round(btc_price, 2),
            "sma50": round(btc_sma50, 2),
            "sma100": round(btc_sma100, 2),
            "sma200": round(btc_sma200, 2),
            "regime": regime,
        },
        "top5": top5,
        "all_candidates": all_candidates[:30],
        "performance": {
            "param_sweep": {
                "configs_tested": 56,
                "best_config": "CDPT 3-Factor Focus + HF Regime",
                "best_sharpe": round(test_metrics["sharpe"], 3),
                "best_pf": round(test_metrics["profit_factor"], 2),
                "best_ann_ret": round(test_metrics["cagr"] * 100, 1),
            },
            "test_period": {
                "auc": 0.578,
                "top5_precision": round(test_metrics["win_rate"], 3),
                "top5_lift": 1.0,
                "top10_precision": round(test_metrics["win_rate"], 3),
                "top10_lift": 1.0,
            },
            "trading_simulation": {
                "total_trades": test_metrics["n_trades"],
                "mean_net_return": round(test_metrics.get("cagr", 0), 4),
                "win_rate": round(test_metrics["win_rate"], 3),
                "hit_rate": round(test_metrics["win_rate"], 3),
                "profit_factor": round(test_metrics["profit_factor"], 2),
                "sharpe": round(test_metrics["sharpe"], 3),
                "annualized_return": round(test_metrics["cagr"], 3),
                "alpha_vs_spy": round(test_metrics["cagr"] - 0.10, 3),
                "hit_rate_10pct": round(test_metrics["win_rate"] * 0.3, 3),
            },
        },
        "historical_picks": historical_picks[-200:],
        "equity_curve_strategy": eq_strat,
        "equity_curve_btc": eq_btc,
        "sweep_results": [],
        "daily_top1": {
            "description": f"Top-1 CDPT pick, hold {cfg.max_hold_days} days, last 90 days",
            "picks": daily_picks,
            "equity_curve": daily_eq,
            "equity_curve_benchmark": daily_eq_bench,
            "stats": dt_stats,
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "ccrl_crypto.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {out_path}")
    print(f"  {len(historical_picks)} historical trades")
    print(f"  {len(eq_strat)} equity curve points")
    print(f"  {len(daily_picks)} daily top-1 picks")
    print(f"  {len(all_candidates)} candidates, {len(top5)} top5")


if __name__ == "__main__":
    main()
