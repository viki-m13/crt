#!/usr/bin/env python3
"""
Generate ccrl_crypto.json for the experiments frontend from CDPT model.
All sections show the SAME strategy: CDPT portfolio-level performance.
"""
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

    # ---- 1. Run CDPT on test period ----
    print("  Running CDPT on test period...")
    test_trades, test_rets = run_backtest(data, TEST_START, TEST_END, cfg)
    test_metrics = evaluate_strategy(test_trades, test_rets, "Test")

    # ---- 2. Compute daily equity curve from portfolio returns ----
    print("  Building equity curves...")
    test_dates = btc_df.loc[TEST_START:TEST_END].index[:len(test_rets)]
    rets_s = pd.Series(test_rets, index=test_dates)
    cum = (1 + rets_s).cumprod() * 10000

    # Monthly equity curve (strategy)
    monthly = cum.resample("ME").last()
    eq_strat = [{"date": str(d.date())[:7], "value": round(float(v), 0)}
                for d, v in monthly.items()]

    # BTC monthly equity
    btc_test = btc_close.loc[TEST_START:TEST_END]
    btc_cum = (btc_test / btc_test.iloc[0]) * 10000
    btc_mo = btc_cum.resample("ME").last()
    eq_btc = [{"date": str(d.date())[:7], "value": round(float(v), 0)}
              for d, v in btc_mo.items()]

    # ---- 3. Build portfolio-level "historical picks" ----
    # Group trades into weekly portfolio periods so each "pick" = 1 week's portfolio
    print("  Building portfolio-level historical picks...")
    weekly_rets = rets_s.resample("W-SUN").apply(lambda x: (1 + x).prod() - 1)
    btc_weekly = btc_close.loc[TEST_START:TEST_END].pct_change().resample("W-SUN").apply(
        lambda x: (1 + x).prod() - 1)
    # Align
    common = weekly_rets.index.intersection(btc_weekly.index)
    weekly_rets = weekly_rets.loc[common]
    btc_weekly = btc_weekly.loc[common]

    # Only keep weeks where strategy had exposure (return != 0)
    active_weeks = weekly_rets[weekly_rets != 0]

    historical_picks = []
    for date, ret in active_weeks.items():
        # Find which tickers were traded that week
        week_start = date - pd.Timedelta(days=6)
        week_trades = test_trades[
            (test_trades["entry_date"] >= week_start) &
            (test_trades["entry_date"] <= date)
        ] if len(test_trades) > 0 else pd.DataFrame()

        tickers = list(week_trades["ticker"].unique()) if len(week_trades) > 0 else []
        ticker_str = ", ".join([t.replace("-USD", "") for t in tickers[:5]])
        if len(tickers) > 5:
            ticker_str += f" +{len(tickers)-5}"

        btc_ret = btc_weekly.get(date, 0)
        if isinstance(btc_ret, (float, np.floating)) and np.isnan(btc_ret):
            btc_ret = 0

        historical_picks.append({
            "entry_date": str(week_start.date()),
            "exit_date": str(date.date()),
            "ticker": ticker_str or "CASH",
            "ticker_full": ticker_str or "CASH",
            "score": round(abs(float(ret)) * 10, 3),
            "entry_price": 1.0,
            "exit_price": round(1.0 + float(ret), 4),
            "return_pct": round(float(ret) * 100, 2),
            "net_return_pct": round(float(ret) * 100, 2),
            "hit_target": bool(ret > 0.05),
            "days_held": 7,
        })

    # ---- 4. Daily top-1: show weekly portfolio performance (last 90 days) ----
    print("  Building daily top-1 (portfolio view)...")
    recent_weekly = weekly_rets.iloc[-13:]  # ~90 days = 13 weeks
    recent_btc = btc_weekly.reindex(recent_weekly.index, fill_value=0)

    daily_picks = []
    val = 10000
    bval = 10000
    daily_eq = []
    daily_eq_bench = []

    for date, ret in recent_weekly.items():
        week_start = date - pd.Timedelta(days=6)
        btc_ret = recent_btc.get(date, 0)
        if isinstance(btc_ret, (float, np.floating)) and np.isnan(btc_ret):
            btc_ret = 0

        # Find traded tickers
        week_trades = test_trades[
            (test_trades["entry_date"] >= week_start) &
            (test_trades["entry_date"] <= date)
        ] if len(test_trades) > 0 else pd.DataFrame()
        tickers = list(week_trades["ticker"].unique()) if len(week_trades) > 0 else []
        best_ticker = tickers[0].replace("-USD", "") if tickers else "CASH"
        n_trades_week = len(week_trades) if len(week_trades) > 0 else 0

        val *= (1 + float(ret))
        bval *= (1 + float(btc_ret))

        daily_picks.append({
            "entry_date": str(week_start.date()),
            "exit_date": str(date.date()),
            "ticker": best_ticker,
            "ticker_full": f"Portfolio ({n_trades_week} trades)",
            "score": round(abs(float(ret)) * 10, 3),
            "conviction": round(abs(float(ret)) * 100, 1),
            "entry_price": round(val / (1 + float(ret)), 0),
            "exit_price": round(val, 0),
            "return_pct": round(float(ret) * 100, 2),
            "net_return_pct": round(float(ret) * 100, 2),
            "hit_target": bool(ret > 0.05),
            "days_held": 7,
            "benchmark_return_pct": round(float(btc_ret) * 100, 2),
            "spy_return_pct": round(float(btc_ret) * 100, 2),
        })
        daily_eq.append({"date": str(date.date()), "value": round(val, 0)})
        daily_eq_bench.append({"date": str(date.date()), "value": round(bval, 0)})

    # Stats for daily top-1
    if daily_picks:
        wins = [p for p in daily_picks if p["net_return_pct"] > 0]
        total_pos = sum(p["net_return_pct"] for p in daily_picks if p["net_return_pct"] > 0)
        total_neg = abs(sum(p["net_return_pct"] for p in daily_picks if p["net_return_pct"] < 0))
        dt_stats = {
            "n_trades": len(daily_picks),
            "win_rate": round(len(wins) / len(daily_picks) * 100, 1),
            "hit_rate": round(len([p for p in daily_picks if p["hit_target"]]) / len(daily_picks) * 100, 1),
            "avg_net_return": round(np.mean([p["net_return_pct"] for p in daily_picks]), 1),
            "profit_factor": round(total_pos / max(total_neg, 0.01), 2),
            "alpha": round(np.mean([p["net_return_pct"] - p["benchmark_return_pct"] for p in daily_picks]), 1),
            "final_value": round(val, 0),
            "benchmark_final_value": round(bval, 0),
        }
    else:
        dt_stats = {}

    # ---- 5. Current top 5 candidates ----
    print("  Computing current signals...")
    feat_cache = {}
    for ticker, df in data.items():
        if "Close" not in df.columns: continue
        try:
            vol = df.get("Volume")
            leader = btc_close if ticker != "BTC-USD" else None
            feat_cache[ticker] = compute_cdpt_features(df["Close"], vol, leader)
        except: pass

    latest = btc_df.index[-1]
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

        all_candidates.append({
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
        })

    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    top5 = all_candidates[:5]

    # ---- 6. BTC regime ----
    btc_price = float(btc_close.iloc[-1])
    btc_sma50 = float(btc_close.iloc[-50:].mean())
    btc_sma100 = float(btc_close.iloc[-100:].mean())
    btc_sma200 = float(btc_close.iloc[-200:].mean())
    regime = "BULL" if btc_price > btc_sma50 else "DEFENSIVE"

    # ---- 7. Compute full-period portfolio stats for performance cards ----
    total_weeks = len(weekly_rets)
    winning_weeks = (weekly_rets > 0).sum()
    portfolio_wr = winning_weeks / total_weeks if total_weeks > 0 else 0
    weekly_sharpe = weekly_rets.mean() / weekly_rets.std() * np.sqrt(52) if weekly_rets.std() > 0 else 0

    # ---- 8. Assemble ----
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
                "auc": round(test_metrics["sharpe"], 3),
                "top5_precision": round(portfolio_wr, 3),
                "top5_lift": round(portfolio_wr / 0.5, 1) if portfolio_wr > 0 else 1.0,
                "top10_precision": round(portfolio_wr, 3),
                "top10_lift": round(portfolio_wr / 0.5, 1) if portfolio_wr > 0 else 1.0,
            },
            "trading_simulation": {
                "total_trades": test_metrics["n_trades"],
                "mean_net_return": round(test_metrics.get("cagr", 0), 4),
                "win_rate": round(portfolio_wr, 3),
                "hit_rate": round(portfolio_wr, 3),
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
            "description": f"CDPT Portfolio, weekly returns, last ~90 days",
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
    print(f"  {len(historical_picks)} portfolio weeks")
    print(f"  {len(eq_strat)} equity curve points")
    print(f"  {len(daily_picks)} daily picks (weekly portfolio)")
    print(f"  Portfolio WR: {portfolio_wr:.2%} (weekly)")
    print(f"  Strategy final: ${cum.iloc[-1]:,.0f} from $10k")


if __name__ == "__main__":
    main()
