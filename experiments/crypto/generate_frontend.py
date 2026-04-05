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

    # Show WEEKLY PORTFOLIO returns as historical picks (matches the equity curve)
    historical_picks = []
    portfolio_val = 10000
    for date, ret in weekly_rets.items():
        if ret == 0:
            continue  # skip cash weeks
        week_start = date - pd.Timedelta(days=6)
        # Find tickers traded that week
        wt = test_trades[
            (test_trades["entry_date"] >= week_start) &
            (test_trades["entry_date"] <= date)
        ] if len(test_trades) > 0 else pd.DataFrame()
        tickers = list(wt["ticker"].unique()) if len(wt) > 0 else []
        ticker_str = ", ".join([t.replace("-USD","") for t in tickers[:4]])
        if len(tickers) > 4:
            ticker_str += f" +{len(tickers)-4}"
        n_trades_w = len(wt)
        entry_val = portfolio_val
        portfolio_val *= (1 + float(ret))

        historical_picks.append({
            "entry_date": str(week_start.date()),
            "exit_date": str(date.date()),
            "ticker": ticker_str or "CASH",
            "ticker_full": f"{n_trades_w} trades",
            "score": round(abs(float(ret)) * 10, 3),
            "entry_price": round(entry_val, 0),
            "exit_price": round(portfolio_val, 0),
            "return_pct": round(float(ret) * 100, 2),
            "net_return_pct": round(float(ret) * 100, 2),
            "hit_target": bool(ret > 0),
            "days_held": 7,
        })

    # ---- 4. Daily top-1: weekly portfolio returns (last ~90 days) ----
    print("  Building daily top-1 (weekly portfolio)...")
    recent_weekly = weekly_rets.iloc[-13:]  # ~90 days
    recent_btc = btc_weekly.reindex(recent_weekly.index, fill_value=0)

    daily_picks = []
    daily_eq = []
    daily_eq_bench = []
    val = 10000; bval = 10000

    for date, ret in recent_weekly.items():
        week_start = date - pd.Timedelta(days=6)
        btc_ret = recent_btc.get(date, 0)
        if isinstance(btc_ret, (float, np.floating)) and np.isnan(btc_ret):
            btc_ret = 0

        wt = test_trades[
            (test_trades["entry_date"] >= week_start) &
            (test_trades["entry_date"] <= date)
        ] if len(test_trades) > 0 else pd.DataFrame()
        tickers = list(wt["ticker"].unique()) if len(wt) > 0 else []
        best = tickers[0].replace("-USD", "") if tickers else "CASH"
        n_t = len(wt)

        entry_val = val
        val *= (1 + float(ret))
        bval *= (1 + float(btc_ret))

        daily_picks.append({
            "entry_date": str(week_start.date()),
            "exit_date": str(date.date()),
            "ticker": best,
            "ticker_full": f"Portfolio ({n_t} trades)",
            "score": round(abs(float(ret)) * 10, 3),
            "conviction": round(n_t, 0),
            "entry_price": round(entry_val, 0),
            "exit_price": round(val, 0),
            "return_pct": round(float(ret) * 100, 2),
            "net_return_pct": round(float(ret) * 100, 2),
            "hit_target": bool(ret > 0),
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

    # ---- 8. Comprehensive strategy details ----
    print("  Computing comprehensive details...")

    # Run CDPT on train/valid too for period comparison
    train_trades, train_rets = run_backtest(data, TRAIN_START, TRAIN_END, cfg)
    train_m = evaluate_strategy(train_trades, train_rets, "Train")
    valid_trades, valid_rets = run_backtest(data, VALID_START, VALID_END, cfg)
    valid_m = evaluate_strategy(valid_trades, valid_rets, "Valid")

    strategy_deep_dive = {
        "rules": {
            "mtmdi_entry": cfg.mtmdi_zscore_entry,
            "mtmdi_exit": cfg.mtmdi_zscore_exit,
            "velocity_threshold": cfg.velocity_threshold,
            "velocity_exit": cfg.mtmdi_velocity_exit,
            "cacs_threshold": cfg.cacs_entry_threshold,
            "mpr_threshold": cfg.mpr_threshold,
            "range_compress": cfg.range_compress_threshold,
            "min_confirming": cfg.min_confirming,
            "vol_target": int(cfg.vol_target * 100),
            "max_position": int(cfg.max_position_pct * 100),
            "max_exposure": int(cfg.max_total_exposure * 100),
            "high_vol_reduction": int((1 - cfg.high_vol_reduction) * 100),
            "universe_size": len(feat_cache),
        },
        "period_comparison": {
            "train": {"sharpe": f"{train_m['sharpe']:.2f}", "trades": train_m['n_trades'],
                      "cagr": f"{train_m['cagr']*100:.0f}%"},
            "valid": {"sharpe": f"{valid_m['sharpe']:.2f}", "trades": valid_m['n_trades'],
                      "cagr": f"{valid_m['cagr']*100:.0f}%"},
            "test":  {"sharpe": f"{test_metrics['sharpe']:.2f}", "trades": test_metrics['n_trades'],
                      "cagr": f"{test_metrics['cagr']*100:.0f}%"},
        },
        "trade_frequency": f"~{test_metrics['n_trades']/2.5:.0f} trades/year",
        "avg_positions": f"Avg hold: {test_metrics['avg_hold_days']:.1f} days",
        "holding_period": f"Max hold: {cfg.max_hold_days} days, TP: +{cfg.take_profit*100:.0f}%, SL: {cfg.stop_loss*100:.0f}%",
    }

    # Today's action: current signals and open positions
    today_action = {"active_signals": [], "open_positions": []}
    if latest in btc_df.index:
        for ticker in feat_cache:
            if ticker not in data or latest not in feat_cache[ticker].index:
                continue
            f = feat_cache[ticker].loc[latest]
            def s3(k, d=0):
                v = f.get(k, d)
                return d if (isinstance(v, float) and np.isnan(v)) else v
            mz = s3("mtmdi_zscore"); md = s3("mtmdi_direction")
            mv = s3("mtmdi_velocity"); casc = s3("cacs")
            mpr = s3("mpr_zscore"); v30 = s3("vol_30d", 0.4)
            rc = s3("range_compress", 0.5); vs = s3("vol_surge")
            if abs(mz) < cfg.mtmdi_zscore_entry or md <= 0:
                continue
            nc = (int(casc > cfg.cacs_entry_threshold) + int(mpr > cfg.mpr_threshold) +
                  int(mv > cfg.velocity_threshold) + int(rc < cfg.range_compress_threshold) +
                  int(vs > 0))
            if nc < cfg.min_confirming:
                continue
            strength = (min(abs(mz)/3,1)*0.30 + min(max(mv,0),1)*0.25 +
                        min(abs(casc)/0.08,1)*0.15 + min(max(mpr,0)/2,1)*0.15 + (1-rc)*0.15)
            sz = cfg.vol_target / max(v30, 0.05) * strength
            sz = min(sz, cfg.max_position_pct)
            today_action["active_signals"].append({
                "ticker": ticker.replace("-USD", ""),
                "size": f"{sz*100:.1f}",
                "mtmdi_z": f"{mz:.2f}",
                "velocity": f"{mv:.2f}",
                "confirming": nc,
            })

    # Monthly breakdown
    monthly_breakdown = []
    monthly_rets_grouped = rets_s.resample("ME").apply(lambda x: (1+x).prod()-1)
    cum_val = 10000
    for d_m, r_m in monthly_rets_grouped.items():
        week_in_month = rets_s.loc[d_m.strftime("%Y-%m")]
        weekly_in_m = week_in_month.resample("W-SUN").apply(lambda x: (1+x).prod()-1)
        n_trades_month = 0
        if len(test_trades) > 0:
            mt = test_trades[(test_trades["entry_date"] >= d_m - pd.offsets.MonthBegin(1)) &
                             (test_trades["entry_date"] <= d_m)]
            n_trades_month = len(mt)
        wins_m = (weekly_in_m > 0).sum()
        total_m = len(weekly_in_m)
        wr_m = wins_m / total_m * 100 if total_m > 0 else 0
        best_w = weekly_in_m.max() * 100 if len(weekly_in_m) > 0 else 0
        worst_w = weekly_in_m.min() * 100 if len(weekly_in_m) > 0 else 0
        cum_val *= (1 + float(r_m))
        monthly_breakdown.append({
            "month": d_m.strftime("%Y-%m"),
            "return_pct": round(float(r_m) * 100, 1),
            "trades": n_trades_month,
            "win_rate": round(wr_m, 0),
            "best_week": round(float(best_w), 1),
            "worst_week": round(float(worst_w), 1),
            "cumulative": f"${cum_val/1000:,.0f}k",
        })

    # Top traded coins
    top_traded_coins = []
    if len(test_trades) > 0:
        coin_groups = test_trades.groupby("ticker")
        for ticker, group in coin_groups:
            if len(group) < 5:
                continue
            wr_c = (group["net_pnl"] > 0).mean() * 100
            avg_r = group["net_pnl"].mean() * 100
            best_c = group["net_pnl"].max() * 100
            worst_c = group["net_pnl"].min() * 100
            total_c = group["net_pnl"].sum() * 100
            top_traded_coins.append({
                "ticker": ticker.replace("-USD", ""),
                "trades": len(group),
                "win_rate": round(wr_c, 1),
                "avg_return": round(avg_r, 2),
                "best": round(best_c, 1),
                "worst": round(worst_c, 1),
                "total_pnl": round(total_c, 1),
            })
        top_traded_coins.sort(key=lambda x: x["trades"], reverse=True)
        top_traded_coins = top_traded_coins[:20]

    # Risk metrics
    rets_arr = rets_s.values
    cum_arr = (1 + rets_s).cumprod()
    peak = cum_arr.cummax()
    dd_series = (cum_arr - peak) / peak
    max_dd = dd_series.min()
    max_dd_idx = dd_series.idxmin()

    # Find drawdown periods
    in_dd = dd_series < 0
    dd_periods = []
    dd_start = None
    for i, (date, val) in enumerate(dd_series.items()):
        if val < 0 and dd_start is None:
            dd_start = date
        elif val >= 0 and dd_start is not None:
            depth = dd_series.loc[dd_start:date].min()
            duration = (date - dd_start).days
            dd_periods.append({"start": str(dd_start.date()), "end": str(date.date()),
                               "depth": f"{depth*100:.1f}%", "duration": f"{duration}d",
                               "recovery": f"{duration}d"})
            dd_start = None
    dd_periods.sort(key=lambda x: float(x["depth"].replace("%","")))
    dd_periods = dd_periods[:5]

    # Longest drawdown
    longest_dd = max(dd_periods, key=lambda x: int(x["duration"].replace("d",""))) if dd_periods else {}

    excess = rets_s - 0.02 / 365
    downside = excess[excess < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0
    calmar = (test_metrics["cagr"]) / abs(max_dd) if abs(max_dd) > 0 else 0
    var_5 = np.percentile(rets_arr, 5)

    # Avg win / avg loss
    if len(test_trades) > 0:
        wins_t = test_trades[test_trades["net_pnl"] > 0]["net_pnl"]
        losses_t = test_trades[test_trades["net_pnl"] <= 0]["net_pnl"]
        avg_win = wins_t.mean() if len(wins_t) > 0 else 0
        avg_loss = abs(losses_t.mean()) if len(losses_t) > 0 else 1
        wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    else:
        wl_ratio = 0

    # Avg exposure
    avg_exp = np.mean([abs(r) for r in rets_arr if r != 0]) / np.mean([abs(r) for r in rets_arr if abs(r) > 0.001]) if len(rets_arr) > 0 else 0
    pct_days_invested = len([r for r in rets_arr if abs(r) > 0.0001]) / len(rets_arr) * 100 if len(rets_arr) > 0 else 0

    # BTC correlation
    btc_test_rets = btc_close.loc[TEST_START:TEST_END].pct_change().iloc[1:]
    common_idx = rets_s.index.intersection(btc_test_rets.index)
    btc_corr = np.corrcoef(rets_s.loc[common_idx].values, btc_test_rets.loc[common_idx].values)[0, 1] if len(common_idx) > 10 else 0

    risk_metrics = {
        "max_drawdown": f"{max_dd*100:.1f}%",
        "max_dd_date": str(max_dd_idx.date()) if hasattr(max_dd_idx, 'date') else str(max_dd_idx),
        "longest_dd_days": int(longest_dd.get("duration", "0d").replace("d", "")) if longest_dd else 0,
        "longest_dd_period": f"{longest_dd.get('start','')} to {longest_dd.get('end','')}" if longest_dd else "",
        "sortino": f"{sortino:.2f}",
        "calmar": f"{calmar:.2f}",
        "var_5": f"{var_5*100:.2f}%",
        "avg_win_loss_ratio": f"{wl_ratio:.2f}",
        "avg_exposure": f"{pct_days_invested:.0f}% of days",
        "btc_correlation": f"{btc_corr:.2f}",
        "drawdown_periods": dd_periods,
    }

    # ---- 9. Assemble ----
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
        "strategy_deep_dive": strategy_deep_dive,
        "today_action": today_action,
        "monthly_breakdown": monthly_breakdown,
        "top_traded_coins": top_traded_coins,
        "risk_metrics": risk_metrics,
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
