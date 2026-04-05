#!/usr/bin/env python3
"""Generate ccrl_sectors.json for the experiments frontend."""
import os, sys, json, datetime, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data
from sector_strategy_v2 import (
    ASRPStrategy, compute_metrics, spy_bh_metrics,
    BENCHMARK, SAFE_HAVENS, SECTOR_ETFS, SECTOR_NAMES,
    backtest, SLIPPAGE_BPS,
)


def backtest_c2c(data, start, end, weight_fn):
    """Close-to-close, no entry-day return."""
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    daily_rets = []
    current_w = {}
    last_month = None
    just_reb = False

    for date in dates:
        idx = spy.index.get_loc(date)
        if idx < 252:
            daily_rets.append(0.0)
            continue
        month = date.month
        reb = (last_month is not None and month != last_month)
        last_month = month

        if just_reb:
            daily_rets.append(0.0)
            just_reb = False
            continue

        dr = 0.0
        for t, w in current_w.items():
            df = data.get(t)
            if df is not None and date in df.index:
                si = df.index.get_loc(date)
                if si > 0:
                    dr += (df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1) * w
        daily_rets.append(dr)

        if reb:
            current_w = weight_fn(date)
            just_reb = True

    return pd.Series(daily_rets, index=dates)


def backtest_t1(data, start, end, weight_fn, tx_bps=5):
    """T+1 open execution (most conservative)."""
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    slip = tx_bps / 10000
    daily_rets = []
    current_w = {}
    pending_w = None
    last_month = None

    for date in dates:
        idx = spy.index.get_loc(date)
        if idx < 252:
            daily_rets.append(0.0)
            continue

        if pending_w is not None:
            dr = 0.0
            for t, w in current_w.items():
                if t not in pending_w or abs(pending_w.get(t, 0) - w) > 0.005:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            prev_c = df.iloc[si - 1]["Close"]
                            today_o = df.loc[date, "Open"] if "Open" in df.columns else prev_c
                            dr += (today_o * (1 - slip) / prev_c - 1) * w
                else:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            dr += (df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1) * w
            for t, w in pending_w.items():
                if t not in current_w or abs(current_w.get(t, 0) - w) > 0.005:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        buy = today_o * (1 + slip)
                        if buy > 0:
                            dr += (df.loc[date, "Close"] / buy - 1) * w
            current_w = pending_w
            pending_w = None
            daily_rets.append(dr)
        else:
            dr = 0.0
            for t, w in current_w.items():
                df = data.get(t)
                if df is not None and date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        dr += (df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1) * w
            daily_rets.append(dr)

        month = date.month
        reb = (last_month is not None and month != last_month)
        last_month = month
        if reb:
            pending_w = weight_fn(date)

    return pd.Series(daily_rets, index=dates)


def fmt_pct(v):
    return f"{v*100:.1f}%" if not (math.isnan(v) if isinstance(v, float) else False) else "?"


def main():
    print("Generating sector frontend data...")
    data = load_data()
    strategy = ASRPStrategy(data)
    spy_close = data[BENCHMARK]["Close"]

    periods = [
        ("train", "2010-01-01", "2019-12-31"),
        ("valid", "2020-01-01", "2022-12-31"),
        ("test", "2023-01-01", "2026-03-15"),
    ]

    # Run both execution modes
    print("  Running C2C backtest...")
    c2c_results = {}
    for name, s, e in periods:
        st = ASRPStrategy(data)
        wf = st.make_signal_fn()
        rets = backtest_c2c(data, s, e, wf)
        m = compute_metrics(rets)
        spy_m = spy_bh_metrics(data, s, e)
        c2c_results[name] = {"metrics": m, "spy": spy_m, "rets": rets}
        print(f"    {name}: Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1%}")

    print("  Running T+1 open backtest...")
    t1_results = {}
    for name, s, e in periods:
        st = ASRPStrategy(data)
        wf = st.make_signal_fn()
        rets = backtest_t1(data, s, e, wf)
        m = compute_metrics(rets)
        spy_m = spy_bh_metrics(data, s, e)
        t1_results[name] = {"metrics": m, "spy": spy_m, "rets": rets}
        print(f"    {name}: Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1%}")

    # Execution comparison block
    exec_comp = {
        "close_to_close": {},
        "t1_open": {},
        "spy_bh": {},
    }
    for name in ["train", "valid", "test"]:
        cm = c2c_results[name]["metrics"]
        tm = t1_results[name]["metrics"]
        sm = c2c_results[name]["spy"]
        exec_comp["close_to_close"][f"{name}_sharpe"] = f"{cm['sharpe']:.2f}"
        exec_comp["close_to_close"][f"{name}_cagr"] = fmt_pct(cm["cagr"])
        exec_comp["close_to_close"][f"{name}_dd"] = fmt_pct(cm["max_dd"])
        exec_comp["t1_open"][f"{name}_sharpe"] = f"{tm['sharpe']:.2f}"
        exec_comp["t1_open"][f"{name}_cagr"] = fmt_pct(tm["cagr"])
        exec_comp["t1_open"][f"{name}_dd"] = fmt_pct(tm["max_dd"])
        exec_comp["spy_bh"][f"{name}_sharpe"] = f"{sm['sharpe']:.2f}"
        exec_comp["spy_bh"][f"{name}_cagr"] = fmt_pct(sm["cagr"])
        exec_comp["spy_bh"][f"{name}_dd"] = fmt_pct(sm["max_dd"])

    # Walk-forward year-by-year (C2C)
    print("  Running year-by-year...")
    wf_years = []
    for year in range(2011, 2027):
        s, e = f"{year}-01-01", f"{year}-12-31"
        try:
            st = ASRPStrategy(data)
            wf = st.make_signal_fn()
            rets = backtest_c2c(data, s, e, wf)
            m = compute_metrics(rets)
            sm = spy_bh_metrics(data, s, e)
            wf_years.append({
                "year": year,
                "asrp_sharpe": f"{m['sharpe']:.2f}",
                "spy_sharpe": f"{sm['sharpe']:.2f}",
                "asrp_cagr": fmt_pct(m["cagr"]),
                "asrp_cagr_num": m["cagr"],
                "spy_cagr": fmt_pct(sm["cagr"]),
                "max_dd": fmt_pct(m["max_dd"]),
            })
        except:
            pass

    # Equity curves (test period) — C2C, T+1, SPY
    test_rets = c2c_results["test"]["rets"]
    cum = (1 + test_rets).cumprod() * 10000
    monthly = cum.resample("ME").last()
    eq_strat = [{"date": str(d.date())[:7], "value": round(float(v), 0)} for d, v in monthly.items()]

    t1_rets = t1_results["test"]["rets"]
    t1_cum = (1 + t1_rets).cumprod() * 10000
    t1_monthly = t1_cum.resample("ME").last()
    eq_t1 = [{"date": str(d.date())[:7], "value": round(float(v), 0)} for d, v in t1_monthly.items()]

    spy_test = spy_close.loc["2023-01-01":"2026-03-15"]
    spy_cum = (spy_test / spy_test.iloc[0]) * 10000
    spy_mo = spy_cum.resample("ME").last()
    eq_spy = [{"date": str(d.date())[:7], "value": round(float(v), 0)} for d, v in spy_mo.items()]

    # Current holdings
    print("  Computing current holdings...")
    latest = spy_close.index[-1]
    current_weights = strategy.get_weights(latest)
    bear = strategy.is_bear(latest)
    positions = []
    for t, w in sorted(current_weights.items(), key=lambda x: -x[1]):
        ptype = "Safe Haven" if t in SAFE_HAVENS else ("Sector ETF" if t in SECTOR_ETFS else "Stock")
        positions.append({
            "ticker": t,
            "type": ptype,
            "weight": f"{w*100:.1f}%",
        })
    eq_pct = sum(w for t, w in current_weights.items() if t not in SAFE_HAVENS)
    hedge_pct = sum(w for t, w in current_weights.items() if t in SAFE_HAVENS)

    current_holdings = {
        "regime": "BEAR" if bear else "BULL",
        "equity_pct": f"{eq_pct*100:.0f}%",
        "hedge_pct": f"{hedge_pct*100:.0f}%",
        "n_stocks": sum(1 for t in current_weights if t not in SAFE_HAVENS and t not in SECTOR_ETFS),
        "next_rebalance": "1st of month",
        "positions": positions[:20],
    }

    # Historical picks (monthly rebalances as trades)
    print("  Building historical picks...")
    weekly_rets = test_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    historical_picks = []
    port_val = 10000
    for d, ret in weekly_rets.items():
        entry_val = port_val
        port_val *= (1 + float(ret))
        historical_picks.append({
            "entry_date": str((d - pd.offsets.MonthBegin(1)).date()),
            "exit_date": str(d.date()),
            "ticker": "ASRP Portfolio",
            "ticker_full": "ASRP Portfolio",
            "score": round(abs(float(ret)) * 10, 3),
            "entry_price": round(entry_val, 0),
            "exit_price": round(port_val, 0),
            "return_pct": round(float(ret) * 100, 2),
            "net_return_pct": round(float(ret) * 100, 2),
            "hit_target": bool(ret > 0),
            "days_held": 30,
        })

    # Top5 = top allocations (group stocks into "Equities" bucket)
    # Show: safe havens individually, equities as aggregate, top 2 stocks
    top5 = []
    stock_total_w = sum(w for t, w in current_weights.items() if t not in SAFE_HAVENS and t not in SECTOR_ETFS)
    top_stocks = [(t, w) for t, w in sorted(current_weights.items(), key=lambda x: -x[1])
                  if t not in SAFE_HAVENS and t not in SECTOR_ETFS][:3]
    # Safe havens first
    for t, w in sorted(current_weights.items(), key=lambda x: -x[1]):
        if t not in SAFE_HAVENS:
            continue
        price = data[t]["Close"].iloc[-1] if t in data else 0
        ret_30d = (data[t]["Close"].iloc[-1] / data[t]["Close"].iloc[-22] - 1) if t in data and len(data[t]) > 22 else 0
        top5.append({
            "ticker": t, "ticker_full": f"{t} (Safe Haven)",
            "score": round(float(w), 4), "price": round(float(price), 2),
            "conviction": round(float(w * 100), 1),
            "returns": {"30d": round(float(ret_30d * 100), 1)},
            "vol_21d": 0, "drawdown": 0, "position_in_range": 0,
        })
    # Then top stocks
    for t, w in top_stocks:
        price = data[t]["Close"].iloc[-1] if t in data else 0
        ret_30d = (data[t]["Close"].iloc[-1] / data[t]["Close"].iloc[-22] - 1) if t in data and len(data[t]) > 22 else 0
        top5.append({
            "ticker": t, "ticker_full": f"{t} (Stock)",
            "score": round(float(w), 4), "price": round(float(price), 2),
            "conviction": round(float(w * 100), 1),
            "returns": {"30d": round(float(ret_30d * 100), 1)},
            "vol_21d": 0, "drawdown": 0, "position_in_range": 0,
        })
    top5 = top5[:5]

    # Performance cards (use C2C test)
    test_m = c2c_results["test"]["metrics"]
    test_spy = c2c_results["test"]["spy"]

    # Daily top1 (monthly returns)
    recent_monthly = weekly_rets.iloc[-6:]
    dt_picks = []
    val = 10000; bval = 10000
    dt_eq = []; dt_eq_b = []
    spy_monthly = spy_close.loc["2023-01-01":"2026-03-15"].pct_change().resample("ME").apply(lambda x: (1+x).prod()-1)
    for d, ret in recent_monthly.items():
        sr = spy_monthly.get(d, 0)
        if isinstance(sr, float) and np.isnan(sr): sr = 0
        val *= (1 + float(ret)); bval *= (1 + float(sr))
        dt_picks.append({
            "entry_date": str((d - pd.offsets.MonthBegin(1)).date()),
            "exit_date": str(d.date()),
            "ticker": "ASRP",
            "ticker_full": "Portfolio",
            "score": round(abs(float(ret))*10, 3),
            "conviction": 0,
            "entry_price": round(val/(1+float(ret)), 0),
            "exit_price": round(val, 0),
            "return_pct": round(float(ret)*100, 2),
            "net_return_pct": round(float(ret)*100, 2),
            "hit_target": bool(ret > 0),
            "days_held": 30,
            "benchmark_return_pct": round(float(sr)*100, 2),
            "spy_return_pct": round(float(sr)*100, 2),
        })
        dt_eq.append({"date": str(d.date()), "value": round(val, 0)})
        dt_eq_b.append({"date": str(d.date()), "value": round(bval, 0)})
    dt_wins = [p for p in dt_picks if p["net_return_pct"] > 0]
    dt_stats = {
        "n_trades": len(dt_picks),
        "win_rate": round(len(dt_wins)/max(len(dt_picks),1)*100, 1),
        "hit_rate": round(len(dt_wins)/max(len(dt_picks),1)*100, 1),
        "avg_net_return": round(np.mean([p["net_return_pct"] for p in dt_picks]), 1) if dt_picks else 0,
        "profit_factor": round(sum(p["net_return_pct"] for p in dt_picks if p["net_return_pct"]>0)/max(abs(sum(p["net_return_pct"] for p in dt_picks if p["net_return_pct"]<0)),0.01),2),
        "alpha": round(np.mean([p["net_return_pct"]-p["benchmark_return_pct"] for p in dt_picks]),1) if dt_picks else 0,
        "final_value": round(val, 0),
        "benchmark_final_value": round(bval, 0),
    }

    # Assemble
    print("  Assembling JSON...")
    output = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "ASRP",
        "strategy_full_name": "Adaptive Sector Risk Parity (ASRP)",
        "target": "Multi-factor stock selection + cross-asset risk parity",
        "best_config": {"rebalance": "Monthly", "regime_sma": 100, "stocks": 30},
        "model": {
            "ensemble_size": 1,
            "features": 4,
            "strategy": "Momentum-quality stock selection + TLT/GLD/IEF risk parity",
            "stocks": len(strategy.stocks),
            "sectors": len(SECTOR_ETFS),
        },
        "spy": {
            "price": round(float(spy_close.iloc[-1]), 2),
            "sma100": round(float(strategy.spy_sma.iloc[-1]), 2),
            "regime": "BEAR" if bear else "BULL",
        },
        "top5": top5,
        "performance": {
            "param_sweep": {
                "configs_tested": 16,
                "best_config": "ASRP C2C",
                "best_sharpe": test_m["sharpe"],
                "best_ann_ret": round(test_m["cagr"] * 100, 1),
            },
            "test_period": {
                "auc": test_m["sharpe"],
                "top5_precision": round(len([r for r in weekly_rets if r > 0]) / max(len(weekly_rets), 1), 3),
                "top5_lift": round(test_m["sharpe"] / max(test_spy["sharpe"], 0.01), 1),
            },
            "trading_simulation": {
                "total_trades": len(weekly_rets),
                "win_rate": round(len([r for r in weekly_rets if r > 0]) / max(len(weekly_rets), 1), 3),
                "profit_factor": round(sum(r for r in weekly_rets if r > 0) / max(abs(sum(r for r in weekly_rets if r < 0)), 0.001), 2),
                "sharpe": test_m["sharpe"],
                "annualized_return": test_m["cagr"],
                "alpha_vs_spy": round(test_m["cagr"] - test_spy["cagr"], 3),
                "hit_rate_target": round(len([r for r in weekly_rets if r > 0]) / max(len(weekly_rets), 1), 3),
            },
        },
        "execution_comparison": exec_comp,
        "walk_forward_years": wf_years,
        "current_holdings": current_holdings,
        "all_candidates": positions[:20],
        "historical_picks": historical_picks,
        "equity_curve_strategy": eq_strat,
        "equity_curve_t1": eq_t1,
        "equity_curve_spy": eq_spy,
        "daily_top1": {
            "description": "ASRP monthly portfolio returns (last 6 months)",
            "picks": dt_picks,
            "equity_curve": dt_eq,
            "equity_curve_benchmark": dt_eq_b,
            "stats": dt_stats,
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "docs", "data", "ccrl_sectors.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"  Saved to {out_path}")
    print(f"  C2C Test: Sharpe {c2c_results['test']['metrics']['sharpe']:.3f}")
    print(f"  T+1 Test: Sharpe {t1_results['test']['metrics']['sharpe']:.3f}")


if __name__ == "__main__":
    main()
