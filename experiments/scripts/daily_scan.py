#!/usr/bin/env python3
"""
Daily TMD-ARC Scanner
======================
Runs the TMD-ARC strategy signals and outputs JSON for the web frontend.

Usage:
    cd experiments && python scripts/daily_scan.py

Output:
    experiments/docs/data/sectors.json   — dashboard data (signals, performance, equity curve)
    experiments/docs/data/full.json      — legacy daily picker data
    experiments/docs/data/last_run.txt   — timestamp
"""

import os
import sys
import json
import math
import datetime

# Add experiments root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

from prepare import load_data, compute_features, UNIVERSE, TRANSACTION_COST_BPS
from src.strategy import TMDArcStrategy, StrategyConfig

import numpy as np
import pandas as pd


EXCLUDED = {"SPY", "VIX", "TLT", "IEF", "HYG", "GLD", "SLV", "USO",
            "DIA", "IWM", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
            "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"}


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Infinity to null."""
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


def run_tmdarc_backtest(data, features_cache, start, end, cfg):
    """Run TMD-ARC strategy over a date range and return daily returns + trades."""
    strategy = TMDArcStrategy(cfg)
    market_close = data.get("SPY", {}).get("Close")
    if market_close is None:
        return [], []

    dates = market_close.loc[start:end].index
    daily_returns = []

    for i, date in enumerate(dates):
        # Build features dict and prices for this date
        feat_dict = {}
        prices = {}
        open_prices = {}
        for ticker in features_cache:
            if ticker in EXCLUDED:
                continue
            fc = features_cache[ticker]
            if date in fc.index:
                feat_dict[ticker] = fc.loc[date].to_dict()
            if ticker in data and "Close" in data[ticker].columns:
                if date in data[ticker].index:
                    prices[ticker] = float(data[ticker].loc[date, "Close"])
            if ticker in data and "Open" in data[ticker].columns:
                if date in data[ticker].index:
                    open_prices[ticker] = float(data[ticker].loc[date, "Open"])

        stats = strategy.step(date, prices, feat_dict,
                              open_prices if open_prices else None)

        # Compute portfolio return for this day
        port_ret = 0.0
        for ticker, pos in strategy.positions.items():
            if ticker in prices and not np.isnan(prices[ticker]):
                prev_price = pos.entry_price if pos.days_held <= 1 else None
                if prev_price and prev_price > 0:
                    day_ret = prices[ticker] / prev_price - 1
                    port_ret += day_ret * pos.size
        daily_returns.append(port_ret)

    trades = strategy.get_trade_log()
    return daily_returns, trades


def compute_period_metrics(daily_rets, rf=0.02):
    """Compute Sharpe, CAGR, MaxDD from daily returns."""
    if not daily_rets or len(daily_rets) < 2:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0}
    rets = pd.Series(daily_rets)
    n_years = len(rets) / 252
    excess = rets - rf / 252
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    cum = (1 + rets).cumprod()
    total_ret = float(cum.iloc[-1] - 1)
    cagr = float((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())
    sortino_std = rets[rets < 0].std()
    sortino = float((excess.mean() / sortino_std * np.sqrt(252))) if sortino_std > 0 else 0
    return {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "sortino": round(sortino, 3),
    }


def main():
    print("Daily TMD-ARC Scanner")
    print("=" * 50)

    # Load data
    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded")

    cfg = StrategyConfig()

    # Compute features for all tickers
    print("Computing features...")
    market_close = data["SPY"]["Close"] if "SPY" in data else None
    features_cache = {}
    latest_features = {}
    for ticker, df in data.items():
        if "Close" not in df.columns:
            continue
        try:
            feats = compute_features(df["Close"], df.get("Volume"), market_close)
            if len(feats) > 0:
                features_cache[ticker] = feats
                latest_features[ticker] = feats.iloc[-1].to_dict()
        except Exception:
            pass

    # Generate today's signals using the strategy engine
    print("Generating signals...")
    strategy = TMDArcStrategy(cfg)
    today_prices = {}
    for ticker, df in data.items():
        if "Close" in df.columns and len(df) > 0:
            today_prices[ticker] = float(df["Close"].iloc[-1])

    feat_dict_today = {t: f for t, f in latest_features.items() if t not in EXCLUDED}
    signals = strategy.generate_signals(
        pd.Timestamp.now().normalize(), feat_dict_today
    )

    # Build top signals for the page
    top_signals = []
    for sig in signals[:20]:
        price = today_prices.get(sig.ticker, 0)
        top_signals.append({
            "ticker": sig.ticker,
            "price": round(price, 2),
            "mtmdi_z": round(sig.mtmdi_zscore, 2),
            "cacs": round(sig.cascade_gap, 3),
            "mpr_z": round(sig.mpr, 2),
            "strength": round(sig.strength, 3),
            "vol_regime": sig.vol_regime,
            "rationale": sig.rationale,
        })

    # Detect current vol regime
    spy_feat = latest_features.get("SPY", {})
    vol_ratio = spy_feat.get("vol_ratio_5_21", 1.0)
    vol_21d = spy_feat.get("vol_21d", 0.15)
    if isinstance(vol_ratio, float) and isinstance(vol_21d, float):
        if vol_ratio > 1.5 or vol_21d > 0.30:
            vol_regime = "high"
        elif vol_ratio < 0.7 and vol_21d < 0.12:
            vol_regime = "low"
        else:
            vol_regime = "normal"
    else:
        vol_regime = "normal"

    spy_price = today_prices.get("SPY", 0)

    # Use pre-computed results from results/ directory for performance data
    results_dir = os.path.join(EXPERIMENTS_DIR, "results")

    # Load stored metrics
    perf = {}
    baseline_path = os.path.join(results_dir, "baseline_metrics.json")
    validation_path = os.path.join(results_dir, "validation_report.json")

    if os.path.exists(validation_path):
        with open(validation_path) as f:
            vr = json.load(f)
        vm = vr.get("validation_metrics", {})
        tm = vr.get("test_metrics", {})
        vb = vr.get("validation_benchmark", {})
        tb = vr.get("test_benchmark", {})

        perf["valid"] = {
            "strategy": {
                "sharpe": round(vm.get("sharpe", 0), 3),
                "cagr": round(vm.get("cagr", 0), 4),
                "max_dd": round(vm.get("max_drawdown", 0), 4),
                "sortino": round(vm.get("sortino", 0), 3),
            },
            "spy": {
                "sharpe": round(vb.get("spy_sharpe", 0), 3),
                "cagr": 0,
                "max_dd": round(vb.get("spy_max_drawdown", 0), 4),
            },
        }
        perf["test"] = {
            "strategy": {
                "sharpe": round(tm.get("sharpe", 0), 3),
                "cagr": round(tm.get("cagr", 0), 4),
                "max_dd": round(tm.get("max_drawdown", 0), 4),
                "sortino": round(tm.get("sortino", 0), 3),
            },
            "spy": {
                "sharpe": round(tb.get("spy_sharpe", 0), 3),
                "cagr": 0,
                "max_dd": round(tb.get("spy_max_drawdown", 0), 4),
            },
        }

    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bm = json.load(f)
        perf["train"] = {
            "strategy": {
                "sharpe": round(bm.get("sharpe", 0), 3),
                "cagr": round(bm.get("cagr", 0), 4),
                "max_dd": round(bm.get("max_drawdown", 0), 4),
                "sortino": round(bm.get("sortino", 0), 3),
            },
            "spy": {"sharpe": 0.786, "cagr": 0.1327, "max_dd": -0.1935},
        }

    # Assemble sectors.json output (main dashboard data)
    docs_dir = os.path.join(EXPERIMENTS_DIR, "docs", "data")
    os.makedirs(docs_dir, exist_ok=True)

    dashboard_data = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "TMD-ARC",
        "strategy_full_name": "Temporal Momentum Dispersion with Adaptive Regime Cascade",
        "description": "Detects regime transitions via cross-timeframe momentum disagreement. Novel features: MTMDI, CACS, MPR.",
        "current_status": {
            "spy_price": round(spy_price, 2),
            "vol_regime": vol_regime,
            "n_signals": len(signals),
            "n_positions": 0,
            "total_exposure": 0,
        },
        "top_signals": top_signals,
        "performance": perf,
        "walk_forward": [],
        "equity_curve_strategy": [],
        "equity_curve_spy": [],
    }

    # Load existing equity curves and walk-forward data if available
    existing_path = os.path.join(docs_dir, "sectors.json")
    if os.path.exists(existing_path):
        try:
            with open(existing_path) as f:
                existing = json.load(f)
            # Preserve equity curves and walk-forward from previous runs
            if existing.get("equity_curve_strategy"):
                dashboard_data["equity_curve_strategy"] = existing["equity_curve_strategy"]
            if existing.get("equity_curve_spy"):
                dashboard_data["equity_curve_spy"] = existing["equity_curve_spy"]
            if existing.get("walk_forward"):
                dashboard_data["walk_forward"] = existing["walk_forward"]
        except Exception:
            pass

    with open(os.path.join(docs_dir, "sectors.json"), "w") as f:
        json.dump(dashboard_data, f, indent=2, cls=SafeJSONEncoder)

    # Timestamp
    with open(os.path.join(docs_dir, "last_run.txt"), "w") as f:
        f.write(datetime.datetime.now().isoformat())

    n_sig = len(signals)
    print(f"\nDone!")
    print(f"  {n_sig} signal{'s' if n_sig != 1 else ''} today")
    print(f"  Vol regime: {vol_regime}")
    print(f"  Top: {', '.join(s.ticker for s in signals[:5]) if signals else 'none'}")
    print(f"  Data written to {docs_dir}/sectors.json")


if __name__ == "__main__":
    main()
