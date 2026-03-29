#!/usr/bin/env python3
"""
Daily Best Stock Picker — Scanner
===================================
Runs the Daily Best Stock Picker strategy and outputs JSON for the web frontend.

Usage:
    cd experiments && python scripts/daily_scan.py

Output:
    experiments/docs/data/full.json     — complete dashboard data
    experiments/docs/data/tickers/*.json — per-ticker detail
    experiments/docs/data/last_run.txt  — timestamp
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

# Import the strategy from train.py
from train import (
    Config, check_market, get_daily_pick, score_stock,
    run_picker_backtest, period_stats, EXCLUDED,
)
from prepare import load_data, compute_features, TEST_START, TEST_END

import numpy as np


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


def main():
    print("Daily Best Stock Picker — Scanner")
    print("=" * 50)

    # Load data
    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded")

    cfg = Config()

    # Compute features
    market_close = data["SPY"]["Close"] if "SPY" in data else None
    latest_features = {}
    for ticker, df in data.items():
        if "Close" not in df.columns:
            continue
        try:
            feats = compute_features(df["Close"], df.get("Volume"), market_close)
            if len(feats) > 0:
                latest_features[ticker] = feats.iloc[-1].to_dict()
        except Exception:
            pass

    # Today's pick
    today_result, today_top5 = get_daily_pick(latest_features, cfg)
    market_ok = check_market(latest_features, cfg)

    # All stocks with scores
    all_stocks = []
    for ticker, feat in latest_features.items():
        if ticker in EXCLUDED:
            continue
        vals = [feat.get(k, np.nan) for k in ["ret_252d", "ret_126d", "ret_63d",
                "position_in_52w_range", "drawdown_252d", "vol_21d"]]
        if any(np.isnan(v) for v in vals):
            continue
        score = score_stock(feat)
        price = data[ticker]["Close"].iloc[-1] if ticker in data else 0
        qualifies = (market_ok
            and feat["ret_252d"] >= cfg.min_ret_252d and feat["ret_126d"] >= cfg.min_ret_126d
            and feat["ret_63d"] >= cfg.min_ret_63d and feat.get("ret_21d", -1) >= cfg.min_ret_21d
            and feat["position_in_52w_range"] >= cfg.min_pos_range
            and feat["drawdown_252d"] >= cfg.max_drawdown_252d
            and cfg.min_vol_21d <= feat["vol_21d"] <= cfg.max_vol_21d
            and feat.get("dd_change_5d", -1) >= cfg.min_dd_change_5d)
        all_stocks.append({
            "ticker": ticker, "price": round(float(price), 2),
            "score": round(float(score), 3),
            "is_pick": bool(today_result and today_result[0] == ticker),
            "qualifies": bool(qualifies),
            "position_in_range": round(float(feat["position_in_52w_range"]) * 100, 1),
            "vol_21d": round(float(feat["vol_21d"]) * 100, 1),
            "drawdown": round(float(feat["drawdown_252d"]) * 100, 1),
            "returns": {
                "5d": round(float(feat.get("ret_5d", 0)) * 100, 1),
                "21d": round(float(feat.get("ret_21d", 0)) * 100, 1),
                "63d": round(float(feat["ret_63d"]) * 100, 1),
                "126d": round(float(feat["ret_126d"]) * 100, 1),
                "252d": round(float(feat["ret_252d"]) * 100, 1),
            },
            "conditions": {
                "trend_1y": bool(feat["ret_252d"] >= cfg.min_ret_252d),
                "trend_6m": bool(feat["ret_126d"] >= cfg.min_ret_126d),
                "trend_3m": bool(feat["ret_63d"] >= cfg.min_ret_63d),
                "near_high": bool(feat["position_in_52w_range"] >= cfg.min_pos_range),
                "low_dd": bool(feat["drawdown_252d"] >= cfg.max_drawdown_252d),
                "vol_ok": bool(cfg.min_vol_21d <= feat["vol_21d"] <= cfg.max_vol_21d),
            },
        })
    all_stocks.sort(key=lambda s: s["score"], reverse=True)

    # Run test period for recent history
    print("Running test period backtest for recent picks...")
    test_picks = run_picker_backtest(data, TEST_START, TEST_END, cfg)
    test_valid = test_picks[test_picks["ticker"].notna()]
    test_with_fwd = test_picks[test_picks["fwd_3m"].notna() & test_picks["ticker"].notna()]

    recent_picks = []
    for _, row in test_valid.tail(60).iterrows():
        recent_picks.append({
            "date": str(row["date"].date()), "ticker": row["ticker"],
            "score": row["score"], "entry_price": row["entry_price"],
            "return_3m": round(float(row["fwd_3m"]) * 100, 2) if row["fwd_3m"] is not None else None,
            "return_6m": round(float(row["fwd_6m"]) * 100, 2) if row["fwd_6m"] is not None else None,
        })

    # Assemble output
    docs_dir = os.path.join(EXPERIMENTS_DIR, "docs", "data")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(os.path.join(docs_dir, "tickers"), exist_ok=True)

    full_data = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "DailyPicker",
        "strategy_full_name": "Daily Best Stock Picker",
        "n_tickers": len(all_stocks),
        "market_regime": "BULLISH" if market_ok else "WAIT",
        "todays_pick": {
            "ticker": today_result[0] if today_result else None,
            "score": round(today_result[1], 3) if today_result else 0,
            "price": round(float(data[today_result[0]]["Close"].iloc[-1]), 2) if today_result else 0,
        },
        "top5": [{"ticker": t, "score": round(s, 3),
                  "price": round(float(data[t]["Close"].iloc[-1]), 2) if t in data else 0}
                 for t, s, _ in (today_top5 if today_result else [])],
        "performance": {
            "test": period_stats(test_picks),
        },
        "recent_picks": recent_picks[-30:],
        "all_stocks": all_stocks,
        "qualifying": [s for s in all_stocks if s["qualifies"]],
    }

    with open(os.path.join(docs_dir, "full.json"), "w") as f:
        json.dump(full_data, f, indent=2, cls=SafeJSONEncoder)

    for stock in all_stocks[:30]:
        ticker = stock["ticker"]
        if ticker not in data:
            continue
        chart = [{"date": str(dt.date()), "price": round(float(row["Close"]), 2)}
                 for dt, row in data[ticker].tail(252).iterrows()]
        with open(os.path.join(docs_dir, "tickers", f"{ticker}.json"), "w") as f:
            json.dump({**stock, "chart": chart}, f, indent=2, cls=SafeJSONEncoder)

    # Timestamp
    with open(os.path.join(docs_dir, "last_run.txt"), "w") as f:
        f.write(datetime.datetime.now().isoformat())

    pick_name = today_result[0] if today_result else "NONE (wait)"
    print(f"\nDone!")
    print(f"  Today's pick: {pick_name}")
    print(f"  Market: {'BULLISH' if market_ok else 'WAIT'}")
    print(f"  {len([s for s in all_stocks if s['qualifies']])} qualifying stocks")
    print(f"  Data written to {docs_dir}/")


if __name__ == "__main__":
    main()
