#!/usr/bin/env python3
"""
train.py — Daily Best Stock Picker
====================================
Answers one question: "What is the best stock to buy TODAY?"

For monthly DCA users. Shows the single highest-quality momentum
stock whenever you check. Hold for at least 3-6 months.

HOW IT WORKS:
1. Scan ~100 large-cap stocks daily
2. Filter for multi-timeframe quality momentum:
   - Market healthy (SPY near highs, trending up)
   - Stock in strong uptrend across 3m/6m/12m timeframes
   - Stock near 52-week high (trend intact)
   - Low volatility, minimal drawdown
3. Rank by composite quality-momentum score
4. Show #1 pick (or "no pick — wait" if market uncertain)

VERIFIED NO LEAKAGE:
- All features use backward-looking rolling windows only
- Forward returns used ONLY for evaluation, never selection
- Same parameters across train/valid/test (no period-specific tuning)
- 63-day buffer between data splits

Run: python train.py
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass

from prepare import (
    load_data, compute_features, evaluate_strategy,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    TEST_START, TEST_END,
    TRANSACTION_COST_BPS,
)


@dataclass
class Config:
    # --- Market health (SPY) ---
    spy_pos_range_min: float = 0.65
    spy_ret_126d_min: float = 0.01
    spy_ret_63d_min: float = 0.0
    spy_ret_21d_min: float = -0.04

    # --- Stock quality (all must pass) ---
    min_ret_252d: float = 0.10
    min_ret_126d: float = 0.05
    min_ret_63d: float = 0.02
    min_ret_21d: float = -0.03
    min_pos_range: float = 0.70
    max_drawdown_252d: float = -0.10
    min_vol_21d: float = 0.06
    max_vol_21d: float = 0.32
    min_dd_change_5d: float = -0.03

    # --- Evaluation hold period ---
    hold_days: int = 63  # 3 months


EXCLUDED = {"SPY", "VIX", "TLT", "IEF", "HYG", "GLD", "SLV", "USO",
            "DIA", "IWM", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
            "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"}


def check_market(features_dict, cfg):
    spy = features_dict.get("SPY")
    if spy is None:
        return False
    p = spy.get("position_in_52w_range", np.nan)
    r126 = spy.get("ret_126d", np.nan)
    r63 = spy.get("ret_63d", np.nan)
    r21 = spy.get("ret_21d", np.nan)
    if any(np.isnan(v) for v in [p, r126, r63, r21]):
        return False
    return (p >= cfg.spy_pos_range_min and r126 >= cfg.spy_ret_126d_min
            and r63 >= cfg.spy_ret_63d_min and r21 >= cfg.spy_ret_21d_min)


def score_stock(feat):
    r252 = max(feat.get("ret_252d", 0), 0)
    r126 = max(feat.get("ret_126d", 0), 0)
    r63 = max(feat.get("ret_63d", 0), 0)
    pr = max(feat.get("position_in_52w_range", 0), 0)
    dd = feat.get("drawdown_252d", -1)
    vol = feat.get("vol_21d", 0.5)
    mom = (min(r252 / 0.30, 1.0) * 0.35 + min(r126 / 0.20, 1.0) * 0.30
           + min(r63 / 0.10, 1.0) * 0.20 + min(pr, 1.0) * 0.15)
    vp = min(max(0, (vol - 0.20) / 0.20), 1.0)
    dp = min(max(0, abs(dd) / 0.10), 1.0)
    return max(0, mom * (1 - 0.3 * vp) * (1 - 0.3 * dp))


def get_daily_pick(features_dict, cfg):
    if not check_market(features_dict, cfg):
        return None, []
    candidates = []
    for ticker, feat in features_dict.items():
        if ticker in EXCLUDED:
            continue
        vals = [feat.get(k, np.nan) for k in ["ret_252d", "ret_126d", "ret_63d",
                "ret_21d", "position_in_52w_range", "drawdown_252d", "vol_21d", "dd_change_5d"]]
        if any(np.isnan(v) for v in vals):
            continue
        if (feat["ret_252d"] < cfg.min_ret_252d or feat["ret_126d"] < cfg.min_ret_126d
            or feat["ret_63d"] < cfg.min_ret_63d or feat["ret_21d"] < cfg.min_ret_21d
            or feat["position_in_52w_range"] < cfg.min_pos_range
            or feat["drawdown_252d"] < cfg.max_drawdown_252d
            or feat["vol_21d"] < cfg.min_vol_21d or feat["vol_21d"] > cfg.max_vol_21d
            or feat.get("dd_change_5d", -1) < cfg.min_dd_change_5d):
            continue
        candidates.append((ticker, score_stock(feat), feat))
    candidates.sort(key=lambda x: x[1], reverse=True)
    if not candidates:
        return None, []
    return candidates[0], candidates[:5]


def run_picker_backtest(data_dict, start_date, end_date, cfg=None):
    if cfg is None:
        cfg = Config()
    market_close = data_dict.get("SPY", pd.DataFrame()).get("Close")
    features_cache = {}
    for ticker, df in data_dict.items():
        if "Close" not in df.columns:
            continue
        try:
            features_cache[ticker] = compute_features(df["Close"], df.get("Volume"), market_close)
        except Exception:
            pass

    dates = data_dict["SPY"].loc[start_date:end_date].index
    picks = []
    for date in dates:
        fd = {}
        for ticker, feats in features_cache.items():
            if date in feats.index:
                fd[ticker] = feats.loc[date].to_dict()
        result, top5 = get_daily_pick(fd, cfg)
        if result is None:
            picks.append({"date": date, "ticker": None, "score": 0})
            continue
        ticker, score, feat = result
        entry_price = data_dict[ticker].loc[date, "Close"]
        df = data_dict[ticker]
        idx = df.index.get_loc(date)

        # Check multiple forward return horizons
        fwd = {}
        for h_name, h_days in [("3m", 63), ("6m", 126), ("1y", 252)]:
            fi = idx + h_days
            if fi < len(df):
                fwd[h_name] = round(float(df.iloc[fi]["Close"] / entry_price - 1), 4)
            else:
                fwd[h_name] = None

        picks.append({
            "date": date, "ticker": ticker, "score": round(score, 3),
            "entry_price": round(float(entry_price), 2),
            "fwd_3m": fwd.get("3m"), "fwd_6m": fwd.get("6m"), "fwd_1y": fwd.get("1y"),
        })
    return pd.DataFrame(picks)


def period_stats(picks_df):
    has_pick = picks_df[picks_df["ticker"].notna()]
    total_days = len(picks_df)
    result = {"pick_rate": f"{len(has_pick)}/{total_days} ({len(has_pick)/total_days:.0%})"}
    for horizon in ["3m", "6m", "1y"]:
        col = f"fwd_{horizon}"
        if col not in has_pick.columns:
            result[horizon] = {"n": 0}
            continue
        valid = has_pick[has_pick[col].notna()]
        if len(valid) == 0:
            result[horizon] = {"n": 0}
            continue
        n_pos = (valid[col] > 0).sum()
        result[horizon] = {
            "n": len(valid),
            "hit_rate": f"{n_pos}/{len(valid)} ({n_pos/len(valid):.1%})",
            "avg": f"{valid[col].mean():.1%}",
            "median": f"{valid[col].median():.1%}",
            "min": f"{valid[col].min():.1%}",
            "max": f"{valid[col].max():.1%}",
        }
    return result


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded")
    cfg = Config()

    print(f"\nDaily Best Stock Picker")
    print(f"  Market: SPY range>{cfg.spy_pos_range_min}, ret_126d>{cfg.spy_ret_126d_min}")
    print(f"  Stock:  ret_252d>{cfg.min_ret_252d}, ret_126d>{cfg.min_ret_126d}, "
          f"range>{cfg.min_pos_range}, vol<{cfg.max_vol_21d}")

    STOCK_PERIODS = [
        ("1999-2003", "1999-06-01", "2003-12-31"),
        ("2004-2007", "2004-01-01", "2007-12-31"),
        ("2008-2009", "2008-01-01", "2009-12-31"),
        ("TRAIN", TRAIN_START, TRAIN_END),
        ("VALID", VALID_START, VALID_END),
        ("TEST", TEST_START, TEST_END),
        ("FULL", "1999-06-01", TEST_END),
    ]
    for pname, s, e in STOCK_PERIODS:
        print(f"\n{'='*60}")
        print(f"{pname}: {s} to {e}")
        print(f"{'='*60}")
        picks = run_picker_backtest(data, s, e, cfg)
        stats = period_stats(picks)
        print(f"  Picks: {stats['pick_rate']}")
        for h in ["3m", "6m", "1y"]:
            hs = stats[h]
            if hs.get("n", 0) > 0:
                print(f"  {h.upper()} return: hit={hs['hit_rate']}, "
                      f"avg={hs['avg']}, med={hs['median']}, "
                      f"range=[{hs['min']}, {hs['max']}]")

    # ============================================================
    # GENERATE WEB DATA
    # ============================================================
    print(f"\n{'='*60}")
    print("Generating web data...")

    market_close = data["SPY"]["Close"]
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

    today_result, today_top5 = get_daily_pick(latest_features, cfg)
    market_ok = check_market(latest_features, cfg)

    # Build all stocks list
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

    # Historical picks — show performance-to-date, not just "pending"
    test_picks = run_picker_backtest(data, TEST_START, TEST_END, cfg)
    test_valid = test_picks[test_picks["ticker"].notna()]

    recent_picks = []
    for _, row in test_valid.tail(90).iterrows():
        ticker = row["ticker"]
        entry_price = row["entry_price"]
        # Get current/latest price for this stock
        current_price = float(data[ticker]["Close"].iloc[-1]) if ticker in data else entry_price
        perf_to_date = round((current_price / entry_price - 1) * 100, 2)
        days_since = (data["SPY"].index[-1] - row["date"]).days

        recent_picks.append({
            "date": str(row["date"].date()), "ticker": ticker,
            "score": row["score"], "entry_price": entry_price,
            "current_price": round(current_price, 2),
            "perf_to_date": perf_to_date,
            "days_held": days_since,
            "return_3m": round(float(row["fwd_3m"]) * 100, 2) if row["fwd_3m"] is not None else None,
            "return_6m": round(float(row["fwd_6m"]) * 100, 2) if row["fwd_6m"] is not None else None,
        })

    # Consolidate picks by ticker — group entries for chart view
    picks_by_ticker = {}
    for p in recent_picks:
        t = p["ticker"]
        if t not in picks_by_ticker:
            picks_by_ticker[t] = {"ticker": t, "entries": [], "count": 0}
        picks_by_ticker[t]["entries"].append({
            "date": p["date"], "price": p["entry_price"], "score": p["score"],
            "perf": p["perf_to_date"], "return_3m": p["return_3m"],
            "return_6m": p["return_6m"],
        })
        picks_by_ticker[t]["count"] += 1
    # Sort by most frequently picked
    picks_consolidated = sorted(picks_by_ticker.values(), key=lambda x: -x["count"])

    # Performance across ALL periods
    all_period_stats = {}
    for pname, s, e in [("1999-2003", "1999-06-01", "2003-12-31"),
                         ("2004-2007", "2004-01-01", "2007-12-31"),
                         ("2008-2009", "2008-01-01", "2009-12-31"),
                         ("train", TRAIN_START, TRAIN_END),
                         ("valid", VALID_START, VALID_END),
                         ("test", TEST_START, TEST_END),
                         ("full", "1999-06-01", TEST_END)]:
        try:
            pp = run_picker_backtest(data, s, e, cfg)
            all_period_stats[pname] = period_stats(pp)
        except Exception as ex:
            print(f"  Warning: {pname} failed: {ex}")

    docs_dir = os.path.join(os.path.dirname(__file__), "docs", "data")
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
        "performance": all_period_stats,
        "recent_picks": recent_picks,
        "picks_by_ticker": picks_consolidated[:20],
        "all_stocks": all_stocks,
        "qualifying": [s for s in all_stocks if s["qualifies"]],
    }

    # Custom JSON serializer to handle NaN (not valid JSON)
    def clean_nan(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        return obj

    with open(os.path.join(docs_dir, "full.json"), "w") as f:
        json.dump(clean_nan(full_data), f, indent=2)

    for stock in all_stocks[:30]:
        ticker = stock["ticker"]
        if ticker not in data:
            continue
        chart = [{"date": str(dt.date()), "price": round(float(row["Close"]), 2)}
                 for dt, row in data[ticker].tail(504).iterrows()]
        # Embed pick entry points for this ticker (for chart dots)
        ticker_picks = picks_by_ticker.get(ticker, {}).get("entries", [])
        ticker_data = {**stock, "chart": chart, "pick_entries": ticker_picks}
        with open(os.path.join(docs_dir, "tickers", f"{ticker}.json"), "w") as f:
            json.dump(clean_nan(ticker_data), f, indent=2)

    print(f"  Today's pick: {today_result[0] if today_result else 'NONE (wait)'}")
    print(f"  {len([s for s in all_stocks if s['qualifies']])} qualifying, {len(all_stocks)} scanned")
    print(f"  Data written to {docs_dir}/")
