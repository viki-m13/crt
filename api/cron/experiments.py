"""
Vercel serverless cron — Experiments Daily Best Stock Picker.

Self-contained: inlines all logic from experiments/train.py + experiments/prepare.py
to avoid import path issues in Vercel's serverless environment.

Runs the Daily Best Stock Picker strategy, then pushes experiments/docs/data/ to GitHub.
Schedule: 22:30 UTC Mon-Fri (5:30pm ET)
"""

import os
import sys
import json
import base64
import datetime
import traceback
from http.server import BaseHTTPRequestHandler
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import requests as req


# ============================================================
# CONFIG
# ============================================================

REPO = "viki-m13/crt"
BRANCH = "main"
GITHUB_API = "https://api.github.com"

# Universe (from experiments/prepare.py)
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "XLB", "XLRE", "XLC",
    "TLT", "IEF", "HYG", "GLD", "SLV", "USO",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM",
    "CSCO", "VZ", "ADBE", "CRM", "CMCSA", "PFE", "NFLX", "INTC",
    "ABT", "KO", "PEP", "TMO", "MRK", "ABBV", "COST", "AVGO", "ACN",
    "CVX", "LLY", "MCD", "WMT", "DHR", "TXN", "NEE", "BMY", "QCOM",
    "UNP", "HON", "LOW", "AMGN", "LIN", "RTX",
    "ORCL", "PM", "UPS", "CAT", "GS", "MS", "BLK", "ISRG", "MDT",
    "DE", "ADP", "GILD", "BKNG", "SYK", "MMM", "GE", "CB", "CI",
    "SO", "DUK", "MO", "CL", "ITW", "FIS", "USB", "SCHW", "PNC",
    "CME", "AON", "ICE", "NSC", "EMR", "APD", "SHW", "ETN", "ECL",
    "WM", "ROP", "LRCX", "KLAC", "AMAT", "MCHP", "SNPS", "CDNS",
    "FTNT", "PANW", "NOW", "WDAY",
]

EXCLUDED = {
    "SPY", "VIX", "TLT", "IEF", "HYG", "GLD", "SLV", "USO",
    "DIA", "IWM", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
}

MOMENTUM_WINDOWS = [5, 10, 21, 63, 126, 252]


@dataclass
class Config:
    spy_pos_range_min: float = 0.65
    spy_ret_126d_min: float = 0.01
    spy_ret_63d_min: float = 0.0
    spy_ret_21d_min: float = -0.04
    min_ret_252d: float = 0.10
    min_ret_126d: float = 0.05
    min_ret_63d: float = 0.02
    min_ret_21d: float = -0.03
    min_pos_range: float = 0.70
    max_drawdown_252d: float = -0.10
    min_vol_21d: float = 0.06
    max_vol_21d: float = 0.32
    min_dd_change_5d: float = -0.03
    hold_days: int = 63


# ============================================================
# GITHUB PUSH HELPERS
# ============================================================

def github_headers():
    token = os.environ.get("GH_TOKEN", "")
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def get_file_sha(path):
    """Get current SHA of a file in the repo (needed for updates)."""
    url = f"{GITHUB_API}/repos/{REPO}/contents/{path}"
    r = req.get(url, headers=github_headers(), params={"ref": BRANCH})
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def push_file(path, content_bytes, message):
    """Create or update a file in the repo via GitHub Contents API."""
    url = f"{GITHUB_API}/repos/{REPO}/contents/{path}"
    encoded = base64.b64encode(content_bytes).decode("utf-8")
    body = {
        "message": message,
        "content": encoded,
        "branch": BRANCH,
    }
    sha = get_file_sha(path)
    if sha:
        body["sha"] = sha
    r = req.put(url, headers=github_headers(), json=body)
    r.raise_for_status()
    return r.status_code


def push_directory(local_dir, repo_dir, commit_msg):
    """Walk a local directory and push all files to the corresponding repo path."""
    pushed = []
    for root, dirs, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_dir)
            repo_path = f"{repo_dir}/{rel}"
            with open(local_path, "rb") as f:
                content = f.read()
            push_file(repo_path, content, commit_msg)
            pushed.append(repo_path)
    return pushed


# ============================================================
# FEATURE COMPUTATION (inlined from experiments/prepare.py)
# ============================================================

def compute_features(close, volume=None, market_close=None):
    """Compute feature set for one stock. All features use ONLY past data."""
    features = {}

    # Momentum returns
    for w in MOMENTUM_WINDOWS:
        features[f"ret_{w}d"] = np.log(close / close.shift(w))

    # MTMDI (Multi-Timeframe Momentum Dispersion Index)
    rets_df = pd.DataFrame({
        f"ret_{w}d": np.log(close / close.shift(w)) for w in MOMENTUM_WINDOWS
    })
    z_scored = pd.DataFrame(index=close.index)
    for col in rets_df.columns:
        rm = rets_df[col].rolling(252, min_periods=126).mean()
        rs = rets_df[col].rolling(252, min_periods=126).std().clip(lower=1e-8)
        z_scored[col] = (rets_df[col] - rm) / rs

    features["mtmdi"] = z_scored.std(axis=1)
    n_fast = len(MOMENTUM_WINDOWS) // 2
    features["mtmdi_direction"] = (
        z_scored.iloc[:, :n_fast].mean(axis=1) -
        z_scored.iloc[:, n_fast:].mean(axis=1)
    )
    mm = features["mtmdi"].rolling(252, min_periods=126).mean()
    ms = features["mtmdi"].rolling(252, min_periods=126).std().clip(lower=1e-8)
    features["mtmdi_zscore"] = (features["mtmdi"] - mm) / ms

    # MPR (Momentum Persistence Ratio)
    ret_fast = close.pct_change(5)
    ret_slow = close.pct_change(63)
    avg_fast = ret_fast / 5
    avg_slow = ret_slow / 63
    features["mpr"] = (avg_fast / avg_slow.clip(lower=1e-8)).clip(-10, 10)
    mpr_m = features["mpr"].rolling(252, min_periods=63).mean()
    mpr_s = features["mpr"].rolling(252, min_periods=63).std().clip(lower=1e-8)
    features["mpr_zscore"] = (features["mpr"] - mpr_m) / mpr_s

    # Volatility
    log_ret = np.log(close / close.shift(1))
    features["vol_5d"] = log_ret.rolling(5).std() * np.sqrt(252)
    features["vol_21d"] = log_ret.rolling(21).std() * np.sqrt(252)
    features["vol_63d"] = log_ret.rolling(63).std() * np.sqrt(252)
    features["vol_ratio_5_21"] = features["vol_5d"] / features["vol_21d"].clip(lower=1e-8)
    features["vol_ratio_21_63"] = features["vol_21d"] / features["vol_63d"].clip(lower=1e-8)

    # Volume
    if volume is not None:
        vol_ma20 = volume.rolling(20).mean().clip(lower=1)
        features["volume_relative"] = volume / vol_ma20
        features["volume_trend"] = volume.rolling(5).mean() / vol_ma20

    # Drawdown
    rmax = close.rolling(252, min_periods=21).max()
    features["drawdown_252d"] = (close - rmax) / rmax
    features["dd_change_5d"] = features["drawdown_252d"] - features["drawdown_252d"].shift(5)
    rmin = close.rolling(252, min_periods=21).min()
    features["position_in_52w_range"] = (
        (close - rmin) / (rmax - rmin).clip(lower=1e-8)
    )

    # CACS (Cross-Asset Cascade Score)
    if market_close is not None:
        stock_ret = close.pct_change()
        market_ret = market_close.pct_change()
        common = stock_ret.index.intersection(market_ret.index)
        sr = stock_ret.reindex(common)
        mr = market_ret.reindex(common)
        cov = sr.rolling(21, min_periods=10).cov(mr)
        var = mr.rolling(21, min_periods=10).var().clip(lower=1e-10)
        beta = cov / var
        leader_move = market_close.pct_change(21).reindex(common)
        stock_move = close.pct_change(21).reindex(common)
        features["cacs"] = (leader_move * beta - stock_move).reindex(close.index)
        features["cacs_beta"] = beta.reindex(close.index)

    result = pd.DataFrame(features, index=close.index)
    result = result.dropna(subset=["mtmdi", "vol_21d"])
    return result


# ============================================================
# STRATEGY LOGIC (inlined from experiments/train.py)
# ============================================================

def check_market(features_dict, cfg):
    """Check if market (SPY) is healthy enough for picks."""
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
    """Composite quality-momentum score for a single stock."""
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
    """Get the best stock pick for today, or None if market is unhealthy."""
    if not check_market(features_dict, cfg):
        return None, []
    candidates = []
    for ticker, feat in features_dict.items():
        if ticker in EXCLUDED:
            continue
        vals = [feat.get(k, np.nan) for k in [
            "ret_252d", "ret_126d", "ret_63d", "ret_21d",
            "position_in_52w_range", "drawdown_252d", "vol_21d", "dd_change_5d",
        ]]
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


def clean_nan(obj):
    """Replace NaN/Inf with None for valid JSON serialization."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return clean_nan(float(obj))
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj


# ============================================================
# MAIN SCAN LOGIC
# ============================================================

def run_experiments_scan():
    """Download data, compute features, pick best stock, write JSON, push to GitHub."""
    cfg = Config()

    # Download data via yfinance
    today = datetime.date.today().isoformat()
    data = {}
    for ticker in UNIVERSE:
        try:
            df = yf.download(
                ticker, start="2008-01-01", end=today,
                progress=False, auto_adjust=True,
            )
            if df is None or len(df) < 100:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df
        except Exception:
            continue

    if "SPY" not in data:
        raise RuntimeError("Failed to download SPY data")

    # Compute latest features for all tickers
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
            continue

    today_result, today_top5 = get_daily_pick(latest_features, cfg)
    market_ok = check_market(latest_features, cfg)

    # Build all stocks list
    all_stocks = []
    for ticker, feat in latest_features.items():
        if ticker in EXCLUDED:
            continue
        vals = [feat.get(k, np.nan) for k in [
            "ret_252d", "ret_126d", "ret_63d",
            "position_in_52w_range", "drawdown_252d", "vol_21d",
        ]]
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

    # Write output files to /tmp (Vercel writable directory)
    docs_dir = "/tmp/experiments_docs_data"
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
        "all_stocks": all_stocks,
        "qualifying": [s for s in all_stocks if s["qualifies"]],
    }

    with open(os.path.join(docs_dir, "full.json"), "w") as f:
        json.dump(clean_nan(full_data), f, indent=2)

    for stock in all_stocks[:30]:
        ticker = stock["ticker"]
        if ticker not in data:
            continue
        chart = [{"date": str(dt.date()), "price": round(float(row["Close"]), 2)}
                 for dt, row in data[ticker].tail(252).iterrows()]
        with open(os.path.join(docs_dir, "tickers", f"{ticker}.json"), "w") as f:
            json.dump(clean_nan({**stock, "chart": chart}), f, indent=2)

    # Push to GitHub
    pushed = push_directory(docs_dir, "experiments/docs/data", "chore: experiments daily scan update")
    return pushed, full_data.get("todays_pick", {}).get("ticker")


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            pushed, pick = run_experiments_scan()
            body = json.dumps({
                "ok": True,
                "pick": pick,
                "pushed": len(pushed),
                "files": pushed[:20],
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        except Exception as e:
            body = json.dumps({
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
