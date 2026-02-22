#!/usr/bin/env python3
"""
Daily Stock Guide — on-demand analysis API server.

Provides a single endpoint:
    GET /api/analyze?ticker=AAPL

Runs the same full analysis pipeline as the daily scan for any valid
yfinance ticker and returns the detail + row JSON.

Usage:
    pip install flask flask-cors
    python scripts/api_server.py

The frontend calls this when a user searches for a ticker not in the
pre-computed dataset.
"""

import os
import sys
import json
import traceback

# Add parent dir so we can import daily_scan
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import pandas as pd
import yfinance as yf

import daily_scan as ds

app = Flask(__name__)
CORS(app)

# Cache SPY data across requests (it doesn't change within a session)
_cache = {}


def _get_market_data():
    """Download and cache SPY + market regime data."""
    if "spy" in _cache:
        return _cache["spy"], _cache["mkt"], _cache["ohlcv_spy"]

    print("[API] Downloading SPY market data...")
    spy_data = ds.download_ohlcv_period(
        [ds.BENCH], period=ds.PERIOD, interval=ds.INTERVAL, chunk_size=1
    )
    O_spy = spy_data["Open"]
    H_spy = spy_data["High"]
    L_spy = spy_data["Low"]
    C_spy = spy_data["Close"]
    V_spy = spy_data["Volume"]
    A_spy = spy_data["AdjClose"]

    PX_spy = A_spy if (not A_spy.empty and ds.BENCH in A_spy.columns) else C_spy

    spy_px = PX_spy[ds.BENCH].dropna()
    spy_h = H_spy[ds.BENCH].reindex(spy_px.index).dropna()
    spy_l = L_spy[ds.BENCH].reindex(spy_px.index).dropna()
    spy_px = spy_px.reindex(spy_h.index).reindex(spy_l.index).dropna()
    mkt = ds.compute_market_regime(spy_px, spy_h, spy_l)

    _cache["spy"] = spy_px
    _cache["mkt"] = mkt
    _cache["ohlcv_spy"] = spy_data
    print(f"[API] SPY data cached ({len(spy_px)} bars)")
    return spy_px, mkt, spy_data


def analyze_ticker(ticker: str):
    """Run the full analysis pipeline for a single ticker.

    Returns (row_dict, detail_dict) or raises on failure.
    """
    ticker = ticker.strip().upper().replace(".", "-")
    if not ticker:
        raise ValueError("Empty ticker")

    spy_px, mkt, _ = _get_market_data()

    # Download ticker data
    print(f"[API] Downloading data for {ticker}...")
    data = ds.download_ohlcv_period(
        [ticker, ds.BENCH], period=ds.PERIOD, interval=ds.INTERVAL, chunk_size=2
    )
    O, H, L, C, V, A = (
        data["Open"], data["High"], data["Low"],
        data["Close"], data["Volume"], data["AdjClose"],
    )

    if C.empty or ticker not in C.columns:
        raise ValueError(f"No price data found for '{ticker}'. Check the symbol is valid on Yahoo Finance.")

    PX = A if (not A.empty and ticker in A.columns) else C

    feature_cols = [
        "dd_lt", "pos_lt", "dd_st", "pos_st", "atr_pct", "volu_z", "gap", "trend_st",
        "idio_dd_lt", "idio_pos_lt", "idio_dd_st", "idio_pos_st",
        "mkt_trend", "mkt_vol", "mkt_dd", "mkt_atr_pct",
    ]
    zwin = max(63, ds.LB_ST)

    # Score — bypass the MIN_WASHOUT_TODAY gate for on-demand lookups
    original_gate = ds.MIN_WASHOUT_TODAY
    ds.MIN_WASHOUT_TODAY = 0
    # Also add ticker to ALWAYS_PLOT temporarily so it bypasses the gate
    original_always = list(ds.ALWAYS_PLOT)
    if ticker not in ds.ALWAYS_PLOT:
        ds.ALWAYS_PLOT.append(ticker)

    try:
        result = ds.score_one_ticker(
            ticker, O, H, L, C, V, PX, spy_px, mkt, feature_cols, zwin
        )
    finally:
        ds.MIN_WASHOUT_TODAY = original_gate
        ds.ALWAYS_PLOT = original_always

    if result is None:
        raise ValueError(
            f"Could not score '{ticker}'. The stock may not have enough price history "
            f"(need {ds.MIN_HISTORY_BARS}+ trading days) or sufficient trading volume."
        )

    row, detail = result
    return ds.sanitize_for_json(row), ds.sanitize_for_json(detail)


@app.route("/api/analyze", methods=["GET"])
def api_analyze():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "Missing 'ticker' parameter"}), 400

    try:
        row, detail = analyze_ticker(ticker)
        return jsonify({"row": row, "detail": detail})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"[API] Daily Stock Guide analysis server starting on port {port}")
    print(f"[API] Try: http://localhost:{port}/api/analyze?ticker=AAPL")
    app.run(host="0.0.0.0", port=port, debug=False)
