"""
Vercel serverless function — on-demand stock analysis.

Endpoint:  GET /api/analyze?ticker=AAPL

Runs the same full analysis pipeline as the daily scan for any valid
yfinance ticker and returns the detail + row JSON.
"""

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Make scripts/ importable
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

import numpy as np
import pandas as pd

import daily_scan as ds

# ---------------------------------------------------------------------------
# Module-level cache — persists across warm Vercel invocations
# ---------------------------------------------------------------------------
_spy_cache = {}


def _get_market_data():
    """Download and cache SPY + market regime data."""
    if "spy_px" in _spy_cache:
        return (
            _spy_cache["spy_px"],
            _spy_cache["mkt"],
            _spy_cache["O"],
            _spy_cache["H"],
            _spy_cache["L"],
            _spy_cache["C"],
            _spy_cache["V"],
            _spy_cache["PX"],
        )

    data = ds.download_ohlcv_period(
        [ds.BENCH], period="10y", interval=ds.INTERVAL, chunk_size=1
    )
    O, H, L, C, V, A = (
        data["Open"], data["High"], data["Low"],
        data["Close"], data["Volume"], data["AdjClose"],
    )
    PX = A if (not A.empty and ds.BENCH in A.columns) else C

    spy_px = PX[ds.BENCH].dropna()
    spy_h = H[ds.BENCH].reindex(spy_px.index).dropna()
    spy_l = L[ds.BENCH].reindex(spy_px.index).dropna()
    spy_px = spy_px.reindex(spy_h.index).reindex(spy_l.index).dropna()
    mkt = ds.compute_market_regime(spy_px, spy_h, spy_l)

    _spy_cache.update(
        spy_px=spy_px, mkt=mkt, O=O, H=H, L=L, C=C, V=V, PX=PX
    )
    return spy_px, mkt, O, H, L, C, V, PX


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def analyze_ticker(ticker: str):
    """Run full analysis pipeline for a single ticker. Returns (row, detail)."""
    ticker = ticker.strip().upper().replace(".", "-")
    if not ticker:
        raise ValueError("Empty ticker")

    spy_px, mkt, spy_O, spy_H, spy_L, spy_C, spy_V, spy_PX = _get_market_data()

    # Download ticker data (+ SPY again so DataFrames share the index)
    data = ds.download_ohlcv_period(
        [ticker, ds.BENCH], period="10y", interval=ds.INTERVAL, chunk_size=2
    )
    O, H, L, C, V, A = (
        data["Open"], data["High"], data["Low"],
        data["Close"], data["Volume"], data["AdjClose"],
    )

    if C.empty or ticker not in C.columns:
        raise ValueError(
            f"No price data found for '{ticker}'. "
            "Check the symbol is valid on Yahoo Finance."
        )

    PX = A if (not A.empty and ticker in A.columns) else C

    feature_cols = [
        "dd_lt", "pos_lt", "dd_st", "pos_st",
        "atr_pct", "volu_z", "gap", "trend_st",
        "idio_dd_lt", "idio_pos_lt", "idio_dd_st", "idio_pos_st",
        "mkt_trend", "mkt_vol", "mkt_dd", "mkt_atr_pct",
    ]
    zwin = max(63, ds.LB_ST)

    # Bypass the washout gate for on-demand lookups
    original_gate = ds.MIN_WASHOUT_TODAY
    original_always = list(ds.ALWAYS_PLOT)
    ds.MIN_WASHOUT_TODAY = 0
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
            f"Could not score '{ticker}'. The stock may not have enough "
            f"price history (need {ds.MIN_HISTORY_BARS}+ trading days) "
            "or sufficient trading volume."
        )

    row, detail = result
    return ds.sanitize_for_json(row), ds.sanitize_for_json(detail)


# ---------------------------------------------------------------------------
# Vercel handler
# ---------------------------------------------------------------------------
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        ticker = query.get("ticker", [""])[0].strip().upper()

        if not ticker:
            self._respond(400, {"error": "Missing 'ticker' parameter"})
            return

        try:
            row, detail = analyze_ticker(ticker)
            self._respond(200, {"row": row, "detail": detail})
        except ValueError as e:
            self._respond(404, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._respond(500, {"error": f"Analysis failed: {str(e)}"})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # -- helpers --

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _respond(self, code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)
