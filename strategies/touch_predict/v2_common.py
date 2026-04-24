"""v2 data layer + features for TouchPredictor (OHLCV-aware).

Loads from the touch_predict/data/ohlcv cache (populated by v2_backfill.py).
Computes Connors-style features: RSI(2), Bollinger, volume-Z, short-term
returns — plus the slower features (SMA200, RSI14, dd252, up252) reused
from credit_spread.common.

Everything below is CAUSAL (close[t] features use only close[0..t]).
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OHLCV_DIR = os.path.join(_HERE, "data", "ohlcv")

# Reuse the NYSE-calendar helpers + fold masks from the existing engine.
# Load common.py via a distinct module name to avoid name collisions.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_cs_common",
    os.path.join(os.path.dirname(_HERE), "credit_spread", "common.py"),
)
_cs = importlib.util.module_from_spec(_spec)  # type: ignore
sys.modules["_cs_common"] = _cs
_spec.loader.exec_module(_cs)  # type: ignore
FOLD_YEARS = _cs.FOLD_YEARS
WARMUP_DAYS = _cs.WARMUP_DAYS
MIN_TRAIN_SAMPLES = _cs.MIN_TRAIN_SAMPLES
MIN_TEST_SAMPLES = _cs.MIN_TEST_SAMPLES
fold_mask = _cs.fold_mask
train_mask_for_fold = _cs.train_mask_for_fold
forward_max_close = _cs.forward_max_close
forward_min_close = _cs.forward_min_close
actual_options_expiry = _cs.actual_options_expiry


def list_tickers() -> list[str]:
    return sorted(
        f[:-5] for f in os.listdir(OHLCV_DIR) if f.endswith(".json")
    )


@dataclass
class OhlcvSeries:
    ticker: str
    dates: np.ndarray    # datetime64[D]
    open:  np.ndarray
    high:  np.ndarray
    low:   np.ndarray
    close: np.ndarray
    volume: np.ndarray


def load_series(ticker: str) -> OhlcvSeries | None:
    p = os.path.join(OHLCV_DIR, f"{ticker}.json")
    try:
        with open(p, "r") as fh:
            blob = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    dates = blob.get("dates") or []
    close = blob.get("close") or []
    if len(dates) != len(close) or len(dates) < WARMUP_DAYS + 100:
        return None
    d = np.array(dates, dtype="datetime64[D]")
    o = np.array(blob.get("open") or close, dtype="float64")
    h = np.array(blob.get("high") or close, dtype="float64")
    l = np.array(blob.get("low") or close, dtype="float64")
    c = np.array(close, dtype="float64")
    v = np.array(blob.get("volume") or [0.0] * len(close), dtype="float64")
    mask = (c > 0) & np.isfinite(c)
    return OhlcvSeries(ticker, d[mask], o[mask], h[mask], l[mask], c[mask], v[mask])


# -------- rolling helpers (fast, vectorized) ---------------------------


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype="float64")
    if len(x) < n:
        return out
    csum = np.cumsum(x, dtype="float64")
    csum = np.concatenate(([0.0], csum))
    out[n - 1 :] = (csum[n:] - csum[:-n]) / n
    return out


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype="float64")
    if len(x) < n:
        return out
    m = _rolling_mean(x, n)
    m2 = _rolling_mean(x * x, n)
    var = np.maximum(m2 - m * m, 0.0)
    out = np.sqrt(var)
    return out


def _rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype="float64")
    if len(x) < n:
        return out
    view = np.lib.stride_tricks.sliding_window_view(x, window_shape=n)
    out[n - 1 :] = view.min(axis=1)
    return out


def _rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype="float64")
    if len(x) < n:
        return out
    view = np.lib.stride_tricks.sliding_window_view(x, window_shape=n)
    out[n - 1 :] = view.max(axis=1)
    return out


def _rsi(close: np.ndarray, n: int) -> np.ndarray:
    """Wilder's RSI with a `n`-period smoothing. Vectorized over the
    accumulation step with a simple Python loop for the recursion."""
    out = np.full_like(close, np.nan, dtype="float64")
    diff = np.diff(close)
    up = np.maximum(diff, 0.0)
    dn = np.maximum(-diff, 0.0)
    if len(close) <= n:
        return out
    avg_up = up[:n].mean()
    avg_dn = dn[:n].mean()
    for i in range(n, len(close)):
        if i > n:
            avg_up = (avg_up * (n - 1) + up[i - 1]) / n
            avg_dn = (avg_dn * (n - 1) + dn[i - 1]) / n
        if avg_dn == 0:
            out[i] = 100.0
        else:
            rs = avg_up / avg_dn
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


# -------- v2 features ---------------------------------------------------


@dataclass
class V2Features:
    # slow / trend features
    sma5:    np.ndarray
    sma20:   np.ndarray
    sma200:  np.ndarray
    trend:   np.ndarray          # close / sma200
    rsi2:    np.ndarray          # Connors 2-period RSI
    rsi14:   np.ndarray
    dd252:   np.ndarray          # 1 - close / max(close over 252d)
    up252:   np.ndarray          # close / min(close over 252d) - 1
    # Bollinger (20, 2σ)
    boll_mid:   np.ndarray
    boll_lower: np.ndarray
    boll_upper: np.ndarray
    # volume
    vol20:      np.ndarray       # 20-day avg volume
    vol50:      np.ndarray       # 50-day avg volume
    vol_z20:    np.ndarray       # today's vol / 20-day avg (ratio)
    vol_z50:    np.ndarray       # today's vol / 50-day avg (ratio)
    # short-term returns
    ret_1d:     np.ndarray
    ret_5d:     np.ndarray
    # range
    tr_pct:     np.ndarray       # (high-low)/close
    # realized volatility (for profit estimation)
    realized_vol: float          # annualized stdev of last 60-day log rets


def compute_features(s: OhlcvSeries) -> V2Features:
    close = s.close
    high = s.high
    low = s.low
    vol = s.volume

    sma5   = _rolling_mean(close, 5)
    sma20  = _rolling_mean(close, 20)
    sma200 = _rolling_mean(close, 200)
    trend  = close / sma200
    rsi2   = _rsi(close, 2)
    rsi14  = _rsi(close, 14)
    hi252  = _rolling_max(close, 252)
    lo252  = _rolling_min(close, 252)
    dd252  = 1.0 - close / hi252
    up252  = close / lo252 - 1.0
    boll_mid   = sma20
    boll_std   = _rolling_std(close, 20)
    boll_lower = boll_mid - 2.0 * boll_std
    boll_upper = boll_mid + 2.0 * boll_std
    vol20  = _rolling_mean(vol, 20)
    vol50  = _rolling_mean(vol, 50)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_z20 = vol / vol20
        vol_z50 = vol / vol50
    ret_1d = np.full_like(close, np.nan)
    ret_1d[1:] = close[1:] / close[:-1] - 1.0
    ret_5d = np.full_like(close, np.nan)
    ret_5d[5:] = close[5:] / close[:-5] - 1.0
    tr_pct = (high - low) / close

    # Realized vol: annualized stdev of last 60-day log returns.
    logret = np.diff(np.log(np.maximum(close, 1e-9)))
    if len(logret) >= 60:
        rv = float(np.std(logret[-60:], ddof=1)) * math.sqrt(252.0)
    else:
        rv = float("nan")

    return V2Features(
        sma5=sma5, sma20=sma20, sma200=sma200, trend=trend,
        rsi2=rsi2, rsi14=rsi14, dd252=dd252, up252=up252,
        boll_mid=boll_mid, boll_lower=boll_lower, boll_upper=boll_upper,
        vol20=vol20, vol50=vol50, vol_z20=vol_z20, vol_z50=vol_z50,
        ret_1d=ret_1d, ret_5d=ret_5d, tr_pct=tr_pct,
        realized_vol=rv,
    )


# -------- forward touch buffers (reused) -------------------------------


def buffer_up(close: np.ndarray, h: int) -> np.ndarray:
    """UP touch buffer: max(close[t+1..t+h])/close[t] - 1."""
    m = forward_max_close(close, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = m / close - 1.0
    return np.where(np.isnan(b), np.nan, b)


def buffer_down(close: np.ndarray, h: int) -> np.ndarray:
    """DOWN touch buffer: 1 - min(close[t+1..t+h])/close[t]."""
    m = forward_min_close(close, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = 1.0 - m / close
    return np.where(np.isnan(b), np.nan, b)


# -------- market-context features (SPY) -------------------------------
#
# Helper to load SPY-aligned context features (5d return, rsi2) so rules
# can condition on market state. SPY is in the liquid universe.


_SPY_CACHE: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None


def spy_context():
    """Return (dates, spy_close, spy_ret_5d) arrays. Cached."""
    global _SPY_CACHE
    if _SPY_CACHE is not None:
        return _SPY_CACHE
    s = load_series("SPY")
    if s is None:
        return None
    ret_5d = np.full_like(s.close, np.nan)
    ret_5d[5:] = s.close[5:] / s.close[:-5] - 1.0
    _SPY_CACHE = (s.dates, s.close, ret_5d)
    return _SPY_CACHE


def align_to_spy(stock_dates: np.ndarray,
                 spy_dates: np.ndarray,
                 spy_ret_5d: np.ndarray) -> np.ndarray:
    """For each stock_date, return SPY's 5d-return on that date (or NaN
    if date missing). Uses np.searchsorted for O(n log m) lookup."""
    idx = np.searchsorted(spy_dates, stock_dates)
    idx = np.clip(idx, 0, len(spy_dates) - 1)
    out = np.full_like(stock_dates, fill_value=np.nan, dtype="float64")
    match = spy_dates[idx] == stock_dates
    out[match] = spy_ret_5d[idx[match]]
    return out
