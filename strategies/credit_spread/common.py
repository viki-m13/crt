"""
Common utilities for the CreditFloor credit-spread research pipeline.

Data source: docs/data/tickers/*.json (daily closes going back ~11 years).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import numpy as np


TICKERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs",
    "data",
    "tickers",
)

# Horizons in trading days. 21 ~ 1 month, 42 ~ 2 months, 63 ~ 3 months,
# 126 ~ 6 months. Chosen to cover typical credit-spread tenors.
HORIZONS = [21, 42, 63, 126]

# Minimum training samples per (ticker, horizon) before a fold is usable.
# Keeps us from "learning" a 100% buffer on a handful of rows.
MIN_TRAIN_SAMPLES = 250

# Minimum total OOS test samples pooled across folds. If we have less than
# this, the 100% claim is too weak.
MIN_TEST_SAMPLES = 50

# Max strike buffer we'll ever recommend. A 25% OTM short strike is too far
# to be profitable. Realistically credit spreads use 1-10% buffers.
MAX_BUFFER = 0.25

# Conformal safety margin added to the worst historical drawdown-to-expiry.
# 1% absolute. Purely additive; does not scale with price.
SAFETY_EPS = 0.01

# Walk-forward fold boundaries (start of test year, UTC). Training is
# everything strictly before; testing is the year. We also enforce a
# purge gap equal to the horizon so train windows cannot overlap test dates.
FOLD_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

# Warmup: drop the first 252 days of each ticker to let long SMAs /
# features stabilize.
WARMUP_DAYS = 252


def expiry_date(last_trading_day: str | np.datetime64, horizon_days: int) -> str:
    """Project the expiration date N trading days forward from `last_trading_day`.

    Uses the NYSE calendar from `pandas_market_calendars`, which
    natively tracks weekends, federal holidays, exchange closures, and
    any special closures (e.g. 2012 Hurricane Sandy, 2018 Bush mourning
    day). `horizon_days` counts *actual NYSE sessions* after the last
    trading day — so if horizon=21 and the last session is 2026-04-23,
    the return is the 21st subsequent session date.

    Returns an ISO date string (YYYY-MM-DD).
    """
    import pandas as pd
    import pandas_market_calendars as mcal

    d = pd.Timestamp(str(last_trading_day)[:10])
    # Fetch the next N sessions strictly after `d`. We ask for a
    # generously wide window (up to ~2 calendar years) so this works
    # for any horizon up to ~500 sessions.
    start = (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (d + pd.Timedelta(days=int(horizon_days * 2) + 60)).strftime("%Y-%m-%d")
    nyse = mcal.get_calendar("NYSE")
    sessions = nyse.valid_days(start_date=start, end_date=end)
    if len(sessions) < horizon_days:
        # Extremely wide horizon; widen the window.
        end = (d + pd.Timedelta(days=int(horizon_days * 3) + 120)).strftime("%Y-%m-%d")
        sessions = nyse.valid_days(start_date=start, end_date=end)
    return sessions[horizon_days - 1].strftime("%Y-%m-%d")


def list_tickers() -> list[str]:
    return sorted(
        f[:-5]
        for f in os.listdir(TICKERS_DIR)
        if f.endswith(".json")
    )


@dataclass
class TickerSeries:
    ticker: str
    dates: np.ndarray  # datetime64[D]
    close: np.ndarray  # float64


def load_series(ticker: str) -> TickerSeries | None:
    path = os.path.join(TICKERS_DIR, f"{ticker}.json")
    try:
        with open(path, "r") as fh:
            blob = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    s = blob.get("series") or {}
    dates = s.get("dates") or []
    prices = s.get("prices") or []
    if len(dates) != len(prices) or len(dates) < WARMUP_DAYS + max(HORIZONS) + 100:
        return None
    d = np.array(dates, dtype="datetime64[D]")
    p = np.array(prices, dtype="float64")
    # Sanity: drop any non-positive prices.
    mask = (p > 0) & np.isfinite(p)
    if not mask.all():
        d = d[mask]
        p = p[mask]
    return TickerSeries(ticker=ticker, dates=d, close=p)


# -------------------------- feature engineering --------------------------
#
# All features below are computed at time t using ONLY close[0..t] (no
# look-ahead). They are used both as regime gates (train/test eligibility)
# and, later, for visualization.


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Simple moving average. out[t] = mean(x[t-n+1..t]). NaN for t < n-1."""
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
    # Unbiased not important; we just need a stable scale estimate.
    m = _rolling_mean(x, n)
    m2 = _rolling_mean(x * x, n)
    var = np.maximum(m2 - m * m, 0.0)
    out = np.sqrt(var)
    return out


def _rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype="float64")
    if len(x) < n:
        return out
    # O(n log n) via slice; dataset sizes are small so simple is fine.
    for i in range(n - 1, len(x)):
        out[i] = np.max(x[i - n + 1 : i + 1])
    return out


def _rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype="float64")
    if len(x) < n:
        return out
    for i in range(n - 1, len(x)):
        out[i] = np.min(x[i - n + 1 : i + 1])
    return out


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder's RSI. Causal; first n values are NaN."""
    out = np.full_like(close, np.nan, dtype="float64")
    d = np.diff(close)
    up = np.maximum(d, 0.0)
    dn = np.maximum(-d, 0.0)
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


@dataclass
class Features:
    trend: np.ndarray      # close/SMA200
    vol20: np.ndarray      # stdev of log returns, 20d, annualized-ish
    rsi14: np.ndarray      # RSI(14)
    dd252: np.ndarray      # 1 - close/max(close[t-252..t])
    mom_126: np.ndarray    # close[t]/close[t-126] - 1


def compute_features(close: np.ndarray) -> Features:
    sma200 = _rolling_mean(close, 200)
    trend = close / sma200
    logret = np.concatenate(([0.0], np.diff(np.log(close))))
    vol20 = _rolling_std(logret, 20)
    rsi14 = _rsi(close, 14)
    hi252 = _rolling_max(close, 252)
    dd252 = 1.0 - close / hi252
    mom_126 = np.full_like(close, np.nan)
    mom_126[126:] = close[126:] / close[:-126] - 1.0
    return Features(trend=trend, vol20=vol20, rsi14=rsi14, dd252=dd252, mom_126=mom_126)


# -------------------------- forward targets --------------------------
#
# For a horizon h, we care about TWO forward statistics at index t:
#   close_fwd(t, h)  = close[t + h]                   (expiration close)
#   min_fwd(t, h)    = min(close[t+1 .. t+h])        (path minimum)
#
# For a short credit spread short-strike K with expiry at t+h to be
# "100% safe" in the strictest American-exercise sense, we require
#   min_fwd(t, h) >= K   (path stays above the strike)
#
# For European-style / at-expiration matching the user's phrasing,
# we require
#   close_fwd(t, h) >= K
#
# We build both and let the research pipeline use the path-minimum as
# the primary (tighter) criterion.


def forward_min_close(close: np.ndarray, h: int) -> np.ndarray:
    """min(close[t+1..t+h]). NaN where the window runs off the end."""
    n = len(close)
    out = np.full(n, np.nan, dtype="float64")
    for t in range(0, n - h):
        out[t] = np.min(close[t + 1 : t + h + 1])
    return out


def forward_close(close: np.ndarray, h: int) -> np.ndarray:
    """close[t+h]. NaN where the window runs off the end."""
    n = len(close)
    out = np.full(n, np.nan, dtype="float64")
    for t in range(0, n - h):
        out[t] = close[t + h]
    return out


def worst_buffer_path(close: np.ndarray, h: int) -> np.ndarray:
    """Fractional buffer required so that min_fwd(t,h) >= close[t] * (1-buf).

        buf(t, h) = max(0, 1 - min_fwd(t,h)/close[t])

    In words: what fraction below today's price did the stock touch in
    the next h trading days? Always >= 0 (since if it never went below,
    buf = 0).
    """
    m = forward_min_close(close, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        buf = 1.0 - m / close
    buf = np.where(np.isnan(buf), np.nan, np.maximum(buf, 0.0))
    return buf


def worst_buffer_expiry(close: np.ndarray, h: int) -> np.ndarray:
    """Fractional buffer required so that close[t+h] >= close[t]*(1-buf)."""
    f = forward_close(close, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        buf = 1.0 - f / close
    buf = np.where(np.isnan(buf), np.nan, np.maximum(buf, 0.0))
    return buf


# -------------------------- regime gating --------------------------
#
# Proprietary regime filter. A date is in the "safe" regime iff:
#   - trend >= 1.00           (close above SMA200)
#   - dd252 <= 0.20           (within 20% of 52w high)
#   - vol20 present           (enough history)
#
# The research pipeline reports performance both with and without this
# filter and picks whichever yields a tighter 100% buffer.


def regime_mask(f: Features, require_uptrend: bool) -> np.ndarray:
    if not require_uptrend:
        return np.isfinite(f.trend) & np.isfinite(f.dd252) & np.isfinite(f.vol20)
    mask = (
        np.isfinite(f.trend)
        & np.isfinite(f.dd252)
        & np.isfinite(f.vol20)
        & (f.trend >= 1.00)
        & (f.dd252 <= 0.20)
    )
    return mask


def year_of(dt64: np.datetime64) -> int:
    return int(str(dt64)[:4])


def fold_mask(dates: np.ndarray, year: int) -> np.ndarray:
    """Return boolean mask of rows whose date falls within calendar year."""
    start = np.datetime64(f"{year}-01-01")
    end = np.datetime64(f"{year + 1}-01-01")
    return (dates >= start) & (dates < end)


def train_mask_for_fold(dates: np.ndarray, year: int, h: int) -> np.ndarray:
    """All rows t strictly before year, AND whose forward window
    (t+h) closed strictly before year. This purges any sample whose
    realized target overlaps the test window.
    """
    cutoff = np.datetime64(f"{year}-01-01")
    # Training sample is valid if dates[t] < cutoff AND the forward
    # window (t+h trading days ahead) is also < cutoff. Since dates is
    # the trading-day index, we approximate t+h by shifting the array.
    n = len(dates)
    end_dates = np.full(n, np.datetime64("9999-12-31"), dtype="datetime64[D]")
    end_dates[: n - h] = dates[h:]
    return (dates < cutoff) & (end_dates < cutoff)


def today_index(dates: np.ndarray) -> int:
    return len(dates) - 1
