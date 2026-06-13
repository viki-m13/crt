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


# Price-series directory. Default: the main site's ticker JSONs
# (2015+, append-maintained). Override with CS_DATA_DIR to run the
# engine/replay on an alternate panel (e.g. the full-history cache
# written by fetch_full_history.py for deep replay validation).
TICKERS_DIR = os.environ.get("CS_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs",
    "data",
    "tickers",
)

# Horizons in trading days. 7 ~ 1 week, 10 ~ bi-weekly, 14 ~ 2 weeks,
# 21 ~ 1 month, 42 ~ 2 months, 63 ~ 3 months, 126 ~ 6 months, 252 ~ 1
# year. The short end lets higher-vol names (AAPL, NVDA, DIS, BA, etc.)
# qualify — in shorter windows their worst historical move is much
# smaller than in 21d+ windows, so they can often fit inside the 25%
# buffer cap. Long horizons rarely clear the v3 publication gates but
# are evaluated so any duration can qualify when conditions allow.
HORIZONS = [7, 10, 14, 21, 42, 63, 126, 252]

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
# purge gap equal to the horizon so train windows cannot overlap test
# dates. Folds run from 2006 (the engine runs on the full-history panel
# and per-ticker MIN_TRAIN_SAMPLES auto-skips folds a young ticker
# can't support); this matches the deep-replay validation configuration
# exactly (see VALIDATION.md).
FOLD_YEARS = list(range(2006, 2027))

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


import functools


@functools.lru_cache(maxsize=1)
def _nyse_valid_days_big():
    """Cache a big window of NYSE trading days once per process (covers
    1980-01-01 through 2028-12-31 so deep-history replays resolve
    correct expiries). Individual call sites then filter in-memory
    instead of rebuilding holidays on every call.
    """
    import pandas as pd
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    sessions = nyse.valid_days(start_date="1980-01-01", end_date="2028-12-31")
    # Strip tz → naive Timestamps (cheaper comparisons downstream)
    return pd.DatetimeIndex([pd.Timestamp(s.tz_localize(None) if s.tzinfo else s)
                             for s in sessions])


@functools.lru_cache(maxsize=4096)
def actual_options_expiry(
    last_trading_day: str | np.datetime64,
    horizon_sessions: int,
) -> tuple[str, str, int]:
    """Snap a session-count horizon to the nearest STANDARD options expiry
    on the NYSE calendar.

    Rules:
      - Horizons <= 20 sessions  → weekly expiry: next Friday (that's a
        trading day) on-or-after the session-projected date.
      - Horizons >= 21 sessions  → monthly expiry: the 3rd-Friday-of-a-
        month trading day on-or-after the session-projected date.
      - If the canonical Friday is a market holiday (rare — Good Friday
        is the only consistent case), fall back to the trading day
        immediately before it (which is how the exchange actually
        handles Good Friday expiry — settles on Thursday).

    Returns (expiry_iso, kind, calendar_days_to_expiry) where:
      - expiry_iso: 'YYYY-MM-DD' of the real Friday options expiry
      - kind: 'weekly' | 'monthly'
      - calendar_days_to_expiry: integer calendar days from
        last_trading_day to expiry (used for BS time-to-expiry).

    NOTE: the walk-forward backtest uses session-count horizons; the
    actual options-expiry displayed here can be 0-4 sessions later
    than the session projection. This small slop is absorbed by the
    1% conformal safety margin.
    """
    import pandas as pd

    sessions = _nyse_valid_days_big()
    d = pd.Timestamp(str(last_trading_day)[:10])
    # Find session index of `d` (or the session on-or-after it)
    idx = int(sessions.searchsorted(d, side="left"))
    # Session-projected target: horizon_sessions forward from `d`
    # (counting d as session 0)
    tgt_idx = min(idx + horizon_sessions, len(sessions) - 1)
    target = sessions[tgt_idx]

    if horizon_sessions <= 20:
        kind = "weekly"
        # First trading-day Friday on-or-after target. Walk forward in
        # the cached sessions index.
        exp = None
        for j in range(tgt_idx, min(tgt_idx + 40, len(sessions))):
            s = sessions[j]
            if s.weekday() == 4:  # Friday
                exp = s
                break
        if exp is None:
            exp = target
    else:
        kind = "monthly"
        # 3rd-Friday (trading day) on-or-after target. Generate candidate
        # 3rd Fridays in upcoming months; match against the cached
        # sessions to detect Good-Friday displacement.
        exp = None
        y, m = target.year, target.month
        for _ in range(8):
            all_fridays = pd.date_range(f"{y}-{m:02d}-01",
                                         (pd.Timestamp(f"{y}-{m:02d}-01") + pd.offsets.MonthEnd(1)),
                                         freq="W-FRI")
            if len(all_fridays) >= 3:
                third_friday_cal = all_fridays[2]
                # Binary-search the cached session index
                sidx = sessions.searchsorted(third_friday_cal)
                if sidx < len(sessions) and sessions[sidx] == third_friday_cal:
                    cand = third_friday_cal
                else:
                    # Good Friday: expiry rolls to Thursday before
                    cand = sessions[max(sidx - 1, 0)]
                if cand >= target:
                    exp = cand
                    break
            m += 1
            if m > 12:
                m = 1
                y += 1
        if exp is None:
            exp = target

    exp_naive = pd.Timestamp(exp)
    cal_days = (exp_naive.normalize() - d.normalize()).days
    return exp_naive.strftime("%Y-%m-%d"), kind, int(cal_days)


@functools.lru_cache(maxsize=4096)
def covered_options_expiry(
    last_trading_day: str | np.datetime64,
    horizon_sessions: int,
) -> tuple[str, str, int, int] | None:
    """Snap a session-count horizon DOWN to the latest STANDARD options
    expiry that the certified window still covers.

    The legacy ``actual_options_expiry`` snapped *up* (first expiry
    on-or-after the session-projected date), which meant the engine
    certified an h-session window but sold an expiry up to ~18 sessions
    further out — e.g. an h=21 signal published 2020-01-23 was assigned
    the 2020-03-20 expiry and rode straight into the COVID crash that
    its 21-session validation never covered. Snapping down closes that
    hole: the expiry returned here is always at most ``horizon_sessions``
    NYSE sessions after the publish date, so the conformal buffer's
    guarantee window is a superset of the actual trade window.

    Rules:
      - Horizons <= 20 sessions → latest weekly (Friday) expiry within
        the window (publish, target].
      - Horizons >= 21 sessions → latest monthly (3rd-Friday) expiry
        within the window; if a monthly doesn't fit, fall back to the
        latest weekly Friday.
      - Good-Friday style holidays: expiry rolls to the trading day
        before the canonical Friday (matching exchange practice).

    Returns (expiry_iso, kind, calendar_days_to_expiry,
    sessions_to_expiry), or None if no standard expiry exists at all in
    the window (can only happen for very short horizons over exchange
    holidays — callers should skip the rung).
    """
    import pandas as pd

    sessions = _nyse_valid_days_big()
    d = pd.Timestamp(str(last_trading_day)[:10])
    idx = int(sessions.searchsorted(d, side="left"))
    tgt_idx = min(idx + horizon_sessions, len(sessions) - 1)

    def _friday_roll_slot(ts: pd.Timestamp) -> pd.Timestamp | None:
        """If ts stands in for a holiday Friday (ts is the last session
        strictly before a non-trading calendar Friday in the same week),
        return that Friday; else None."""
        nxt = ts + pd.Timedelta(days=1)
        while nxt.weekday() != 4:
            nxt += pd.Timedelta(days=1)
        if (nxt - ts).days > 6:
            return None
        sidx = sessions.searchsorted(nxt)
        is_trading = sidx < len(sessions) and sessions[sidx] == nxt
        if is_trading:
            return None
        prev_sess = sessions[max(sessions.searchsorted(nxt) - 1, 0)]
        return nxt if prev_sess == ts else None

    def _is_weekly_slot(ts: pd.Timestamp) -> bool:
        # Regular Friday, or the session standing in for a holiday Friday
        # (e.g. Maundy Thursday before Good Friday).
        return ts.weekday() == 4 or _friday_roll_slot(ts) is not None

    def _is_third_friday_slot(ts: pd.Timestamp) -> bool:
        # ts is a trading day; it is a "monthly expiry" if it's the 3rd
        # Friday of its month, OR the trading day standing in for a 3rd
        # Friday that is a holiday (Good Friday case).
        if ts.weekday() == 4 and 15 <= ts.day <= 21:
            return True
        rolled = _friday_roll_slot(ts)
        return rolled is not None and 15 <= rolled.day <= 21 and rolled.month == ts.month

    best = None
    for j in range(tgt_idx, idx, -1):  # latest first, strictly after publish day
        s = sessions[j]
        if horizon_sessions <= 20:
            if _is_weekly_slot(s):
                best = (j, s, "weekly")
                break
        else:
            if _is_third_friday_slot(s):
                best = (j, s, "monthly")
                break
    if best is None and horizon_sessions >= 21:
        for j in range(tgt_idx, idx, -1):
            s = sessions[j]
            if s.weekday() == 4:
                best = (j, s, "weekly")
                break
    if best is None:
        return None
    j, exp, kind = best
    exp_naive = pd.Timestamp(exp)
    cal_days = (exp_naive.normalize() - d.normalize()).days
    sessions_to_exp = j - idx
    if sessions_to_exp <= 0 or cal_days <= 0:
        return None
    return exp_naive.strftime("%Y-%m-%d"), kind, int(cal_days), int(sessions_to_exp)


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
    dd252: np.ndarray      # 1 - close/max(close[t-252..t])   (down from 252d high)
    up252: np.ndarray      # close/min(close[t-252..t]) - 1   (up from 252d low)
    mom_126: np.ndarray    # close[t]/close[t-126] - 1
    mom_252: np.ndarray    # close[t]/close[t-252] - 1


def compute_features(close: np.ndarray) -> Features:
    sma200 = _rolling_mean(close, 200)
    trend = close / sma200
    logret = np.concatenate(([0.0], np.diff(np.log(close))))
    vol20 = _rolling_std(logret, 20)
    rsi14 = _rsi(close, 14)
    hi252 = _rolling_max(close, 252)
    lo252 = _rolling_min(close, 252)
    dd252 = 1.0 - close / hi252
    up252 = close / lo252 - 1.0
    mom_126 = np.full_like(close, np.nan)
    mom_126[126:] = close[126:] / close[:-126] - 1.0
    mom_252 = np.full_like(close, np.nan)
    mom_252[252:] = close[252:] / close[:-252] - 1.0
    return Features(
        trend=trend, vol20=vol20, rsi14=rsi14,
        dd252=dd252, up252=up252,
        mom_126=mom_126, mom_252=mom_252,
    )


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
    """min(close[t+1..t+h]). NaN where the window runs off the end.
    Vectorized via a sliding-window trick: rolling min over the NEXT h
    values. ~100× faster than the per-t Python loop."""
    n = len(close)
    out = np.full(n, np.nan, dtype="float64")
    if n <= h:
        return out
    # Build a (n-h, h) view of close[t+1..t+h] via stride tricks, then
    # reduce along axis=1. np.lib.stride_tricks.sliding_window_view
    # is O(1) for the view and O((n-h)*h) for the min reduction.
    tail = close[1:]                                 # length n-1
    view = np.lib.stride_tricks.sliding_window_view(tail, window_shape=h)
    # view shape: (n-h, h), where view[t] = close[t+1..t+h]
    out[: n - h] = view.min(axis=1)
    return out


def forward_max_close(close: np.ndarray, h: int) -> np.ndarray:
    """max(close[t+1..t+h]). NaN where the window runs off the end.
    Vectorized. See forward_min_close."""
    n = len(close)
    out = np.full(n, np.nan, dtype="float64")
    if n <= h:
        return out
    tail = close[1:]
    view = np.lib.stride_tricks.sliding_window_view(tail, window_shape=h)
    out[: n - h] = view.max(axis=1)
    return out


def forward_close(close: np.ndarray, h: int) -> np.ndarray:
    """close[t+h]. NaN where the window runs off the end."""
    n = len(close)
    out = np.full(n, np.nan, dtype="float64")
    for t in range(0, n - h):
        out[t] = close[t + h]
    return out


def worst_buffer_path(close: np.ndarray, h: int) -> np.ndarray:
    """PUT side — how far *below* today's price the stock went.

    Fractional buffer required so that min_fwd(t,h) >= close[t] * (1-buf):

        buf_down(t, h) = max(0, 1 - min_fwd(t,h)/close[t])

    Used when selling put credit spreads (short strike below spot).
    """
    m = forward_min_close(close, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        buf = 1.0 - m / close
    buf = np.where(np.isnan(buf), np.nan, np.maximum(buf, 0.0))
    return buf


def worst_buffer_path_up(close: np.ndarray, h: int) -> np.ndarray:
    """CALL side — how far *above* today's price the stock went.

    Fractional buffer required so that max_fwd(t,h) <= close[t] * (1+buf):

        buf_up(t, h) = max(0, max_fwd(t,h)/close[t] - 1)

    Used when selling call credit spreads (short strike above spot).
    Symmetric to worst_buffer_path: every construction, purge, and
    walk-forward fold semantic is identical, just mirrored along the
    price axis.
    """
    m = forward_max_close(close, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        buf = m / close - 1.0
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
    """PUT-side regime: uptrend + not deeply drawn-down."""
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


def regime_mask_call(f: Features, require_bearish: bool) -> np.ndarray:
    """CALL-side regime: below-SMA200 + hasn't rallied hard off 1-year low.

    Symmetric to the put gate:
        put gate  = (close >= SMA200) & (<=20% off 52w high)
        call gate = (close <= SMA200) & (<=20% up from 52w low)

    Intuition: for a short call strike to stay safe, the stock needs to
    not rally sharply. Stocks below their 200-SMA that haven't already
    started a recovery are the lowest-rally-probability population.
    """
    if not require_bearish:
        return np.isfinite(f.trend) & np.isfinite(f.up252) & np.isfinite(f.vol20)
    mask = (
        np.isfinite(f.trend)
        & np.isfinite(f.up252)
        & np.isfinite(f.vol20)
        & (f.trend <= 1.00)
        & (f.up252 <= 0.20)
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
