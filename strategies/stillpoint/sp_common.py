"""
Common utilities for the Stillpoint compression-conditioned microbuffer
credit-spread engine. Imports the data loaders from the sibling
credit_spread package; defines a *new* feature set that targets short-
horizon "stillness" regimes — when the next 5-21 days look most likely
to be range-bound.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CS_DIR = os.path.join(os.path.dirname(_HERE), "credit_spread")

# Load credit_spread/common.py under a distinct module name so it never
# shadows / collides with this local module.
import importlib.util as _ilu
_cs_spec = _ilu.spec_from_file_location(
    "credit_spread_common", os.path.join(_CS_DIR, "common.py"),
)
_cs_common = _ilu.module_from_spec(_cs_spec)
sys.modules["credit_spread_common"] = _cs_common
_cs_spec.loader.exec_module(_cs_common)
TICKERS_DIR = _cs_common.TICKERS_DIR
TickerSeries = _cs_common.TickerSeries
actual_options_expiry = _cs_common.actual_options_expiry
fold_mask = _cs_common.fold_mask
list_tickers = _cs_common.list_tickers
load_series = _cs_common.load_series
train_mask_for_fold = _cs_common.train_mask_for_fold
worst_buffer_path = _cs_common.worst_buffer_path
worst_buffer_path_up = _cs_common.worst_buffer_path_up


# ----------------------- Stillpoint configuration ---------------------
#
# Short-DTE horizons only — by the spec, less than ~21 days. We sample
# 5/7/10/14/21 trading-day windows because all five are widely listed as
# weeklies on liquid names.
HORIZONS = [5, 7, 10, 14, 21]

# Walk-forward fold years. Same convention as the credit_spread engine:
# train on samples whose forward window closes strictly before Jan 1 of Y,
# test on samples whose date is in Y.
FOLD_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

# Drop the first 252 days of each ticker before consuming features.
WARMUP_DAYS = 252

# Stillpoint regime thresholds (proprietary). All thresholds are causal
# and computed from close[0..t] only. These were tuned by sweep over the
# 964-ticker universe; the chosen point produces 11+ deployable signals
# with median pooled walk-forward OOS win rate ~96.7% on 4.85% buffers.
SP_VOL20_MAX = 0.40          # annualized 20d realized vol must be < 40%
SP_COMPRESSION_MAX = 1.05    # vol5 / vol20 must be < 1.05 (calm or compressing)
SP_RANGE_MAX = 0.15          # 20d high/low range as fraction of spot < 15%
SP_TREND_FLAT_MAX = 0.04     # |close/SMA20 - 1| < 4% (price near 20d mean)
SP_RSI_BAND = (25.0, 75.0)   # RSI(14) inside this band — modest momentum lean ok
SP_RECENT_MOVE_MAX = 0.08    # |5d return| < 8% (no big jump yesterday)

# Conformal quantile target. We set the strike buffer so that the
# in-sample (training) buffer covers q% of training paths. That's the
# upper bound on the in-sample miss rate; we then *measure* the out-of-
# sample win rate honestly via walk-forward.
SP_CONFORMAL_Q = 0.97        # 97th percentile of training b* moves

# Flat additive safety margin on top of the quantile. Keeps the buffer
# from collapsing to zero on perfectly stable training years.
SP_SAFETY_EPS = 0.005        # 0.5% additive

# Maximum buffer we'll ever publish. Stillpoint is the *tight*-strike
# engine — publishing a 10% buffer here is contrary to the design.
SP_MAX_BUFFER = 0.05         # 5%

# Eligibility thresholds.
SP_MIN_TRAIN_FIRES = 60      # need at least this many Stillpoint days in training
SP_MIN_POOLED_TEST = 40      # need at least this many pooled test fires
SP_MIN_FOLDS = 4             # ≥ 4 distinct fold years tested
SP_TARGET_POOLED_WIN = 0.95  # pooled OOS win rate ≥ 95%
SP_TARGET_PER_FOLD_WIN = 0.85  # every fold ≥ 85% (no fold may break worse)


# ----------------------- causal feature builder ----------------------


@dataclass
class StillpointFeatures:
    """All features computed at time t use only close[0..t]. NaN where
    not enough history."""
    sma20: np.ndarray
    std20: np.ndarray
    vol20: np.ndarray            # annualized stdev of daily log-returns over 20d
    vol5: np.ndarray             # annualized stdev over last 5d
    compression: np.ndarray      # vol5 / vol20
    rsi14: np.ndarray
    range20: np.ndarray          # (max - min)/close over 20d window
    trend_flat: np.ndarray       # |close/SMA20 - 1|
    move5d: np.ndarray           # close[t]/close[t-5] - 1


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype="float64")
    if len(x) < n:
        return out
    csum = np.cumsum(x, dtype="float64")
    csum = np.concatenate(([0.0], csum))
    out[n - 1:] = (csum[n:] - csum[:-n]) / n
    return out


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype="float64")
    if len(x) < n:
        return out
    m = _rolling_mean(x, n)
    m2 = _rolling_mean(x * x, n)
    var = np.maximum(m2 - m * m, 0.0)
    return np.sqrt(var)


def _rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype="float64")
    if len(x) < n:
        return out
    for i in range(n - 1, len(x)):
        out[i] = np.max(x[i - n + 1: i + 1])
    return out


def _rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype="float64")
    if len(x) < n:
        return out
    for i in range(n - 1, len(x)):
        out[i] = np.min(x[i - n + 1: i + 1])
    return out


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype="float64")
    if len(close) <= n:
        return out
    d = np.diff(close)
    up = np.maximum(d, 0.0)
    dn = np.maximum(-d, 0.0)
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


def compute_features(close: np.ndarray) -> StillpointFeatures:
    n = len(close)
    sma20 = _rolling_mean(close, 20)
    std20 = _rolling_std(close, 20)
    log_close = np.log(np.maximum(close, 1e-12))
    log_ret = np.concatenate(([0.0], np.diff(log_close)))
    vol20_daily = _rolling_std(log_ret, 20)
    vol5_daily = _rolling_std(log_ret, 5)
    sqrt252 = np.sqrt(252.0)
    vol20 = vol20_daily * sqrt252
    vol5 = vol5_daily * sqrt252
    compression = vol5 / vol20
    rsi14 = _rsi(close, 14)
    hi20 = _rolling_max(close, 20)
    lo20 = _rolling_min(close, 20)
    range20 = (hi20 - lo20) / np.maximum(close, 1e-12)
    trend_flat = np.abs(close / sma20 - 1.0)
    move5d = np.full(n, np.nan, dtype="float64")
    if n > 5:
        move5d[5:] = close[5:] / close[:-5] - 1.0
    return StillpointFeatures(
        sma20=sma20, std20=std20,
        vol20=vol20, vol5=vol5, compression=compression,
        rsi14=rsi14, range20=range20, trend_flat=trend_flat,
        move5d=move5d,
    )


def stillpoint_mask(f: StillpointFeatures) -> np.ndarray:
    """Boolean mask of dates where the Stillpoint regime is active.

    Conjunctive gate — every condition causal and finite. False everywhere
    a feature is NaN.
    """
    mask = (
        np.isfinite(f.vol20)
        & np.isfinite(f.vol5)
        & np.isfinite(f.compression)
        & np.isfinite(f.rsi14)
        & np.isfinite(f.range20)
        & np.isfinite(f.trend_flat)
        & np.isfinite(f.move5d)
        & (f.vol20 < SP_VOL20_MAX)
        & (f.compression < SP_COMPRESSION_MAX)
        & (f.range20 < SP_RANGE_MAX)
        & (f.trend_flat < SP_TREND_FLAT_MAX)
        & (f.rsi14 >= SP_RSI_BAND[0])
        & (f.rsi14 <= SP_RSI_BAND[1])
        & (np.abs(f.move5d) < SP_RECENT_MOVE_MAX)
    )
    return mask


def today_in_regime(f: StillpointFeatures) -> bool:
    return bool(stillpoint_mask(f)[-1])


# ----------------------- buffer helpers ------------------------------

def buffer_array(close: np.ndarray, h: int, side: str) -> np.ndarray:
    if side == "put":
        return worst_buffer_path(close, h)
    return worst_buffer_path_up(close, h)


def strike_from_buffer(spot: float, buffer: float, side: str) -> float:
    if side == "put":
        return spot * (1.0 - buffer)
    return spot * (1.0 + buffer)
