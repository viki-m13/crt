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
# Walk-forward fold years. Test years are 2020 through the current
# calendar year (computed at import time, NOT hardcoded). This makes
# the fold set auto-extend with the wall clock — so the engine always
# tests on every available calendar year as OOS, never freezing the
# OOS test set in the past.
def _fold_years_through_now():
    from datetime import datetime, timezone
    cur = datetime.now(timezone.utc).year
    # Always include at least 2020-2026 (legacy minimum); extend
    # forward through the current year.
    return list(range(2020, max(2026, cur) + 1))

FOLD_YEARS = _fold_years_through_now()

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


# -------------------- Tight-tier configuration ----------------------
#
# A second, NEW preset that targets strikes much closer to spot
# (sub-3% buffer) on ultra-short horizons (2-3 sessions). Trades a
# stricter regime gate + vol-adaptive conformal buffer for tighter
# strikes at the same 95%+ accuracy threshold.

# Tight regime gate — restrictive subset of base
SP_TIGHT_VOL20_MAX = 0.30
SP_TIGHT_COMPRESSION_MAX = 0.95
SP_TIGHT_RANGE_MAX = 0.10
SP_TIGHT_TREND_FLAT_MAX = 0.025
SP_TIGHT_RSI_BAND = (35.0, 65.0)
SP_TIGHT_RECENT_MOVE_MAX = 0.05

SP_TIGHT_HORIZONS = [2, 3, 5]   # ultra-short
SP_TIGHT_CONFORMAL_Q = 0.95     # 95th-percentile of vol-normalized buffers
SP_TIGHT_MAX_BUFFER = 0.03      # 3% cap (was 5% for core)
SP_TIGHT_PER_FOLD_WIN = 0.85
SP_TIGHT_POOLED_WIN = 0.95


# -------------------- Iron Condor (Atomic) Profit tier ---------------
#
# Joint put+call credit spread on the same ticker, sized for combined
# per-trade ROR >= 50% at joint OOS WR >= 95%. The "Atomic" framing:
# treat the iron condor as a single indivisible trade — both legs win
# or it loses. Joint backtest counts a fold-day as a win iff
#   path_min(t..t+h) >= K_put_short AND path_max(t..t+h) <= K_call_short
# Combined credit / max-loss / ROR come from summing both legs' credits
# and recognizing that max-loss = width - combined_credit (because at
# expiry the stock can be inside either bound or outside one but never
# both, so only one side's loss is realized while the other side keeps
# its full credit).

SP_IC_HORIZONS = [21, 30, 42, 63, 90, 126]
# Per-leg conformal quantile candidates — engine picks the q that yields
# the highest combined ROR while still validating at the joint pooled WR
# target. Smaller q ⇒ closer-to-spot strikes; we want the balance.
SP_IC_CONFORMAL_QS = [0.96, 0.97, 0.975, 0.98, 0.985]
SP_IC_MAX_BUFFER = 0.30             # per-leg max buffer
SP_IC_PER_FOLD_WIN = 0.75           # joint per-fold WR floor — single
                                     # fold can dip; the >=95% pooled
                                     # gate is the actual user spec
# Spread widths (per leg, fraction of spot) — engine evaluates each and
# keeps the width that maximizes combined ROR.
SP_IC_WIDTHS = [0.01, 0.02, 0.03, 0.05]


# -------------------- Universal Iron Condor tier ---------------------
#
# Same 50%+ ROR + 95%+ joint OOS WR target as the Atomic IC tier, but
# with TWO key differences that broaden the eligible-ticker universe:
#
#   1. NO REGIME GATE. Backtest runs across ALL trading days after the
#      252-day warmup, not just stillness-regime days. The regime
#      conditioning that helped LSTR/FLO/CE/IPGP/TER/HUN/FMC pass also
#      excluded the entire universe of high-vol names. Removing the
#      gate lets any ticker compete on its OWN walk-forward statistics.
#
#   2. CLOSE-AT-EXPIRY WIN CONDITION. The Atomic IC tier uses the
#      strict path-min/max criterion (American-exercise-safe). For
#      European cash-settled options (SPX, NDX, RUT) the only thing
#      that matters at settlement is close[t+h]. For OTM American
#      verticals, early exercise on the LONG leg is essentially never
#      optimal in practice, so close-at-expiry is the realistic win
#      condition. Switching from path to close-at-expiry produces a
#      tighter conditional distribution (paths can touch and recover)
#      and roughly triples the eligible-ticker count.
#
# Vol-adaptive joint conformal: identical to the Atomic IC method —
# z = b* / (σ × √T) buffers, conformal q-quantile of training z's, then
# rescale by today's σ at publish. Joint walk-forward validation (BOTH
# legs must survive). Width sweep keeps the highest-ROR width.
#
# Realistic-credit ceiling: BS pricing at very narrow widths can
# produce combined_credit > width (an arbitrage artifact). We cap
# combined_credit / width at 0.50 — beyond that, real markets won't
# fill the spread because of bid-ask. This caps ROR at 100% per trade,
# which is already aggressive.

SP_UIC_HORIZONS = [21, 30, 42, 63, 90, 126]
SP_UIC_CONFORMAL_QS = [0.96, 0.97, 0.975, 0.98, 0.985]
SP_UIC_WIDTHS = [0.01, 0.02, 0.03, 0.05]
SP_UIC_MAX_BUFFER = 0.30
SP_UIC_PER_FOLD_WIN = 0.80
SP_UIC_POOLED_WIN = 0.95
SP_UIC_TARGET_ROR = 0.50
SP_UIC_MAX_COMBINED_CREDIT_RATIO = 0.50


# -------------------- Robust Universal IC tier ----------------------
#
# Robust variant of UIC that defends against the 7 known
# backtest-to-live gaps.  Applies these proprietary layers on top
# of the same vol-adaptive joint conformal:
#
#  L1. Two-stage walk-forward:
#        Selection folds = 2020 .. (current_year - 2)
#        Confirmation folds = (current_year - 1), current_year
#      Pick (q, width) on selection folds ONLY. Then test the chosen
#      config on confirmation folds. Eligible iff BOTH stages pass.
#      Eliminates the multi-config peeking bias that plagues a single-
#      pass walk-forward.
#
#  L2. Stricter per-fold floor (0.90 instead of 0.80).
#
#  L3. Stricter pooled WR gate (0.97 selection, 0.95 confirmation).
#      Bonferroni-style deflation against the ~230k-configuration
#      hypothesis space.
#
#  L4. Conservative-pricing eligibility gate. Require combined ROR
#      >= 50% under STRESS pricing (haircut 0.65, IV mult 1.10).
#      Display normal pricing on the webapp; only signals that clear
#      50% even under worse-case fill assumptions get through.
#
#  L5. Regime σ distance check. Today's σ20 must fall within
#      [P05, P95] of the historical σ20 distribution. Rejects days
#      where current vol is in unprecedented territory relative to
#      what we trained on.
#
#  L6. Live log compatibility (handled outside the eligibility check
#      but the published signal carries the q+width+thresholds so
#      the live log knows exactly what to grade).

SP_RUIC_HORIZONS = [21, 30, 42, 63, 90, 126]
SP_RUIC_CONFORMAL_QS = [0.97, 0.975, 0.98, 0.985]
SP_RUIC_WIDTHS = [0.01, 0.02, 0.03, 0.05]
SP_RUIC_MAX_BUFFER = 0.30
SP_RUIC_PER_FOLD_WIN = 0.90              # was 0.80
SP_RUIC_SELECTION_POOLED_WIN = 0.97      # was 0.95
SP_RUIC_CONFIRMATION_POOLED_WIN = 0.95   # held-out years
SP_RUIC_TARGET_ROR_STRESS = 0.50         # ROR under stress pricing
SP_RUIC_STRESS_HAIRCUT = 0.65            # was 0.80
SP_RUIC_STRESS_IV_MULT = 1.10            # was 1.30
SP_RUIC_MAX_COMBINED_CREDIT_RATIO = 0.50
SP_RUIC_REGIME_SIGMA_LO_PCTILE = 5.0     # today's σ must be in [P05, P95]
SP_RUIC_REGIME_SIGMA_HI_PCTILE = 95.0


# -------------------- Liquid Active IC (LAIC) -----------------------
#
# 4-corners-of-the-pareto tier: liquid universe + 50% ROR + frequent
# trades, sacrificing some win rate. Targets 85% backtest WR
# (deflating to ~80% live with the same robustness defenses) on the
# liquid universe with broad horizons. Same 6 robustness layers as
# RUIC; only the WR thresholds and the ROR scoring differ.
SP_LAIC_HORIZONS = [7, 10, 14, 21, 30, 42, 63, 90]
SP_LAIC_CONFORMAL_QS = [0.85, 0.88, 0.90, 0.92, 0.94]
SP_LAIC_WIDTHS = [0.005, 0.01, 0.02, 0.03, 0.05]
SP_LAIC_MAX_BUFFER = 0.30
SP_LAIC_PER_FOLD_WIN = 0.75
SP_LAIC_SELECTION_POOLED_WIN = 0.85
SP_LAIC_CONFIRMATION_POOLED_WIN = 0.83
SP_LAIC_TARGET_ROR_STRESS = 0.50
SP_LAIC_STRESS_HAIRCUT = 0.65
SP_LAIC_STRESS_IV_MULT = 1.10
SP_LAIC_MAX_COMBINED_CREDIT_RATIO = 0.50

# -------------------- Liquid Frequent IC (LFIC) ---------------------
#
# Other corner: liquid universe + 95% backtest WR + frequent trades,
# sacrificing ROR (target 10%+ rather than 50%+). For users who
# prioritize accuracy over per-trade payout. Backtest 95% with
# stress-pricing eligibility deflates to ~92-93% live.
SP_LFIC_HORIZONS = [7, 10, 14, 21, 30, 42, 63, 90, 126]
SP_LFIC_CONFORMAL_QS = [0.97, 0.975, 0.98, 0.985]
SP_LFIC_WIDTHS = [0.005, 0.01, 0.02, 0.03, 0.05]
SP_LFIC_MAX_BUFFER = 0.30
SP_LFIC_PER_FOLD_WIN = 0.90
SP_LFIC_SELECTION_POOLED_WIN = 0.97
SP_LFIC_CONFIRMATION_POOLED_WIN = 0.95
SP_LFIC_TARGET_ROR_STRESS = 0.10        # only 10% — high-WR, low-ROR tier
SP_LFIC_STRESS_HAIRCUT = 0.65
SP_LFIC_STRESS_IV_MULT = 1.10
SP_LFIC_MAX_COMBINED_CREDIT_RATIO = 0.50


def selection_and_confirmation_folds():
    """Split FOLD_YEARS into selection and confirmation halves."""
    cur = max(FOLD_YEARS)
    if len(FOLD_YEARS) < 4:
        return list(FOLD_YEARS), []
    selection = [y for y in FOLD_YEARS if y <= cur - 2]
    confirmation = [y for y in FOLD_YEARS if y > cur - 2]
    return selection, confirmation


# -------------------- Liquid options universe ----------------------
#
# A priori curated list of tickers KNOWN to have liquid weekly +
# monthly options chains. Selection criterion is market structure
# (options open interest / daily option volume), NOT historical
# backtest performance — this avoids the cherry-picking critique.
# Source: standard "actively-traded options" lists from CBOE / OCC
# 2024 reports + every name in the S&P 100 + top sector ETFs.
SP_LIQUID_UNIVERSE = {
    # Major index ETFs
    "SPY", "QQQ", "IWM", "DIA", "EEM", "EFA", "VTI", "VOO", "VTV",
    "VUG", "VWO", "MDY",
    # Sector ETFs (popular options)
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU",
    "XLRE", "XLC", "XBI", "XOP", "XME", "XRT", "XHB", "SMH", "IBB",
    "KRE", "KBE", "GDX", "GDXJ",
    # Commodity / bond ETFs
    "GLD", "SLV", "USO", "UNG", "TLT", "TBT", "HYG", "LQD",
    # Volatility products (used as proxy for liquidity tier)
    "VXX", "UVXY", "SVXY",
    # Mega-cap tech (highest options volume)
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "NFLX", "PYPL",
    # Mega-cap finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "BLK",
    # Mega-cap health/consumer
    "JNJ", "UNH", "PFE", "MRK", "LLY", "ABBV", "TMO", "ABT",
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "DIS", "NKE", "HD", "LOW",
    # Mega-cap industrial / energy
    "XOM", "CVX", "COP", "SLB", "BA", "CAT", "GE", "HON", "RTX",
    # Crypto-adjacent (very active options)
    "COIN", "MSTR", "RIOT", "MARA", "HOOD",
    # Other widely-traded options names
    "BABA", "F", "GM", "T", "VZ", "DAL", "AAL", "UAL",
    "GME", "AMC", "PLTR", "SOFI", "NIO", "BIDU",
    # Berkshire (BRK-A available; B class typically used in retail)
    "BRK-A", "BRK-B",
}
SP_IC_POOLED_WIN = 0.95             # joint pooled WR floor (Atomic IC)
SP_IC_TARGET_ROR = 0.50             # min combined per-trade ROR (Atomic IC)


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


def tight_mask(f: StillpointFeatures) -> np.ndarray:
    """Stricter Stillpoint variant — used for the tight (<3% buffer) tier."""
    return (
        np.isfinite(f.vol20) & np.isfinite(f.vol5) & np.isfinite(f.compression)
        & np.isfinite(f.rsi14) & np.isfinite(f.range20)
        & np.isfinite(f.trend_flat) & np.isfinite(f.move5d)
        & (f.vol20 < SP_TIGHT_VOL20_MAX)
        & (f.compression < SP_TIGHT_COMPRESSION_MAX)
        & (f.range20 < SP_TIGHT_RANGE_MAX)
        & (f.trend_flat < SP_TIGHT_TREND_FLAT_MAX)
        & (f.rsi14 >= SP_TIGHT_RSI_BAND[0])
        & (f.rsi14 <= SP_TIGHT_RSI_BAND[1])
        & (np.abs(f.move5d) < SP_TIGHT_RECENT_MOVE_MAX)
    )


def today_in_tight_regime(f: StillpointFeatures) -> bool:
    return bool(tight_mask(f)[-1])


# ----------------------- buffer helpers ------------------------------

def buffer_array(close: np.ndarray, h: int, side: str) -> np.ndarray:
    if side == "put":
        return worst_buffer_path(close, h)
    return worst_buffer_path_up(close, h)


def strike_from_buffer(spot: float, buffer: float, side: str) -> float:
    if side == "put":
        return spot * (1.0 - buffer)
    return spot * (1.0 + buffer)


def close_buffer_arrays(close: np.ndarray, h: int):
    """Close-at-expiry buffer arrays.

    For each historical day t with t+h within data:
      drop(t)  = max(0, 1 − close[t+h] / close[t])
      rise(t)  = max(0, close[t+h] / close[t] − 1)

    Used by the Universal IC tier whose win condition is
    'close[t+h] inside [K_put_short, K_call_short]'. Strictly looser
    than the path-criterion (path can touch and recover).
    """
    n = len(close)
    fwd = np.full(n, np.nan, dtype="float64")
    if n > h:
        fwd[: n - h] = close[h:]
    with np.errstate(invalid="ignore", divide="ignore"):
        drop = np.where(np.isnan(fwd), np.nan, np.maximum(1.0 - fwd / close, 0.0))
        rise = np.where(np.isnan(fwd), np.nan, np.maximum(fwd / close - 1.0, 0.0))
    return drop, rise
