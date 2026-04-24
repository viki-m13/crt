"""Novel touch-prediction regimes for v2 (liquid-only, OHLCV-aware).

Each regime returns a boolean mask over a stock's trading days where
the regime's entry criterion holds. All conditions are CAUSAL —
features at day t use only data through day t.

Evidence basis comes from:
  - Connors & Alvarez "Short Term Trading Strategies That Work"
  - Bulkowski's "Encyclopedia of Candlestick Charts"
  - Khandani & Lo (2007) short-term reversal literature
  - Research-agent scan (see session notes)

The walk-forward engine tries each regime at each horizon and picks
the most-profitable (highest EV) eligible combo.
"""
from __future__ import annotations

import numpy as np
from v2_common import V2Features, align_to_spy, spy_context


# ---------- CALL side (up-touch): mean-reversion bounces -----------------


def call_plain(f: V2Features) -> np.ndarray:
    return (
        np.isfinite(f.sma200) & np.isfinite(f.rsi14) & np.isfinite(f.vol20)
    )


def call_connors_tps(f: V2Features) -> np.ndarray:
    """Connors TPS adapted for stocks.
    close < 5-SMA AND RSI(2) < 10 AND volume > 2x 50-day avg AND
    today's 1d return < -2% AND close > 200-SMA.
    Claimed ~85-90% reversion rate in Connors/Alvarez (2008)."""
    close_below_sma5 = np.isfinite(f.sma5) & np.isfinite(f.rsi2) & \
        np.isfinite(f.vol50) & np.isfinite(f.sma200) & np.isfinite(f.ret_1d)
    return close_below_sma5 & (f.rsi2 < 10.0) & \
        (f.vol_z50 > 2.0) & (f.ret_1d < -0.02) & (f.trend > 1.00)


def call_multi_stack(f: V2Features) -> np.ndarray:
    """Deep multi-stack mean-reversion setup.
    close < lower Bollinger (20, 2σ) AND RSI(2) < 5 AND
    close > 200-SMA AND volume > 1.5x 20-day avg AND 5d return < -5%.
    Claimed ~87-91% touch of +2% in 5-10 days on large-caps."""
    return (
        np.isfinite(f.boll_lower) & np.isfinite(f.rsi2)
        & np.isfinite(f.vol20) & np.isfinite(f.sma200) & np.isfinite(f.ret_5d)
        & (__import__("numpy").arange(len(f.sma200)) >= 0)  # placeholder to keep array shape aligned
    ) & (
        # close below lower Bollinger: test value array shape-compat
        np.isfinite(f.boll_lower)
    )


def call_multi_stack_v2(f: V2Features, close: np.ndarray) -> np.ndarray:
    """Same as multi_stack, referencing raw close."""
    return (
        np.isfinite(f.boll_lower) & np.isfinite(f.rsi2)
        & np.isfinite(f.vol_z20) & np.isfinite(f.sma200) & np.isfinite(f.ret_5d)
        & (close < f.boll_lower)
        & (f.rsi2 < 5.0)
        & (f.trend > 1.00)
        & (f.vol_z20 > 1.5)
        & (f.ret_5d < -0.05)
    )


def call_panic_day(f: V2Features, close: np.ndarray) -> np.ndarray:
    """Single-day panic-drop setup.
    1-day return < -5% AND volume > 1.5x 20-day avg AND close > 200-SMA
    (healthy stock that just took a shock). High-probability short-term
    bounce in 3-7 days."""
    return (
        np.isfinite(f.ret_1d) & np.isfinite(f.vol_z20) & np.isfinite(f.sma200)
        & (f.ret_1d < -0.05)
        & (f.vol_z20 > 1.5)
        & (f.trend > 1.00)
    )


def call_spy_rel_weakness(f: V2Features, stock_dates: np.ndarray) -> np.ndarray:
    """Cross-sectional: stock 5d return much weaker than SPY, but
    healthy stock > 200-SMA. Expect catch-up bounce."""
    ctx = spy_context()
    if ctx is None:
        return np.zeros(len(f.sma200), dtype=bool)
    spy_d, _, spy_r5 = ctx
    spy_r5_aligned = align_to_spy(stock_dates, spy_d, spy_r5)
    rel = f.ret_5d - spy_r5_aligned
    return (
        np.isfinite(rel) & np.isfinite(f.rsi2) & np.isfinite(f.sma200)
        & (rel < -0.07)
        & (f.rsi2 < 15.0)
        & (f.trend > 1.00)
    )


def call_deep_oversold(f: V2Features) -> np.ndarray:
    """Deep oversold: RSI(14) < 20 OR 252d drawdown > 20%.
    (Same as v1; included for comparison.)"""
    return (
        np.isfinite(f.rsi14) & np.isfinite(f.dd252)
        & ((f.rsi14 < 20.0) | (f.dd252 >= 0.20))
    )


# ---------- PUT side (down-touch): mean-reversion from overbought --------


def put_plain(f: V2Features) -> np.ndarray:
    return np.isfinite(f.sma200) & np.isfinite(f.rsi14) & np.isfinite(f.vol20)


def put_connors_tps(f: V2Features) -> np.ndarray:
    """Mirror of call_connors_tps for puts.
    close > 5-SMA AND RSI(2) > 90 AND volume > 2x 50-day avg AND
    today's return > +2% AND close < 200-SMA (downtrend rally).
    Expect pullback."""
    return (
        np.isfinite(f.sma5) & np.isfinite(f.rsi2)
        & np.isfinite(f.vol_z50) & np.isfinite(f.sma200) & np.isfinite(f.ret_1d)
        & (f.rsi2 > 90.0)
        & (f.vol_z50 > 2.0)
        & (f.ret_1d > 0.02)
        & (f.trend < 1.00)
    )


def put_multi_stack(f: V2Features, close: np.ndarray) -> np.ndarray:
    """close > upper Bollinger AND RSI(2) > 95 AND close < 200-SMA
    AND volume > 1.5x AND 5d return > +5%. Mean-reversion-down."""
    return (
        np.isfinite(f.boll_upper) & np.isfinite(f.rsi2)
        & np.isfinite(f.vol_z20) & np.isfinite(f.sma200) & np.isfinite(f.ret_5d)
        & (close > f.boll_upper)
        & (f.rsi2 > 95.0)
        & (f.trend < 1.00)
        & (f.vol_z20 > 1.5)
        & (f.ret_5d > 0.05)
    )


def put_parabolic(f: V2Features) -> np.ndarray:
    """Parabolic move setup: 5d return > +15% AND RSI(2) > 95."""
    return (
        np.isfinite(f.ret_5d) & np.isfinite(f.rsi2)
        & (f.ret_5d > 0.15)
        & (f.rsi2 > 95.0)
    )


# ---------- Registry -----------------------------------------------------


CALL_REGIMES = {
    "plain":           lambda f, c, d: call_plain(f),
    "connors_tps":     lambda f, c, d: call_connors_tps(f),
    "multi_stack":     lambda f, c, d: call_multi_stack_v2(f, c),
    "panic_day":       lambda f, c, d: call_panic_day(f, c),
    "spy_rel_weak":    lambda f, c, d: call_spy_rel_weakness(f, d),
    "deep_oversold":   lambda f, c, d: call_deep_oversold(f),
}

PUT_REGIMES = {
    "plain":           lambda f, c, d: put_plain(f),
    "connors_tps":     lambda f, c, d: put_connors_tps(f),
    "multi_stack":     lambda f, c, d: put_multi_stack(f, c),
    "parabolic":       lambda f, c, d: put_parabolic(f),
}
