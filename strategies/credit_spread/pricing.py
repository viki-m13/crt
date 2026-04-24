"""Simple Black-Scholes credit-spread pricing for CreditFloor.

We publish a short strike K = spot * (1 ± buffer). To estimate the
credit a user would realistically receive for selling a vertical credit
spread at that strike, we:

    1. Compute realized annualized volatility from the stock's last
       ``VOL_WINDOW`` trading days of log returns.
    2. Apply a mild IV-skew uplift (``IV_MULT``) — real OTM options
       trade at higher IV than realized.
    3. Price the short leg at strike K using Black-Scholes (r=0).
    4. Price the long leg one spread-width further OTM
       (K_long = K - W for puts, K_long = K + W for calls).
       Width W is ``WIDTH_PCT`` of spot (defaults to 5%, matches common
       brokerage defaults for $5-wide spreads on ~$100 stocks).
    5. Credit = (short - long) * ``HAIRCUT`` (20% conservative haircut
       for bid-ask slippage).
    6. Max loss per contract = (W * 100) - (credit * 100).
    7. Return-on-risk per trade  = credit / (W - credit).
    8. Annualized ROR = ROR * (365 / calendar_days_to_expiry).

Important sign conventions (triple-checked):
    - Put: payoff = max(K - S_T, 0); put price = K*N(-d2) - S*N(-d1)
      with d1 = [ln(S/K) + (σ²/2)T] / (σ√T), d2 = d1 - σ√T, r=0.
    - Call: payoff = max(S_T - K, 0); call price = S*N(d1) - K*N(d2).
    - For a put spread (short higher K, long lower K-W): credit is
      positive because the higher-strike put has greater intrinsic +
      time value. K_short > K_long.
    - For a call spread (short lower K, long higher K+W): credit is
      positive because the lower-strike call is more expensive.
      K_short < K_long.

The published strike K is always at the SHORT leg — that is
    - PUT  credit spread: K_short = spot*(1 - buffer) <= spot
    - CALL credit spread: K_short = spot*(1 + buffer) >= spot
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


VOL_WINDOW = 60           # trading days of log returns for realized vol
IV_MULT = 1.30            # implied vol = realized * this (mild skew uplift)
WIDTH_PCT = 0.05          # spread width = spot * this fraction
HAIRCUT = 0.80            # credit multiplier (conservative for bid-ask)
CALENDAR_DAYS_PER_YEAR = 365.0


def _ncdf(x: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes put price with r=0, no dividends."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)  # intrinsic
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * _ncdf(-d2) - S * _ncdf(-d1)


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call price with r=0, no dividends."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * _ncdf(d2)


def realized_vol(closes: np.ndarray, window: int = VOL_WINDOW) -> float | None:
    """Annualized stdev of daily log returns over the last `window` days.
    Returns None if insufficient data.
    """
    if len(closes) < window + 1:
        return None
    tail = closes[-window - 1 :]
    if (tail <= 0).any():
        return None
    log_ret = np.diff(np.log(tail))
    # Sample stdev (ddof=1), annualized by sqrt(252).
    s = float(np.std(log_ret, ddof=1))
    return s * math.sqrt(252.0)


@dataclass
class ProfitEstimate:
    """All numbers are per-share (one contract = 100 shares)."""
    side: str            # 'put' or 'call'
    spot: float
    short_strike: float
    long_strike: float
    width: float         # = |short_strike - long_strike|
    buffer_pct: float    # short-strike distance from spot (informational)
    realized_vol: float  # annualized
    implied_vol: float   # what we used for BS
    t_years: float       # calendar-days / 365
    horizon_sessions: int
    short_price: float   # BS value of short leg
    long_price: float    # BS value of long leg
    credit: float        # (short - long) * haircut
    max_loss: float      # width - credit
    return_on_risk: float   # credit / max_loss  (per-trade)
    annualized_ror: float   # ror * (365 / calendar_days_to_expiry)


def estimate_profit(
    side: str,
    spot: float,
    buffer: float,                # as a fraction, e.g. 0.1429 for 14.29%
    horizon_sessions: int,
    realized_sigma: float,
    calendar_days_to_expiry: int,
    width_pct: float = WIDTH_PCT,
    iv_mult: float = IV_MULT,
    haircut: float = HAIRCUT,
) -> ProfitEstimate | None:
    """Compute a conservative estimate of the credit and return-on-risk
    for a vertical credit spread at our published strike.

    `buffer` is the short-leg OTM fraction our rule certifies:
        put:  short strike = spot * (1 - buffer)
        call: short strike = spot * (1 + buffer)

    `width_pct` is the spread width in fractions of spot. The long leg
    sits one ``width`` further OTM than the short leg (deeper OTM in
    both put and call cases), so max loss per contract is bounded by
    the width.
    """
    if side not in ("put", "call"):
        return None
    if spot <= 0 or buffer < 0 or horizon_sessions <= 0:
        return None
    if realized_sigma is None or realized_sigma <= 0:
        return None
    if calendar_days_to_expiry <= 0:
        return None

    T = calendar_days_to_expiry / CALENDAR_DAYS_PER_YEAR
    iv = realized_sigma * iv_mult
    width = spot * width_pct
    if side == "put":
        k_short = spot * (1.0 - buffer)
        k_long  = k_short - width
        if k_long <= 0:
            # Deep-OTM put with width pushing past zero; skip.
            return None
        p_short = bs_put(spot, k_short, T, iv)
        p_long  = bs_put(spot, k_long,  T, iv)
    else:
        k_short = spot * (1.0 + buffer)
        k_long  = k_short + width
        p_short = bs_call(spot, k_short, T, iv)
        p_long  = bs_call(spot, k_long,  T, iv)

    # Sanity: short leg (closer to spot) should be worth more than the
    # further-OTM long leg. If numerical noise flips it, clamp.
    raw_credit = max(p_short - p_long, 0.0)
    credit = raw_credit * haircut
    max_loss = max(width - credit, 0.01)  # floor to avoid div-by-zero
    ror = credit / max_loss
    ann_ror = ror * (CALENDAR_DAYS_PER_YEAR / calendar_days_to_expiry)

    return ProfitEstimate(
        side=side,
        spot=spot,
        short_strike=k_short,
        long_strike=k_long,
        width=width,
        buffer_pct=buffer * 100.0,
        realized_vol=realized_sigma,
        implied_vol=iv,
        t_years=T,
        horizon_sessions=horizon_sessions,
        short_price=p_short,
        long_price=p_long,
        credit=credit,
        max_loss=max_loss,
        return_on_risk=ror,
        annualized_ror=ann_ror,
    )
