"""Long-option OTM pricing for TouchPredictor.

For a long-call trade predicting the stock will touch S*(1+b) within h
sessions, we want to BUY an OTM call at some strike K = S*(1+k) with
0 < k < b. Premium is cheap at higher k, but intrinsic at touch
(T - K) = (b - k) * S shrinks. ROI = profit / premium has an interior
maximum we find by grid search.

Symmetric for puts: target = S*(1-b), strike K = S*(1-k), k < b.

Profit model (conservative, for ranking purposes):
    profit_per_share = max(target_price - strike, 0) - premium       (call)
                     = max(strike - target_price, 0) - premium       (put)
    max_loss         = premium              (long options cap loss at premium paid)
    ROI              = profit / premium     (multiple of premium received as profit)

This assumes we sell at the moment of touch, ignoring any residual
time value on top of intrinsic (conservative — real sales typically
capture some TV too). BS uses IV = realized_vol * IV_MULT with no
dividends and r=0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


VOL_WINDOW = 60              # trading-day lookback for realized vol
IV_MULT = 1.30               # IV = realized * this (skew uplift)
HAIRCUT = 0.80               # credit-side haircut NOT used here; long-side
                             # slippage is modeled by inflating premium:
PREMIUM_SLIPPAGE = 1.15      # pay 15% more than BS mid (conservative)
CALENDAR_DAYS_PER_YEAR = 365.0

# Grid for strike-placement fraction k/b ∈ (0, 1). We sweep in [0.05, 0.95]
# and pick the k that maximizes ROI. Never allow k >= buffer (option
# would expire worthless even on a hit at exactly target).
STRIKE_GRID = np.linspace(0.05, 0.95, 19)  # 0.05, 0.10, ..., 0.95

# Minimum tradable premium per share. Very cheap options DO fill in
# practice, just with wider relative slippage. The user's whole thesis
# is "super cheap OTM" so we keep this floor low and let the slippage
# multiplier do the work.
MIN_PREMIUM_PER_SHARE = 0.02


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * _ncdf(d2)


def bs_put(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * _ncdf(-d2) - S * _ncdf(-d1)


def realized_vol(closes: np.ndarray, window: int = VOL_WINDOW) -> float | None:
    if len(closes) < window + 1:
        return None
    tail = closes[-window - 1 :]
    if (tail <= 0).any():
        return None
    log_ret = np.diff(np.log(tail))
    s = float(np.std(log_ret, ddof=1))
    return s * math.sqrt(252.0)


@dataclass
class OtmPlay:
    side: str           # 'call' or 'put'
    spot: float
    buffer: float       # certified touch buffer (fraction)
    target_price: float
    k_frac: float       # strike placement: k / buffer (in (0, 1))
    strike: float
    t_years: float
    realized_vol: float
    implied_vol: float
    premium: float      # BS * PREMIUM_SLIPPAGE (what we PAY)
    profit: float       # (target - strike) - premium (call); symmetric for put
    max_loss: float     # = premium
    roi: float          # profit / premium


def _play_at(
    side: str,
    spot: float,
    buffer: float,
    k_frac: float,
    T_years: float,
    sigma: float,
    realized_sigma: float,
) -> OtmPlay | None:
    """Compute an OtmPlay at a specific k_frac (strike placement fraction
    of buffer). Returns None if invalid."""
    if k_frac <= 0 or k_frac >= 1:
        return None
    k = buffer * k_frac
    if side == "call":
        target = spot * (1.0 + buffer)
        strike = spot * (1.0 + k)
        bs = bs_call(spot, strike, T_years, sigma)
    else:
        target = spot * (1.0 - buffer)
        strike = spot * (1.0 - k)
        if strike <= 0:
            return None
        bs = bs_put(spot, strike, T_years, sigma)
    if bs <= 0:
        return None
    # Slippage scales with how cheap the option is — penny options have
    # wider relative bid-ask. Apply 1.15× for normal-priced, up to 1.50×
    # for <$0.20 options. The grid search still finds the ROI-optimal
    # placement after this adjustment.
    if bs < 0.20:
        slip = 1.50
    elif bs < 0.50:
        slip = 1.30
    else:
        slip = PREMIUM_SLIPPAGE
    premium = bs * slip
    if premium < MIN_PREMIUM_PER_SHARE:
        return None
    intrinsic_at_touch = (buffer - k) * spot
    profit = intrinsic_at_touch - premium
    roi = profit / premium
    return OtmPlay(
        side=side,
        spot=spot,
        buffer=buffer,
        target_price=target,
        k_frac=k_frac,
        strike=strike,
        t_years=T_years,
        realized_vol=realized_sigma,
        implied_vol=sigma,
        premium=premium,
        profit=profit,
        max_loss=premium,
        roi=roi,
    )


def best_otm_play(
    side: str,
    spot: float,
    buffer: float,
    calendar_days_to_expiry: int,
    realized_sigma: float | None,
    iv_mult: float = IV_MULT,
) -> OtmPlay | None:
    """Grid-search over k_frac in STRIKE_GRID for the ROI-maximizing OTM
    call or put. Returns None if no profitable play exists.

    `buffer` is the certified touch buffer (fraction).
    `calendar_days_to_expiry` drives the BS time-to-expiry.
    `realized_sigma` is annualized realized vol; IV used = that * iv_mult.
    """
    if side not in ("call", "put"):
        return None
    if spot <= 0 or buffer <= 0 or calendar_days_to_expiry <= 0:
        return None
    if realized_sigma is None or realized_sigma <= 0:
        return None

    T = calendar_days_to_expiry / CALENDAR_DAYS_PER_YEAR
    iv = realized_sigma * iv_mult

    best: OtmPlay | None = None
    for k_frac in STRIKE_GRID:
        p = _play_at(side, spot, buffer, float(k_frac), T, iv, realized_sigma)
        if p is None:
            continue
        if p.roi <= 0:
            continue  # skip unprofitable placements
        if best is None or p.roi > best.roi:
            best = p
    return best
