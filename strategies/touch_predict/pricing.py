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
IV_MULT = 1.15               # IV = realized * this (mild skew uplift for
                             #   liquid names; retail bid-ask on monthly
                             #   3rd-Friday expiries is usually tight)
HAIRCUT = 0.80               # credit-side haircut NOT used here; long-side
                             # slippage is modeled by inflating premium:
PREMIUM_SLIPPAGE = 1.05      # pay 5% more than BS mid for normal-priced
                             #   contracts (tightest bid-ask on $>0.50)
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


# --- Per-leg smile + bid-ask slippage for credit-spread pricing -------------
# Used by option_c_research.py for both the historical backtest and live
# signal pricing. The OtmPlay (long-option) path below is unchanged — its
# slippage model already operates on a single bought leg.

SKEW_ALPHA_PUT  = 0.20
SKEW_ALPHA_CALL = 0.05

NET_BID_ASK_FLOOR = 0.05      # $/share net spread bid-ask minimum
NET_BID_ASK_FRAC  = 0.10      # bid-ask widens linearly with mid above floor
MIN_TRADEABLE_FILL = 0.05     # rungs below this are flagged untradeable
QUALITY_THIN_MAX   = 0.10
QUALITY_MODEST_MAX = 0.30


def iv_at_strike(spot: float, K: float, T: float, atm_iv: float, side: str,
                 alpha_put: float = SKEW_ALPHA_PUT,
                 alpha_call: float = SKEW_ALPHA_CALL) -> float:
    """Per-strike IV under a multiplicative log-moneyness smile.

    iv(K, T) = atm_iv * (1 + alpha * max(0, ±log(spot/K)) / sqrt(T))

    For puts the uplift is on K below spot; for calls on K above spot.
    K on the wrong side of spot returns atm_iv (no uplift).
    """
    if T <= 0 or K <= 0 or spot <= 0 or atm_iv <= 0:
        return atm_iv
    x = math.log(spot / K) / math.sqrt(T)
    if side == "put":
        return atm_iv * (1.0 + alpha_put * max(0.0, x))
    return atm_iv * (1.0 + alpha_call * max(0.0, -x))


def tenor_haircut(T: float) -> float:
    """Tenor-aware fraction of BS-mid you can capture as a limit fill on
    a vertical credit spread. Short-dated weeklies fill near mid, LEAPS
    markets are thin and fill far below.
    """
    if T <= 0:
        return 0.80
    if T < 0.10:
        return 0.80
    if T < 0.25:
        return 0.72
    if T < 0.50:
        return 0.65
    if T < 1.00:
        return 0.58
    return 0.50


def expected_fill_credit(mid_credit: float, T: float) -> tuple[float, float]:
    """BS-mid → expected limit-order fill, modeling tenor + bid-ask.
    Returns (fill, bid_ask_estimate)."""
    if mid_credit <= 0 or T <= 0:
        return 0.0, NET_BID_ASK_FLOOR
    bid_ask = max(NET_BID_ASK_FLOOR, NET_BID_ASK_FRAC * mid_credit)
    bid_fill = max(0.0, mid_credit - bid_ask / 2.0)
    haircut_fill = mid_credit * tenor_haircut(T)
    return min(bid_fill, haircut_fill), bid_ask


def credit_quality(fill_credit: float) -> str:
    """Classify per-share fill credit as rich/modest/thin."""
    if fill_credit < QUALITY_THIN_MAX:
        return "thin"
    if fill_credit < QUALITY_MODEST_MAX:
        return "modest"
    return "rich"


def credit_spread_price(side: str, spot: float, K_short: float, K_long: float,
                         T: float, atm_iv: float
                         ) -> tuple[float, float, float, float, float, float]:
    """Full credit-spread pricing bundle under per-leg smile + slippage.

    Returns:
        (mid_credit, fill_credit, bid_ask, max_loss, sigma_short, sigma_long)
    All per share. Width is implicit in (K_short, K_long).
    """
    if T <= 0 or atm_iv <= 0 or min(spot, K_short, K_long) <= 0:
        return 0.0, 0.0, NET_BID_ASK_FLOOR, max(abs(K_long - K_short), 0.01), atm_iv, atm_iv
    sigma_s = iv_at_strike(spot, K_short, T, atm_iv, side)
    sigma_l = iv_at_strike(spot, K_long,  T, atm_iv, side)
    if side == "put":
        s = bs_put(spot, K_short, T, sigma_s)
        l = bs_put(spot, K_long,  T, sigma_l)
    else:
        s = bs_call(spot, K_short, T, sigma_s)
        l = bs_call(spot, K_long,  T, sigma_l)
    mid = max(s - l, 0.0)
    fill, bid_ask = expected_fill_credit(mid, T)
    width = abs(K_long - K_short)
    max_loss = max(width - fill, 0.01)
    return mid, fill, bid_ask, max_loss, sigma_s, sigma_l


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
    # wider relative bid-ask. Normal-priced liquid-option pricing is
    # close to BS mid; very cheap options get penalized more.
    if bs < 0.10:
        slip = 1.30
    elif bs < 0.25:
        slip = 1.15
    else:
        slip = PREMIUM_SLIPPAGE   # 1.05 by default
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
