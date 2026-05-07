"""Black-Scholes + smile credit-spread pricing for CreditFloor.

To estimate the credit a user would realistically receive for selling a
vertical credit spread at our published strikes, we layer three things on
top of Black-Scholes:

    1. ATM IV anchor: ``atm_iv = realized_vol * IV_MULT``. Realized vol
       is the stdev of the last ``VOL_WINDOW`` daily log returns.

    2. Per-leg IV smile (``iv_at_strike``). Equity puts have a put-side
       smile that uplifts deep-OTM IV materially above ATM. Without
       this, BS-flat under-prices the further-OTM long leg of a
       vertical, which inflates the implied credit. The smile is a
       multiplicative function of normalized log-moneyness:

           iv(K, T) = atm_iv * (1 + alpha * max(0, ±log(spot/K))/√T)

       with ``alpha = SKEW_ALPHA_PUT`` for puts (uplift when K<spot) and
       ``alpha = SKEW_ALPHA_CALL`` for calls (uplift when K>spot, much
       milder for liquid US equities).

    3. Tenor-aware bid-ask slippage. Real fills sit closer to the bid
       than the BS mid; the gap widens with tenor (LEAPS markets are
       thinner) and with thin credits (a $0.05 minimum bid-ask eats most
       of a dime credit). ``expected_fill_credit`` returns
       ``min(mid * tenor_haircut, mid - bid_ask/2)`` — whichever is more
       conservative.

The published short strike is always K_short — for puts K_short = spot*
(1 - buffer) ≤ spot; for calls K_short = spot*(1 + buffer) ≥ spot. The
long leg sits one ``WIDTH_PCT * spot`` further OTM in both cases, so
max loss per contract is bounded by the spread width.

Important sign conventions (triple-checked):
    - Put: payoff = max(K - S_T, 0); put price = K*N(-d2) - S*N(-d1)
      with d1 = [ln(S/K) + (σ²/2)T] / (σ√T), d2 = d1 - σ√T, r=0.
    - Call: payoff = max(S_T - K, 0); call price = S*N(d1) - K*N(d2).
    - Put spread (short higher K, long lower K-W): credit positive.
    - Call spread (short lower K, long higher K+W): credit positive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


VOL_WINDOW = 60           # trading days of log returns for realized vol
IV_MULT = 1.30            # ATM IV = realized * this (mild skew uplift)
WIDTH_PCT = 0.05          # spread width = spot * this fraction
HAIRCUT = 0.80            # legacy credit multiplier (kept for back-compat)
CALENDAR_DAYS_PER_YEAR = 365.0

# --- Smile / slippage configuration -----------------------------------------
# Multiplicative IV uplift slope per unit normalized log-moneyness. Equities
# have a steep put-side smile (alpha~0.20 captures it modestly) and a much
# flatter call wing.
SKEW_ALPHA_PUT  = 0.20
SKEW_ALPHA_CALL = 0.05

# Spread net bid-ask floor. Even highly liquid options quote at least a
# $0.05 spread on the combo; that floor dominates for thin credits.
NET_BID_ASK_FLOOR = 0.05      # $/share net spread bid-ask minimum
NET_BID_ASK_FRAC  = 0.10      # $/share bid-ask fraction of mid above floor

# Below this fill, we mark a rung as untradeable (the limit order won't
# get the credit the model is suggesting in any reasonable market).
MIN_TRADEABLE_FILL = 0.05     # $/share

# Quality bucket thresholds (in $/share fill credit).
QUALITY_THIN_MAX  = 0.10
QUALITY_MODEST_MAX = 0.30


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


def iv_at_strike(spot: float, K: float, T: float, atm_iv: float, side: str,
                 alpha_put: float = SKEW_ALPHA_PUT,
                 alpha_call: float = SKEW_ALPHA_CALL) -> float:
    """Per-strike implied vol under a multiplicative log-moneyness smile.

    iv(K, T) = atm_iv * (1 + alpha * max(0, ±log(spot/K)) / sqrt(T))

    For puts the uplift is on K below spot (x = log(spot/K) > 0); for
    calls it is on K above spot. K on the wrong side returns atm_iv.
    """
    if T <= 0 or K <= 0 or spot <= 0 or atm_iv <= 0:
        return atm_iv
    x = math.log(spot / K) / math.sqrt(T)
    if side == "put":
        return atm_iv * (1.0 + alpha_put * max(0.0, x))
    return atm_iv * (1.0 + alpha_call * max(0.0, -x))


def tenor_haircut(T: float) -> float:
    """Tenor-aware fraction of BS-mid you can realistically capture as
    a limit-order fill on a vertical credit spread. Short-dated weeklies
    fill very close to mid; LEAPS markets are thin and fill far below.
    """
    if T <= 0:
        return HAIRCUT
    if T < 0.10:    # < ~5 weeks
        return 0.80
    if T < 0.25:    # < ~3 months
        return 0.72
    if T < 0.50:    # < ~6 months
        return 0.65
    if T < 1.00:    # < ~1 year
        return 0.58
    return 0.50     # LEAPS


def expected_fill_credit(mid_credit: float, T: float) -> tuple[float, float]:
    """Convert a BS-mid spread credit to an expected limit-order fill.

    Two regimes, take the more conservative one:
        - tenor regime: fill = mid * tenor_haircut(T)
        - bid-side regime: fill = mid - max(floor, frac*mid) / 2
    Returns (fill, bid_ask_estimate). Returns (0, floor) if untradeable.
    """
    if mid_credit <= 0 or T <= 0:
        return 0.0, NET_BID_ASK_FLOOR
    bid_ask = max(NET_BID_ASK_FLOOR, NET_BID_ASK_FRAC * mid_credit)
    bid_fill = max(0.0, mid_credit - bid_ask / 2.0)
    haircut_fill = mid_credit * tenor_haircut(T)
    return min(bid_fill, haircut_fill), bid_ask


def credit_quality(fill_credit: float) -> str:
    """Classify a per-share fill credit as rich/modest/thin."""
    if fill_credit < QUALITY_THIN_MAX:
        return "thin"
    if fill_credit < QUALITY_MODEST_MAX:
        return "modest"
    return "rich"


def credit_spread_mid(side: str, spot: float, K_short: float, K_long: float,
                      T: float, atm_iv: float) -> float:
    """BS-mid net credit per share for a vertical credit spread, with the
    per-leg IV smile applied. Returns ``max(short - long, 0)``.

    Used by callers that want to keep their own haircut/slippage logic
    (e.g. Stillpoint's stress vs normal IV pathways) but need the smile
    correction at the leg level.
    """
    if T <= 0 or atm_iv <= 0 or min(spot, K_short, K_long) <= 0:
        return 0.0
    sigma_s = iv_at_strike(spot, K_short, T, atm_iv, side)
    sigma_l = iv_at_strike(spot, K_long,  T, atm_iv, side)
    if side == "put":
        s = bs_put(spot, K_short, T, sigma_s)
        l = bs_put(spot, K_long,  T, sigma_l)
    else:
        s = bs_call(spot, K_short, T, sigma_s)
        l = bs_call(spot, K_long,  T, sigma_l)
    return max(s - l, 0.0)


@dataclass
class ProfitEstimate:
    """All numbers are per-share (one contract = 100 shares)."""
    side: str
    spot: float
    short_strike: float
    long_strike: float
    width: float
    buffer_pct: float
    realized_vol: float
    implied_vol: float       # ATM IV used as the smile anchor
    short_iv: float          # per-leg IV under the smile
    long_iv: float
    t_years: float
    horizon_sessions: int
    short_price: float       # BS mid (with per-leg smile IV)
    long_price: float
    mid_credit: float        # short - long, before slippage
    credit: float            # expected limit-order fill (post slippage)
    bid_ask_estimate: float  # net spread bid-ask, $/share
    max_loss: float          # width - credit
    return_on_risk: float    # credit / max_loss  (per-trade)
    annualized_ror: float    # ror * (365 / calendar_days_to_expiry)
    quality: str             # 'rich' | 'modest' | 'thin'
    tradeable: bool          # fill >= MIN_TRADEABLE_FILL


def estimate_profit(
    side: str,
    spot: float,
    buffer: float,
    horizon_sessions: int,
    realized_sigma: float,
    calendar_days_to_expiry: int,
    width_pct: float = WIDTH_PCT,
    iv_mult: float = IV_MULT,
    haircut: float | None = None,   # ignored; kept for back-compat
) -> ProfitEstimate | None:
    """Estimate the credit, max loss, and return-on-risk for a vertical
    credit spread sold at the published short strike.

    `buffer` is the short-leg OTM fraction the rule certifies:
        put:  short strike = spot * (1 - buffer); long is `width` lower
        call: short strike = spot * (1 + buffer); long is `width` higher

    The credit is computed as the BS midpoint with the per-leg IV smile,
    then converted into an expected limit-order fill via
    ``expected_fill_credit``. The legacy ``haircut`` keyword is accepted
    but ignored — the new model is parameter-free at the call site.
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
    atm_iv = realized_sigma * iv_mult
    width = spot * width_pct
    if side == "put":
        k_short = spot * (1.0 - buffer)
        k_long  = k_short - width
        if k_long <= 0:
            return None
    else:
        k_short = spot * (1.0 + buffer)
        k_long  = k_short + width

    sigma_s = iv_at_strike(spot, k_short, T, atm_iv, side)
    sigma_l = iv_at_strike(spot, k_long,  T, atm_iv, side)
    if side == "put":
        p_short = bs_put(spot, k_short, T, sigma_s)
        p_long  = bs_put(spot, k_long,  T, sigma_l)
    else:
        p_short = bs_call(spot, k_short, T, sigma_s)
        p_long  = bs_call(spot, k_long,  T, sigma_l)

    mid_credit = max(p_short - p_long, 0.0)
    fill_credit, bid_ask = expected_fill_credit(mid_credit, T)
    max_loss = max(width - fill_credit, 0.01)
    ror = fill_credit / max_loss
    ann_ror = ror * (CALENDAR_DAYS_PER_YEAR / calendar_days_to_expiry)
    quality = credit_quality(fill_credit)
    tradeable = fill_credit >= MIN_TRADEABLE_FILL

    return ProfitEstimate(
        side=side,
        spot=spot,
        short_strike=k_short,
        long_strike=k_long,
        width=width,
        buffer_pct=buffer * 100.0,
        realized_vol=realized_sigma,
        implied_vol=atm_iv,
        short_iv=sigma_s,
        long_iv=sigma_l,
        t_years=T,
        horizon_sessions=horizon_sessions,
        short_price=p_short,
        long_price=p_long,
        mid_credit=mid_credit,
        credit=fill_credit,
        bid_ask_estimate=bid_ask,
        max_loss=max_loss,
        return_on_risk=ror,
        annualized_ror=ann_ror,
        quality=quality,
        tradeable=tradeable,
    )
