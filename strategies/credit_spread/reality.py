"""Reality layer — verify every published rung against the ACTUAL
listed options chain before publication.

The price-history engine works on modeled contracts: theoretical
Friday expirations, exact-dollar strikes (e.g. $266.18), Black-Scholes
credits. Real chains are discrete and thinner: many names have only
monthly expirations, strikes come in $0.50/$1/$2.50/$5 increments, and
far-OTM options often have a ZERO bid — meaning the modeled credit is
not collectible at all. This module closes that gap, fail-closed:

  1. Expiry: the ticker's real listed expirations (yfinance) are
     snapped DOWN — the latest expiration within the certified
     h-session window. No listed expiration inside the window -> the
     rung is dropped.
  2. Strikes: snapped to real listed strikes in the SAFE direction
     (put short strike rounds down, call short strike rounds up, so
     the real cushion is always >= the certified one). The long leg
     goes to the listed strike nearest the model's protection level.
  3. Pricing: from the real quotes. The published credit is the
     NATURAL credit — sell at the short leg's bid, buy at the long
     leg's ask — i.e. the fill you can get without negotiating,
     minus commissions. Mid is reported alongside.
  4. Liquidity gates: short bid > 0, long ask > 0, open interest >=
     MIN_OI on both legs, natural net credit >= the tradeability
     floor.

Anything that cannot be verified (chain fetch failure, no expiration
in window, zero bid, thin OI, credit below floor) is NOT published.
If Yahoo's options API is down entirely, the scan publishes nothing
and says so — a day with no signals is a result; a signal that does
not exist in reality is a bug.
"""
from __future__ import annotations

import bisect
import os
import time
from dataclasses import dataclass

import numpy as np

from common import _nyse_valid_days_big

# Liquidity gates. These are the difference between "a contract exists"
# and "you can actually trade it near the quoted credit". Tunable via
# env for experiments; defaults chosen to keep real S&P-500-caliber
# names while removing thin ones and phantom quotes.
MIN_OI = int(os.environ.get("CS_MIN_OI", "25"))          # open interest per leg
# Short-leg bid/ask spread must be a small fraction of the spread width
# — a wide short-leg market means the quoted bid is unreliable/stale and
# the natural credit will not fill.
MAX_SHORT_SPREAD_FRAC = float(os.environ.get("CS_MAX_SHORT_SPREAD_FRAC", "0.40"))
# Underlying average daily dollar volume floor (from results/adv.json).
# 0 disables the underlying gate (e.g. if the ADV map is unavailable).
# $100M/day = "very liquid". The liquidity-impact study (VALIDATION.md
# §17) shows filtering to this floor holds accuracy and ROR on both
# tiers (and both improve slightly vs no filter), so it is effectively
# free — hence the default sits here rather than at a permissive $50M.
MIN_ADV_USD = float(os.environ.get("CS_MIN_ADV_USD", str(100e6)))
CHAIN_RETRIES = 3
COMMISSION_PER_SHARE = 0.0132


@dataclass
class RealSpread:
    expiry: str                # real listed expiration (YYYY-MM-DD)
    sessions_to_expiry: int
    cal_days_to_expiry: int
    short_strike: float        # real listed strikes
    long_strike: float
    width: float
    short_bid: float
    short_ask: float
    long_bid: float
    long_ask: float
    short_oi: int
    long_oi: int
    short_spread_frac: float   # (short_ask-short_bid)/width
    adv_usd: float             # underlying avg daily $ volume (0 if unknown)
    natural_credit: float      # short_bid - long_ask (per share)
    mid_credit: float
    net_natural_credit: float  # natural - commissions
    max_loss: float            # width - natural_credit
    ror_natural: float         # net_natural / max_loss
    real_buffer_pct: float     # cushion at the REAL short strike
    wing: dict | None          # real crash-wing quote, when affordable
    quote_time: str


def _sessions_between(d0: str, d1: str) -> int:
    import pandas as pd
    sessions = _nyse_valid_days_big()
    i0 = int(sessions.searchsorted(pd.Timestamp(d0), side="left"))
    i1 = int(sessions.searchsorted(pd.Timestamp(d1), side="right")) - 1
    return i1 - i0


class ChainCache:
    """Per-ticker cached access to the real options chain."""

    def __init__(self) -> None:
        self._tickers: dict[str, object] = {}
        self._expirations: dict[str, list[str]] = {}
        self._chains: dict[tuple[str, str], object] = {}
        self.failures: list[str] = []
        # why model-passing rungs were dropped by reality verification
        self.drops: dict[str, int] = {}
        # underlying average daily $ volume map (results/adv.json)
        self.adv: dict[str, float] = {}
        _adv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "results", "adv.json")
        try:
            import json as _json
            self.adv = _json.load(open(_adv_path)).get("adv_usd", {})
        except Exception:  # noqa: BLE001
            self.adv = {}

    def drop(self, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1

    def _t(self, ticker: str):
        import yfinance as yf
        if ticker not in self._tickers:
            self._tickers[ticker] = yf.Ticker(ticker)
        return self._tickers[ticker]

    def expirations(self, ticker: str) -> list[str] | None:
        if ticker in self._expirations:
            return self._expirations[ticker]
        last = None
        for attempt in range(CHAIN_RETRIES):
            try:
                exps = list(self._t(ticker).options)
                self._expirations[ticker] = exps
                return exps
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(1.5 * (attempt + 1))
        self.failures.append(f"{ticker}: expirations: {last}")
        return None

    def chain(self, ticker: str, expiry: str, side: str):
        """Return the strike-sorted puts/calls DataFrame or None."""
        key = (ticker, expiry)
        if key not in self._chains:
            last = None
            for attempt in range(CHAIN_RETRIES):
                try:
                    self._chains[key] = self._t(ticker).option_chain(expiry)
                    last = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    time.sleep(1.5 * (attempt + 1))
            if last is not None:
                self.failures.append(f"{ticker} {expiry}: chain: {last}")
                self._chains[key] = None
        ch = self._chains[key]
        if ch is None:
            return None
        tab = ch.puts if side == "put" else ch.calls
        return tab.sort_values("strike").reset_index(drop=True)


def _leg(tab, strike: float) -> dict | None:
    row = tab[tab["strike"] == strike]
    if row.empty:
        return None
    r = row.iloc[0]
    def f(x):
        v = r.get(x)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if v != v else v  # NaN -> 0
    return {"bid": f("bid"), "ask": f("ask"),
            "oi": int(f("openInterest")), "vol": int(f("volume"))}


def verify_rung(cache: ChainCache, ticker: str, side: str, spot: float,
                publish_date: str, horizon_sessions: int,
                model_short: float, model_long: float,
                wing_model_strike: float | None,
                min_net: float = 0.05) -> RealSpread | None:
    """Snap a model rung onto the real chain; None if it doesn't exist
    in reality with a collectible, liquid credit."""
    # Underlying liquidity floor (coarse): structurally thin names have
    # no continuous options market regardless of listed strikes. Only
    # gated when the ADV map is present and has this ticker.
    adv = float(cache.adv.get(ticker, 0.0))
    if MIN_ADV_USD > 0 and cache.adv and ticker in cache.adv and adv < MIN_ADV_USD:
        cache.drop("underlying_adv_below_floor")
        return None
    exps = cache.expirations(ticker)
    if not exps:
        cache.drop("chain_unavailable")
        return None
    # latest listed expiration within the certified window
    best = None
    for e in exps:
        if e <= publish_date:
            continue
        sess = _sessions_between(publish_date, e)
        if 0 < sess <= horizon_sessions:
            if best is None or e > best[0]:
                best = (e, sess)
    if best is None:
        cache.drop("no_listed_expiration_in_window")
        return None
    expiry, sessions = best
    tab = cache.chain(ticker, expiry, side)
    if tab is None or tab.empty:
        cache.drop("chain_unavailable")
        return None
    strikes = tab["strike"].tolist()

    # short strike: safe direction (put rounds down, call rounds up)
    if side == "put":
        i = bisect.bisect_right(strikes, model_short) - 1
    else:
        i = bisect.bisect_left(strikes, model_short)
    if i < 0 or i >= len(strikes):
        cache.drop("no_listed_strike_near_model")
        return None
    ks = float(strikes[i])
    # long leg: listed strike nearest the model's protection level,
    # never equal to (or on the wrong side of) the short strike
    j = min(range(len(strikes)), key=lambda k: abs(strikes[k] - model_long))
    kl = float(strikes[j])
    if side == "put" and kl >= ks:
        if i - 1 < 0:
            cache.drop("no_listed_strike_near_model")
            return None
        kl = float(strikes[i - 1])
    if side == "call" and kl <= ks:
        if i + 1 >= len(strikes):
            cache.drop("no_listed_strike_near_model")
            return None
        kl = float(strikes[i + 1])

    s_leg = _leg(tab, ks)
    l_leg = _leg(tab, kl)
    if s_leg is None or l_leg is None:
        cache.drop("no_listed_strike_near_model")
        return None
    # liquidity gates
    if s_leg["bid"] <= 0 or l_leg["ask"] <= 0:
        cache.drop("zero_bid_or_no_ask")
        return None
    if s_leg["oi"] < MIN_OI or l_leg["oi"] < MIN_OI:
        cache.drop("open_interest_below_min")
        return None

    width = abs(ks - kl)
    # Short-leg market must be tight relative to the spread width — a
    # wide short-leg quote means the bid the natural credit relies on is
    # unreliable/stale (phantom liquidity).
    short_spread = s_leg["ask"] - s_leg["bid"]
    short_spread_frac = short_spread / width if width > 0 else 9.99
    if short_spread_frac > MAX_SHORT_SPREAD_FRAC:
        cache.drop("short_leg_spread_too_wide")
        return None

    natural = s_leg["bid"] - l_leg["ask"]
    mid = (s_leg["bid"] + s_leg["ask"]) / 2 - (l_leg["bid"] + l_leg["ask"]) / 2
    net = natural - COMMISSION_PER_SHARE
    if net < min_net or width <= 0:
        cache.drop("natural_credit_below_floor")
        return None
    max_loss = max(width - natural, 0.01)

    # crash wing on the real chain (optional; never blocks the rung)
    wing = None
    if wing_model_strike is not None and wing_model_strike > 0:
        jw = min(range(len(strikes)), key=lambda k: abs(strikes[k] - wing_model_strike))
        kw = float(strikes[jw])
        on_safe_side = kw < kl if side == "put" else kw > kl
        w_leg = _leg(tab, kw) if on_safe_side else None
        if w_leg is not None and w_leg["ask"] > 0:
            cost = 0.5 * w_leg["ask"] + 0.0066
            if net - cost >= min_net:
                wing = {"strike": kw, "ratio": 0.5, "ask": w_leg["ask"],
                        "oi": w_leg["oi"], "est_cost_per_share": cost,
                        "net_credit_after_wing": net - cost}

    import pandas as pd
    cal_days = int((pd.Timestamp(expiry) - pd.Timestamp(publish_date)).days)
    buffer_pct = (spot - ks) / spot * 100.0 if side == "put" else (ks - spot) / spot * 100.0
    return RealSpread(
        expiry=expiry, sessions_to_expiry=sessions, cal_days_to_expiry=cal_days,
        short_strike=ks, long_strike=kl, width=width,
        short_bid=s_leg["bid"], short_ask=s_leg["ask"],
        long_bid=l_leg["bid"], long_ask=l_leg["ask"],
        short_oi=s_leg["oi"], long_oi=l_leg["oi"],
        short_spread_frac=round(short_spread_frac, 4), adv_usd=adv,
        natural_credit=natural, mid_credit=mid, net_natural_credit=net,
        max_loss=max_loss, ror_natural=net / max_loss,
        real_buffer_pct=buffer_pct, wing=wing,
        quote_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
