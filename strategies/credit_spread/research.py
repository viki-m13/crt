"""
CreditFloor — Walk-Forward Conformal Strike Engine (put + call).

Proprietary approach — symmetric two-sided
------------------------------------------

We want to sell a credit spread whose short strike K sits as close as
possible to today's price S, but still "never breached" by the stock
over the next h trading days.

    Put side  (short K BELOW spot, bullish-bounded)
        buffer b_put(t,h)  = 1 - min(close[t+1..t+h]) / close[t]
        strike              = S * (1 - b_hat)
        win iff  min_fwd >= strike

    Call side (short K ABOVE spot, bearish-bounded)
        buffer b_call(t,h) = max(close[t+1..t+h]) / close[t] - 1
        strike              = S * (1 + b_hat)
        win iff  max_fwd <= strike

The walk-forward, conformal-buffer, purged-gap, regime-gated protocol
is identical for both sides; only the buffer definition, the strike
arithmetic, and the regime gate differ (put uses uptrend gate, call
uses below-SMA200 + no-recovery-rally gate). Everything else — folds,
safety margin, cap, warmup, leakage controls, pool validation — is
shared and defined once.

Algorithm (per ticker, per horizon h, per side):

    1. Split history into walk-forward folds by calendar year.
    2. For fold year Y:
        a. Training set = dates t with t+h < Jan 1 Y     (purged gap).
        b. Test set     = dates t with Jan 1 Y <= t < Jan 1 (Y+1)
                          and t+h <= last available date.
        c. b_hat_fold = max( b*(t,h)  over training set ) + SAFETY_EPS.
        d. For each test date t, WIN iff b*(t,h) <= b_hat_fold.
    3. Pool ALL fold test outcomes. A (ticker, side, horizon) combo is
       "CreditFloor eligible" iff:
        - every fold's test win rate = 100%, with >=1 win per fold
        - at least MIN_TEST_SAMPLES pooled test samples
        - final b_hat (fit on full history) <= MAX_BUFFER
    4. Live buffer b_final = max( b*(t,h) over ALL history ) + EPS
       >= every fold-time b_hat, so OOS win rate is a lower bound on
       what live would have seen.
    5. Regime variants: plain and regime-gated. Per-horizon we keep
       whichever *passes* validation with the smaller b_final. The
       regime variant is only deployable live when today's features
       also satisfy the gate.

Pool validation is done per side AND globally — any loss anywhere
fails the entire method (fail-closed).

Leakage controls (identical both sides)
---------------------------------------
    - Features at t use only close[0..t].
    - Targets use only (t, t+h]; never mixed into features.
    - Purged training: any sample whose forward window crosses into
      the test year is dropped from training.
    - Fixed safety margin (1%) and buffer cap (25%) — never tuned.
    - Variant choice (plain vs regime) is made on buffer size, not
      on win rate.
    - 252-day warmup per ticker.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from common import (
    FOLD_YEARS,
    HORIZONS,
    MAX_BUFFER,
    MIN_TEST_SAMPLES,
    MIN_TRAIN_SAMPLES,
    SAFETY_EPS,
    WARMUP_DAYS,
    Features,
    TickerSeries,
    _nyse_valid_days_big,
    actual_options_expiry,
    compute_features,
    covered_options_expiry,
    expiry_date,
    fold_mask,
    list_tickers,
    load_series,
    regime_mask,
    regime_mask_call,
    train_mask_for_fold,
    worst_buffer_path,
    worst_buffer_path_up,
)
from pricing import (
    IV_MULT,
    MIN_TRADEABLE_FILL,
    STRESS_IV_MULT,
    bs_call,
    bs_put,
    estimate_profit,
    iv_at_strike,
    realized_vol,
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- v3 publication layer ---------------------------
#
# "Sigma-Clear": the walk-forward conformal machinery below decides
# WHICH (ticker, side, horizon) combos are stable enough to consider
# (folds 100% OOS at the certified buffer, >=50 pooled tests, certified
# buffer <= 25%). The v3 layer then decides WHAT strike is actually
# published and whether the rung is tradeable at all. Frozen on the
# 2008-2018 design window of the deep daily replay and validated once,
# untouched, on 2019-2026 (see VALIDATION.md):
#
#   published buffer  b = K_SIGMA * sigma60_daily * sqrt(h) + 1%
#   gates:            b >= HIST_CLEAR * (worst h-day move in history)
#                     b <= 25% (h <= 21)  /  45% (h >= 42)
#                     expiry snapped DOWN to a covered standard expiry
#                     conservative net fill >= $0.05/share (after
#                       commissions, tenor haircut, bid-ask)
#                     underlying has listed options (optionable.json)
#                     >= ~10 years of listed price history
#                     series fresh (<= 5 sessions stale)
#
# Honest performance of this exact rule in the deep replay (1 contract
# per deduped trade, conservative fills): 2008-2018 design 0 losses /
# 554 trades; 2019-2026 validation 9 losses / 1,508 trades (99.4%),
# net P&L positive overall and in 7 of 8 validation years. This is NOT
# a 100% guarantee and is never published as one.
ENGINE_VERSION = "v3-sigmaclear"
K_SIGMA = 2.5
HIST_CLEAR = 0.8
CAP_SHORT = 0.25          # h <= 21
CAP_LONG = 0.45           # h >= 42
MIN_HISTORY_CAL_DAYS = 3652   # ~10 years listed
MAX_STALE_SESSIONS = 5

OPTIONABLE_PATH = os.path.join(OUTPUT_DIR, "optionable.json")

# Reality layer (reality.py): every rung that passes the model gates is
# verified against the ACTUAL listed options chain — real expiration
# inside the certified window, real strikes snapped in the safe
# direction, real bid/ask with open-interest floors, natural credit
# above the tradeability floor. Rungs that don't exist in reality are
# not published. CS_SKIP_REALITY=1 disables the layer (offline tests
# and replay work only — never production).
SKIP_REALITY = os.environ.get("CS_SKIP_REALITY") == "1"
_CHAIN_CACHE = None


def _chain_cache():
    global _CHAIN_CACHE
    if _CHAIN_CACHE is None:
        from reality import ChainCache
        _CHAIN_CACHE = ChainCache()
    return _CHAIN_CACHE


def load_optionable() -> dict[str, bool]:
    """Fail-closed map of ticker -> has listed options."""
    try:
        with open(OPTIONABLE_PATH) as fh:
            return json.load(fh).get("optionable", {})
    except (OSError, json.JSONDecodeError):
        print(f"[WARN] {OPTIONABLE_PATH} missing/unreadable — "
              "publishing NOTHING (fail-closed). Run fetch_optionable.py.",
              file=sys.stderr)
        return {}


def series_fresh(end_date: str) -> bool:
    """True iff end_date is within MAX_STALE_SESSIONS NYSE sessions of
    the latest completed session (delisted/stale series fail closed)."""
    import pandas as pd
    from datetime import datetime, timezone
    sessions = _nyse_valid_days_big()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    i_today = int(sessions.searchsorted(today, side="right")) - 1
    i_end = int(sessions.searchsorted(pd.Timestamp(end_date[:10]), side="right")) - 1
    return (i_today - i_end) <= MAX_STALE_SESSIONS


def v3_cap(h: int) -> float:
    return CAP_LONG if h >= 42 else CAP_SHORT


def v3_published_buffer(sigma: float | None, hist_max: float, h: int) -> float | None:
    """The Sigma-Clear published buffer, or None if the rung fails the
    structural gates (history clearance / cap)."""
    if sigma is None or sigma <= 0 or not np.isfinite(hist_max):
        return None
    b = K_SIGMA * (sigma / math.sqrt(252.0)) * math.sqrt(h) + 0.01
    if b < HIST_CLEAR * hist_max:
        return None
    if b > v3_cap(h):
        return None
    return b


class Side(str, Enum):
    PUT = "put"
    CALL = "call"


@dataclass
class FoldResult:
    year: int
    n_train: int
    n_test: int
    b_hat: float
    wins: int
    losses: int
    worst_test_buf: float


@dataclass
class VariantResult:
    variant: str               # "plain" or "regime"
    horizon: int
    side: str                  # "put" or "call"
    folds: list[FoldResult] = field(default_factory=list)
    pooled_wins: int = 0
    pooled_losses: int = 0
    b_final: float = 0.0
    b_final_history_max: float = 0.0
    eligible: bool = False


@dataclass
class SideResult:
    """Per-side output for a single ticker."""
    side: str
    best: dict[str, Any] | None = None
    all_eligible: list[dict[str, Any]] = field(default_factory=list)
    # Watchlist: (variant, horizon) combos whose backtest PASSES
    # (every fold 100%, final buf <= cap, enough samples) but whose
    # regime gate doesn't match TODAY, so we can't deploy. Sorted by
    # tightest buffer. Useful to surface "about to be dynamic" names
    # like AAPL waiting to break below SMA200.
    watchlist: list[dict[str, Any]] = field(default_factory=list)
    variants: dict[str, dict[int, VariantResult]] = field(default_factory=dict)


@dataclass
class TickerResult:
    ticker: str
    n_days: int
    start_date: str
    end_date: str
    today_close: float
    put: SideResult = field(default_factory=lambda: SideResult(side=Side.PUT.value))
    call: SideResult = field(default_factory=lambda: SideResult(side=Side.CALL.value))
    # Annualized stdev of daily log returns from the last 60 sessions.
    # None if insufficient data. Used only for profit estimates; the
    # walk-forward is independent of this.
    realized_vol: float | None = None


# ----------------------------- side-aware primitives -----------------------------


def _buffer_array(close: np.ndarray, h: int, side: Side) -> np.ndarray:
    if side == Side.PUT:
        return worst_buffer_path(close, h)
    return worst_buffer_path_up(close, h)


def _regime_mask(feats: Features, side: Side, require: bool) -> np.ndarray:
    if side == Side.PUT:
        return regime_mask(feats, require_uptrend=require)
    return regime_mask_call(feats, require_bearish=require)


def _today_regime_ok(feats: Features, side: Side) -> bool:
    if side == Side.PUT:
        return bool(
            np.isfinite(feats.trend[-1])
            and np.isfinite(feats.dd252[-1])
            and feats.trend[-1] >= 1.00
            and feats.dd252[-1] <= 0.20
        )
    return bool(
        np.isfinite(feats.trend[-1])
        and np.isfinite(feats.up252[-1])
        and feats.trend[-1] <= 1.00
        and feats.up252[-1] <= 0.20
    )


def _strike_from_buffer(spot: float, buffer: float, side: Side) -> float:
    if side == Side.PUT:
        return spot * (1.0 - buffer)
    return spot * (1.0 + buffer)


# ----------------------------- core evaluation -----------------------------


def evaluate_variant(
    ts: TickerSeries,
    feats: Features,
    horizon: int,
    regime: bool,
    side: Side,
) -> VariantResult:
    vr = VariantResult(
        variant="regime" if regime else "plain",
        horizon=horizon,
        side=side.value,
    )
    close = ts.close
    dates = ts.dates
    buf = _buffer_array(close, horizon, side)
    rmask_all = _regime_mask(feats, side, require=regime)
    warmup = np.zeros(len(dates), dtype=bool)
    warmup[WARMUP_DAYS:] = True
    base = rmask_all & warmup & np.isfinite(buf)

    for year in FOLD_YEARS:
        tr = base & train_mask_for_fold(dates, year, horizon)
        te = base & fold_mask(dates, year)
        n = len(dates)
        test_ok = np.zeros(n, dtype=bool)
        test_ok[: n - horizon] = True
        te = te & test_ok
        if tr.sum() < MIN_TRAIN_SAMPLES:
            continue
        if te.sum() == 0:
            continue
        b_train_worst = float(buf[tr].max())
        b_hat = min(max(b_train_worst + SAFETY_EPS, 0.0), 0.99)
        test_buf = buf[te]
        wins = int((test_buf <= b_hat).sum())
        losses = int((test_buf > b_hat).sum())
        vr.folds.append(
            FoldResult(
                year=year,
                n_train=int(tr.sum()),
                n_test=int(te.sum()),
                b_hat=b_hat,
                wins=wins,
                losses=losses,
                worst_test_buf=float(test_buf.max()),
            )
        )

    vr.pooled_wins = sum(f.wins for f in vr.folds)
    vr.pooled_losses = sum(f.losses for f in vr.folds)

    if base.sum() >= MIN_TRAIN_SAMPLES:
        vr.b_final_history_max = float(buf[base].max())
        vr.b_final = min(vr.b_final_history_max + SAFETY_EPS, 0.99)
    else:
        vr.b_final_history_max = float("nan")
        vr.b_final = float("nan")

    total = vr.pooled_wins + vr.pooled_losses
    vr.eligible = bool(
        vr.folds
        and vr.pooled_losses == 0
        and total >= MIN_TEST_SAMPLES
        and np.isfinite(vr.b_final)
        and vr.b_final <= MAX_BUFFER
        and all(f.losses == 0 for f in vr.folds)
        and all(f.wins > 0 for f in vr.folds)
    )
    return vr


def _vr_payload(vr: VariantResult, spot: float, side: Side, today_regime_ok: bool) -> dict[str, Any]:
    return {
        "side": side.value,
        "variant": vr.variant,
        "horizon": vr.horizon,
        "buffer": vr.b_final,
        "buffer_pct": vr.b_final * 100.0,
        "strike": _strike_from_buffer(spot, vr.b_final, side),
        "today_close": spot,
        "regime_ok_today": today_regime_ok,
        "pooled_wins": vr.pooled_wins,
        "pooled_losses": vr.pooled_losses,
        "n_test": vr.pooled_wins + vr.pooled_losses,
        "n_folds": len(vr.folds),
        "history_worst_buffer": vr.b_final_history_max,
    }


def process_ticker_side(
    ts: TickerSeries,
    feats: Features,
    side: Side,
) -> SideResult:
    spot = float(ts.close[-1])
    today_ok = _today_regime_ok(feats, side)
    sr = SideResult(side=side.value)

    best: tuple[float, VariantResult] | None = None
    per_horizon_best: dict[int, VariantResult] = {}
    # Watch: per horizon, track the tightest backtest-eligible combo
    # that was blocked only because today's regime gate doesn't match.
    # Keyed by horizon so each horizon contributes at most one watch.
    per_horizon_watch: dict[int, VariantResult] = {}
    for regime in (False, True):
        key = "regime" if regime else "plain"
        sr.variants[key] = {}
        for h in HORIZONS:
            vr = evaluate_variant(ts, feats, h, regime, side)
            sr.variants[key][h] = vr
            if not vr.eligible:
                continue
            if regime and not today_ok:
                # Backtest passes, but regime gate blocks deployment
                # today. Record for watchlist.
                prev = per_horizon_watch.get(h)
                if prev is None or vr.b_final < prev.b_final:
                    per_horizon_watch[h] = vr
                continue
            prev = per_horizon_best.get(h)
            if prev is None or vr.b_final < prev.b_final:
                per_horizon_best[h] = vr
            if best is None or vr.b_final < best[0]:
                best = (vr.b_final, vr)

    sr.all_eligible = [
        _vr_payload(per_horizon_best[h], spot, side, today_ok)
        for h in sorted(per_horizon_best.keys())
    ]
    # Only keep watch items when there's no deployable combo for that
    # horizon — avoid double-surfacing on the webapp.
    sr.watchlist = [
        _vr_payload(per_horizon_watch[h], spot, side, today_ok)
        for h in sorted(per_horizon_watch.keys())
        if h not in per_horizon_best
    ]
    if best is not None:
        sr.best = _vr_payload(best[1], spot, side, today_ok)
    return sr


def process_ticker(ticker: str) -> TickerResult | None:
    ts = load_series(ticker)
    if ts is None:
        return None
    feats = compute_features(ts.close)
    tr = TickerResult(
        ticker=ticker,
        n_days=len(ts.dates),
        start_date=str(ts.dates[0]),
        end_date=str(ts.dates[-1]),
        today_close=float(ts.close[-1]),
    )
    tr.put = process_ticker_side(ts, feats, Side.PUT)
    tr.call = process_ticker_side(ts, feats, Side.CALL)
    # Stash realized vol on the ticker result so the serializer can use
    # it for profit estimates without re-computing.
    tr.realized_vol = realized_vol(ts.close)
    return tr


# ----------------------------- serialization -----------------------------


def _profit_block(prof) -> dict[str, Any]:
    return {
        "realized_vol_pct":       prof.realized_vol * 100.0,
        "implied_vol_pct":        prof.implied_vol  * 100.0,
        "short_iv_pct":           prof.short_iv * 100.0,
        "long_iv_pct":            prof.long_iv  * 100.0,
        "short_strike":           prof.short_strike,
        "long_strike":            prof.long_strike,
        "spread_width":           prof.width,
        "mid_credit_per_share":   prof.mid_credit,
        "est_credit_per_share":   prof.credit,
        "net_credit_per_share":   prof.net_credit,
        "bid_ask_per_share":      prof.bid_ask_estimate,
        "est_max_loss_per_share": prof.max_loss,
        "return_on_risk_pct":     prof.return_on_risk * 100.0,
        "annualized_ror_pct":     prof.annualized_ror * 100.0,
        "credit_quality":         prof.quality,
        "tradeable":              prof.tradeable,
    }


def _rung_dict(
    r: TickerResult, sr: SideResult, e: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a full PUBLISHED rung record, applying the v3 Sigma-Clear
    publication layer. Returns None when the rung fails any publication
    gate (no covered expiry, history clearance, cap, or conservative
    net fill below the tradeable minimum)."""
    h = e["horizon"]
    hist_max = e["history_worst_buffer"]
    b_pub = v3_published_buffer(r.realized_vol, hist_max, h)
    if b_pub is None:
        return None
    snap = covered_options_expiry(r.end_date, h)
    if snap is None:
        return None
    exp_iso, kind, cal_days, sessions_to_exp = snap

    # Conservative fill at the PUBLISHED strike. Tradeability gate is
    # on the net (post-commission) credit.
    prof = estimate_profit(
        side=sr.side,
        spot=r.today_close,
        buffer=b_pub,
        horizon_sessions=h,
        realized_sigma=r.realized_vol,
        calendar_days_to_expiry=cal_days,
    )
    if prof is None or prof.net_credit < MIN_TRADEABLE_FILL:
        return None
    # Stress pricing: bare realized vol, zero volatility risk premium.
    stress = estimate_profit(
        side=sr.side,
        spot=r.today_close,
        buffer=b_pub,
        horizon_sessions=h,
        realized_sigma=r.realized_vol,
        calendar_days_to_expiry=cal_days,
        iv_mult=STRESS_IV_MULT,
    )

    # Optional crash-wing overlay (VALIDATION.md §7): half a unit of a
    # long option one width past the long leg, same expiry — attached
    # only when the net credit still clears the tradeability floor
    # after paying for it. Cuts the worst case ~29% and makes deep
    # breaches convex (a COVID-size move flips to a profit) for ~7.5%
    # of P&L; design-window zero-loss invariant preserved.
    wing = None
    wing_ratio = 0.5
    k_wing = (prof.long_strike - prof.width if sr.side == "put"
              else prof.long_strike + prof.width)
    if k_wing > 0:
        T_yrs = cal_days / 365.0
        atm_iv = r.realized_vol * IV_MULT
        iv_w = iv_at_strike(r.today_close, k_wing, T_yrs, atm_iv, sr.side)
        mid_w = (bs_put(r.today_close, k_wing, T_yrs, iv_w) if sr.side == "put"
                 else bs_call(r.today_close, k_wing, T_yrs, iv_w))
        ask_w = mid_w + max(0.05, 0.10 * mid_w) / 2.0 + 0.0066
        net_after = prof.net_credit - wing_ratio * ask_w
        if net_after >= MIN_TRADEABLE_FILL:
            wing = {
                "strike": k_wing,
                "ratio": wing_ratio,
                "est_cost_per_share": wing_ratio * ask_w,
                "net_credit_after_wing": net_after,
            }

    spot = r.today_close
    strike = spot * (1.0 - b_pub) if sr.side == "put" else spot * (1.0 + b_pub)

    # Reality verification: the rung must exist on the live chain with
    # a collectible natural credit, or it is not published.
    real = None
    if not SKIP_REALITY:
        from dataclasses import asdict as _asdict
        from reality import verify_rung
        rs = verify_rung(
            _chain_cache(), r.ticker, sr.side, spot, r.end_date, h,
            prof.short_strike, prof.long_strike, k_wing if k_wing > 0 else None,
            min_net=MIN_TRADEABLE_FILL,
        )
        if rs is None:
            return None
        real = _asdict(rs)

    # When the reality layer is active, the PUBLISHED contract is the
    # real one (real expiration, real short strike — snapped in the
    # safe direction, so the real cushion >= the certified one). The
    # model values stay alongside for transparency. The live log
    # records these primary fields, so it tracks the real contract.
    if real is not None:
        pub_strike = real["short_strike"]
        pub_expiry = real["expiry"]
        pub_cal = real["cal_days_to_expiry"]
        pub_sess = real["sessions_to_expiry"]
        pub_buffer_pct = real["real_buffer_pct"]
        pub_kind = kind
    else:
        pub_strike, pub_expiry = strike, exp_iso
        pub_cal, pub_sess = cal_days, sessions_to_exp
        pub_buffer_pct, pub_kind = b_pub * 100.0, kind

    base = {
        "engine": ENGINE_VERSION,
        "horizon": h,
        "expiry_date": pub_expiry,
        "expiry_type": pub_kind,
        "calendar_days_to_expiry": pub_cal,
        "sessions_to_expiry": pub_sess,
        # Published strike — the contract a user would actually trade.
        "strike": pub_strike,
        "buffer_pct": pub_buffer_pct,
        "model_strike": strike,
        "model_expiry_date": exp_iso,
        "model_buffer_pct": b_pub * 100.0,
        # Certified conformal stats for transparency: the walk-forward
        # machinery's own never-breached buffer and strike.
        "certified_buffer_pct": e["buffer_pct"],
        "certified_strike": e["strike"],
        "history_worst_buffer_pct": hist_max * 100.0,
        "sigma_distance": K_SIGMA,
        "variant": e["variant"],
        "n_test": e["n_test"],
        "n_folds": e["n_folds"],
        "pooled_wins": e["pooled_wins"],
        "pooled_losses": e["pooled_losses"],
        "folds": [
            {
                "year": f.year,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "b_hat_pct": f.b_hat * 100.0,
                "wins": f.wins,
                "losses": f.losses,
                "worst_test_buf_pct": f.worst_test_buf * 100.0,
            }
            for f in sr.variants[e["variant"]][e["horizon"]].folds
        ],
        "profit": _profit_block(prof),
        "stress_profit": _profit_block(stress) if stress is not None else None,
        # With the reality layer active, the wing quote comes from the
        # real chain too (or is absent when real quotes can't pay for it).
        "crash_wing": (real["wing"] if real is not None else wing),
        # Verified live-chain contract (real expiration, real strikes,
        # real quotes; the natural credit is the collectible one).
        "real": real,
    }
    return base


def _signal_payload(r: TickerResult, sr: SideResult) -> dict[str, Any] | None:
    """Assemble the published signal for one (ticker, side): the ladder
    of rungs that clear the v3 publication layer. Returns None when no
    rung clears (the combo stays certified-but-unpublishable)."""
    assert sr.best is not None
    ladder = [d for d in (_rung_dict(r, sr, e) for e in sr.all_eligible)
              if d is not None]
    if not ladder:
        return None
    # Primary pick: highest net return-on-risk among published rungs
    # (real natural-credit ROR when the reality layer priced the rung).
    def _ror(d):
        if d.get("real"):
            return d["real"]["ror_natural"] * 100.0
        return d["profit"]["return_on_risk_pct"]
    primary_rung = max(ladder, key=_ror)
    return {
        "ticker": r.ticker,
        "today_close": r.today_close,
        "end_date": r.end_date,
        "side": sr.side,
        "engine": ENGINE_VERSION,
        "realized_vol_pct": (r.realized_vol * 100.0) if r.realized_vol else None,
        # primary (highest net-ROR) pick
        "strike": primary_rung["strike"],
        "buffer_pct": primary_rung["buffer_pct"],
        "certified_buffer_pct": primary_rung["certified_buffer_pct"],
        "horizon": primary_rung["horizon"],
        "expiry_date": primary_rung["expiry_date"],
        "expiry_type": primary_rung["expiry_type"],
        "calendar_days_to_expiry": primary_rung["calendar_days_to_expiry"],
        "variant": primary_rung["variant"],
        "n_test": primary_rung["n_test"],
        "n_folds": primary_rung["n_folds"],
        "pooled_wins": primary_rung["pooled_wins"],
        "pooled_losses": primary_rung["pooled_losses"],
        "history_worst_buffer_pct": primary_rung["history_worst_buffer_pct"],
        "regime_ok_today": sr.best["regime_ok_today"],
        "profit": primary_rung["profit"],
        "stress_profit": primary_rung["stress_profit"],
        "real": primary_rung.get("real"),
        "folds": primary_rung["folds"],
        "ladder": ladder,
    }


def _pool_validate(eligibles: list[tuple[TickerResult, SideResult]]) -> tuple[int, int]:
    """Pool every eligible (ticker, side) OOS sample; return (wins, losses)."""
    wins = losses = 0
    for r, sr in eligibles:
        assert sr.best is not None
        v = sr.best["variant"]
        h = sr.best["horizon"]
        vr = sr.variants[v][h]
        for f in vr.folds:
            wins += f.wins
            losses += f.losses
    return wins, losses


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]

    optionable = load_optionable()

    t0 = time.time()
    results: list[TickerResult] = []
    put_elig: list[tuple[TickerResult, SideResult]] = []
    call_elig: list[tuple[TickerResult, SideResult]] = []
    put_watch: list[tuple[TickerResult, SideResult]] = []
    call_watch: list[tuple[TickerResult, SideResult]] = []
    n_not_optionable = n_stale = n_young = 0
    for i, t in enumerate(tickers, 1):
        # Ticker-level publication gates (fail-closed):
        #   1. underlying must have a listed options chain;
        if not optionable.get(t, False):
            n_not_optionable += 1
            continue
        try:
            r = process_ticker(t)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] {t}: {exc}", file=sys.stderr)
            continue
        if r is None:
            continue
        #   2. series must be fresh (delisted/stale series fail closed);
        if not series_fresh(r.end_date):
            n_stale += 1
            continue
        #   3. >= ~10 years of listed history (a conformal max learned
        #      on a bull-market-only IPO history is not trustworthy).
        if (np.datetime64(r.end_date[:10], "D")
                - np.datetime64(r.start_date[:10], "D")).astype(int) < MIN_HISTORY_CAL_DAYS:
            n_young += 1
            continue
        results.append(r)
        if r.put.best is not None:
            put_elig.append((r, r.put))
        elif r.put.watchlist:
            put_watch.append((r, r.put))
        if r.call.best is not None:
            call_elig.append((r, r.call))
        elif r.call.watchlist:
            call_watch.append((r, r.call))
        if i % 100 == 0:
            print(
                f"  {i}/{len(tickers)}  puts={len(put_elig)}  "
                f"calls={len(call_elig)}  "
                f"watch(put/call)={len(put_watch)}/{len(call_watch)}  "
                f"elapsed={time.time()-t0:.1f}s"
            )

    put_wins, put_losses = _pool_validate(put_elig)
    call_wins, call_losses = _pool_validate(call_elig)
    pool_wins = put_wins + call_wins
    pool_losses = put_losses + call_losses

    # Sort: tightest certified buffer first, tie-break on larger sample.
    put_elig.sort(key=lambda x: (x[1].best["buffer"], -x[1].best["n_test"]))
    call_elig.sort(key=lambda x: (x[1].best["buffer"], -x[1].best["n_test"]))
    # Watchlists sort by tightest buffer too.
    put_watch.sort(key=lambda x: x[1].watchlist[0]["buffer"])
    call_watch.sort(key=lambda x: x[1].watchlist[0]["buffer"])

    # Apply the v3 publication layer: only signals with >=1 rung that
    # clears every gate (history clearance, cap, covered expiry,
    # conservative net fill) are published.
    put_signals = [p for p in (_signal_payload(r, sr) for r, sr in put_elig)
                   if p is not None]
    call_signals = [p for p in (_signal_payload(r, sr) for r, sr in call_elig)
                    if p is not None]

    print()
    print(f"Tickers processed:       {len(results)}  "
          f"(skipped: {n_not_optionable} non-optionable, {n_stale} stale, "
          f"{n_young} <10y history)")
    print(f"Put-side certified:      {len(put_elig)}  "
          f"({put_wins}/{put_wins+put_losses} OOS, losses={put_losses})")
    print(f"Call-side certified:     {len(call_elig)} "
          f"({call_wins}/{call_wins+call_losses} OOS, losses={call_losses})")
    print(f"Published (v3 layer):    puts={len(put_signals)} calls={len(call_signals)}")
    reality_summary = None
    if not SKIP_REALITY and _CHAIN_CACHE is not None:
        nf = len(_CHAIN_CACHE.failures)
        reality_summary = {
            "chain_fetch_failures": nf,
            "drops": dict(sorted(_CHAIN_CACHE.drops.items())),
        }
        print(f"Reality layer:           chain fetch failures={nf} "
              f"drops={reality_summary['drops']}")
        for f in _CHAIN_CACHE.failures[:10]:
            print(f"  [chain] {f}", file=sys.stderr)
        if nf and not (put_signals or call_signals):
            print("WARNING: zero signals AND chain failures — Yahoo options "
                  "API may be down; treat today's empty book as unverified.",
                  file=sys.stderr)

    lean = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": ENGINE_VERSION,
        "summary": {
            "n_tickers_processed": len(results),
            "horizons": HORIZONS,
            "fold_years": FOLD_YEARS,
            "safety_eps": SAFETY_EPS,
            "max_buffer": MAX_BUFFER,
            "engine": ENGINE_VERSION,
            "v3_layer": {
                "k_sigma": K_SIGMA,
                "hist_clear": HIST_CLEAR,
                "cap_short": CAP_SHORT,
                "cap_long": CAP_LONG,
                "min_net_fill_per_share": MIN_TRADEABLE_FILL,
                "min_history_cal_days": MIN_HISTORY_CAL_DAYS,
            },
            # Why model-passing rungs were dropped by live-chain
            # verification today (None when the layer was skipped).
            "reality": reality_summary,
            # Honest, replay-derived expectations for this exact rule
            # set (deduped independent trades, conservative fills; see
            # strategies/credit_spread/VALIDATION.md). The fold stats
            # below describe the conformal certification machinery,
            # NOT a forward win-rate claim.
            "validated": {
                "design_window": "2008-2018",
                "design_trades": 554,
                "design_losses": 0,
                "validation_window": "2019-2026",
                "validation_trades": 1508,
                "validation_losses": 9,
                "validation_win_rate": 0.994,
                "note": ("Win rate is NOT 100%. Residual tail risk is "
                          "real: systemic crashes (Mar 2020) and "
                          "single-name events (M&A, earnings) can and "
                          "do breach published strikes ~0.5% of the "
                          "time out-of-sample. Defined-risk spreads "
                          "cap each loss at width minus credit."),
            },
            "put": {
                "n_certified": len(put_elig),
                "n_published": len(put_signals),
                "pooled_wins": put_wins,
                "pooled_losses": put_losses,
                "pooled_win_rate": put_wins/(put_wins+put_losses) if put_wins+put_losses else None,
            },
            "call": {
                "n_certified": len(call_elig),
                "n_published": len(call_signals),
                "pooled_wins": call_wins,
                "pooled_losses": call_losses,
                "pooled_win_rate": call_wins/(call_wins+call_losses) if call_wins+call_losses else None,
            },
            "combined": {
                "n_certified": len(put_elig) + len(call_elig),
                "n_published": len(put_signals) + len(call_signals),
                "pooled_wins": pool_wins,
                "pooled_losses": pool_losses,
                "pooled_win_rate": pool_wins/(pool_wins+pool_losses) if pool_wins+pool_losses else None,
            },
        },
        "put_signals":  put_signals,
        "call_signals": call_signals,
        # Watchlist: tickers whose backtest passes but today's regime
        # gate fails. Surface so users can see what's about to turn on.
        "put_watchlist": [
            {
                "ticker": r.ticker,
                "today_close": r.today_close,
                "end_date": r.end_date,
                "rungs": [
                    {**e,
                     "expiry_date": actual_options_expiry(r.end_date, e["horizon"])[0],
                     "expiry_type": actual_options_expiry(r.end_date, e["horizon"])[1]}
                    for e in sr.watchlist
                ],
            }
            for r, sr in put_watch
        ],
        "call_watchlist": [
            {
                "ticker": r.ticker,
                "today_close": r.today_close,
                "end_date": r.end_date,
                "rungs": [
                    {**e,
                     "expiry_date": actual_options_expiry(r.end_date, e["horizon"])[0],
                     "expiry_type": actual_options_expiry(r.end_date, e["horizon"])[1]}
                    for e in sr.watchlist
                ],
            }
            for r, sr in call_watch
        ],
    }
    lean_path = os.path.join(OUTPUT_DIR, "signals.json")
    with open(lean_path, "w") as fh:
        json.dump(lean, fh, indent=2)
    print(f"Wrote {lean_path}  "
          f"(published puts={len(put_signals)}, calls={len(call_signals)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
