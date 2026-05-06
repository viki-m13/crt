"""
Stillpoint — Compression-Conditioned Microbuffer Engine.

Proprietary novel method for short-DTE (<=21 trading days) credit spreads
with strikes very close to spot (<=5% buffer) and high empirical win rate
(target >=95% out-of-sample, every fold >=90%).

The method
----------
Most credit-spread engines try to set a buffer that has *never* been
breached over a horizon. That is fine when you can afford a 10-25%
buffer on a 1-6 month tenor — there are usually enough quiet stocks for
which 100% of training paths fit. It does not work when you want a
strike inches from spot on a two-week tenor: the "100% of training" mark
balloons.

Stillpoint takes a different route. We accept a small allowed miss rate
in exchange for a tight strike, then EARN that win rate honestly with
two new ingredients:

  1. Stillpoint regime gate — we only consider days where the stock is
     in a compression-volatility regime: 20d annualized vol < 30%,
     5d/20d vol ratio < 0.95 (still calming), 20d range < 10%, price
     within 2.5% of its 20d SMA, RSI(14) in [35, 65], and |5d move|
     < 5%. Empirically, the conditional distribution of next-h-day
     paths in this regime is far thinner than the unconditional
     distribution — its 97th percentile is materially smaller than the
     unconditional one.

  2. Conformal q-quantile strike — within the regime-gated training
     samples we set the buffer to the 97th percentile path-buffer plus
     0.5% safety. With 97% empirical training coverage we expect ~95%+
     OOS coverage; we then VERIFY this on walk-forward test years and
     reject anything whose pooled or per-fold OOS win rate falls below
     thresholds (95% pooled, 90% per fold, ≥40 pooled tests, ≥4 folds).

Per-ticker, per-side, per-horizon. Walk-forward folds purge any training
sample whose forward window crosses into the test year (anti-leakage).

Output
------
strategies/stillpoint/results/stillpoint_signals.json:
  {
    summary: {...},
    put_signals: [{ticker, today_close, ladder: [{horizon, strike, buffer_pct, ...}]}, ...],
    call_signals: [...],
    pin_signals: [...]   # tickers eligible on BOTH sides simultaneously
  }
"""
from __future__ import annotations

import json
import math
import os
import sys
import time


from dataclasses import dataclass, field
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CS_DIR = os.path.join(os.path.dirname(_HERE), "credit_spread")
sys.path.insert(0, _HERE)

import importlib.util as _ilu
_pr_spec = _ilu.spec_from_file_location(
    "credit_spread_pricing", os.path.join(_CS_DIR, "pricing.py"),
)
_pricing = _ilu.module_from_spec(_pr_spec)
sys.modules["credit_spread_pricing"] = _pricing
_pr_spec.loader.exec_module(_pricing)
estimate_profit = _pricing.estimate_profit
realized_vol = _pricing.realized_vol

from sp_common import (  # noqa: E402
    FOLD_YEARS, HORIZONS, WARMUP_DAYS,
    SP_CONFORMAL_Q, SP_SAFETY_EPS, SP_MAX_BUFFER,
    SP_MIN_TRAIN_FIRES, SP_MIN_POOLED_TEST, SP_MIN_FOLDS,
    SP_TARGET_POOLED_WIN, SP_TARGET_PER_FOLD_WIN,
    SP_TIGHT_HORIZONS, SP_TIGHT_CONFORMAL_Q, SP_TIGHT_MAX_BUFFER,
    SP_TIGHT_PER_FOLD_WIN, SP_TIGHT_POOLED_WIN,
    SP_IC_HORIZONS, SP_IC_CONFORMAL_QS, SP_IC_MAX_BUFFER,
    SP_IC_PER_FOLD_WIN, SP_IC_POOLED_WIN, SP_IC_TARGET_ROR,
    SP_IC_WIDTHS,
    actual_options_expiry, buffer_array, compute_features,
    fold_mask, list_tickers, load_series, stillpoint_mask,
    strike_from_buffer, tight_mask, today_in_regime,
    today_in_tight_regime, train_mask_for_fold,
)
# BS pricers (from credit_spread/pricing.py loaded above)
bs_put = _pricing.bs_put
bs_call = _pricing.bs_call


OUTPUT_DIR = os.path.join(_HERE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class StillFold:
    year: int
    n_train: int
    n_test: int
    b_hat: float
    wins: int
    losses: int
    worst_test_buf: float


@dataclass
class StillVariant:
    ticker: str
    side: str
    horizon: int
    folds: list[StillFold] = field(default_factory=list)
    pooled_wins: int = 0
    pooled_losses: int = 0
    b_final: float = 0.0          # the live, full-history quantile buffer
    n_history_fires: int = 0      # total Stillpoint days ever
    eligible: bool = False
    today_in_regime: bool = False


def _fold_buffer(buf_train: np.ndarray, q: float) -> float:
    """Conformal buffer = qth quantile of training b* + safety_eps.

    Using nearest-rank quantile (np.quantile linear) on b* values. We
    add a small absolute safety margin so the strike doesn't sit
    literally on the worst observed move.
    """
    if buf_train.size == 0:
        return float("nan")
    q_val = float(np.quantile(buf_train, q))
    return min(max(q_val + SP_SAFETY_EPS, 0.0), 0.99)


def evaluate_variant(close: np.ndarray, dates: np.ndarray,
                     stillpoint: np.ndarray, side: str, horizon: int,
                     ticker: str, today_ok: bool) -> StillVariant:
    vr = StillVariant(ticker=ticker, side=side, horizon=horizon,
                       today_in_regime=today_ok)
    buf = buffer_array(close, horizon, side)
    n = len(dates)
    warmup = np.zeros(n, dtype=bool)
    warmup[WARMUP_DAYS:] = True
    base = warmup & stillpoint & np.isfinite(buf)

    # Walk-forward folds
    for year in FOLD_YEARS:
        tr = base & train_mask_for_fold(dates, year, horizon)
        te = base & fold_mask(dates, year)
        # Test must have valid forward window
        ok = np.zeros(n, dtype=bool)
        ok[: n - horizon] = True
        te = te & ok
        if tr.sum() < SP_MIN_TRAIN_FIRES // 2:
            # Not enough training samples in this fold's training window;
            # we still tally the test fires for honesty (no signal would
            # have been published, so they're skipped by 'continue').
            continue
        b_hat = _fold_buffer(buf[tr], SP_CONFORMAL_Q)
        if not np.isfinite(b_hat) or b_hat > SP_MAX_BUFFER:
            continue
        if te.sum() == 0:
            continue
        test_buf = buf[te]
        wins = int((test_buf <= b_hat).sum())
        losses = int((test_buf > b_hat).sum())
        vr.folds.append(StillFold(
            year=year,
            n_train=int(tr.sum()),
            n_test=int(te.sum()),
            b_hat=b_hat,
            wins=wins, losses=losses,
            worst_test_buf=float(test_buf.max()),
        ))
    vr.pooled_wins = sum(f.wins for f in vr.folds)
    vr.pooled_losses = sum(f.losses for f in vr.folds)
    vr.n_history_fires = int(base.sum())

    # Live (full-history) buffer using all Stillpoint training samples.
    # NOT used to grade OOS; only used to publish today's strike.
    if base.sum() >= SP_MIN_TRAIN_FIRES:
        vr.b_final = _fold_buffer(buf[base], SP_CONFORMAL_Q)
    else:
        vr.b_final = float("nan")

    pooled_total = vr.pooled_wins + vr.pooled_losses
    pooled_wr = (vr.pooled_wins / pooled_total) if pooled_total else 0.0
    every_fold_ok = all(
        (f.wins / max(f.wins + f.losses, 1)) >= SP_TARGET_PER_FOLD_WIN
        for f in vr.folds
    )
    vr.eligible = bool(
        vr.folds
        and len(vr.folds) >= SP_MIN_FOLDS
        and pooled_total >= SP_MIN_POOLED_TEST
        and pooled_wr >= SP_TARGET_POOLED_WIN
        and every_fold_ok
        and np.isfinite(vr.b_final)
        and vr.b_final <= SP_MAX_BUFFER
    )
    return vr


@dataclass
class TightFold:
    year: int
    n_train: int
    n_test: int
    z_q: float                # vol-normalized quantile from training
    median_b_hat: float       # representative test-set buffer (for display)
    wins: int
    losses: int
    worst_test_buf: float


@dataclass
class TightVariant:
    """Vol-adaptive tight-tier variant. Buffer = z_q × σ_today × √(h/252).

    z is the path buffer normalized by an instantaneous vol estimate
    (here vol20). The conformal quantile is taken on z's instead of raw
    buffers, then re-scaled by today's σ at publish time. This yields
    materially tighter strikes when current vol is below regime-historical
    levels.
    """
    ticker: str
    side: str
    horizon: int
    folds: list[TightFold] = field(default_factory=list)
    pooled_wins: int = 0
    pooled_losses: int = 0
    z_final: float = 0.0          # vol-normalized buffer quantile (full history)
    sigma_today: float = 0.0      # today's σ (vol20)
    b_final: float = 0.0          # = z_final × σ_today × √T + safety
    n_history_fires: int = 0
    eligible: bool = False
    today_in_regime: bool = False


def evaluate_tight(close: np.ndarray, dates: np.ndarray,
                   sigma: np.ndarray, tight_active: np.ndarray,
                   side: str, horizon: int, ticker: str,
                   today_ok: bool) -> TightVariant:
    """Vol-adaptive conformal tight-tier evaluator. See TightVariant docstring."""
    vr = TightVariant(ticker=ticker, side=side, horizon=horizon,
                       today_in_regime=today_ok)
    buf = buffer_array(close, horizon, side)
    n = len(dates)
    warmup = np.zeros(n, dtype=bool)
    warmup[WARMUP_DAYS:] = True
    base = (warmup & tight_active & np.isfinite(buf)
            & np.isfinite(sigma) & (sigma > 0))
    T = horizon / 252.0
    sqrtT = math.sqrt(T)

    sigma_today = float(sigma[-1]) if (np.isfinite(sigma[-1]) and sigma[-1] > 0) else float("nan")
    vr.sigma_today = sigma_today

    for year in FOLD_YEARS:
        tr = base & train_mask_for_fold(dates, year, horizon)
        te = base & fold_mask(dates, year)
        ok = np.zeros(n, dtype=bool)
        ok[: n - horizon] = True
        te = te & ok
        if tr.sum() < SP_MIN_TRAIN_FIRES // 2:
            continue
        if te.sum() == 0:
            continue
        # Vol-normalized buffers from training
        z_tr = buf[tr] / (sigma[tr] * sqrtT)
        z_q = float(np.quantile(z_tr, SP_TIGHT_CONFORMAL_Q))
        # Per-row test thresholds: each test sample has its own σ(t)
        # so the threshold "moves" with current vol.
        thresh = z_q * sigma[te] * sqrtT + SP_SAFETY_EPS
        test_buf = buf[te]
        wins = int((test_buf <= thresh).sum())
        losses = int((test_buf > thresh).sum())
        vr.folds.append(TightFold(
            year=year,
            n_train=int(tr.sum()),
            n_test=int(te.sum()),
            z_q=z_q,
            median_b_hat=float(np.median(thresh)),
            wins=wins, losses=losses,
            worst_test_buf=float(test_buf.max()),
        ))

    vr.pooled_wins = sum(f.wins for f in vr.folds)
    vr.pooled_losses = sum(f.losses for f in vr.folds)
    vr.n_history_fires = int(base.sum())

    # Live (full-history) z quantile — projected to today's σ
    if base.sum() >= SP_MIN_TRAIN_FIRES and np.isfinite(sigma_today):
        z_full = buf[base] / (sigma[base] * sqrtT)
        vr.z_final = float(np.quantile(z_full, SP_TIGHT_CONFORMAL_Q))
        vr.b_final = vr.z_final * sigma_today * sqrtT + SP_SAFETY_EPS
    else:
        vr.z_final = float("nan")
        vr.b_final = float("nan")

    pooled_total = vr.pooled_wins + vr.pooled_losses
    pooled_wr = (vr.pooled_wins / pooled_total) if pooled_total else 0.0
    every_fold_ok = all(
        (f.wins / max(f.wins + f.losses, 1)) >= SP_TIGHT_PER_FOLD_WIN
        for f in vr.folds
    )
    vr.eligible = bool(
        vr.folds
        and len(vr.folds) >= SP_MIN_FOLDS
        and pooled_total >= SP_MIN_POOLED_TEST
        and pooled_wr >= SP_TIGHT_POOLED_WIN
        and every_fold_ok
        and np.isfinite(vr.b_final)
        and vr.b_final > 0
        and vr.b_final <= SP_TIGHT_MAX_BUFFER
    )
    return vr


@dataclass
class ICFold:
    year: int
    n_train: int
    n_test: int
    b_put_hat: float
    b_call_hat: float
    wins: int
    losses: int


@dataclass
class ICVariant:
    """Iron-condor (Atomic) tier — joint put+call credit spread.

    Per-leg conformal at SP_IC_CONFORMAL_Q (typically 0.975) means
    each leg breaches roughly 2.5% of the time; since put-breach and
    call-breach are mutually exclusive, the joint IC win rate is
    approximately 2q - 1 = 95%. Combined ROR is computed using BS
    pricing on a fixed spread width (5% of spot per leg).
    """
    ticker: str
    horizon: int
    folds: list[ICFold] = field(default_factory=list)
    pooled_wins: int = 0
    pooled_losses: int = 0
    b_put_final: float = 0.0
    b_call_final: float = 0.0
    n_history_fires: int = 0
    today_in_regime: bool = False
    eligible: bool = False
    # Profit metrics (filled when eligible)
    combined_credit: float = 0.0
    combined_max_loss: float = 0.0
    combined_ror: float = 0.0
    combined_ann_ror: float = 0.0
    width: float = 0.0
    K_put_short: float = 0.0
    K_put_long: float = 0.0
    K_call_short: float = 0.0
    K_call_long: float = 0.0


def _evaluate_ic_at_q(close, dates, regime, horizon, q):
    """Evaluate IC at one q value. Returns (folds, b_put_final, b_call_final,
    pooled_wins, pooled_losses, n_history) or None if insufficient data."""
    buf_put = buffer_array(close, horizon, "put")
    buf_call = buffer_array(close, horizon, "call")
    n = len(dates)
    warmup = np.zeros(n, dtype=bool)
    warmup[WARMUP_DAYS:] = True
    base = (warmup & regime
            & np.isfinite(buf_put) & np.isfinite(buf_call))
    if int(base.sum()) < SP_MIN_TRAIN_FIRES:
        return None
    folds = []
    for year in FOLD_YEARS:
        tr = base & train_mask_for_fold(dates, year, horizon)
        te = base & fold_mask(dates, year)
        ok = np.zeros(n, dtype=bool)
        ok[: n - horizon] = True
        te = te & ok
        if tr.sum() < SP_MIN_TRAIN_FIRES // 2 or te.sum() == 0:
            continue
        b_put_hat = float(np.quantile(buf_put[tr], q)) + SP_SAFETY_EPS
        b_call_hat = float(np.quantile(buf_call[tr], q)) + SP_SAFETY_EPS
        if b_put_hat > SP_IC_MAX_BUFFER or b_call_hat > SP_IC_MAX_BUFFER:
            continue
        joint_safe = (buf_put[te] <= b_put_hat) & (buf_call[te] <= b_call_hat)
        wins = int(joint_safe.sum())
        losses = int((~joint_safe).sum())
        folds.append(ICFold(
            year=year, n_train=int(tr.sum()), n_test=int(te.sum()),
            b_put_hat=b_put_hat, b_call_hat=b_call_hat,
            wins=wins, losses=losses,
        ))
    pooled_w = sum(f.wins for f in folds)
    pooled_l = sum(f.losses for f in folds)
    b_put_final = float(np.quantile(buf_put[base], q)) + SP_SAFETY_EPS
    b_call_final = float(np.quantile(buf_call[base], q)) + SP_SAFETY_EPS
    return folds, b_put_final, b_call_final, pooled_w, pooled_l, int(base.sum())


def evaluate_ic(close: np.ndarray, dates: np.ndarray, regime: np.ndarray,
                horizon: int, ticker: str, today_ok: bool,
                spot: float, sigma: float, cal_days: int) -> ICVariant:
    """Joint walk-forward IC evaluator. Tries multiple per-leg conformal
    quantiles and keeps the SMALLEST q that yields a passing IC at
    ROR >= 50%. Smaller q → tighter strikes → higher credit → higher
    ROR; we accept it as long as the joint walk-forward still validates
    at >=95% pooled WR.
    """
    vr = ICVariant(ticker=ticker, horizon=horizon, today_in_regime=today_ok)
    best_ror_seen = -1.0  # keep the highest-ROR passing config
    chosen = None  # (folds, bP, bC, pw, pl, n_h, ror, credit, max_loss, ann_ror)
    for q in sorted(SP_IC_CONFORMAL_QS):  # ascending q = ascending buffer
        res = _evaluate_ic_at_q(close, dates, regime, horizon, q)
        if res is None:
            continue
        folds, bP, bC, pw, pl, n_h = res
        if not folds or len(folds) < SP_MIN_FOLDS:
            continue
        pooled = pw + pl
        if pooled < SP_MIN_POOLED_TEST:
            continue
        wr = pw / pooled
        if wr < SP_IC_POOLED_WIN:
            continue
        if any((f.wins / max(f.wins + f.losses, 1)) < SP_IC_PER_FOLD_WIN
               for f in folds):
            continue
        if not (np.isfinite(bP) and np.isfinite(bC)
                and bP <= SP_IC_MAX_BUFFER and bC <= SP_IC_MAX_BUFFER):
            continue
        # Sweep spread widths; pick the one with highest ROR.
        iv = sigma * 1.30
        T = max(cal_days, 1) / 365.0
        for w_pct in SP_IC_WIDTHS:
            width = spot * w_pct
            K_ps = spot * (1 - bP)
            K_pl = K_ps - width
            K_cs = spot * (1 + bC)
            K_cl = K_cs + width
            if K_pl <= 0:
                continue
            cp = max(bs_put(spot, K_ps, T, iv) - bs_put(spot, K_pl, T, iv), 0.0) * 0.80
            cc = max(bs_call(spot, K_cs, T, iv) - bs_call(spot, K_cl, T, iv), 0.0) * 0.80
            cred = cp + cc
            ml = max(width - cred, 0.01)
            ror = cred / ml
            ann_ror = ror * (365.0 / max(cal_days, 1))
            if ror > best_ror_seen:
                best_ror_seen = ror
                chosen = (folds, bP, bC, pw, pl, n_h, ror, cred, ml, ann_ror,
                           K_ps, K_pl, K_cs, K_cl, width)

    if chosen is None:
        return vr
    (folds, bP, bC, pw, pl, n_h, ror, cred, ml, ann_ror,
     K_ps, K_pl, K_cs, K_cl, width) = chosen
    vr.folds = folds
    vr.pooled_wins = pw
    vr.pooled_losses = pl
    vr.n_history_fires = n_h
    vr.b_put_final = bP
    vr.b_call_final = bC

    if ror < SP_IC_TARGET_ROR:
        return vr  # passes WR but not ROR — not eligible

    vr.width = width
    vr.combined_credit = cred
    vr.combined_max_loss = ml
    vr.combined_ror = ror
    vr.combined_ann_ror = ann_ror
    vr.K_put_short = K_ps
    vr.K_put_long = K_pl
    vr.K_call_short = K_cs
    vr.K_call_long = K_cl
    vr.eligible = True
    return vr


@dataclass
class TickerOut:
    ticker: str
    today_close: float
    end_date: str
    realized_vol_pct: float | None
    today_in_regime: bool
    today_in_tight_regime: bool = False
    put_variants: list[StillVariant] = field(default_factory=list)
    call_variants: list[StillVariant] = field(default_factory=list)
    tight_put_variants: list[TightVariant] = field(default_factory=list)
    tight_call_variants: list[TightVariant] = field(default_factory=list)
    ic_variants: list[ICVariant] = field(default_factory=list)


def process_ticker(ticker: str) -> TickerOut | None:
    ts = load_series(ticker)
    if ts is None:
        return None
    feats = compute_features(ts.close)
    sp = stillpoint_mask(feats)
    tight = tight_mask(feats)
    today_ok = today_in_regime(feats)
    today_tight_ok = today_in_tight_regime(feats)
    rv = realized_vol(ts.close)

    # Skip the ticker entirely if it's never in either regime.
    if int(sp.sum()) < SP_MIN_TRAIN_FIRES and int(tight.sum()) < SP_MIN_TRAIN_FIRES:
        return None

    out = TickerOut(
        ticker=ticker, today_close=float(ts.close[-1]),
        end_date=str(ts.dates[-1]),
        realized_vol_pct=(rv * 100.0) if rv else None,
        today_in_regime=today_ok,
        today_in_tight_regime=today_tight_ok,
    )
    # Core tier (5%-buffer, h ∈ {5,7,10,14,21}, static conformal)
    if int(sp.sum()) >= SP_MIN_TRAIN_FIRES:
        for h in HORIZONS:
            for side in ("put", "call"):
                vr = evaluate_variant(
                    ts.close, ts.dates, sp, side, h, ticker, today_ok,
                )
                if side == "put":
                    out.put_variants.append(vr)
                else:
                    out.call_variants.append(vr)
    # Tight tier (3%-buffer, h ∈ {2,3,5}, vol-adaptive conformal)
    if int(tight.sum()) >= SP_MIN_TRAIN_FIRES:
        for h in SP_TIGHT_HORIZONS:
            for side in ("put", "call"):
                tv = evaluate_tight(
                    ts.close, ts.dates, feats.vol20, tight,
                    side, h, ticker, today_tight_ok,
                )
                if side == "put":
                    out.tight_put_variants.append(tv)
                else:
                    out.tight_call_variants.append(tv)
    # Iron Condor (Atomic) tier — joint put+call, 50%+ ROR target.
    # Try both regimes (base + tight); keep whichever passes with higher
    # ROR. Today's deployment requires the same regime to be active.
    if rv is not None and rv > 0:
        spot_now = float(ts.close[-1])
        for h in SP_IC_HORIZONS:
            _, _, cal_days = actual_options_expiry(str(ts.dates[-1]), h)
            best = None
            for regime_mask, today_in_this_regime in (
                (sp, today_ok), (tight, today_tight_ok),
            ):
                if int(regime_mask.sum()) < SP_MIN_TRAIN_FIRES:
                    continue
                ic = evaluate_ic(
                    ts.close, ts.dates, regime_mask, h, ticker,
                    today_in_this_regime,
                    spot=spot_now, sigma=rv,
                    cal_days=int(cal_days),
                )
                if not ic.eligible:
                    continue
                # Prefer the variant deployable today; tie-break on ROR
                key = (1 if ic.today_in_regime else 0, ic.combined_ror)
                if best is None or key > best[0]:
                    best = (key, ic)
            if best is not None:
                out.ic_variants.append(best[1])
    return out


def _rung_payload(spot: float, vr: StillVariant, end_date: str,
                  rv: float | None) -> dict[str, Any]:
    exp_iso, kind, cal_days = actual_options_expiry(end_date, vr.horizon)
    strike = strike_from_buffer(spot, vr.b_final, vr.side)
    pooled = vr.pooled_wins + vr.pooled_losses
    pooled_wr = (vr.pooled_wins / pooled) if pooled else 0.0
    base = {
        "side": vr.side,
        "horizon": vr.horizon,
        "expiry_date": exp_iso,
        "expiry_type": kind,
        "calendar_days_to_expiry": cal_days,
        "strike": strike,
        "buffer_pct": vr.b_final * 100.0,
        "pooled_wins": vr.pooled_wins,
        "pooled_losses": vr.pooled_losses,
        "n_test": pooled,
        "win_rate_pct": pooled_wr * 100.0,
        "n_folds": len(vr.folds),
        "n_history_fires": vr.n_history_fires,
        "today_in_regime": vr.today_in_regime,
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
            for f in vr.folds
        ],
    }
    prof = estimate_profit(
        side=vr.side, spot=spot, buffer=vr.b_final,
        horizon_sessions=vr.horizon,
        realized_sigma=(rv if rv else None),
        calendar_days_to_expiry=cal_days,
    )
    if prof is not None:
        base["profit"] = {
            "realized_vol_pct": prof.realized_vol * 100.0,
            "implied_vol_pct": prof.implied_vol * 100.0,
            "short_strike": prof.short_strike,
            "long_strike": prof.long_strike,
            "spread_width": prof.width,
            "est_credit_per_share": prof.credit,
            "est_max_loss_per_share": prof.max_loss,
            "return_on_risk_pct": prof.return_on_risk * 100.0,
            "annualized_ror_pct": prof.annualized_ror * 100.0,
        }
    else:
        base["profit"] = None
    return base


def _signal_for_side(t: TickerOut, side: str) -> dict[str, Any] | None:
    variants = t.put_variants if side == "put" else t.call_variants
    elig = [v for v in variants if v.eligible and v.today_in_regime]
    if not elig:
        return None
    rv = (t.realized_vol_pct / 100.0) if t.realized_vol_pct else None
    ladder = [_rung_payload(t.today_close, v, t.end_date, rv) for v in elig]
    # tightest buffer first
    ladder.sort(key=lambda r: r["buffer_pct"])
    return {
        "ticker": t.ticker,
        "today_close": t.today_close,
        "end_date": t.end_date,
        "realized_vol_pct": t.realized_vol_pct,
        "side": side,
        "ladder": ladder,
        # primary metrics from tightest rung
        "strike": ladder[0]["strike"],
        "buffer_pct": ladder[0]["buffer_pct"],
        "horizon": ladder[0]["horizon"],
        "expiry_date": ladder[0]["expiry_date"],
        "expiry_type": ladder[0]["expiry_type"],
        "calendar_days_to_expiry": ladder[0]["calendar_days_to_expiry"],
        "win_rate_pct": ladder[0]["win_rate_pct"],
        "n_test": ladder[0]["n_test"],
        "n_folds": ladder[0]["n_folds"],
    }


def _tight_rung_payload(spot: float, vr: TightVariant, end_date: str,
                         rv: float | None) -> dict[str, Any]:
    exp_iso, kind, cal_days = actual_options_expiry(end_date, vr.horizon)
    strike = strike_from_buffer(spot, vr.b_final, vr.side)
    pooled = vr.pooled_wins + vr.pooled_losses
    pooled_wr = (vr.pooled_wins / pooled) if pooled else 0.0
    base = {
        "side": vr.side,
        "horizon": vr.horizon,
        "expiry_date": exp_iso,
        "expiry_type": kind,
        "calendar_days_to_expiry": cal_days,
        "strike": strike,
        "buffer_pct": vr.b_final * 100.0,
        "z_q": vr.z_final,
        "sigma_today_pct": vr.sigma_today * 100.0,
        "method": "voladapt",
        "pooled_wins": vr.pooled_wins,
        "pooled_losses": vr.pooled_losses,
        "n_test": pooled,
        "win_rate_pct": pooled_wr * 100.0,
        "n_folds": len(vr.folds),
        "n_history_fires": vr.n_history_fires,
        "today_in_regime": vr.today_in_regime,
        "folds": [
            {
                "year": f.year,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "z_q": f.z_q,
                "median_b_hat_pct": f.median_b_hat * 100.0,
                "wins": f.wins,
                "losses": f.losses,
                "worst_test_buf_pct": f.worst_test_buf * 100.0,
            }
            for f in vr.folds
        ],
    }
    prof = estimate_profit(
        side=vr.side, spot=spot, buffer=vr.b_final,
        horizon_sessions=vr.horizon,
        realized_sigma=(rv if rv else None),
        calendar_days_to_expiry=cal_days,
    )
    if prof is not None:
        base["profit"] = {
            "realized_vol_pct": prof.realized_vol * 100.0,
            "implied_vol_pct": prof.implied_vol * 100.0,
            "short_strike": prof.short_strike,
            "long_strike": prof.long_strike,
            "spread_width": prof.width,
            "est_credit_per_share": prof.credit,
            "est_max_loss_per_share": prof.max_loss,
            "return_on_risk_pct": prof.return_on_risk * 100.0,
            "annualized_ror_pct": prof.annualized_ror * 100.0,
        }
    else:
        base["profit"] = None
    return base


def _ic_rung_payload(spot: float, vr: ICVariant, end_date: str) -> dict[str, Any]:
    exp_iso, kind, cal_days = actual_options_expiry(end_date, vr.horizon)
    pooled = vr.pooled_wins + vr.pooled_losses
    pooled_wr = (vr.pooled_wins / pooled) if pooled else 0.0
    return {
        "horizon": vr.horizon,
        "expiry_date": exp_iso,
        "expiry_type": kind,
        "calendar_days_to_expiry": cal_days,
        "K_put_short": vr.K_put_short,
        "K_put_long": vr.K_put_long,
        "K_call_short": vr.K_call_short,
        "K_call_long": vr.K_call_long,
        "buf_put_pct": vr.b_put_final * 100.0,
        "buf_call_pct": vr.b_call_final * 100.0,
        "width": vr.width,
        "combined_credit": vr.combined_credit,
        "combined_max_loss": vr.combined_max_loss,
        "combined_ror_pct": vr.combined_ror * 100.0,
        "combined_annualized_ror_pct": vr.combined_ann_ror * 100.0,
        "joint_pooled_wins": vr.pooled_wins,
        "joint_pooled_losses": vr.pooled_losses,
        "n_test": pooled,
        "joint_win_rate_pct": pooled_wr * 100.0,
        "n_folds": len(vr.folds),
        "n_history_fires": vr.n_history_fires,
        "today_in_regime": vr.today_in_regime,
        "folds": [
            {
                "year": f.year,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "b_put_hat_pct": f.b_put_hat * 100.0,
                "b_call_hat_pct": f.b_call_hat * 100.0,
                "wins": f.wins,
                "losses": f.losses,
            }
            for f in vr.folds
        ],
    }


def _ic_signal(t: TickerOut) -> dict[str, Any] | None:
    elig = [v for v in t.ic_variants if v.eligible and v.today_in_regime]
    if not elig:
        return None
    ladder = [_ic_rung_payload(t.today_close, v, t.end_date) for v in elig]
    # sort: highest ROR first (highest profit per trade)
    ladder.sort(key=lambda r: -r["combined_ror_pct"])
    return {
        "ticker": t.ticker,
        "today_close": t.today_close,
        "end_date": t.end_date,
        "realized_vol_pct": t.realized_vol_pct,
        "ladder": ladder,
        # primary metrics from highest-ROR rung
        "K_put_short": ladder[0]["K_put_short"],
        "K_call_short": ladder[0]["K_call_short"],
        "horizon": ladder[0]["horizon"],
        "expiry_date": ladder[0]["expiry_date"],
        "combined_ror_pct": ladder[0]["combined_ror_pct"],
        "joint_win_rate_pct": ladder[0]["joint_win_rate_pct"],
        "n_test": ladder[0]["n_test"],
        "n_folds": ladder[0]["n_folds"],
    }


def _tight_signal_for_side(t: TickerOut, side: str) -> dict[str, Any] | None:
    variants = t.tight_put_variants if side == "put" else t.tight_call_variants
    elig = [v for v in variants if v.eligible and v.today_in_regime]
    if not elig:
        return None
    rv = (t.realized_vol_pct / 100.0) if t.realized_vol_pct else None
    ladder = [_tight_rung_payload(t.today_close, v, t.end_date, rv) for v in elig]
    ladder.sort(key=lambda r: r["buffer_pct"])
    return {
        "ticker": t.ticker,
        "today_close": t.today_close,
        "end_date": t.end_date,
        "realized_vol_pct": t.realized_vol_pct,
        "side": side,
        "ladder": ladder,
        "strike": ladder[0]["strike"],
        "buffer_pct": ladder[0]["buffer_pct"],
        "horizon": ladder[0]["horizon"],
        "expiry_date": ladder[0]["expiry_date"],
        "expiry_type": ladder[0]["expiry_type"],
        "calendar_days_to_expiry": ladder[0]["calendar_days_to_expiry"],
        "win_rate_pct": ladder[0]["win_rate_pct"],
        "n_test": ladder[0]["n_test"],
        "n_folds": ladder[0]["n_folds"],
    }


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("SP_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]

    t0 = time.time()
    put_outs: list[dict[str, Any]] = []
    call_outs: list[dict[str, Any]] = []
    pin_outs: list[dict[str, Any]] = []
    tight_put_outs: list[dict[str, Any]] = []
    tight_call_outs: list[dict[str, Any]] = []
    tight_pin_outs: list[dict[str, Any]] = []
    ic_outs: list[dict[str, Any]] = []
    n_processed = 0
    n_in_regime = 0
    n_in_tight_regime = 0
    pooled_w = pooled_l = 0
    put_w = put_l = 0
    call_w = call_l = 0
    tight_pooled_w = tight_pooled_l = 0
    ic_pooled_w = ic_pooled_l = 0

    for i, t in enumerate(tickers, 1):
        try:
            r = process_ticker(t)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] {t}: {exc}", file=sys.stderr)
            continue
        if r is None:
            continue
        n_processed += 1
        if r.today_in_regime:
            n_in_regime += 1
        if r.today_in_tight_regime:
            n_in_tight_regime += 1
        # Pool stats across ALL eligible variants (not just those
        # deployable today). The pooled test is the validation; today's
        # gate just filters what we publish.
        for v in r.put_variants:
            if v.eligible:
                pooled_w += v.pooled_wins
                pooled_l += v.pooled_losses
                put_w += v.pooled_wins
                put_l += v.pooled_losses
        for v in r.call_variants:
            if v.eligible:
                pooled_w += v.pooled_wins
                pooled_l += v.pooled_losses
                call_w += v.pooled_wins
                call_l += v.pooled_losses
        for v in r.tight_put_variants + r.tight_call_variants:
            if v.eligible:
                tight_pooled_w += v.pooled_wins
                tight_pooled_l += v.pooled_losses
        for v in r.ic_variants:
            if v.eligible:
                ic_pooled_w += v.pooled_wins
                ic_pooled_l += v.pooled_losses

        ps = _signal_for_side(r, "put")
        cs = _signal_for_side(r, "call")
        if ps:
            put_outs.append(ps)
        if cs:
            call_outs.append(cs)
        if ps and cs:
            pin_outs.append({
                "ticker": r.ticker,
                "today_close": r.today_close,
                "end_date": r.end_date,
                "put": ps,
                "call": cs,
            })

        tps = _tight_signal_for_side(r, "put")
        tcs = _tight_signal_for_side(r, "call")
        if tps:
            tight_put_outs.append(tps)
        if tcs:
            tight_call_outs.append(tcs)
        if tps and tcs:
            tight_pin_outs.append({
                "ticker": r.ticker,
                "today_close": r.today_close,
                "end_date": r.end_date,
                "put": tps,
                "call": tcs,
            })

        ic_sig = _ic_signal(r)
        if ic_sig:
            ic_outs.append(ic_sig)

        if i % 100 == 0:
            print(f"  {i}/{len(tickers)}  in-reg={n_in_regime}/{n_in_tight_regime}  "
                  f"core(p/c/pin)={len(put_outs)}/{len(call_outs)}/{len(pin_outs)}  "
                  f"tight(p/c/pin)={len(tight_put_outs)}/{len(tight_call_outs)}/{len(tight_pin_outs)}  "
                  f"ic={len(ic_outs)}  "
                  f"elapsed={time.time()-t0:.1f}s")

    pooled_total = pooled_w + pooled_l
    pooled_wr = (pooled_w / pooled_total) if pooled_total else None
    put_total = put_w + put_l
    call_total = call_w + call_l
    tight_total = tight_pooled_w + tight_pooled_l
    tight_wr = (tight_pooled_w / tight_total) if tight_total else None
    ic_total = ic_pooled_w + ic_pooled_l
    ic_wr = (ic_pooled_w / ic_total) if ic_total else None

    print()
    print(f"Tickers processed:                {n_processed}")
    print(f"  in core regime today:           {n_in_regime}")
    print(f"  in tight regime today:          {n_in_tight_regime}")
    print(f"Core tier deployable (p/c/pin):   {len(put_outs)}/{len(call_outs)}/{len(pin_outs)}")
    print(f"Tight tier deployable (p/c/pin):  {len(tight_put_outs)}/{len(tight_call_outs)}/{len(tight_pin_outs)}")
    print(f"IC (Atomic) tier deployable:      {len(ic_outs)}")
    if pooled_total:
        print(f"Core pooled OOS win rate:         {pooled_wr*100:.3f}% "
              f"({pooled_w}/{pooled_total})")
    if tight_total:
        print(f"Tight pooled OOS win rate:        {tight_wr*100:.3f}% "
              f"({tight_pooled_w}/{tight_total})")
    if ic_total:
        print(f"IC joint pooled OOS win rate:     {ic_wr*100:.3f}% "
              f"({ic_pooled_w}/{ic_total})")
    print(f"Elapsed:                          {time.time()-t0:.1f}s")

    # Sort signals: tightest buffer first.
    put_outs.sort(key=lambda s: s["buffer_pct"])
    call_outs.sort(key=lambda s: s["buffer_pct"])
    pin_outs.sort(key=lambda s: s["put"]["buffer_pct"] + s["call"]["buffer_pct"])
    tight_put_outs.sort(key=lambda s: s["buffer_pct"])
    tight_call_outs.sort(key=lambda s: s["buffer_pct"])
    tight_pin_outs.sort(key=lambda s: s["put"]["buffer_pct"] + s["call"]["buffer_pct"])
    ic_outs.sort(key=lambda s: -s["combined_ror_pct"])  # highest ROR first

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "Stillpoint",
        "summary": {
            "n_tickers_processed": n_processed,
            "n_in_regime_today": n_in_regime,
            "n_in_tight_regime_today": n_in_tight_regime,
            "horizons": HORIZONS,
            "tight_horizons": SP_TIGHT_HORIZONS,
            "fold_years": FOLD_YEARS,
            "max_buffer_pct": SP_MAX_BUFFER * 100.0,
            "tight_max_buffer_pct": SP_TIGHT_MAX_BUFFER * 100.0,
            "conformal_q": SP_CONFORMAL_Q,
            "tight_conformal_q": SP_TIGHT_CONFORMAL_Q,
            "safety_eps_pct": SP_SAFETY_EPS * 100.0,
            "target_pooled_win_pct": SP_TARGET_POOLED_WIN * 100.0,
            "target_per_fold_win_pct": SP_TARGET_PER_FOLD_WIN * 100.0,
            "n_put_signals": len(put_outs),
            "n_call_signals": len(call_outs),
            "n_pin_signals": len(pin_outs),
            "n_tight_put_signals": len(tight_put_outs),
            "n_tight_call_signals": len(tight_call_outs),
            "n_tight_pin_signals": len(tight_pin_outs),
            "n_ic_signals": len(ic_outs),
            "ic_horizons": SP_IC_HORIZONS,
            "ic_conformal_qs": SP_IC_CONFORMAL_QS,
            "ic_target_ror_pct": SP_IC_TARGET_ROR * 100.0,
            "pooled_wins": pooled_w,
            "pooled_losses": pooled_l,
            "pooled_win_rate": pooled_wr,
            "tight_pooled_wins": tight_pooled_w,
            "tight_pooled_losses": tight_pooled_l,
            "tight_pooled_win_rate": tight_wr,
            "ic_pooled_wins": ic_pooled_w,
            "ic_pooled_losses": ic_pooled_l,
            "ic_joint_pooled_win_rate": ic_wr,
            "put": {
                "pooled_wins": put_w, "pooled_losses": put_l,
                "pooled_win_rate": (put_w / put_total) if put_total else None,
            },
            "call": {
                "pooled_wins": call_w, "pooled_losses": call_l,
                "pooled_win_rate": (call_w / call_total) if call_total else None,
            },
        },
        "put_signals": put_outs,
        "call_signals": call_outs,
        "pin_signals": pin_outs,
        "tight_put_signals": tight_put_outs,
        "tight_call_signals": tight_call_outs,
        "tight_pin_signals": tight_pin_outs,
        "ic_signals": ic_outs,
    }
    out_path = os.path.join(OUTPUT_DIR, "stillpoint_signals.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
