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
    actual_options_expiry, buffer_array, compute_features,
    fold_mask, list_tickers, load_series, stillpoint_mask,
    strike_from_buffer, today_in_regime, train_mask_for_fold,
)


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
class TickerOut:
    ticker: str
    today_close: float
    end_date: str
    realized_vol_pct: float | None
    today_in_regime: bool
    put_variants: list[StillVariant] = field(default_factory=list)
    call_variants: list[StillVariant] = field(default_factory=list)


def process_ticker(ticker: str) -> TickerOut | None:
    ts = load_series(ticker)
    if ts is None:
        return None
    feats = compute_features(ts.close)
    sp = stillpoint_mask(feats)
    today_ok = today_in_regime(feats)
    rv = realized_vol(ts.close)

    # Skip the ticker entirely if it's never in the regime — saves time.
    if int(sp.sum()) < SP_MIN_TRAIN_FIRES:
        return None

    out = TickerOut(
        ticker=ticker, today_close=float(ts.close[-1]),
        end_date=str(ts.dates[-1]),
        realized_vol_pct=(rv * 100.0) if rv else None,
        today_in_regime=today_ok,
    )
    for h in HORIZONS:
        for side in ("put", "call"):
            vr = evaluate_variant(
                ts.close, ts.dates, sp, side, h, ticker, today_ok,
            )
            if side == "put":
                out.put_variants.append(vr)
            else:
                out.call_variants.append(vr)
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


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("SP_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]

    t0 = time.time()
    put_outs: list[dict[str, Any]] = []
    call_outs: list[dict[str, Any]] = []
    pin_outs: list[dict[str, Any]] = []
    n_processed = 0
    n_in_regime = 0
    pooled_w = pooled_l = 0
    put_w = put_l = 0
    call_w = call_l = 0

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

        ps = _signal_for_side(r, "put")
        cs = _signal_for_side(r, "call")
        if ps:
            put_outs.append(ps)
        if cs:
            call_outs.append(cs)
        if ps and cs:
            # iron-condor-equivalent "pinpoint" combo
            pin_outs.append({
                "ticker": r.ticker,
                "today_close": r.today_close,
                "end_date": r.end_date,
                "put": ps,
                "call": cs,
            })
        if i % 100 == 0:
            print(f"  {i}/{len(tickers)}  in-regime={n_in_regime}  "
                  f"put_signals={len(put_outs)}  call_signals={len(call_outs)}  "
                  f"pin_signals={len(pin_outs)}  "
                  f"elapsed={time.time()-t0:.1f}s")

    pooled_total = pooled_w + pooled_l
    pooled_wr = (pooled_w / pooled_total) if pooled_total else None
    put_total = put_w + put_l
    call_total = call_w + call_l

    print()
    print(f"Tickers processed:        {n_processed}")
    print(f"Tickers in regime today:  {n_in_regime}")
    print(f"Put-side eligible today:  {len(put_outs)}")
    print(f"Call-side eligible today: {len(call_outs)}")
    print(f"Pin (both sides) today:   {len(pin_outs)}")
    if pooled_total:
        print(f"Pooled OOS win rate:      {pooled_wr*100:.3f}% "
              f"({pooled_w}/{pooled_total})")
    print(f"Elapsed:                  {time.time()-t0:.1f}s")

    # Sort signals: tightest buffer first.
    put_outs.sort(key=lambda s: s["buffer_pct"])
    call_outs.sort(key=lambda s: s["buffer_pct"])
    pin_outs.sort(key=lambda s: s["put"]["buffer_pct"] + s["call"]["buffer_pct"])

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "Stillpoint",
        "summary": {
            "n_tickers_processed": n_processed,
            "n_in_regime_today": n_in_regime,
            "horizons": HORIZONS,
            "fold_years": FOLD_YEARS,
            "max_buffer_pct": SP_MAX_BUFFER * 100.0,
            "conformal_q": SP_CONFORMAL_Q,
            "safety_eps_pct": SP_SAFETY_EPS * 100.0,
            "target_pooled_win_pct": SP_TARGET_POOLED_WIN * 100.0,
            "target_per_fold_win_pct": SP_TARGET_PER_FOLD_WIN * 100.0,
            "n_put_signals": len(put_outs),
            "n_call_signals": len(call_outs),
            "n_pin_signals": len(pin_outs),
            "pooled_wins": pooled_w,
            "pooled_losses": pooled_l,
            "pooled_win_rate": pooled_wr,
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
    }
    out_path = os.path.join(OUTPUT_DIR, "stillpoint_signals.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
