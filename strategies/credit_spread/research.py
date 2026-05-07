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
    actual_options_expiry,
    compute_features,
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
from pricing import estimate_profit, realized_vol


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def _rung_dict(
    r: TickerResult, sr: SideResult, e: dict[str, Any],
) -> dict[str, Any]:
    """Build a full rung record: walk-forward stats + actual options
    expiry + profit estimate (BS)."""
    exp_iso, kind, cal_days = actual_options_expiry(r.end_date, e["horizon"])
    base = {
        "horizon": e["horizon"],
        "expiry_date": exp_iso,
        "expiry_type": kind,
        "calendar_days_to_expiry": cal_days,
        "strike": e["strike"],
        "buffer_pct": e["buffer_pct"],
        "variant": e["variant"],
        "n_test": e["n_test"],
        "n_folds": e["n_folds"],
        "pooled_wins": e["pooled_wins"],
        "pooled_losses": e["pooled_losses"],
        "history_worst_buffer_pct": e["history_worst_buffer"] * 100.0,
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
    }
    # Profit estimate (may be None if vol unavailable or long leg invalid)
    prof = estimate_profit(
        side=sr.side,
        spot=r.today_close,
        buffer=e["buffer"],
        horizon_sessions=e["horizon"],
        realized_sigma=r.realized_vol,
        calendar_days_to_expiry=cal_days,
    )
    if prof is not None:
        base["profit"] = {
            "realized_vol_pct":       prof.realized_vol * 100.0,
            "implied_vol_pct":        prof.implied_vol  * 100.0,
            "short_iv_pct":           prof.short_iv * 100.0,
            "long_iv_pct":            prof.long_iv  * 100.0,
            "short_strike":           prof.short_strike,
            "long_strike":            prof.long_strike,
            "spread_width":           prof.width,
            "mid_credit_per_share":   prof.mid_credit,
            "est_credit_per_share":   prof.credit,
            "bid_ask_per_share":      prof.bid_ask_estimate,
            "est_max_loss_per_share": prof.max_loss,
            "return_on_risk_pct":     prof.return_on_risk * 100.0,
            "annualized_ror_pct":     prof.annualized_ror * 100.0,
            "credit_quality":         prof.quality,
            "tradeable":              prof.tradeable,
        }
    else:
        base["profit"] = None
    return base


def _signal_payload(r: TickerResult, sr: SideResult) -> dict[str, Any]:
    assert sr.best is not None
    best = sr.best
    primary_rung = _rung_dict(r, sr, best)
    ladder = [_rung_dict(r, sr, e) for e in sr.all_eligible]
    return {
        "ticker": r.ticker,
        "today_close": r.today_close,
        "end_date": r.end_date,
        "side": sr.side,
        "realized_vol_pct": (r.realized_vol * 100.0) if r.realized_vol else None,
        # primary (tightest-buffer) pick
        "strike": primary_rung["strike"],
        "buffer_pct": primary_rung["buffer_pct"],
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
        "regime_ok_today": best["regime_ok_today"],
        "profit": primary_rung["profit"],
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

    t0 = time.time()
    results: list[TickerResult] = []
    put_elig: list[tuple[TickerResult, SideResult]] = []
    call_elig: list[tuple[TickerResult, SideResult]] = []
    put_watch: list[tuple[TickerResult, SideResult]] = []
    call_watch: list[tuple[TickerResult, SideResult]] = []
    for i, t in enumerate(tickers, 1):
        try:
            r = process_ticker(t)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] {t}: {exc}", file=sys.stderr)
            continue
        if r is None:
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

    print()
    print(f"Tickers processed:       {len(results)}")
    print(f"Put-side eligible:       {len(put_elig)}  "
          f"({put_wins}/{put_wins+put_losses} OOS, losses={put_losses})")
    print(f"Call-side eligible:      {len(call_elig)} "
          f"({call_wins}/{call_wins+call_losses} OOS, losses={call_losses})")
    print(f"Combined OOS tests:      {pool_wins + pool_losses}")
    print(f"Combined OOS win rate:   "
          f"{(pool_wins/(pool_wins+pool_losses)*100) if pool_wins+pool_losses else 0:.3f}%")

    # Sort: tightest buffer first, tie-break on larger test sample count.
    put_elig.sort(key=lambda x: (x[1].best["buffer"], -x[1].best["n_test"]))
    call_elig.sort(key=lambda x: (x[1].best["buffer"], -x[1].best["n_test"]))
    # Watchlists sort by tightest buffer too.
    put_watch.sort(key=lambda x: x[1].watchlist[0]["buffer"])
    call_watch.sort(key=lambda x: x[1].watchlist[0]["buffer"])

    lean = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "n_tickers_processed": len(results),
            "horizons": HORIZONS,
            "fold_years": FOLD_YEARS,
            "safety_eps": SAFETY_EPS,
            "max_buffer": MAX_BUFFER,
            "put": {
                "n_eligible": len(put_elig),
                "pooled_wins": put_wins,
                "pooled_losses": put_losses,
                "pooled_win_rate": put_wins/(put_wins+put_losses) if put_wins+put_losses else None,
            },
            "call": {
                "n_eligible": len(call_elig),
                "pooled_wins": call_wins,
                "pooled_losses": call_losses,
                "pooled_win_rate": call_wins/(call_wins+call_losses) if call_wins+call_losses else None,
            },
            "combined": {
                "n_eligible": len(put_elig) + len(call_elig),
                "pooled_wins": pool_wins,
                "pooled_losses": pool_losses,
                "pooled_win_rate": pool_wins/(pool_wins+pool_losses) if pool_wins+pool_losses else None,
            },
        },
        "put_signals":  [_signal_payload(r, sr) for r, sr in put_elig],
        "call_signals": [_signal_payload(r, sr) for r, sr in call_elig],
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
          f"(puts={len(put_elig)}, calls={len(call_elig)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
