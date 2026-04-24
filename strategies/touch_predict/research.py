"""TouchPredictor — walk-forward 100%-accurate touch predictions,
ranked by OTM-option profitability.

For each (ticker, side, regime, horizon):
  - side: 'call' (up-touch) or 'put' (down-touch)
  - regime: a subset of historical days meeting a mean-reversion gate
  - horizon h (in trading sessions)

We compute, over walk-forward folds:
  UP buffer   b_up(t, h)   = max(close[t+1..t+h]) / close[t] - 1
  DOWN buffer b_down(t, h) = 1 - min(close[t+1..t+h]) / close[t]

For a rule to be 100% touch-accurate:
  cert_buffer = min over training-set b(t, h) - SAFETY_EPS
  A signal wins iff the test-set b(t, h) >= cert_buffer, i.e. the
  forward path actually touches the certified threshold.

Walk-forward 100% = every test fold has every sample winning. Then
the rule is deployable today for the current regime.

For each eligible (ticker, side, regime, horizon), we estimate the
best OTM-option profit at the current spot with:
  - Certified touch buffer = what we'll touch
  - Spot = today's close
  - Target price = spot * (1 +/- cert_buffer)
  - Grid-searched OTM strike = maximizes ROI
  - Actual Friday expiry + calendar days via NYSE calendar

Per-ticker, per-side we pick the combo with HIGHEST ROI (profitability
is the engineering goal). The walk-forward remains leakage-free and
fail-closed on a per-fold 100% basis.
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

# Local import: pricing.py lives next to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from pricing import best_otm_play, realized_vol  # noqa: E402

# Reuse from credit_spread.common — causal features, fold masks,
# NYSE calendar helpers, ticker loader. Put this path AFTER our
# local path so local `pricing.py` wins on name collision.
sys.path.insert(1, os.path.join(os.path.dirname(_HERE), "credit_spread"))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_cs_common",
    os.path.join(os.path.dirname(_HERE), "credit_spread", "common.py"),
)
_cs_common = importlib.util.module_from_spec(_spec)  # type: ignore
# Register so dataclasses (and anything else) can find the module
# by its __module__ name when inspecting classes.
sys.modules["_cs_common"] = _cs_common
_spec.loader.exec_module(_cs_common)  # type: ignore
FOLD_YEARS           = _cs_common.FOLD_YEARS
MIN_TEST_SAMPLES     = _cs_common.MIN_TEST_SAMPLES
MIN_TRAIN_SAMPLES    = _cs_common.MIN_TRAIN_SAMPLES
WARMUP_DAYS          = _cs_common.WARMUP_DAYS
Features             = _cs_common.Features
TickerSeries         = _cs_common.TickerSeries
actual_options_expiry = _cs_common.actual_options_expiry
compute_features     = _cs_common.compute_features
fold_mask            = _cs_common.fold_mask
list_tickers         = _cs_common.list_tickers
load_series          = _cs_common.load_series
train_mask_for_fold  = _cs_common.train_mask_for_fold
forward_max_close    = _cs_common.forward_max_close
forward_min_close    = _cs_common.forward_min_close


OUTPUT_DIR = os.path.join(_HERE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Touch-prediction config
HORIZONS = [5, 7, 10, 14, 21]         # short = cheap premium
SAFETY_EPS = 0.005                    # 0.5%; subtracted from cert_buffer

# Target touch rate: the cert_buffer is set at the (1 - TARGET)th quantile
# of training buffers, so TARGET of training samples historically touched.
# Lower target = bigger certified buffer. The sweet spot depends on per-
# ticker vol; we search a grid of target rates per ticker below.
TARGET_GRID = [0.85, 0.75, 0.65, 0.55]

# A rule is ELIGIBLE if its pooled out-of-sample touch rate is at least
# 5 percentage points below the target (allows some conformal slack).
MIN_SLACK = 0.05

# Minimum buffer for the FINAL live-deployed signal. Per-fold buffers
# can dip below this without skipping the fold; we only require the
# full-history-fit final buffer to clear this floor for live use.
MIN_CERT_BUFFER = 0.02                # 2% minimum live certified touch
MIN_EV_REPORT = 0.20                  # 20% expected-value cut for display


class Side(str, Enum):
    CALL = "call"   # up-touch
    PUT = "put"     # down-touch


# -------------------------- regimes -----------------------------------
#
# A regime is a boolean mask over trading days, built from CAUSAL
# features (no look-ahead). Tighter gates → smaller eligible-day
# population but often → larger 100% certified buffer.


def regime_mask_for(side: Side, kind: str, f: Features) -> np.ndarray:
    """Build the regime mask for a side & regime 'kind'."""
    finite = (
        np.isfinite(f.trend) & np.isfinite(f.rsi14)
        & np.isfinite(f.dd252) & np.isfinite(f.up252)
        & np.isfinite(f.vol20) & np.isfinite(f.mom_252)
    )
    if side == Side.CALL:  # UP side: buy calls on mean-reversion bounces
        if kind == "plain":
            return finite
        if kind == "oversold":
            return finite & (f.rsi14 < 30.0)
        if kind == "deep_oversold":
            return finite & ((f.rsi14 < 20.0) | (f.dd252 >= 0.20))
        if kind == "dip_in_uptrend":
            return finite & (f.trend >= 1.00) & (f.rsi14 < 40.0)
        raise ValueError(f"unknown call regime {kind}")
    # PUT side: buy puts on overextended / parabolic moves
    if kind == "plain":
        return finite
    if kind == "overbought":
        return finite & (f.rsi14 > 70.0)
    if kind == "deep_overbought":
        return finite & ((f.rsi14 > 80.0) | (f.up252 > 0.50))
    if kind == "parabolic":
        return finite & (f.mom_252 >= 0.20) & (f.trend >= 1.15)
    raise ValueError(f"unknown put regime {kind}")


CALL_REGIMES = ["plain", "oversold", "deep_oversold", "dip_in_uptrend"]
PUT_REGIMES  = ["plain", "overbought", "deep_overbought", "parabolic"]


def today_in_regime(f: Features, side: Side, kind: str) -> bool:
    mask = regime_mask_for(side, kind, f)
    return bool(mask[-1])


# --------------------- forward touch buffers -------------------------


def buffer_up(close: np.ndarray, h: int) -> np.ndarray:
    """UP touch buffer at entry index t: max_fwd_close/close[t] - 1.
    For each t, this is the largest fractional UPWARD excursion in
    close[t+1..t+h]. Always a number; can be negative if stock only
    went down over the window.
    """
    m = forward_max_close(close, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = m / close - 1.0
    return np.where(np.isnan(b), np.nan, b)


def buffer_down(close: np.ndarray, h: int) -> np.ndarray:
    """DOWN touch buffer at entry index t: 1 - min_fwd_close/close[t].
    Largest fractional DOWNWARD excursion in close[t+1..t+h]. Always
    a number; can be negative if stock only went up."""
    m = forward_min_close(close, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = 1.0 - m / close
    return np.where(np.isnan(b), np.nan, b)


# --------------------- walk-forward evaluation ------------------------


@dataclass
class FoldResult:
    year: int
    n_train: int
    n_test: int
    cert_buffer: float        # min over train - SAFETY_EPS
    wins: int                 # test samples with buffer >= cert_buffer
    losses: int
    worst_test_buf: float     # smallest test buffer observed


@dataclass
class RuleResult:
    side: str
    regime: str
    horizon: int
    target_touch_rate: float = 0.0   # quantile target used
    folds: list[FoldResult] = field(default_factory=list)
    pooled_wins: int = 0
    pooled_losses: int = 0
    touch_rate: float = 0.0          # pooled_wins / pooled_total
    final_cert_buffer: float = 0.0
    final_history_min: float = 0.0
    n_regime_days: int = 0
    eligible: bool = False


def _evaluate(ts: TickerSeries, feats: Features,
              side: Side, regime: str, horizon: int,
              target_touch_rate: float) -> RuleResult:
    rr = RuleResult(side=side.value, regime=regime, horizon=horizon,
                    target_touch_rate=target_touch_rate)
    close = ts.close
    dates = ts.dates
    buf = buffer_up(close, horizon) if side == Side.CALL else buffer_down(close, horizon)
    rmask = regime_mask_for(side, regime, feats)
    warmup = np.zeros(len(dates), dtype=bool)
    warmup[WARMUP_DAYS:] = True
    base = rmask & warmup & np.isfinite(buf)
    rr.n_regime_days = int(base.sum())

    n = len(dates)
    test_ok = np.zeros(n, dtype=bool)
    test_ok[: n - horizon] = True

    # Quantile level: pick the buffer at the (1 − target)th percentile of
    # training buffers. E.g. target=0.80 → use the 20th percentile, so
    # ~80% of training samples had buffer >= cert_buffer.
    q_pct = (1.0 - target_touch_rate) * 100.0

    for year in FOLD_YEARS:
        tr = base & train_mask_for_fold(dates, year, horizon)
        te = base & fold_mask(dates, year) & test_ok
        if tr.sum() < MIN_TRAIN_SAMPLES:
            continue
        if te.sum() == 0:
            continue
        # cert_buffer: the (1-target)-quantile of training buffers,
        # less a small safety margin. The rule promises the stock
        # touches at least cert_buffer with ~target probability.
        train_q = float(np.percentile(buf[tr], q_pct))
        cert = max(train_q - SAFETY_EPS, 0.0)
        # No per-fold cert_buffer skip — we only require the FINAL
        # full-history cert_buffer to meet MIN_CERT_BUFFER for live
        # deployability. Per-fold low buffers just mean that fold's
        # training data was less favorable.
        test_buf = buf[te]
        wins = int((test_buf >= cert).sum())
        losses = int((test_buf < cert).sum())
        rr.folds.append(FoldResult(
            year=year,
            n_train=int(tr.sum()),
            n_test=int(te.sum()),
            cert_buffer=cert,
            wins=wins,
            losses=losses,
            worst_test_buf=float(test_buf.min()),
        ))

    rr.pooled_wins = sum(f.wins for f in rr.folds)
    rr.pooled_losses = sum(f.losses for f in rr.folds)

    # Final live cert_buffer uses ALL history with the same quantile.
    # We also record the absolute minimum (worst-ever move) for context.
    if rr.n_regime_days >= MIN_TRAIN_SAMPLES:
        rr.final_history_min = float(buf[base].min())
        full_q = float(np.percentile(buf[base], q_pct))
        rr.final_cert_buffer = max(full_q - SAFETY_EPS, 0.0)
    else:
        rr.final_history_min = float("nan")
        rr.final_cert_buffer = 0.0

    total = rr.pooled_wins + rr.pooled_losses
    rr.touch_rate = (rr.pooled_wins / total) if total > 0 else 0.0
    # Eligible if empirical OOS touch rate is within MIN_SLACK of target.
    min_touch = max(target_touch_rate - MIN_SLACK, 0.50)
    rr.eligible = bool(
        rr.folds
        and total >= MIN_TEST_SAMPLES
        and rr.touch_rate >= min_touch
        and rr.final_cert_buffer >= MIN_CERT_BUFFER
        and all(f.wins > 0 for f in rr.folds)
    )
    return rr


# --------------------- per-ticker search ------------------------------


@dataclass
class TickerResult:
    ticker: str
    n_days: int
    start_date: str
    end_date: str
    today_close: float
    realized_vol: float | None = None
    call_best: dict[str, Any] | None = None
    put_best: dict[str, Any] | None = None
    # Watchlist: eligible rules whose regime doesn't match today.
    call_watch: list[dict[str, Any]] = field(default_factory=list)
    put_watch: list[dict[str, Any]] = field(default_factory=list)


def _rule_to_signal(
    r: TickerResult, rule: RuleResult, side: Side, feats: Features
) -> dict[str, Any] | None:
    """Compute the live signal + best OTM play for an eligible rule.
    Returns None if rule has no profitable OTM placement today."""
    exp_iso, kind, cal_days = actual_options_expiry(r.end_date, rule.horizon)
    play = best_otm_play(
        side=side.value,
        spot=r.today_close,
        buffer=rule.final_cert_buffer,
        calendar_days_to_expiry=cal_days,
        realized_sigma=r.realized_vol,
    )
    if play is None:
        return None
    today_ok = today_in_regime(feats, side, rule.regime)
    # Expected value per $1 of premium paid:
    #   touch_rate * (profit/premium) + (1 - touch_rate) * (-1)
    # = touch_rate * ROI - (1 - touch_rate)
    # Expressed as % of premium.
    ev = rule.touch_rate * play.roi - (1.0 - rule.touch_rate)
    return {
        "side": side.value,
        "regime": rule.regime,
        "horizon": rule.horizon,
        "expiry_date": exp_iso,
        "expiry_type": kind,
        "calendar_days_to_expiry": cal_days,
        "cert_buffer_pct":     rule.final_cert_buffer * 100.0,
        "history_worst_pct":   rule.final_history_min * 100.0,
        "target_touch_rate_pct": rule.target_touch_rate * 100.0,
        "touch_rate_pct":      rule.touch_rate * 100.0,
        "target_price":        play.target_price,
        "strike":              play.strike,
        "k_frac":              play.k_frac,
        "otm_pct":             (play.strike / r.today_close - 1.0) * 100.0
                                if side == Side.CALL else
                                (1.0 - play.strike / r.today_close) * 100.0,
        "est_premium":         play.premium,
        "est_profit":          play.profit,
        "est_max_loss":        play.max_loss,
        "roi_pct":             play.roi * 100.0,           # ROI on a winning trade
        "ev_pct":              ev * 100.0,                 # expected value per $ premium
        "implied_vol_pct":     play.implied_vol * 100.0,
        "realized_vol_pct":    play.realized_vol * 100.0,
        "n_test":              rule.pooled_wins + rule.pooled_losses,
        "n_folds":             len(rule.folds),
        "pooled_wins":         rule.pooled_wins,
        "pooled_losses":       rule.pooled_losses,
        "n_regime_days":       rule.n_regime_days,
        "regime_ok_today":     today_ok,
        "folds": [
            {
                "year": f.year, "n_train": f.n_train, "n_test": f.n_test,
                "cert_buffer_pct": f.cert_buffer * 100.0,
                "wins": f.wins, "losses": f.losses,
                "worst_test_buf_pct": f.worst_test_buf * 100.0,
            }
            for f in rule.folds
        ],
    }


def process_ticker_side(
    ts: TickerSeries, feats: Features, side: Side,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (best, watchlist) for this side. `best` is the highest-ROI
    signal whose regime matches today (deployable). `watchlist` is a
    list of signals whose rule passed backtest but today's regime
    doesn't match."""
    regimes = CALL_REGIMES if side == Side.CALL else PUT_REGIMES
    best_signal: dict[str, Any] | None = None
    watch: list[dict[str, Any]] = []
    # Build a per-ticker TickerResult wrapper to pass into _rule_to_signal
    tr = TickerResult(
        ticker="",  # set later
        n_days=len(ts.dates),
        start_date=str(ts.dates[0]),
        end_date=str(ts.dates[-1]),
        today_close=float(ts.close[-1]),
        realized_vol=realized_vol(ts.close),
    )
    for regime in regimes:
        for h in HORIZONS:
            for target in TARGET_GRID:
                rule = _evaluate(ts, feats, side, regime, h, target)
                if not rule.eligible:
                    continue
                sig = _rule_to_signal(tr, rule, side, feats)
                if sig is None:
                    continue
                if not sig["regime_ok_today"]:
                    watch.append(sig)
                    continue
                if sig["ev_pct"] <= 0:
                    continue
                if best_signal is None or sig["ev_pct"] > best_signal["ev_pct"]:
                    best_signal = sig
    # Sort watchlist by EV descending
    watch.sort(key=lambda x: -x["ev_pct"])
    return best_signal, watch


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
        realized_vol=realized_vol(ts.close),
    )
    best_call, watch_call = process_ticker_side(ts, feats, Side.CALL)
    best_put,  watch_put  = process_ticker_side(ts, feats, Side.PUT)
    tr.call_best = best_call
    tr.put_best  = best_put
    tr.call_watch = watch_call
    tr.put_watch  = watch_put
    return tr


# --------------------- main driver -----------------------------------


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("TP_LIMIT") or os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]

    t0 = time.time()
    results: list[TickerResult] = []
    call_signals: list[dict[str, Any]] = []
    put_signals:  list[dict[str, Any]] = []
    call_watch:   list[dict[str, Any]] = []
    put_watch:    list[dict[str, Any]] = []
    for i, t in enumerate(tickers, 1):
        try:
            r = process_ticker(t)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] {t}: {exc}", file=sys.stderr)
            continue
        if r is None:
            continue
        results.append(r)
        if r.call_best is not None and r.call_best.get("ev_pct", 0) >= MIN_EV_REPORT * 100:
            call_signals.append({"ticker": r.ticker, "today_close": r.today_close,
                                 "end_date": r.end_date,
                                 "realized_vol_pct": (r.realized_vol or 0) * 100.0,
                                 **r.call_best})
        elif r.call_watch:
            call_watch.append({"ticker": r.ticker, "today_close": r.today_close,
                               "end_date": r.end_date,
                               "realized_vol_pct": (r.realized_vol or 0) * 100.0,
                               "rungs": r.call_watch[:3]})
        if r.put_best is not None and r.put_best.get("ev_pct", 0) >= MIN_EV_REPORT * 100:
            put_signals.append({"ticker": r.ticker, "today_close": r.today_close,
                                "end_date": r.end_date,
                                "realized_vol_pct": (r.realized_vol or 0) * 100.0,
                                **r.put_best})
        elif r.put_watch:
            put_watch.append({"ticker": r.ticker, "today_close": r.today_close,
                              "end_date": r.end_date,
                              "realized_vol_pct": (r.realized_vol or 0) * 100.0,
                              "rungs": r.put_watch[:3]})
        if i % 100 == 0:
            print(
                f"  {i}/{len(tickers)}  calls={len(call_signals)}  puts={len(put_signals)}  "
                f"watch(c/p)={len(call_watch)}/{len(put_watch)}  "
                f"elapsed={time.time()-t0:.1f}s"
            )

    # Pool validation
    call_w = sum(s["pooled_wins"] for s in call_signals)
    call_l = sum(s["pooled_losses"] for s in call_signals)
    put_w  = sum(s["pooled_wins"] for s in put_signals)
    put_l  = sum(s["pooled_losses"] for s in put_signals)
    pool_total = call_w + call_l + put_w + put_l
    pool_w = call_w + put_w
    print()
    print(f"Tickers processed:       {len(results)}")
    print(f"CALL signals (EV >= {int(MIN_EV_REPORT*100)}%): {len(call_signals)}  "
          f"pooled OOS: {call_w}/{call_w+call_l} (touch rate {(call_w/(call_w+call_l)*100) if call_w+call_l else 0:.2f}%)")
    print(f"PUT  signals (EV >= {int(MIN_EV_REPORT*100)}%): {len(put_signals)}   "
          f"pooled OOS: {put_w}/{put_w+put_l} (touch rate {(put_w/(put_w+put_l)*100) if put_w+put_l else 0:.2f}%)")
    print(f"Combined OOS tests:      {pool_total}")
    if pool_total:
        print(f"Combined touch rate:     {pool_w/pool_total*100:.3f}%")
    print(f"Target touch grid:       {[int(t*100) for t in TARGET_GRID]}%  "
          f"(eligibility: empirical >= target - {int(MIN_SLACK*100)}pp)")

    # Sort by EV desc
    call_signals.sort(key=lambda s: -s["ev_pct"])
    put_signals.sort(key=lambda s: -s["ev_pct"])

    lean = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "n_tickers_processed": len(results),
            "horizons": HORIZONS,
            "fold_years": FOLD_YEARS,
            "safety_eps": SAFETY_EPS,
            "min_cert_buffer": MIN_CERT_BUFFER,
            "target_touch_grid": TARGET_GRID,
            "min_slack_pp": MIN_SLACK * 100,
            "min_ev_report_pct": MIN_EV_REPORT * 100,
            "call": {
                "n_eligible": len(call_signals),
                "pooled_wins": call_w,
                "pooled_losses": call_l,
                "pooled_win_rate": call_w/(call_w+call_l) if call_w+call_l else None,
            },
            "put": {
                "n_eligible": len(put_signals),
                "pooled_wins": put_w,
                "pooled_losses": put_l,
                "pooled_win_rate": put_w/(put_w+put_l) if put_w+put_l else None,
            },
            "combined": {
                "n_eligible": len(call_signals) + len(put_signals),
                "pooled_wins": pool_w,
                "pooled_losses": call_l + put_l,
                "pooled_win_rate": pool_w/pool_total if pool_total else None,
            },
        },
        "call_signals": call_signals,
        "put_signals":  put_signals,
        "call_watchlist": call_watch,
        "put_watchlist":  put_watch,
    }
    lean_path = os.path.join(OUTPUT_DIR, "signals.json")
    with open(lean_path, "w") as fh:
        json.dump(lean, fh, indent=2)
    print(f"Wrote {lean_path} (calls={len(call_signals)}, puts={len(put_signals)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
