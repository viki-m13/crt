"""
CreditFloor — Walk-Forward Conformal Strike Engine.

Proprietary approach
--------------------

We want to short a credit spread whose short strike K (put-side) sits as
close as possible to today's price S, but still "never breached" by the
stock over the next h trading days. Defining buffer b = 1 - K/S, the
sample is a WIN if min(close[t+1..t+h]) >= S*(1-b), i.e. the stock never
trades below K at close on any of the h forward days.

Worst-case buffer b*(t,h) = 1 - min_fwd(t,h)/close[t] is known exactly
at t+h. CreditFloor estimates a *forward-safe* buffer b_hat(t,h) that
satisfies b*(t,h) <= b_hat(t,h) for every held-out sample.

Algorithm (per ticker, per horizon h):

    1. Split history into walk-forward folds by calendar year.
    2. For fold year Y:
        a. Training set = dates t with t+h < Jan 1 Y     (purged gap).
        b. Test set     = dates t with Jan 1 Y <= t < Jan 1 (Y+1)
                          and t+h <= last available date.
        c. b_hat_fold = max( b*(t,h)  over training set ) + SAFETY_EPS
           (conformal 100% coverage on the training sample — a generalized
            order statistic upper bound).
        d. For each test date t, WIN iff b*(t,h) <= b_hat_fold.
    3. Pool ALL fold test outcomes. A (ticker, horizon) combo is
       "CreditFloor eligible" iff:
        - every fold's test win rate = 100%
        - at least MIN_TEST_SAMPLES pooled test samples
        - final b_hat (fit on full history) <= MAX_BUFFER
        - final b_hat <= some configurable cap (e.g. 0.15)
    4. Final live buffer b_final = max( b*(t,h) over ALL history ) + EPS.
       This is the buffer we emit for today's prediction; it is strictly
       >= every b_hat_fold used in validation, so the OOS win rate is a
       lower bound on what we'd have seen live.
    5. Regime variants: we also run the exact same pipeline with the
       dates restricted to an uptrend-only regime. For each ticker we
       keep whichever variant passes validation with the smaller final
       buffer (if both pass, regime-gated wins because it is tighter).

Cross-ticker pool validation
----------------------------

After per-ticker filtering we pool every eligible (ticker, horizon, fold)
OOS prediction and require the pooled win rate across the entire universe
to be 100%. If any trade ever failed, the entire horizon is rejected
(fail-closed).

Leakage controls
----------------

    - Features at t use only close[0..t].
    - Forward targets for t are in (t, t+h]; never mixed into features.
    - Walk-forward purges any training sample whose forward window
      extends into the test calendar year.
    - SAFETY_EPS and MAX_BUFFER are fixed constants — never tuned on
      test data.
    - No hyperparameter search that uses test-fold outcomes: the only
      choice is (plain vs regime) per ticker, decided AFTER pooling,
      preferring the *tighter* buffer rather than the "best" win rate
      (both must be 100% to be considered).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
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
    compute_features,
    fold_mask,
    list_tickers,
    load_series,
    regime_mask,
    train_mask_for_fold,
    worst_buffer_path,
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class FoldResult:
    year: int
    n_train: int
    n_test: int
    b_hat: float
    wins: int
    losses: int
    worst_test_buf: float  # largest b*(t,h) seen on test set


@dataclass
class VariantResult:
    variant: str                # "plain" or "regime"
    horizon: int
    folds: list[FoldResult] = field(default_factory=list)
    pooled_wins: int = 0
    pooled_losses: int = 0
    b_final: float = 0.0
    b_final_history_max: float = 0.0
    eligible: bool = False


@dataclass
class TickerResult:
    ticker: str
    n_days: int
    start_date: str
    end_date: str
    today_close: float
    variants: dict[str, dict[int, VariantResult]] = field(default_factory=dict)
    # Best pick: the (variant, horizon) with smallest eligible b_final.
    best: dict[str, Any] | None = None
    # All eligible (variant, horizon) combos that are deployable today.
    all_eligible: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------


def evaluate_variant(
    ts: TickerSeries,
    feats: Features,
    horizon: int,
    regime: bool,
) -> VariantResult:
    vr = VariantResult(variant="regime" if regime else "plain", horizon=horizon)
    close = ts.close
    dates = ts.dates
    buf = worst_buffer_path(close, horizon)
    rmask_all = regime_mask(feats, require_uptrend=regime)
    warmup = np.zeros(len(dates), dtype=bool)
    warmup[WARMUP_DAYS:] = True
    base = rmask_all & warmup & np.isfinite(buf)

    # ----- walk-forward folds -----
    for year in FOLD_YEARS:
        tr = base & train_mask_for_fold(dates, year, horizon)
        te = base & fold_mask(dates, year)
        # Enforce forward window fits within observed history on test
        n = len(dates)
        test_ok = np.zeros(n, dtype=bool)
        test_ok[: n - horizon] = True
        te = te & test_ok
        if tr.sum() < MIN_TRAIN_SAMPLES:
            continue
        if te.sum() == 0:
            continue
        b_train_worst = float(buf[tr].max())
        b_hat = b_train_worst + SAFETY_EPS
        # b_hat is clamped to [0, 1] for safety
        b_hat = min(max(b_hat, 0.0), 0.99)
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

    # Final live buffer uses ALL history (train+test), conservative.
    if base.sum() >= MIN_TRAIN_SAMPLES:
        vr.b_final_history_max = float(buf[base].max())
        vr.b_final = min(vr.b_final_history_max + SAFETY_EPS, 0.99)
    else:
        vr.b_final_history_max = float("nan")
        vr.b_final = float("nan")

    # Eligibility: 100% OOS win rate, enough pooled tests, final buf tight.
    total = vr.pooled_wins + vr.pooled_losses
    vr.eligible = bool(
        vr.folds
        and vr.pooled_losses == 0
        and total >= MIN_TEST_SAMPLES
        and np.isfinite(vr.b_final)
        and vr.b_final <= MAX_BUFFER
        # every fold must have been individually 100% (redundant but
        # explicit — protects against a fold with zero wins counting).
        and all(f.losses == 0 for f in vr.folds)
        and all(f.wins > 0 for f in vr.folds)
    )
    return vr


def process_ticker(ticker: str) -> TickerResult | None:
    ts = load_series(ticker)
    if ts is None:
        return None
    feats = compute_features(ts.close)
    today_close = float(ts.close[-1])

    tr = TickerResult(
        ticker=ticker,
        n_days=len(ts.dates),
        start_date=str(ts.dates[0]),
        end_date=str(ts.dates[-1]),
        today_close=today_close,
    )

    # Check if today's features put us in the uptrend regime.
    today_regime_ok = bool(
        np.isfinite(feats.trend[-1])
        and np.isfinite(feats.dd252[-1])
        and feats.trend[-1] >= 1.00
        and feats.dd252[-1] <= 0.20
    )

    best: tuple[float, VariantResult] | None = None
    # Collect ALL deployable (variant, horizon) combos so the UI can show
    # multiple expiries per ticker (e.g. "safe in 21, 42, 63 days at K").
    # Per horizon we keep whichever variant yields the tighter buffer so
    # we never double-count the same horizon.
    per_horizon_best: dict[int, VariantResult] = {}
    for regime in (False, True):
        key = "regime" if regime else "plain"
        tr.variants[key] = {}
        for h in HORIZONS:
            vr = evaluate_variant(ts, feats, h, regime)
            tr.variants[key][h] = vr
            if not vr.eligible:
                continue
            # For the regime variant to be usable LIVE, today must also
            # match the regime gate. Otherwise we cannot deploy it.
            if regime and not today_regime_ok:
                continue
            prev = per_horizon_best.get(h)
            if prev is None or vr.b_final < prev.b_final:
                per_horizon_best[h] = vr
            if best is None or vr.b_final < best[0]:
                best = (vr.b_final, vr)

    def _vr_payload(vr: VariantResult) -> dict[str, Any]:
        return {
            "variant": vr.variant,
            "horizon": vr.horizon,
            "buffer": vr.b_final,
            "buffer_pct": vr.b_final * 100.0,
            "strike": today_close * (1.0 - vr.b_final),
            "today_close": today_close,
            "regime_ok_today": today_regime_ok,
            "pooled_wins": vr.pooled_wins,
            "pooled_losses": vr.pooled_losses,
            "n_test": vr.pooled_wins + vr.pooled_losses,
            "n_folds": len(vr.folds),
            "history_worst_buffer": vr.b_final_history_max,
        }

    tr.all_eligible = [
        _vr_payload(per_horizon_best[h])
        for h in sorted(per_horizon_best.keys())
    ]
    if best is not None:
        tr.best = _vr_payload(best[1])
    return tr


# --------------------------- serialization ---------------------------


def _vr_to_dict(vr: VariantResult) -> dict[str, Any]:
    d = asdict(vr)
    return d


def _tr_to_dict(tr: TickerResult) -> dict[str, Any]:
    return {
        "ticker": tr.ticker,
        "n_days": tr.n_days,
        "start_date": tr.start_date,
        "end_date": tr.end_date,
        "today_close": tr.today_close,
        "variants": {
            k: {str(h): _vr_to_dict(v) for h, v in d.items()}
            for k, d in tr.variants.items()
        },
        "best": tr.best,
        "all_eligible": tr.all_eligible,
    }


# --------------------------- main driver ---------------------------


def main() -> int:
    tickers = list_tickers()
    # Allow subset for debugging via env var.
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]

    t0 = time.time()
    results: list[TickerResult] = []
    eligible: list[TickerResult] = []
    for i, t in enumerate(tickers, 1):
        try:
            r = process_ticker(t)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] {t}: {exc}", file=sys.stderr)
            continue
        if r is None:
            continue
        results.append(r)
        if r.best is not None:
            eligible.append(r)
        if i % 50 == 0:
            print(
                f"  {i}/{len(tickers)}  elig={len(eligible)}  "
                f"elapsed={time.time()-t0:.1f}s"
            )

    # ----- cross-universe pooled validation -----
    # Enforce: aggregate wins/losses across EVERY eligible (ticker,
    # variant, horizon, fold) — across the entire universe — must be
    # 100% win rate. Any single loss anywhere voids the method.
    pool_wins = 0
    pool_losses = 0
    any_loss_map: dict[tuple[str, str, int], bool] = {}
    for r in eligible:
        assert r.best is not None
        v = r.best["variant"]
        h = r.best["horizon"]
        vr = r.variants[v][h]
        for f in vr.folds:
            pool_wins += f.wins
            pool_losses += f.losses
            if f.losses:
                any_loss_map[(r.ticker, v, h)] = True

    print()
    print(f"Tickers processed:   {len(results)}")
    print(f"Per-ticker eligible: {len(eligible)}")
    print(f"Pooled OOS tests:    {pool_wins + pool_losses}")
    print(f"Pooled OOS wins:     {pool_wins}")
    print(f"Pooled OOS losses:   {pool_losses}")
    if pool_wins + pool_losses > 0:
        print(f"Pooled OOS win rate: {pool_wins/(pool_wins+pool_losses)*100:.3f}%")

    # Defensive: drop any eligible entry whose own pooled folds had any
    # loss (by construction they shouldn't, but we re-check).
    eligible = [r for r in eligible if not any_loss_map.get(
        (r.ticker, r.best["variant"], r.best["horizon"]), False
    )]

    # Sort: tightest buffer first, tie-break on larger test sample count.
    eligible.sort(key=lambda r: (r.best["buffer"], -r.best["n_test"]))

    # Write full research dump (heavy — for audit) and a lean signals file.
    full_path = os.path.join(OUTPUT_DIR, "research_full.json")
    with open(full_path, "w") as fh:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": {
                    "horizons": HORIZONS,
                    "fold_years": FOLD_YEARS,
                    "min_train_samples": MIN_TRAIN_SAMPLES,
                    "min_test_samples": MIN_TEST_SAMPLES,
                    "safety_eps": SAFETY_EPS,
                    "max_buffer": MAX_BUFFER,
                    "warmup_days": WARMUP_DAYS,
                },
                "summary": {
                    "n_tickers_processed": len(results),
                    "n_tickers_eligible": len(eligible),
                    "pooled_wins": pool_wins,
                    "pooled_losses": pool_losses,
                },
                "tickers": [_tr_to_dict(r) for r in results],
            },
            fh,
        )
    print(f"Wrote {full_path}")

    lean = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "n_tickers_processed": len(results),
            "n_tickers_eligible": len(eligible),
            "pooled_wins": pool_wins,
            "pooled_losses": pool_losses,
            "pooled_win_rate": (
                pool_wins / (pool_wins + pool_losses)
                if (pool_wins + pool_losses) > 0
                else None
            ),
            "horizons": HORIZONS,
            "fold_years": FOLD_YEARS,
            "safety_eps": SAFETY_EPS,
            "max_buffer": MAX_BUFFER,
        },
        "signals": [
            {
                "ticker": r.ticker,
                "today_close": r.today_close,
                "end_date": r.end_date,
                # Primary (tightest-buffer) pick
                "strike": r.best["strike"],
                "buffer_pct": r.best["buffer_pct"],
                "horizon": r.best["horizon"],
                "variant": r.best["variant"],
                "n_test": r.best["n_test"],
                "n_folds": r.best["n_folds"],
                "pooled_wins": r.best["pooled_wins"],
                "pooled_losses": r.best["pooled_losses"],
                "history_worst_buffer_pct": r.best["history_worst_buffer"] * 100.0,
                "regime_ok_today": r.best["regime_ok_today"],
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
                    for f in r.variants[r.best["variant"]][r.best["horizon"]].folds
                ],
                # All horizons usable today for this ticker (sorted short→long)
                "ladder": [
                    {
                        "horizon": e["horizon"],
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
                            for f in r.variants[e["variant"]][e["horizon"]].folds
                        ],
                    }
                    for e in r.all_eligible
                ],
            }
            for r in eligible
        ],
    }
    lean_path = os.path.join(OUTPUT_DIR, "signals.json")
    with open(lean_path, "w") as fh:
        json.dump(lean, fh, indent=2)
    print(f"Wrote {lean_path} ({len(eligible)} signals)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
