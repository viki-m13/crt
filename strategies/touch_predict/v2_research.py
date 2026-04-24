"""TouchPredictor v2 — runs the novel regime grid on the liquid
universe, picks the most-profitable (highest EV) rule per ticker/side
that is ALSO consistently profitable (every fold positive $ P&L).

Reports a ranked list of rules with:
  - Regime name
  - Empirical touch rate (out-of-sample pooled)
  - Per-fold $ P&L (all must be > 0 to qualify)
  - Estimated OTM premium + profit + ROI + EV

This is the "quality over quantity" pass: we'd rather ship 3 rules
with +500% EV and zero losing years than 50 rules averaging +25%.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_common import (
    FOLD_YEARS, MIN_TEST_SAMPLES, MIN_TRAIN_SAMPLES, WARMUP_DAYS,
    OhlcvSeries, V2Features,
    actual_options_expiry, align_to_spy, buffer_down, buffer_up,
    compute_features, fold_mask, list_tickers, load_series,
    spy_context, train_mask_for_fold,
)
from v2_regimes import CALL_REGIMES, PUT_REGIMES
from pricing import best_otm_play


HORIZONS = [5, 7, 10, 14, 21]
TARGET_GRID = [0.90, 0.85, 0.75, 0.65]
SAFETY_EPS = 0.005
MIN_CERT_BUFFER = 0.015       # 1.5% minimum target move
MIN_PREMIUM = 0.50            # liquid-name premium floor
MIN_TOUCH_RATE = 0.80         # empirical pooled touch rate must clear this
MIN_ROI_PER_WIN = 0.30        # 30% ROI/win minimum
REQUIRE_PROFITABLE_EVERY_FOLD = True   # consistency requirement


def _evaluate(s: OhlcvSeries, feats: V2Features,
              side: str, regime_name: str,
              regime_mask: np.ndarray,
              horizon: int, target_rate: float) -> dict | None:
    close = s.close
    dates = s.dates
    buf = buffer_up(close, horizon) if side == "call" else buffer_down(close, horizon)
    warmup = np.zeros(len(dates), dtype=bool)
    warmup[WARMUP_DAYS:] = True
    base = regime_mask & warmup & np.isfinite(buf)
    if base.sum() < MIN_TRAIN_SAMPLES:
        return None

    q_pct = (1.0 - target_rate) * 100.0

    folds = []
    for year in FOLD_YEARS:
        tr = base & train_mask_for_fold(dates, year, horizon)
        te = base & fold_mask(dates, year)
        n = len(dates)
        test_ok = np.zeros(n, dtype=bool); test_ok[: n - horizon] = True
        te = te & test_ok
        if tr.sum() < MIN_TRAIN_SAMPLES or te.sum() == 0:
            continue
        train_q = float(np.percentile(buf[tr], q_pct))
        cert = max(train_q - SAFETY_EPS, 0.0)
        test_buf = buf[te]
        wins = int((test_buf >= cert).sum())
        losses = int((test_buf < cert).sum())
        folds.append({
            "year": year, "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "cert_buffer": cert, "wins": wins, "losses": losses,
        })

    pooled_wins   = sum(f["wins"] for f in folds)
    pooled_losses = sum(f["losses"] for f in folds)
    total_tests   = pooled_wins + pooled_losses
    if total_tests == 0 or not folds:
        return None

    touch_rate = pooled_wins / total_tests
    if touch_rate < MIN_TOUCH_RATE:
        return None

    # Final live cert_buffer: full-history quantile
    full_q = float(np.percentile(buf[base], q_pct))
    final_cert = max(full_q - SAFETY_EPS, 0.0)
    if final_cert < MIN_CERT_BUFFER:
        return None

    # Price the OTM option play at the final cert_buffer with today's spot
    spot = float(close[-1])
    last_date = str(dates[-1])
    exp_iso, exp_type, cal_days = actual_options_expiry(last_date, horizon)
    play = best_otm_play(
        side=side, spot=spot, buffer=final_cert,
        calendar_days_to_expiry=cal_days,
        realized_sigma=feats.realized_vol,
    )
    if play is None:
        return None
    if play.premium < MIN_PREMIUM:
        return None
    if play.roi < MIN_ROI_PER_WIN:
        return None

    # Consistency check: every fold's P&L must be positive
    fold_pnls = []
    for f in folds:
        pnl = f["wins"] * play.profit + f["losses"] * (-play.premium)
        fold_pnls.append({**f, "pnl": pnl})
    if REQUIRE_PROFITABLE_EVERY_FOLD and any(fp["pnl"] <= 0 for fp in fold_pnls):
        return None

    ev = touch_rate * play.roi - (1.0 - touch_rate)

    return {
        "ticker":       s.ticker,
        "side":         side,
        "regime":       regime_name,
        "horizon":      horizon,
        "target_rate":  target_rate,
        "spot":         spot,
        "expiry":       exp_iso,
        "expiry_type":  exp_type,
        "cal_days":     cal_days,
        "cert_buffer":  final_cert,
        "target_price": spot * (1 + final_cert) if side == "call" else spot * (1 - final_cert),
        "strike":       play.strike,
        "k_frac":       play.k_frac,
        "premium":      play.premium,
        "profit":       play.profit,
        "max_loss":     play.max_loss,
        "roi_win":      play.roi,
        "ev":           ev,
        "touch_rate":   touch_rate,
        "pooled_wins":  pooled_wins,
        "pooled_losses": pooled_losses,
        "n_folds":      len(folds),
        "folds":        fold_pnls,
        "realized_vol": feats.realized_vol,
        "iv_used":      play.implied_vol,
    }


def process_ticker(ticker: str) -> list[dict]:
    s = load_series(ticker)
    if s is None:
        return []
    feats = compute_features(s)
    results = []
    for side, regime_map in (("call", CALL_REGIMES), ("put", PUT_REGIMES)):
        for rname, rfn in regime_map.items():
            try:
                mask = rfn(feats, s.close, s.dates)
            except Exception as exc:
                print(f"  [ERR] {ticker} {side} {rname}: {exc}", file=sys.stderr)
                continue
            if not mask.any():
                continue
            for h in HORIZONS:
                for tgt in TARGET_GRID:
                    r = _evaluate(s, feats, side, rname, mask, h, tgt)
                    if r is not None:
                        results.append(r)
    return results


def main() -> int:
    tickers = list_tickers()
    # Ensure SPY context is pre-cached so rel-weakness regime is fast.
    _ = spy_context()
    t0 = time.time()
    all_results = []
    for i, t in enumerate(tickers, 1):
        try:
            all_results.extend(process_ticker(t))
        except Exception as exc:
            print(f"[ERR] {t}: {exc}", file=sys.stderr)
        if i % 20 == 0:
            print(f"  {i}/{len(tickers)}  rules_found={len(all_results)}  elapsed={time.time()-t0:.1f}s")

    # Sort by EV descending
    all_results.sort(key=lambda r: -r["ev"])
    # Keep top 1 per (ticker, side) by EV to get diversity in output
    best_by_tk_side = {}
    for r in all_results:
        k = (r["ticker"], r["side"])
        if k not in best_by_tk_side or r["ev"] > best_by_tk_side[k]["ev"]:
            best_by_tk_side[k] = r
    top_diverse = sorted(best_by_tk_side.values(), key=lambda r: -r["ev"])

    # Report
    print(f"\n{'='*100}")
    print(f"Rules found: {len(all_results)}  (unique ticker/side: {len(top_diverse)})")
    print(f"{'='*100}")
    total_tests = sum(r["pooled_wins"] + r["pooled_losses"] for r in top_diverse)
    total_wins  = sum(r["pooled_wins"] for r in top_diverse)
    total_pnl   = sum(sum(f["pnl"] for f in r["folds"]) for r in top_diverse)
    total_prem  = sum((r["pooled_wins"] + r["pooled_losses"]) * r["premium"] for r in top_diverse)
    if total_tests:
        print(f"Pooled touch rate:    {total_wins/total_tests*100:.2f}%")
        print(f"Total $ P&L:          ${total_pnl:+,.0f}")
        print(f"Total $ premium:      ${total_prem:,.0f}")
        print(f"Pooled ROI:           {total_pnl/total_prem*100:+.1f}%")
    print()
    print(f"{'ticker':<6} {'side':<4} {'regime':<14} {'h':>3} {'tgt%':>5} {'touch%':>6} {'prem$':>6} {'prof$':>6} {'ROI%':>5} {'EV%':>6} {'$P&L':>7} {'nF':>3}")
    print("-" * 120)
    for r in top_diverse[:40]:
        pnl = sum(f["pnl"] for f in r["folds"])
        print(f"{r['ticker']:<6} {r['side']:<4} {r['regime']:<14} "
              f"{r['horizon']:>3} {r['target_rate']*100:>4.0f}% "
              f"{r['touch_rate']*100:>5.1f}% "
              f"{r['premium']:>6.2f} {r['profit']:>6.2f} "
              f"{r['roi_win']*100:>4.0f}% {r['ev']*100:>+5.0f}% "
              f"${pnl:>+6.0f} {r['n_folds']:>3}")

    # Write output
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "horizons": HORIZONS, "target_grid": TARGET_GRID,
            "min_touch_rate": MIN_TOUCH_RATE, "min_cert_buffer": MIN_CERT_BUFFER,
            "min_premium": MIN_PREMIUM, "min_roi_per_win": MIN_ROI_PER_WIN,
            "require_profitable_every_fold": REQUIRE_PROFITABLE_EVERY_FOLD,
        },
        "summary": {
            "n_tickers_scanned": len(tickers),
            "n_rules_found": len(all_results),
            "n_unique_ticker_side": len(top_diverse),
            "pooled_touch_rate": (total_wins/total_tests) if total_tests else None,
            "pooled_roi": (total_pnl/total_prem) if total_prem else None,
            "total_dollar_pnl": total_pnl,
            "total_dollar_premium": total_prem,
        },
        "rules": top_diverse,
    }
    out_path = os.path.join(_HERE, "results", "v2_signals.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
