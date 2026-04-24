"""TouchPredictor v2 POOLED — cross-sectional pooling across liquid tickers.

Per-ticker stacked regimes (Connors TPS, multi_stack, panic_day) fire
~7 times per ticker over 11 years — too rare for per-ticker walk-forward.

Solution: POOL across the 94 liquid tickers. Each (ticker, day) matching
the regime is one training/test sample. Walk-forward by calendar year.
The certified touch buffer is the cross-ticker (1 - target)-quantile of
training buffers. A live signal today = any ticker whose features match
the regime TODAY.

Output: per-regime stats (not per-ticker), plus today's live fires.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_common import (
    FOLD_YEARS, WARMUP_DAYS,
    actual_options_expiry, buffer_down, buffer_up,
    compute_features, fold_mask, list_tickers, load_series,
    spy_context, train_mask_for_fold,
)
from v2_regimes import CALL_REGIMES, PUT_REGIMES
from pricing import best_otm_play


HORIZONS = [5, 7, 10, 14, 21]
TARGET_GRID = [0.90, 0.85, 0.75, 0.65]
SAFETY_EPS = 0.005
MIN_CERT_BUFFER = 0.015
MIN_POOLED_TRAIN = 30       # drastically reduced now that we pool
MIN_POOLED_TEST = 20
MIN_TOUCH_RATE = 0.75       # target min empirical touch rate


def _gather(side: str, regime_name: str, regime_fn):
    """Build (dates, buffers[h], regime_mask, ticker_id) arrays across
    all liquid tickers, concatenated. Returns dict per horizon."""
    all_dates  = []
    all_mask   = []
    all_ticker = []
    all_close  = []
    all_spot_features = []  # list of (ticker, dict) for live lookup
    buffers    = {h: [] for h in HORIZONS}

    _ = spy_context()  # pre-cache

    for t in list_tickers():
        s = load_series(t)
        if s is None: continue
        f = compute_features(s)
        try:
            m = regime_fn(f, s.close, s.dates)
        except Exception as exc:
            print(f"  [ERR] regime {regime_name} on {t}: {exc}", file=sys.stderr)
            continue
        warmup = np.zeros(len(s.dates), dtype=bool); warmup[WARMUP_DAYS:] = True
        mm = m & warmup
        if not mm.any():
            continue
        all_dates.append(s.dates[mm])
        all_mask.append(mm)
        all_ticker.extend([t] * int(mm.sum()))
        all_close.append(s.close[mm])
        for h in HORIZONS:
            buf = buffer_up(s.close, h) if side == "call" else buffer_down(s.close, h)
            buffers[h].append(buf[mm])

    if not all_dates:
        return None
    return {
        "dates":   np.concatenate(all_dates),
        "tickers": np.array(all_ticker),
        "close":   np.concatenate(all_close),
        "buffers": {h: np.concatenate(buffers[h]) for h in HORIZONS},
    }


def _evaluate_pooled(side: str, regime_name: str,
                     pool: dict, horizon: int, target_rate: float) -> dict | None:
    dates = pool["dates"]
    buf = pool["buffers"][horizon]
    close = pool["close"]

    valid = np.isfinite(buf)
    if valid.sum() < MIN_POOLED_TRAIN * 2:
        return None

    q_pct = (1.0 - target_rate) * 100.0

    # Walk-forward by year
    folds = []
    for year in FOLD_YEARS:
        tr = valid & (dates < np.datetime64(f"{year}-01-01"))
        te = valid & (dates >= np.datetime64(f"{year}-01-01")) \
                  & (dates < np.datetime64(f"{year+1}-01-01"))
        if tr.sum() < MIN_POOLED_TRAIN or te.sum() == 0:
            continue
        train_q = float(np.percentile(buf[tr], q_pct))
        cert = max(train_q - SAFETY_EPS, 0.0)
        test_buf = buf[te]
        wins   = int((test_buf >= cert).sum())
        losses = int((test_buf < cert).sum())
        folds.append({"year": year, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                      "cert_buffer": cert, "wins": wins, "losses": losses})

    pooled_wins   = sum(f["wins"] for f in folds)
    pooled_losses = sum(f["losses"] for f in folds)
    total = pooled_wins + pooled_losses
    if total < MIN_POOLED_TEST or not folds:
        return None
    touch_rate = pooled_wins / total
    if touch_rate < MIN_TOUCH_RATE:
        return None

    full_q = float(np.percentile(buf[valid], q_pct))
    final_cert = max(full_q - SAFETY_EPS, 0.0)
    if final_cert < MIN_CERT_BUFFER:
        return None

    return {
        "side": side, "regime": regime_name, "horizon": horizon,
        "target_rate": target_rate,
        "cert_buffer": final_cert,
        "touch_rate": touch_rate,
        "pooled_wins": pooled_wins, "pooled_losses": pooled_losses,
        "n_folds": len(folds), "folds": folds,
        "pooled_total": total,
    }


def _find_live_fires(regime_name: str, regime_fn, side: str) -> list[dict]:
    """Return tickers whose regime fires today. For each, note the
    feature values that triggered it."""
    fires = []
    for t in list_tickers():
        s = load_series(t)
        if s is None: continue
        f = compute_features(s)
        try:
            m = regime_fn(f, s.close, s.dates)
        except Exception:
            continue
        if m[-1]:  # fires today
            fires.append({
                "ticker": t, "spot": float(s.close[-1]),
                "as_of": str(s.dates[-1]), "realized_vol": f.realized_vol,
                "rsi2": float(f.rsi2[-1]) if np.isfinite(f.rsi2[-1]) else None,
                "rsi14": float(f.rsi14[-1]) if np.isfinite(f.rsi14[-1]) else None,
                "ret_1d": float(f.ret_1d[-1]) if np.isfinite(f.ret_1d[-1]) else None,
                "ret_5d": float(f.ret_5d[-1]) if np.isfinite(f.ret_5d[-1]) else None,
                "vol_z20": float(f.vol_z20[-1]) if np.isfinite(f.vol_z20[-1]) else None,
                "trend":  float(f.trend[-1]) if np.isfinite(f.trend[-1]) else None,
                "dd252":  float(f.dd252[-1]) if np.isfinite(f.dd252[-1]) else None,
            })
    return fires


def main() -> int:
    print("Gathering pooled buffers per regime...")
    t0 = time.time()

    all_rules: list[dict] = []

    for side, regime_map in (("call", CALL_REGIMES), ("put", PUT_REGIMES)):
        for rname, rfn in regime_map.items():
            t_start = time.time()
            pool = _gather(side, rname, rfn)
            if pool is None:
                print(f"  {side}/{rname}: no fires")
                continue
            n = len(pool["dates"])
            print(f"  {side}/{rname}: {n:,} pooled (ticker,day) fires (gather {time.time()-t_start:.1f}s)")
            # Try each horizon × target
            for h in HORIZONS:
                for tgt in TARGET_GRID:
                    r = _evaluate_pooled(side, rname, pool, h, tgt)
                    if r is None:
                        continue
                    all_rules.append(r)

    # Sort by touch_rate desc, then by cert_buffer desc
    all_rules.sort(key=lambda r: (-r["touch_rate"], -r["cert_buffer"]))

    print(f"\nTotal qualifying rules: {len(all_rules)}  ({time.time()-t0:.1f}s total)")
    print(f"\n{'side':<4} {'regime':<14} {'h':>3} {'tgt%':>5} {'touch%':>6} {'buf%':>6} {'tests':>6} {'folds':>5}")
    print("-"*60)
    for r in all_rules[:40]:
        print(f"{r['side']:<4} {r['regime']:<14} {r['horizon']:>3} "
              f"{r['target_rate']*100:>4.0f}% {r['touch_rate']*100:>5.1f}% "
              f"{r['cert_buffer']*100:>5.2f}% {r['pooled_total']:>6} "
              f"{r['n_folds']:>5}")

    # For top rules, find today's live fires and price OTM play
    # (only evaluate top rules to avoid thrash)
    live_sig_count = 0
    for r in all_rules[:20]:
        fires = _find_live_fires(r['regime'],
                                 (CALL_REGIMES if r['side']=='call' else PUT_REGIMES)[r['regime']],
                                 r['side'])
        exp_iso, exp_type, cal_days = ("", "", 0)
        live = []
        for fi in fires:
            if fi['realized_vol'] is None or np.isnan(fi['realized_vol']):
                continue
            exp_iso, exp_type, cal_days = actual_options_expiry(fi['as_of'], r['horizon'])
            play = best_otm_play(side=r['side'], spot=fi['spot'],
                                 buffer=r['cert_buffer'],
                                 calendar_days_to_expiry=cal_days,
                                 realized_sigma=fi['realized_vol'])
            if play is None:
                continue
            ev = r['touch_rate'] * play.roi - (1.0 - r['touch_rate'])
            live.append({**fi, "strike": play.strike, "premium": play.premium,
                         "profit": play.profit, "roi": play.roi, "ev": ev,
                         "expiry": exp_iso, "expiry_type": exp_type,
                         "cal_days": cal_days})
        r["live_fires"] = live
        live_sig_count += len(live)

    print(f"\nTotal LIVE signals across top 20 rules today: {live_sig_count}")
    print(f"\nTop live signals by EV:")
    flat_live = [(r, fi) for r in all_rules[:20] for fi in r.get('live_fires', [])]
    flat_live.sort(key=lambda p: -p[1]['ev'])
    for r, fi in flat_live[:20]:
        print(f"  {fi['ticker']:<6} {r['side']:<4} {r['regime']:<14} h={r['horizon']}d  "
              f"spot=${fi['spot']:.2f}  K=${fi['strike']:.2f}  prem=${fi['premium']:.2f}  "
              f"touch={r['touch_rate']*100:.1f}%  ROI/win={fi['roi']*100:.0f}%  EV={fi['ev']*100:+.0f}%")

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "horizons": HORIZONS, "target_grid": TARGET_GRID,
            "min_touch_rate": MIN_TOUCH_RATE, "min_cert_buffer": MIN_CERT_BUFFER,
            "min_pooled_test": MIN_POOLED_TEST,
        },
        "pooled_rules": all_rules,
        "live_count": live_sig_count,
    }
    out_path = os.path.join(_HERE, "results", "v2_pooled.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
