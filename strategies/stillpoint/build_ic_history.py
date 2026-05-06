"""Generate the complete historical Atomic Iron Condor signal log.

Walks the entire universe; for every (ticker, horizon, regime, q) combo
that passes IC eligibility (joint pooled OOS WR >=95%, every fold >=80%,
combined ROR >=50%), enumerates EVERY historical day on which the
regime gate was active and the trade would have fired.

For each fire, records:
  - publish_date  : the historical day the regime was active
  - ticker, horizon, regime (base|tight), q, width_pct
  - K_put_short, K_put_long, K_call_short, K_call_long
  - spot at fire, sigma at fire (vol20)
  - estimated combined credit, max-loss, ROR (BS-priced)
  - actual outcome at expiry: WIN (path stayed inside both strikes) or
    LOSS (one side breached) — known for fires where the forward
    window is contained in our data
  - actual close at expiry, actual path-min, actual path-max

Output: strategies/stillpoint/results/atomic_ic_history.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sp_common import (
    FOLD_YEARS, WARMUP_DAYS,
    SP_IC_HORIZONS, SP_IC_CONFORMAL_QS, SP_IC_MAX_BUFFER,
    SP_IC_PER_FOLD_WIN, SP_IC_POOLED_WIN, SP_IC_TARGET_ROR, SP_IC_WIDTHS,
    SP_SAFETY_EPS, SP_MIN_TRAIN_FIRES, SP_MIN_POOLED_TEST, SP_MIN_FOLDS,
    buffer_array, compute_features, fold_mask, list_tickers, load_series,
    stillpoint_mask, tight_mask, train_mask_for_fold,
)

import importlib.util as _ilu
_pr_spec = _ilu.spec_from_file_location(
    "credit_spread_pricing",
    os.path.join(os.path.dirname(_HERE), "credit_spread", "pricing.py"),
)
_pricing = _ilu.module_from_spec(_pr_spec)
sys.modules["credit_spread_pricing"] = _pricing
_pr_spec.loader.exec_module(_pricing)
bs_put = _pricing.bs_put
bs_call = _pricing.bs_call
realized_vol = _pricing.realized_vol


def best_ic_config(close, dates, regime, h, spot_today, sigma):
    """Return the highest-ROR IC config that passes eligibility, or None.

    Returns dict with keys: q, b_put, b_call, width_pct, ror, pooled_wr,
    pooled_n, fold_wrs.
    """
    bP = buffer_array(close, h, "put")
    bC = buffer_array(close, h, "call")
    n = len(dates)
    warmup = np.zeros(n, dtype=bool); warmup[WARMUP_DAYS:] = True
    base = warmup & regime & np.isfinite(bP) & np.isfinite(bC)
    if int(base.sum()) < SP_MIN_TRAIN_FIRES:
        return None
    iv = sigma * 1.30
    cal_days = max(int(h * 1.45), 1)
    T = cal_days / 365.0
    best = None
    for q in SP_IC_CONFORMAL_QS:
        pw = pl = 0; fold_count = 0; fold_wrs = []
        for y in FOLD_YEARS:
            tr = base & train_mask_for_fold(dates, y, h)
            te = base & fold_mask(dates, y)
            ok = np.zeros(n, dtype=bool); ok[: n - h] = True
            te = te & ok
            if tr.sum() < 30 or te.sum() == 0:
                continue
            bp = float(np.quantile(bP[tr], q)) + SP_SAFETY_EPS
            bc = float(np.quantile(bC[tr], q)) + SP_SAFETY_EPS
            if bp > SP_IC_MAX_BUFFER or bc > SP_IC_MAX_BUFFER:
                continue
            fold_count += 1
            j = (bP[te] <= bp) & (bC[te] <= bc)
            w = int(j.sum()); l = int((~j).sum())
            pw += w; pl += l; fold_wrs.append(w / max(w + l, 1))
        pooled = pw + pl
        if pooled < SP_MIN_POOLED_TEST or fold_count < SP_MIN_FOLDS:
            continue
        wr = pw / pooled
        if wr < SP_IC_POOLED_WIN:
            continue
        if any(fwr < SP_IC_PER_FOLD_WIN for fwr in fold_wrs):
            continue
        bp_f = float(np.quantile(bP[base], q)) + SP_SAFETY_EPS
        bc_f = float(np.quantile(bC[base], q)) + SP_SAFETY_EPS
        if bp_f > SP_IC_MAX_BUFFER or bc_f > SP_IC_MAX_BUFFER:
            continue
        # Best width
        K_ps = spot_today * (1 - bp_f); K_cs = spot_today * (1 + bc_f)
        best_ror_w = 0; best_w = None
        for w_pct in SP_IC_WIDTHS:
            width = spot_today * w_pct
            K_pl = K_ps - width; K_cl = K_cs + width
            if K_pl <= 0:
                continue
            cp = max(bs_put(spot_today, K_ps, T, iv) - bs_put(spot_today, K_pl, T, iv), 0) * 0.80
            cc = max(bs_call(spot_today, K_cs, T, iv) - bs_call(spot_today, K_cl, T, iv), 0) * 0.80
            cred = cp + cc
            ml = max(width - cred, 0.01)
            ror = cred / ml
            if ror > best_ror_w:
                best_ror_w = ror; best_w = w_pct
        if best_ror_w < SP_IC_TARGET_ROR:
            continue
        cand = {"q": q, "b_put": bp_f, "b_call": bc_f, "width_pct": best_w,
                "ror": best_ror_w, "pooled_wr": wr, "pooled_n": pooled,
                "fold_wrs": fold_wrs, "n_folds": fold_count}
        if best is None or cand["ror"] > best["ror"]:
            best = cand
    return best


def main():
    tickers = list_tickers()
    limit = os.environ.get("SP_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]

    eligible_combos = []  # all (ticker, h, regime_name, config)
    fires = []            # all historical fires across all combos

    n_processed = 0
    for ti, t in enumerate(tickers, 1):
        ts = load_series(t)
        if ts is None: continue
        f = compute_features(ts.close)
        rv = realized_vol(ts.close)
        if rv is None or rv <= 0: continue
        n_processed += 1
        spot_now = float(ts.close[-1])
        warmup = np.zeros(len(ts.dates), dtype=bool); warmup[WARMUP_DAYS:] = True

        for regime_name, regime in (("base", stillpoint_mask(f)),
                                     ("tight", tight_mask(f))):
            if int(regime.sum()) < SP_MIN_TRAIN_FIRES:
                continue
            for h in SP_IC_HORIZONS:
                cfg = best_ic_config(ts.close, ts.dates, regime, h, spot_now, rv)
                if cfg is None:
                    continue
                eligible_combos.append({
                    "ticker": t, "horizon": h, "regime": regime_name,
                    "q": cfg["q"], "width_pct": cfg["width_pct"],
                    "buffer_put_pct": cfg["b_put"] * 100.0,
                    "buffer_call_pct": cfg["b_call"] * 100.0,
                    "joint_oos_wr_pct": cfg["pooled_wr"] * 100.0,
                    "n_oos_tests": cfg["pooled_n"],
                    "estimated_ror_pct": cfg["ror"] * 100.0,
                    "n_folds": cfg["n_folds"],
                })
                # Enumerate every regime-active day after warmup
                buf_put = buffer_array(ts.close, h, "put")
                buf_call = buffer_array(ts.close, h, "call")
                base = warmup & regime & np.isfinite(buf_put) & np.isfinite(buf_call)
                idx = np.where(base)[0]
                # vol20 series for sigma at fire
                v20 = f.vol20
                for i in idx:
                    fire_date = str(ts.dates[i])
                    spot_at = float(ts.close[i])
                    sigma_at = float(v20[i]) if np.isfinite(v20[i]) else float("nan")
                    K_ps_fire = spot_at * (1 - cfg["b_put"])
                    K_cs_fire = spot_at * (1 + cfg["b_call"])
                    width = spot_at * cfg["width_pct"]
                    K_pl_fire = K_ps_fire - width
                    K_cl_fire = K_cs_fire + width
                    # Recompute estimated credit + ROR at fire-day vol if available
                    if np.isfinite(sigma_at) and sigma_at > 0:
                        iv_fire = sigma_at * 1.30
                        cal_days = max(int(h * 1.45), 1)
                        T_fire = cal_days / 365.0
                        cp = max(bs_put(spot_at, K_ps_fire, T_fire, iv_fire) - bs_put(spot_at, K_pl_fire, T_fire, iv_fire), 0) * 0.80
                        cc = max(bs_call(spot_at, K_cs_fire, T_fire, iv_fire) - bs_call(spot_at, K_cl_fire, T_fire, iv_fire), 0) * 0.80
                        cred_fire = cp + cc
                        ml_fire = max(width - cred_fire, 0.01)
                        ror_fire = cred_fire / ml_fire
                    else:
                        cred_fire = float("nan"); ml_fire = float("nan"); ror_fire = float("nan")
                    # Outcome (if forward window inside data)
                    outcome = None
                    close_at_expiry = path_min = path_max = None
                    if i + h < len(ts.dates):
                        window = ts.close[i + 1 : i + 1 + h]
                        path_min = float(window.min())
                        path_max = float(window.max())
                        close_at_expiry = float(ts.close[i + h])
                        won = (path_min >= K_ps_fire) and (path_max <= K_cs_fire)
                        outcome = "WIN" if won else "LOSS"
                    fires.append({
                        "publish_date": fire_date,
                        "ticker": t, "horizon": h, "regime": regime_name,
                        "q": cfg["q"], "width_pct": cfg["width_pct"],
                        "spot": spot_at, "sigma_vol20_pct": sigma_at * 100.0 if np.isfinite(sigma_at) else None,
                        "K_put_short": K_ps_fire, "K_put_long": K_pl_fire,
                        "K_call_short": K_cs_fire, "K_call_long": K_cl_fire,
                        "buffer_put_pct": cfg["b_put"] * 100.0,
                        "buffer_call_pct": cfg["b_call"] * 100.0,
                        "width": width,
                        "est_credit": cred_fire,
                        "est_max_loss": ml_fire,
                        "est_ror_pct": ror_fire * 100.0 if np.isfinite(ror_fire) else None,
                        "outcome": outcome,
                        "close_at_expiry": close_at_expiry,
                        "path_min": path_min,
                        "path_max": path_max,
                    })
        if ti % 100 == 0:
            print(f"  {ti}/{len(tickers)}  combos={len(eligible_combos)}  fires={len(fires)}")

    # Sort fires by date descending
    fires.sort(key=lambda r: (r["publish_date"], r["ticker"]), reverse=True)
    eligible_combos.sort(key=lambda r: -r["estimated_ror_pct"])

    summary = {
        "n_tickers_processed": n_processed,
        "n_eligible_combos": len(eligible_combos),
        "n_fires_total": len(fires),
        "n_fires_resolved_win": sum(1 for f in fires if f["outcome"] == "WIN"),
        "n_fires_resolved_loss": sum(1 for f in fires if f["outcome"] == "LOSS"),
        "n_fires_unresolved": sum(1 for f in fires if f["outcome"] is None),
        "min_pooled_wr_pct": SP_IC_POOLED_WIN * 100.0,
        "min_per_fold_wr_pct": SP_IC_PER_FOLD_WIN * 100.0,
        "min_combined_ror_pct": SP_IC_TARGET_ROR * 100.0,
        "max_per_leg_buffer_pct": SP_IC_MAX_BUFFER * 100.0,
        "horizons_tested": SP_IC_HORIZONS,
        "conformal_qs_tested": SP_IC_CONFORMAL_QS,
        "spread_widths_tested_pct": [w * 100.0 for w in SP_IC_WIDTHS],
    }

    out = {
        "summary": summary,
        "eligible_combos": eligible_combos,
        "fires": fires,
    }

    out_path = os.path.join(_HERE, "results", "atomic_ic_history.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print()
    print(f"Wrote {out_path}")
    print(f"  eligible_combos: {len(eligible_combos)}")
    print(f"  total fires: {len(fires)}")
    print(f"  resolved: {summary['n_fires_resolved_win']} wins / "
          f"{summary['n_fires_resolved_loss']} losses / "
          f"{summary['n_fires_unresolved']} unresolved")
    if summary['n_fires_resolved_win'] + summary['n_fires_resolved_loss']:
        wr = summary['n_fires_resolved_win'] / (summary['n_fires_resolved_win'] + summary['n_fires_resolved_loss'])
        print(f"  resolved fires WR: {wr*100:.3f}%")
    print()
    print("Most recent 25 fires:")
    print(f"{'date':<12} {'tk':<6} {'reg':<5} {'h':<3} {'q':<6} {'bP':>6} {'bC':>6} {'ROR':>6} {'outcome':<8}")
    for r in fires[:25]:
        out_str = r["outcome"] or "pending"
        print(f"{r['publish_date'][:10]:<12} {r['ticker']:<6} {r['regime']:<5} {r['horizon']:<3} "
              f"{r['q']:<6} {r['buffer_put_pct']:>5.2f}% {r['buffer_call_pct']:>5.2f}% "
              f"{r.get('est_ror_pct') or 0:>5.1f}% {out_str:<8}")


if __name__ == "__main__":
    main()
