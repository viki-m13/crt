"""Iron-condor sweep: combined put+call credit spread sized for 50%+ ROR
at 95%+ joint OOS WR.

Joint win condition (per historical day t):
  path_min(t..t+h) >= K_put_short    AND    path_max(t..t+h) <= K_call_short

Combined credit = credit_put + credit_call (both haircut applied)
Combined max loss = max(width_put, width_call) - combined_credit
Combined ROR = combined_credit / combined_max_loss

We sweep joint backtest over (regime, horizon, buffer cap, conformal q),
require:
  - joint walk-forward OOS WR >= 0.95 pooled, >= 0.85 per fold
  - estimated combined ROR >= 50%
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sp_common import (
    FOLD_YEARS, WARMUP_DAYS,
    buffer_array, compute_features, fold_mask, list_tickers, load_series,
    train_mask_for_fold,
)

# Reuse credit_spread/pricing.py
import importlib.util as _ilu
_pr_spec = _ilu.spec_from_file_location(
    "credit_spread_pricing",
    os.path.join(os.path.dirname(_HERE), "credit_spread", "pricing.py"),
)
_pricing = _ilu.module_from_spec(_pr_spec)
sys.modules["credit_spread_pricing"] = _pricing
_pr_spec.loader.exec_module(_pricing)
estimate_profit = _pricing.estimate_profit
realized_vol = _pricing.realized_vol
bs_put = _pricing.bs_put
bs_call = _pricing.bs_call
credit_spread_mid = _pricing.credit_spread_mid  # smile-aware leg pricing


def _mask(f, **kw):
    return (
        np.isfinite(f.vol20) & np.isfinite(f.vol5) & np.isfinite(f.compression)
        & np.isfinite(f.rsi14) & np.isfinite(f.range20)
        & np.isfinite(f.trend_flat) & np.isfinite(f.move5d)
        & (f.vol20 < kw["vol20_max"])
        & (f.compression < kw["comp_max"])
        & (f.range20 < kw["range_max"])
        & (f.trend_flat < kw["flat_max"])
        & (f.rsi14 >= kw["rsi_lo"]) & (f.rsi14 <= kw["rsi_hi"])
        & (np.abs(f.move5d) < kw["recent_max"])
    )


def ic_credit_estimate(spot, buf_put, buf_call, h, sigma, width_pct=0.05,
                       iv_mult=1.30, haircut=0.80):
    """Combined IC credit and max-loss per share."""
    cal_days = max(int(h * 1.45), 1)
    T = cal_days / 365.0
    iv = sigma * iv_mult
    width = spot * width_pct
    # Put leg
    K_p_short = spot * (1 - buf_put)
    K_p_long = K_p_short - width
    if K_p_long <= 0: return None
    credit_put = credit_spread_mid("put", spot, K_p_short, K_p_long, T, iv) * haircut
    # Call leg
    K_c_short = spot * (1 + buf_call)
    K_c_long = K_c_short + width
    credit_call = credit_spread_mid("call", spot, K_c_short, K_c_long, T, iv) * haircut
    combined_credit = credit_put + credit_call
    # Max loss for IC: width - combined credit (one side breaches; other keeps full credit)
    max_loss = max(width - combined_credit, 0.01)
    ror = combined_credit / max_loss
    ann_ror = ror * (365.0 / cal_days)
    return {
        "credit": combined_credit,
        "max_loss": max_loss,
        "width": width,
        "ror": ror,
        "ann_ror": ann_ror,
        "credit_put": credit_put,
        "credit_call": credit_call,
        "K_p_short": K_p_short, "K_c_short": K_c_short,
    }


def main():
    tickers = list_tickers()
    limit = os.environ.get("SP_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    print(f"Universe: {len(tickers)}")

    base = dict(vol20_max=0.40, comp_max=1.05, range_max=0.15,
                flat_max=0.04, rsi_lo=25.0, rsi_hi=75.0, recent_max=0.08)
    tight = dict(vol20_max=0.30, comp_max=0.95, range_max=0.10,
                 flat_max=0.025, rsi_lo=35.0, rsi_hi=65.0, recent_max=0.05)
    regimes = {"base": base, "tight": tight}

    # Per-leg quantile (joint WR is approximately 2q-1 for indep MR)
    qs = [0.97, 0.975, 0.98]
    horizons = [21, 30, 42, 63, 90, 126]
    bufs = [0.10, 0.15, 0.20, 0.25]
    safety = 0.005
    target_joint = 0.95
    target_per_fold_joint = 0.85
    min_train = 60
    min_pooled = 40
    min_folds = 4

    all_signals = []  # list of dicts with config + signal
    by_config = defaultdict(list)

    for ti, t in enumerate(tickers, 1):
        ts = load_series(t)
        if ts is None: continue
        f = compute_features(ts.close)
        warmup = np.zeros(len(ts.dates), dtype=bool)
        warmup[WARMUP_DAYS:] = True
        rv = realized_vol(ts.close)
        if rv is None or rv <= 0: continue
        spot = float(ts.close[-1])

        for rname, rg in regimes.items():
            sp = _mask(f, **rg)
            if int(sp.sum()) < min_train: continue
            for h in horizons:
                buf_put = buffer_array(ts.close, h, "put")
                buf_call = buffer_array(ts.close, h, "call")
                ok = np.zeros(len(ts.dates), dtype=bool)
                ok[: len(ts.dates) - h] = True
                base_mask = (warmup & sp & np.isfinite(buf_put) & np.isfinite(buf_call))
                if int(base_mask.sum()) < min_train: continue

                fold_data = []
                for year in FOLD_YEARS:
                    tr = base_mask & train_mask_for_fold(ts.dates, year, h)
                    te = base_mask & fold_mask(ts.dates, year) & ok
                    if tr.sum() < 30 or te.sum() == 0: continue
                    fold_data.append((tr, te))
                if len(fold_data) < min_folds: continue

                for q in qs:
                    pooled_w = pooled_l = 0
                    fold_wrs = []
                    for tr, te in fold_data:
                        b_put = float(np.quantile(buf_put[tr], q)) + safety
                        b_call = float(np.quantile(buf_call[tr], q)) + safety
                        # JOINT win: both legs safe
                        joint_safe = (buf_put[te] <= b_put) & (buf_call[te] <= b_call)
                        w = int(joint_safe.sum())
                        l = int((~joint_safe).sum())
                        pooled_w += w
                        pooled_l += l
                        fold_wrs.append(w / max(w + l, 1))
                    pooled = pooled_w + pooled_l
                    if pooled < min_pooled: continue
                    wr = pooled_w / pooled
                    if wr < target_joint: continue
                    if any(fwr < target_per_fold_joint for fwr in fold_wrs): continue

                    # Live (full-history) buffers
                    b_put_final = float(np.quantile(buf_put[base_mask], q)) + safety
                    b_call_final = float(np.quantile(buf_call[base_mask], q)) + safety
                    if b_put_final > 0.30 or b_call_final > 0.30: continue

                    # IC ROR estimate
                    ic = ic_credit_estimate(spot, b_put_final, b_call_final, h, rv)
                    if ic is None: continue

                    for mb in bufs:
                        if b_put_final <= mb and b_call_final <= mb:
                            sig = {
                                "ticker": t, "regime": rname, "h": h, "q": q,
                                "max_buf": mb,
                                "buf_put": b_put_final, "buf_call": b_call_final,
                                "wr": wr, "n_test": pooled,
                                "ror": ic["ror"], "ann_ror": ic["ann_ror"],
                                "credit": ic["credit"], "max_loss": ic["max_loss"],
                            }
                            by_config[(rname, h, q, mb)].append(sig)
                            all_signals.append(sig)
                            break  # one tier per signal
        if ti % 100 == 0:
            print(f"# {ti}/{len(tickers)}  total_signals={len(all_signals)}")

    print()
    print(f"{'reg':<6} {'h':<4} {'q':<5} {'maxB':<5} "
          f"{'#elig':>5} {'#>=50%':>6} {'medROR':>7} {'medAnn':>9}")
    print("-" * 70)
    for key, sigs in sorted(by_config.items(),
                              key=lambda kv: -sum(1 for s in kv[1] if s["ror"]>=0.50)):
        rname, h, q, mb = key
        n50 = sum(1 for s in sigs if s["ror"] >= 0.50)
        med_ror = float(np.median([s["ror"] for s in sigs]))
        med_ann = float(np.median([s["ann_ror"] for s in sigs]))
        if not sigs: continue
        print(f"{rname:<6} {h:<4} {q:<5} {mb*100:>4.1f}% "
              f"{len(sigs):>5} {n50:>6} {med_ror*100:>6.2f}% {med_ann*100:>8.1f}%")

    print()
    print("Top 30 IC signals by per-trade ROR (with WR >= 95%):")
    print(f"{'tk':<6} {'reg':<6} {'h':<4} {'bufP':>6} {'bufC':>6} {'wr':>6} {'ROR':>7} {'AnnROR':>9}")
    all_signals.sort(key=lambda s: -s["ror"])
    seen = set()
    for s in all_signals:
        k = (s["ticker"], s["regime"], s["h"])
        if k in seen: continue
        seen.add(k)
        print(f"{s['ticker']:<6} {s['regime']:<6} {s['h']:<4} {s['buf_put']*100:>5.2f}% "
              f"{s['buf_call']*100:>5.2f}% {s['wr']*100:>5.2f}% {s['ror']*100:>6.2f}% "
              f"{s['ann_ror']*100:>8.1f}%")
        if len(seen) >= 30: break


if __name__ == "__main__":
    main()
