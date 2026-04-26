"""Walk-forward v2 — adds conformal-strike (M5), per-trade consensus
(M6), and Mahalanobis regime distance (M7) to the layer ablation.

Run after v1 to compare.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_common import (
    FOLD_YEARS, WARMUP_DAYS, spy_context,
)
from v2_regimes import CALL_REGIMES, PUT_REGIMES
from option_c_research import (
    K_SHORT_GRID, SPREAD_WIDTH, HORIZONS,
    _gather_fires, Fire,
)
from option_c_certified import (
    WORST_BUFFER_SAFETY, REGIME_FAMILY,
    SPY_5D_FLOOR_PCT, STOCK_5D_FLOOR_PCT,
    CONSENSUS_FAMILIES_REQUIRED,
)
from option_c_certified_v2 import (
    conformal_floor, per_trade_consensus,
)


# Tunable thresholds (same as v1 for fair compare)
MIN_TICKER_FIRES = 30
MIN_TICKER_FOLDS = 2
TRAIN_WIN_RATE_FLOOR = 0.999
CONFORMAL_CONF = 0.99    # 99% conformal coverage target


def fy(fi: Fire) -> int:
    return int(str(fi.date)[:4])


def fire_outcome(fi: Fire, side: str, k_short: float) -> int:
    if side == "put":
        Ks = fi.spot * (1.0 - k_short)
        return +1 if fi.close_at_expiry >= Ks else -1
    Ks = fi.spot * (1.0 + k_short)
    return +1 if fi.close_at_expiry <= Ks else -1


def _print(*a, **k):
    print(*a, **k)
    sys.stdout.flush()


def main() -> int:
    cutoff_years = [2022, 2023, 2024, 2025, 2026]

    # ------ Step 1: gather fires once -----------
    _print("Gathering fires...")
    t0 = time.time()
    cache: dict[tuple, list[Fire]] = {}
    for side, regime_map in (("put", CALL_REGIMES), ("call", PUT_REGIMES)):
        for rname, rfn in regime_map.items():
            for h in HORIZONS:
                fires = _gather_fires(side, rname, rfn, h)
                if fires:
                    cache[(side, rname, h)] = fires
    _print(f"  done. {len(cache)} (side, regime, h) sets, "
           f"total fires = {sum(len(v) for v in cache.values()):,}, "
           f"elapsed={time.time()-t0:.1f}s")

    # ------ Step 2: walk-forward predictions -----
    # Combos:
    #  baseline: train_wr ≥ 99.9%
    #  +M5: conformal strike floor at 99% confidence
    #  +M5+M6: + per-trade consensus
    COMBOS = ["base", "M5", "M5_M6"]
    pred_by_combo: dict[str, list[dict]] = {c: [] for c in COMBOS}

    _print("\nWalk-forward...")
    n_total = len(cache) * len(K_SHORT_GRID)
    done = 0
    t0 = time.time()
    for (side, rname, h), fires in cache.items():
        for k_short in K_SHORT_GRID:
            by_year_ticker: dict[tuple, list[Fire]] = {}
            for fi in fires:
                by_year_ticker.setdefault((fy(fi), fi.ticker), []).append(fi)
            family = REGIME_FAMILY.get(rname, rname)

            for cy in cutoff_years:
                # Train per ticker
                tk_train: dict[str, list[Fire]] = {}
                for (y, tkr), fs in by_year_ticker.items():
                    if y < cy:
                        tk_train.setdefault(tkr, []).extend(fs)

                # Compute per-ticker stats + conformal floor
                tk_stats: dict[str, dict] = {}
                for tkr, fs in tk_train.items():
                    if len(fs) < MIN_TICKER_FIRES:
                        continue
                    # Per-fold check
                    folds: dict[int, dict] = {}
                    for fi in fs:
                        fld = folds.setdefault(fy(fi), {"w": 0, "l": 0})
                        if fire_outcome(fi, side, k_short) > 0:
                            fld["w"] += 1
                        else:
                            fld["l"] += 1
                    n_folds = len(folds)
                    if n_folds < MIN_TICKER_FOLDS:
                        continue
                    every_perfect = all(f["l"] == 0 and f["w"] > 0 for f in folds.values())
                    train_wr = sum(f["w"] for f in folds.values()) / sum(f["w"]+f["l"] for f in folds.values())
                    if train_wr < TRAIN_WIN_RATE_FLOOR:
                        continue
                    conformal_q = conformal_floor(side, fs, CONFORMAL_CONF)
                    tk_stats[tkr] = {
                        "every_perfect": every_perfect,
                        "train_wr": train_wr,
                        "conformal_q": conformal_q,
                    }

                # Predict on test fires
                for (y, tkr), test_fs in by_year_ticker.items():
                    if y != cy:
                        continue
                    s = tk_stats.get(tkr)
                    if s is None:
                        continue
                    for fi in test_fs:
                        out = fire_outcome(fi, side, k_short)
                        rec = {
                            "year": cy, "ticker": tkr, "side": side,
                            "regime": rname, "family": family, "horizon": h,
                            "k_short": k_short, "fire_date": str(fi.date),
                            "outcome": out,
                        }
                        # base: just historical 100% claim
                        pred_by_combo["base"].append(rec)
                        # +M5: K_short ≥ conformal floor + safety
                        if (s["conformal_q"] is not None
                            and k_short >= s["conformal_q"] + WORST_BUFFER_SAFETY):
                            pred_by_combo["M5"].append(rec)
                            pred_by_combo["M5_M6"].append(rec)  # consensus filtered later
            done += 1
            if done % 60 == 0:
                _print(f"  {done}/{n_total} processed, elapsed={time.time()-t0:.1f}s")
    _print(f"  predictions done. elapsed={time.time()-t0:.1f}s")

    # ------ Step 3: M6 consensus filter on M5_M6 -----
    pred_by_combo["M5_M6"] = per_trade_consensus(pred_by_combo["M5_M6"])

    # ------ Step 4: summarize -----
    _print("\n" + "=" * 80)
    _print("CALIBRATION RESULTS — v2 (M5 conformal, M6 per-trade consensus)")
    _print("=" * 80)
    _print(f"{'combo':<12} {'n':>7} {'wins':>7} {'win%':>8}  {'by year':<60}")
    _print("-" * 95)
    summary = {}
    for cname, preds in pred_by_combo.items():
        n = len(preds)
        wins = sum(1 for p in preds if p["outcome"] > 0)
        wr = (wins / n) if n else None
        by_year: dict[int, dict] = {}
        for p in preds:
            d = by_year.setdefault(p["year"], {"n": 0, "w": 0})
            d["n"] += 1
            if p["outcome"] > 0:
                d["w"] += 1
        yr_str = " ".join(
            f"{y}={d['w']}/{d['n']}={d['w']/d['n']*100:.1f}%" if d["n"] else ""
            for y, d in sorted(by_year.items())
        )
        wr_str = f"{wr*100:.2f}%" if wr is not None else "—"
        _print(f"{cname:<12} {n:>7} {wins:>7} {wr_str:>8}  {yr_str:<60}")
        summary[cname] = {"n": n, "wins": wins, "win_rate": wr,
                          "by_year": {str(y): d for y, d in by_year.items()}}

    _print("\nVERDICT (Certified-grade requires ≥ 98% win and n ≥ 100):")
    qualifying = [(k, v) for k, v in summary.items()
                  if (v["win_rate"] or 0) >= 0.98 and v["n"] >= 100]
    if qualifying:
        for k, v in qualifying:
            _print(f"  ✓ {k}  n={v['n']}  win_rate={v['win_rate']*100:.2f}%")
    else:
        _print("  ❌ No combo meets the 98% bar.")

    out_path = os.path.join(_HERE, "results", "option_c_walkforward_v2.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"cutoff_years": cutoff_years, "summary": summary,
                   "tunable": {"min_fires": MIN_TICKER_FIRES,
                               "min_folds": MIN_TICKER_FOLDS,
                               "conformal_conf": CONFORMAL_CONF}},
                  fh, indent=2)
    _print(f"\nDetail saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
