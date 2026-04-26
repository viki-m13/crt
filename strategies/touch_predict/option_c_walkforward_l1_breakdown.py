"""L1 calibration breakdown by k_short and horizon.

The headline walk-forward result said L1 calibrates at 99.57% across
5.75M predictions. We need to verify L1 isn't trivial: that 1-5% OTM
(rich-credit) strikes also calibrate, not just 25-50% OTM strikes
where premium is pennies.

Outputs three breakdowns of L1 walk-forward predictions:
  1) Calibration by k_short bucket
  2) Calibration by horizon
  3) Calibration by (k_short, horizon) cell
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_regimes import CALL_REGIMES, PUT_REGIMES
from option_c_research import K_SHORT_GRID, HORIZONS, _gather_fires
from option_c_certified import WORST_BUFFER_SAFETY, REGIME_FAMILY
from option_c_walkforward import (
    MIN_TICKER_FIRES, MIN_TICKER_FOLDS, TRAIN_WIN_RATE_FLOOR,
    vectorize_set, _per_ticker_stats,
)


def _print(*a, **k):
    print(*a, **k)
    sys.stdout.flush()


def main() -> int:
    cutoff_years = [2022, 2023, 2024, 2025, 2026]

    _print("Gathering fires (vectorized)...")
    t0 = time.time()
    cache: dict[tuple, dict] = {}
    for side, regime_map in (("put", CALL_REGIMES), ("call", PUT_REGIMES)):
        for rname, rfn in regime_map.items():
            for h in HORIZONS:
                fires = _gather_fires(side, rname, rfn, h)
                if fires:
                    cache[(side, rname, h)] = vectorize_set(fires, side)
    total = sum(len(v["spots"]) for v in cache.values())
    _print(f"  {len(cache)} sets, {total:,} fires, "
           f"elapsed={time.time()-t0:.1f}s")

    # Buckets: count predictions and wins keyed by (k_short, h, year)
    cell_n: dict[tuple, int] = defaultdict(int)
    cell_w: dict[tuple, int] = defaultdict(int)

    _print("\nWalk-forward L1 (vectorized)...")
    t0 = time.time()
    iters = 0
    n_iter_total = len(cache) * len(K_SHORT_GRID) * len(cutoff_years)

    for (side, rname, h), bag in cache.items():
        moves = bag["moves"]
        years = bag["years"]
        tcodes = bag["tickers"]
        n_tk = bag["n_tickers"]

        for k_short in K_SHORT_GRID:
            wins_arr = (moves <= k_short).astype(np.int8)

            for cy in cutoff_years:
                train_mask = years < cy
                test_mask = years == cy
                if not train_mask.any() or not test_mask.any():
                    iters += 1
                    continue

                t_total, t_wins, t_folds, t_every, t_worst = _per_ticker_stats(
                    years[train_mask], tcodes[train_mask],
                    moves[train_mask], wins_arr[train_mask],
                    n_tk,
                )
                t_wr = np.where(t_total > 0, t_wins / np.maximum(t_total, 1), 0.0)
                claim_100 = (
                    (t_wr >= TRAIN_WIN_RATE_FLOOR)
                    & (t_total >= MIN_TICKER_FIRES)
                    & (t_folds >= MIN_TICKER_FOLDS)
                )
                passes_L1 = (t_worst > 0) & (k_short >= t_worst + WORST_BUFFER_SAFETY)
                eligible = claim_100 & passes_L1   # L1 + claim

                test_tk = tcodes[test_mask]
                test_w = wins_arr[test_mask]
                if not eligible[test_tk].any():
                    iters += 1
                    continue

                mask = eligible[test_tk]
                test_w_eligible = test_w[mask]
                n = int(mask.sum())
                w = int(test_w_eligible.sum())
                key = (k_short, h, int(cy))
                cell_n[key] += n
                cell_w[key] += w
                iters += 1
        if iters and iters % 500 == 0:
            _print(f"  iter {iters}/{n_iter_total}, elapsed={time.time()-t0:.1f}s")
    _print(f"  done. elapsed={time.time()-t0:.1f}s")

    # ---- Breakdown 1: by k_short ----
    _print("\n" + "=" * 80)
    _print("L1 calibration by k_short bucket (lower k_short = richer credit)")
    _print("=" * 80)
    _print(f"{'k_short':>9}  {'n':>10}  {'wins':>10}  {'win%':>8}  worst-year")
    _print("-" * 80)
    by_k: dict[float, dict] = defaultdict(lambda: {"n": 0, "w": 0, "by_year": defaultdict(lambda: {"n": 0, "w": 0})})
    for (k_short, h, y), n in cell_n.items():
        w = cell_w[(k_short, h, y)]
        by_k[k_short]["n"] += n
        by_k[k_short]["w"] += w
        by_k[k_short]["by_year"][y]["n"] += n
        by_k[k_short]["by_year"][y]["w"] += w
    by_k_summary = {}
    for k_short in sorted(by_k):
        s = by_k[k_short]
        wr = s["w"] / s["n"] if s["n"] else None
        ywrs = [d["w"] / d["n"] for d in s["by_year"].values() if d["n"] >= 50]
        worst_yr = min(ywrs) if ywrs else None
        wr_str = f"{wr*100:.2f}%" if wr is not None else "—"
        worst_str = f"{worst_yr*100:.2f}%" if worst_yr is not None else "—"
        _print(f"{k_short*100:>7.0f}%  {s['n']:>10,}  {s['w']:>10,}  {wr_str:>8}  {worst_str}")
        by_k_summary[str(k_short)] = {
            "n": s["n"], "wins": s["w"], "win_rate": wr,
            "worst_year": worst_yr,
            "by_year": {str(y): dict(d) for y, d in s["by_year"].items()},
        }

    # ---- Breakdown 2: by horizon ----
    _print("\n" + "=" * 80)
    _print("L1 calibration by horizon (days to expiry)")
    _print("=" * 80)
    _print(f"{'horizon':>9}  {'n':>10}  {'wins':>10}  {'win%':>8}  worst-year")
    _print("-" * 80)
    by_h: dict[int, dict] = defaultdict(lambda: {"n": 0, "w": 0, "by_year": defaultdict(lambda: {"n": 0, "w": 0})})
    for (k_short, h, y), n in cell_n.items():
        w = cell_w[(k_short, h, y)]
        by_h[h]["n"] += n
        by_h[h]["w"] += w
        by_h[h]["by_year"][y]["n"] += n
        by_h[h]["by_year"][y]["w"] += w
    by_h_summary = {}
    for h in sorted(by_h):
        s = by_h[h]
        wr = s["w"] / s["n"] if s["n"] else None
        ywrs = [d["w"] / d["n"] for d in s["by_year"].values() if d["n"] >= 50]
        worst_yr = min(ywrs) if ywrs else None
        wr_str = f"{wr*100:.2f}%" if wr is not None else "—"
        worst_str = f"{worst_yr*100:.2f}%" if worst_yr is not None else "—"
        _print(f"{h:>9}  {s['n']:>10,}  {s['w']:>10,}  {wr_str:>8}  {worst_str}")
        by_h_summary[str(h)] = {
            "n": s["n"], "wins": s["w"], "win_rate": wr,
            "worst_year": worst_yr,
        }

    # ---- Breakdown 3: by (k_short, horizon) cell ----
    _print("\n" + "=" * 80)
    _print("L1 calibration by (k_short × horizon) — only cells with n ≥ 100")
    _print("=" * 80)
    by_kh: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "w": 0})
    for (k_short, h, y), n in cell_n.items():
        w = cell_w[(k_short, h, y)]
        by_kh[(k_short, h)]["n"] += n
        by_kh[(k_short, h)]["w"] += w
    # Print as grid
    ks_sorted = sorted({k for k, _ in by_kh})
    hs_sorted = sorted({h for _, h in by_kh})
    header = "k_short \\ h " + " ".join(f"{h:>6}" for h in hs_sorted)
    _print(header)
    _print("-" * len(header))
    by_kh_summary = {}
    for k_short in ks_sorted:
        cells = []
        for h in hs_sorted:
            s = by_kh.get((k_short, h), {"n": 0, "w": 0})
            if s["n"] >= 100:
                wr = s["w"] / s["n"]
                cells.append(f"{wr*100:>5.1f}%")
                by_kh_summary[f"{k_short}_{h}"] = {
                    "n": s["n"], "wins": s["w"], "win_rate": wr,
                }
            elif s["n"] > 0:
                cells.append(f"  n={s['n']:<3}".replace(" ", ".") if s["n"] < 100 else f"{s['n']:>6}")
            else:
                cells.append(" " * 6)
        _print(f"{k_short*100:>5.0f}%      " + " ".join(cells))

    # ---- Verdict: where does L1 calibrate? ----
    _print("\n" + "=" * 80)
    _print("VERDICT — does L1 calibrate where it matters (low k_short)?")
    _print("=" * 80)
    rich_credit_ks = [k for k in K_SHORT_GRID if k <= 0.05]
    pass_count = 0
    fail_count = 0
    for k in rich_credit_ks:
        s = by_k_summary.get(str(k))
        if s is None or s["n"] < 100:
            _print(f"  k_short={k*100:.0f}%  n={s['n'] if s else 0}  insufficient sample")
            continue
        wr = s["win_rate"] or 0
        wy = s["worst_year"] or 0
        ok = wr >= 0.98 and wy >= 0.95
        flag = "✓" if ok else "✗"
        _print(f"  {flag} k_short={k*100:.0f}%  n={s['n']:,}  win={wr*100:.2f}%  "
               f"worst-year={wy*100:.2f}%")
        if ok:
            pass_count += 1
        else:
            fail_count += 1
    _print(f"\nRich-credit k_short ≤ 5% buckets: {pass_count} pass, {fail_count} fail")
    if fail_count == 0 and pass_count > 0:
        _print("  → L1 IS NON-TRIVIAL: it delivers ≥98% even on tight strikes")
    elif pass_count == 0:
        _print("  → L1 IS TRIVIAL: only deep-OTM works, rich-credit fails")
    else:
        _print("  → L1 IS PARTIALLY VALID: mixed performance on rich credit")

    out_path = os.path.join(_HERE, "results", "option_c_walkforward_l1_breakdown.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "cutoff_years": cutoff_years,
            "tunable": {"min_fires": MIN_TICKER_FIRES,
                        "min_folds": MIN_TICKER_FOLDS,
                        "train_win_rate_floor": TRAIN_WIN_RATE_FLOOR},
            "by_k_short": by_k_summary,
            "by_horizon": by_h_summary,
            "by_k_short_x_horizon": by_kh_summary,
        }, fh, indent=2)
    _print(f"\nDetail saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
