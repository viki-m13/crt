"""Per-cell certification — identify exactly which (side, k_short, horizon)
buckets walk-forward-calibrate at ≥98% with worst-year ≥95%.

The idea: rather than trying to make a single rule generalize across the
whole grid, certify only the specific grid cells that empirically deliver.
A trade is "Certified" iff its (side, k_short, h) cell passed walk-forward.
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

# Certification thresholds
MIN_CELL_N = 200            # need 200+ predictions to certify a cell
CELL_WIN_RATE_BAR = 0.98    # cell-level claim
CELL_WORST_YEAR_BAR = 0.95  # no single cutoff year may break below this


def _print(*a, **k):
    print(*a, **k)
    sys.stdout.flush()


def main() -> int:
    cutoff_years = [2022, 2023, 2024, 2025, 2026]

    _print("Gathering fires...")
    t0 = time.time()
    cache: dict[tuple, dict] = {}
    for side, regime_map in (("put", CALL_REGIMES), ("call", PUT_REGIMES)):
        for rname, rfn in regime_map.items():
            for h in HORIZONS:
                fires = _gather_fires(side, rname, rfn, h)
                if fires:
                    cache[(side, rname, h)] = vectorize_set(fires, side)
    total = sum(len(v["spots"]) for v in cache.values())
    _print(f"  {len(cache)} sets, {total:,} fires, elapsed={time.time()-t0:.1f}s")

    # Track (side, k_short, h, year) → (n, w)
    cell_n: dict[tuple, int] = defaultdict(int)
    cell_w: dict[tuple, int] = defaultdict(int)
    # For the LATEST cutoff year, record (side, k_short, h, ticker)
    # tuples that pass claim_100 + L1. Production whitelists these.
    LATEST_CY = max(cutoff_years)
    eligible_latest: set[tuple] = set()

    _print("\nWalk-forward L1 by (side, k_short, h, year)...")
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
                eligible = claim_100 & passes_L1

                test_tk = tcodes[test_mask]
                test_w = wins_arr[test_mask]
                if not eligible[test_tk].any():
                    iters += 1
                    continue

                mask = eligible[test_tk]
                n = int(mask.sum())
                w = int(test_w[mask].sum())
                key = (side, k_short, h, int(cy))
                cell_n[key] += n
                cell_w[key] += w

                # For latest cutoff: record (side, k_short, h, ticker)
                # whitelist for production tier-assignment.
                if cy == LATEST_CY:
                    elig_idx = np.where(eligible)[0]
                    for ti in elig_idx:
                        eligible_latest.add(
                            (side, float(k_short), int(h), bag["ticker_names"][int(ti)])
                        )
                iters += 1

    _print(f"  done. elapsed={time.time()-t0:.1f}s")

    # Aggregate per (side, k_short, h)
    cells: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "w": 0, "by_year": {}})
    for (side, k, h, y), n in cell_n.items():
        w = cell_w[(side, k, h, y)]
        c = cells[(side, k, h)]
        c["n"] += n
        c["w"] += w
        c["by_year"][y] = (n, w)

    # Mark certified cells
    certified = []
    for (side, k, h), s in cells.items():
        if s["n"] < MIN_CELL_N:
            continue
        wr = s["w"] / s["n"]
        ywrs = [w / n for n, w in s["by_year"].values() if n >= 30]
        if not ywrs:
            continue
        wy = min(ywrs)
        if wr >= CELL_WIN_RATE_BAR and wy >= CELL_WORST_YEAR_BAR:
            certified.append({
                "side": side, "k_short": k, "horizon": h,
                "n": s["n"], "wins": s["w"], "win_rate": wr,
                "worst_year": wy,
                "by_year": {str(y): {"n": n, "w": w}
                            for y, (n, w) in s["by_year"].items()},
            })
    certified.sort(key=lambda c: (c["side"], c["k_short"], c["horizon"]))

    _print("\n" + "=" * 90)
    _print(f"CERTIFIED CELLS — n ≥ {MIN_CELL_N}, win ≥ {CELL_WIN_RATE_BAR*100:.0f}%, "
           f"worst-year ≥ {CELL_WORST_YEAR_BAR*100:.0f}%")
    _print("=" * 90)
    _print(f"{'side':<5} {'k_short':>7} {'horizon':>7} {'n':>10} {'wins':>10} "
           f"{'win%':>8} {'worst-yr':>10}")
    _print("-" * 90)
    for c in certified:
        _print(f"{c['side']:<5} {c['k_short']*100:>6.0f}% {c['horizon']:>7} "
               f"{c['n']:>10,} {c['wins']:>10,} {c['win_rate']*100:>7.2f}% "
               f"{c['worst_year']*100:>9.2f}%")
    _print(f"\nTotal certified cells: {len(certified)}")

    # Group by side
    by_side = defaultdict(list)
    for c in certified:
        by_side[c["side"]].append((c["k_short"], c["horizon"]))
    for side in ("put", "call"):
        cells_list = sorted(by_side[side])
        _print(f"\n{side.upper()} side ({len(cells_list)} certified cells):")
        # Group by k_short for readable display
        ks_to_hs = defaultdict(list)
        for k, h in cells_list:
            ks_to_hs[k].append(h)
        for k in sorted(ks_to_hs):
            hs = sorted(ks_to_hs[k])
            _print(f"  k_short={k*100:.0f}%  horizons={hs}")

    # Identify rich-credit cells
    rich = [c for c in certified if c["k_short"] <= 0.10]
    _print(f"\nRich-credit certified cells (k_short ≤ 10%): {len(rich)}")
    for c in rich:
        _print(f"  ✓ {c['side']:<4} k={c['k_short']*100:.0f}% h={c['horizon']:>3}d  "
               f"n={c['n']:,}  win={c['win_rate']*100:.2f}%  "
               f"worst-yr={c['worst_year']*100:.2f}%")

    # Whitelist: only tickers in certified cells. Each entry is a
    # (side, k_short, h, ticker) tuple eligible at the latest cutoff.
    cert_cell_keys = {(c["side"], c["k_short"], c["horizon"]) for c in certified}
    cert_trades = sorted(
        (t for t in eligible_latest if (t[0], t[1], t[2]) in cert_cell_keys),
        key=lambda t: (t[0], t[1], t[2], t[3]),
    )
    _print(f"\nCertified-trade whitelist (cutoff={LATEST_CY}, "
           f"only cells that themselves certified): {len(cert_trades)}")

    out_path = os.path.join(_HERE, "results", "option_c_certified_cells.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "cutoff_years": cutoff_years,
            "latest_cutoff": LATEST_CY,
            "thresholds": {
                "min_cell_n": MIN_CELL_N,
                "cell_win_rate_bar": CELL_WIN_RATE_BAR,
                "cell_worst_year_bar": CELL_WORST_YEAR_BAR,
                "min_ticker_fires": MIN_TICKER_FIRES,
                "min_ticker_folds": MIN_TICKER_FOLDS,
                "train_win_rate_floor": TRAIN_WIN_RATE_FLOOR,
            },
            "certified_cells": certified,
            "certified_trades": [
                {"side": s, "k_short": k, "horizon": h, "ticker": tk}
                for (s, k, h, tk) in cert_trades
            ],
        }, fh, indent=2)
    _print(f"\nDetail saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
