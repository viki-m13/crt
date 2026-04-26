"""Walk-forward calibration v4 — fully numpy-vectorized.

Goal
----
A "Certified" rule should deliver ≥98% real-future win rate when it
claims ≥98%. Walk-forward across cutoff years 2022-2026 measures
whether layer combos calibrate on truly-unseen data.

Vectorization
-------------
For each (side, regime, h) cache entry we have arrays of length N
(spots, closes, years, ticker_codes, moves). For every (k_short,
cutoff_year) iteration we use only numpy primitives:
  - boolean masks for train/test split
  - np.bincount for per-ticker counts
  - np.maximum.reduceat for per-ticker max-move
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
from v2_common import spy_context
from v2_regimes import CALL_REGIMES, PUT_REGIMES
from option_c_research import K_SHORT_GRID, SPREAD_WIDTH, HORIZONS, _gather_fires
from option_c_certified import (
    WORST_BUFFER_SAFETY, REGIME_FAMILY,
    SPY_5D_FLOOR_PCT, CONSENSUS_FAMILIES_REQUIRED,
)


# Tunable thresholds
MIN_TICKER_FIRES = 30
MIN_TICKER_FOLDS = 2
TRAIN_WIN_RATE_FLOOR = 0.999  # rule must claim ~100% historically


def _print(*a, **k):
    print(*a, **k)
    sys.stdout.flush()


def vectorize_set(fires_list, side: str):
    n = len(fires_list)
    spots = np.empty(n, dtype="float64")
    closes = np.empty(n, dtype="float64")
    years = np.empty(n, dtype="int32")
    tcodes = np.empty(n, dtype="int32")
    book = {}
    for i, fi in enumerate(fires_list):
        spots[i] = fi.spot
        closes[i] = fi.close_at_expiry
        years[i] = int(str(fi.date)[:4])
        if fi.ticker not in book:
            book[fi.ticker] = len(book)
        tcodes[i] = book[fi.ticker]
    if side == "put":
        moves = (spots - closes) / spots
    else:
        moves = (closes - spots) / spots
    moves = np.clip(moves, 0.0, None)
    return {
        "spots": spots, "closes": closes, "years": years,
        "tickers": tcodes, "moves": moves,
        "ticker_names": list(book.keys()),
        "n_tickers": len(book),
        "side": side,
    }


def _per_ticker_stats(years_train: np.ndarray,
                       tcodes_train: np.ndarray,
                       moves_train: np.ndarray,
                       wins_train: np.ndarray,
                       n_tickers: int):
    """Compute per-ticker: total_fires, total_wins, n_folds, every_perfect,
    worst_move (all fully vectorized)."""
    # total fires + total wins per ticker
    total = np.bincount(tcodes_train, minlength=n_tickers).astype(np.int64)
    wins = np.bincount(tcodes_train, weights=wins_train, minlength=n_tickers).astype(np.int64)

    # n_folds = distinct (ticker, year) pairs per ticker
    # Encode (ticker, year) into a single int — assume year fits in int32
    if len(tcodes_train) > 0:
        ty_keys = tcodes_train.astype(np.int64) * 10000 + years_train.astype(np.int64)
        unique_keys = np.unique(ty_keys)
        # n_folds[t] = count of unique keys whose ticker == t
        unique_tickers = (unique_keys // 10000).astype(np.int64)
        n_folds = np.bincount(unique_tickers, minlength=n_tickers)
        # every_perfect: for each (ticker, year) bucket, count losses;
        # ticker is "every perfect" iff every bucket has 0 losses AND ≥1 win.
        # Compute per-(ticker, year) loss count via bincount on ty_keys.
        # Use a continuous index for each unique key.
        idx_in_unique = np.searchsorted(unique_keys, ty_keys)
        bucket_loss = np.bincount(idx_in_unique, weights=(1 - wins_train),
                                   minlength=len(unique_keys))
        bucket_total = np.bincount(idx_in_unique, minlength=len(unique_keys))
        bucket_has_loss = (bucket_loss > 0)
        # Aggregate: for each ticker, are all of its buckets loss-free?
        # Loop tickers to compute this; n_tickers is small (~94).
        every_perfect = np.ones(n_tickers, dtype=bool)
        any_loss_per_ticker = np.zeros(n_tickers, dtype=bool)
        for ui, has_loss in enumerate(bucket_has_loss):
            t = int(unique_tickers[ui])
            if has_loss:
                any_loss_per_ticker[t] = True
        every_perfect = ~any_loss_per_ticker
    else:
        n_folds = np.zeros(n_tickers, dtype=np.int64)
        every_perfect = np.zeros(n_tickers, dtype=bool)

    # worst_move per ticker — use np.maximum.at (in-place, vectorized)
    worst = np.zeros(n_tickers, dtype=np.float64)
    np.maximum.at(worst, tcodes_train, moves_train)
    return total, wins, n_folds, every_perfect, worst


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
    total = sum(v["n_tickers"] * 0 + len(v["spots"]) for v in cache.values())
    _print(f"  {len(cache)} sets, {total:,} fires, "
           f"elapsed={time.time()-t0:.1f}s")

    COMBO_NAMES = ["base", "L1", "L2", "L1_L2"]
    pred_by_combo: dict[str, list[dict]] = {c: [] for c in COMBO_NAMES}

    _print("\nWalk-forward (vectorized)...")
    t0 = time.time()
    iters = 0
    n_iter_total = len(cache) * len(K_SHORT_GRID) * len(cutoff_years)

    for (side, rname, h), bag in cache.items():
        moves = bag["moves"]
        years = bag["years"]
        tcodes = bag["tickers"]
        names = bag["ticker_names"]
        n_tk = bag["n_tickers"]
        family = REGIME_FAMILY.get(rname, rname)

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
                passes_L2 = t_every

                # Vectorize test loop too: for each test fire, look up
                # ticker flags and emit prediction record.
                test_tk = tcodes[test_mask]
                test_w = wins_arr[test_mask]
                test_y = years[test_mask]
                # Mask test fires whose ticker passes claim_100
                claims_per_test = claim_100[test_tk]
                if not claims_per_test.any():
                    iters += 1
                    continue
                base_mask = claims_per_test
                l1_mask = base_mask & passes_L1[test_tk]
                l2_mask = base_mask & passes_L2[test_tk]
                l12_mask = l1_mask & passes_L2[test_tk]

                for combo, mask in zip(COMBO_NAMES, [base_mask, l1_mask, l2_mask, l12_mask]):
                    if not mask.any():
                        continue
                    # Extract eligible test fires for this combo
                    idx = np.where(mask)[0]
                    for i in idx:
                        pred_by_combo[combo].append({
                            "year": int(test_y[i]),
                            "ticker": names[int(test_tk[i])],
                            "side": side, "regime": rname, "family": family,
                            "horizon": h, "k_short": k_short,
                            "outcome": int(test_w[i]) * 2 - 1,
                        })

                iters += 1
            if iters and iters % 500 == 0:
                _print(f"  iter {iters}/{n_iter_total}, "
                       f"elapsed={time.time()-t0:.1f}s, "
                       f"preds (base={len(pred_by_combo['base'])}, "
                       f"L1+L2={len(pred_by_combo['L1_L2'])})")
    _print(f"  predictions complete. elapsed={time.time()-t0:.1f}s")
    _print(f"  prediction counts: " + ", ".join(
        f"{c}={len(pred_by_combo[c])}" for c in COMBO_NAMES))

    # Apply L4 (consensus) post-hoc
    def consensus_filter(preds):
        by_key = defaultdict(set)
        for p in preds:
            k = (p["ticker"], p["year"], p["side"])
            by_key[k].add(p["family"])
        keep = {k for k, fams in by_key.items()
                if len(fams) >= CONSENSUS_FAMILIES_REQUIRED}
        return [p for p in preds if (p["ticker"], p["year"], p["side"]) in keep]

    _print("\n" + "=" * 95)
    _print("CALIBRATION RESULTS — walk-forward, vectorized")
    _print("=" * 95)
    _print(f"{'combo':<14} {'n':>8} {'wins':>8} {'win%':>8}  {'by year':<55}")
    _print("-" * 95)

    summary = {}
    for cname, preds in pred_by_combo.items():
        for variant, postfn in [
            ("", lambda x: x),
            ("+L4", consensus_filter),
        ]:
            sub = postfn(preds)
            n = len(sub)
            wins = sum(1 for p in sub if p["outcome"] > 0)
            wr = (wins / n) if n else None
            yearly = defaultdict(lambda: {"n": 0, "w": 0})
            for p in sub:
                yearly[p["year"]]["n"] += 1
                if p["outcome"] > 0:
                    yearly[p["year"]]["w"] += 1
            yr_str = " ".join(
                f"{y}={d['w']}/{d['n']}={d['w']/d['n']*100:.1f}%" if d["n"] else ""
                for y, d in sorted(yearly.items())
            )
            label = cname + variant
            wr_str = f"{wr*100:.2f}%" if wr is not None else "—"
            _print(f"{label:<14} {n:>8} {wins:>8} {wr_str:>8}  {yr_str:<55}")
            summary[label] = {
                "n": n, "wins": wins, "win_rate": wr,
                "by_year": {str(y): dict(d) for y, d in yearly.items()},
            }

    _print("\nVERDICT (Certified-grade requires ≥ 98% win and n ≥ 100):")
    qual = sorted(((k, v) for k, v in summary.items()
                   if (v["win_rate"] or 0) >= 0.98 and v["n"] >= 100),
                  key=lambda x: -x[1]["win_rate"])
    if qual:
        for k, v in qual[:5]:
            _print(f"  ✓ {k:<14}  n={v['n']:>5}  win_rate={v['win_rate']*100:.2f}%")
    else:
        _print("  ❌ No combo meets the 98% bar with sufficient sample.")
        top = sorted(((k, v) for k, v in summary.items() if v["n"] >= 50),
                      key=lambda x: -(x[1]["win_rate"] or 0))[:8]
        _print("  Top combos by win rate (n ≥ 50):")
        for k, v in top:
            _print(f"    {k:<14}  n={v['n']:>5}  win_rate={(v['win_rate'] or 0)*100:.2f}%")

    out_path = os.path.join(_HERE, "results", "option_c_walkforward.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"cutoff_years": cutoff_years, "summary": summary,
                   "tunable": {"min_fires": MIN_TICKER_FIRES,
                               "min_folds": MIN_TICKER_FOLDS,
                               "train_win_rate_floor": TRAIN_WIN_RATE_FLOOR}},
                  fh, indent=2)
    _print(f"\nDetail saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
