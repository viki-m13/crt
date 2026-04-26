"""Walk-forward calibration harness for Option C "Certified-Core" rules.

Goal
----
A "Certified" rule should deliver ≥98% real-future win rate when it
claims ≥98%. To verify, we walk-forward test multiple LAYER COMBOS
(ablations) and measure which combos actually calibrate.

Method
------
For each cutoff year Y in {2022, 2023, 2024, 2025, 2026}:

    1. Take only historical fires with date < Y as the "training set"
       and fires in year Y as the "held-out test set."
    2. For each (ticker, side, regime, horizon, k_short):
        a. From the training set: compute per-ticker win rate, n_fires,
           n_folds, every_fold_perfect, worst-historical-move.
        b. Apply each LAYER COMBO to determine if the rule is
           "certified" / "near" / "fail" using TRAINING data only.
    3. For each "certified" rule, look at the held-out test set fires:
       count actual wins/losses → empirical Y-year win rate.
    4. Aggregate across all 5 cutoff years per layer combo.
    5. Calibration metric: do "certified ≥98%" claims actually
       deliver ≥98% on held-out fires?

The 8 layer combos we ablate are subsets of:
    L1: worst-historical-drawdown floor (K_short ≥ worst + 1%)
    L2: per-ticker fold coverage (≥200 fires, ≥5 folds, every fold perfect)
    L3: macro regime conformity gate (SPY > SMA200, etc.) — applied at
        publish time per fire date in held-out set
    L4: multi-rule consensus (≥2 distinct families)

We run combos:
    {none, L1, L2, L4, L1+L2, L1+L4, L2+L4, L1+L2+L3+L4}
and compare to the baseline (existing engine: empirical training
win-rate quantile only).
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_common import (
    FOLD_YEARS, WARMUP_DAYS,
    OhlcvSeries, V2Features,
    actual_options_expiry, compute_features, fold_mask, list_tickers,
    load_series, spy_context, train_mask_for_fold,
)
from v2_regimes import CALL_REGIMES, PUT_REGIMES
from option_c_research import (
    K_SHORT_GRID, SPREAD_WIDTH, IV_MULT, HAIRCUT,
    HORIZONS, _credit_and_maxloss, _trade_pnl, _gather_fires, Fire,
)
from option_c_certified import (
    WORST_BUFFER_SAFETY, MIN_TICKER_FIRES, MIN_TICKER_FOLDS,
    SPY_5D_FLOOR_PCT, STOCK_5D_FLOOR_PCT,
    REGIME_FAMILY, CONSENSUS_FAMILIES_REQUIRED,
)


# ------------- helpers ----------------------------------------------


def fire_year(fi: Fire) -> int:
    return int(str(fi.date)[:4])


def split_train_test(fires: list[Fire], cutoff_year: int) -> tuple[list[Fire], list[Fire]]:
    train = [fi for fi in fires if fire_year(fi) < cutoff_year]
    test = [fi for fi in fires if fire_year(fi) == cutoff_year]
    return train, test


def fire_outcome(fi: Fire, side: str, k_short: float, k_long: float) -> int:
    """Return +1 if the spread won at expiry, -1 otherwise."""
    if side == "put":
        Ks = fi.spot * (1.0 - k_short)
        return +1 if fi.close_at_expiry >= Ks else -1
    else:
        Ks = fi.spot * (1.0 + k_short)
        return +1 if fi.close_at_expiry <= Ks else -1


def fire_pnl(fi: Fire, side: str, k_short: float, k_long: float, sigma: float) -> float:
    """Realized $/share P&L for one fire (uses fire-date σ for credit)."""
    T = max(1, int(round(7))) / 365.0  # tiny T proxy; we just need pnl direction
    # We'll compute credit using actual horizon (in calendar days
    # approximation) and the fire-time sigma.
    # For walk-forward, we just need win/loss for the calibration metric;
    # exact $ P&L isn't strictly needed to answer "did 98% claim hold?"
    side_win = fire_outcome(fi, side, k_short, k_long)
    return 1.0 if side_win > 0 else 0.0


def per_ticker_train_stats(side: str, train_fires: list[Fire],
                            k_short: float, k_long: float) -> dict[str, dict]:
    """Per-ticker train-set summary: n_fires, n_folds (distinct years),
    every_fold_perfect, worst-historical-move."""
    out: dict[str, dict] = {}
    for fi in train_fires:
        tk = out.setdefault(fi.ticker, {
            "fires": [],
            "folds": {},
        })
        tk["fires"].append(fi)
        y = fire_year(fi)
        f = tk["folds"].setdefault(y, {"wins": 0, "losses": 0})
        if fire_outcome(fi, side, k_short, k_long) > 0:
            f["wins"] += 1
        else:
            f["losses"] += 1

    summarized: dict[str, dict] = {}
    for tkr, info in out.items():
        n_fires = sum(f["wins"] + f["losses"] for f in info["folds"].values())
        n_folds = len(info["folds"])
        every_perfect = all(
            f["losses"] == 0 and f["wins"] > 0 for f in info["folds"].values()
        )
        # Worst-historical-move on training set
        worst = 0.0
        for fi in info["fires"]:
            if side == "put":
                m = max(0.0, (fi.spot - fi.close_at_expiry) / fi.spot)
            else:
                m = max(0.0, (fi.close_at_expiry - fi.spot) / fi.spot)
            if m > worst:
                worst = m
        # Win rate on training set
        total = n_fires
        wins = sum(f["wins"] for f in info["folds"].values())
        train_win_rate = (wins / total) if total > 0 else 0.0
        summarized[tkr] = {
            "n_fires": n_fires,
            "n_folds": n_folds,
            "every_fold_perfect": every_perfect,
            "worst_historical_move": worst,
            "train_win_rate": train_win_rate,
        }
    return summarized


# ------------- layer combo predicates -----------------------------


def passes_layer_1(stats: dict, k_short: float) -> bool:
    """Worst-historical-drawdown floor: K_short ≥ worst + safety."""
    if stats["worst_historical_move"] <= 0:
        return False
    return k_short >= stats["worst_historical_move"] + WORST_BUFFER_SAFETY


def passes_layer_2(stats: dict) -> bool:
    """Per-ticker fold coverage."""
    return (
        stats["n_fires"] >= MIN_TICKER_FIRES
        and stats["n_folds"] >= MIN_TICKER_FOLDS
        and stats["every_fold_perfect"]
    )


def evaluate_combo(layers: set[str], stats: dict, k_short: float) -> bool:
    """Do layers 1 and/or 2 pass for this rule? (L3 macro applied
    later, L4 consensus applied later.)"""
    if "L1" in layers and not passes_layer_1(stats, k_short):
        return False
    if "L2" in layers and not passes_layer_2(stats):
        return False
    return True


# ------------- walk-forward driver ------------------------------------


def run_walkforward(layer_combos: list[set[str]],
                    cutoff_years: list[int],
                    use_macro: dict[frozenset, bool],
                    use_consensus: dict[frozenset, bool]) -> dict:
    """Walk-forward across cutoff years, ablating layer combinations.

    Returns a dict mapping combo_name → list of {year, claimed_n,
    actual_wins, actual_total, claim_lower_bound}.
    """
    results: dict[str, dict] = {
        "_".join(sorted(c)) if c else "none": {
            "by_year": {},
            "all_predictions": [],
        }
        for c in layer_combos
    }

    # Pre-fetch everything once. For each (regime, horizon, k_short),
    # gather the full fire set and split per cutoff year.
    print(f"Walk-forward: cutoff years = {cutoff_years}")
    print(f"Combos: {[' & '.join(sorted(c)) if c else 'none' for c in layer_combos]}")
    t0 = time.time()
    n_combos_done = 0

    # We iterate (side, regime, horizon, k_short) combos to gather
    # predictions for every layer combo at every cutoff year.
    for side, regime_map in (("put", CALL_REGIMES), ("call", PUT_REGIMES)):
        for rname, rfn in regime_map.items():
            for h in HORIZONS:
                fires = _gather_fires(side, rname, rfn, h)
                if not fires:
                    continue
                for k_short in K_SHORT_GRID:
                    k_long = k_short + SPREAD_WIDTH
                    family = REGIME_FAMILY.get(rname, rname)
                    for cy in cutoff_years:
                        train_fires, test_fires = split_train_test(fires, cy)
                        if not test_fires:
                            continue
                        # Train stats per ticker
                        stats = per_ticker_train_stats(side, train_fires, k_short, k_long)
                        # Per ticker decide eligibility per combo
                        # and accumulate test outcomes.
                        for fi in test_fires:
                            tkr = fi.ticker
                            tk_stats = stats.get(tkr)
                            if tk_stats is None:
                                continue   # ticker had no training history yet
                            # If "every_fold_perfect" required (in L2)
                            # but train win rate < 1.0, then L2 fails.
                            outcome = fire_outcome(fi, side, k_short, k_long)
                            for layers in layer_combos:
                                cname = "_".join(sorted(layers)) if layers else "none"
                                if not evaluate_combo(layers, tk_stats, k_short):
                                    continue
                                # Record prediction (we'll later
                                # consolidate consensus across rules
                                # at the same (ticker, year, horizon)).
                                results[cname]["all_predictions"].append({
                                    "year": cy,
                                    "ticker": tkr,
                                    "side": side,
                                    "regime": rname,
                                    "family": family,
                                    "horizon": h,
                                    "k_short": k_short,
                                    "fire_date": str(fi.date),
                                    "outcome": outcome,
                                })
                    n_combos_done += 1
                    if n_combos_done % 50 == 0:
                        print(f"  scanned {n_combos_done} (regime, h, k) combos  "
                              f"elapsed={time.time()-t0:.1f}s")

    return results


# ------------- consensus + macro post-processing ---------------------


def _spy_macro_at(spy_dates, spy_close, spy_ret_5d, fire_date_str: str) -> bool:
    """Was the SPY-macro gate satisfied on this date?"""
    when = np.datetime64(fire_date_str)
    idx = int(np.searchsorted(spy_dates, when))
    if idx >= len(spy_dates) or idx < 200:
        return False
    today = float(spy_close[idx])
    sma200 = float(np.mean(spy_close[idx - 200 : idx]))
    if today < sma200:
        return False
    r5 = float(spy_ret_5d[idx]) if np.isfinite(spy_ret_5d[idx]) else float("nan")
    if not np.isfinite(r5) or r5 < SPY_5D_FLOOR_PCT:
        return False
    return True


def apply_macro_filter(predictions: list[dict]) -> list[dict]:
    """Filter predictions whose fire_date doesn't pass the SPY macro gate."""
    ctx = spy_context()
    if ctx is None:
        return predictions
    spy_dates, spy_close, spy_ret_5d = ctx
    return [
        p for p in predictions
        if _spy_macro_at(spy_dates, spy_close, spy_ret_5d, p["fire_date"])
    ]


def apply_consensus_filter(predictions: list[dict]) -> list[dict]:
    """Keep only predictions where ≥CONSENSUS_FAMILIES_REQUIRED distinct
    regime families fire on the same (ticker, year, side) — proxy for
    'multiple rules agree'."""
    by_key: dict[tuple, set[str]] = {}
    for p in predictions:
        k = (p["ticker"], p["year"], p["side"])
        by_key.setdefault(k, set()).add(p["family"])
    qualifying = {k for k, families in by_key.items()
                  if len(families) >= CONSENSUS_FAMILIES_REQUIRED}
    return [p for p in predictions
            if (p["ticker"], p["year"], p["side"]) in qualifying]


# ------------- summarize ---------------------------------------------


def summarize_calibration(name: str, predictions: list[dict]) -> dict:
    """Compute: total predictions, total wins, empirical win rate."""
    n = len(predictions)
    if n == 0:
        return {"name": name, "n": 0, "wins": 0, "win_rate": None}
    wins = sum(1 for p in predictions if p["outcome"] > 0)
    return {
        "name": name,
        "n": n,
        "wins": wins,
        "win_rate": wins / n,
    }


def by_year(predictions: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in predictions:
        d = out.setdefault(p["year"], {"n": 0, "wins": 0})
        d["n"] += 1
        if p["outcome"] > 0:
            d["wins"] += 1
    for y, d in out.items():
        d["win_rate"] = (d["wins"] / d["n"]) if d["n"] else None
    return out


# ------------- main --------------------------------------------------


def main() -> int:
    cutoff_years = [2022, 2023, 2024, 2025, 2026]
    base_layer_combos = [
        set(),                  # none — pure empirical (no extra layers)
        {"L1"},                 # worst-historical floor only
        {"L2"},                 # per-ticker fold coverage only
        {"L1", "L2"},           # both structural layers
    ]

    results = run_walkforward(base_layer_combos, cutoff_years, {}, {})

    # For each base combo, also compute +L3 (macro) and +L4 (consensus)
    # variants by post-processing the predictions list.
    print("\n\n" + "=" * 72)
    print("WALK-FORWARD CALIBRATION RESULTS")
    print("=" * 72)
    print(f"{'combo':<24} {'n':>7} {'wins':>7} {'win_rate':>10} {'by year':<28}")
    print("-" * 72)
    expanded: dict[str, dict] = {}
    for combo_name, info in results.items():
        preds = info["all_predictions"]
        for variants in [
            (combo_name, preds),
            (combo_name + "+L3", apply_macro_filter(preds)),
            (combo_name + "+L4", apply_consensus_filter(preds)),
            (combo_name + "+L3+L4", apply_consensus_filter(apply_macro_filter(preds))),
        ]:
            vname, vpreds = variants
            stats = summarize_calibration(vname, vpreds)
            yearly = by_year(vpreds)
            year_str = " ".join(
                f"{y}={d['wins']}/{d['n']}={(d['win_rate'] or 0)*100:.1f}%"
                for y, d in sorted(yearly.items())
            )
            wr_pct = (stats['win_rate'] or 0) * 100
            print(f"{vname:<24} {stats['n']:>7} {stats['wins']:>7} "
                  f"{wr_pct:>9.2f}%  {year_str}")
            expanded[vname] = stats

    # Find combos that meet the "Certified ≥98% claim → ≥98% delivery" bar
    print("\nCALIBRATION VERDICT")
    print("-" * 72)
    print("(combo qualifies as 'Certified-grade' if win rate ≥ 98% AND n ≥ 100)")
    print()
    qualifying = []
    for vname, s in expanded.items():
        if s["n"] is None or s["n"] < 100:
            continue
        wr = s["win_rate"] or 0
        if wr >= 0.98:
            qualifying.append((vname, s))
            print(f"  ✓ {vname:<24} n={s['n']:>5}  win_rate={wr*100:.2f}%")
    if not qualifying:
        print("  ❌ NO COMBO meets the ≥98% calibration bar with sufficient n.")
        print("  (Weighted average pulled below by edge cases or insufficient samples.)")

    # Save full results
    out_path = os.path.join(_HERE, "results", "option_c_walkforward.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "cutoff_years": cutoff_years,
            "summary": expanded,
            "qualifying_combos": [name for name, _ in qualifying],
        }, fh, indent=2)
    print(f"\nDetail written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
