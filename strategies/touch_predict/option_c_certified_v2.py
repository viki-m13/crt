"""Certified-Core v2 — adds conformal strike + per-trade consensus +
regime-distance gate (Mahalanobis) on top of the v1 4-layer guard.

Activated only if walk-forward shows v1 doesn't calibrate.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from option_c_research import K_SHORT_GRID, SPREAD_WIDTH, _gather_fires, Fire
from option_c_certified import WORST_BUFFER_SAFETY, REGIME_FAMILY


# ---------- Conformal strike selection -------------------------------

def conformal_floor(side: str, train_fires: list[Fire], confidence: float = 0.99) -> float | None:
    """Return the conformal lower bound on the post-fire move
    distribution at the given confidence. By construction, the true
    underlying probability that a future fire's move ≤ this value is
    at least `confidence` (under exchangeability).

    For puts: we return the upper bound on the down-move distribution.
        K_short ≥ this returned value (as fraction of spot).
    For calls: upper bound on the up-move distribution.
    """
    if not train_fires:
        return None
    moves = []
    for fi in train_fires:
        if side == "put":
            m = max(0.0, (fi.spot - fi.close_at_expiry) / fi.spot)
        else:
            m = max(0.0, (fi.close_at_expiry - fi.spot) / fi.spot)
        moves.append(m)
    moves.sort()
    n = len(moves)
    # Conformal quantile under exchangeability: ⌈(N+1) × confidence⌉
    idx = min(n - 1, max(0, int(math.ceil((n + 1) * confidence)) - 1))
    return moves[idx]


# ---------- Mahalanobis regime distance ------------------------------

def mahalanobis_dist(point: np.ndarray, mean: np.ndarray,
                      inv_cov: np.ndarray) -> float:
    diff = point - mean
    return float(math.sqrt(max(0.0, diff @ inv_cov @ diff)))


def regime_distance_train(features: list[np.ndarray]):
    """Build (mean, inv_cov) for the historical regime feature distribution."""
    if not features:
        return None
    X = np.vstack(features)
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = cov + np.eye(cov.shape[0]) * 1e-6   # ridge for invertibility
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return None
    return mean, inv_cov


# ---------- Per-trade consensus --------------------------------------

def per_trade_consensus(predictions: list[dict], strike_tol_pct: float = 0.05,
                        min_families: int = 2):
    """Filter predictions: for each (ticker, year, side), find groups of
    predictions whose K_short_frac is within ± strike_tol of each other
    AND whose horizons agree (within 30 calendar days), AND that
    represent ≥ min_families distinct regime families. Only those
    predictions pass.

    A prediction is part of multiple groups if it satisfies the closeness
    criteria with several others. We'll use the simplest version:
    bucket by k_short rounded to nearest 1%, group by (ticker, year, side).
    """
    by_key: dict[tuple, list[dict]] = {}
    for p in predictions:
        ks_bucket = round(p["k_short"] * 100)  # nearest 1% bucket
        k = (p["ticker"], p["year"], p["side"], p["horizon"], ks_bucket)
        by_key.setdefault(k, []).append(p)
    keep_keys = set()
    for k, ps in by_key.items():
        families = {p["family"] for p in ps}
        if len(families) >= min_families:
            keep_keys.add(k)
    return [p for p in predictions
            if (p["ticker"], p["year"], p["side"], p["horizon"],
                round(p["k_short"] * 100)) in keep_keys]
