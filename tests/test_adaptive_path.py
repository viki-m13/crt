#!/usr/bin/env python3
"""Leakage, denominator and calibration checks for the adaptive-path engine.

Everything this engine reports is a precision figure, and every way a
precision figure goes wrong is a way of accidentally seeing the future. The
checks here are the ones where a bug makes the result BETTER, which is why
they have to be mechanical:

  - the evidence gate must never look at an outcome that had not matured
  - the conservative lower bound must not certify a tiny sample
  - a delisted name must count against us, not vanish from the denominator
  - features at bar i must be unchanged by anything after bar i

    python tests/test_adaptive_path.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research", "adaptive_path"))

import engine as E  # noqa: E402

FAIL: list[str] = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    print(("  ok    " if cond else "  FAIL  ") + label + " " + str(detail))
    if not cond:
        FAIL.append(label)


def panel(n=1400, k=6, seed=5, dead=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    cols = [f"S{j}" for j in range(k)]
    px = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n))) for c in cols},
        index=idx)
    if dead:                      # kill a ticker part-way through
        px.iloc[dead[1]:, px.columns.get_loc(dead[0])] = np.nan
    return px


def main():
    print("=" * 78)
    print("ADAPTIVE PATH ENGINE")
    print("=" * 78)

    print("\n--- 1. the config refuses what it cannot honour ---")
    for bad, why in (((29,), "a horizon under 30 sessions"),
                     ((30, 30), "duplicate horizons")):
        try:
            E.Config(horizons=bad)
            check(False, f"rejects {why}")
        except ValueError:
            check(True, f"rejects {why}")

    print("\n--- 2. the conservative bound cannot certify thin evidence ---")
    check(E.effective_block_wilson(1.0, 1) < 0.60,
          "one perfect block does not prove 90%",
          f"{E.effective_block_wilson(1.0, 1):.3f}")
    check(E.effective_block_wilson(1.0, 0) == 0.0, "zero blocks proves nothing")
    lo_small = E.effective_block_wilson(0.95, 12)
    lo_big = E.effective_block_wilson(0.95, 500)
    check(lo_small < lo_big, "more blocks tighten the bound",
          f"{lo_small:.3f} -> {lo_big:.3f}")
    check(lo_big < 0.95, "the bound always sits below the point estimate",
          f"{lo_big:.4f} < 0.95")
    check(E.effective_block_wilson(0.95, 12, 0.001)
          < E.effective_block_wilson(0.95, 12, 0.05),
          "a stricter alpha gives a lower bound (the family correction bites)")

    print("\n--- 3. THE GATE ONLY SEES MATURED FORECASTS ---")
    cfg = E.Config()
    # 150 dates x 20 names: enough rows AND enough non-overlapping blocks for
    # the bound to actually clear 0.90. Anything smaller SHOULD fail the gate,
    # which is the point of the minimum_gate_* settings.
    ndates, per = 150, 20
    ii = np.repeat(np.arange(0, ndates * 100, 100), per)
    p_cal = np.tile(np.r_[np.full(12, 0.97), np.full(4, 0.7),
                          np.full(4, 0.3)], ndates)
    up = np.tile(np.r_[np.ones(12), np.zeros(4), np.ones(2),
                       np.zeros(2)], ndates)
    hist = pd.DataFrame({"i": ii, "exit_i": ii + 30, "p_cal": p_cal, "up": up})
    g, det = E.evidence_gate(hist, cfg, 30)
    check(np.isfinite(g), "a gate opens on strong, well-spread evidence", g)
    top = [c for c in det["candidates"] if c["threshold"] == 0.95][0]
    check(top["conservative_lower"] < top["precision"],
          "the gate uses the lower bound, never the point estimate",
          f"{top['conservative_lower']:.3f} < {top['precision']:.3f}")

    thin = hist.head(30)
    gt, dt = E.evidence_gate(thin, cfg, 30)
    check(not np.isfinite(gt), "too few rows keeps the gate shut", dt["reason"])
    check(not np.isfinite(E.evidence_gate(pd.DataFrame(), cfg, 30)[0]),
          "no history at all keeps the gate shut")

    # matured_history is the only door to the gate, and it is an < comparison
    parts = [pd.DataFrame({"i": [0, 100], "exit_i": [30, 130], "up": [1.0, 1.0]})]
    m = E.matured_history(parts, 100)
    check(list(m.exit_i) == [30], "a forecast maturing AT as_of is excluded",
          f"kept exit_i={list(m.exit_i)}")
    check(E.matured_history(parts, 10).empty, "nothing has matured on day 10")

    print("\n--- 4. a delisted name counts against us ---")
    # The engine refuses to fit on thin data (correctly), so the fixture has
    # to be big enough to actually exercise the loop: 40 names, 8 years.
    px = panel(n=2100, k=40, dead=("S3", 1500))
    cfg2 = E.Config(horizons=(30,), first_fit_year=2000, evaluation_year=2005,
                    minimum_training_rows=150, step=21, refit_every=252,
                    names_per_date=40, minimum_calibration_dates=6,
                    min_names_per_date=20)
    out = E.run_prequential(px, 30, cfg2, verbose=False)
    check(not out.empty, "the loop produces forecasts", f"{len(out)} rows")
    dl = out[out.delisted]
    check(len(dl) > 0, "the dead ticker does generate observations", len(dl))
    check((dl.up == 0).all(),
          "and every one of them is labelled NOT higher, not dropped")
    check(out.up.notna().all(), "no observation is silently missing an outcome")

    print("\n--- 5. no look-ahead in the features ---")
    # REGRESSION: features at bar i must not move when the future is rewritten
    base = panel(seed=9)
    f1 = E.build_features(base)
    tampered = base.copy()
    tampered.iloc[900:] *= 4.0
    f2 = E.build_features(tampered)
    worst, where = 0.0, ""
    for name in E.FEATURES:
        a, b = f1[name][:900], f2[name][:900]
        d = np.nanmax(np.abs(a - b)) if np.isfinite(a).any() else 0.0
        if d > worst:
            worst, where = d, name
    check(worst < 1e-9,
          "quadrupling every bar after 900 changes no feature before it",
          f"worst drift {worst:.2e} ({where})")

    print("\n--- 6. the path band is scored on the WHOLE path ---")
    lp = np.log(panel(seed=3).to_numpy())
    rows = pd.DataFrame({"i": [300], "col": [0], "sigma": [0.015], "mu": [0.0]})
    sc = E.path_scores(lp, rows, 30)
    steps = np.arange(1, 31)
    actual = lp[300 + steps, 0] - lp[300, 0]
    manual = np.max(np.abs(actual) / (0.015 * np.sqrt(steps)))
    check(abs(sc["flat_path_score"][0] - manual) < 1e-9,
          "the score is the worst standardised deviation over every close",
          f"{sc['flat_path_score'][0]:.4f}")
    check(sc["flat_path_score"][0] >= abs(actual[-1]) / (0.015 * np.sqrt(30)) - 1e-9,
          "and is never smaller than the endpoint-only score")
    check(np.isnan(E.path_scores(lp, pd.DataFrame(
        {"i": [len(lp) - 5], "col": [0], "sigma": [.01], "mu": [0.]}),
        30)["path_score"][0]), "an unfinished path scores NaN, not zero")

    print("\n--- 7. weights and quantiles ---")
    w = E.date_weights(np.array([1, 1, 1, 2]), np.zeros(4), 756)
    check(abs(w[:3].sum() - w[3]) < 1e-9,
          "a date with 3 rows carries the same total weight as one with 1")
    old = E.date_weights(np.array([1, 2]), np.array([0, 3024]), 756)
    check(old[0] > old[1], "older evidence weighs less", f"{old[0]:.3f} vs {old[1]:.3f}")
    check(abs(E.weighted_quantile(np.array([1., 2, 3, 4]), 0.5) - 2.5) < 0.6,
          "weighted quantile is sane")
    check(np.isnan(E.weighted_quantile(np.array([np.nan]), 0.5)),
          "all-NaN gives NaN, not a made-up number")

    print("\n--- 8. the mixture adapts and stays recoverable ---")
    raw = np.column_stack([np.full(50, 0.5), np.full(50, 0.9), np.full(50, 0.1)])
    hist2 = pd.DataFrame({
        "i": np.arange(300), "exit_i": np.arange(300) + 30,
        "p_base": 0.5, "p_linear": 0.9, "p_nonlinear": 0.1,
        "p_mixed": 0.5, "up": np.tile([1.0, 1.0, 1.0, 0.0], 75)})
    p, wts = E.adaptive_probability(raw, hist2, 400, E.Config())
    check(abs(wts.sum() - 1.0) < 1e-9, "expert weights sum to 1", f"{wts.sum():.6f}")
    check(wts.min() >= 0.05,
          "no expert is ever zeroed out, so it can recover", f"min {wts.min():.3f}")
    check(wts[1] > wts[2], "the expert that was right gets more weight",
          f"linear {wts[1]:.3f} > nonlinear {wts[2]:.3f}")
    p0, w0 = E.adaptive_probability(raw, pd.DataFrame(), 400, E.Config())
    check(abs(w0 - 1 / 3).max() < 1e-9,
          "with no history the experts are weighted equally")

    print("\n" + "=" * 78)
    print(f"{N[0] - len(FAIL)}/{N[0]} checks passed")
    if FAIL:
        for f in FAIL:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
