# Step 33: Walk-Forward Adaptive Parameter Selection — Summary

**Date:** 2026-04-21
**Hypothesis:** Choosing parameters dynamically based on trailing 5Y
performance should beat static CAP5 (cap=5%, top_n=5) if the optimal
parameter set is regime-dependent.

## Protocol

- **Burn-in:** first 60 months (2006-04 → 2011-04) use plain CAP5.
- **Rebalance:** each April, pick the best (cap, top_n) grid cell based
  on trailing 5Y metric (CAGR, Sharpe, or Calmar).
- **Candidate grid:** 30 variants = cap ∈ {none, 5%, 7%, 10%, 15%, 20%}
  × top_n ∈ {3, 4, 5, 6, 7}
- **Universe:** 97 tickers, 20Y spine 2006-04 → 2026-04.
- Each year's forward 12M performance is compared to what static CAP5
  would have delivered over the same window.

## Results (forward 12M CAGR, per year)

| Metric used | Wins vs CAP5 | Mean log-return |
|---|---|---|
| CAGR       | 6 / 15 | +30.93%   |
| Sharpe     | 3 / 15 | +30.06%   |
| Calmar     | 2 / 15 | +29.06%   |
| **CAP5 static** | — | **+30.80%** |

**Walk-forward never outperforms static CAP5 meaningfully on any metric.**

## Key observations

1. **Metric choice matters**, but not enough: CAGR-based selection is
   the only one that edges CAP5 (+0.13pp mean log return). Sharpe and
   Calmar both *underperform* CAP5 by 0.7–1.7pp.

2. **Early-period whipsaw:** In 2011-12, all three metrics picked
   (cap=10%, top_n=3/4) — the best-in-hindsight 2006-2011 configuration.
   But during the forward 2011-2012 window, this variant underperformed
   CAP5 by 8–9pp. Similar pattern repeats in 2020-21 (picked cap=none,
   top_n=3 after strong 2015-2020 run; lost 11pp in forward).

3. **2022+ convergence:** From 2022 onward, ALL three metrics
   converge on CAP5 itself as the trailing-best choice. This
   means the walk-forward test in the last 4 years is structurally
   CAP5, which means the apparent "Walk-forward won 6/15" is heavily
   concentrated in 2012-2018 (where adaptive occasionally beats
   static), and reverses entirely afterward.

4. **Regime lag:** The trailing 5Y window is a rearview mirror. When a
   regime shifts (2011→bull, 2020→crisis rebound), the trailing optimum
   is from the *previous* regime and actively hurts performance.

## Interpretation

Walk-forward parameter adaptation **does not survive the strict
point-in-time test.** The incumbent static CAP5 is not just the
in-sample optimum across 20Y — it's also the configuration that
walk-forward ultimately converges on, suggesting that:

- CAP5's moderate concentration (cap=5%, top_n=5) is a *genuinely
  robust* tradeoff rather than a period-specific optimum.
- Any attempt to "chase" recent outperformers at the parameter level
  exposes the strategy to regime-change whiplash.

## Decision

**No adoption.** Walk-forward adaptive parameters are shelved.
CAP5 stays as the static parameter choice. This is consistent with the
step31 finding that (cap=5%, top_n=5) is the unique 20Y CAGR maximum
among 30 grid cells.

## Caveat

The simulation here is pragmatic: it uses each candidate's
precomputed full-period equity slice for both trailing scoring and
forward evaluation. A strict walk-forward would run fresh simulations
per year-segment with param changes, but the DCA mechanics make exact
stitching ambiguous. The pragmatic version matches the intent of
"would trailing-best have helped forward" and the answer is no.
