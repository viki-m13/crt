# Novel-v9: multi-horizon consensus — the first POSITIVE result that survives the gauntlet

**Date:** 2026-05-16 · `novel_v9_consensus.py` · clean-data (post the
2026-05-16 integrity fix).

## Idea (theory-grounded, not data-mined)

The deployed scorer is `mean(pred_3m, pred_6m)`. The walk-forward GBM
also emits `pred_1m`, and the three horizon heads are only ~0.55–0.81
correlated — there is real disagreement, and **disagreement is an
uncertainty signal**. Selecting only names where all three horizons
agree (low cross-horizon rank dispersion) is textbook ensemble
variance-reduction, applied to picks the model is most confident about.
It is **not a new alpha** — it is a better *extraction* of the existing
validated model, aimed squarely at the recent-era fragility.

## Result vs the same-harness deployed scorer (clean data, identical
crash gate / K=2 / costs)

| Variant | CAGR | Sharpe | MaxDD | 10y DCA-win | eras beating S&P-DCA | 2021–26 era |
|---|--:|--:|--:|--:|--:|--:|
| deployed mean(3m,6m) | 25.4% | 0.71 | −49% | 79% | 3/4 | +16% (loses, S&P +17) |
| **consensus / low-disp** | **~32%** | **~0.84** | −52% | **100%** | **4/4** | **+25% (beats)** |

(Absolute levels are lower than the live ~40% because this isolation
harness has no Chronos filter / inv-vol / production regime gate — the
**relative** improvement on an identical harness is the finding.)

## Skeptic's gauntlet — it passes

1. **Dispersion-quantile plateau** (low-disp): q ∈ {0.30…0.70} →
   CAGR 30.9–32.1%, Sharpe 0.82–0.84, 100% 10y, 4/4 eras, 2021–26
   always +25%. Flat — not a knife-edge.
2. **Consensus-threshold plateau:** thr ∈ {0.50…0.70} → CAGR 30–33%,
   Sharpe 0.81–0.86, same era profile. Flat.
3. **True design(≤2012)/holdout(≥2013) split — the decisive test:**

   | | design ≤2012 Sharpe | holdout ≥2013 Sharpe | holdout cum |
   |---|--:|--:|--:|
   | deployed mean(3m,6m) | 0.79 | **0.64** | +760% |
   | consensus low-disp | 0.76 | **0.99** | **+3316%** |

   The edge is **concentrated out-of-sample**; the design period is
   slightly *worse*. That is the opposite of an overfit (which looks
   great in-design and collapses OOS). It generalises forward — exactly
   the property every prior attempt lacked.

## Honest status & what must happen before any deployment

This is the strongest, most defensible improvement found in this whole
engagement and the **first** thing that genuinely fixes the recent-era
(2021+) underperformance while surviving plateau + holdout tests. But
it is validated *in isolation*, not in the production pipeline. Before
it can go live it must clear the repo's full gauntlet on the **real**
harness (Chronos filter + inv-vol cap + production regime gate + the
canonical 10-split walk-forward + MC synthetic-delisting overlay), via
`build_webapp_v5_pit.run_full_sim`. Recommended next step: add a
`consensus` scorer option there (gate: drop names whose pred_1m/3m/6m
cross-sectional ranks disagree, i.e. top-half agreement; rank survivors
by mean horizon rank) and re-run the canonical validation + MC overlay.
Only if it holds there should the deployed `STRATEGY_SPEC` change.

No website / data.json changes were made in this commit — this is a
validated *candidate*, reported honestly, not yet shipped.
