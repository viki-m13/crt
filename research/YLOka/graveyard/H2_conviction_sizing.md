# H2 — Conviction-spread sizing — KILLED

**Hypothesis**: weight ∝ softmax(λ · z-score of GBM score) — concentrate when one pick stands out, equal-weight when picks are interchangeable. Add cash slice when top score < q25 of universe.

**Result** (research window 2003-09 → 2024-04, K=3, h=6):
| Variant | CAGR | Sharpe | MaxDD |
|---|---:|---:|---:|
| baseline (ew) | 40.78% | 0.953 | -49.83% |
| λ=0.5 | 39.52% | 0.870 | -61.51% |
| λ=1.5 | 34.53% | 0.745 | -71.55% |
| λ=0.5 + cash floor q25 | 39.52% | 0.870 | -61.51% |

**Why it failed**:
- The GBM `pred_3m`/`pred_6m` rank order is meaningful but the score *gap* between #1 and #3 does not predict relative outperformance.
- Concentrating into the top score lifts variance without lifting expected return → MaxDD blows out by 12-22 pp at no CAGR gain.
- Cash floor never activated meaningfully because the top score is rarely below q25 of the universe (the GBM is pulling from a 350+ name pool; q25 is low).

**Don't repeat**: equal-weighting top-K is NOT a v3 oversight — it's the right answer because the model's score is well-calibrated as a rank but not as a magnitude.
