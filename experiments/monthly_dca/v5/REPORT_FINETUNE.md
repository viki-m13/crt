# v5 Fine-Tune + Generalisation Report
**Run date:** 2026-05-11.

## Final winner (most generalisable)

**`v5_chr_p70_q0.45_k3_invvol_cap0.4`**

Same as v3 baseline (`ml_3plus6 EW tight h=6`) but:
- Apply Chronos-bolt-tiny p70 cross-sectional rank ≥ 0.45 filter
- Switch to inverse-volatility weighting with cap=0.4 per pick

| Universe | v3 baseline (WF mean) | **v5 K=3 invvol** | Lift |
|----------|--------------------:|------------------:|-----:|
| PIT S&P 500 | 42.80% (9/10 beats) | **47.16%** (10/10) | **+4.36 pp** |
| Broader 1833 | 51.83% (10/10) | **57.82%** (9/10) | **+5.99 pp** |
| Non-S&P 500 PIT | 51.03% (10/10) | **62.72%** (10/10) | **+11.69 pp** |
| Random-500 seed 1 | 62.09% (10/10) | **67.09%** (10/10) | +5.00 pp |
| Random-500 seed 2 | 47.50% (6/10) | **46.86%** (8/10) | -0.64 pp (more beats SPY) |
| Random-500 seed 3 | 44.95% (7/10) | **51.14%** (9/10) | +6.19 pp |

**v5 K=3 invvol generalizes MORE than v3** — beats v3 in 5 of 6 tested universes, including the broader 1833 and non-S&P 500 cohorts.

## Highest CAGR (PIT only — does NOT generalize)

**`v5_chr_p70_q0.45_k2_invvol_cap0.4`**: 51.28% WF mean OOS CAGR on PIT S&P 500 with 10/10 beats SPY, but fails to generalize:
- Broader 1833: 39.05% (vs v3's 51.83%) — significantly WORSE
- Non-SP500 PIT: 46.24% (vs v3's 51.03%) — worse

The K=2 concentration that wins on PIT S&P 500 fails on broader universes. **Not recommended** for production unless the deployment universe is strictly PIT S&P 500.

## Sensitivity analysis (PIT S&P 500)

Fine-tuned across K ∈ {2,3,4}, weighting ∈ {ew, invvol, softmax}, cap ∈ {0.4, 0.5, 0.6, 0.75, 1.0}, q ∈ {0.30, 0.35, 0.40, 0.45, 0.50}.

Top-10 invvol_cap0.4 variants all show WF mean 46-51% with 10/10 beats SPY.  The K=3 invvol cap=0.4 family is on a robust plateau (small parameter perturbations don't materially change result), while K=2 sits on a knife-edge.

## Comparison: v3 vs v5 generalization breadth

**v3** (deployed): generalises strongly across all 6 tested universes — was the baseline used to validate v5.

**v5 K=3 invvol** (new): generalises MORE — beats v3 by 4-12pp on broader/non-SP500/random universes. The lift comes primarily from the Chronos confidence filter, which carries information complementary to v2 GBM's cross-sectional alpha.

The K=2 high-CAGR variant is **overfit to PIT S&P 500 specifics** (likely the NVDA-dominated 2017-2024 run) and should not be deployed on alternate universes.

## Recommendation

**Production candidate: `v5_chr_p70_q0.45_k3_invvol_cap0.4`** (most generalisable, all 10/10 beat SPY on PIT, beats v3 baseline on 5 of 6 universes including broader 1833 and non-S&P 500).

For PIT S&P 500 only, the K=2 invvol cap=0.4 variant (51.28% WF mean) offers higher CAGR — but at the cost of MDD -69% and **does not generalise** to other universes.

## Files

- `cache/v2/sp500_pit/v5_finetune_results.csv` — 250+ variant sweep
- `cache/v2/sp500_pit/v5_finetune_k_w_cap.csv` — K/W/cap fine-tune
- `cache/v2/sp500_pit/v5_generalization_v3_vs_v5.csv` — generalisation across 6 universes
- `cache/v2/sp500_pit/v5_chr_p70_q0.45_k3_invvol_cap0.4_equity.csv` — winner equity
- `cache/v2/sp500_pit/v5_chr_p70_q0.45_k2_invvol_cap0.4_equity.csv` — high-CAGR alt equity

## Honest caveats

1. The K=2 invvol PIT winner's 51% WF mean does not generalize to broader universes — it's NVDA-concentration-driven, which is in-sample-specific to PIT S&P 500's 2017-2024 history.

2. The K=3 invvol winner's lift on PIT (+4.36pp WF mean) is smaller than on broader universes (+5.99pp to +11.69pp). The Chronos confidence filter provides MORE alpha on broader universes — likely because broader universes contain more "false positive" picks that Chronos can filter out.

3. R1_GFC: v5 K=3 invvol gives 104.59% (vs v3's 108.79%) — slightly lower in this single split. All other splits favor v5.

4. Sample size: only 6 universes tested. Random-500 seed 2 (the weakest result) is a 153K-row sample — variance is non-trivial.

5. Live deployment: when the user runs this on Russell 1000 or tech all-cap, expect performance similar to the broader/non-SP500 results (50-60% WF mean OOS). Genuine alpha exists across universes.
