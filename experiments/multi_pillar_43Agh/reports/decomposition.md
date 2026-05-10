# Pillar Decomposition (multi_pillar_43Agh)

Date: 2026-05-10. PIT S&P 500, 2003-09 → 2025-12.

This decomposes the contribution of each pillar to the headline metrics.
Each row's delta is **vs the V6 winner** (invvol + cash yield), which is
the closest deployable Pareto-improved baseline.

## Pillar standalone tests (Phase 3 first-cut)

| Configuration                  | CAGR     | Sharpe | MaxDD   | WF mean CAGR | Δ CAGR | Δ Sharpe | Δ MaxDD |
|--------------------------------|---------:|-------:|--------:|-------------:|-------:|---------:|--------:|
| V3 deployed (EW, no cash yld)  | 39.77%   | 0.955  | -49.83% | 42.80%       | +1.57  | -0.015   | -3.84   |
| **V6 winner (invvol+cy)**      | **38.20%** | **0.971** | **-45.98%** | **42.48%** |  0.00 |  0.000  |  0.00 |
| Pillar 1 only (drop 30%)       | 20.10%   | 1.091  | -36.71% | 22.92%       |-18.10  | +0.120   | +9.27  |
| Pillar 2 only (trend gate)     | 19.25%   | 0.904  | -48.28% | 20.71%       |-18.95  | -0.067   | -2.29  |
| Pillar 3 only (novel 0.5)      | 28.30%   | 0.802  | -54.92% | 27.04%       | -9.90  | -0.169   | -8.94  |
| Pillar 4 only (archetype 0.5)  | 29.55%   | 0.826  | -52.68% | 31.03%       | -8.65  | -0.144   | -6.69  |

## Pillar combinations

| Configuration                  | CAGR     | Sharpe | MaxDD   | WF mean CAGR | Δ CAGR | Δ Sharpe | Δ MaxDD |
|--------------------------------|---------:|-------:|--------:|-------------:|-------:|---------:|--------:|
| Pillars 1+2                    | 18.46%   | 1.059  | -40.61% | 21.84%       |-19.74  | +0.088   | +5.38  |
| Pillars 1+2+4                  | 16.01%   | 0.843  | -33.97% | 19.00%       |-22.19  | -0.128   |+12.02  |
| Pillars 1+2+3+4                | 18.99%   | 0.987  | -43.10% | 24.25%       |-19.21  | +0.016   | +2.88  |

## Sweep over Pillar 1 alone (drop %)

| drop_failure_pct | CAGR | Sharpe | MaxDD | WF mean CAGR | beats SPY |
|-----------------:|-----:|-------:|------:|-------------:|----------:|
| 0%   (= V6)       | 38.2% | 0.971 | -46.0% | 42.5% | 9/10 |
| 10%               | 14.6% | 0.696 | -67.7% | 16.7% | 7/10 |
| 20%               | 19.7% | 0.969 | -45.8% | 21.0% | 7/10 |
| 30%               | 20.1% | 1.091 | -36.7% | 22.9% | 8/10 |
| **40%**           | **21.1%** | **1.200** | **-35.5%** | 22.3% | 8/10 |

Note: drop=10% looks anomalously bad — it removes some of V3's deep-value
"failure-like-but-rebound" picks but not enough to gain the Sharpe benefit.
30-40% drop is the bigger plateau.

## Sweep over Pillar 1 as **score penalty** (no filter)

| w_failure | CAGR | Sharpe | MaxDD | WF mean CAGR | beats SPY |
|----------:|-----:|-------:|------:|-------------:|----------:|
| 0.0       | 38.2% | 0.971 | -46.0% | 42.5% | 9/10 |
| 0.2       | 31.6% | 0.886 | -46.0% | 35.3% | 9/10 |
| 0.4       | 28.9% | 0.870 | -46.0% | 31.3% | 9/10 |
| 0.6       | 24.8% | 0.856 | -43.1% | 25.3% | 8/10 |

The **soft penalty** preserves the deep-value picks better than the hard
filter — `w_failure=0.2` gives 31.6% CAGR (vs 21% with the hard 40% drop)
but Sharpe is lower. There is no penalty weight that strictly Pareto-
improves V6.

## K (concentration) sweep with drop=20%, w_failure=0.20

| K | CAGR | Sharpe | MaxDD | beats SPY |
|--:|-----:|-------:|------:|----------:|
| 3 | 18.8% | 0.990 | -43.9% | 7/10 |
| 4 | 20.3% | 1.158 | -39.8% | 8/10 |
| **5** | **20.1%** | **1.217** | **-37.1%** | 7/10 |

Higher K with diversification + failure filter = best Sharpe in this
sweep (1.217). But CAGR plateaus at ~20%.

## Headline conclusions

1. **Pillar 1 contributes** Sharpe (+0.13) and MaxDD reduction (+9pp), at a
   CAGR cost (-18pp) when applied as a hard filter. As a soft score penalty
   the trade-off is gentler but never Pareto.
2. **Pillar 2 (trend gate)** contributes nothing positive on this universe
   — it cuts the same deep-value rebound picks as Pillar 1 without
   capturing the same Sharpe benefit.
3. **Pillar 3 (novel features fast version)** dilutes the ML signal. The
   60-day SPY-corr / lag-1 persistence / abs-skew at 0.5 weight loses ~10pp
   CAGR and 0.17 Sharpe. The hypothesised information was either (a) not
   captured by the fast surrogates, or (b) already absorbed by ML.
4. **Pillar 4 (forensic archetype)** at 0.5 weight loses 9pp CAGR and 0.14
   Sharpe. The archetype features (vol_1y, pullback, vol_contraction, etc.)
   are already in the ML model's input.
5. **Composite Pillars 1+2+3+4** does NOT compound benefits — losses
   dominate. Best variant in the full sweep:

   `combo_drop20_wf0_wa2_wn0_wc2`: CAGR 22.2%, Sharpe 0.99, MaxDD -44%

   — meaningfully worse than V6 winner on every metric except CAGR
   per Sharpe-unit (which is roughly equal).

## Decomposition statement

Of the **38.20% V6-winner CAGR**, the multi-pillar architecture as built
**does not extract more**. The dominant CAGR contribution comes from:

- Base ML signal (`ml_3plus6`): +28-30pp vs SPY
- Tight regime gate (cash in 4 hostile months): +1-2pp (small absolute,
  large in 2008/2020 splits)
- Inverse-vol weighting on K=3 picks: -1.6pp full CAGR, +0.015 Sharpe,
  +3.8pp MaxDD reduction
- Cash yield credit (3% on bills): +0.05pp (trivial, honest)

The multi-pillar pillars do NOT add positive CAGR on top of this.

## Why does the deep-value rebound capture dominate?

Empirical examination of V3's pick history (`v6/results/v3_baseline_picks.csv`):
- 2009: MBI, GNW, THC — all picked at deeply distressed levels with
  `mom_12_1 ≈ -0.40 to -0.60`. ML score was high because the model had
  learned that mid-bear deep-value with technical stabilisation is a
  rebound signature. They 5-bagged in 9 months.
- 2020: PEG, AIG, ETN — picked during March-April 2020 panic.
- 2022: CCL, WBD, KEY — picked during inflation-bear, all rebounded
  meaningfully by mid-2023.

A failure filter cuts these picks BECAUSE they look like failures
(deep pullback, vol expansion, distance from 52w high). But the ML model
distinguishes the rebound subset from the death-spiral subset using
patterns the hand-engineered failure score doesn't capture. The hand-
engineered failure score is **redundant in the rebound case** and
**counterproductive in the deep-value rebound case**.

Same logic applies to Pillar 2 (trend gate) and Pillar 4 (archetype
matching). The forensic archetype centroid IS itself a "deep-value
recovering name" pattern (high vol, deep pullback, base-building) but
because we apply it as a SCORE BOOST rather than a filter, it tends
to over-emphasise names that match the median-winner pattern at the
cost of names that match an off-median pattern but still ML-rank high.

## What would change the picture

1. **Higher-frequency rebal** (weekly): trend confirmation arrives faster;
   trend gate may add value.
2. **Real fundamentals** (Beneish M, Altman Z, Sloan accruals): may
   discriminate rebound vs death-spiral better than price-only features.
3. **A separate model for the bear-rebound regime**: train a dedicated
   model only on bear-period deep-value rebounds; switch in during
   the correct regime. The current ML model is one-size-fits-all.
4. **True novel-math features** (full TDA, full HMM): the fast surrogates
   used here did not capture the hypothesised non-linear structure.

This is the binding constraint on the 100%+ CAGR target on PIT S&P 500
at monthly rebal: there is no orthogonal cheap signal in price-only
features, on this universe, at this frequency, that the ML model has
not already absorbed.
