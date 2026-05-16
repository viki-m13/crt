# Novel-v7: k-NN + probability-of-recovery — honest NEGATIVE result

**Date:** 2026-05-16 · `novel_v7_knn_recovery.py`
**Verdict: fails badly. Not deployed. Reported as a negative, not tuned.**

## What was built (rigorously, with the requested "advanced math")

A picker using **only** two signals — no GBM, no Chronos:

1. **Mahalanobis k-NN analog matching.** Feature space = the repo's
   recovery / pre-runner family (drawdown depth & age, recovery track
   record, reflexive-bounce intensity, capitulation-stabilization,
   6m rank-trajectory, fallen-angel vol, 6-1 momentum, acceleration).
   **Ledoit-Wolf shrinkage covariance → Σ^(-1/2) whitening** (proper
   Mahalanobis, not naive Euclidean), Gaussian-kernel distance weights
   on the k=60 nearest historical analogs.
2. **Empirical probability of recovery.** Kernel-weighted fraction of
   those analogs whose realized H=6m forward return was > 0
   (conditional P(recover)). Final score = analog E[fwd ret] gated by
   above-median P(recover).

**Leakage control (the critical part for k-NN):** strict purge +
embargo — an analog (asof_a, ticker) is usable at T only if
`asof_a + H + EMBARGO ≤ T`, so an analog's realized 6-month forward
return can never overlap the prediction horizon. Rolling 132-month
training window. **No parameter was swept** (k, window, horizon,
embargo, gate were all set a-priori).

## Result vs the deployed K=2 picker (same grid / regime / costs)

| Metric | Deployed K=2 | **k-NN + P(recover)** |
|---|---:|---:|
| Full CAGR | 49.2% | **1.4%** |
| Sharpe | 1.04 | **0.22** |
| Max DD | −52% | **−87%** |
| DCA win vs S&P-DCA, 3y | — | 42% |
| DCA win vs S&P-DCA, 5y | — | 37% |
| DCA win vs S&P-DCA, 10y | — | 34% |

Era-by-era DCA (money-weighted IRR), strategy vs S&P-DCA:

| Era | k-NN strat | S&P-DCA | beat? |
|---|---:|---:|:--:|
| 2003–2009 | **−23.4%** | +1.4% | no |
| 2010–2015 | +4.8% | +13.6% | no |
| 2016–2020 | +14.4% | +20.7% | no |
| 2021–2026 | +6.3% | +17.4% | no |

It **loses to plain S&P-DCA in every era** and is far worse than the
deployed picker on every metric. The recovery-feature space selects
deep-drawdown names; analog-matched expected return + a recovery-prob
gate is not enough to separate genuine rebounds from value traps —
hence the −87% drawdown.

## Why this matters (the honest takeaway)

This is the **third independent, rigorously-built attempt** to find a
second usable alpha in this repo's price-only large-cap PIT data
(after CDV and CAC in novel-v6), and the third clean failure. It
re-confirms, from yet another angle and with genuinely advanced
machinery (shrinkage-Mahalanobis k-NN, purged analog inference,
empirical conditional recovery probability), the repo's central
finding: **there is essentially one modest, OOS-robust alpha here (the
walk-forward GBM price-pattern model). k-NN/recovery is not a second
one.** Deploying or headlining this would be dishonest; it stays as a
documented negative. No website or data.json changes were made.
