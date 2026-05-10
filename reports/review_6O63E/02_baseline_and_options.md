# 02 — v3 baseline and the three candidate options

## v3 baseline (deployed today)

```
Strategy: strategy_rotation (ml_3plus6 scorer)
Universe: PIT S&P 500 (cache/v2/sp500_pit/sp500_membership_monthly.parquet)
K_normal=3, K_recovery=3, K_bull=3
Weighting: equal-weight
Regime gate: tight
Hold months: 6
Cost: 10bp round-trip per pick
```

Reproduced metrics (exact to 16 decimal places against the published JSON):

```
cagr_full       0.3977406189318511   → 39.77%
wf_mean_cagr    0.42800538003320804  → 42.80%
wf_median_cagr  0.39897919814733074  → 39.90%
wf_min_cagr     0.14492104439496223  → 14.49%
wf_max_cagr     1.0879351638068608   → 108.79%
sharpe          0.9553637477926258   → 0.955
max_dd          -0.49828619285029263 → −49.83%
n_cash_months   4
wf_n_pos        10/10
wf_n_beats_spy  9/10
```

## Option A — invvol weighting + 3% cash yield (`G0nfM` v6)

```python
V6Config(
    scorer="ml_3plus6", universe="sp500_pit", regime_gate="tight",
    k_normal=3, k_recovery=3, k_bull=3,
    weighting="invvol",      # NEW: 1/vol_1y weights, normalised to sum=1
    hold_months=6, cost_bps=10.0,
    cash_yield_yr=0.03,      # NEW: T-bill yield while in cash
)
```

Mechanism: textbook risk-parity at the pick level. Doesn't change which stocks are picked — only their weight. The lowest-vol pick gets ~45% of the basket, the highest-vol pick gets ~20%, instead of an exact 33.33%. The basket's realised vol drops without losing the cross-sectional alpha. The 3% T-bill credit during the 4 cash months adds ~0.1pp CAGR — trivial but honest.

**Headline reproduction on home (sp500_pit)**: CAGR 38.20% (−1.57pp), Sharpe 0.971 (+0.016), MaxDD −45.98% (+3.85pp), WF mean 42.48% (tied), WF min 20.92% (+6.43pp), beats SPY 9/10.

## Option B — invvol + bull-regime K=2 (`uDXqh` v8)

```python
V6Config(
    scorer="ml_3plus6", universe="sp500_pit", regime_gate="tight",
    k_normal=3, k_recovery=3, k_bull=2,     # NEW: K=2 in bull regime only
    weighting="invvol",
    hold_months=6, cost_bps=10.0,
    cash_yield_yr=0.03,
)
```

Mechanism: same as A, plus the basket concentrates to 2 picks (instead of 3) when SPY 12m momentum ≥ 10% AND SPY > 200-day MA. The bull regime is ~14.6% of months over 2003-2025 (39 of 268 active months). The agent's reasoning: in strong bull regimes, picks tend to follow the market up and concentration risk is lower.

**Headline reproduction on home**: CAGR 41.54% (+1.77pp), Sharpe 1.017 (+0.062), MaxDD −45.98% (+3.85pp), WF mean 46.15% (+3.35pp), WF min 24.16% (+9.67pp), **beats SPY 10/10**.

**Caveat**: the kb=2 cell was picked from 45 (kn, kr, kb) combinations evaluated on the same data. The agent acknowledges modest overfit risk.

## Option C — v3 + Chronos-bolt-tiny p70 filter (`zc4cv` v5)

```python
# Step 1: load Chronos-bolt-tiny predictions (zero-shot, p70 quantile of 3m forecast)
# Step 2: rank stocks cross-sectionally by chronos_p70 within each month
# Step 3: keep only stocks with chronos_p70_rank ≥ 0.4 (top 60%)
# Step 4: apply v3 (ml_3plus6, K=3 EW tight h=6) to the filtered subset
```

Mechanism: Chronos-bolt-tiny is a 9M-parameter zero-shot time-series foundation model from Amazon, trained on a diverse public time-series corpus. It produces probabilistic forecasts of future price paths. The p70 quantile of its 64-day forward forecast captures "expected upside". Filtering v3's pick pool to stocks where Chronos also expects positive 3-month returns removes the bottom 40% of stocks that v3 likes but Chronos disagrees with.

**Headline reproduction on home**: CAGR 44.81% (+5.04pp), Sharpe 1.036 (+0.081), MaxDD −49.83% (same), WF mean 45.86% (+3.06pp), WF min 17.01% (+2.52pp), **beats SPY 10/10** (R3 split flips from −1.5pp to +0.99pp).

**Caveats**:
1. q=0.4 was selected from a sweep of 30+ chronos × quantile × hold × K cells on the same data.
2. Chronos-bolt-tiny was released in 2024 — the training corpus may overlap parts of 2003-2023.
3. Adds external HuggingFace dependency at every rebalance (~3s CPU per month).

## Combinations also tested

- **A+C**: apply Chronos filter, then pick top-3 by v3 score, weight by invvol, 3% cash yield. Stacks the two principled changes.
- **B+C**: apply Chronos filter, then pick top-K (with kb=2 in bull), weight by invvol, 3% cash yield.

## What was NOT validated

- v7 hedge stack — different product
- v8 k=1+TLT — agent disclaims
- v8b leveraged — leverage, not alpha
- FHtzX Pre-Runner — WF mean below baseline
- 43Agh multi-pillar — every pillar reduces CAGR
- O0MtP ETF universes — explicitly "do not deploy on ETFs"

These do not improve v3 by their own reports. Including them in the deep validation would have been wasteful.
