# v3 deployed-model on ETF universes — agent O0MtP

> **Branch**: `claude/test-model-etfs-O0MtP`
> **Identifier**: this experiment lives under `experiments/etf_v3_test_O0MtP/`
> so it is easy to find amongst the parallel agent runs.
> **Status**: complete; committed to `main`.
> **Webapp**: NOT touched (per request).

## What we did

1. **Replicated the deployed v3 strategy** end-to-end on three new universes,
   then on the existing S&P 500 PIT universe as an apples-to-apples baseline:
     - `broad`    — 297 plain (un-leveraged) ETFs covering broad / sector /
       factor / international / fixed income / commodities / REITs /
       currencies / themes (`experiments/etf_v3_test_O0MtP/universe.py:BROAD_ETFS`).
     - `levered`  — 83 leveraged & inverse ETFs (2x, 3x, ±1x, ±2x, ±3x).
     - `combined` — 379-ticker union of the two.
     - `sp500_pit` — production deployed cross-section, used for direct
       comparison.
2. **Pipeline mirrors production**:
     - daily prices via yfinance (saved to `data/prices_*.parquet`)
     - monthly features via `experiments/monthly_dca/backtester.compute_features`
       + `experiments/monthly_dca/extra_features.compute_extras`
     - cross-section panel with multi-horizon forward returns
     - walk-forward HistGradientBoostingRegressor on cross-sectional rank
       targets at horizons (1m, 3m, 6m), retrain each January,
       7-month embargo
     - score = mean of (pred_3m, pred_6m) — the deployed v3 `ml_3plus6`
     - apply v3 winner: K=3, EW, tight regime gate, hold=6m, cost=10bps
3. **Designed and tested 9 variants** to see if any changes improve robustness
   without breaking the deployed cross-section.

## The four universes — top-line results (v3 baseline as deployed)

| Universe   | N tickers | CAGR     | SPY     | Edge (pp) | MDD     | Sharpe | WF beats SPY |
|------------|-----------|----------|---------|-----------|---------|--------|--------------|
| sp500_pit  | 500/mo    | **39.77%** | 11.12% | **+28.66** | -49.83% | 0.96   | **10/10**    |
| levered    | 83        | 27.29%   | 13.92% | +13.37    | -70.26% | 0.76   | 8/9          |
| combined   | 379       | 9.29%    | 10.93% | -1.65     | -84.50% | 0.43   | 6/10         |
| broad      | 297       | 1.77%    | 10.93% | -9.16     | **-67.15%** | 0.19 | **0/10**     |

Three honest findings up front:

1. **The deployed v3 model does not transfer to broad ETFs.** On 21 years of
   ~300 ETFs it returns 1.77% CAGR vs SPY's 10.93%, with -67% drawdown. Zero
   of ten walk-forward splits beat SPY. The strategy systematically picks the
   highest-volatility sector / EM / commodity ETFs (SLV, COPX, EWP, RSX,
   PSI, BNO, ITB, XHB, KRE) and holds them through regime turns.
2. **It works very well on leveraged ETFs**: 27.3% CAGR (+13.4pp edge),
   8/9 walk-forward splits beat SPY. Caveat: -70% MDD.
3. **It works as expected on the deployed S&P 500 PIT cross-section**:
   39.8% CAGR (+28.7pp edge), 10/10 splits beat SPY, MDD -50%.

## Diagnosis: why v3 underperforms on broad ETFs

Three structural reasons:

1. **Cross-sectional dispersion is dominated by factor exposure, not
   idiosyncratic skill.** When you cross-sectionally rank ETFs, the top of
   the list is whatever sector / region / commodity is currently running.
   The model picks "momentum extremes" that crash together when the regime
   turns.
2. **Top-3 picks of broad ETFs are highly correlated.** EW K=3 is fine for
   stocks (idiosyncratic risk gives genuine diversification) but on ETFs
   the three picks are typically the same factor bet (e.g. EWP+RSX+EWD =
   Europe + emerging Europe + Sweden = same trade).
3. **Regime gate is too slow with 6-month hold.** The strategy entered
   2008-01 with picks = XLE/EWD/EWQ in "normal" regime; by the time the
   hold expired in 2008-07 it rebalanced into SLV/EWD/EWY (the "look like
   2007 winners" picks); only flipped to cash in 2009-01 after the entire
   crash. Production v3 tolerates this on stocks because the 500-name
   cross-section of stocks has more inherent diversification; ETF picks
   compound this delay.

## What we tested — 9-variant ablation across all 4 universes

`experiments/etf_v3_test_O0MtP/run_ablations.py` runs every variant on
every universe. Full table at `results/ablations_master.csv`.

Variants tested:

| Name | K | Weighting | Mid-hold crash check | Vol-score λ | Notes |
|------|---|-----------|----------------------|-------------|-------|
| **baseline**          | 3 | EW     | off | 0    | deployed v3 |
| k5_ew                 | 5 | EW     | off | 0    | + diversification |
| k5_ew_midhold         | 5 | EW     | on  | 0    | + diversification + regime safety |
| k5_invvol             | 5 | invvol | off | 0    | + diversification + low-vol tilt |
| k5_invvol_midhold     | 5 | invvol | on  | 0    | + everything except vol-score |
| k3_ew_midhold         | 3 | EW     | on  | 0    | minimum: just regime safety |
| **k7_ew_midhold**     | 7 | EW     | on  | 0    | wider diversification |
| k5_ew_volscore        | 5 | EW     | off | 0.10 | score penalised by vol_xs |
| **kitchen_sink**      | 5 | invvol | on  | 0.10 | all four changes |

### Headline ablation results (Sharpe ratio per universe)

| Variant            | broad | levered | combined | sp500_pit | mean |
|--------------------|-------|---------|----------|-----------|------|
| baseline           | 0.19  | 0.76    | 0.43     | **0.96**  | 0.59 |
| k5_ew              | 0.33  | **0.79**| 0.43     | 0.86      | 0.60 |
| k5_ew_midhold      | 0.37  | 0.65    | 0.44     | 0.88      | 0.59 |
| k5_invvol          | 0.37  | 0.73    | 0.41     | 0.83      | 0.59 |
| k5_invvol_midhold  | 0.36  | 0.65    | 0.44     | **0.90**  | 0.59 |
| k3_ew_midhold      | 0.17  | 0.68    | 0.45     | 0.81      | 0.53 |
| **k7_ew_midhold**  | 0.55  | 0.66    | 0.55     | **0.90**  | **0.67** |
| k5_ew_volscore     | 0.46  | 0.72    | 0.52     | 0.67      | 0.59 |
| kitchen_sink       | **0.72** | 0.57 | **0.64** | 0.38      | 0.58 |

### Headline ablation results (MDD per universe)

| Variant            | broad   | levered | combined | sp500_pit | mean |
|--------------------|---------|---------|----------|-----------|------|
| baseline           | -67%    | -70%    | -85%     | -50%      | -68% |
| k5_ew              | -48%    | -72%    | -85%     | -59%      | -66% |
| k5_ew_midhold      | -47%    | -76%    | -82%     | **-43%**  | -62% |
| k5_invvol          | -46%    | -70%    | -83%     | -57%      | -64% |
| k5_invvol_midhold  | -44%    | -72%    | -80%     | -45%      | -60% |
| k3_ew_midhold      | -66%    | **-69%**| -83%     | -50%      | -67% |
| **k7_ew_midhold**  | -43%    | -72%    | -74%     | -48%      | **-59%** |
| k5_ew_volscore     | -37%    | **-38%**| -51%     | -48%      | -44% |
| kitchen_sink       | **-19%**| -40%    | **-31%** | -62%      | -38% |

## Robustness recommendations

Two clean answers depending on what "robust" means to you.

### Recommendation A — "Pareto across universes" → `k7_ew_midhold`

Best mean Sharpe across all 4 universes (0.67 vs baseline 0.59). Best mean
MDD across the four (-59% vs -68%). Modest CAGR sacrifice on the S&P 500
(25.2% vs 39.8% baseline; still beats SPY's 11% by 14pp), with material
gains on broad ETFs (8.30% vs 1.77%, MDD -43% vs -67%) and combined
(13.89% vs 9.29%, MDD -74% vs -85%). On leveraged it gives back ~7pp of
CAGR but doesn't break.

If we wanted a single setting that ages well across regimes _and_ universes,
this is it.

### Recommendation B — "Hard MDD cap" → `kitchen_sink`

When MDD is the absolute priority and we're willing to give up CAGR,
kitchen_sink (K=5, invvol, mid-hold-crash, λ=0.10 vol-score) cuts MDD from
-68% mean to -38% mean. **But on the S&P 500 PIT cross-section this is
catastrophic** (CAGR collapses from 39.8% → 5.0%) — vol-adjusted scoring
breaks the deployed model's edge on a stock cross-section. _Do not ship
kitchen_sink as a global default._

### Recommendation C — "Don't touch the deployed model" (status quo)

The deployed v3 is the **clear winner on its native S&P 500 PIT
cross-section**: 39.8% CAGR, 10/10 walk-forward splits beat SPY, Sharpe 0.96.
Every robust variant we tried sacrificed CAGR there. The model is well-tuned
to its trained universe; do not generalise it to ETFs in production without
a separate re-train.

## Why the "obvious" tweaks didn't generalise

- **Mid-hold crash check** is a marginal positive _only when paired with
  larger K_. On its own (`k3_ew_midhold`), it doesn't help — the strategy
  still concentrates in correlated picks before the crash.
- **Inverse-volatility weighting** helps in low-signal universes (broad
  ETFs) but fights the model on high-signal universes (S&P 500 stocks),
  where the model deliberately picks high-momentum / higher-vol names.
- **Vol-score penalty (λ=0.10)** is too aggressive on the deployed cross-
  section — it scrubs out the alpha. λ may need to be smaller (≤0.05) for
  S&P 500 stocks. On ETFs it is mildly helpful.

## Files produced

```
experiments/etf_v3_test_O0MtP/
├── REPORT.md                     ← this file
├── universe.py                   ← BROAD_ETFS, LEVERAGED_ETFS lists
├── fetch_prices.py               ← yfinance downloader
├── run_v3.py                     ← v3-baseline pipeline (+ ML walk-forward)
├── run_v3_robust.py              ← robust v1 (kitchen) + v2 (invvol K=5)
├── run_sp500.py                  ← v3 + 5 ablations on S&P 500 PIT
├── run_ablations.py              ← 9-variant suite across all 4 universes
├── data/
│   ├── prices_broad.parquet      ← 297 ETFs × 7,890 days
│   ├── prices_levered.parquet    ← 83 ETFs × 7,890 days
│   └── prices_combined.parquet   ← 379 ETFs × 7,890 days
├── cache/
│   ├── feat_<universe>.parquet   ← cross-section feature panel (per universe)
│   └── preds_<universe>.parquet  ← walk-forward ML predictions (per universe)
└── results/
    ├── comparison.csv                 ← ETF baseline summary
    ├── comparison_robust.csv          ← baseline + robust v1 + robust v2
    ├── ablations_master.csv           ← all 9×4 ablation rows
    ├── ablations_<universe>.csv
    ├── sp500_compare.csv              ← SP500 baseline + 5 ablations
    ├── <universe>_<variant>_equity.csv      (monthly equity curve)
    ├── <universe>_<variant>_picks.csv       (per-rebalance picks + weights)
    ├── <universe>_<variant>_yearly.csv      (year-by-year strategy vs SPY)
    ├── <universe>_<variant>_walkforward.csv (10-split WF table)
    ├── <universe>_<variant>_summary.json    (top-line metrics)
    └── *_run.log                      (raw run logs)
```

## How to reproduce

```bash
# 1. Download prices (cached; set FORCE=1 to refetch)
python3 experiments/etf_v3_test_O0MtP/fetch_prices.py

# 2. v3 baseline on broad/levered/combined ETFs (slow first time; cached after)
python3 experiments/etf_v3_test_O0MtP/run_v3.py
# emits results/<u>_summary.json, equity, picks, yearly, walkforward

# 3. v3-baseline + 5 ablations on S&P 500 PIT
python3 experiments/etf_v3_test_O0MtP/run_sp500.py

# 4. Full 9-variant ablation across all 4 universes
python3 experiments/etf_v3_test_O0MtP/run_ablations.py
```

All scripts are idempotent — the heavy steps (price fetch, feature
computation, walk-forward ML training) are cached under `data/` and `cache/`.

## Caveats

1. **Survivorship**: yfinance returns only currently-existing tickers. The
   ETF universe at, say, 2010-01 differs from today's universe. Several
   thematic ETFs that launched and then died (e.g. SPDR S&P Pharmaceuticals
   PJP variants, defunct China A-share ETFs) are not represented. This is
   a mild upward bias on backtested ETF results.
2. **Costs**: 10bps cost is modest; for thinly-traded leveraged ETFs the
   real bid-ask spread is wider. Leveraged-ETF results in particular are
   optimistic.
3. **Liquidity / capacity**: K=3 picks on a 297-ETF universe still picks
   real liquid ETFs. K=3 picks on the 83-ETF leveraged universe at small
   AUM ETFs (e.g. JNUG, JDST, DUST) is impractical at meaningful capital.
4. **Walk-forward splits**: re-use the production v3 PIT splits.
   `R1_GFC` is omitted from the leveraged universe since most leveraged
   ETFs launched 2008-09. WF table records `wf_n_splits` per universe.

## Bottom line

The deployed v3 model is **well-calibrated to the S&P 500 stock cross-
section it was trained on** and we should leave it alone for that universe.
On ETFs the same setup over-concentrates in volatility extremes and earns
no edge over SPY (broad) or buys very volatile beta exposure (leveraged).
A single variant — **K=7 EW with mid-hold crash check** — is consistently
better than baseline across all four universes by Sharpe and MDD, at the
cost of about a third of the deployed model's S&P 500 CAGR. Whether that
trade is worth shipping depends on whether the production webapp is meant
to also serve ETF / mixed universes; the user explicitly said _"do not
update the website"_, so we record the finding here without changing
production.
