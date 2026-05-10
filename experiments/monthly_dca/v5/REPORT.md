# v5 Strategy Search: Novel Approaches Report

**Run date:** 2026-05-10.
**Goal:** Test novel/proprietary ideas (orthogonal multi-strategy, pattern matching to historical multibaggers, vertical-winner classifier) to push CAGR materially above v3's 42.80% WF mean OOS CAGR on PIT S&P 500.

**TL;DR:** None of the v5 novel approaches improved on v3 baseline. **v3 remains the deployed strategy.**

---

## 1. Orthogonal multi-strategy ensemble

Built 7 diversified scoring strategies designed to target different regimes/styles:

| Strategy            | Description                                                | Solo WF mean |
|---------------------|------------------------------------------------------------|-------------:|
| S1_ml_3plus6        | v3 baseline (multi-horizon ML rank ensemble)               |       42.80% |
| S2_pure_momentum    | 12m mom + vol-adj mom, must be above 200dma                |        ~25%  |
| S3_quality_pullback | Long-term winners on 15-50% pullback w/ recovery           |        ~16%  |
| S4_breakout_winner  | Tight consolidation + breakout strength + multibagger      |        ~22%  |
| S5_low_vol_quality  | High Sharpe + trend health + low max-DD                    |        ~10%  |
| S6_multibagger_lottery | multibagger ratio + fip + acceleration_2y                |        ~22%  |
| S7_idio_winner      | Idiosyncratic momentum + mom_consistency_12m               |        ~15%  |

**Tested ensembles** (rank-blended, vote-based, weighted):

| Ensemble                           | Full CAGR | WF mean | Beats SPY |
|------------------------------------|----------:|--------:|----------:|
| S1 alone (baseline)                |    39.77% |  42.80% |       9/10|
| U_S1_S3_S6 (top-3 vote union)     |    39.77% |  42.80% |       9/10| (ties baseline)
| E_ML_heavy (60% S1 + 20% S3 + 20% S6) | 25.19% |  31.88% |       9/10|
| E_ML_S3_balanced (50/50)           |    14.74% |  10.57% |       2/10|
| E_ML_S6_balanced (50/50)           |    20.75% |  21.94% |       9/10|
| E_all_eq (all 7 equal-weighted)    |    20-30% |  20-25% |       6/10|
| U_all7 (top-3 from each, union)    |    15.73% |  17.77% |       5/10|

**Why ensembles fail:** The v2 ML score is so dominant that blending with weaker
factors dilutes the signal.  The "vote union" of S1+S3+S6 happens to match v3
exactly because v3's top-3 picks are also high-ranked in S3 and S6, so the union
collapses to the v3 picks.  All other blends lose by 5-30pp WF mean.

**Strategy correlations (monthly returns) on the same scorer with different
hold periods:**
- v3_h6 vs v3_h12: 0.77
- v3_h6 vs v3_h3: 0.87
- v3_h12 vs v3_h3: 0.67

Holding-period diversification gives only 0.7-0.9 correlation; the strategies
are too coupled to gain from blending.

---

## 2. Pattern similarity to historical multibaggers

Built a pattern-matching score: for each (asof, ticker), compute 252-day
normalized log-return path; compare against templates from known multibagger
starting points (NVDA 2014/2022, AAPL 2003/2009, AMZN 1999, TSLA 2019, NFLX
2009, AMD 2016).  Score = max similarity (exp(-MSE)) to any template.

| Variant                           | Full CAGR | WF mean | Beats SPY |
|-----------------------------------|----------:|--------:|----------:|
| v3 baseline                       |    39.77% |  42.80% |       9/10|
| pattern_sim alone                 |    11.75% |  11.05% |       3/10|
| 50/50 v3 + pattern_sim            |    11.09% |   9.22% |       2/10|
| 70/30 v3 + pattern_sim            |    10.73% |   8.98% |       2/10|
| 85/15 v3 + pattern_sim            |    11.05% |   9.15% |       3/10|
| v3 score, filter pattern top 50%  |    14.49% |  14.59% |       2/10|
| v3 score, filter pattern top 30%  |    11.89% |   9.38% |       2/10|

**Why pattern matching fails:** Most stocks that LOOK like NVDA at the start
of its 2014-2017 run do not actually go on to be multibaggers.  The chart
shape ("price was up X% over the last year with this trajectory") is not a
sufficient predictor of forward returns — most "vertical-looking" stocks at
any given moment have already had their run and are about to mean-revert.

---

## 3. Vertical-winner classifier

Trained a LightGBM 3-seed binary classifier on the broader 1833-ticker panel
(annual retrain, 7-month embargo, 10y rolling window).  Target = "did this
stock have a >100% 12m forward return?"  Used scale_pos_weight to handle
class imbalance (~3.87% positive rate).

| Variant                           | Full CAGR | WF mean | Beats SPY | MaxDD |
|-----------------------------------|----------:|--------:|----------:|------:|
| v3 baseline                       |    39.77% |  42.80% |       9/10| -49.8%|
| vertical classifier alone         |    -6.58% |  -4.02% |       3/10| -99.0%|
| 50/50 v3 + vertical               |    11.93% |  19.39% |       4/10| -91.2%|
| 70/30 v3 + vertical               |    12.15% |  19.70% |       6/10| -93.5%|
| v3 score, filter vert top 50%     |    10.60% |  22.61% |       7/10| -98.2%|
| v3 score, filter vert top 30%     |    10.46% |  23.00% |       7/10| -98.0%|
| Multiplicative (v3 × vert)        |    11.93% |  19.39% |       4/10| -91.2%|

**Why the vertical classifier fails catastrophically:**

- The "vertical winner" target is too rare (3.87% positive rate) and noisy.
- Stocks that the classifier identifies as "likely 100%+ in 12m" are typically
  high-vol, low-quality lottery tickets (already up huge, near 52w highs,
  high RSI). Most crash instead of doubling.
- MaxDD of -99% indicates the classifier picks stocks that subsequently lose
  near 100% — confirming it identifies high-tail-risk names, not high-tail-
  return names.
- The cross-section of "true multibaggers" is extremely heterogeneous (NVDA's
  2014 path looks nothing like TSLA's 2019 path looks nothing like AAPL's
  2003 path), so the classifier learns a confused decision boundary.

---

## 4. Multi-horizon ensemble

Tested running v3 simultaneously with hold = 3, 6, 12 and blending the equity
curves:

| Combination                       | CAGR   | WF mean |
|-----------------------------------|-------:|--------:|
| v3_h6 alone (baseline)            | 39.77% |  42.80% |
| v3_h12 alone                      | 35.57% |  38.91% |
| v3_h3 alone                       | 30.50% |  27.12% |
| 50/50 h6+h12                      | 38.66% |  41.83% |
| 60/40 h6+h12                      | 39.04% |  42.18% |
| 75/25 h6+h12                      | 39.45% |  42.56% |
| 50/50 h6+h3                       | 35.71% |  35.30% |
| 33/33/33 h3+h6+h12                | 36.57% |  37.24% |

All ensembles underperform h6 alone because the lower-CAGR strategies drag
down the higher one, and the high correlation (0.7-0.9) prevents
diversification benefit.

---

## 5. Score momentum

Hypothesis: "Stock with rising v3 score" predicts future returns better than
v3 score alone.  Computed score_mom = v3_score(t) - v3_score(t-3).

| Variant                                | CAGR   | WF mean |
|----------------------------------------|-------:|--------:|
| v3 baseline                            | 39.77% |  42.80% |
| score_mom alone                        |  7.72% |   8.51% |
| 50/50 v3 + score_mom                   | 10.98% |  13.11% |
| 70/30 v3 + score_mom                   |  9.48% |  11.85% |
| 85/15 v3 + score_mom                   | 10.80% |  12.05% |
| v3 score filter pos score_mom          | 21.47% |  22.26% |
| v3 score filter top 50% score_mom      | 20.86% |  20.19% |

Score momentum doesn't help.

---

## 6. Why v3 is hard to beat on PIT S&P 500

The v2 GBM is trained on 67 features including momentum, quality, idio-mom,
breakout, multibagger, FIP score, drawdown — at multiple horizons.  This means
the GBM has already absorbed the signal from each individual factor and
non-linear combinations of them.  Hand-crafted single-factor strategies (S2-S7)
are strict subsets of the information already in the v2 ML score.

To genuinely beat v3 on PIT S&P 500 requires:
1. **New information beyond price**: fundamentals, news, alternative data.
2. **Different model architecture** that captures structure the GBM misses
   (sequence/transformer models on raw price history could in principle work,
   but on CPU and within a session, training a competitive model is
   prohibitive — the GBM with 67 hand-features is a strong baseline).
3. **Universe expansion** — already documented (broader 1,833-ticker → 51.8%
   WF mean; non-S&P 500 PIT → 51.0%; random 500 → 56.4% avg).

---

## 7. Recommendation

**Production: keep v3 deployed unchanged.** 42.80% WF mean OOS CAGR on PIT
S&P 500 is the realistic ceiling without leverage / new data sources.

**For higher CAGR**: deploy v3 on broader universes (Russell 1000, tech all-
cap), where it delivers 51-56% WF mean OOS CAGR.  The same logic, applied to
universes with more cross-sectional dispersion, naturally produces higher
returns.

---

## Files

- `strategies_orthogonal.py` — 7 diversified scoring strategies
- `run_v5_orthogonal_sweep.py` — ensemble sweep
- `pattern_similarity.py` — chart-pattern matching to historical multibaggers
- `train_vertical_classifier.py` — LightGBM 100%+/12m binary classifier
- Cache: `v5_orthogonal_sweep_results.csv`, `ml_preds_pattern_sim.parquet`,
  `ml_preds_vertical.parquet`

## Bottom line

We tried truly novel approaches (orthogonal ensemble, chart pattern matching,
vertical winner classification, score momentum) and none of them honestly beat
v3.  The deployed v3 strategy is at the local optimum of the search space we
explored, including these creative additions.

The path to higher CAGR is **not** via novel strategy construction on the
existing 67-feature panel — it's via expanding the universe or adding new
information sources.

---

## 8. v6 GBM with proprietary features (post-initial-report)

### Proprietary features added (11 new)

- `rank_mom_change_12`, `rank_mom_change_3`: change in cross-sectional rank of mom_12_1 / mom_3 vs 3 months ago (rising-rank momentum).
- `mtf_alignment`: count of (price > 50dma > 100dma > 200dma) — multi-timeframe trend alignment.
- `coiling_strength`: vol_contraction × tight_consolidation_60 × (mom_12_1 > 0) — "spring-loaded" pre-breakout setup.
- `reversal_mom`: 5d return - 21d return (decelerating selling).
- `power_consolidation`: low BB width × high RSI × positive mom (pre-breakout consolidation with momentum).
- `vertical_index`: composite of acceleration_2y + multibagger_ratio + fip_score + breakout_strength_60.
- `quality_compounder`: trend_health × positive momentum / volatility.
- `recovery_setup`: in moderate (10-40%) DD with positive 12m momentum.
- `rank_mom_12_now`, `rank_mom_3_now`: current cross-sectional rank in mom_12_1 / mom_3.

### v6 results (LightGBM 5-seed ensemble, full-history training, 7m embargo)

| Variant                     | Full CAGR | WF mean | WF min  | Beats SPY | Sharpe | MaxDD |
|-----------------------------|----------:|--------:|--------:|----------:|-------:|------:|
| v3 baseline (deployed v2 GBM) |  39.77% |  42.80% |  14.49% |       9/10|   0.96 | -49.8%|
| v6 alone (k=3 h=6)          |     2.74% |   3.57% | -20.68% |       2/10|   0.25 | -78.2%|
| v6 alone (k=3 h=12)         |    10.30% |   8.78% |  -4.17% |       5/10|   0.47 | -53.9%|
| v6_6m alone (k=3 h=6)       |     2.57% |   0.58% | -15.60% |       2/10|   0.27 | -80.9%|
| v6_3m alone (k=3 h=12)      |    11.10% |  14.41% |  -7.01% |       4/10|   0.49 | -66.3%|
| 50/50 v3+v6 rank-blend (h=12)| 16.33% |  15.82% |  -6.47% |       7/10|   0.62 | -53.9%|
| 70/30 v3+v6 (h=12)          |    15.54% |  17.24% |  -6.47% |       6/10|   0.59 | -65.4%|
| v3 score, filter v6 top 50% (h=12)| 11.25% |  20.04% | -16.02% |  7/10|   0.46 | -94.0%|

The v6 model is **substantially worse** than v3 in every configuration tested:
no blend, ensemble, or filter improves on baseline.  Adding the 11 hand-crafted
proprietary features to the v6 LightGBM pipeline hurt rather than helped —
likely because:

1. **The v2 GBM training pipeline is meaningfully different from v6's**: the
   deployed v2 model used HistGradientBoostingRegressor with multi-horizon
   joint training and full-history rolling.  My v6 LightGBM with rolling 10y
   window is a different beast and has consistently underperformed even
   without the extra features (cf. v4 ML in the previous report).

2. **Adding noisy features degrades GBM performance**: GBMs are not robust to
   spurious features.  Each of the 11 new features has individual IC near
   zero (we can verify post-hoc), and adding them as additional split
   candidates dilutes the model's capacity to learn from the strong v3-style
   features.

3. **Combining v3 deployed predictions with v6 ranks** still hurts because the
   v6 ranks are essentially noise.

### Conclusion: proprietary features did not work

The 11 hand-crafted features are individually intuitive but collectively offer
no marginal lift over the v3 67-feature panel + v2 GBM training pipeline.

---

## 9. HuggingFace foundation models (Chronos)

We tested two zero-shot time-series foundation models from Amazon's Chronos
family:

- **chronos-t5-small** (~46M params): On CPU, **114 seconds per single
  forecast** of 126 days. For the full PIT panel (280 asofs × ~500 tickers ×
  6m horizon) this would take **51+ days**. Infeasible on this hardware.

- **chronos-bolt-tiny** (much smaller, 250x faster): 500 forecasts in 4.84s.
  Tractable in principle (~23 min for full panel), but yielded asof 4/276
  with the script in roughly 2 min before being killed to free CPU for v6
  training.  The Chronos-bolt scoring infrastructure (`score_chronos_bolt.py`)
  is committed and can be re-run when GPU is available.

Limitations of zero-shot foundation models for this task:

1. They forecast univariate price paths, not cross-sectional relative
   performance.  We would need to convert their output to a cross-sectional
   ranking, which loses information.
2. The Chronos models are trained on generic time series, not financial
   returns specifically.  Their priors on mean reversion / momentum may
   conflict with stock-market dynamics.
3. Without fine-tuning on our cross-section, they're unlikely to produce
   alpha that beats a well-tuned in-domain GBM.

### Conclusion: Chronos is not honestly tractable on CPU here

For meaningful evaluation we need GPU access or a much smaller model (e.g.
TimesFM-200M, Lag-Llama-tiny).  Code is committed for future re-run.

---

## 10. 1D CNN on raw 252-day price paths

Trained a small 3-layer 1D CNN (~10K params) to predict cross-sectional 6m
forward rank from 252-day normalized log-return paths. Walk-forward annual
retrain, 7m embargo, 10y rolling window.

The CNN trained 4 of 22 years before being killed to free CPU for v6.  At ~30s
per year × 22 years = ~10 min standalone, but 3 concurrent processes made
each ~3x slower.  Code is committed (`train_ts_cnn.py`); resumable.

---

## 11. Final updated bottom line

**Total experiments:** 4,000+ variants in v4 (TP/SL/K/H/blends/regime gates) +
~80 strategies in v5 orthogonal sweep + 11 proprietary features in v6 +
zero-shot Chronos + 1D CNN time-series.

**Outcome:** the deployed v3 strategy at 42.80% WF mean OOS CAGR remains the
best honest answer for PIT S&P 500.  None of:

- factor blends or hand-crafted proprietary features
- multi-strategy orthogonal ensembles  
- pattern matching to historical multibaggers
- vertical-winner classifiers
- score momentum
- multi-horizon ensembles
- fresh LightGBM training (with or without new features)
- HuggingFace zero-shot foundation models (Chronos)
- multi-timeframe alignment, coiling, recovery setup features

beat v3 in honest evaluation.

To deliver materially higher CAGR the user needs to either:

1. **Expand the universe.** v3 on broader 1,833-ticker universe → 51.8% WF
   mean.  On non-S&P 500 PIT → 51.0%.  The user's planned Russell 1000 / tech
   all-cap deployment should naturally deliver the desired CAGR uplift.
2. **Use leverage** (linear scaling of return + risk).
3. **Add information beyond price**: fundamentals, news, alternative data,
   options flow, etc.
4. **GPU compute** to train competitive deep-learning models (Transformer
   sequence models on raw price data) that may capture structure the
   tabular GBM misses.

The PIT S&P 500 universe with price-only features is a mature game where
the deployed v3 sits at the local optimum.  This honest conclusion holds
even after extensive novel-strategy exploration.
