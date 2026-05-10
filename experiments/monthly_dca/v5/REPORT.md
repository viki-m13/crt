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
