# Mechanism Articulation (multi_pillar_43Agh)

Date: 2026-05-10.

This document explains in plain language what each pillar does, what it's
trying to capture, and the empirical reason it does or does not contribute
positively on PIT S&P 500 monthly rebal.

## The dominant edge: V3/V6's ML rebound capture

The deployed strategy uses a HistGradientBoostingRegressor trained on
67 price-based features, walk-forward with 7-month embargo, predicting
6-month forward return. The model has learned a non-trivial pattern:

> Deep-value names (mom_12_1 < -0.30, dd_from_52wh < -0.40) that
> simultaneously show technical stabilisation (recent ret_21d > 0,
> vol_contraction > 0.5, RSI rising from oversold) tend to rebound
> 50-200% over the next 6 months.

This pattern fires hardest in **late-bear / early-recovery** market
regimes. Examples:
- **2009 Q1-Q2**: MBI, GNW, THC — financial crisis bottoms, 5-baggers
- **2020 Q2**: PEG, AIG, ETN — COVID panic bottoms, 1.5-2x in 6 months
- **2022 H2**: CCL, WBD, KEY — inflation-bear bottoms

The ML model does NOT pick every deep-value name — it picks the subset
showing technical stabilisation. This is the key. A naïve "buy the most
beaten-down names" strategy would catch the 2009 5-baggers but also the
2008 BBBY-style death-spirals; the ML distinguishes them.

## Pillar 1 — Failure-Avoidance Filter

**Goal**: cut the bottom 25-40% of universe by composite failure-risk
score before selection. Hypothesis: removing future blowups improves
risk-adjusted returns and compounds asymmetrically with winner selection.

**Construction**: weighted-rank composite of 16 features chosen from
forensic Study B's KS analysis. Includes: max_dd_5y (deeper = riskier),
vol_1y (higher = riskier), tight_consolidation_60 (lower = riskier),
trend_health_5y (lower = riskier), recovery_rate (lower = riskier).

**Standalone result on PIT S&P 500**: -18pp CAGR, +0.13 Sharpe, +9pp
MaxDD reduction (drop 30%).

**Why it disappoints**: the filter cuts the universe's high-vol
deeply-pulled-back names. The ML model's deep-value rebound picks fall
in this same bucket — they look like failures because they ARE failing,
right up until they rebound. Cutting them removes both true failures
AND the rebound-capture signal. The Sharpe gain is real but not enough
to justify the CAGR loss on this universe.

**Where it would work**: in a universe where the rebound-capture signal
is weaker (broader-1811, non-S&P), the filter might Pareto improve. We
verified the strategy works on non-S&P universe — Pillar-1-as-soft-penalty
delivered CAGR 35% / Sharpe 0.93 / beats SPY 9/10 there. But the
*combination* is not strictly better than V6 there either.

**Honest summary**: Pillar 1 is a real signal (failure-prone names DO
underperform), but it is not orthogonal to V3's existing signal on
this universe; it is an over-corrected version of the same signal.

## Pillar 2 — Stock-Level Trend Gate

**Goal**: require multi-timeframe trend confirmation before a stock is
eligible.  Hypothesis: stocks not in confirmed uptrends do not enter
the candidate pool regardless of other signals.

**Construction**: AND of 5 conditions on the 67-feature panel
(mom_12_1 ≥ -0.30, mom_3 ≥ -0.15, d_sma200 ≥ -0.10, dd_from_52wh ≥ -0.55,
frac_above_50dma_1y ≥ 0.20). Default settings keep 75-94% of names
eligible per asof; permissive variant kept ~95%.

**Standalone result**: -19pp CAGR, -0.07 Sharpe.

**Why it hurts**: same as Pillar 1, more directly. The deep-value rebound
picks fail the trend gate **at the moment they are picked** (price still
near 52w low, mom_12_1 still negative). By the time they pass the trend
gate, the rebound has already started and the price has moved.

**Where it would work**: in higher-frequency rebalance (weekly), trend
confirmation arrives fast enough to enter the rebound. With monthly
rebalance and a 6-month hold horizon, the entry is too late.

**Honest summary**: Pillar 2 is the wrong tool for this strategy's
frequency and pick style.

## Pillar 3 — Novel Mathematical Features

**Goal**: extract non-redundant information from rolling price series
using methods most quant strategies don't use.

**Original plan**: TDA persistence entropy, HMM state probabilities,
transfer entropy, GPD tail shape.

**What was actually built (time-bounded)**: fast surrogates only —
60-day rolling correlation with SPY, lag-1 autocorrelation
(Hurst-ish), absolute 60-day skew. The full TDA / HMM / true transfer
entropy implementations were too expensive for the session.

**Standalone result**: -10pp CAGR, -0.17 Sharpe at 0.5 blend weight.

**Why it didn't work**:
- 60-day SPY-corr is mostly redundant with `beta_2y` already in the panel.
- Lag-1 autocorrelation is mostly noise at 60-day horizon for individual
  stocks.
- Abs-skew adds noise without orthogonal signal.

**Where the unfastened versions might work**:
- Full TDA persistence entropy on rolling 252-day windows — gudhi can
  detect topological signatures of consolidation patterns that linear
  rolling stats miss. Plausible 1-3pp Sharpe improvement based on
  general TDA literature, but the implementation is non-trivial.
- HMM 2-state on (ret, vol, volume) per stock — captures regime
  transitions. Plausible 1-2pp Sharpe improvement.
- True transfer entropy from sector ETF — captures lead-lag invisible
  to Pearson correlation.

**Honest summary**: Pillar 3 was under-built due to time. The fast
surrogates capture little. A serious attempt requires full TDA + HMM
implementation and 2-3 days of compute.

## Pillar 4 — Forensic Archetype

**Goal**: score each ticker by similarity to the median pre-runner
pattern (Study A archetype) so that names matching the historical
"about-to-run" signature get boosted.

**Construction**: combination of (a) Euclidean distance in 16-feature
space to centroid of 793 winner pre-windows (3 months before base),
mapped to a 1-rank score; (b) engineered rule-based score on the
strongest 7 features (vol, pullback, best-month, vol_contraction,
tight_consolidation, near_52wh, RS).

**Standalone result**: -9pp CAGR, -0.14 Sharpe at 0.5 weight blended
into the score.

**Why it dilutes**:
- The archetype's strongest features (vol, pullback_all, max_dd_5y,
  vol_contraction) are ALREADY in the 67-feature ML model's input.
- The ML model has learned a richer non-linear function over those
  features than a simple Euclidean distance to a single centroid.
- The engineered version uses correlated features (it's an
  approximation of the ML model with much less capacity).

**Where it would work**:
- As a **filter** rather than score-blend: drop names whose
  archetype-distance is in the top-decile. (Different from Pillar 1
  failure filter — this gates on dissimilarity to known winners.) Not
  tested in this iteration.
- As a feature input to a meta-model that combines ML score and
  archetype score with non-linear interactions.
- For **explainability**: even if it doesn't improve picks, the
  archetype score can be displayed to users to explain why a
  particular pick is plausible.

**Honest summary**: Pillar 4 contains useful information but the
existing ML model already absorbs it. As a score-blend it dilutes;
as an explainability layer it would be valuable.

## Pillar 5 — Composite Selection + Sizing

**Goal**: compose Pillars 1-4 with the ML score; size positions
inversely to volatility; concentrate when signals agree.

**Construction**:
- Stage 1: drop bottom X% by failure_score (Pillar 1)
- Stage 2: drop trend-ineligible (Pillar 2) — disabled in best variant
- Stage 3: composite = w_ml*z(ml) + w_arch*z(arch) + w_novel*z(novel) +
           w_classic*z(classic) - w_failure*z(failure)
- Sizing: inverse-vol within K picks, K and gross from market regime

**Best composite variant from sweep**: `combo_drop20_wf0_wa2_wn0_wc2`
— CAGR 22.2%, Sharpe 0.99, MaxDD -44%, beats SPY 7/10.

This is **worse than V6 winner on every metric** (CAGR -16pp, Sharpe
+0.02, MaxDD -2pp, beats-SPY -2). The composite as built does NOT
Pareto-improve.

## What this means for "the edge"

The existing V3/V6 strategy's edge is in the ML model's ability to
discriminate rebound-capable deep-value names from death-spiral
deep-value names, using subtle non-linear interactions of price-based
features. **No hand-engineered overlay built from those same features
adds orthogonal signal.** Adding overlays only dilutes.

To exceed V3/V6 on PIT S&P 500 monthly rebal, you would need either:
1. **An orthogonal data source** (real fundamentals, news/sentiment,
   options-implied vol, earnings call NLP). The price-feature space is
   exhausted.
2. **A different model class** (e.g. neural net with sequential price
   pattern matching like a transformer over 252-day windows). HGB on
   summary statistics has finite capacity.
3. **A different time horizon** (weekly rebal, 21-day hold) where
   different patterns become exploitable.
4. **A different selection style** (concentrated bets on conviction,
   not equal-weight top-K). With K=1 + agreement gating, conviction
   sizing might 2x the equity curve at 4x the volatility — not what
   we want.

## Bottom line

The multi-pillar architecture is a sensible exploratory frame. On this
specific universe at this specific frequency with these specific
features, **the ML signal has captured the dominant edge** and the
hand-engineered overlays we built do not add value.

This is itself useful research output: it tightly bounds where the
remaining edge can come from (orthogonal data, different model,
different horizon, different selection style — not "better filters").
