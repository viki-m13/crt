# 02 — Inventor's Notebook

The forensic study told us something the published trend literature gets
**wrong**. Anchor that finding, then invent.

## Step 1: What's already known (the floor)

Catalog of standard ideas (any new strategy that overlaps too closely
fails the novelty bar):

1. **12-1 momentum** (Jegadeesh-Titman, Asness): rank by 12-month return
   skipping last month. Buy top decile.
2. **Tight-base breakout** (O'Neill / Concretum): consolidation +
   contraction of volatility + breakout above resistance with rising
   volume. *The Concretum trend article belongs here.*
3. **Quality momentum** (AQR): combine momentum with profitability or
   stable earnings.
4. **Idiosyncratic momentum** (Blitz): residualize against market/sector
   before ranking.
5. **52-week-high proximity** (George & Hwang): nearness to 52wh as a
   psychological anchor.
6. **Industry / sector momentum**.
7. **Pullback-to-trend / mean-reversion-in-trend** ("buy the dip"):
   short-term oversold within long-term uptrend. **The current strategy's
   `pullback_in_winner` and `quality_pullback` are in this family.**
8. **Earnings post-announcement drift** (PEAD): ride positive surprises.
9. **Low-vol effect**: low-vol stocks earn excess.
10. **Acceleration** (3m mom – 12m mom): identifying inflection.
11. **Trend-following on indices / sectors**.
12. **Cross-sectional rank momentum / IBD-style RS rating**.

What this strategy must NOT be: any of (1)–(12) reweighted, ensembled,
or LightGBM'd. Those each have <0.5 Sharpe in OOS univariate tests on
S&P 500 since 2010 — the canonical playbook is exhausted on this universe.

## Step 2: Forensic study — pre-3x-runner footprint

Built `research/forensics/find_runners.py` and
`research/forensics/preruner_signatures.py`.

**Result.** Identified 1,724 historical 3x-in-12-month runners across
the 1833-ticker panel from 1997–2026. Compared their feature distributions
at run-start to 1,500 stratified-by-year non-runner controls.

### What the data screams

| Feature                  | Median runner | Median non-runner | AUC   |
|--------------------------|--------------:|------------------:|------:|
| **vol_3m**               | **0.79**      | 0.28              | **0.91** |
| **vol_1y**               | 0.71          | 0.30              | 0.91 |
| **dd_from_52wh**         | -55%          | -11%              | 0.89 (inverse) |
| **bb_width_pct**         | 0.34          | 0.11              | 0.88 |
| **accel** (5d - 21d ret) | +0.13         | -0.01             | 0.77 |
| **drawdown_age_days**    | 169 days      | 77 days           | 0.67 |
| **vol_expansion_24m**    | 1.30          | 0.98              | 0.68 |

**Stable across all four eras** (1997-2007 / 2008-2014 / 2015-2019 /
2020-2025). The numbers wiggle but the direction never flips.

### What this contradicts

The pre-3x-runner footprint is the **opposite** of the published
"tight-base breakout" / "anatomy of a trend" prescription:

- ❌ Published: tight consolidation, low realized vol, BB squeeze
  → Reality: **HIGH realized vol, WIDE BB**.
- ❌ Published: near 52wh, momentum already positive
  → Reality: **45-70% off 52wh, 12-1 momentum NEGATIVE.**
- ❌ Published: cleanly trending up
  → Reality: **been falling for ~6 months, just stopped getting
  worse.**

The conventional wisdom catches *late-stage* breakouts — i.e. the period
after the run has begun. It misses *pre-runners* almost entirely, which
is why most published trend strategies under-perform their backtests.

### What this is consistent with

A two-mechanism story:

**M1. Volatility risk premium for fallen angels.**  Institutional risk
managers force selling on stocks whose realized vol exceeds risk
budgets. This selling is mechanical, not informational. It mis-prices
high-vol stocks downward. When the selling pressure abates, the
mis-pricing reverses — fastest in the names with the most accumulated
mechanical selling.

**M2. Dispersion compression after capitulation.**  Bear phases
compress cross-sectional dispersion *and* steepen the price-of-bad-news
curve. When dispersion normalizes, the steepest discounts unwind first.

The signal we're looking for isn't "buy oversold". It's: **"buy stocks
that LOOK like the pre-runner archetype, but where the
mechanical-selling phase has just ended."**

The hardest problem isn't *finding* candidates with this footprint —
the forensic table makes that easy. It's *separating runners from
continued-failures* among candidates that all look the same on day one.

## Step 3: 22 invention candidates

I push past the obvious. Each candidate is judged on novelty,
mechanism plausibility, feasibility (price-only, S&P 500, 25y),
expected edge.

### A. Novel representations

**C1. Cross-Sectional Rank Trajectory (CRT).**
For every stock at every month-end T, compute its 21-day return rank
percentile across the universe. Look at the time series of percentiles
over the last 6 month-ends — call this the rank trajectory. Compute the
Spearman correlation between time and rank. Stocks with persistently
upward rank trajectories — even if absolute price is still declining —
are *covertly winning the relative game*. This is conceptually
different from absolute momentum because a stock can be at -30% YTD
yet still be improving its rank in a falling market.
Mechanism: institutional rotation reaches the "next layer" of laggards
before absolute breakout confirms.
Novelty: high. Standard rank-momentum uses point-in-time rank, not
the *time-derivative* of rank.
Score: **9/10 build it.**

**C2. Pre-Runner Archetype K-NN (PRA).**
From the 1,724 historical pre-runners, compute a feature vector at each
run-start. Cluster into ~20 archetypes. For each candidate at rebalance
T, measure distance to nearest archetype centroid. Long the names
closest to the archetypes. Train/test purged so future runners aren't
in the train fold.
Mechanism: case-based reasoning at scale.
Novelty: high. KNN against historical archetypes ≠ feature-zoo
regression.
Score: **8/10 build it.**

**C3. Distributional shape of recent returns (DSR).**
Instead of return mean / vol, use the *shape* of the daily-return
distribution over last 60 days: skew, excess kurtosis, ratio of squared
positive returns to squared negative returns ("volatility asymmetry").
The intuition: pre-runners' return distribution is becoming more
positively asymmetric *before* the mean turns positive.
Mechanism: distribution-shape change leads mean change.
Novelty: medium-high. Skew-of-returns is published; *combination of
skew + vol-asymmetry + tail-ratio + transition speed* is not.
Score: **7/10.**

**C4. Topological persistence of return trajectories.**
Embed the 252-day return path in 2D using a sliding window. Compute
persistent homology features (longest 0-dim loop, number of
0-persistence classes). Pre-runner topology should differ from
non-runner topology.
Mechanism: complexity-theoretic distinction.
Novelty: very high.
Feasibility: requires `gudhi` or `ripser` — extra install. Defer.

**C5. Wavelet / spectral signature.**
Decompose 252-day returns into frequency components. Pre-runners may
show specific power at intermediate frequencies (60-120d cycles)
indicating accumulation/distribution waves.
Novelty: high.
Feasibility: needs `pywt`; medium effort.

### B. Mechanism-based signals

**C6. Capitulation-stabilization transition (CST).**
Explicit model of the transition from "high-vol downtrend" to
"high-vol sideways". Operationalized as: 60-day OLS slope of log-prices
recently approaching zero from negative; 60-day rolling standard
deviation roughly constant; recent 21-day high not falling further.
Mechanism: the stabilization itself is the signal — institutions are
done forcing selling.
Novelty: medium. The components exist; the *gating condition* does not.
Score: **8/10.**

**C7. Reflexive bounce intensity (RBI).**
For each big down-day (return < -3% or < -2σ) in last 60d, count whether
at least one >+2% day occurred within the next 5 days. RBI = #(yes) /
#(big down days). High RBI = informed dip-buying = no death spiral.
Mechanism: presence of buyers absorbing supply.
Novelty: high. Not in published lit.
Score: **8/10.**

**C8. Cross-sectional vol-asymmetry leader.**
For each stock, compute (sum of squared up-day returns) / (sum of
squared down-day returns) over 60d. Then take the cross-sectional rank
of that ratio. Pre-runners cluster in the top quartile.
Novelty: medium-high.

**C9. Down-day-volume / up-day-volume ratio (need volume).**
Wyckoff accumulation: when price is falling but down-day volume is
LIGHTER than up-day volume, supply is being absorbed.
Novelty: low (Wyckoff is well-known) but *quantitative* version not
in standard quant packages. Need volume data.
Score: 6/10 — depends on volume fetch.

**C10. Open-to-close vs close-to-close decomposition.**
Overnight return ("open[T] / close[T-1] - 1") + intraday return
("close[T] / open[T] - 1"). When overnight returns turn positive while
intraday remains negative — informed flow accumulating between sessions.
Novelty: medium.
Feasibility: needs OHLC data. Defer.

### C. Network / graph signals

**C11. Lag-correlation neighborhood emergence.**
Build a 252-day lag-1 return correlation matrix. For each stock, compute
its degree-1 neighborhood (top-10 most correlated peers). When a stock's
neighborhood is averaging strong recent returns *while it itself is
not*, the stock is "downstream" of a moving sector and is about to
catch up.
Mechanism: lead-lag information diffusion.
Novelty: high.
Score: **7/10.**

**C12. Sector-flow position.**
For each ticker, infer sector by clustering on 252d returns. Score by
"distance from sector median 21-day return" — stocks that have lagged
their inferred peers despite belonging to the same flow are catch-up
candidates.
Novelty: medium.

**C13. Eigenvector centrality of return co-movement.**
Build a return-correlation graph and compute each stock's eigenvector
centrality. Stocks with rising centrality are becoming more integrated
into market flow — this often precedes sector-wide moves.
Novelty: high but speculative.

### D. Regime-aware

**C14. Forensic regime classifier.**
Cluster the 1700+ pre-runner events into 3-4 regime classes (e.g.
"crash bottom rebound", "fallen-angel rotation", "tech post-pullback",
"value rerate"). Detect current SPY+VIX regime; pick the matching
runner-archetype set.
Novelty: medium-high.
Score: **7/10.**

**C15. Dispersion-conditional selection.**
When cross-sectional dispersion of 21-day returns is HIGH, pick the
deepest-drawdown rebound candidates. When LOW, pick the
strongest-trend names. Dispersion as the regime switch.
Novelty: medium-high.

### E. Adaptive / online learning

**C16. Online feature-weight updating.**
Keep a rolling 24-month window of (feature_at_T, future_3mo_return).
At each rebalance, fit a robust regression and use the in-sample
coefficients as the weights for next month. Avoids stale historical
weights when regime shifts.
Novelty: medium.

**C17. Survival-analysis based exit timing.**
Fit a Cox-style survival model on past runners' "time-to-peak"
distributions, conditioned on entry features. At entry, predict the
expected time-to-peak; use it as the holding period rather than fixed
12 months or trailing stop.
Novelty: high (almost no published quant uses survival analysis for
exit timing).

### F. Counterintuitive combinations

**C18. Vol × drawdown × stabilization × trend health.**
The 4-way interaction: (high vol) × (deep drawdown) × (selling
decelerating) × (long-term trend healthy).
Each individually has known but moderate edge. The 4-way interaction
selects the *exact* pre-runner archetype while filtering out
falling-knife traps.
Novelty: medium-high (it's the geometric mean of forensic factors).
**This is the natural baseline-plus.**
Score: **8/10.**

**C19. Asymmetric volatility-of-volatility.**
Compute 60-day std of rolling 5-day realized vol. Pre-runners cluster
at *moderate* VoV (high vol but stable), not extreme VoV. Adds a
filter to remove "exploding" / "imploding" names.
Novelty: medium.

### G. Information-asymmetry proxies (price-only)

**C20. Closing-price gravitation index.**
Compute "% of last 60 days where close was above open" — measures
intraday accumulation.
Feasibility: needs OHLC. Defer.

**C21. Range-expansion asymmetry.**
For each day, compute (high-low) / 21-day-mean (high-low). Track
whether range expansion is correlated with positive close vs negative
close. Pre-runners show range expansion correlated with positive close.
Feasibility: needs OHLC. Defer.

### H. Pattern recognition

**C22. 1D CNN / Transformer on price-volume sequences.**
Train a small model (~50k params) on 252-day price-volume sequences,
labels = future 3x in 12mo (binary). Use purged WF.
Novelty: medium (the architecture is known; the application to
cross-sectional pre-runner selection less so).
Score: 6/10 — high effort, moderate expected lift; defer in favor
of more interpretable C1, C2, C7, C18.

## Step 4: Scoring + selection of top 5

Priority order based on (novelty × mechanism plausibility × feasibility):

| ID  | Name                                | Novelty | Plaus. | Feas. | Pick |
|-----|-------------------------------------|--------:|-------:|------:|-----:|
| C1  | Cross-Sectional Rank Trajectory     | 9       | 8      | 9     | ★★★★★ |
| C18 | Vol × DD × Accel × Trend (interact) | 6       | 9      | 10    | ★★★★ |
| C7  | Reflexive Bounce Intensity          | 8       | 8      | 9     | ★★★★ |
| C2  | Pre-Runner Archetype K-NN           | 8       | 7      | 8     | ★★★★ |
| C6  | Capitulation–Stabilization Transit. | 7       | 8      | 9     | ★★★ |
| C19 | Asymmetric Vol-of-Vol               | 6       | 7      | 9     | ★★ |
| C14 | Forensic Regime Classifier          | 7       | 7      | 7     | ★★ |

**Top 5 picks for Phase 2:** C1 (CRT), C18 (interaction), C7 (RBI),
C2 (PRA-KNN), C6 (CST). Combine winners.

## Step 5: The invention thesis

The *core* invention is a single signal: **Cross-Sectional Rank
Trajectory (CRT)** — the time-derivative of a stock's cross-sectional
return-rank percentile.

Combined with the **Pre-Runner Footprint Filter** (C18: high vol +
deep drawdown + selling decelerating + long-term healthy) — which is
where the forensic study explicitly anchors the strategy.

CRT measures *covert leadership emergence*: stocks improving their
relative-strength rank monotonically over 6 months, even if absolute
price is still in drawdown. The filter prevents the strategy from
buying generic dumpster fires.

Why it isn't arbitraged:
- Most factor zoos use point-in-time rank levels, not the
  time-derivative of rank.
- Many mainstream momentum users *avoid* names in drawdown by
  construction, exactly because the conventional 12-1 signal punishes
  recent losers. They miss the CRT signal because they screen out the
  candidates upstream.
- The signal is *visible only at the cross-section level over time* —
  retail tools and most factor backtesting tools work per-stock, not
  on the cross-sectional rank trajectory.
- Information cost: the signal requires a multi-month rank history
  computed across a clean PIT universe at every rebalance. Easy in
  principle, friction in practice.
- Capacity: signal is on S&P 500 names; capacity is high.

Phase 2 will build, test, and combine the candidates.
