# 02 — Invention: Multi-Pillar Stock Selection (43Agh)

Date: 2026-05-10. Branch: `claude/multi-pillar-stock-strategy-43Agh`.
Namespace: `multi_pillar_43Agh`.

This document is the design for the new strategy. Read after `00_repo_map.md`
and `01_engine_audit.md`.

---

## 1. The architecture

A 3-stage pipeline as in the brief, materialised as five composable pillars
on top of the v6 engine:

```
                                    ┌─────────────────────────────┐
                                    │   PIT S&P 500 universe at T │
                                    └──────────────┬──────────────┘
                                                   │
                ┌──────────────────────────────────▼─────────────────────────────┐
                │ STAGE 1 — Failure-avoidance filter (Pillar 1)                  │
                │   composite failure-risk score; drop bottom 25-40% by score    │
                └──────────────────────────────────┬─────────────────────────────┘
                                                   │
                ┌──────────────────────────────────▼─────────────────────────────┐
                │ STAGE 2 — Trend & regime gate (Pillar 2)                       │
                │   • market regime → cash / risk-on / risk-on+concentrated      │
                │   • per-stock trend confirmation → eligibility                 │
                └──────────────────────────────────┬─────────────────────────────┘
                                                   │
                ┌──────────────────────────────────▼─────────────────────────────┐
                │ STAGE 3 — High-conviction selection (Pillars 3, 4, 5)          │
                │   composite_score = α·ML_3plus6                                │
                │                   + β·forensic_archetype                       │
                │                   + γ·novel_math                               │
                │                   + δ·classic momentum/quality                 │
                │   pick top-K, K varies with signal agreement (3/5/8)           │
                └──────────────────────────────────┬─────────────────────────────┘
                                                   │
                                ┌──────────────────▼──────────────────┐
                                │ Sizing (inverse-vol, per-pick cap,  │
                                │ regime-conditional gross exposure)  │
                                └─────────────────────────────────────┘
```

The engine runs the same `simulate()`. The new code lives in
`experiments/multi_pillar_43Agh/strategy/` and produces a `score_panel`
that the engine consumes.

---

## 2. Hypothesis (where the edge comes from)

The existing V3 strategy delivers ~40% CAGR with a -50% MaxDD. ~9,000
variants in v6/v7 sweeps are tightly clustered around it. The remaining
edge is **not** in better hyperparameters of the same scorer; it's in:

1. **Failure avoidance.** Of the V3 -50% MaxDD, a large fraction comes
   from a small handful of picks that imploded (MBI -65%, GNW -55%,
   THC -45% in 2008-09; CCL -50%, WBD -55% in 2022). A failure-avoidance
   filter that catches *most* of these saves more drawdown than any
   regime gate can.

2. **Stock-level trend.** V3 picks deep-value rebound names — many of
   which are still in confirmed downtrends. A subset will rebound (the
   ML model captures this); a subset will keep falling. A stock-level
   trend confirmation eliminates the second subset without losing the
   first (rebounds usually only become picks once they've stopped
   falling).

3. **Forensic archetype features.** V3 uses 67 generic price/momentum
   features. Pre-runner stocks have a recognisable signature (tight
   consolidation → volume expansion → breakout above prior pivot, with
   relative-strength leadership). A score that matches this archetype
   directly should add information not captured by mean-IC factors.

4. **Concentration scaling.** V3 picks K=3 always. When all top-3 picks
   align on multiple pillars (failure-OK + trend-up + archetype-match
   + ML-high), capital should concentrate further (K=2 or K=1). When
   pillars disagree, K should expand to 5-8 (or go to cash).

5. **Cash sleeve in hostile tape.** V3 is in cash 4 months of 22 years —
   far too rarely. The 2008 GFC drawdown is -50% precisely because the
   strategy stayed deployed. A regime gate that goes to cash for 6-12
   months in 2008 changes the equity curve materially.

Each of these is a small-to-medium edge alone (1-5pp CAGR, 0.05-0.15
Sharpe). **Combined, the brief's claim is that they compose to a
materially higher Sharpe and lower DD.** Whether they hit the aggressive
100% / Sharpe 3+ target on PIT S&P 500 is an honest open question; the
project will report the actual number, not massage it.

---

## 3. Pillars in detail

### Pillar 1 — Failure-Avoidance Filter

Output: `failure_score(asof, ticker) ∈ [0, 1]`. Higher = more failure-prone.

Components (forensic Study B drives the weights):
- **Technical breakdown**: trend collapse (mom_12_1 << 0 AND mom_3 << 0
  AND not yet recovering), distance from 52w high large and stretching,
  vol expanding, RS rank in bottom decile and dropping.
- **Vol-adjusted depth**: tail-ratio_24m > X AND drawdown_age_days > Y
  (slow-bleed pattern, more dangerous than sharp drop).
- **Quality proxies (price-only)**: low `quality_score_5y`,
  bad `recovery_rate`, multi-year `excess_5y_logret < 0`.
- **Forensic features from Study B** (TBD — derived in Phase 1).

We **do not** drop picks unconditionally; we drop the bottom 25-40% of the
universe by failure_score before Stage 3 considers them. This is the
"failure avoidance compounds asymmetrically" insight from the brief.

### Pillar 2 — Trend and Regime

#### Market regime
Three regimes: **risk_on**, **mixed**, **hostile**. Decision tree on
SPY features (see `regime.py`):
- `hostile` if `spy_dsma200 < -0.05` AND `spy_mom_12_1 < 0` AND
  `spy_dd_from_52wh < -0.15` (or any 21d-loss > 8%).
- `risk_on` if `spy_dsma200 > 0` AND `spy_mom_12_1 > 0.10` AND
  breadth (% above 200dma) > 0.55.
- `mixed` otherwise.

Plus a **breadth + dispersion** confirmation: count of S&P names with
positive 6m momentum (proxy for breadth), and cross-sectional return
dispersion (proxy for stock-picker tape).

In `hostile`: 100% T-bills. (`gross = 0`, cash credit applied.)
In `mixed`: K=5-8 picks, gross 0.6-0.8.
In `risk_on`: K=3 picks, gross 1.0.

#### Stock-level trend gate
A pick is **eligible** only if:
- `mom_12_1 > -0.10` (allow mild pullback but not deep downtrend)
- `mom_3 > -0.05` (recent action not catastrophic)
- `d_sma200 > -0.10` (within 10% of 200dma, not far below)
- `dd_from_52wh > -0.50` (not in death-spiral)
- `frac_above_50dma_1y > 0.30` (multi-month trend up at least sometimes)

This is **multi-timeframe trend confirmation** as the brief specifies.
The threshold values are conservative enough to keep V3's deep-value
rebounds (MBI/GNW/THC in early 2009 had pre-pick `mom_12_1 ≈ -0.40` —
those would be cut, **but** at the moment ML picked them they had often
moved up sharply already in the prior 21 days, so `mom_3 > -0.05`
might still admit them). **This will be empirically tuned in
Phase 2 P2.**

### Pillar 3 — Novel Mathematical Features

Implement at least 4 of the 7 in the brief, evaluate marginal IC:

1. **Topological persistence entropy** of rolling 60-day return series
   (gudhi). Hypothesis: pre-breakout consolidations have low persistence
   entropy (clean basing pattern); post-breakout uptrends rise.
2. **HMM 2-state probabilities** on (return, vol, volume) per stock.
   Feature: P(state=trend-up) over the last 21 days; transition activity
   in last 5 days (regime change).
3. **Transfer entropy** from sector ETF (XLK, XLE, XLF, …) to the stock,
   on 21-day rolling window. Measures lead-lag information flow.
4. **GPD tail shape** (scipy.stats.genpareto) on the left tail of 252-day
   returns. Shape > 0 = heavier tail = higher failure risk.

Less likely but optional:
- Correlation-network persistent homology (computationally expensive
  for 500-name universe at monthly frequency)
- Rough volatility / jump-diffusion (too many params, fitting fragile)
- Causal effect of earnings surprise (no earnings data in this engine)

Each feature passes a **standalone IC test** before inclusion. Anything
that doesn't add OOS information goes to `research/graveyard/` with
the experiment write-up.

### Pillar 4 — Forensic Archetype

From Phase 1 Study A, extract the signature of pre-runners over the 6-18
months before the run. Two implementations:

A. **Engineered**: build per-month features that explicitly compute
   "how closely does this ticker's last 6-12 months match the runner
   archetype". E.g., (range_pos_1y in [0.55, 0.75]) AND
   (vol_contraction > 0.50) AND (rs_12m_spy > 0.20) AND (mom_consistency_12m > 0.60)
   → score 1.0; else partial score.

B. **Learned**: nearest-neighbour distance in the (already-computed)
   67-feature space to centroids of known pre-runner cases. With careful
   leakage control: use only Study A cases whose pre-window ENDS before T.

We combine both; they capture different aspects.

### Pillar 5 — Composite Selection + Sizing

Composite score is a weighted sum on the **cleaned, trending universe**
(post-Pillar-1 filter and Pillar-2 stock-level gate):

```
composite(t, asof) =
    w_ml       * z(ml_3plus6(t, asof))
  + w_archetype * z(archetype(t, asof))
  + w_novel    * z(novel_math_composite(t, asof))
  + w_classic  * z(classic_mom_q(t, asof))      # mom_12_1, quality_score_5y
```

`z()` = cross-sectional z-score per asof. Initial weights:
`w_ml=1.0, w_archetype=0.6, w_novel=0.4, w_classic=0.4`. The relative
weights and signs are chosen by **walk-forward grid search** with
strict embargo (Phase 2 P5). They are NOT tuned to the frozen holdout
(2025-01 → 2026-05), which is touched once at Phase 5 only.

Sizing scales with signal agreement. Define `agreement(asof) ∈ [0,1]` =
fraction of top-K picks where all 4 pillars (ML, archetype, novel, trend)
score in their top-quintile. If `agreement > 0.7` → K=3, gross=1. If
`0.4 < agreement ≤ 0.7` → K=5, gross=0.8. If `agreement ≤ 0.4` → K=8,
gross=0.6 (and only if regime is `risk_on`; else cash).

### Sizing weights

Inverse-vol within the K picks (proven Pareto improvement in v6) plus the
gross scaler from the agreement rule, plus the regime gross overlay
(0 in hostile, 0.6-0.8 in mixed, 1.0 in risk_on).

---

## 4. What we will NOT re-attempt (graveyard guard)

From v6/v7 reports, these are documented dead ends:
- Stricter Faber/strict-DD market regime gates (lose recovery alpha)
- Crash fallback to SPY or TLT (both crashed in 2008/2022)
- Trailing stop on portfolio drawdown (forces exit during recovery)
- SPY-DD continuous gross scaling (asymmetric loss on recovery)
- Sticky cash re-entry (misses the V-rally)
- Vol penalty on score (mostly noise)
- K=4-5 standalone (without agreement scaling)
- Pure pullback filters (cuts deep-value rebound)

Multi-pillar avoids these failure modes by construction:
- Failure filter is technical AND quality-based, not just pullback
- Trend gate is multi-timeframe and uses ratio rules, not bright lines
- Cash decision is regime-conditional + signal-conditional (NEW)
- Concentration scales with agreement, not regime alone (NEW)

---

## 5. Calibration philosophy

Every weight, threshold, and hyperparameter is fit **only on data with
asof < frozen_holdout_start**. Frozen holdout = 2025-01-01 → 2026-05-07.
This is touched exactly once at Phase 5.

For walk-forward fit (in Phase 2/3), we use the same 10 splits as v6,
but each pillar's hyperparameters are tuned within an inner walk-forward
that excludes the test window of the outer split. **Nested cross-validation.**

---

## 6. Honest priors on the target

The brief's aggressive target is 100%+ CAGR / Sharpe 3+. Existing baselines:
- 12-1 momentum: 15-18% CAGR, Sharpe ≈ 0.6, MaxDD -50%
- AQR-style trend on equities: 15-20%, Sharpe 0.8, MaxDD -30%
- V3 deployed: 39.8% CAGR, Sharpe 0.96, MaxDD -50%
- V6 winner: 38.2% CAGR, Sharpe 0.97, MaxDD -46%
- V7 safer: 29.6% CAGR, Sharpe 1.10, MaxDD -29%

Reaching 100% CAGR Sharpe 3 on PIT S&P 500 with monthly rebal, K≤8,
no leverage, no shorting — would imply a 2-3× CAGR improvement and
3× Sharpe improvement. The brief acknowledges this is exploratory.

**Honest expectation**: a well-built multi-pillar can add 5-15pp CAGR
and 0.2-0.5 Sharpe over V3, with a meaningful MaxDD reduction. That
puts us in the 45-55% CAGR / 1.2-1.5 Sharpe / -25 to -35% MaxDD range
on PIT S&P 500.  Anything beyond that is upside; we report the actual
number after the gauntlet.

If the actual number falls in that range: that's a real Pareto improvement.
If the actual number hits 100% CAGR / Sharpe 3: the leakage red-team is
treated as an obligation, not a celebration.
