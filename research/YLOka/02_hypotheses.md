# 02 — Hypotheses for Beating v3 (Price-Only)

**Branch**: `claude/rebuild-stock-selection-YLOka`
**Constraint**: existing daily OHLCV panel only — no fundamentals, no options, no analyst data, no short interest.
**Baseline to beat**: v3 `ml_3plus6` — full-OOS CAGR 39.77%, Sharpe 0.955, MaxDD -49.83%, ~1.45× annual turnover, 4 cash months over 268.
**Objective**: maximize walk-forward OOS CAGR over the research window (2003-09 → 2024-04). Tie-break Sharpe. Constraint MaxDD ≤ -50%.
**Frozen holdout**: 2024-05 → 2026-04 (24 months). Used **once** in Phase 4.

## Strategic frame — where the unsearched edge lives

v3 already captures ~80% of the price-only signal that any single GBM target can extract. The 48-feature set covers momentum, quality, drawdown, breakout, and tail asymmetry. ~600 v4–v7 variants tried scorer-blends, ensembles, regime gates, stops, hedges. Reading those failures, three themes recur:

1. **Single-target ML over-converges**. Predicting next-3m or next-6m return collapses the cross-section to a similar set of "obvious winners". Different *targets* produce different rankings — and ensembling targets that disagree should help when the correlation drops.
2. **The crash gate is too hard / too late**. The current `tight` gate fires on a `21d ≤ -8%` threshold — usually mid-crash. Earlier warning + softer tilts (de-risk to 1 pick or to lower-vol picks before going to zero) is unexplored.
3. **Equal weighting throws away the model's conviction**. The ML score gap between pick #1 and pick #3 is often large. Conviction-weighted sizing was tried in v6 (invvol) but not properly conditioned on score-spread.

The Concretum / Mulvaney trend piece reinforces three meta-lessons that translate from CTA to stock selection:
- Long-lookback breakouts (130d Donchian, not 20d) catch durable regime changes.
- Right-tailed payoff: 35% win rate is fine if winners are 4–10x losers — design for tail capture, not hit rate.
- Discrete on/off signals beat continuous tilts when the underlying truth is regime-switching.

## Hypothesis ranking

Ranked by **expected edge × feasibility** for *this session and the next two*. "Cheap" = uses cached `ml_preds_v2.parquet` + `monthly_returns_clean.parquet`, no GBM retraining (~30s per backtest). "Medium" = needs new feature computation but can reuse cached prices. "Heavy" = requires GBM retraining or new feature pipeline.

| # | Hypothesis | Cost | Expected edge | Risk |
|---|---|---|---|---|
| **H1** | **Multi-target rank ensemble (1m + 3m + 6m + 12m)** with disagreement-aware weighting | Medium | High | Targets correlate; new 12m head needs labels |
| **H2** | **Conviction-spread sizing**: weight ∝ z-score of model output, with cash slice when top score is weak | Cheap | High | Loses some "best of 3" diversification |
| **H3** | **Acceleration overlay on v3 picks**: only buy when 3m pred AND short-term acceleration agree | Cheap | Med-High | Reduces buys; may shrink turnover too much |
| **H4** | **Soft-cash continuum**: scale exposure to risk_off score (0..1), not hard cash/full | Cheap | Med-High | Adds parameter; may be over-fittable |
| **H5** | **Asymmetric exit**: hold winners past 6m if score still top-decile; force exit losers at -15% | Cheap | Med | Stop-loss whipsaw is documented in v7 |
| **H6** | **Donchian-130 breakout filter**: pick from intersection of (top-K ML score) ∩ (price within 5% of 130d high) | Cheap | Med | May reduce candidate pool too much |
| **H7** | **Cross-decile dispersion regime**: when XS dispersion of momentum is wide → concentrate (K=3); when narrow → diversify (K=10) | Cheap | Med | Dispersion is itself noisy |
| **H8** | **Overnight vs intraday return decomposition**: features built from open/close gaps; literature shows S&P winners' overnight return dominates | Medium | Med-High | Open data quality varies pre-2010 |
| **H9** | **Volume-thrust persistence**: 5d up-volume ratio + breadth confirmation as tilt | Medium | Med | Volume data quality varies |
| **H10** | **Sector-neutralized residual momentum**: regress ticker mom on sector mom; pick highest residual | Medium | Med | Sector tags PIT not in current cache |
| **H11** | **Bayesian shrinkage of GBM scores**: shrink scores per-ticker by historical-prediction error; reduces overconfident picks | Medium | Med | Adds complexity without obvious lift |
| **H12** | **Adaptive K via signal strength**: K = 1..5 based on top-score gap to median | Cheap | Med | Already partly tried in v3 (k_recovery) |
| **H13** | **Earnings-week avoidance** (price-derived): identify earnings dates from intraday gap patterns; skip rebalance on earnings month | Medium | Low-Med | Inferring earnings dates from prices is noisy |
| **H14** | **Dual-momentum anti-correlation pair**: top ML pick paired with bottom-vol low-correlation hedge | Medium | Low-Med | Drag in bull regimes |
| **H15** | **Lead-lag from mega-cap on small-cap**: when AAPL/MSFT/NVDA momentum diverges from S&P breadth, tilt to lagging mid-caps | Medium | Med | Regime-dependent, hard to validate |
| **H16** | **Path-quality filter**: require trend_r2_12m > 0.5 AND sharpe_12m > 0 — drop "noisy uptrend" picks | Cheap | Low-Med | Already in feature set; may already be priced |
| **H17** | **Pyramiding on winners**: rebalance on month T+3 to add to picks above water | Cheap | Low | Adds turnover; v3 already 6m hold |

### Top-5 to run in Phase 2 this session/next

1. **H2** — conviction-spread sizing. **Cheapest with highest plausible lift**. Engine modification only.
2. **H4** — soft-cash continuum. **Cheap, attacks the all-or-nothing crash gate** which is a known v3 weakness.
3. **H1** — multi-target ensemble. **Medium cost; foundational** — adds a 12m-target head and an asymmetric (downside) target head; reweights the ensemble by recent IC.
4. **H6** — Donchian-130 breakout filter. **Cheap, novel angle** that v3 does not encode (no breakout-over-N feature).
5. **H3** — acceleration overlay. **Cheap, simple intersection** — easiest A/B test against the H6 filter.

## Detailed cards (ordered by Phase-2 priority)

### H1 — Multi-target rank ensemble with disagreement-aware weighting

**Thesis**: v3 averages 3m and 6m predictions but treats them symmetrically. Different forward horizons isolate different sources of edge: 1m captures mean-reversion + earnings drift, 3m captures momentum, 6m captures trend, 12m captures stage-2 Weinstein. When all 4 horizons agree on a ticker, conviction is real; when they disagree, the signal is noise. Also: an *asymmetric* target — predict the *probability* of being in the top-quintile next 6m vs predict the magnitude — focuses the model on a more learnable quantity.

**Implementation**:
- Reuse `ml_preds_v2.parquet` for 1m / 3m / 6m heads (already trained).
- Train a 12m-rank head (~30 min one-off) and a top-quintile-classifier head (~30 min one-off). Seven-month embargo for both to extend the 6m embargo invariant (use 13m embargo for 12m head).
- Score = `mean(rank(pred_h)) for h in horizons`, weighted by `softmax(IC(h, last 24m))` — agreement-aware.

**Required data**: existing panel + new training run for 12m and classifier targets.

**Predicted failure mode**: targets correlate too highly (3m-rank corr w/ 6m-rank ≈ 0.9 historically) → ensemble = single horizon; minor lift only.

**Success criterion**: walk-forward OOS CAGR ≥ 41% AND Sharpe ≥ 1.0 on research window (vs 39.77 / 0.96 baseline).

### H2 — Conviction-spread sizing

**Thesis**: equal-weighting top-3 ignores that some months produce a "stand-out" pick (top score 2σ above #4) and others produce three near-equal candidates. When there's a stand-out, concentrating into it raises CAGR; when picks are interchangeable, equal-weighting is correct. Also: when even the *best* pick has weak conviction (top-score below historical 25th percentile), the right action is partial cash.

**Implementation**:
- Score normalization: `z = (top_K_score - median_universe_score) / mad_universe_score`.
- Weights: `softmax(λ · z_top_K)` capped at 0.6 per name.
- Cash slice: `cash_weight = max(0, 1 - top_score / hist_25pct)` → smooth de-risk.
- Two parameters: `λ` (tilt strength) and the cash-slice threshold.

**Required data**: `ml_preds_v2.parquet` (existing).

**Predicted failure mode**: concentration risk — one bad pick at high weight tanks the month. v3's equal-weight is partly *because* equal-weighting was the lowest-MaxDD config in v3 search.

**Success criterion**: CAGR ≥ 41% OR (CAGR ≥ 40 AND Sharpe ≥ 1.05), with MaxDD ≤ -52%.

### H3 — Acceleration overlay on v3 picks

**Thesis**: v3's score is steady-state (3m + 6m forward prediction). Layering a *short-term acceleration* check ("is this name speeding up right now?") catches names that are both medium-term predicted-good AND currently in a momentum thrust. The Mulvaney lesson: pyramid on winners; intersect "this is about to be a winner" with "it's already doing it now".

**Implementation**: filter top-K picks by `accel > 0` AND `mom_3 > mom_6_1 / 2`; if filter empties basket, fall back to v3 unfiltered.

**Required data**: existing `accel`, `mom_3`, `mom_6_1` features.

**Predicted failure mode**: too restrictive in chop — basket is empty too often, capital sits idle.

**Success criterion**: CAGR ≥ 40.5% AND turnover ≤ 1.6×.

### H4 — Soft-cash continuum

**Thesis**: the `tight` gate fires on `21d ≤ -8%` (already mid-crash) and goes 100% cash. A continuous risk-off score (0–1) using SPY DD-from-52wh, vol-12m, and breadth proxy can:
- De-risk earlier (50% cash at DD -5%, 100% cash at DD -15%)
- Avoid the binary "all in / all out" whipsaw on shallow corrections
- Capture the mean-reversion bounce by re-entering smoothly

**Implementation**:
```
risk_off = clip( max(
   max(0, -spy_dd_from_52wh - 0.05) / 0.10,   # DD from 52wh: -5%→0, -15%→1
   max(0, spy_vol_12m - 0.20) / 0.15,         # vol: 20%→0, 35%→1
   max(0, -spy_breadth_pct_above_50 + 0.40) / 0.30   # breadth fallout
), 0, 1)
basket_weight = 1 - risk_off
```
Equity portfolio gets `basket_weight`; cash gets `risk_off` (with T-bill yield).

**Required data**: SPY price-derived metrics (existing); breadth requires aggregating across panel (cheap).

**Predicted failure mode**: continuous gate either under- or over-reacts vs the binary one. Three risk-off inputs mean three knobs, easy to overfit.

**Success criterion**: Sharpe ≥ 1.05 with CAGR ≥ 38% (acknowledging the de-risk gives back some upside) AND MaxDD better than v3's -50%.

### H6 — Donchian-130 breakout filter

**Thesis**: from Concretum/Mulvaney, 130d Donchian capture catches durable trends. For long-only stock picking: a name that is *currently* within 5% of its 130d high AND has a top-quartile ML score is more likely to extend the trend than a "deep value" pick at the bottom of its 130d range. This filters out v3's tendency to pick recovering laggards in the late stage of a regime.

**Implementation**: from picks ranked by ML score, require `dist_to_130d_high ≤ 5%`. If filter empties, take the top-K by score.

**Required data**: 130d rolling high — easy to compute from existing prices.

**Predicted failure mode**: blocks deep-value rebound picks that v3 likes (per v6 report, deep value IS a real edge). Net could be flat.

**Success criterion**: CAGR ≥ 40.5% AND fewer monster monthly drawdowns (worst-month return better than v3's worst).

### H5, H7–H17

(Carrying brief cards from the table; will expand to full cards if any reach Phase 2.)

- **H5** asymmetric exit — hold winners past 6m if pred top-decile; force-exit losers at -15% intra-period (daily monitoring).
- **H7** dispersion regime — wide XS momentum dispersion → K=3; narrow → K=10.
- **H8** overnight/intraday split — features from open vs close decomposition.
- **H9** volume-thrust persistence — 5d up-volume ratio confirmation.
- **H10** sector-neutralized residual momentum.
- **H11** Bayesian shrinkage of GBM scores per-ticker by historical error.
- **H12** adaptive K via top-vs-median gap.
- **H13** earnings-week avoidance via gap-pattern inference.
- **H14** dual-momentum anti-correlation pair.
- **H15** mega-cap → mid-cap lead-lag tilt.
- **H16** path-quality filter (`trend_r2_12m > 0.5`).
- **H17** pyramid on winners.

## Multiple-testing discipline

- **Frozen holdout**: 2024-05 → 2026-04 — never queried during research.
- **Research window**: 2003-09 → 2024-04 (~250 months).
- All experiments logged to `backtests/YLOka/experiment_log.csv` with SHA, config hash, target window, and metrics.
- Phase 4 = run the chosen winner ONCE on the holdout. If it underperforms expectations, pick the next-best Phase-2 candidate by research-window CAGR — *no retuning*.

## Engineering setup

- **Test harness**: a thin wrapper that consumes `(asof, ticker, score)` and returns equity curve + metrics. Avoids reimplementing simulator. Use `experiments/monthly_dca/v2/sp500_pit_extended_sweep.simulate_variant`.
- **Score modification**: every hypothesis is a function `score(panel) -> Series` that takes the panel and returns a per-(asof, ticker) score. Existing scorer for v3 = `(pred_3m + pred_6m) / 2`; H2/H3/H6 modify the pick selection downstream.
- **Reproduction harness**: `strategy/YLOka/run_experiment.py` writes a manifest per run to `backtests/YLOka/runs/<ts>_<config-hash>/manifest.json` with metrics + git SHA.
- **Smoke tests**: `tests/YLOka/test_*.py` — PIT membership invariants, feature lag invariants, walk-forward splitter invariants, cost-model accounting. Must pass before any experiment is logged.

## What I will NOT pursue

- **Long-short**: out of scope for the v3 deployed posture; would need borrow modeling and is not the path to higher CAGR (per v6 long-short experiments).
- **Crypto / multi-asset diversification**: separate product on the site.
- **Options overlays**: no options data per user direction.
- **Daily-resolution intra-month rebalancing**: v7 attempted this; whipsaw cost > benefit at the deployed scale.
