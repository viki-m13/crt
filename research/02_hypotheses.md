# Hypotheses — pushing OOS WF CAGR above v3's 42.8%

Date: 2026-05-10. Scope locked by user: **higher concentration (k≤3)** and
**monthly or weekly frequency** on PIT S&P 500. No leverage, no down-cap.

Engine: `experiments/monthly_dca/v6/lib_engine.py` (parity with v3).
Floors that any winner must clear:
- WF min CAGR ≥ 0%
- WF mean Sharpe ≥ 1.0
- MaxDD ≥ -50% (full-window)
- ≥ 8/10 WF splits beat SPY

Baseline (v3 deployed, reproduced numerically):
- CAGR full 39.77%, Sharpe 0.96, MaxDD -49.83%, WF mean 42.80%, WF min 14.49%, 10/10 pos, 9/10 beat SPY.

## Tier 1 — uses existing ML preds, fast to evaluate

### H1. Concentration sweep k ∈ {1, 2, 3}
- **Thesis**: top ML-rank picks contain the most signal; lower k drops
  noisy lower-ranked names and concentrates on the best.
- **Data**: existing `ml_preds_v2.parquet`.
- **Predicted failure**: k=1 mean CAGR up but Sharpe and min split
  collapse; one bad pick destroys a year. Probably k=2 is the
  sweet-spot for "honest CAGR with floors".
- **Verdict criterion**: WF mean ↑ AND WF min ≥ 0% AND ≥8/10 beat SPY.

### H2. Hold horizon × k. {1m, 2m, 3m, 6m} × {1, 2, 3}
- **Thesis**: shorter holds rotate out of fading winners faster, capture
  more of the explosive-mover early phase.
- **Data**: existing.
- **Predicted failure**: shorter hold → higher turnover → cost drag.
  10 bps round-trip × 12 vs ×2 holds matters.
- **Verdict criterion**: WF mean ↑ net of cost.

### H3. Scorer sweep at k=1: ml_h3 / ml_h6 / ml_3plus6 / ml_3plus6plus1
- **Thesis**: 3m predictor is more reactive; 6m is more stable. The mean
  smooths but may dilute the sharper signal.
- **Predicted failure**: ml_h3 alone is noisier and cost-drag heavier.
- **Verdict**: best on WF mean subject to floors.

### H4. Regime gate at k=1: tight / strict_dd / safer / combo / faber
- **Thesis**: at k=1 a single bad-month exposure during a crash is
  catastrophic; an earlier-firing gate may keep WF min ≥ 0%.
- **Predicted failure**: earlier gate misses recoveries (R5_COVID's
  +62% comes from staying in 2020).
- **Verdict**: at k=1, best gate × WF min trade-off.

### H5. Conviction-weighted concentration (variable k)
- **Thesis**: when top-1 score is far above top-2 (large gap), bet 1
  pick; when scores cluster, diversify across 3.
- **Predicted failure**: rank gap is noisy on a small N=500 universe.
- **Verdict**: WF mean vs flat-k controls.

### H6. Staggered ensemble (sub-baskets)
- **Thesis**: two parallel sub-baskets at k=1, each rebalancing every
  6m but offset by 3m, doubles the effective rebalance cadence without
  changing per-pick variance. (Same for k=1 with 5 sub-baskets offset
  monthly → equivalent to k=5 monthly with 5m hold but with crash-gate
  applied uniformly.)
- **Data**: existing preds, just re-aggregate.
- **Predicted failure**: ensemble averages out exactly to a longer-k
  monthly. Has to be tested empirically.

### H7. Half-cash on 'warning' regime
- **Thesis**: between full-bull and crash there's a 'warning' state
  (SPY dsma200 < 0 OR 6m mom < 0) where exposure should be reduced
  but not zero. v6 has this knob (`half_cash_warning`).
- **Predicted failure**: 'warning' triggers too often, dilutes returns.
- **Verdict**: WF mean CAGR ↑ AND WF min ↑.

## Tier 2 — requires new code; high EV if ML retrain stays honest

### H8. Weekly rebalance with monthly preds (cheap weekly variant)
- **Thesis**: weekly rebalance to follow the monthly-updated top-K with
  smart turnover. Rebalance-only-on-change. Should shave drawdowns by
  exiting fading names sooner.
- **Predicted failure**: monthly preds don't change between asof
  ticks, so weekly is just monthly in disguise + extra cost.
- **Decision**: skip if monthly preds aren't refreshed weekly.

### H9. **TRUE weekly walk-forward** — weekly features, weekly preds, 1–4 week hold
- **Thesis**: the deepest scope-allowed change. Build weekly feature
  panel from `prices_extended.parquet`, retrain GBM monthly with weekly
  test slices, test 1w/2w/4w hold periods at k=1, k=2.
- **Data**: build new — weekly close panel, weekly fwd returns,
  weekly cross-sectional ranks. Reuse PIT membership at week-end.
- **Cost model**: must upgrade. 10 bps round-trip is fine for monthly
  S&P 500; weekly compounds to ~52× per year. Use 5 bps each side
  (10 bps round-trip) and confirm that survives.
- **Risks (high)**:
  - Embargo: weekly target horizon ∈ {5d, 10d, 20d}. Embargo must be
    > target horizon. With 4w forward target need ≥ 5w embargo.
  - Overfitting: 100s more training rows but each row is more
    autocorrelated. Use purged k-fold (de Prado) per `purged_walkforward.py`.
  - Cost model is now first-order; need ADV-scaled slippage.
- **Predicted failure**: noise dominates; cost eats the alpha; fit
  degrades.
- **Verdict**: WF mean CAGR ↑ vs monthly k=1 net of upgraded costs.

### H10. Multi-target ML — fwd return + "explosive" classifier
- **Thesis**: bipartite signal. (a) regressor predicts continuous
  return, (b) classifier predicts P(top-decile 6m return). Final
  score = α·z(reg) + (1-α)·z(cls). Filters out names with high
  predicted return that come from "low-vol drift" rather than
  "explosive run".
- **Data**: train both heads in walk-forward. Need
  `experiments/monthly_dca/cache/v2/ml_preds_v2.parquet` to grow with a
  classifier head. Re-fit walk-forward.
- **Predicted failure**: classifier and regressor are highly
  correlated, no marginal info. Or the rank correlation between
  P(top-decile) and actual return is too weak.

### H11. Trend × volume thrust feature
- **Thesis**: explosive movers (NVDA-style) are characterised by
  multi-week up-volume above a baseline. Add an `obv_mom_3m`,
  `vol_thrust_20d` feature to the existing 67. Re-fit walk-forward.
- **Data**: volume column in `prices_extended.parquet` (already there).
- **Predicted failure**: existing momentum features already absorb most
  of this signal — incremental IC is small.

### H12. Earnings-drift proxy (price-only, no fundamentals)
- **Thesis**: large gap-up days within last 2 months act as proxy for
  positive earnings surprise. Stocks just past such a gap have post-
  earnings drift edge over weeks. v3 already has `earnings_drift_proxy`
  — extend it (3m, 6m windows; gap magnitude buckets).

## Tier 3 — speculative

### H13. Shorter-target ML (1m forward) retrained semi-annually
- **Thesis**: faster signal. Predicted failure: severe label noise at
  1m, GBM overfits. Likely killed quickly.

### H14. Regime-conditional concentration
- **Thesis**: bull→k=1, normal→k=2, recovery→k=3, crash→cash. Uses
  regime gate's existing classification.

## Plan

1. Run Tier 1 sweep first (cheap; reuses all infrastructure). Identify
   the 1–3 most promising configurations.
2. If Tier 1 hits the floor with mean ≥ 60–70% WF CAGR, proceed to
   final validation. Triple-digit on monthly + S&P 500 is a stretch.
3. Build H9 (weekly true walk-forward) only if Tier 1 results suggest
   compounding gains. This is a 1–2 day build with proper purged
   k-fold + embargo.
4. Tier 2 H10/H11/H12 are incremental feature engineering — moderate
   priority, attempt if Tier 1 gives signal that ML head benefits from
   more features.

## Tracking

`backtests/experiment_log.csv` — every run with config, metrics, git
SHA. `research/exp_NN_<name>.md` — narrative per experiment. Failed
runs go to `research/graveyard/`.
