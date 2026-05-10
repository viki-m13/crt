# Exp 02 — TLT crash fallback + add-on stack + staggered ensemble

Date: 2026-05-10. Driver: `experiments/monthly_dca/v8/run_tier{2,3}_*.py`.
Builds on exp_01 winner (`ml_3plus6plus1`, k=1, hold=1, safer regime, invvol).

## Hypothesis

The exp_01 winner sits in cash during 4 crash months, earning 0%. Two
substitutions are obvious: (a) earn T-bill yield in cash; (b) allocate
to a crash-anti-correlated asset (TLT, long Treasuries). Long Treasuries
historically rallied during equity crashes (2008, 2020 — though 2022
broke the pattern).

## Method

47 single-knob and stacked variants on top of exp_01 winner:
- crash fallback ∈ {cash, spy, tlt}
- additive: drawdown_de_risk, smart_reentry, min_pick_mom, vol_target,
  trailing_stop, vol_penalty, sticky_cash, half_cash_warning,
  cash_yield, spy_dd_scale, quality_blend, regime_gate variants,
  hold_horizon variants, k variants
- 6 staggered ensembles (n_legs ∈ {2,3,4,6}, hold ∈ {2,3,4,6})

All other dimensions held to exp_01 winner.

## Result

**`22_fallback_tlt` is the new champion** (Pareto-improves exp_01
winner across the board):

| Metric              | v3 baseline | exp_01 best | **exp_02 best (TLT)** | Δ vs v3   |
|---------------------|------------:|------------:|----------------------:|----------:|
| Full CAGR           | 39.77%      | 38.64%      | **40.27%**            | +0.50pp   |
| Sharpe              | 0.955       | 1.058       | **1.084**             | +0.129    |
| MaxDD               | -49.83%     | -45.01%     | **-44.49%**           | +5.34pp   |
| **WF mean CAGR**    | **42.80%**  | **48.28%**  | **50.16%**            | **+7.36pp** |
| WF min CAGR         | 14.49%      | 9.54%       | **17.38%**            | +2.89pp   |
| WF mean Sharpe      | 1.031       | 1.058       | **1.084**             | +0.053    |
| **WF n beats SPY**  | **9/10**    | **10/10**   | **10/10**             | **+1**    |

**Strict Pareto improvement** on every metric vs v3 baseline.

## What worked

1. **`crash_fallback="tlt"`**: replaces 4 cash months with 100% TLT
   allocation. Adds defensive positive return during stress.
   - +1.88pp WF mean (48.28 → 50.16)
   - +7.84pp WF min (9.54 → 17.38)
2. **Trailing stop 25%** has no effect (never triggers) but doesn't hurt.

## What didn't move the needle

- `cash_yield_yr=3%`, `smart_reentry`, `sticky_cash`: identical to TLT
  alone because the TLT branch already covers the cash period.
- `dderisk` / `vol_target` / `vol_penalty`: cap upside more than they
  reduce downside; net negative.
- `min_pick_mom`: shaves WF mean. The GBM already prefers names with
  decent momentum; the floor filter cuts winners that have a recent
  short-term dip.
- Hold horizon h ∈ {2, 3, 6}: all materially worse than h=1.
- k ∈ {2, 3} with conviction / softmax weighting: barely lower mean
  but materially lower sharpe and breadth.
- Tight / combo regimes with TLT fallback: high mean (~49%) but MaxDD
  blows out to -90% because crash fires less often → more time
  unhedged in equity.

## Staggered ensemble verdict

n_legs ∈ {2,3,4,6}, hold_per_leg ∈ {2,3,4,6}: averaging legs averages
WF mean down. n=2/h=2 gets 37.6% (vs 50.2% solo); none Pareto-improve.
Concentration > diversification at this granularity.

## Verdict

**KEEP `22_fallback_tlt` as the leader.** Logged in
`backtests/experiment_log.csv`. Output CSVs in
`experiments/monthly_dca/v8/results/tier{2,3}*.csv`.

## Hitting the ceiling on monthly + existing preds

We've gone from v3's 42.8% → 50.2% on WF mean OOS CAGR using ONLY
choices on top of the existing GBM preds (concentration, horizon,
scorer mix, regime, fallback). Three more knobs that we can't unlock
without code changes:

1. **Train GBM with shorter target / more features** — likely +3-10pp
   WF mean.
2. **Weekly walk-forward** — the big scope-allowed change. Requires
   building weekly features, weekly fwd returns, weekly retrain with
   proper purged k-fold + embargo. Highest EV, highest cost.
3. **New features**: explicit "pre-explosion" detectors (vol thrust,
   breakout follow-through, gap drift). Likely +1-3pp.

Triple-digit (100%+) WF mean OOS CAGR on PIT S&P 500 monthly is, based
on this empirical ceiling, unlikely without the weekly track.
