# Exp 03 — TRUE weekly walk-forward GBM (KILLED)

Date: 2026-05-10. Branch: `claude/rebuild-stock-selection-2qHxY`.
Driver: `experiments/monthly_dca/v8/weekly/{build,fit,run}_*.py`.

## Hypothesis (from `02_hypotheses.md` H9)

A genuine weekly walk-forward — weekly features, weekly fwd targets,
weekly retrain — would unlock another 20-50pp of WF mean OOS CAGR by
firing on signal flips that the monthly cadence misses.

## Method

Built end-to-end weekly pipeline:

1. **Weekly feature panel** (`build_weekly_features.py`): 22 PIT-clean
   features across 1583 weekly Friday asofs × 1824 tickers (1.9M rows).
   Vectorised across tickers from `prices_extended.parquet`.
2. **Weekly PIT membership** (`build_weekly_membership_and_spy.py`):
   weekly view of monthly PIT membership; SPY weekly features for the
   regime gate. 607k rows.
3. **Walk-forward GBM** (`fit_weekly_gbm.py`): HistGBM on cross-
   sectional rank target (4-week forward return). **Embargo 6 weeks**,
   retrain every 13 weeks (quarterly). Train cutoff at retrain time T:
   `asof < T - 6 weeks`. 84 retrains, 409k OOS preds across 1085 weeks.
4. **Weekly simulator** (`lib_weekly.py`): mirrors v6 monthly engine.
   Friday-close to Friday-close fills; cost = `cost_bps × n_changed`
   round-trips on every rebalance; honest NaN handling
   (treat NaN-return as 0%, not -100%, with 2-week ffill on prices).
5. **Variant sweep** (`run_weekly_baseline.py`): 13 configs across
   k ∈ {1,2,3}, hold ∈ {1,2,4} weeks, regime ∈ {safer, tight, strict_dd},
   crash fallback ∈ {cash, tlt}, half-cash on warning.

## Result

**0 of 13 weekly variants pass the floors.** Best WF mean OOS CAGR is
17.1% (k=3, h=4, safer, TLT) — half of v3 monthly's 42.8% baseline.

| variant                       | WF mean | WF min  | Sharpe | MaxDD   | beats SPY |
|-------------------------------|--------:|--------:|-------:|--------:|----------:|
| 08_k3_h4_safer_tlt            | 17.12%  | -2.23%  | 0.61   | -60.2%  | 5/10      |
| 06_k2_h2_safer_tlt            | 13.62%  | -9.97%  | 0.43   | -68.9%  | 5/10      |
| 04_k1_h4_safer_tlt            | 12.79%  | -34.57% | 0.24   | -86.9%  | 4/10      |
| 03_k1_h2_safer_tlt            | 12.42%  | -15.83% | 0.38   | -82.1%  | 4/10      |
| 09_k1_h1_tight_tlt            |  7.55%  | -23.96% | 0.45   | -97.3%  | 3/10      |
| 01_k1_h1_safer_tlt            | -4.59%  | -24.53% | 0.18   | -95.7%  | 2/10      |
| **monthly k=1 TLT (exp_02)**  | **50.16%** | **17.38%** | **1.08** | **-44.5%** | **10/10** |

## Why it fails

1. **Cost drag at k=1 weekly**: 10 bps round-trip × ~52 rebalances/yr ×
   100% turnover ≈ 10pp/yr drag. Even at 5 bps the geometry is harsh.
2. **Horizon mismatch**: 4-week target ranks predict 4-week forward
   returns; using them to pick on 1-week holds means we're betting on
   the noisy 1/4 slice. The signal-to-noise crashes.
3. **Single-pick weekly variance**: at k=1 weekly, a single bad week on
   one ticker is a 5-15% portfolio loss. The annual GBM-monthly
   equivalent has ~12 such "bets per year"; the weekly equivalent has
   ~52, and cross-sectional rank predictions don't distinguish well
   between adjacent weeks.
4. **Full-window CAGRs are misleading**: `09_k1_h1_tight_tlt` hits
   cagr_full = 101.85% but its WF min is -23.96% and 7/10 splits are
   below the floor. That single-window number is path-dependent on
   2008-09 + 2020-21 windfalls; per-split it falls apart.

## Verdict

**KILL** the weekly track for this scope. Records preserved for
reproducibility:

- `experiments/monthly_dca/v8/weekly/cache/features_weekly.parquet` (1.9M)
- `experiments/monthly_dca/v8/weekly/cache/sp500_membership_weekly.parquet`
- `experiments/monthly_dca/v8/weekly/cache/spy_features_weekly.parquet`
- `experiments/monthly_dca/v8/weekly/cache/weekly_preds.parquet` (409k OOS preds)
- `experiments/monthly_dca/v8/weekly/results/weekly_baseline.csv`

## What might fix it (not pursued in this run)

- **Horizon-matched targets**: 1w target → 1w hold; 2w target → 2w hold.
  The `build_targets_alt.py` script is the start; never finished due to
  scope cut.
- **Lower cost model**: 5 bps each side / 5 bps round-trip would cut
  drag in half. Honest only if execution actually achieves that
  (institutional VWAP fills).
- **Smart-turnover weekly**: only sell tickers that drop out of top-K
  beyond a rank-buffer. Probably +5-10pp.
- **k≥5 weekly with vol target**: dilute single-pick variance.
- **Different feature set for weekly**: the 22 features are derived
  from a feature library tuned for monthly horizons. Short-horizon
  signals (5d-20d momentum + reversal, microstructure) likely better.

The monthly k=1 + TLT-fallback winner already Pareto-improves v3
(WF mean 42.80% → 50.16%, 9/10 → 10/10 beats SPY); unlocking weekly
would require a separate research effort.
