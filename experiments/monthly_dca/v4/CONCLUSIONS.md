# v4 Strategy Search: Conclusions

**Date:** 2026-05-10
**TL;DR:** Extensive search confirms v3 is near-optimal on PIT S&P 500.
**No production change recommended.**

## Final summary table

| Configuration                                   | Full CAGR | WF mean OOS | WF min  | Beats SPY | Sharpe | MaxDD |
|-------------------------------------------------|----------:|------------:|--------:|----------:|-------:|------:|
| **v3 baseline (deployed): ml_3plus6 K3 h6 EW**  | **39.77%** | **42.80%**  | 14.49%  | 9/10      | 0.96   | -49.8%|
| v3 + invvol cap=0.40 (best honest variant)      |    38.66% |    42.84%   | **22.00%**| 9/10    | **0.98** | -48.4%|
| K=2 h=7 (overfit spike, not recommended)        |    45.48% |    52.21%   |  6.70%  | 9/10      | 0.94   | -81.2%|
| Fresh v4 LightGBM ML (poor)                     |    10.18% |    10.02%   | -4.42%  | 3/10      | 0.45   | -61.4%|
| TP=+50% (honest semantics)                      |    31.61% |    34.49%   |  7.82%  | 8/10      | 0.94   | -43.6%|
| SL=-30% (honest semantics)                      |    31.66% |    34.80%   | 16.78%  | 10/10     | 0.98   | -52.3%|

## Key findings

1. **The v2 GBM model is well-tuned.**  Adding hand-picked factors (quality,
   momentum, idio-mom, breakout) all hurt performance. The ML score already
   captures these signals.
2. **K=3 EW h=6 is the robust optimum.** K=1, 2, 4, 5 all underperform.
   h≠6 (3, 7, 9, 12) all underperform except as overfit spikes.
3. **Take-profit and stop-loss don't help when implemented honestly.** The
   apparent +21pp lift from TP=+50% in initial tests was a simulator bug
   (post-TP returns were double-counted).  Corrected semantics: TP and SL
   both reduce CAGR.
4. **A fresh LightGBM did not beat the existing v2 GBM.** The published v2
   (HistGradientBoostingRegressor on 1m/3m/6m horizons, full-history training)
   appears to be the strongest configuration on this data.
5. **Inverse-volatility weighting + cap=0.4 is the only honest improvement.**
   Trades 1pp full CAGR for 7.5pp lift in WF min. This is a robustness
   improvement, not a CAGR improvement.

## What would unlock higher CAGR

PIT S&P 500 has limited cross-sectional dispersion. To honestly target higher
CAGR:

1. **Expand universe.** The same v3 strategy on the broader 1,833-ticker
   universe delivers 51.8% WF mean OOS CAGR. On non-S&P 500 PIT it delivers
   51.0%. On random 500 subsets, average 56.4%.  But the user has specified
   PIT S&P 500 as the primary universe.

2. **New asset classes.** Stock picking on equities is a mature game.
   Crypto, futures, and options have higher dispersion and may deliver
   higher CAGR.

3. **Leverage.** 2× leverage on v3 would give ~85% WF mean CAGR but with
   ~100% MaxDD risk.

4. **New information sources.** Fundamentals, news, alternative data not in
   the current 67-feature set.

## Production decision

**Keep v3 deployed unchanged.** The 42.80% WF mean OOS CAGR with 9/10
splits beating SPY is the best honest answer for PIT S&P 500.

The v4 search confirms the deployed strategy is near-optimal. Any future
iteration should focus on:

- Universe expansion (with explicit user opt-in)
- New information sources beyond price
- New asset classes
