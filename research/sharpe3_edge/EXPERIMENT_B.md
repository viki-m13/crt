# Experiment B: continued invention, registered before B returns

Experiment A's Nasdaq ledgers are now observed; its exact rules and results are retained. Adaptive IPD produced Sharpe 0.282, fixed60 0.761, and information-only equals fixed60: ranking by unscaled strength starves the rare transient state. These are NOT Sharpe3 results. S&P A is still being executed. This extension is openly post-A research; no assertion of a virgin holdout.

Two new hypotheses: (1) asymmetric market-response capacity identifies names absorbing selling pressure rather than merely low-volatility names; (2) improving breadth and residual-rank transitions identify dissemination of information before simple trailing momentum. This adds operational tests, not a proven or globally unprecedented edge.

## New frozen methods

- ipd_balanced60: alternate five-session signal dates reserved for information-only or transient-only; prevents score-scale dominance. No substitution if that state has no candidates.
- absorption60: market21<0, residual21>0, drawdown>-30%, and top quartile of trailing upside-response minus downside-response beta. Rank by that response gap plus residual21 / residual vol. Response betas are rolling126 covariance against clipped positive/negative market returns, respectively.
- breadth_release30: current eligible-stock breadth above its value at the latest scheduled origin at least21 sessions back by >.10, market21>-.05, and market below200-session geometric mean. Select stocks with negative63-session return and positive5-session residual, ranked by their residual5/residual-vol minus residual21 z-score.
- rotation60: breadth>.45 and rising, positive residual21, improvement of within-date residual21 percentile over residual63 percentile >.25, volatility percentile<.60. Rank by that percentile improvement plus path-efficiency percentile.
- convexity60: positive residual63, residual21>residual63/3, residual5>0 but <residual21/2, path-efficiency percentile>.50, residual-vol21/63<1. Rank by volatility-normalized residual acceleration.
- barbell60: alternate absorption60 and convexity60 opportunities; no fallback.

Capital-clock comparison: the original fixed assigned sleeve can remain cash until its next 30-session slot despite an expired holding. Work-conserving scheduling instead chooses the first available sleeve in circular order at each five-session decision date. NO sale is accelerated, no holding is rebalanced, each lot still holds >=30 sessions, and no borrowing funds stock purchases. This is an implementation comparison, NOT claimed new information. All B mechanisms and ipd_adaptive/ipd_fixed60/momentum60/equal60 controls receive this same scheduler.

Run base 25bp one-way, plus 0 and50bp; delayed entry5; and the existing explicit half-capital/1%-assumed-borrow SPY-hedge diagnostic. Keep every result. B uses the identical incomplete archived universes and evaluation periods, with no feature/threshold revisions after B returns. No state probability or causal-economic attribution is inferred merely from these formulas. The state opportunity code, not a claim of novelty, defines the experiment.

Additional validation: new features cannot change under future stock/benchmark/membership mutations; future holdings may not alter past orders; work-conserving mode must reconcile cash and lot P&L and preserve duration; original A output must reproduce with the new optional scheduler disabled. The all-member benchmark is allowed to repeat names across sleeves (necessary to represent the market), unlike stock-selection strategies. Inverted60 in A is an inverted-rank/noncandidate long control, NOT a synthetic negative-return short portfolio.
