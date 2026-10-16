# Sharpe-3 edge search — frozen protocol

Objective: invent and test a genuinely distinct stock-selection edge whose net out-of-sample portfolio Sharpe can reach >=3 without manufacturing the statistic through leverage, tiny samples, hindsight thresholds, or volatility suppression. A 30 NYSE-session minimum holding/forecast horizon remains binding. Zero exposure is allowed.

## First edge: Information Persistence Disagreement (IPD)

Hypothesis: large stock-specific moves contain two economically different processes that ordinary momentum/reversal models mix together. Temporary liquidity/positioning pressure should mean-revert; persistent information repricing should continue. The edge is to infer which process generated the move *before* observing its future path.

At each decision date decompose a stock's trailing return into market/breadth component and idiosyncratic residual. Characterize the residual shock by: multi-scale sign persistence; acceleration; path efficiency; overnight/intraday proxy where available; volume surprise; volatility response; cross-sectional breadth disagreement; distance from prior trend; drawdown/recovery geometry; and historical recurrence. Construct two independent causal scores:

- TRANSIENT: unusually large residual move + poor path efficiency + volatility/volume burst + broad market disagreement + rapid partial recovery. Expected edge: reversal.
- INFORMATION: residual move + high path efficiency + multi-horizon agreement + breadth/sector confirmation where available + limited immediate reversal. Expected edge: continuation.

A third model estimates *classification ambiguity*. Trade only when transient-vs-information separation is large and historical matured evidence for that state is stable. This is not a claim that the idea is globally unprecedented; novelty means a new project-specific mechanism rather than another parameter sweep of the existing CRT classifiers.

## Portfolio

Long continuation states and long reversal-after-negative-transient states. The symmetric short legs are researched separately and are excluded from the primary portfolio unless borrow data/cost assumptions are defensible. Market beta is estimated from trailing data and optionally hedged with the benchmark; report both hedged and unhedged. Positions are equal-risk capped; no leverage is used to raise Sharpe. Entry sensitivity uses next-session close. Minimum holding 30 sessions; candidate exits 30/60/90/126/180/252 sessions are selected only from matured prior evidence and locked at entry.

## Validation

Use PIT historical membership and dead/stopped-trading names wherever available. Discovery/calibration/test are chronological with full horizon purge. Any adaptive horizon, state threshold, hedge ratio, and abstention threshold must use matured past data only. Evaluate net of conservative turnover costs; report 0/10/25/50bp one-way sensitivity. Report CAGR, annualized arithmetic return, annualized volatility, Sharpe with zero and cash hurdle, Sortino, max drawdown, turnover, exposure, number of positions, unique entry dates, non-overlapping maximum-horizon blocks, annual returns, worst year, and beta.

Required controls: market benchmark; equal-weight eligible universe; existing CRT v3/v6 where comparable; plain residual momentum; plain reversal; volatility-filtered momentum; random same-date names; sign-flipped signal; shuffled training labels/states. Test delayed entry, delistings/missing outcomes pessimistically, and leave-one-era-out behavior.

## Search discipline

Phase A tests IPD exactly as stated. If it fails, preserve the result and invent orthogonal mechanisms rather than tuning until Sharpe=3. Candidate next mechanisms: cross-sectional dispersion compression/expansion, post-event drift conditional on information quality, residual trend convexity, volatility-managed relative momentum, crowding unwind/re-entry, and event-clock state machines using timestamped filings. Every mechanism gets a pre-outcome specification and complete result table.

A historical Sharpe >=3 is only accepted as a research success if it survives an untouched temporal evaluation, costs, next-close entry, multiple eras, adequate observations, and plausible implementation. Otherwise it is recorded as an exploratory lead. No result will be described as achieved until executed evidence supports it.