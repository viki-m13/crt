# Downside state recurrence — frozen experiment, 2026-09-09

User mandate: continue inventing and testing selective stock endpoint forecasts. This experiment tests the bearish tail: the archived endpoint must be strictly below the issue reference after >=30 NYSE sessions. No guaranteed accuracy; no short-sale order or production edit. Code is namespaced separately from the concurrently discovered `research/downside` experiment, pinned at f507f5eb241a2bc3a1134819cfd32503d59d0098. Its new results have not been read when registering this extension.

## New mechanism
Cross-era state recurrence: a stock becomes a bearish candidate only after contemporaneous relative weakness or a failed recovery. Compare its current normalized state with earlier PIT stock states rather than visual chart shapes. Estimate the frequency of VERIFIED endpoint declines among nearest past states, balancing contributions by issue date; then require support from distinct historical eras and agreement with a direct endpoint classifier. This is a project-specific experiment, not a claim of worldwide novelty.

## Frozen candidate set
Use the existing failure_first pinned archive/calendar/data adapter; no full-history completeness or future-price eligibility. Five-session issue grid. At each date union: five weakest relative-63-session names; five lowest trend-acceleration names; five weakest relative-63 names among failed-recovery states. Failed recovery means price below 200-session geometric mean, relative-63 return negative, positive 21-session return, and negative five-session return. Candidates are chosen before labels, maximum 15 per date, stable ticker tie-breaking. Report full-universe AND candidate-set matched controls; candidate selection is part of every tested method.

## State and history
Distance vector fixed: r5/r21/r63/r126/r252, vol63, vol_ratio, dd, ma50, ma200, rel21_rank, rel63_rank, rel126_rank, semivol_ratio, trend_accel, breadth, market21, market63, market200, market_vol63. Normalize using median and interquartile scale of prior reference states only; floor scale 1e-6; clip normalized coordinates at +/-8. No future normalization. Reference states: trailing 2,520 sessions with every fourth five-session origin (20-session grid), and feature date < annual fit date minus 30. Nearest-neighbor tree uses no outcomes. Retrieve at most 512 closest references; each horizon filters these to exit_i < fit_i before using labels. Use first 128 admissible neighbors; less than 32 rows or 12 distinct dates means unsupported, no estimated score. The same candidate search is used by every recurrence variant. Small-horizon history is not replaced by hindsight-selected long-horizon history.

Decline label = matured AND observed positive endpoint price AND return < 0. Flat and unresolved are not confirmed declines. Pending labels never enter fitting. Preserve unknown counts; primary denominator treats unresolved matured selections as failed verified-down predictions. This is a conservative observed-outcome target, not missing-price imputation.

## Fixed variants
1. direct: ordinary binary LightGBM, same candidate universe; reuse audited annual full-horizon-purged train/calibration convention and hyperparameters from failure_first, with target strictly negative observed return.
2. recurrence: 128 nearest matured reference states; inverse within-date count balances contributions; 756-session half-life across dates. Add two pseudo-dates at 50% probability. Weights normalized to effective date count before smoothing.
3. regime_recurrence: same estimator after restricting neighbors to the candidate's current bull/stress regime, then support checks.
4. cross_era: minimum of regime_recurrence and estimates on reference origin blocks before/after fit_i-1260; both eras must have >=32 rows and >=12 dates.
5. consensus: minimum of direct and regime_recurrence scores. Agreement is not independence.
6. confirmed_recovery (primary): consensus restricted to the frozen failed-recovery state.
7. robust_consensus: minimum of direct and cross_era.

Raw recurrence and minimum-combination outputs are estimates/scores, NOT certified probabilities. Direct scores use separate past-data Platt calibration. No new calibration fitted on current test outcomes. This experiment tests whether local support improves rare-tail precision; it cannot claim a formal conformal guarantee.

## Policy and evaluation
Eight horizons 30/60/90/126/180/252/504/756. Seven fixed score cutoffs .60/.70/.80/.85/.90/.95/.975. At each issue date choose each stock's shortest supported horizon exceeding the cutoff, then highest score with ticker tie-break. Max one issuance per policy/date; no reissue through the prior locked deadline. Report all 49 adaptive-horizon policies, not the best as confirmatory evidence. Secondary adaptive95 chooses a method/cutoff only using completed past virtual-policy predictions, nonoverlap diagnostics and multiplicity/repeated-look adjusted exact-binomial lower-bound screen; it can abstain entirely and is not production certification.

Evaluate S&P origins 2013+ and Nasdaq 2018+ using all available data. Endpoints after input cutoff pending. Previously reused historical data are NOT virgin research holdouts. Display 2024+ separately. Include strict lower-than-issue, next-close-reference, -20bp/-50bp/-5%/-10% margin sensitivities, interim upside/path incompleteness, number of distinct dates/tickers/years, connected overlap episodes and greedy nonoverlap count. Compute expected same-date/horizon full-universe and candidate controls and volatility-matched controls. Date-block paired uncertainty is descriptive, not a universal guarantee under arbitrary dependence.

## Falsification and reproducibility
Regression tests for future-price feature mutation, future-label mutation, exact endpoint vs touch, flat/unknown/pending labels, horizon lock, no forced picks, minimum recurrence support, per-date neighbor weighting, era support and candidate selection invariance. Repeat selected real annual folds exactly. Fixed label-permutation control (joint within training/reference dates to preserve market state) on 2024+ in both universes; it is a stock-selection null, not destruction of market predictability. Keep all runs and config/source/input hashes. A clean GitHub rerun is same-data reproduction, not new market evidence.

## Limitations
No new SEC, dilution or actual bankruptcy events have been assembled; no claim to have tested that event-based route. Archived adjusted/mixed prices and missing constituents/terminal payouts still block literal unadjusted-price certification. Neither a high decline hit rate nor avoiding a large interim rally proves a profitable executable short: borrow/recall/dividend/financing/fill data are absent. No current stock recommendations from stale March/May 2026 inputs. Existing main/production and other repos are unchanged.
