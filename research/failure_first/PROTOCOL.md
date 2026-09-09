# Failure-first veto experiment — frozen before new outcomes

2026-09-09. Base main 704bb2bcccc7d560ead84c29c13d10bf83c63322. Research only. This protocol is written before computing this experiment's outcomes. Existing archives have already been studied: none of this historical sample is a virgin research holdout.

## Question
Does decomposing losses improve selective stock-and-horizon prediction over an ordinary direction model with identical inputs? Endpoint must be strictly higher than issue reference. Horizons: 30,60,90,126,180,252,504,756 actual NYSE sessions. No pick quota; at most one stock each fixed 5-session decision date. Do not reissue a stock before its prior deadline. A deadline never moves.

## Important mathematical correction
Predicting 'not higher' with one binary model is only relabeling 'higher', not new information. Also, requiring each of five failure risks below 5% does NOT imply their union is below 5%. Do not multiply survival probabilities assuming independence. The actual hypothesis is that explicitly modeling different exhaustive loss pathways provides incremental information. Test it against a same-feature binary control and include the naive maximum-channel rule as a negative ablation.

## Available data and honest scope
Use the pinned public CRT archives used by SIEVE, with byte hashes checked, historic membership and NYSE calendar. S&P historical membership and Nasdaq matrix are separate runs, not independent universes because they overlap. Mixed/adjusted close and missing terminal payouts remain limitations. The pilot tests PRICE-OBSERVABLE failure types, not actual bankruptcy, dilution or SEC-event causes. Those cannot be identified from these prices. Current data are stale and cannot authorize today's buys. No changes to live production or other repositories.

## Models and labels
Train a direct binary endpoint-failure model and five separate one-versus-rest failure heads. Mutually exclusive and exhaustive precedence: (1) matured missing endpoint, (2) resolved endpoint loss >=20%, (3) remaining endpoint loss with benchmark nonpositive, (4) remaining endpoint loss after a >=10% interim gain, (5) all other nonpositive endpoints. Strictly positive endpoints are success, including recovered interim losses. Unknown interim paths enter the residual failure class, not a made-up causal class. Auxiliary head predicts a >=20% interim drawdown from entry or incomplete path. It is a veto ablation, not the requested endpoint target.

Inputs are past-only SIEVE price states, cross-sectional ranks, market state and additional downside/upside semivolatility, skew, trend efficiency and drawdown-state measures. No post-issue feature or future endpoint-availability filter. Price-ineligible current rows are reported, not replaced by survivors.

Fixed LightGBM binary heads: 80 trees, 7 leaves, depth 3, learning rate .05, min child samples 150, L2=20, max bin 63, seed 20260909; no class balancing, random CV, early stopping or hyperparameter search. Constant smoothed priors for single/rare class training. Fit annually. At each fit, reserve the most recent 252 sessions of fully matured issue dates for probability calibration; train strictly before the FIRST calibration outcome begins (horizon purge). Training window 2520 sessions, fixed every fourth 5-session origin, date-balanced weights with 756-session half-life. Minimum 24 training dates and 26 calibration dates. Fit calibration on out-of-training log-odds only. Training and calibration label end timestamps must predate the relevant boundary. Never re-fit probabilities from the evaluation labels.

## Predeclared variants and thresholds
1. binary: calibrated ordinary endpoint-failure control.
2. failure_sum: calibrated sum of separate failure-head risks (no independence assumption).
3. failure_veto (primary): maximum of calibrated failure_sum and calibrated binary risk.
4. failure_veto_path: primary plus calibrated interim-path risk veto.
5. naive_channels: maximum individual failure risk, deliberately INVALID as union probability; report to demonstrate the aggregation trap.

Evaluate success-score thresholds .70,.80,.85,.90,.95,.975, all reported. Select shortest qualifying horizon per stock, then smallest risk, then shortest horizon, then ticker. Emit only after the cutoff; do not force daily picks. The primary target is failure_veto at >.95, not the best threshold discovered afterward. Also run a strict historical evidence screen using only prior matured issued predictions and non-overlapping intervals. Clearly label conditional statistical assumptions; no automatic production certification.

## Evaluation
S&P evaluation origins 2013+; Nasdaq 2018+ with unsupported early fits recorded as abstentions. Report 2024+ separately but do NOT call it untouched: earlier projects have studied it. Selection happens before joining the current forecast's outcome. Missing matured endpoints fail pessimistically; pending endpoints are excluded and counted separately. Report observed-only precision too, and entry at next session close, plus 20/50bp endpoint-margin sensitivities. No claim of executable trading profitability.

Controls: exact expected uniform-random success rate from contemporaneously eligible stocks at each selected horizon (stronger than one random seed), same-volatility-decile random rate, lowest-volatility stock, ordinary binary, and SIEVE same-window policy results. For causal incremental selection test compare all methods at matched decision dates as well. Uncertainty: paired moving-time-block bootstrap including zero-pick dates and a greedily non-overlapping forecast subset, plus era/unique ticker/missingness/concentration diagnostics. Cluster diagnostics do not prove independence.

## Validation tests
Future-price mutation; future-label mutation; train/calibration/evaluation purge; no future-completeness selection; exhaustive failure taxonomy; zero/negative/flat and missing prices; immature vs unresolved; reference mismatch; union aggregation counterexample; repeated ticker lock; shortest horizon; deterministic ties; no-quota abstention; non-overlap across boundaries; strict JSON; consistent shuffled input order; label-shuffle negative control; planted-signal positive control; identical direct-label complement; repeatability. Any discovered correctness fix must be documented and rerun, not tuned to improve results.

## Deliverables
Code and regression tests, prediction-level outputs, complete threshold table, data coverage, machine-readable run ledger, source/config/input hashes, null and ablation results, reproducible commands, GitHub branch and PR. No 95% claim absent adequate evidence and no promise of background work.