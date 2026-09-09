# Cash-information transport extension — frozen before model results

This extends the disclosure-clock hypothesis to a portfolio return objective. Prior price-only results are known; the new SEC facts have been obtained, but this learner has not been fitted or evaluated. Previously used 2021+ market data remain reused history, never relabeled untouched.

## Proposed edge

Credible cash-backed operating improvements can propagate through related businesses at different reporting dates. Compare a company's own last known filing with the recent filings of its contemporaneously correlated peers. A stock that has not repriced while its cash economics and peer information improve is a proposed information-lag opportunity. This is a hypothesis about sequential information absorption; residual returns, machine learning and graph/peer information are established ideas. No global novelty claim.

## Four feature-family tests, one fixed learning architecture

P: price-only residual-return learner (control).
F: add latest as-filed revenue growth, operating margin/change, cash-flow margin/change, disclosure age and cash-flow duration.
D: F plus explicit cash/price disagreement interactions and cash credibility flags.
T: D plus mean peer operating-margin changes, cash-margin changes and revenue growth, transported through a past-only covariance graph. Peers are the eight most positively correlated contemporaneously eligible S&P members with disclosed facts, excluding self. Graph is estimated on preceding 252 observed daily returns and refreshed every21 sessions. Newly disclosed peer facts update only after the filing-date availability lag. None of these features is a causal guarantee or future knowledge.

Weekly decision grid (five sessions). Keep facts for at most180 calendar days from filing. Require252 valid prior returns, known historical membership, known original disclosure availability and observed current quote. Record the current-resolver/survivorship mapping gaps; do not claim a full historical issuer master.

For each horizon30/60: forecast the next-close-entry, fixed-horizon stock return minus past-estimated beta times the matching benchmark return, scaled by prior stock volatility and gross hedge notional. Targets may be fitted only after their endpoint is strictly before the annual fit date. Missing historical target prices are withheld from supervised loss, recorded as missing, and NEVER used to filter test candidates. Actual held missing marks retain the conservative stress liquidation in the ledger. Model: deterministic LightGBM Huber regressor, 100 trees, seven leaves, depth3, learning rate.04, minimum leaf200, L2=25, seed20260911. Train on prior2,520 sessions with each issue date equal total weight and a756-session half-life. No early stopping or outcome-driven hyperparameter search.

Two fixed absolute normalized-score cutoffs .05/.15. Two book constructions: (1) long-only at most20 positive-score stocks, equal risk by prior63-session volatility, at least5 names otherwise cash; (2) long/short at most20 each side with scores above/below the cutoff, at least5 each, half gross per side, then an explicit SPY hedge offsets the estimated beta and the whole portfolio is scaled to gross<=1. No unverifiable abstract factor return is booked as a cash asset. All share-unit quantities are set at the signal close, execute the next close, and expire30/60 sessions after entry. Same5bp/3%-borrow/French-RF account. This yields32 configurations, all reported.

Before opening test strategy results, choose the configuration by validation2016–2020 net excess Sharpe, requiring positive validation Sharpe, no missing-held-price liquidations and max gross<=2. Then start a fresh test account in2021; no carried hindsight-selected positions. Show all test rows but do not switch the winner using them. Annually refit causally during both validation and test. Full daily marked P&L, cash, borrow, financing, netted turnover and costs determine Sharpe; not model IC or winning-trade-only returns.

## Checks

Future-fact and future-price mutation must leave prior states/predictions unchanged. Every fit records maximum label-maturity date. Compare P versus F/D/T on identical data support, report all data/identity exclusions, fixed training-label-permutation control in the recent segment, sign-reversal and random-matched schedule controls for the pretest selection, cost and execution-delay sensitivities, era metrics and63-session block uncertainty. If no validation candidate qualifies, say so and do not promote the highest test row. Actual field timestamps and observed raw facts differ from proof of original API realtime availability; that limitation stays explicit.

The aim remains a real economic edge and net Sharpe>=3, not a convenient accounting identity or a retuned historical headline. No live orders or production edits.
