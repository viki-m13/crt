# Direct-downside experiment — frozen protocol, 2026-09-09

This is a new executed-work plan registered BEFORE computing this experiment's market outcomes. It does not assert successful results. Base code/input lineage: failure-first branch at b95b0b1259f3dff1840d51c66a3b35f059178e3c. Only research files may change; no production edits, live trades or other-repository writes.

## Exact target and denominator
Predict a strict decline in an individual stock's observed archived endpoint price relative to its issue-time reference. Issued horizon is immutable and >=30 NYSE sessions. Flat is not down; missing is not down; a recovered intraperiod crash is not endpoint success. Matured unobserved endpoints are reported separately and count as failed verified-down predictions in the primary pessimistic metric. Also report resolved-only and best/worst missing-outcome bounds. Pending outcomes are not successes. This is a verified-price-change research estimand, not evidence about literal unadjusted prices while the supplied adjusted/mixed data remain uncertified.

## Eight fixed mechanisms (same eligible universe and chronology)
1. direct: supervised binary verified endpoint-decline control.
2. competing: jointly normalized mutually exclusive outcomes: observed nondecline; >=20% terminal decline; other decline with a falling benchmark; remaining decline; unresolved endpoint. Sum only the decline classes, then calibrate on separate past data. This avoids interpreting missing prices as successful bearish predictions.
3. recent: direct classifier trained only on the latest 1,260-session subset of the ordinary 2,520-session training history.
4. rebound: direct score, vetoed unless price is below its 200-session geometric moving mean, relative 63-session momentum is negative, 21-session return is positive, and its recent 63-session high is below its preceding 63-session high. Tests deterioration plus failed recovery, not a free threshold sweep.
5. persistence: separately learn the stricter event that price is below reference at one-half, three-quarters AND the full horizon; calibrate that event on separate past data. It is a subset target, not a theorem that estimated probability lower-bounds true endpoint probability.
6. residual_cdf: learn conditional median volatility-scaled log return on observed outcomes; evaluate the weighted residual CDF at the zero-return barrier from calibration data only. Multiply by a calibrated endpoint-observation probability via the conditional probability chain rule, not an independence assumption.
7. consensus: minimum of direct, recent and competing calibrated estimates; agreement is a selection hypothesis, not independent votes.
8. squeeze_veto: consensus, additionally requiring predicted risk of a >=20% adverse upside path (or incomplete path) <=10%. Short squeeze is only a price-path proxy, not a causal attribution or an execution model.

New past-only geometry features: rebound efficiency; lower-high/lower-low state; downside persistence/run lengths; up/down market response asymmetry; peak age; residual return autocorrelation; negative-shock recovery; trend/volatility acceleration; and within-date peer ranks. No future financial events are inferred from prices. No claim of global never-before-done novelty.

## Fixed search, models and complete policy
Horizons: 30,60,90,126,180,252,504,756. Cutoffs: .60,.70,.80,.85,.90,.95,.975. All 56 mechanism/cutoff policies are reported, not just the best observed row. Each issue date: choose each stock's earliest eligible horizon meeting the cutoff, then the highest-scored stock with deterministic ticker tie breaking. At most one issue per five-session decision date; zero is allowed; no reissue of a ticker through its prior deadline. Later model updates never move an issued endpoint.

Annual refitting; purged trailing 2,520-session training window with every fourth five-session training origin, and a separate latest 252-session calibration-issue window whose outcomes are fully known. Training labels end strictly before the first calibration issue; calibration labels end strictly before outer evaluation begins. Recent model uses half the training history. Tree settings fixed: 80 trees, seven leaves, max depth 3, learning rate .05, minimum leaf size 150, L2=20, max bins 63, seed 20260910, CPU deterministic two threads. Date-balanced weights with 756-session half-life. No early stopping, random-row validation or outcome-driven parameter changes. Platt calibration is fitted outside base-model training. Weak support produces smoothed constants, not zero estimated risk.

A separate adaptive95 evidence policy may pick among the fixed mechanisms/cutoffs using only previously issued fully matured policy predictions; uncertainty and nonoverlap support are required, with a multiplicity-adjusted diagnostic. It may abstain always. It is not a production certificate: overlapping markets and selection invalidate naive IID guarantees. The exact adaptive policy, not per-horizon cherry-picking, is evaluated.

## Data and limitations
Use all available pinned S&P historical-member and Nasdaq historical-member inputs with the existing audited calendar/cutoff adapter. Do not choose securities based on future endpoint availability, later returns, or present-day membership. Missing historical constituents remain a coverage limitation. Historical inputs already inspected by other researchers are reused history, NOT a virgin holdout. Report the 2024+ segment separately, without calling it untouched project-wide. Current-data/feed, corporate-action and prospective evidence blocks remain in force. Optional broader data not present in the pinned input set cannot be silently substituted.

## Required comparisons and falsification
Exact same-date/horizon expected uniform-stock control; same-volatility/relative-momentum peer control; the strongest observed row must be compared against these. Report all horizons, per-era counts, calibration, drawdown/adverse upside, 20/50bp endpoint-margin and next-close-reference sensitivity. Report price decreases, not short-sale P&L (borrow, availability, recalls, dividends, financing and execution are not observed).

Test future price mutation, future label mutation, exact horizon/deadline, flat/unknown/pending separation, probability-class normalization, identity of selected-policy replays, and fixed training-label permutation controls in both universes' recent segment. Count unique tickers, years, issue dates, connected overlap episodes, greedy nonoverlap intervals and report date-block bootstrap paired uncertainty. No confidence claim from an all-success degenerate bootstrap. Record computational/negative results and all attempts. Clean-run reproduction is reproducibility, not independent market evidence.

## Success criterion
>95% realized verified-down precision is a necessary point estimate, not sufficient proof. It also needs adequate temporally distinct evidence, robust controls, clean price/corporate-action data, and genuinely new prospective confirmation before any recommendation can claim validated >95% confidence. No quota, no invented picks, no guarantees. Failing this experiment does not rule out different future inventions.

Primary methodological references: scikit-learn probability calibration documentation; LightGBM objective/parameter documentation; Geifman and El-Yaniv, Selective Classification for Deep Neural Networks, arXiv:1705.08500. Those sources do not establish a finance-specific guarantee for this experiment.
