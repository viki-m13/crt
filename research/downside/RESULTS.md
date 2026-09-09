# Direct-downside experiment: measured results, 2026-09-09

**The >95% selective downside-accuracy target was NOT established.** Eight mechanisms were implemented, run and checked, not merely proposed. No current buy or short recommendation is authorized.

This predicts a known endpoint strictly below its issue-time reference. It is not the earlier failure-first experiment selecting the safest buys. Flat prices are not declines. Missing endpoints are not declines. An interim crash followed by recovery above reference is not a successful downside prediction.

## 95% score threshold: the decisive result

Seven of eight methods emitted no forecasts at the 95% estimated-score cutoff in either universe. The remaining method, `residual_cdf`, emitted six S&P predictions, of which only **two were correct: 2/6 = 33.3%**. It emitted none in Nasdaq. Every 97.5% cutoff and both separate `adaptive95` historical-evidence policies abstained.

The six S&P estimates were 96.0–97.0%, all issued in 2018 with 504-session horizons. Actual realized precision was 33.3%. Their same-date/horizon random expectation was 37.0%, and the volatility/momentum-matched expectation was 31.0%. Raising a model's displayed score threshold did not manufacture an accurate bearish subset.

## Meaningful-sized versus tiny samples

For orientation, the highest observed full-period precision among rows with **at least 30 matured predictions** was persistence@0.6 in both universes. This display filter is descriptive and post-result; it does not select a validated winning policy. Every one of the 56 fixed policies and the adaptive policy is preserved in `policy_frontier.csv`, including empty and unsuccessful rows.

| Universe / policy | Correct/matured | Precision | Same-date random | Volatility + momentum peers | Nonoverlap intervals |
|---|---:|---:|---:|---:|---:|
| S&P persistence@0.6 | 23/43 | 53.5% | 36.9% | 47.9% | 4 |
| Nasdaq persistence@0.6 | 17/33 | 51.5% | 41.9% | 47.1% | 2 |
| S&P failed-rebound@0.6 | 63/130 | 48.5% | 38.3% | 42.4% | 10 |
| Nasdaq failed-rebound@0.6 | 44/95 | 46.3% | 39.0% | 40.9% | 11 |
| S&P residual_cdf@0.95 | 2/6 | 33.3% | 37.0% | 31.0% | 1 |
| Nasdaq residual_cdf@0.85 | 6/9 | 66.7% | 47.5% | 63.0% | 3 |
| Nasdaq failed-rebound@0.8 | 2/2 | 100% | 45.7% | 47.5% | 2 |

The persistence model targets being below reference at half-horizon, three-quarter-horizon and endpoint. The table evaluates its issued selections on the requested weaker endpoint-only target; a 60% model score is not a demonstrated 60% success rate.

The two perfect Nasdaq failed-rebound calls were LCID on 2023-07-11 and BIIB on 2023-12-21, both 30 sessions. They are different intervals but the same year. Two successes give only a 22.4% one-sided 95% exact-binomial lower bound even under an unjustified independence assumption. A bootstrap of [100%,100%] from two successes is degenerate, not certainty. The S&P instance of that same policy was 0/1.

Nasdaq residual_cdf@0.85 fell from **6/9 to 4/9** when using the next close as the reference and keeping the original deadline. Requiring a decline greater than 50 basis points gave 5/9. One erroneous bearish selection, MRNA in June 2021, rose **126.2%** by its endpoint and 140.3% at the maximum daily close along the path. These are archived price changes, not executed short-sale returns.

## Implemented mechanisms and novelty boundary

1. Direct verified endpoint-decline classifier.
2. Competing-outcome model: observed nondecline, severe decline, other decline with a falling benchmark, remaining decline, and unresolved endpoint. Only observed-decline classes enter the probability sum.
3. Recent-history classifier using a shorter training window.
4. Failed-rebound selector: below the 200-session geometric moving mean, negative relative 63-session momentum, positive 21-session return, and a lower recent 63-session high.
5. Persistent decline across multiple checkpoints.
6. Conditional median plus a held-out residual distribution evaluated at the zero-return barrier, combined with the estimated endpoint-observation probability using the probability chain rule.
7. Consensus/minimum score across direct, recent and competing models.
8. Consensus plus an adverse-upside/incomplete-path veto.

These are modeling and selection hypotheses, not proofs that combined scores are calibrated or independent. They are a project-specific implementation/comparison, not a global novelty claim. Eighteen new causal geometry/market-response features augment the prior 37, giving 55 total. No SEC filings, verified dilution/bankruptcy causes or new fundamental data were ingested in this price-observable batch.

## Scope and chronology

| Universe | Eligible historical tickers | Feature rows | Outer stock-horizon estimates | Fitted annual/horizon folds | Evaluation | Archive cutoff |
|---|---:|---:|---:|---:|---|---|
| S&P historical membership | 732 | 483,699 | 2,457,584 | 112 | 2013+ | 2026-03-20 |
| Nasdaq historical membership | 163 | 51,033 | 234,428 | 54 | 2018+ | 2026-05-07 |

Total: **2,692,012 stock-horizon estimates** and 166 fitted annual/horizon folds. These are not independent bets. Universes, mechanisms and horizons overlap. Nasdaq had 18 unsupported annual/horizon folds that abstained. This is not the other agent's separate 10,991-stock dataset.

Horizon candidates: **30/60/90/126/180/252/504/756 NYSE sessions**. Score cutoffs: **60/70/80/85/90/95/97.5%**. Each stock receives its shortest qualifying horizon, then the highest-scored eligible stock is selected. At most one issuance per five-session decision date, no pick quota, and no reissue through the prior locked deadline.

The protocol was committed before these outcomes in `d7a089e6d8160bc0e51dd0dd677ceca2be56d618`. Annual fits use a 2,520-session training window and a separate 252-session calibration-origin window. Training outcomes finish before calibration begins; calibration outcomes finish before test fitting. Fixed tree settings: 80 trees, depth 3, seven leaves, learning rate .05, minimum leaf 150, L2=20, 63 bins and seed 20260910. Date balancing retains 756-session recency decay. No post-result model or threshold tuning occurred.

## Denominators, controls and uncertainty

Matured unobserved endpoints count as failed verified-down predictions and are separately reported as unknown. Resolved-only and all-unknowns-down sensitivity rates are retained. Pending forecasts are excluded, not counted as wins. Known flat outcomes fail. Zero matured forecasts means undefined accuracy.

Random is the exact expected decline rate for a uniform eligible stock on the selected issue date/horizon, not one fortunate seed. Peers match volatility and relative-momentum quintiles. Benchmark and weakest-momentum controls are also retained. Control pools use the full available eligible universe rather than the smaller universe remaining after each policy's own ticker lockouts. They are selection-context benchmarks, not executable cash-managed portfolios.

Reports include next-close-reference and 20/50bp margin sensitivity, horizon mix, pending/unknowns, no-pick intervals, years, tickers, connected overlap episodes and greedy nonoverlap intervals. A next-close reference leaves one fewer holding session at the original endpoint and is an implementation sensitivity, not a separately optimized policy.

The paired date-block intervals are conditional diagnostics, not distribution-free certificates or a correction for every exploratory comparison. The S&P persistence policy's 5.6-point excess over peers had a wide diagnostic interval of about **-15.8 to +21.4 points**. Nasdaq persistence's peer-excess interval also included zero. Tiny all-success bootstrap rows cannot establish certainty. `adaptive95` uses only matured virtual-policy histories with nonoverlap, minimum support and multiplicity/repeated-look accounting, but still makes no theorem-level future-market guarantee; it issued nothing.

## Recent falsification controls

One fixed joint training-label permutation per universe was tested for all eight horizons over 2024–2026. Calibration remained on real past labels. This is a falsification control, not a multi-permutation significance test. The real forecasts were additionally replayed from 2024 with empty ticker locks so starting states match the null. No retraining or tuning was performed for that comparison.

Examples with identical recent start:
- S&P direct@0.6: real 18/31, null no matured predictions.
- S&P competing@0.6: real 20/41, null 2/4.
- S&P failed-rebound@0.6: real 21/28 (75.0%); Nasdaq same policy real 8/20 (40.0%). This recent pocket does not replicate across universes.
- Nasdaq direct@0.6: real 24/49 versus null 8/18.
- Nasdaq residual_cdf@0.6: real 17/37 versus null 6/17.

Both shuffled experiments emitted zero predictions at the 95% cutoff. Sparse null subsets can look perfect: S&P null residual_cdf@0.7 was 1/1. All null and matched-start real policy comparisons are preserved in the validation bundle and reproducible with `compare_null.py`.

## Verification completed

**108 tests passed**: 50 inherited failure-first plus 58 new downside/resume cases. All four real/shuffled full runs passed mechanical audits. Endpoint labels were independently recomputed from prices. Every selected stock/date/horizon was replayed. Future-price, benchmark, membership and label corruption left the eligible past features/predictions unchanged. Planted negative signals were detected; changing the positive/negative label alone did not create information.

| Audit run | Label rows checked | Forecast rows checked | Selected variant rows replayed | Exact fold repeated |
|---|---:|---:|---:|---:|
| S&P real | 3,869,592 | 2,457,584 | 2,068 | 23,998 |
| Nasdaq real | 408,264 | 234,428 | 1,384 | 4,945 |
| S&P shuffled | 3,869,592 | 430,376 | 70 | 23,998 |
| Nasdaq shuffled | 408,264 | 93,232 | 370 | 4,945 |

GitHub Actions run **34403873937 completed successfully for both universes**, independently installing dependencies, downloading/hash-verifying all eight inputs, running 108 tests per runner, fitting all real models and auditing results. Downloaded artifacts matched every local selection, abstention, observed outcome and summary within 1e-12; the largest difference was about 2.1e-15. See `CI_VERIFICATION.md`. Same-data reproduction is not independent forward evidence. The shuffled controls were run and audited locally, not rerun by that CI workflow.

## Limitations and conclusion

Historical membership does not eliminate missing constituent prices, ticker/corporate-action uncertainty, or the mixed/dividend-adjusted target. NDX membership starts in 2015 and is incomplete early; S&P snapshots are monthly. Sources are stale. No short-borrow availability, recalls, dividends, financing, margin or fills are modeled. Price-direction precision is not short-sale profitability.

The reused 2024+ history is not a virgin research holdout. No model has been approved for current recommendations. Main and production remain untouched.

**These eight fixed implementations did not produce a validated >95% downside selector.** The results reject this batch on these archives, not every possible future price-based or filing/event-based invention. Failed-rebound and persistence filters show limited, regime-sensitive leads rather than a qualifying tool. The complete policy grid, reproducible code, selected ledgers, calibration, null results and audits preserve successes and failures alike.
