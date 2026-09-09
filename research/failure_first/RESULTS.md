# Failure-first veto network: actual results, 2026-09-09

**The >95% future-accuracy target is NOT established.** This experiment is implemented and executed; it is no longer just an idea document.

## Primary result

At the preregistered 95% estimated-success cutoff:

| Method | S&P wins / matured | Nasdaq wins / matured | Interpretation |
|---|---:|---:|---|
| Ordinary binary control | 0/0 | 22/36 = 61.1% | Nominal confidence is not achieved accuracy |
| Failure-sum ablation | 0/0 | 7/7 = 100% | All seven originated in 2022; one overlapping episode |
| **Failure-veto primary** | **0/0** | **2/2 = 100%** | Both picks are among those seven; not extra evidence |
| Failure-veto plus path veto | 0/0 | 0/0 | Abstained |
| Naive max-channel rule | 110/159 = 69.2% | 49/72 = 68.1% | Invalid aggregation, deliberately tested as a control |
| Separate historical evidence screen | 0/0 | 0/0 | **No statistically qualified recommendations** |

Zero matured forecasts means undefined accuracy, not 100%. No primary or failure-sum 95%-score forecasts were issued from 2024 onward in these archives.

## Why the seven wins are not a validated 95% system

All were issued in **2022** with **252- or 504-session** endpoints. Only **one** can be retained by the greedy non-overlap diagnostic. Even under an unjustified IID assumption, seven wins from seven give a one-sided 95% exact-binomial lower bound of only **65.2%**. A bootstrap [100%,100%] from all-success observations is degenerate, not certainty.

Using the **next session close as entry**, with the same locked endpoint, changes the ablation to **6/7 = 85.7%**. Requiring a 50bp positive endpoint margin also changes it to 6/7. The two stricter primary picks survive both checks. These are sensitivities, not a fill-based trading backtest.

Historical selections below are NOT current buy recommendations:

| Issue | Ticker | Sessions | Archived endpoint return | Worst complete path return | Next-close entry wins? | Primary also selected? |
|---|---|---:|---:|---:|---|---|
| 2022-01-18 | ADP | 252 | +5.00% | -13.31% | yes | yes |
| 2022-03-16 | TXN | 504 | +0.42% | -16.05% | **no** | no |
| 2022-06-03 | ADI | 504 | +49.06% | -16.86% | yes | no |
| 2022-06-17 | VRSN | 252 | +39.88% | +0.15% | yes | yes |
| 2022-07-12 | AAPL | 504 | +62.52% | -14.03% | yes | no |
| 2022-10-05 | MSFT | 504 | +69.29% | -14.02% | yes | no |
| 2022-10-12 | COST | 504 | +98.92% | -3.30% | yes | no |

The exact same-date/horizon random expectation for the seven was **70.8%**. Matching their volatility deciles increases it to **85.9%**. Much of the apparent precision is consistent with favorable risk-profile selection; this does not establish an independent predictive edge. TXN's original gain was just 0.425%, and disappears at the next-close entry.

## The substantive mathematical finding

Predicting failure with one binary model merely relabels predicting success. Our planted-signal/complement test confirms this. The tested hypothesis was instead that an **exhaustive decomposition** of endpoint losses helps identify rare reliable subsets.

The five disjoint price-observable labels are unresolved endpoint; severe terminal loss; other loss with a falling benchmark; other loss after an interim gain; and remaining loss. They are not verified SEC, dilution or bankruptcy causes. A sixth auxiliary model detects interim drawdown/incomplete-path risk. Positive endpoints remain successes even when interim crashes recover.

Crucially, five disjoint failure risks of 3% total **15%**, not 3%. The naive rule that every individual failure head looks safe can manufacture reassuring scores. That control scored only 69.2% and 68.1% at its nominal 95% cutoff. The actual variants use calibrated summed risk, and optionally veto it with a separately calibrated direct failure or path-risk model. No independence assumption is made.

## Scope and leakage control

| Universe | Unique eligible tickers | Eligible feature rows | Outer stock-horizon predictions | Fitted annual/horizon folds | Unsupported folds | Evaluation | Price cutoff |
|---|---:|---:|---:|---:|---:|---|---|
| S&P historical membership | 732 | 483,699 | 2,457,584 | 112 | 0 | 2013+ | 2026-03-20 |
| Nasdaq historical membership | 163 | 51,033 | 234,428 | 54 | 18 | 2018+ | 2026-05-07 |

This is NOT the other agent's separate 10,991-stock panel. The two tested universes overlap, and historical constituent coverage is incomplete.

Eight horizons: **30/60/90/126/180/252/504/756 NYSE sessions**. All methods, thresholds, hyperparameters and selection rules were registered before outcomes in commit `caae36172b75df7d97ace23a1c9fe3b9b01632ff`. No outcome-based threshold or hyperparameter changes were made.

Annual LightGBM heads use 80 trees, seven leaves, depth three, learning rate .05, minimum leaf size 150, L2=20, fixed seed 20260909. Training is date-balanced with a 756-session half-life and 2,520-session rolling history. The newest 252-session calibration issue window is separate from training. **Training outcome dates end before the first calibration issue; calibration outcomes end before the test fit date.** There is no random-row cross-validation or early stopping.

Eligibility/features are fixed using past data. Future endpoint availability does not decide who may be predicted. Missing matured endpoints are failures; immature labels remain pending. The shortest qualifying horizon is selected per stock, then the lowest-risk stock. At most one issuance per five-session decision date, no minimum quota, and no reissuing a ticker through its locked deadline.

## Complete policy grid

Thresholds are model-score cutoffs, NOT realized accuracy. `Random` is the exact expected success rate of a uniform same-date/horizon eligible stock; `Vol` also matches the selected stock's volatility decile. `Nonoverlap` is a greedy diagnostic, not proof of independence. Policies overlap; their sample counts must not be summed as independent evidence. Issued minus matured is pending.

### S&P

| Policy | Issued | Wins/matured | Hit rate | Random | Vol | Nonoverlap |
|---|---:|---:|---:|---:|---:|---:|
| binary@0.7 | 615 | 352/540 | 65.2% | 64.7% | 65.3% | 18 |
| binary@0.8 | 405 | 262/378 | 69.3% | 68.4% | 69.9% | 7 |
| binary@0.85 | 301 | 200/297 | 67.3% | 69.9% | 69.5% | 5 |
| binary@0.9 | 142 | 93/142 | 65.5% | 72.6% | 68.0% | 3 |
| binary@0.95 | 0 | 0/0 | — | — | — | 0 |
| binary@0.975 | 0 | 0/0 | — | — | — | 0 |
| evidence_95 | 0 | 0/0 | — | — | — | 0 |
| failure_sum@0.7 | 615 | 345/549 | 62.8% | 64.7% | 63.0% | 15 |
| failure_sum@0.8 | 403 | 273/386 | 70.7% | 69.1% | 72.3% | 7 |
| failure_sum@0.85 | 311 | 212/311 | 68.2% | 70.8% | 70.8% | 6 |
| failure_sum@0.9 | 181 | 137/181 | 75.7% | 69.6% | 77.1% | 4 |
| failure_sum@0.95 | 0 | 0/0 | — | — | — | 0 |
| failure_sum@0.975 | 0 | 0/0 | — | — | — | 0 |
| failure_veto@0.7 | 614 | 335/539 | 62.2% | 65.1% | 63.7% | 14 |
| failure_veto@0.8 | 385 | 261/367 | 71.1% | 69.5% | 70.8% | 6 |
| failure_veto@0.85 | 284 | 189/284 | 66.5% | 70.6% | 72.3% | 5 |
| failure_veto@0.9 | 130 | 95/130 | 73.1% | 72.5% | 77.0% | 2 |
| failure_veto@0.95 | 0 | 0/0 | — | — | — | 0 |
| failure_veto@0.975 | 0 | 0/0 | — | — | — | 0 |
| failure_veto_path@0.7 | 467 | 271/444 | 61.0% | 64.5% | 63.0% | 10 |
| failure_veto_path@0.8 | 236 | 151/236 | 64.0% | 66.2% | 66.0% | 5 |
| failure_veto_path@0.85 | 172 | 116/172 | 67.4% | 62.1% | 68.4% | 7 |
| failure_veto_path@0.9 | 68 | 53/68 | 77.9% | 65.5% | 78.2% | 3 |
| failure_veto_path@0.95 | 0 | 0/0 | — | — | — | 0 |
| failure_veto_path@0.975 | 0 | 0/0 | — | — | — | 0 |
| naive_channels@0.7 | 665 | 372/655 | 56.8% | 58.3% | 56.7% | 68 |
| naive_channels@0.8 | 665 | 374/646 | 57.9% | 61.1% | 59.9% | 30 |
| naive_channels@0.85 | 657 | 384/605 | 63.5% | 63.2% | 63.1% | 14 |
| naive_channels@0.9 | 571 | 322/474 | 67.9% | 65.8% | 70.6% | 9 |
| naive_channels@0.95 | 251 | 110/159 | 69.2% | 71.5% | 76.3% | 6 |
| naive_channels@0.975 | 22 | 12/16 | 75.0% | 75.4% | 77.8% | 2 |

### Nasdaq

| Policy | Issued | Wins/matured | Hit rate | Random | Vol | Nonoverlap |
|---|---:|---:|---:|---:|---:|---:|
| binary@0.7 | 353 | 150/263 | 57.0% | 60.8% | 59.6% | 10 |
| binary@0.8 | 172 | 78/141 | 55.3% | 56.0% | 55.3% | 8 |
| binary@0.85 | 81 | 43/72 | 59.7% | 57.4% | 55.1% | 5 |
| binary@0.9 | 60 | 35/59 | 59.3% | 59.7% | 59.3% | 3 |
| binary@0.95 | 36 | 22/36 | 61.1% | 57.6% | 59.1% | 1 |
| binary@0.975 | 4 | 2/4 | 50.0% | 68.4% | 49.2% | 1 |
| evidence_95 | 0 | 0/0 | — | — | — | 0 |
| failure_sum@0.7 | 343 | 141/245 | 57.6% | 60.5% | 58.5% | 12 |
| failure_sum@0.8 | 189 | 74/131 | 56.5% | 57.8% | 54.2% | 3 |
| failure_sum@0.85 | 121 | 59/89 | 66.3% | 62.8% | 62.3% | 3 |
| failure_sum@0.9 | 61 | 44/57 | 77.2% | 64.8% | 70.6% | 1 |
| failure_sum@0.95 | 7 | 7/7 | 100.0% | 70.8% | 85.9% | 1 |
| failure_sum@0.975 | 0 | 0/0 | — | — | — | 0 |
| failure_veto@0.7 | 337 | 139/237 | 58.6% | 59.6% | 59.2% | 10 |
| failure_veto@0.8 | 127 | 63/105 | 60.0% | 57.7% | 54.0% | 4 |
| failure_veto@0.85 | 71 | 49/66 | 74.2% | 63.2% | 63.4% | 1 |
| failure_veto@0.9 | 52 | 36/52 | 69.2% | 63.4% | 64.2% | 1 |
| failure_veto@0.95 | 2 | 2/2 | 100.0% | 53.4% | 83.3% | 1 |
| failure_veto@0.975 | 0 | 0/0 | — | — | — | 0 |
| failure_veto_path@0.7 | 189 | 86/145 | 59.3% | 60.5% | 60.3% | 12 |
| failure_veto_path@0.8 | 51 | 29/51 | 56.9% | 54.0% | 55.7% | 2 |
| failure_veto_path@0.85 | 51 | 34/51 | 66.7% | 54.7% | 61.8% | 2 |
| failure_veto_path@0.9 | 35 | 26/35 | 74.3% | 58.0% | 59.9% | 2 |
| failure_veto_path@0.95 | 0 | 0/0 | — | — | — | 0 |
| failure_veto_path@0.975 | 0 | 0/0 | — | — | — | 0 |
| naive_channels@0.7 | 369 | 190/359 | 52.9% | 57.3% | 55.1% | 28 |
| naive_channels@0.8 | 369 | 212/328 | 64.6% | 61.8% | 62.6% | 17 |
| naive_channels@0.85 | 369 | 203/310 | 65.5% | 62.6% | 62.8% | 10 |
| naive_channels@0.9 | 352 | 179/266 | 67.3% | 63.7% | 64.8% | 10 |
| naive_channels@0.95 | 122 | 49/72 | 68.1% | 62.4% | 64.9% | 4 |
| naive_channels@0.975 | 0 | 0/0 | — | — | — | 0 |

## Additional controls

At a 90% score cutoff, the S&P primary wins on **73.1% of 130 shared issuance dates**, versus **63.1%** for ordinary binary; horizons coincide on 121 dates. Nasdaq is **69.2% vs 59.6%** on 52 shared dates, with only 20 coinciding horizons. These are descriptive improvements over the binary control, not independent confirmation of >95% accuracy. The primary's paired date-block excess versus matched random spans zero in both universes.

Prior **SIEVE saved ledgers were recounted**, not refitted. Its Nasdaq 85% row remains **65/79 = 82.3%**, and 95% row **4/5 = 80%**. On shared 85%-cutoff dates, SIEVE was ahead of the new primary (S&P 40/51 vs 24/51; Nasdaq 7/8 vs 5/8), although virtually all horizons differ. This experiment does not establish failure-first as a superior replacement for SIEVE.

**Training-label permutation controls** were run for both universes in 2024–2026. No shuffled binary/sum/primary method issued at 95%. The real versions also had no primary/sum 95% issuances during that window. The shuffled naive Nasdaq model emitted nine forecasts, all pending; none were counted as wins. This is one deterministic shuffle per annual/horizon fit, not a full permutation p-value study.

Uncertainty diagnostics use 2,000 moving-time-block resamples, including no-pick dates, with blocks at least as long as the maximum selected horizon. Greedy non-overlap and conditional exact-binomial lower bounds are reported separately. These diagnostics do not establish independence, stationarity, or forward guarantees. All-success tiny cohorts have degenerate bootstrap precision intervals.

## Validation performed

**50 new regression cases pass.** With the previous SIEVE package's 31 tests, the local total is **81 passed**. Tests cover future prices/labels, constituent isolation, exhaustive failure classes, mathematical union-risk traps, reference/endpoint correctness, next-close entry, missing/pending labels, horizon locks, no-quota abstention, deterministic ties, duplicate/misaligned records, non-overlap boundaries, strict JSON and calibration priors.

A planted-signal model exceeded 97% held-out accuracy on the synthetic fixture. Binary-label inversion gave complementary predictions, confirming that merely predicting failure adds no information. Independent held-out null labels gave chance-level performance and no artificial 95% score bucket.

**57,547 real forecast rows** across four annual/horizon folds were exactly reproduced, then reproduced again unchanged after future-label corruption: S&P h30/2024 and h180/2021, Nasdaq h30/2024 and h126/2023. All saved forecasts passed timing, risk-ordering, reference, locked-horizon and selection audits; all summaries were recomputed from the chosen-forecast ledgers. Both full runs and both shuffled runs passed mechanical audits. Final memory-efficient evaluation reproduced the complete prior Nasdaq summary unchanged.

Six published core files have Git blob hashes matching the executed local files. GitHub Actions workflow `failure-first-validation.yml` additionally performs clean installs, all 50 new tests, end-to-end downloads of the eight immutable/hash-checked inputs, both full model evaluations, and real-fold repetition audits. Its run artifacts contain the complete machine-readable outputs; CI status is available on the branch.

## Corrections and limits

One synthetic null test initially used a deterministic planted-feature boundary; a weak chance split aligned it. The test was corrected to use independent null labels. This did not change the financial model. Two concurrent local jobs exhausted memory; reduced-column/indexed evaluation and serial reruns fixed this without changing the fitted model or thresholds. Additional prediction/outcome timestamp checks were added.

The archives are adjusted/mixed-source, incompletely cover historical constituents, and do not verify terminal corporate actions. They are not certified price-only or fully survivorship-free. Historical data were previously researched; 2024+ is not an untouched holdout. No prospective evidence is available. Price-observable risk groups are not actual SEC/earnings/dilution failure mechanisms. The two universes overlap. Long overlapping forecasts cannot be counted as independent weekly successes.

**No current buy list or live launch is justified.** S&P inputs end March 20, 2026; Nasdaq May 7, 2026. The small 2022 pocket remains a research lead, not an achieved >95% tool. This conclusion rejects the claim for this tested implementation, not every possible failure-first method.
