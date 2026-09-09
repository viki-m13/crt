# Downside state recurrence — executed results, 2026-09-09

**The >95% precision target was not achieved.** All seven mechanisms issued zero forecasts at the 95% score cutoff in both universes. The separate adaptive95 historical evidence screen also issued zero. Zero predictions means undefined accuracy, not perfect performance.

This is implemented and executed research, not another proposal. Protocol commit: `409753505c1c90d915ce9fbcd83b6840bd70a383`. Executable source plus clean-run workflow: `dd917a9dfc0bb13d8068a2449bed365a8c08ee86`. The existing direct-downside branch was discovered and preserved; this extension is isolated in `research/recurrence_downside/`.

## New mechanism

Match normalized historical market/stock states rather than visual chart shapes. Form an ex-ante candidate pool from extreme relative weakness, trend deceleration and failed recovery. Compare each candidate with up to 128 earlier states whose complete forward outcomes were already available at the annual fit date. Balance observations by historical issue date, apply recency decay/shrinkage, and require at least 32 neighbor rows from 12 dates. Variants require the same market regime, support from two historical eras, or agreement with a direct decline classifier. The failed-recovery consensus is the primary hypothesis.

Seven methods and seven fixed score cutoffs produce 49 complete policies per universe, plus the adaptive95 diagnostic. Eight possible locked horizons: 30/60/90/126/180/252/504/756 NYSE sessions. Each policy picks the shortest qualifying horizon per stock, then the highest-scored stock; at most one forecast per five-session date; no reissue through its prior deadline. No minimum pick quota.

Success is a strictly LOWER observed endpoint than the archived issue price. Flat, unresolved and recovered intraperiod declines are not bearish successes. Pending outcomes remain pending. Raw recurrence estimates and minima of scores are not calibrated confidence guarantees.

## Main comparisons at the frozen 0.60 score cutoff

| Method | Universe | Wins/matured | Verified decline | Full-universe random expectation | Candidate-pool random expectation | Next-close reference | Nonoverlap n |
|---|---|---:|---:|---:|---:|---:|---:|
| Direct | S&P | 52/122 | 42.6% | 38.7% | 45.8% | 42.6% | 11 |
| Direct | Nasdaq | 58/114 | 50.9% | 40.3% | 50.1% | 50.9% | 11 |
| Recurrence | S&P | 185/399 | 46.4% | 35.9% | 42.2% | 46.4% | 21 |
| Recurrence | Nasdaq | 56/114 | 49.1% | 40.4% | 46.1% | 50.0% | 22 |
| Failed-recovery consensus (primary) | S&P | 14/31 | 45.2% | 40.4% | 42.7% | 45.2% | 6 |
| Failed-recovery consensus (primary) | Nasdaq | 1/6 | 16.7% | 44.0% | 63.4% | 16.7% | 4 |

Controls are exact expected hit rates at the SAME decision dates and chosen horizons, not one fortunate random seed. Candidate controls use only the frozen weakness/recovery pool, testing incremental information beyond selecting weak stocks. Volatility-decile and volatility-plus-relative-strength controls are also saved.

The modest lead: S&P recurrence at 0.60 has an observed +4.1 percentage points over the candidate pool. Its conditional time-block paired interval is about +1.2 to +7.0 points, but this is NOT multiplicity-adjusted across searched policies and reused historical research; only about 4.4 maximum-horizon blocks exist. It is not proof of a robust tradable edge. Nasdaq's candidate-paired interval includes zero (-5.2 to +8.4 points). The failed-recovery veto did not improve reliability.

## Every nonempty policy, including bad results

All omitted score cutoffs issued no predictions. The full 49-policy tables, zero rows, counts, margins, controls and era metrics are produced in each run's `summary.csv` and `summary.json`; `adaptive95` is a separate 50th row. Do not add samples across these overlapping policies.

| Policy | S&P wins/matured | Nasdaq wins/matured |
|---|---:|---:|
| confirmed_recovery@0.6 | 14/31 | 1/6 |
| consensus@0.6 | 25/59 | 9/25 |
| consensus@0.7 | 0/1 | 0/0 |
| cross_era@0.6 | 14/39 | 1/1 |
| direct@0.6 | 52/122 | 58/114 |
| direct@0.7 | 15/36 | 23/48 |
| direct@0.8 | 0/0 | 2/7 |
| recurrence@0.6 | 185/399 | 56/114 |
| recurrence@0.7 | 14/44 | 1/1 |
| recurrence@0.8 | 2/2 | 0/0 |
| regime_recurrence@0.6 | 168/368 | 51/111 |
| regime_recurrence@0.7 | 14/44 | 1/1 |
| regime_recurrence@0.8 | 1/3 | 0/0 |
| regime_recurrence@0.85 | 1/1 | 0/0 |
| robust_consensus@0.6 | 2/5 | 0/0 |
| EVERY method at 0.95 or 0.975 | 0/0 | 0/0 |
| adaptive95 evidence screen | 0/0 | 0/0 |

## High-selectivity traps

The S&P recurrence 0.80 cell was 2/2: CLF issued 2013-06-04 for 756 sessions and WBD issued 2022-10-31 for 180 sessions. These are historical examples, not current recommendations. WBD declined only 0.539%. Two successes do not establish 95% reliability. The other 1/1 cells are equally insufficient. Increasing the score cutoff did not reliably increase measured precision.

For recurrence at 0.60, the proportion falling more than 5% was only 34.3% S&P / 39.5% Nasdaq; more than 10% was 27.3% / 32.5%. One complete S&P path rose approximately 455.5% above its issue reference along the way. These are archived price-path observations, NOT short-sale P&L. Borrow availability, recalls, dividends, financing and executable fills were not observed.

## Scope

| Universe | Eligible tickers | Eligible feature rows | Full-history candidate rows | Outer candidate-horizon forecasts | Evaluation | Archive cutoff |
|---|---:|---:|---:|---:|---|---|
| S&P historical members | 732 | 483,699 | 15,191 | 70,584 | 2013+ | 2026-03-20 |
| Nasdaq historical members | 163 | 51,033 | 6,287 | 37,160 | 2018+ | 2026-05-07 |

Universes overlap. This is not 895 independent securities and not the separate 10,991-stock panel. Candidate counts include pre-evaluation reference history. Historical coverage, mixed adjustments and terminal corporate actions remain limitations. No new SEC or financial-deterioration event history was assembled.

## Fixed permutation falsification, 2024–2026

Permute labels WITHIN historical issue dates in training/reference data, preserving market-wide date outcomes while destroying their stock assignment. Calibration stays separate and past-only. This is a stock-selection null, not a null of all market predictability.

| Universe | Real recurrence@0.60 recent wins/n | Within-date-permuted wins/n |
|---|---:|---:|
| S&P | 23/50 = 46.0% | 11/25 = 44.0% |
| Nasdaq | 29/44 = 65.9% | 4/10 = 40.0% |

The policies select different dates/horizons under permutation; this is not a paired same-trade significance test. Full null policy tables are preserved. No null variant generated a 95%-score recommendation. A favorable isolated recent segment is not a new holdout and did not retroactively change method/cutoff.

## Completed local validation

- 85 regression tests passed: 35 new recurrence/downside tests plus 50 prior failure-first tests. Synthetic fixtures test mechanics, not financial predictability.
- Independently reconstructed 107,744 real candidate-horizon endpoint outcomes from raw archived prices; checked candidate inclusion, purges, fit dates and locked horizons.
- Replayed 53,116 ordinary-policy issue/abstain decisions and matched every saved stock/date/horizon. The adaptive95 screen has synthetic future-data tests and produced no forecasts.
- Refit 2024 horizons 30/252 in each universe: 2,444 forecast rows matched exactly, including after deliberately corrupting future outcome labels.
- Audited both fixed-label-permutation runs: 23,064 further endpoint rows and 11,172 decisions checked.
- Executable source blobs match published GitHub blob hashes. All eight historical public inputs are SHA-256 verified.

## Reproducibility

Actions run `34405549335` independently performs clean installation, 85 tests, public-input downloads/hashes, both full real runs and actual-fold adversarial audits. Its final completion/comparison is recorded separately in `CI_VERIFICATION.md`; do not infer completion merely from the workflow's existence. The permutation runs were local, not part of that clean runner. Reproduction on the same data is NOT independent market evidence.

## Issues caught

Float32 control-bin boundaries disagreed with float64 row lookups; fixed consistent float64 binning and added a regression test. Entirely abstaining runs needed a stable empty schema/null metrics; fixed and tested. An initial concurrent local S&P process ended after writing annual forecasts but before policy reporting; the cause was not established. Reporting was replayed standalone from identical cached forecasts and validated with raw-price reconstruction and actual-model refits. No model/threshold was changed to improve results.

## Interpretation

The recurrence and weak-stock restriction show modest descriptive separation, not a 95% predictor. Cross-era/failed-recovery restrictions mostly reduced usable evidence instead of creating certainty. This rejects the tested configurations, not future invention. Unknown terminal payouts are not called bankruptcies; absent prices are not called declines. These are reused historical samples, not virgin holdouts. Stale March/May 2026 data cannot produce a current buy/sell list. Main and production remain unchanged.
