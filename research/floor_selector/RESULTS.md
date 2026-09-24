# Executed results — 2026-09-09

**Objective not achieved: no validated >95% stock recommendation.**

Two implemented method families were executed on immutable archived inputs. The first uses return-floor/probability thresholds (eight complete policies); the second uses ranks relative to prior failed forecasts (four variants). The second was designed after viewing the first family's failures and is explicitly exploratory. All tested variants are reported.

## Data and execution scope

- S&P archive: 590,816 stock/horizon forecast outputs, 159 monthly decision dates, price feed through 2026-03-20.
- Nasdaq archive: 75,768 outputs, 101 monthly decision dates, price feed through 2026-05-07.
- Total: **666,584 forecast outputs**, not independent trials; includes multiple horizons, correlated stocks and pending outcomes.
- Annual rolling fits, label-maturity purging, past-only correction, immutable deadlines and no repeat ticker before its previous endpoint.
- The source archive is not certified as price-only or complete for terminal corporate actions. Counts describe this archive, not a complete stock universe.

The following tables cover issue dates from 2019 onward. Missing matured endpoints count pessimistically as failures. Pending outcomes are not in the precision denominator. The policies choose at most one stock per monthly decision; horizons are 30 through 756 trading sessions.

## Experiment 1: return-floor and probability policies

| Policy | S&P higher / matured | S&P pending | Nasdaq higher / matured | Nasdaq pending |
|---|---:|---:|---:|---:|
| rank_one | 49 / 81 (60.5%) | 6 | 42 / 74 (56.8%) | 15 |
| p80 | 35 / 52 (67.3%) | 16 | 40 / 56 (71.4%) | 15 |
| p90 | 2 / 3 (66.7%) | 0 | 3 / 5 (60.0%) | 0 |
| p95 | 0 / 0 (undefined) | 0 | 0 / 0 (undefined) | 0 |
| p975 | 0 / 0 (undefined) | 0 | 0 / 0 (undefined) | 0 |
| p99 | 0 / 0 (undefined) | 0 | 0 / 0 (undefined) | 0 |
| positive floor | 0 / 0 (undefined) | 0 | 0 / 1 (0.0%) | 0 |
| positive floor + p95 | 0 / 0 (undefined) | 0 | 0 / 0 (undefined) | 0 |

The complete strict evidence-screened adaptive policy issued **zero recommendations in either universe**. This is a failure to attain the forecasting goal, not 100% accuracy. Merely tightening raw probability cutoffs did not discover a supported 95% region.

## Experiment 2: failure-memory selection

A forecast is ranked by how rarely comparable strong scores appeared among matured previous failures. Batch selection considers all stock/horizon comparisons. The candidate with the shortest eligible horizon is considered before top-one ranking/cooldown. Scores are not labeled individual 95% probabilities, and no distribution-free market guarantee is claimed.

| Score / comparison history | S&P higher / matured | Nasdaq higher / matured |
|---|---:|---:|
| model score / pooled | 23 / 35 (65.7%) | 3 / 8 (37.5%) |
| model score / market-state matched | **20 / 26 (76.9%)** | 2 / 2 (100%; inadequate sample) |
| lower quantile / pooled | 11 / 17 (64.7%) | 7 / 11 (63.6%) |
| lower quantile / market-state matched | 6 / 9 (66.7%) | 0 / 2 (0.0%) |

These selections all have matured endpoints in this run. The best non-tiny subset is **20/26, not >95%**. Its horizons are 15 calls at 756 sessions, seven at 504, three at 30, and one at 180. Thus most of this result concerns two-to-three-year endpoints, not one month. The Nasdaq 2/2 result does not support a high-confidence accuracy claim. Cross-universe results do not establish robust generalization.

## Trajectory checkpoints

On resolved historical endpoints, the quantile model's median center modestly improved mean absolute log-return error versus predicting no change:

| Diagnostic | S&P | Nasdaq |
|---|---:|---:|
| Model median log-MAE | 0.18687 | 0.22540 |
| Flat-price log-MAE | 0.19453 | 0.23056 |
| Nominal 90% checkpoint-band coverage | **86.6%** | **84.4%** |

Intervals under-covered their nominal rate. These are checkpoint diagnostics across overlapping horizons, not simultaneous daily-path coverage. No validated path guarantee is claimed.

## Current scanner output

`NO_PICK`. Both data freshness/target-basis checks and the precision requirement block an actionable recommendation. A latest-available-session research ranking was generated separately, explicitly labeled historical and unvalidated; it is not a current buy recommendation.

## Validation and reproducibility

**32 deterministic tests passed locally**. Tests cover actual model fitting, training maturity, future-outcome/price mutations, endpoint-versus-touch labels, missing/pending denominators, earliest horizon, confidence abstention, cooldown, immutable records, exchange closures, stale/unverified inputs, batch ranking and all-empty end-to-end output.

Run commands are in README.md; source paths and SHA-256 digests are in input_manifest.json. CSV/JSON artifacts retain selected forecasts and all policies. CI checks code/data integrity, not whether the trading objective has been achieved.

This is reused historical research evidence, not an untouched final holdout or a prospective track record. No main-branch, production or order-placement changes are included.
