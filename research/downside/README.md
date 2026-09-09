# Direct-downside research

**Status: implemented, executed and independently reproduced. The >95% target is not established.** Read `RESULTS.md`, `policy_frontier.csv`, `CI_VERIFICATION.md` and the frozen `PROTOCOL.md`. This is a research CLI, not a live stock scanner or a short-selling system.

## Contract

A forecast is `(ticker, issue timestamp, reference price, fixed horizon)`. The target is a known endpoint strictly below reference. Horizons are 30/60/90/126/180/252/504/756 NYSE sessions. Flat and missing outcomes are not successful declines. Pending forecasts do not count as correct. Each forecast locks its endpoint; later model changes cannot extend a losing deadline. There is no pick quota.

Each complete policy first chooses the shortest qualifying horizon per stock, then the highest-ranked stock. At most one forecast is issued on each five-session decision date. A ticker is not reissued while its previous forecast remains active. Eight mechanisms and seven score thresholds are evaluated; a separate matured-history evidence policy may abstain indefinitely. A score of .95 is not an empirically established 95% success probability.

## Reproduce from the repository root

Use Python 3.13 and run the large universes sequentially on memory-constrained machines. The GitHub workflow gives each universe a separate runner. The inherited failure-first package is required and is present in this branch.

```bash
python -m pip install -r research/downside/requirements.txt
python -m pytest tests/test_failure_first.py tests/test_downside.py tests/test_downside_resume.py -q
python -m research.failure_first.download --output pinned-inputs

python -m research.downside.run --inputs pinned-inputs --output results/sp500 --universe sp500
python -m research.downside.validate --inputs pinned-inputs --directory results/sp500
python -m research.downside.run --inputs pinned-inputs --output results/ndx --universe ndx
python -m research.downside.validate --inputs pinned-inputs --directory results/ndx

python -m research.downside.run --inputs pinned-inputs --output results/null_sp500 --universe sp500 --null --years 2024 2025 2026
python -m research.downside.validate --inputs pinned-inputs --directory results/null_sp500
python -m research.downside.run --inputs pinned-inputs --output results/null_ndx --universe ndx --null --years 2024 2025 2026
python -m research.downside.validate --inputs pinned-inputs --directory results/null_ndx

python -m research.downside.compare_null --full results/sp500 --null results/null_sp500 --output results/recent_sp500
python -m research.downside.compare_null --full results/ndx --null results/null_ndx --output results/recent_ndx
```

`--resume` resumes completed whole horizons only when configuration, universe, null status, input manifest and numerical source hashes match. `--evaluate-only` rebuilds the complete selection policy from saved forecasts. The latter is not a new model-fitting experiment. A smoke test can use `--universe ndx --years 2024 --horizons 30`, but cannot stand in for the full evaluation.

## Files and outputs

- `features.py`: inherited causal states plus failed-recovery/market-response geometry.
- `model.py`: exact downside labels, direct and competing models, recent-history, persistence, conditional residual distribution and consensus/veto mechanisms.
- `policy.py`: outcome-blind stock/horizon selection, lockouts, matched controls and uncertainty diagnostics.
- `run.py`: annual purged fits, forecast checkpoints, evaluation and hash/config metadata.
- `validate.py`: independent endpoint calculation, selected-policy replay and future-data attacks.
- `compare_null.py`: same-start real/shuffled policy comparison without retraining.

Each full run saves features, eight outcome panels and forecast panels, selected `picks.csv`, every `decisions.csv` row, `summary.json/csv`, calibration grids, coverage, fit boundaries, configuration/source/input metadata and `validation.json`. These are research records, not broker fills. The downloaded validation bundle preserves selected ledgers and diagnostics; large reproducible features/all-candidate Parquet panels are not embedded in that smaller bundle.

The full expanded narrative report is supplied in the conversation download. The repository summary and compact frontier preserve all fixed policy results. Actions artifacts expire after 30 days; user-delivered copies preserve them separately.

## Data boundaries

Inputs are pinned to CRT commit `9939e0f1e0f0d601c14b2d21ff14a38da82cc2e2`, with exact byte sizes and SHA-256 in `research/failure_first/input_manifest.json`. They are the existing S&P historical-member panel and the Nasdaq data archived from bonds, not a newly acquired whole-market database. No private-repository contents or new account credentials were copied here.

Current source cutoffs are 2026-03-20 for S&P and 2026-05-07 for Nasdaq. Prices are adjusted/mixed, historical constituent coverage is incomplete, terminal corporate actions are not fully verified, and NDX membership starts in 2015. S&P membership is monthly. These limitations block a literal, current, survivorship-free raw-price claim. Prior data-source licenses/restrictions still apply; this work is research-only.

No actual SEC filing or causal distress-event feature was added. Squeeze and systemic labels describe price paths/co-occurrences, not verified mechanisms. There is no observed short availability, borrow, dividend-liability, recall, financing, margin or execution model. Directional accuracy is not short-sale profitability. Existing history is reused research data, including 2024+; no untouched project-wide holdout or prospective success is claimed.

## Methodological references

- Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- LightGBM objectives and parameters: https://lightgbm.readthedocs.io/en/stable/Parameters.html
- Geifman and El-Yaniv, *Selective Classification for Deep Neural Networks*: https://arxiv.org/abs/1705.08500

These support components of the methodology, not a finance-specific confidence guarantee. Every empirical result is this experiment's own measurement on the pinned inputs. No claim of global novelty or a guaranteed decline is made.
