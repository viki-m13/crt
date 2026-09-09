# Direct-downside independent runtime reproduction — completed

Run **34403873937** completed successfully for BOTH universes. Tested source: `f507f5eb241a2bc3a1134819cfd32503d59d0098`. S&P job `102642035795`; Nasdaq job `102642035490`.

Each clean Ubuntu 24.04 / Python 3.13 runner installed pinned dependencies, ran all **108 regression tests**, downloaded and verified the size/SHA-256 of all eight immutable public inputs, fitted the complete historical eight-horizon real experiment, and performed independent endpoint/replay/future-data audits. S&P tests: 28.96 seconds; Nasdaq: 30.54 seconds. Tests are software validation, not market accuracy.

Artifacts were downloaded and compared programmatically against local outputs. We did not merely accept a green workflow badge.

| Check | S&P | Nasdaq |
|---|---:|---:|
| Selected rows across overlapping policy variants | 2,068 | 1,384 |
| Every stock, issue time, horizon and locked endpoint identical | yes | yes |
| Every issuance and abstention identical | yes | yes |
| Every realized success/unknown/pending status identical | yes | yes |
| Summary/calibration/fit-audit/validation metrics within 1e-12 | yes | yes |
| Largest numeric discrepancy | 2.11e-15 | 2.22e-16 |
| Independently checked endpoint-label rows | 3,869,592 | 408,264 |
| Outer prediction rows audited | 2,457,584 | 234,428 |
| 2024 30-session fold reproduced with future-label attack | 23,998 rows | 4,945 rows |

The selected-row counts aggregate overlapping variants, not independent observations. Floating differences at approximately 1e-15 do not change any decision or reported hit rate.

## Additional local controls

The two fixed shuffled-training experiments were executed and mechanically audited locally, including exact-fold repetition and future-label attacks. They were not separately rerun in this CI workflow. The 2024 real/null comparison resets both ticker-lock states and reuses already-trained real forecasts, without fitting new parameters. Complete comparisons and logs are in the delivered validation bundle.

## Engineering interruptions and equivalence

The constrained local runtime had approximately 6 GB RAM. Concurrent large tasks caused resource termination. Completed horizon checkpoints were preserved; a configuration/source-hash-checked resume re-ran incomplete horizons without altering numerical models. Evaluation/audit loading was then made leaner. The original and lean Nasdaq evaluations match exactly for all picks, decisions, complete summaries and calibration metrics. The independently completed original-source S&P CI run matches the lean local S&P evaluation to the tolerance above. Thus the memory changes did not change model outputs or the selection result.

Only data loading/object lifetime and deterministic resume behavior were changed. `model.py`, `features.py` and `policy.py` remained unchanged. Source and input hashes are recorded in run metadata. Dependency and experiment configuration are pinned.

## Preserved artifacts

- S&P artifact **10125034549**, `direct-downside-sp500`.
- Nasdaq artifact **10124651201**, `direct-downside-ndx`.

Both contain selected forecasts, every issuance/no-pick decision, summaries, calibration tables, metadata, data coverage, fit boundaries, mechanical audit results, and run/test logs. Actions retention is 30 days; copies are included in the user-delivered ZIP.

## Interpretation

Independent runtime reproduction is not independent market evidence. The same archived adjusted/mixed inputs and the same reused history underlie both runs. The goal remains unmet: at the 95% score cutoff, only six S&P distribution-model predictions were emitted and only two were correct; Nasdaq had none. Neither adaptive95 policy issued a forecast. No current trading recommendation or >95% future confidence claim is authorized.
