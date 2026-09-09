# Clean-runtime reproduction — completed 2026-09-09

GitHub Actions run **34405549335** succeeded for S&P and Nasdaq. Executable source commit: `dd917a9dfc0bb13d8068a2449bed365a8c08ee86`. Each clean Ubuntu 24 / Python 3.13 runner installed pinned dependencies, passed **85 tests**, downloaded and SHA-256-verified the eight historical inputs, executed the complete experiment and repeated the real-fold future-label attack.

The artifacts were downloaded and compared programmatically, not merely accepted because CI was green.

| Compared object | S&P | Nasdaq |
|---|---:|---:|
| Raw candidate-horizon forecast rows | 70,584 | 37,160 |
| Selected rows across overlapping policy variants | 1,219 | 445 |
| Issue/abstain decisions, including adaptive95 | 33,250 | 20,950 |
| Stock/date/horizon keys identical | yes | yes |
| Every issue/abstain decision identical | yes | yes |
| Summaries and reliability metrics agree within 1e-12 | yes | yes |
| Every raw forecast agrees within 1e-12 | yes | yes |
| Largest numerical forecast difference | 3.55e-14 | 3.55e-14 |
| Audit counts and actual-fold repeat checks identical | yes | yes |

The first comparison found a reporting-schema difference: original local zero-pick summaries omitted two expected-control fields, while the published code emitted explicit nulls. Local reports were regenerated using the published summary function and unchanged forecast ledgers. No prediction, win count, threshold or model changed. All final comparisons passed. Selected row counts aggregate overlapping policies and are NOT independent sample sizes.

S&P Actions artifact **10125373081**, Nasdaq **10125269803**. The saved user bundle contains their original ZIPs and the machine-readable local comparison. Artifact retention in Actions is 30 days. The within-date label-permutation runs were executed and audited locally; the clean runners reproduced the real-data runs, not the nulls.

Independent runtime reproduction is same-data reproducibility, not independent market evidence or a virgin holdout. Result unchanged: **no predictions at the 95% score cutoff; no adaptive95 recommendations; >95% future precision not established**. No live trades, main edits or production deployment.
