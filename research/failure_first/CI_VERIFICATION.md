# Independent runtime reproduction — completed 2026-09-09

GitHub Actions run **34399034560** completed successfully for BOTH S&P and Nasdaq. Source commit: `c11bc5024cf50de3799db04fa5649e0023eeceaf`. S&P job `102625983071`; Nasdaq job `102625983481`.

Each clean Ubuntu 24 / Python 3.13 runner performed:

- Pinned dependency installation.
- **50 new regression tests: all passed** (S&P 15.10s, Nasdaq 19.21s).
- End-to-end public-data download and size/SHA-256 verification of all eight pinned historical inputs.
- Full eight-horizon historical model training and policy evaluation.
- Full mechanical audit plus exact real-fold repetition and future-label attacks.

## Compared against the original local runs

Both result artifacts were downloaded and compared programmatically, not merely accepted because CI was green.

| Check | S&P | Nasdaq |
|---|---:|---:|
| Chosen forecast rows across all overlapping policy variants | 8,160 | 3,923 |
| Every stock/date/horizon/deadline key identical | yes | yes |
| Every issued/abstaining decision identical | yes | yes |
| All summary counts identical | yes | yes |
| All nested numerical metrics within 1e-12 tolerance | yes | yes |
| Largest selected numerical difference | 1.22e-15 | 1.11e-15 |

These row counts aggregate overlapping variants and are NOT independent prediction sample sizes. Differences at about 1e-15 are floating arithmetic, not different selections or win rates.

The local suite also included 31 legacy SIEVE regression cases: **81 passed total**, of which 50 are new. The two shuffled-training experiments were run and audited locally; the clean GitHub jobs reran the real-model experiments, not the shuffled controls.

## Result remains unchanged

The primary 95%-score gate: no S&P predictions and 2/2 Nasdaq wins. The decomposition-only ablation: 7/7 Nasdaq wins, all issued in 2022 with overlapping one/two-year horizons, reduced to 6/7 with next-close entry or a 50bp endpoint margin. No picks passed the separate historical evidence screen. **This is not validation of >95% future precision or a live-buy authorization.**

Independent runtime reproduction is reproducibility on the SAME historical data, not independent market evidence or a new untouched holdout. The data/corporate-action limitations in README and RESULTS still apply.

## Machine-readable artifacts

- S&P Actions artifact **10122985259**, `failure-first-sp500`.
- Nasdaq Actions artifact **10122741458**, `failure-first-ndx`.

Each includes `picks.csv`, `decisions.csv`, complete summaries, calibration diagnostics, input/config/source metadata, coverage, fit boundaries, validation checks, and the test log. Actions artifact retention is 30 days; the user-delivered results bundle also preserves these copies.

Draft PR **#191** contains this research. Main and production are unchanged.
