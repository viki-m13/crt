# PIT S&P 500 Dataset (sp500_pit)

A **point-in-time, survivorship-bias-reduced S&P 500 dataset** spanning
1995-01 → 2026-05, used to validate v5 (and any future model) against a
panel that includes the historical S&P 500 members yfinance cannot
serve (because their tickers were retired on acquisition or
bankruptcy).

This addresses the "largest honesty gap" called out in
`research/01_engine_audit.md`: the v2 panel had only **611 of the 985
unique tickers** that historically belonged to the S&P 500 (51% in 2003,
96% in 2025). The augmented dataset here lifts coverage to **72% in
2003 → 99.7% in 2025**.

## Contents

All files live under `experiments/monthly_dca/cache/v2/sp500_pit/` to
keep the data and the v2 pipeline in the same tree. This directory is
the manifest / entry point.

### Membership (the truth source)
| File | Size | Description |
|---|---:|---|
| `sp500_membership_monthly.parquet` | 84 KB | (asof, ticker) for every month-end 2003-01 → 2026-04. 985 unique tickers across 280 months. Built by `experiments/monthly_dca/v2/build_sp500_pit_membership.py` from `fja05680/sp500` (1996-01..2019-01 daily snapshots) + `sp500_changes_since_2019.csv` (add/remove changelog 2019-01..present). |
| `sp500_hist_1996_2019.csv` | 7.8 MB | Raw daily PIT history (fja05680). |
| `sp500_changes_since_2019.csv` | 3 KB | Add/remove events since 2019. |
| `sp500_today.csv` | 84 KB | Current S&P 500 sanity-check snapshot. |

### Augmented price panel (the PIT-corrected universe)
| File | Size | Description |
|---|---:|---|
| `prices_extended_pit.parquet` | 71 MB | **The headline file.** Daily auto-adjusted Close, 1962-01 → 2026-05, **1994 tickers** (= 1833 original + 161 backfilled). Drop-in replacement for `experiments/monthly_dca/cache/prices_extended.parquet` in any downstream code. |
| `augmented/monthly_prices_clean.parquet` | 4.9 MB | Month-end last close of the augmented panel, with bad-data filter applied (mirrors `v2/build_dataset.py`). |
| `augmented/monthly_returns_clean.parquet` | 5.5 MB | Month-over-month returns, clipped to [-1.0, 2.0]. |
| `augmented/bad_data_tickers.json` | 1 KB | 18 tickers with masked bad-data months (ticker-reuse signatures). |

### Provenance & coverage
| File | Description |
|---|---|
| `backfilled_tickers.json` | Per-ticker source (fnspid or yfinance), date range, PIT window, rename alias if any. |
| `sp500_backfill_plan.json` | Pre-flight plan: 161 backfilled, 213 unreachable on free data. |
| `sp500_missing_tickers.txt` | The 374 PIT tickers not in the original v2 panel. |
| `sp500_yf_classification.json` | yfinance probe results, classified as valid / partial / reuse-suspect. |
| `sp500_yf_probe.json` | Raw probe data: yfinance row counts and date ranges per missing ticker. |
| `fnspid/fnspid_classification.json` | Same classification scheme applied to FNSPID dataset entries. |
| `coverage_after_backfill.csv` | Year-by-year PIT coverage % (before/after). |

### Augmented model outputs (v3 baseline only — see CAVEATS)
| File | Size | Description |
|---|---:|---|
| `augmented/ml_preds.parquet` | 18 MB | Walk-forward HistGBM predictions (1m/3m/6m horizons) retrained on the augmented cross-section. 405k rows. |
| `augmented/sp500_pit_filter_summary.json` | 560 B | Headline metrics of the v3-baseline pit-filter backtest on augmented preds. |
| `augmented/sp500_pit_filter_equity.csv` | 28 KB | Full equity curve. |
| `augmented/sp500_pit_filter_yearly.csv` | 581 B | Yearly returns. |
| `augmented/sp500_pit_filter_walkforward.csv` | 1.4 KB | 10-split WF metrics. |
| `augmented/sp500_pit_filter_coverage.csv` | 560 B | Per-year PIT coverage in the prediction panel. |

### Regenerable artifacts (gitignored)
These are large and regenerable from the scripts; not committed:
- `fnspid/full_history.zip` (589 MB FNSPID raw) — re-downloaded from Hugging Face on demand by Phase 1.
- `fnspid/extracted/*.csv` (124 per-ticker CSVs).
- `augmented/features/*.parquet` (353 monthly feature files, ~150 MB).
- `augmented/panel_cross_section_v3.parquet` (244 MB).

## Data sources

1. **Existing v2 daily panel** (1833 tickers) — yfinance, captured during normal `extend_history.py` runs. Covers 62% of PIT universe.
2. **FNSPID** (Hugging Face `Zihan1004/FNSPID`, CC BY-NC 4.0) — 7700 US stock CSVs scraped through 2023-12. Provides 108 acquired large-cap names with deep history (AGN, ANTM, ABMD, ALXN, ATVI, ARNC, BLL, AET, plus rename-mapped BHGE→BKR, RTN→RTX, UTX→RTX, SYMC→GEN, DWDP→DD).
3. **yfinance** (incremental) — 53 additional names not in (1) and not in (2). Required dot→dash normalisation for BF.B, BRK.B.

All backfills are **validated by date-overlap with PIT membership** to filter ticker-reuse contamination (e.g. modern ACV is Aberdeen Asia-Pacific Income, NOT historical Alberto-Culver).

## Limitations

- **213 OTC bankruptcy "Q" tickers remain unreachable** on free data. Examples: `AAMRQ` (American Airlines pre-2013), `LEHMQ` (Lehman), `WAMUQ` (Washington Mutual), `ANRZQ` (Alpha Natural Resources), `MTLQQ` (Motorola old), `BSC` (Bear Stearns), `EKDKQ` (Eastman Kodak old). These moved to OTC after bankruptcy and neither Yahoo nor FNSPID indexed those trades. True 100% PIT coverage requires CRSP / Sharadar (paid).
- The augmented panel still has the tail-of-the-tail bankruptcy bias these names would have created. The visible Max DD widening (-52% → -60% on the v3 baseline) is a partial signal of that bias.
- Coverage drops to 51% only in 2003 because that's where the **highest density of pre-2010 acquisitions / bankruptcies** are (financial crisis era).

## How to use

### Drop-in for any model that reads `prices_extended.parquet`

Point the code at the augmented file instead:

```python
# Original:
panel = pd.read_parquet(CACHE / "prices_extended.parquet")
# PIT-corrected:
panel = pd.read_parquet(CACHE / "v2" / "sp500_pit" / "prices_extended_pit.parquet")
```

The augmented panel **strictly extends** the original (every original ticker is still there, in the same column). It only adds 161 columns.

### Reproducing the whole pipeline

See `experiments/monthly_dca/v5/spx_pit/REPORT.md` for the 7-phase recipe.

## CAVEATS on the bundled v3-baseline numbers

The `augmented/sp500_pit_filter_*` files are the result of running the
canonical v3 baseline (k=15, ml_3plus6, tight regime gate) on the
augmented preds. **They are NOT the deployed v5 numbers.** Deployed v5
uses a Chronos-bolt-tiny filter on top of GBM rankings, which is a
separate model that wasn't re-run on the augmented panel in this work
(its scoring file `ml_preds_chronos.parquet` was missing from the
checkout, and Chronos inference takes ~90 min on CPU to regenerate).

For the deployed-v5 PIT validation, the missing step is:

```bash
# Add Chronos predictions on the augmented panel (~90 min on CPU)
# (requires `pip install chronos-forecasting torch`)
python3 experiments/monthly_dca/v5/score_chronos.py  # with paths
                                                     # redirected to augmented/
# Then re-run v5 strategy with Chronos filter
python3 experiments/monthly_dca/v5/build_webapp_v5_pit.py  # with paths
                                                           # redirected to augmented/
```

These steps are not yet wired up here; the v3-baseline run was the
first-iteration honest validation.

## Headline impact (v3 baseline only)

| Metric | Original (51%-cov panel) | Augmented (72%-cov panel) | Δ |
|---|---:|---:|---:|
| Full CAGR | 15.05% | 21.05% | +6.0pp |
| WF mean CAGR | 16.22% | 24.63% | +8.4pp |
| WF beats SPY | 5/10 | 7/10 | +2 |
| Max DD | -52.2% | -60.1% | -7.9pp |

CAGR went **up** because the dominant additions are acquired
large-caps (AGN, ANTM, ABMD, CELG, etc.) with real positive returns up
to acquisition date. Max DD widened, which is where the survivorship
correction shows its real cost.

For v5 (with Chronos), the expected direction is the same but the
magnitudes are open until that re-run lands.

## File index

```
experiments/monthly_dca/cache/v2/sp500_pit/
├── sp500_membership_monthly.parquet       # canonical PIT membership
├── prices_extended_pit.parquet            # the headline file (71 MB)
├── backfilled_tickers.json                # provenance
├── sp500_backfill_plan.json
├── sp500_missing_tickers.txt
├── sp500_yf_classification.json
├── sp500_yf_probe.json
├── coverage_after_backfill.csv
└── augmented/
    ├── monthly_prices_clean.parquet       # monthly resample
    ├── monthly_returns_clean.parquet
    ├── bad_data_tickers.json
    ├── ml_preds.parquet                   # walk-forward GBM (v3 baseline)
    ├── sp500_pit_filter_summary.json      # headline metrics
    ├── sp500_pit_filter_equity.csv
    ├── sp500_pit_filter_yearly.csv
    ├── sp500_pit_filter_walkforward.csv
    └── sp500_pit_filter_coverage.csv
```

Scripts that built each artefact: `experiments/monthly_dca/v5/spx_pit/`.
