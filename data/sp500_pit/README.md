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

### Augmented model outputs
| File | Size | Description |
|---|---:|---|
| `augmented/ml_preds.parquet` | 18 MB | Walk-forward HistGBM predictions (1m/3m/6m horizons) retrained on the augmented cross-section. 405k rows, 274 months. |
| `augmented/ml_preds_chronos.parquet` | small | Chronos-bolt-tiny p50/p70/p90 3m forecasts on the augmented panel. 104k rows. |
| `augmented/sp500_pit_panel.parquet` | 98 MB | Joined panel (members × features × forward-returns) used by v3/v5 backtest. |
| **Deployed v5-winner outputs (the headline)** | | |
| `augmented/v5_winner_summary.json` | small | Full + WF metrics of v5 (Chronos-filter) on PIT panel. |
| `augmented/v5_winner_walkforward.csv` | small | 10-split WF breakdown for v5. |
| `augmented/v5_winner_equity.csv` | small | v5 equity curve. |
| **Deployed v3-winner outputs** | | |
| `augmented/v3_winner_summary.json` | small | Full + WF metrics of v3 (no Chronos) on PIT panel. |
| `augmented/v3_winner_walkforward.csv` | small | 10-split WF breakdown for v3. |
| `augmented/v3_winner_equity.csv` | small | v3 equity curve. |
| **k=15 v3-baseline pit-filter (reference)** | | |
| `augmented/sp500_pit_filter_summary.json` | 560 B | Headline metrics of the simple k=15 v3-baseline. |
| `augmented/sp500_pit_filter_*.csv` | small | Equity / yearly / walkforward / coverage CSVs. |

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

## What was actually validated

Three configs were run on the augmented PIT panel:

1. **Deployed v5-winner** (`v5_chr_p70_q0.45_k3_invvol_cap0.4_h6_tight`)
   — the production strategy: GBM ml_3plus6 ranking + Chronos-bolt-tiny
   p70 filter (q≥0.45) + top-3 picks + invvol weighting (cap 40%) +
   6m hold + tight regime gate. Outputs:
   `augmented/v5_winner_*` files.
2. **Deployed v3-winner** (`ml_3plus6|k3_3_3|ew|tight|h6|cap1.0`)
   — same as v5 minus the Chronos filter. Outputs:
   `augmented/v3_winner_*` files.
3. **k=15 v3-baseline** (the simple `sp500_pit_filter_backtest`) —
   reference comparison. Outputs: `augmented/sp500_pit_filter_*`.

The Chronos predictions were regenerated from scratch on the augmented
panel (`augmented/ml_preds_chronos.parquet`) using the same
chronos-bolt-tiny model. Compute took ~1.5 min on CPU.

### NaN treatment

When a basket holds a ticker through an acquisition, the post-acquisition
monthly return is NaN. The production sweep code (`sp500_pit_extended_sweep.simulate_variant`)
booked **-100%** on NaN — fine on the biased panel where acquired
tickers were simply absent, catastrophic on the augmented panel
where they're now in the universe.

Patched: NaN → **0%** (honest interpretation = cash payout at last-known
price). This is the minimum-bias correction. The v3-aug and v5-aug
runs both apply this patch via monkey-patching `monthly_returns.fillna(0)`
before the sim sees it.

Without the patch v5-aug would show 9% CAGR / 4% WF mean — an
artifactual catastrophe driven entirely by acquisition NaN. With the
patch it shows 33% CAGR / 33% WF mean — the honest survivorship-
corrected number.

## Headline impact

### Deployed v5-winner (`v5_chr_p70_q0.45_k3_invvol_cap0.4_h6_tight`)

| Metric | Original (biased) | Augmented (PIT) | Δ |
|---|---:|---:|---:|
| Full CAGR | 43.86% | **32.92%** | **-10.94pp** |
| Sharpe | 1.06 | 0.92 | -0.14 |
| Max DD | -48.4% | -51.3% | -2.9pp |
| WF mean CAGR | **47.16%** | **32.68%** | **-14.48pp** |
| WF beats SPY | 10/10 | 8/10 | -2 |

### Deployed v3-winner (`ml_3plus6|k3_3_3|ew|tight|h6|cap1.0`)

| Metric | Original (biased) | Augmented (PIT) | Δ |
|---|---:|---:|---:|
| Full CAGR | 39.77% | **31.81%** | -7.96pp |
| WF mean CAGR | **42.80%** | **25.78%** | **-17.02pp** |
| WF beats SPY | 9/10 | 6/10 | -3 |
| Max DD | -49.8% | -61.4% | -11.5pp |

**Takeaway:** the deployed strategies' 43-47% WF mean CAGR overstates
the PIT-honest number by 14-17pp. Corrected WF mean is ~26-33% —
still strong, still beats SPY by ~19pp/yr on average. Chronos filter
helps: v5 loses less to the correction than v3 (-14.5pp vs -17.0pp).

### k=15 v3-baseline (reference only)

| Metric | Original | Augmented | Δ |
|---|---:|---:|---:|
| Full CAGR | 15.05% | 21.05% | +6.0pp |
| WF mean CAGR | 16.22% | 24.63% | +8.4pp |

The k=15 baseline goes UP not down — concentration is the lever. At
k=3, each delisted/acquired pick has 4-5x more weight, so survivorship
correction bites. At k=15, individual delisting events average out.

Full breakdown and per-split numbers in
[`experiments/monthly_dca/v5/spx_pit/REPORT.md`](../../experiments/monthly_dca/v5/spx_pit/REPORT.md).

## Staggered-tranche timing-luck mitigation (Phase 6)

The deployed v5 rebalances every 6 months → 2 entries / year → exposed
to **rebalance-date luck**. 2024 was a particularly bad example: both
the Jan-31 and Jul-31 entries landed on the year's worst picking
moments, while the other 10 months of the year were positive on
average. Full diagnosis:
[`experiments/monthly_dca/v5/spx_pit/TIMING_LUCK.md`](../../experiments/monthly_dca/v5/spx_pit/TIMING_LUCK.md).

|                          | Lump-sum (deployed) | Basic staggered | Crash-aware staggered |
|--------------------------|--------------------:|----------------:|----------------------:|
| Entries / year           |                  ~2 |              12 |                    12 |
| Active tranches          |                   1 |        up to 6  |              up to 6  |
| Crash-regime gate active for legacy tranches | yes | no   |                   yes |
| **Full-window CAGR**     |          **32.92 %** |       27.77 %  |             29.80 %   |
| Sharpe                   |                0.92 |          0.84   |                 0.86  |
| Max DD                   |             -51.3 % |        -53.6 % |             -51.0 %   |
| **2024 edge vs SPY**     |          **-25.0 pp** |   **+55.8 pp** |             -13.9 pp  |

**Recommended path**: crash-aware staggered. Preserves crash protection
(Max DD identical to lump-sum), preserves Sharpe (0.86 vs 0.92),
partially mitigates 2024-style timing-luck (+11 pp better than lump in
2024), at the cost of ~3 pp WF CAGR vs lump-sum.

Outputs:
- `augmented/v5_staggered_*.{json,csv}` — basic 6-tranche stagger
- `augmented/v5_staggered_ca_*.{json,csv}` — crash-aware variant
- `augmented/v5_staggered_vs_lump.json` — side-by-side summary

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
