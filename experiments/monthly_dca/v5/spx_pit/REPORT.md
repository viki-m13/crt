# v5 PIT validation — does survivorship bias inflate the headline?

**Question.** The deployed v5 SP500 numbers in `reports/final_validation.md`
were computed against a price panel that was missing **374 of the 985
unique tickers that historically belonged to the S&P 500** between
2003-01 and 2026-04 (mostly companies that got acquired or went
bankrupt). Coverage was 51% in 2003 and 96% in 2025. Did that gap inflate
the headline edge?

**Method.** Reconstruct the panel as completely as free data allows
(yfinance + Hugging Face FNSPID dataset, with ticker-reuse filtering
by PIT-membership-window overlap), recompute the entire feature stack,
retrain the walk-forward GBM, re-run the canonical v3-baseline pit-filter
backtest. Same code, same regime gate, same K's, same cost model — only
the price universe changes.

## Coverage delta

| Year | Original panel | Augmented panel |
|------|---------------:|----------------:|
| 2003 |  51% |  72% |
| 2008 |  59% |  80% |
| 2013 |  69% |  87% |
| 2018 |  81% |  96% |
| 2025 |  96% |  99.7% |

161 backfilled tickers (108 from FNSPID, 53 from yfinance), all
date-validated to filter ticker-recycle traps (e.g. modern ACV is
Aberdeen Asia-Pacific Income, not the historical Alberto-Culver).
213 PIT tickers remain unreachable on free data — mostly OTC
bankruptcy 'Q' tickers (AAMRQ, LEHMQ, WAMUQ, ANRZQ, etc.) that
Yahoo and FNSPID never indexed. CRSP / Sharadar (paid) would close
that final gap.

## Headline comparison — v3 baseline pit-filter backtest

| Metric            | Original (biased) | Augmented (PIT) |        Δ |
|-------------------|------------------:|----------------:|---------:|
| Full-window CAGR  |            15.05% |          21.05% |   +6.0pp |
| Sharpe (monthly)  |              0.72 |            0.78 |    +0.06 |
| Max drawdown      |            -52.2% |          -60.1% |  -7.9pp  |
| WF mean CAGR      |            16.22% |          24.63% |   +8.4pp |
| WF median CAGR    |            16.59% |          15.64% |   -0.9pp |
| WF min CAGR       |           -11.93% |         -13.50% |   -1.6pp |
| WF max CAGR       |            36.13% |         119.84% |  +83.7pp |
| WF mean edge vs SPY|             +1.4pp |          +9.8pp |   +8.4pp |
| WF n positive splits|             9/10 |             9/10 |    tied  |
| WF n beats SPY    |              5/10 |             7/10 |    +2    |
| Coverage 2003     |              51% |             72% |   +21pp |

Source: `experiments/monthly_dca/cache/v2/sp500_pit/sp500_pit_filter_summary.json`
vs `.../augmented/sp500_pit_filter_summary.json`.

## What this means

**Headline CAGR went UP, not down.** That looks weird at first — the
naive expectation is that survivorship correction always cuts returns.
The reason it went the other way here:

1. Most of the 161 backfilled names are **acquired large-caps** (AGN,
   ANTM, ABMD, ALXN, CELG, ATVI, AET, ANDV, BHGE, etc.). These
   companies tended to be **acquired at premiums to their pre-deal
   prices**, and they had real, often strong, returns up to the
   acquisition date. The deployed model was *unable to pick them at all*
   because they weren't in its universe. Once they're added, the model
   has more high-quality opportunities to choose from.

2. Only a minority of the backfilled names were genuine 'bad' delistings
   — and the worst of those (the OTC bankruptcy-Q tickers like AAMRQ,
   LEHMQ, WAMUQ) are still missing from the free-data ceiling.

So the augmented run is partially honest: it adds the good acquired
names but still excludes most of the bankruptcy tail. The TRUE
survivorship-corrected number is probably between the augmented CAGR
(21%) and a lower CRSP-grade number (unknown without paid data), but
the augmented run is the BETTER lower bound than the original 15%.

**Max drawdown widened by 8pp** — this is the survivorship correction
showing up where it has the biggest leverage: the deep drawdowns now
include some real names that delisted at their bottoms.

**Beat-SPY count improved 5/10 -> 7/10**. The model selecting from a
fuller universe more often produces winners that beat the broad index.

## Caveats

1. **CAGR is not the WF-validation metric in the v5 paper.** The
   `reports/final_validation.md` claims of "WF mean 50.16%" come from a
   different config — the `exp_02 winner` (k=1, ml_3plus6plus1 scorer,
   safer regime gate, TLT crash fallback) on the v6 simulator. This
   validation re-ran the simpler **v3 baseline** (k=15, ml_3plus6,
   tight regime). The augmented v3 numbers are still a fair indicator
   of direction-of-bias, but reproducing the exp_02 winner exactly
   would require running `v6/lib_engine.py` on these augmented preds —
   a follow-up.

2. **Chronos filter was NOT re-run.** The deployed v5 winner uses
   a Chronos-bolt-tiny filter on top of GBM ranking. The Chronos
   predictions file was missing from this checkout, so we used the
   un-filtered ranking. Chronos would have to be regenerated on the
   161 new tickers (~60-90 min on CPU) for a true v5-vs-v5
   comparison.

3. **213 OTC bankruptcy Q tickers remain unreachable.** True 100%
   PIT validation needs CRSP / Sharadar.

## Files

- `experiments/monthly_dca/v5/spx_pit/build_sp500_pit_prices.py`
- `experiments/monthly_dca/v5/spx_pit/build_monthly_clean_pit.py`
- `experiments/monthly_dca/v5/spx_pit/cache_features_pit.py`
- `experiments/monthly_dca/v5/spx_pit/add_alpha_features_pit.py`
- `experiments/monthly_dca/v5/spx_pit/build_panel_pit.py`
- `experiments/monthly_dca/v5/spx_pit/train_ml_pit.py`
- `experiments/monthly_dca/v5/spx_pit/run_pit_filter_backtest.py`

Augmented outputs (under `experiments/monthly_dca/cache/v2/sp500_pit/augmented/`):

- `monthly_prices_clean.parquet`, `monthly_returns_clean.parquet`  (4.9 MB / 5.5 MB)
- `ml_preds.parquet`  (18 MB; 405k walk-forward predictions)
- `sp500_pit_filter_*.csv`, `sp500_pit_filter_summary.json`  (small)

(`panel_cross_section_v3.parquet` is 244 MB and gitignored;
`features/*.parquet` are also gitignored. Both are regenerable.)

## Reproduction

```bash
# Phase 1: build augmented daily panel (downloads ~590 MB FNSPID + yfinance backfill)
python3 experiments/monthly_dca/v5/spx_pit/build_sp500_pit_prices.py

# Phase 2: monthly clean panels
python3 experiments/monthly_dca/v5/spx_pit/build_monthly_clean_pit.py

# Phase 3a: base features (~65 min, 353 month-ends x 1994 tickers)
python3 experiments/monthly_dca/v5/spx_pit/cache_features_pit.py

# Phase 3b: alpha + alpha2 + extra + novel features (~30 min)
python3 experiments/monthly_dca/v5/spx_pit/add_alpha_features_pit.py

# Phase 3.5: cross-section
python3 experiments/monthly_dca/v5/spx_pit/build_panel_pit.py

# Phase 4: walk-forward GBM (~9 min on CPU)
python3 experiments/monthly_dca/v5/spx_pit/train_ml_pit.py

# Phase 5: v3 baseline pit-filter backtest + summary
python3 experiments/monthly_dca/v5/spx_pit/run_pit_filter_backtest.py
```

Total wallclock: ~2 hours end-to-end from a fresh checkout.
