# v5 PIT validation — does survivorship bias inflate the headline?

**Question.** The deployed v5 SP500 numbers in `reports/final_validation.md`
were computed against a price panel that was missing **374 of the 985
unique tickers** that historically belonged to the S&P 500 between
2003-01 and 2026-04 (mostly acquired or bankrupt names whose tickers
Yahoo retired). Coverage was 51% in 2003 and 96% in 2025. Did that gap
inflate the headline edge?

**Method.** Reconstruct the panel as completely as free data allows
(yfinance + FNSPID, with date-overlap validation to filter ticker
reuse), recompute features, retrain the GBM walk-forward, regenerate
Chronos-bolt-tiny predictions on the augmented panel, then re-run
**both** the deployed-v3-winner config and the deployed-v5-winner
config (the one with the Chronos filter). Same scripts, same regime
gate, same K, same cost, same hold horizon — only the price universe
changes.

## Coverage delta

| Year | Original panel | Augmented panel |
|------|---------------:|----------------:|
| 2003 |  51% |  72% |
| 2008 |  59% |  80% |
| 2013 |  69% |  87% |
| 2018 |  81% |  96% |
| 2025 |  96% |  99.7% |

161 backfilled tickers (108 from FNSPID via Hugging Face, 53 from
yfinance), all date-validated to filter ticker-recycle traps (e.g.
modern ACV is Aberdeen Asia-Pacific Income, NOT Alberto-Culver).
213 OTC bankruptcy-Q tickers (AAMRQ, LEHMQ, WAMUQ, ANRZQ, BSC, ...)
remain unreachable on free data — these would require CRSP / Sharadar.

## Headline comparison — deployed v5-winner (with Chronos filter)

The actual deployed strategy:
`v5_chr_p70_q0.45_k3_invvol_cap0.4_h6_tight`
(GBM ml_3plus6 ranking + Chronos-bolt-tiny p70 3m filter at q≥0.45 +
top-3 picks + inverse-vol weighting capped at 40% + 6m hold + tight
regime gate).

| Metric            | Original (biased) | Augmented (PIT) |        Δ |
|-------------------|------------------:|----------------:|---------:|
| Full-window CAGR  |            43.86% |          32.92% | **-10.94pp** |
| Sharpe (monthly)  |              1.06 |            0.92 |   -0.14 |
| Max drawdown      |            -48.4% |          -51.3% |    -2.9pp |
| WF mean CAGR      |        **47.16%** |      **32.68%** | **-14.48pp** |
| WF median CAGR    |            43.49% |          31.60% |  -11.89pp |
| WF min CAGR       |            23.08% |          12.35% |  -10.73pp |
| WF max CAGR       |           104.59% |          63.45% |  -41.14pp |
| WF mean edge vs SPY|            +32.4pp |        +19.3pp |   -13.1pp |
| WF n positive     |              10/10 |           10/10 |    tied  |
| WF n beats SPY    |              10/10 |            8/10 |    -2    |

Source: `experiments/monthly_dca/cache/v2/sp500_pit/v5_winner_summary.json`
vs `.../augmented/v5_winner_summary.json`.

## Headline comparison — deployed v3-winner (no Chronos)

`ml_3plus6|k3_3_3|ew|tight|h6|cap1.0`. Same as v5 minus the Chronos filter.

| Metric            | Original (biased) | Augmented (PIT) |        Δ |
|-------------------|------------------:|----------------:|---------:|
| Full-window CAGR  |            39.77% |          31.81% |   -7.96pp |
| Sharpe            |              0.96 |            0.78 |   -0.17 |
| Max drawdown      |            -49.8% |          -61.4% |   -11.5pp |
| WF mean CAGR      |        **42.80%** |      **25.78%** | **-17.02pp** |
| WF median CAGR    |            39.90% |          14.18% |   -25.72pp |
| WF n beats SPY    |              9/10 |            6/10 |    -3    |

Source: `v3_winner_summary.json` vs `augmented/v3_winner_summary.json`.

## v3-baseline pit-filter (k=15 — for reference)

This was my first comparison. Less informative because k=15 dilutes
concentration risk, so the bias doesn't show as strongly. Included
here so the numbers cross-reference.

| Metric            | Original | Augmented |        Δ |
|-------------------|---------:|----------:|---------:|
| Full CAGR         |   15.05% |    21.05% |   +6.0pp |
| WF mean CAGR      |   16.22% |    24.63% |   +8.4pp |
| WF beats SPY      |     5/10 |      7/10 |    +2    |

## What this means

1. **The deployed v5 model's 47% WF mean CAGR overstates the PIT-honest number
   by ~14.5pp.** The corrected WF mean is ~32.7% — still strongly positive,
   still beats SPY by 19pp/yr on average, still positive in 10/10 splits. But
   the "47%" headline number does not survive an honest PIT correction.

2. **Chronos filter is still doing useful work.** v5 (with Chronos) drops
   less than v3 (without): WF mean -14.5pp vs -17.0pp. So the Chronos signal
   is partially protecting against picking the bad-acquired names.

3. **The cost shows up in WF dispersion, not just headline.** WF median
   on v3 falls from 39.9% to 14.2% — a 26-point haircut on the median split.
   Both v3 and v5 see widening Max DD (acquired/delisted names that previously
   couldn't enter the basket now occasionally do).

4. **A k=15 baseline barely shifts** (and even goes up). Concentration
   is the lever. At k=3, each delisted/acquired pick has 4-5x more impact.

## Treatment of acquired-company NaN returns

For tickers like AGN, ANTM, ABMD, CELG, AET, etc. — these were
ACQUIRED, not bankrupt. At acquisition close, shareholders received cash
(typically at a premium). In the augmented panel, the post-acquisition
monthly return shows NaN.

The production sweep code (`sp500_pit_extended_sweep.simulate_variant`)
booked **-100%** loss on NaN returns. That was harmless on the original
biased panel (which had almost no NaN cases — no acquired tickers in
the universe). On the augmented panel it would have been catastrophic
and dishonest — the strategy would have shown -96% Max DD on every
acquisition wave, which is not what actually happens to a holder.

I patched the augmented runs to book **0%** on NaN returns (the
honest interpretation: cash payout at last-known price). This is the
minimum-bias correction. A more sophisticated treatment would look
up the actual acquisition-close price; that's out of scope here.

If you re-run with NaN→-1.0 (the unpatched production behavior), v5
shows 9% CAGR / 4% WF mean. That number is artifactually low. The
NaN→0 patched run (33% CAGR / 33% WF mean) is the honest one.

## Caveats

1. **Simulation runs 254 (v5) / 250 (v3) months** vs 268 in the
   original. The PIT panel has slightly less feature coverage in the
   earliest months (some backfilled tickers don't have 504 days of
   history before 2003-01).
2. **213 OTC bankruptcy-Q tickers remain unreachable.** These
   contribute the worst delisting outcomes. The augmented numbers
   here are still an upper bound on true PIT performance.
3. **Held-position pricing for genuine delistings** (not acquisitions):
   the rare cases where a ticker is bankrupt and shareholders get $0
   would book 0% under our patch but ~-100% in truth. These are a
   small minority for S&P 500 names (companies usually leave the index
   before bankruptcy).

## Files

Scripts:
- `experiments/monthly_dca/v5/spx_pit/build_sp500_pit_prices.py`
- `experiments/monthly_dca/v5/spx_pit/build_monthly_clean_pit.py`
- `experiments/monthly_dca/v5/spx_pit/cache_features_pit.py`
- `experiments/monthly_dca/v5/spx_pit/add_alpha_features_pit.py`
- `experiments/monthly_dca/v5/spx_pit/build_panel_pit.py`
- `experiments/monthly_dca/v5/spx_pit/train_ml_pit.py`
- `experiments/monthly_dca/v5/spx_pit/build_sp500_pit_panel_aug.py`
- `experiments/monthly_dca/v5/spx_pit/score_chronos_aug.py`
- `experiments/monthly_dca/v5/spx_pit/run_v3_winner_aug.py`
- `experiments/monthly_dca/v5/spx_pit/run_v5_winner_aug.py`
- `experiments/monthly_dca/v5/spx_pit/run_pit_filter_backtest.py`

Augmented outputs (under `experiments/monthly_dca/cache/v2/sp500_pit/augmented/`):
- `monthly_prices_clean.parquet`, `monthly_returns_clean.parquet`
- `ml_preds.parquet` (walk-forward GBM, 405k rows)
- `ml_preds_chronos.parquet` (Chronos-bolt-tiny p50/p70/p90, 104k rows)
- `sp500_pit_panel.parquet` (joined panel)
- `v3_winner_summary.json` + `v3_winner_walkforward.csv` + `v3_winner_equity.csv`
- `v5_winner_summary.json` + `v5_winner_walkforward.csv` + `v5_winner_equity.csv`
- `sp500_pit_filter_*` (the k=15 v3-baseline reference)

## Reproduction

```bash
# Phase 1: build augmented daily panel (~5 min; downloads FNSPID ~590 MB)
python3 experiments/monthly_dca/v5/spx_pit/build_sp500_pit_prices.py

# Phase 2: monthly clean panels (~30 s)
python3 experiments/monthly_dca/v5/spx_pit/build_monthly_clean_pit.py

# Phase 3a: 21-col base features (~65 min)
python3 experiments/monthly_dca/v5/spx_pit/cache_features_pit.py

# Phase 3b: layer alpha + alpha2 + extra + novel features to reach 79 cols
# (~30 min on first run; idempotent after that)
python3 experiments/monthly_dca/v5/spx_pit/add_alpha_features_pit.py

# Phase 3.5: cross-section
python3 experiments/monthly_dca/v5/spx_pit/build_panel_pit.py
python3 experiments/monthly_dca/v5/spx_pit/build_sp500_pit_panel_aug.py

# Phase 4: walk-forward GBM (~9 min on CPU)
python3 experiments/monthly_dca/v5/spx_pit/train_ml_pit.py

# Phase 5a: Chronos-bolt-tiny on augmented panel (~1.5 min)
pip install torch chronos-forecasting
python3 experiments/monthly_dca/v5/spx_pit/score_chronos_aug.py

# Phase 5b: deployed v3-winner with augmented inputs (~30 s)
python3 experiments/monthly_dca/v5/spx_pit/run_v3_winner_aug.py

# Phase 5c: deployed v5-winner with augmented inputs (~30 s)
python3 experiments/monthly_dca/v5/spx_pit/run_v5_winner_aug.py
```

Total wallclock from a clean checkout: ~2 hours, mostly Phase 3a (the
65-min base feature compute on a CPU).
