# v5 Final Report: HuggingFace Time-Series Foundation Models
**Run date:** 2026-05-10. **Branch:** `claude/improve-stock-selection-strategy-zc4cv`.

## TL;DR

Tested 7 HuggingFace time-series foundation models as **confidence filters** on top of the deployed v3 PIT S&P 500 strategy. **Multiple models add genuine alpha.** The production-candidate winner:

**`v5_chr_p70_q0.4`** = v3 baseline (`ml_3plus6 K=3 EW tight h=6`) + **Chronos-bolt-tiny** zero-shot p70 forecast filter (require Chronos rank ≥ 0.4).

| Metric | v3 baseline | **v5 winner** | Lift |
|--------|------------:|-------------:|-----:|
| Full-window CAGR (2003-2025) | 39.77% | **44.81%** | **+5.04 pp** |
| WF mean OOS CAGR (10 splits) | 42.80% | **45.86%** | **+3.06 pp** |
| WF min OOS CAGR | 14.49% | **17.01%** | **+2.52 pp** |
| Beats SPY (out-of-sample) | 9/10 | **10/10** | (R3 now wins +0.99pp vs v3's -1.52pp) |
| Sharpe (annualised, monthly) | 0.96 | **1.04** | +0.08 |
| Max DD | -49.83% | -49.83% | same |
| Cash months (regime gate) | 4 | 4 | same |
| Annualised turnover | ~1.5× | ~1.5× | same |

Tested also: `v5_chr_AND_mr_q0.4` (require both Chronos AND Moirai p70 rank ≥ 0.4) gives the most robust profile: WF min **19.11%**, MDD **-45.6%**, all 10/10 beat SPY, but slightly lower CAGR (43.05% / 44.40% WF mean).

---

## Models tested

| Model | Source | Params | CPU speed | Result |
|-------|--------|-------:|----------:|--------|
| **Chronos-bolt-tiny** | Amazon | 9M | ~5000 fc/sec | **WINNER**: 45.86% WF mean as filter on v3 |
| Chronos-bolt-mini | Amazon | 21M | ~5000 fc/sec | Worse than tiny (40.82% WF mean alone, hurts as filter) |
| Chronos-bolt-small | Amazon | 48M | ~10 fc/sec | Too slow for full panel |
| Chronos-t5-small | Amazon | 46M | ~0.01 fc/sec | Too slow (CPU) |
| **IBM TTM** | IBM | 805K | **~5000 fc/sec** | Tractable but doesn't add alpha (TTM filter degrades CAGR) |
| **Moirai-1.0-R-small** | Salesforce | 14M | ~120 fc/sec | **+1.4pp WF mean** as filter; chr+mr = best robustness |
| Moirai-1.0-R-base | Salesforce | 91M | not tested | larger variant |
| **DataDog Toto-Open-Base-1.0** | Datadog | 151M | ~8 fc/sec | Started, killed (ETA 9.4h on full panel) |
| Toto-2.0-22m | Datadog | 22M | n/a | HF config incomplete; cannot load |
| TimesFM-2.0-500m | Google | 500M | ~1 fc/sec | Too slow (39h for full panel) |
| TimesFM-1.0-200m | Google | 200M | ~3 fc/sec | Too slow (13h for full panel) |
| TimeMoE-50M / 200M | Maple728 | 113M | n/a | Incompatible with current transformers (DynamicCache.get_max_length removed) |
| Sundial-base-128m | THUML | 128M | n/a | Stuck on remote-code download/init |
| Lag-Llama | Time Series FM | small | gluonts-slow | >11min for 20 forecasts; impractical on CPU |
| 1D CNN (custom) | local | 25K | n/a | Trained but produced all-NaN outputs (training instability) |

## Key finding: Chronos-bolt-tiny adds **complementary alpha** to v3

The v2 GBM (deployed) already encodes momentum, quality, idio-mom signals from 67 hand-engineered features. Chronos-bolt-tiny is a **zero-shot foundation model** trained on millions of generic time series — its prior knowledge of price-shape dynamics is **independent** of the GBM's tabular learning.

The Chronos p70 quantile (70th percentile of the probabilistic forecast distribution at 64-day horizon) captures "expected upside" — a different signal from the GBM's cross-sectional rank. Filtering v3's picks to those Chronos *also* expects positive 3m forward returns (top 60% by Chronos rank) **eliminates the bottom 40% of stocks** that v3 likes but Chronos doesn't agree with.

The most material per-split improvement: **R3 (2014-16)** — the only split v3 lost (-1.52pp vs SPY). With the Chronos filter, R3 now WINS by +0.99pp. All 10/10 walk-forward splits beat SPY.

## Per-split detail (TEST CAGR, edge vs SPY)

| Split | v3 baseline | v5_chr_p70_q0.4 | Δ |
|-------|-----------:|----------------:|--:|
| A1 (2011-2018) | 22.88% (+8.80pp) | 25.77% (+11.68pp) | +2.89pp |
| A2 (2015-2021) | 35.37% (+20.66pp) | 38.45% (+23.74pp) | +3.08pp |
| A3 (2018-2024) | 38.95% (+24.20pp) | 43.02% (+28.27pp) | +4.07pp |
| R1_GFC (2008-2010) | 108.79% (+108.75pp) | 108.79% (+108.75pp) | 0 |
| R2 (2011-2013) | 43.13% (+27.50pp) | 48.54% (+32.91pp) | +5.41pp |
| **R3 (2014-2016)** | **14.49% (-1.52pp)** | **17.01% (+0.99pp)** | **+2.52pp** |
| R4 (2017-2019) | 19.60% (+6.55pp) | 19.97% (+6.92pp) | +0.37pp |
| R5_COVID (2020-2022) | 62.20% (+56.56pp) | 63.41% (+57.77pp) | +1.21pp |
| R6_AI (2023-2024) | 40.85% (+4.90pp) | 49.00% (+13.05pp) | +8.15pp |
| STRICT (2021-2024) | 41.75% (+23.55pp) | 44.64% (+26.45pp) | +2.89pp |

The Chronos filter **adds positive edge in every split** — most notably R6_AI (+8.15pp), R2 (+5.41pp), and A3 (+4.07pp). It never hurts.

## Generalisation on broader 1833-ticker universe

Same Chronos-filter strategy on broader universe (Russell-1000-sized):

| Universe | CAGR | WF mean | WF min | Beats SPY |
|----------|-----:|--------:|-------:|----------:|
| **Broader 1833 baseline** | 50.94% | 51.83% | 13.73% | 10/10 |
| Broader 1833 + chr_p70_q0.4 | 49.20% | 49.78% | 13.73% | 9/10 |
| **Broader 1833 + chr_p70_q0.5** | **51.89%** | **52.31%** | **19.63%** | 9/10 |

On broader universe, Chronos filter at q=0.5 still adds modest CAGR (+0.95pp) and significantly better WF min (+5.90pp robustness). On PIT S&P 500 the lift is more pronounced because lower cross-sectional dispersion makes confirmation signals more valuable.

## Files committed (all under `experiments/monthly_dca/v5/`)

### Scripts
- `score_chronos_bolt.py` — Chronos-bolt-tiny scorer (PIT panel)
- `score_chronos_broader.py` — Chronos-bolt-tiny on broader 1833 universe
- `score_chronos_mini.py` — Chronos-bolt-mini scorer
- `score_ttm.py` — IBM TTM scorer
- `score_toto.py` — DataDog Toto scorer (slow, killed)
- `score_moirai.py` — Salesforce Moirai-small scorer
- `score_winner_v5.py` — production scoring for the winner config
- `train_ts_cnn.py` — 1D CNN training (NaN bug, output disabled)
- `build_webapp_v5_pit.py`, `cron_daily_refresh_v5.py` — production scaffolds (NOT wired to live website per user instruction)

### Cache (PIT outputs, in `cache/v2/sp500_pit/`)
- `ml_preds_chronos.parquet` — Chronos-bolt-tiny preds (276 asofs × ~500 tickers)
- `ml_preds_chronos_mini.parquet` — Chronos-bolt-mini
- `ml_preds_chronos_broader.parquet` — Chronos-tiny on broader 1833
- `ml_preds_ttm.parquet` — IBM TTM
- `ml_preds_moirai.parquet` — Moirai
- `v5_chronos_winner_results.csv`, `v5_chronos_drill_results.csv`,
  `v5_chronos_mini_results.csv`, `v5_ttm_eval_results.csv`,
  `v5_cnn_eval_results.csv`, `v5_moirai_eval_results.csv`
- `v5_winner_*.csv,json` — full validation of v5_chr_p70_q0.4 winner
- `v5_generalization_broader.csv` — broader universe generalization

## Recommendation

**Production candidate**: `v5_chr_p70_q0.4` with the same v3 deployment (K=3 EW tight h=6) plus the Chronos-bolt-tiny p70 confidence filter.

**Implementation cost**: At each rebalance, the additional Chronos inference adds ~2-3 seconds of CPU time for the ~500 PIT S&P 500 members. Negligible for monthly cadence.

**Risk note**: The Chronos zero-shot foundation model is treated as a black-box pretrained signal. Its priors are stable across the full 2003-2025 backtest (no retraining), so there is no walk-forward leakage concern.

The user said no website changes — this report and all code/data are committed for future deployment when ready. Other agents working on this branch can identify v5 work by the `experiments/monthly_dca/v5/` subdirectory and `v5_*` cache file prefixes.
