# Daily Stock Guide — research repository

Research, backtests, and infrastructure for the
[dailystockguide.com](https://dailystockguide.com) deployed strategy
(currently **v5** — a Chronos-filtered GBM ranking strategy on a
point-in-time S&P 500 universe). This repo is the data layer, ML
training, validation gauntlets, web frontend, and assorted experiments.

## Quick links

| Topic | Entry point |
|---|---|
| The deployed strategy (v5) | [`experiments/monthly_dca/v5/`](experiments/monthly_dca/v5/) |
| Point-in-time S&P 500 dataset | [`data/sp500_pit/README.md`](data/sp500_pit/README.md) |
| Engine audit (leakage, survivorship) | [`research/01_engine_audit.md`](research/01_engine_audit.md) |
| Final validation (v8 exp_02 winner) | [`reports/final_validation.md`](reports/final_validation.md) |
| PIT v5 validation report | [`experiments/monthly_dca/v5/spx_pit/REPORT.md`](experiments/monthly_dca/v5/spx_pit/REPORT.md) |
| Rebalance-timing-luck analysis | [`experiments/monthly_dca/v5/spx_pit/TIMING_LUCK.md`](experiments/monthly_dca/v5/spx_pit/TIMING_LUCK.md) |
| Data layer overview | [`data/README.md`](data/README.md) |

## Honest performance numbers

The deployed v5 strategy (as of 2026-05-17) is **E1**
(`v5_pit_sp500_E1_win1_win2_5050_chronos_p70_k2_invvol_cap0.4_minhold6_scoredrift`)
— a **50/50 portfolio of two 2-stock sleeves** of the same GBM+Chronos
alpha (K=2, Chronos p70 filter, inverse-vol cap 0.4, rule-based
rebalance min-6m/score-drift) that swap which scorer drives stock
**selection** vs rebalance **timing**: Sleeve A/WIN1 (select=ml_3plus6,
trigger=blend) + Sleeve B/WIN2 (select=blend, trigger=ml_3plus6). The
sleeves rebalance on different dates so their 2-stock variance
decorrelates while the shared alpha compounds; the combined live book
is up to 4 names. Validated on the augmented PIT panel with the
canonical production sim — see
[`experiments/monthly_dca/v5/spx_pit/IMPROVE_FINDINGS.md`](experiments/monthly_dca/v5/spx_pit/IMPROVE_FINDINGS.md).

Canonical production sim (augmented PIT 2003–2025, 10 bps): the prior
single-sleeve WIN1 vs the now-deployed **E1** two-sleeve portfolio:

| | WIN1 (prior) | **E1 (deployed)** |
|---|---:|---:|
| Full-window CAGR (lump-sum) | 51.9% | **51.9%** |
| Sharpe (monthly)            | 1.01  | **1.03** |
| Max DD (accumulating DCA)   | -64%  | **-56%** |
| WF splits beating SPY (CAGR)| 10/10 | **10/10** |
| Non-overlapping eras beating S&P-DCA | 4/4 | **4/4** |
| Rolling 10y / 5y / 3y DCA-win | 100% / 99% / 86% | **100% / 99% / 91%** |
| Worst rolling 5-yr DCA CAGR | +2.5%/yr | **+11.9%/yr** |

**E1** is a strict improvement on single-sleeve WIN1: same CAGR,
higher Sharpe, ~8–10pp shallower interim drawdown, WF 10/10, all four
eras beat S&P-DCA, and the *worst* rolling 5-year DCA outcome roughly
doubles — i.e. the same edge delivered far more consistently. It is
cost-insensitive, sits on a wide 50/50 mix-weight plateau (0.3–0.7),
is more delisting-robust than either sleeve alone, and has the
strongest truly-out-of-sample holdout of any variant (untouched
2013–2026 Sharpe 1.08). The edge is still heavily front-loaded in
2003–2009 and the interim drawdowns are still deep — narrowed, not
removed.

The PIT correction adds 161 acquired/renamed large-caps (AGN, ANTM,
ABMD, CELG, ATVI, AET, …) that the original v2 panel omitted. See
[`data/sp500_pit/`](data/sp500_pit/) for the full dataset and
methodology, and
[`experiments/monthly_dca/v5/spx_pit/IMPROVEMENTS.md`](experiments/monthly_dca/v5/spx_pit/IMPROVEMENTS.md)
for the K=3→K=2 sweep, MC delisting validation, cross-universe
generalization (NDX 8/8 beats QQQ), and the rule-based rebalance
sweep.

## Repository layout

```
crt/
├── README.md                                ← this file
├── api/                                     Vercel serverless API endpoints
├── data/                                    Top-level data hub (entry-point READMEs)
│   ├── README.md                            data layer overview
│   └── sp500_pit/                           PIT S&P 500 dataset (May 2026)
│       └── README.md
├── experiments/                             experiments by family
│   └── monthly_dca/                         the production strategy family
│       ├── cache/                           cached prices, features, predictions
│       │   ├── prices_extended.parquet        daily prices (1833 tickers, the
│       │   │                                  original biased panel)
│       │   ├── features/                      per-month 79-col feature snapshots
│       │   └── v2/sp500_pit/                  PIT membership + PIT-corrected
│       │       │                              dataset and outputs (see data/sp500_pit/)
│       │       ├── sp500_membership_monthly.parquet
│       │       ├── prices_extended_pit.parquet
│       │       └── augmented/                 PIT-corrected outputs
│       ├── v2/                               GBM walk-forward training
│       ├── v4/, v6/, v7/, v8/                successive validation gauntlets
│       └── v5/                               currently deployed strategy
│           ├── score_*.py                    Chronos / time-series forecasters
│           ├── build_webapp_v5_pit.py        production v5 webapp builder
│           ├── score_winner_v5.py            today-only scorer for live picks
│           └── spx_pit/                      May 2026 PIT validation work
│               ├── REPORT.md                 full PIT validation report
│               ├── build_sp500_pit_prices.py Phase 1: augmented daily panel
│               ├── build_monthly_clean_pit.py Phase 2: monthly clean
│               ├── cache_features_pit.py     Phase 3a: base features
│               ├── add_alpha_features_pit.py Phase 3b: alpha+novel features
│               ├── build_panel_pit.py        Phase 3.5: cross-section
│               ├── build_sp500_pit_panel_aug.py Phase 5a: joined panel
│               ├── train_ml_pit.py           Phase 4: GBM retrain
│               ├── score_chronos_aug.py      Phase 5c: Chronos on augmented
│               ├── run_pit_filter_backtest.py Phase 5: k=15 baseline
│               ├── run_v3_winner_aug.py      Phase 5b: deployed v3
│               └── run_v5_winner_aug.py      Phase 5d: deployed v5
├── reports/                                 polished reports
│   ├── executive_summary.md
│   └── final_validation.md                  exp_02 winner (PIT update May 2026)
├── research/                                research notes / audits
│   └── 01_engine_audit.md                   leakage + survivorship audit
├── strategy/                                shared strategy / feature library
│   └── features/novel_features.py           12 novel features (cst, rbi, …)
├── strategies/                              other strategy families
├── crypto/, max/, spreads/                  separate strategy families
└── docs/                                    long-form research docs
```

## How to reproduce the PIT validation

```bash
# 1) Augmented daily panel (~5 min; downloads FNSPID ~590 MB)
python3 experiments/monthly_dca/v5/spx_pit/build_sp500_pit_prices.py

# 2) Monthly clean panels
python3 experiments/monthly_dca/v5/spx_pit/build_monthly_clean_pit.py

# 3a) Base features (~65 min on CPU)
python3 experiments/monthly_dca/v5/spx_pit/cache_features_pit.py

# 3b) Alpha + novel features (~30 min on CPU, idempotent)
python3 experiments/monthly_dca/v5/spx_pit/add_alpha_features_pit.py

# 3.5) Cross-section dataset
python3 experiments/monthly_dca/v5/spx_pit/build_panel_pit.py
python3 experiments/monthly_dca/v5/spx_pit/build_sp500_pit_panel_aug.py

# 4) Walk-forward GBM (~9 min on CPU)
python3 experiments/monthly_dca/v5/spx_pit/train_ml_pit.py

# 5a) Chronos-bolt-tiny on augmented panel (~1.5 min)
pip install torch chronos-forecasting
python3 experiments/monthly_dca/v5/spx_pit/score_chronos_aug.py

# 5b) Deployed v3-winner with PIT correction
python3 experiments/monthly_dca/v5/spx_pit/run_v3_winner_aug.py

# 5c) Deployed v5-winner (with Chronos) with PIT correction
python3 experiments/monthly_dca/v5/spx_pit/run_v5_winner_aug.py
```

Total wall-clock: ~2 hours from a clean checkout, mostly Phase 3a
(the base-feature compute).

## Deployment

The web frontend is at [dailystockguide.com](https://dailystockguide.com).
Deployed via Vercel using `vercel.json` and the serverless functions
in `api/`. Daily refresh is handled by
`experiments/monthly_dca/v5/cron_daily_refresh_v5.py`.

## License

Internal research repository. Third-party data:
- yfinance: Yahoo Finance's terms of service apply.
- FNSPID dataset: CC BY-NC 4.0 (research / non-commercial use).
- fja05680/sp500: MIT.
- Chronos-bolt-tiny: Apache 2.0.
