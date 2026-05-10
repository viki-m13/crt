# Repo map — main stock-selection strategy

Date: 2026-05-10. Branch: `claude/rebuild-stock-selection-2qHxY`.

## TL;DR — what's deployed on the main page

The main page (`docs/index.html`, served at `dailystockguide.com`) advertises:

> ML-driven 3-stock S&P 500 basket rebalanced every 6 months. Walk-forward
> Gradient Boosted Trees + crash-aware regime gate, true point-in-time S&P 500
> universe. **42.80% mean OOS CAGR** across 10 walk-forward splits, 10/10
> positive, 9/10 beat SPY. Validated 2003–2025.

This is **V3-PIT-SP500** (`strategy_version: "v3-pit-sp500"` in
`experiments/docs/monthly-dca/data.json`). Headline metrics from
`experiments/monthly_dca/cache/v2/sp500_pit/v3_winner_summary.json`:

| Metric                    | Value     |
|---------------------------|----------:|
| n_months (full)           | 268       |
| Final equity (start=1)    | 1693.33   |
| **CAGR full (2003-09→2025-12)** | **39.77%** |
| SPY CAGR same window      | 11.94%    |
| Edge full                 | +27.83pp  |
| Sharpe                    | 0.96      |
| MaxDD                     | -49.83%   |
| n_cash_months             | 4         |
| **WF mean CAGR (10 splits)** | **42.80%** |
| WF median / min / max     | 39.90% / 14.49% / 108.79% |
| WF mean edge vs SPY       | +27.99pp  |
| WF n positive / beats-SPY | 10 / 9    |
| Annualised turnover       | ~1.46×    |

Per-split TEST CAGR (`v3_winner_walkforward.csv`):

| Split    | window         | CAGR    | edge    | sharpe | maxDD   |
|----------|----------------|---------|---------|--------|---------|
| A1       | 2011-2018      | 22.88%  | +8.80   | 0.90   | -35.4%  |
| A2       | 2015-2021      | 35.37%  | +20.66  | 0.89   | -35.4%  |
| A3       | 2018-2024      | 38.95%  | +24.20  | 0.90   | -35.4%  |
| R1_GFC   | 2008-2010      | 108.79% | +108.75 | 1.25   | -47.5%  |
| R2       | 2011-2013      | 43.13%  | +27.50  | 1.38   | -21.3%  |
| R3       | 2014-2016      | 14.49%  | -1.52   | 0.73   | -15.0%  |
| R4       | 2017-2019      | 19.60%  | +6.55   | 0.76   | -35.4%  |
| R5_COVID | 2020-2022      | 62.20%  | +56.56  | 1.02   | -30.0%  |
| R6_AI    | 2023-2024      | 40.85%  | +4.90   | 1.35   | -12.5%  |
| STRICT   | 2021-2024      | 41.75%  | +23.55  | 1.12   | -30.0%  |

Only one split clears triple-digit OOS: R1_GFC (108.79%), driven by the
2009 rebound. The user's **stretch ask of triple-digit OOS WF CAGR**
sits roughly 2.5× above the current mean — see `01_engine_audit.md`
for feasibility discussion.

## Pipeline (what the deployed strategy actually does)

```
docs/index.html  (main page)
  └─ docs/monthly_dca.js  (renderer)
      └─ experiments/docs/monthly-dca/data.json  (strategy output)
          └─ experiments/monthly_dca/v2/build_webapp_v3_pit.py  (builder)

Strategy data flow (offline, monthly):
  prices_extended.parquet  (1995-01 → 2026-05, 1833 tickers, daily OHLC)
   └─ monthly_returns_clean.parquet  (bad-month-mask filtered)
   └─ cache/features/{YYYY-MM-DD}.parquet  (67 features per asof, PIT)
       └─ ml_preds_v2.parquet  (walk-forward GBM preds, 372k rows)
           └─ sp500_pit/v3_winner_*.csv  (per-split test, full-window, etc.)

Engines:
  experiments/monthly_dca/v6/lib_engine.py  ← canonical (parity with v3)
  experiments/monthly_dca/backtester.py      (older, used to build features)
  experiments/monthly_dca/v2/ml_strategy.py  (walk-forward fit)
  experiments/monthly_dca/v2/build_sp500_pit_membership.py  (PIT membership)
```

## Strategy spec (V3 winner)

- **Universe**: PIT S&P 500. 985 unique tickers ever-in-index, 280 monthly
  asofs from 2003-01-31. Built strictly point-in-time from
  `sp500_hist_1996_2019.csv` + `sp500_changes_since_2019.csv`.
- **Features**: 67 per ticker per asof, all price-only (momentum, trend,
  recovery, RSI, vol, drawdowns). Each feature value at asof T uses data
  with index ≤ T. Cross-sectional ranks `_xs` rebuilt per asof.
- **Model**: HistGradientBoostingRegressor on `fwd_3m_ret` and
  `fwd_6m_ret`. Score = mean of the two predictions. Annual retrain in
  January. **7-month embargo** (training = `asof < tm - 7 months`).
- **Pick**: top-K=3 by ML score. Equal weight. Hold 6 months.
- **Crash gate (`tight`)**: if SPY 21d ret ≤ -8% OR (SPY 6m mom ≤ -5% AND
  SPY 21d ret ≤ -3%) → cash next month. Triggered 4 times in 268 months.
- **Cost**: 10 bps round-trip applied to every changed pick.
- **Smart turnover**: re-pick every 6 months; if a name is in both old and
  new basket, hold (no sell+buy). Approx annualised turnover 1.46×.

## Variants on disk (state of prior research)

`experiments/monthly_dca/v{2,4,5,6,7}/` plus the root experiments folder
holds extensive prior work (60+ research scripts). Each version:

- **v2**: original PIT-SP500 ML pipeline (the deployed one). Sweep,
  walk-forward, retrain, generalise on broader universes.
- **v3**: monthly-rebalance compounding engine over price-only features
  (different track from v2; see `experiments/monthly_dca/REPORT.md`,
  superseded by v2 PIT for production).
- **v4**: simulator-knob sweeps (no headline win).
- **v5**: orthogonal price-pattern features incl. Chronos-bolt time-series
  embeddings. Concluded **price-pattern signals do not Pareto-improve v3**.
- **v6**: inv-vol weighting + 3% cash-yield credit. **Pareto-improves v3
  on Sharpe (0.97 vs 0.96), MaxDD (-46% vs -50%), WF min CAGR (+6.4pp).**
  WF mean essentially tied at 42.48% vs 42.80%. Better than v3 in 6/8
  generalisation universes. **Engine of record.**
- **v7**: aggressive downside protection (daily stops, conditional drawdown
  insurance via SH overlay, 10% TLT sleeve). Cuts MaxDD to -29% but
  drops CAGR to 29.6% (full) / 32.6% (WF mean). Honest trade-off, not a
  Pareto win. Currently NOT on the main page.

The main page still serves V3. V6 is the de-facto best-of-class and the
engine to extend.

## Files that consume / produce strategy outputs

Build (offline cron, pre-deploy):
- `experiments/monthly_dca/v2/build_sp500_pit_membership.py`  PIT membership
- `experiments/monthly_dca/backtester.py:compute_features`     67-feature panel
- `experiments/monthly_dca/v2/ml_strategy.py:fit_walkforward`  GBM preds
- `experiments/monthly_dca/v2/build_webapp_v3_pit.py`          assemble data.json
- `experiments/monthly_dca/cron_daily_refresh.py`              daily refresh

Serve (production):
- `experiments/docs/monthly-dca/data.json`   ← strategy output
- `docs/monthly_dca.js`                       ← page renderer
- `docs/index.html`                           ← main page

## Other strategies in the repo (out of scope here)

- `strategies/touch_predict/`, `strategies/stillpoint/`,
  `strategies/credit_spread/` — separate strategies, separate pages.
- `crypto/`, `spreads/`, `max/` — separate sub-sites.
- `scripts/daily_scan.py`, `scripts/api_server.py` — older "rebound
  scanner" strategy on `/screener`. Different page.

## Bias-risk hot-spots (where leakage / survivorship can creep in)

1. **PIT membership**: only 2003-2025 has confirmed monthly PIT membership;
   pre-2003 is sparse. Anything claiming pre-2003 OOS on S&P 500 PIT is
   suspect — check it uses the actual PIT membership file.
2. **Universe extension**: `prices_extended.parquet` includes 1,833 tickers
   that *exist today* in some form. The non-S&P 500 universes are
   survivorship-biased — a ticker had to survive to be in there. The MC
   bias overlay (`v3_winner_bias_sensitivity.csv`) tries to model this but
   it's a model, not a dataset.
3. **Earnings/fundamentals**: not used. Price-only. So no earnings-release
   lag bug — but also no earnings-revisions edge.
4. **Execution price**: monthly close-to-close. The "implicit fill" is the
   month-end close. Not next-day open, not VWAP. For a monthly strategy
   this is acceptable; `cost_bps=10` is meant to absorb the gap. For
   higher-frequency variants this would need to change.
5. **Sign-flipped feature** (already-fixed in v6): the v3 `dd_from_52wh`
   was stored as positive magnitude but consumed as signed; the
   `regime_strict_dd` branch never triggered. v3-deployed `tight` gate
   doesn't use this feature so the deployed numbers are unaffected.
   v6 corrected the sign on load.
