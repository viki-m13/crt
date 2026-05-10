# 00 — Repo Map of the Main Strategy

**Branch**: `claude/rebuild-stock-selection-YLOka`
**Author**: Claude (Phase 0)
**Date**: 2026-05-10
**Scope**: only the headline "3 S&P 500 stocks every 6 months" basket strategy on the front page (`docs/index.html`). Crypto, spreads, screener, "max", and "stillpoint" are out of scope.

## TL;DR

- **Strategy name on the wire**: `v3-pit-sp500`, scorer `ml_3plus6`, K=3 picks, 6-month hold, regime gate (`tight`), equal-weight.
- **Headline claim** (front page): 42.80% mean OOS CAGR across 10 walk-forward splits, 10/10 positive, 9/10 beat SPY DCA.
- **Underlying single OOS run** (full 2003→2026): CAGR 39.77%, Sharpe 0.95, MaxDD -49.83%, ~1.45× annual turnover, 4 cash months across the full period.
- **The "10 splits" are time slices of one OOS simulation, not 10 independent experiments.** This is a real, legitimate audit finding (see §6 and `01_engine_audit.md`).
- **Universe is point-in-time S&P 500** with delisted tickers retained pre-delisting. 985 unique tickers across 1995–2026. ✅
- **ML embargo is 7 months** (training cutoff = test_month - 7m for a 6m forward target). ✅
- **Many things are fine**, but the headline metric framing oversells, and execution uses the same bar (month-end close) for signal and trade — a small structural look-ahead.

## 1. Front-end → backend data path

```
docs/index.html
  └── docs/monthly_dca.js (DATA_URL = "/experiments/monthly-dca/data.json")
      └── experiments/monthly-dca/data.json     (~145 KB, written by build pipeline)
          └── built by experiments/monthly_dca/build_webapp_json_v3.py
              ├── reads cache/v2/sp500_pit/v3_ml_3plus6_summary.json (headline numbers)
              ├── reads cache/v2/sp500_pit/v3_ml_3plus6_walkforward.csv (per-split)
              ├── reads cache/v2/sp500_pit/v3_ml_3plus6_*.csv (yearly, drawdowns, …)
              └── runs experiments/monthly_dca/compound_engine.run_compound() to
                   produce live equity curve + current basket
```

The screener (`docs/app.js` → `docs/data/full.json`) is a different product and not in scope.

## 2. Strategy spec (from `data.json`)

- `scorer`: `ml_3plus6` — mean of 3-month-ahead and 6-month-ahead forward-return ML rank predictions.
- `K_normal = K_recovery = K_bull = 3`; cash if regime = `crash`.
- `weighting`: equal-weight.
- `regime_gate`: `tight`.
- `regime_gate_rule`:
  - **crash** if `SPY 21d ≤ -8%` OR (`SPY 6m ≤ -5%` AND `SPY 21d ≤ -3%`)
  - **recovery** if `SPY below 200dma streak ≥ 40d` AND `SPY back above 200dma` AND `SPY 21d > 0`
  - **bull** if `SPY 12m ≥ 10%` AND `above 200dma`
  - else **normal**.
- `hold_months`: 6.
- `cost_bps`: 10 round-trip (5 bps each direction in `compound_engine`).
- `universe`: PIT S&P 500 members at each rebalance month-end.
- `rebalance_rule`: hold 6 months; reform on month T if `(T - last_rebalance) ≥ 6m` or regime crosses to/from cash.

## 3. Universe construction (PIT S&P 500)

- Built by `experiments/monthly_dca/v2/build_sp500_pit_membership.py`.
- Sources:
  - `cache/v2/sp500_pit/sp500_hist_1996_2019.csv` — daily PIT snapshots 1996–2019.
  - `cache/v2/sp500_pit/sp500_changes_since_2019.csv` — explicit add/remove change events 2019→present.
- For each panel month-end, forward-fill from latest pre-event snapshot; apply explicit changes after the cutoff.
- 976 unique tickers in S&P 500 history; ~500/month; 985 in price panel (S&P plus a small set of always-included names).
- Excluded from "pickable": `SPY, QQQ, IWM, VTI, RSP, DIA, BTC-USD, ETH-USD, TQQQ, SQQQ, UPRO, SPXL, SPXS, TZA, TNA, SOXL, SOXS, FAS, FAZ, TMF, TMV, UGL, GLL, BOIL, KOLD`.

## 4. Price panel

- `cache/prices_extended.parquet` (preferred) or `cache/prices.parquet`.
- 8,133 trading days × 1,833 tickers, 1995-01-03 → 2026-05-07 (per V7 report).
- Almost certainly yfinance + manual CSV stitching; sources unstated in code.
- Crypto and leveraged ETFs explicitly excluded from the picking universe.
- Delisted-ticker price availability is partial; `cache/delisted_panel.parquet` (292 KB) holds the supplement.

## 5. Features (48-dim)

Defined at `experiments/monthly_dca/ml_strategy.py:32–48` and (functionally) for the production model in `experiments/monthly_dca/v2/ml_strategy.py`. Cached at `cache/features/<asof>.parquet` per month-end (228 MB total).

Categories:
- **Momentum**: `mom_12_1`, `mom_6_1`, `mom_3`, `mom_3y`, `mom_2y`, `mom_5y`, `mom_accel`, `mom_consistency_12m`, `accel`, `excess_5y_logret`, `rs_3m_spy`, `rs_6m_spy`, `rs_12m_spy`.
- **Trend / quality**: `trend_health_5y`, `frac_above_50dma_1y`, `sharpe_1y`, `sharpe_12m`, `trend_r2_12m`, `sma50_above_200`, `d_sma200`, `d_sma50`, `trend_slope_252`, `recovery_rate`.
- **Drawdown / volatility**: `pullback_1y/3y/5y/all`, `dd_from_52wh`, `vol_1y/12m/3m/6m`, `vol_contraction`, `vol_expansion_24m`, `rsi_14`, `max_below_200_streak`, `drawdown_age_days`, `bb_width_pct`, `bb_width_contraction`, `dist_from_low_1y`.
- **Range / breakout**: `range_pos_1y`, `near_52wh_60d`, `new_52wh`, `below_52wh`.
- **Tail**: `best_month_24m`, `worst_month_24m`, `tail_ratio_24m`, `multibagger_ratio_24m`, `mean_ret_12m`, `beta_2y`, `log_price`.

All features are backward-looking on the price panel; no fundamentals, no analyst data, no short-interest, no options, no news.

## 6. ML pipeline (production)

`experiments/monthly_dca/v2/ml_strategy.py`:
- `HistGradientBoostingRegressor` × 3 horizons (1m, 3m, 6m forward returns).
- Cross-sectional rank label (per-month percentile of forward return) and cross-sectionally rank-transformed features → "regime-free" model that learns relative orderings.
- **Walk-forward retrain**: every January, refit on rows with `asof ≤ test_month − 7 months` (embargo enforces "the 6-month forward label of training rows ends before the test month"). ✅
- Predictions written to `cache/v2/ml_preds_v2.parquet` (`pred_1m`, `pred_3m`, `pred_6m`, and a blended `pred`).
- Production scorer `s1_ml_3plus6 = (pred_3m + pred_6m) / 2` (`v5/strategies_orthogonal.py:41`).

Note: `experiments/monthly_dca/ml_strategy.py` (the top-level one, 3y target) is a separate research artifact and is NOT what feeds the front page. The Explore-agent map confused these; the right file to start from is the v2 one.

## 7. The "10 walk-forward splits"

Defined in `experiments/monthly_dca/v2/sp500_pit_extended_sweep.py:27–38`:

| Split | From | To | Months |
|---|---|---|---|
| A1 | 2011-01 | 2018-12 | 96 |
| A2 | 2015-01 | 2021-12 | 84 |
| A3 | 2018-01 | 2024-12 | 84 |
| R1_GFC | 2008-01 | 2010-12 | 36 |
| R2 | 2011-01 | 2013-12 | 36 |
| R3 | 2014-01 | 2016-12 | 36 |
| R4 | 2017-01 | 2019-12 | 36 |
| R5_COVID | 2020-01 | 2022-12 | 36 |
| R6_AI | 2023-01 | 2024-12 | 24 |
| STRICT | 2021-01 | 2024-12 | 48 |

**These are time-slices of a single OOS equity curve, not 10 independent train/test experiments.** `sp500_pit_v3_validate.py:62–80` (`per_split_eval`) just slices `eq` by date range and computes window-local CAGR / Sharpe / MaxDD. Periods overlap heavily — e.g. R3 ⊂ A1, R6 ⊂ A3, etc. The model itself IS retrained walk-forward (Jan refit + 7m embargo), but the "splits" are presentational, not experimental.

Per-split TEST CAGR (from `cache/v2/sp500_pit/v3_ml_3plus6_walkforward.csv`):

| Split | CAGR | SPY | Edge | Sharpe | MaxDD |
|---|---:|---:|---:|---:|---:|
| A1 | 22.9% | 14.1% | +8.8 | 0.90 | -35.4% |
| A2 | 35.4% | 14.7% | +20.7 | 0.89 | -35.4% |
| A3 | 38.9% | 14.8% | +24.2 | 0.90 | -35.4% |
| R1_GFC | **108.8%** | 0.0% | +108.7 | 1.25 | -47.5% |
| R2 | 43.1% | 15.6% | +27.5 | 1.38 | -21.3% |
| R3 | 14.5% | 16.0% | -1.5 | 0.73 | -15.0% |
| R4 | 19.6% | 13.0% | +6.6 | 0.76 | -35.4% |
| R5_COVID | 62.2% | 5.6% | +56.6 | 1.02 | -30.0% |
| R6_AI | 40.8% | 36.0% | +4.9 | 1.35 | -12.5% |
| STRICT | 41.8% | 18.2% | +23.6 | 1.12 | -30.0% |
| **mean** | **42.8%** | 14.8% | +28.0 | 1.03 | — |
| median | 39.9% | 14.9% | +21.7 | 0.96 | — |

The mean is dragged up by R1_GFC (108.8%) — buying at a generational bottom and riding the V-shape. Median is 39.9%, close to the full-period CAGR.

## 8. Backtest engine

`experiments/monthly_dca/compound_engine.py`:
- Monthly DCA simulation; reinvests proceeds into equal-weight top-K basket.
- `ExitSpec`: `monthly_rebalance`, `trail_25/35/50`, `hold_forever`, `trail35_or_3y`.
- Positions tracked daily for stops; rebalance triggered monthly.
- Costs applied as `cost_bps` per fill (default 5 bps each leg → ~10 bps round-trip).
- **Execution price = month-end close**, used both for the signal computation and the trade. (See engine audit.)
- No ADV-aware slippage, no borrow cost (long-only), no taxes.
- A v7 daily-resolution variant (`v7/daily_stop_validator.py`) exists for accurate intra-month stop simulation; it's not used by the production headline.

## 9. Strategy lineage (v3 → v7)

- **v1–v2 (legacy)**: pull-back analog matching, k-NN washout meter — front-page screener still uses this lineage.
- **v3 (deployed)**: `ml_3plus6` GBM rank ensemble + `tight` regime gate + K=3 + 6m hold, equal-weight, PIT S&P 500. CAGR 39.77%, Sharpe 0.95, MaxDD -49.83%.
- **v4**: scorer-blend / simulator-knob sweeps; explored `ml_136_blend`, `ml_36_qmom`, `ml_36_low_dd`, `ml_36_v2_3plus6_avg`, etc. None broke past v3 (see `v4/CONCLUSIONS.md`).
- **v5**: orthogonal-strategy ensembles (S1 ml_3plus6, S2 pure_momentum, S3 quality_pullback, S4 breakout, S5 low_vol_quality, S6 multibagger_lottery, S7 idio_winner). Some Sharpe gains, no clean CAGR breakout.
- **v6**: invvol weighting + cash yield → CAGR 38.20%, Sharpe 0.97, MaxDD -45.98% (marginal Pareto vs v3). Documented as confirming v3 near-optimal; ML proprietary-features GBM was substantially worse than v3.
- **v7**: aggressive downside protection — daily stops + dynamic SH (CDI) + 10% TLT sleeve. v7_safer: CAGR 29.6%, Sharpe 1.11, MaxDD -29.0%. Strict trade-off; CAGR cost.

≈600 strategy variants explored across v4–v7 according to v6/v7 reports. Multiple-testing exposure is non-trivial.

## 10. Files ranked by audit importance

1. `experiments/monthly_dca/v2/ml_strategy.py` — production GBM, embargo, regime classifier.
2. `experiments/monthly_dca/v2/sp500_pit_extended_sweep.py` — `WF_SPLITS`, `simulate_variant`, scoring & gate utilities.
3. `experiments/monthly_dca/v2/build_sp500_pit_membership.py` — PIT membership construction.
4. `experiments/monthly_dca/compound_engine.py` — execution, costs, exits.
5. `experiments/monthly_dca/v2/sp500_pit_v3_validate.py` — per-split slicing, drawdown ledger, bias overlay.
6. `experiments/monthly_dca/build_webapp_json_v3.py` — JSON output orchestrator.
7. `experiments/monthly_dca/fast_engine.py` / `fast_score.py` — feature loading, IRR utilities.
8. `experiments/monthly_dca/v2/sp500_pit_bias_overlay.py` — synthetic delisting α-sweep.
9. `experiments/monthly_dca/v6/lib_engine.py`, `v7/lib_engine_v7.py` — successor engines (not deployed).
10. `experiments/monthly_dca/v5/strategies_orthogonal.py` — alt strategies S1–S7.

## 11. Open questions for engine audit (Phase 0 §3)

Tracked in `01_engine_audit.md` — TL;DR:
1. Bar-aligned execution (month-end close used for both signal and fill) — small but real one-day look-ahead.
2. "10 splits" framing → revisit headline metric definition.
3. Confirm delisted-ticker delisting-date accuracy for ~80–100 historical members.
4. Confirm the ML rank target uses cross-sectional rank computed only over **eligible** tickers at each as-of (no peeking at non-PIT names).
5. Confirm SPY regime features are all backward-looking (no contemporaneous month-end data).
6. Confirm cost / slippage realism vs. ADV at 1.45× annual turnover and a typical $1k → $1M deployment.
7. Frozen holdout — none currently exists; final-test windows have been re-tuned across v3–v7. Need to lock one before Phase 4.
