# Engine audit — leakage, survivorship, execution

Date: 2026-05-10. Engine of record:
`experiments/monthly_dca/v6/lib_engine.py` (parity-tested against deployed
v3; reproduces V3 metrics exactly).

## Findings

### ✓ Point-in-time S&P 500 membership
- `experiments/monthly_dca/v2/build_sp500_pit_membership.py` reconstructs
  monthly index membership from `sp500_hist_1996_2019.csv` (Bloomberg /
  CRSP-style historical lists) + `sp500_changes_since_2019.csv` (manually
  curated additions/removals).
- 985 unique tickers ever in index across 280 monthly asofs (2003-01 →
  2026-04). Verified by `sp500_membership_count.csv`.
- Each rebalance T uses `mem.asof == T`, **not today's index**.
- Source files committed in `experiments/monthly_dca/cache/v2/sp500_pit/`.

**Caveat.** Pre-2003 PIT membership is not maintained, so any backtest
window starting before 2003 must use the broader 1,833-ticker universe
(survivorship-biased; see below).

### ⚠ Survivorship — partly handled
- Inclusion: `monthly_returns_clean.parquet` has 1,833 tickers including
  many that delisted (`delisted_panel.parquet` exists). Good.
- Returns from delisted names: tickers with bad/incomplete months get
  masked by `bad_month_cells_mask.parquet`. Returns sourced from
  Yahoo Finance via `yfinance` (in `extend_history.py`).
- **Gap**: Yahoo's coverage of delisted names is incomplete. There is no
  proper CRSP/Norgate delisted-with-final-return source in the repo. The
  Monte-Carlo overlay (`v3_winner_bias_sensitivity.csv`, α∈{0..20}%/yr)
  is a *model* of delisting, not measured delisted returns. Honest
  bias-corrected CAGR at α=4%/yr is **28.6%** (full-window v3),
  vs the headline 39.8%.
- Decision needed (see scoping questions): is α=4% MC overlay acceptable,
  or should we pull in a proper delisted-with-final-return dataset
  (Norgate, Sharadar, CRSP) before any "honest" claim?

### ✓ Walk-forward / no look-ahead in features
- All features per `cache/features/{date}.parquet` use only data with
  index ≤ asof (verified in `backtester.py:compute_features`).
- Cross-sectional ranks `_xs` are computed *within* an asof, no leakage.
- ML walk-forward in `ml_strategy.py:200-238`:
  ```python
  cutoff = tm - pd.DateOffset(months=embargo_months)   # embargo=7
  train = big[big["asof"] < cutoff]
  ```
- Targets are 1m / 3m / 6m forward returns. With 6m max horizon and 7m
  embargo, the most-recent training row's target ends ≤ test month T-1m.
  **Embargo is correct — no target leakage.** (Strictly, the training
  cutoff is on `asof`, and a row with `asof = T-7m` has target ending at
  `T-7m + 6m = T-1m`, which is *before* T. Safe.)
- Annual retrain (Jan), so January-T's model was fit on data ending
  ≈Jun T-1.

### ⚠ ML target lookahead — minor risk in cross-sectional rank features
The 67 features include rank transforms grouped by `asof`. Those are
fine. But several momentum windows (e.g., `mom_12_1`, `mom_6_1`) skip the
final month (`_1`) — common practice to avoid 1-month reversal. Make
sure any new features follow the same convention.

### ⚠ Execution price — month-end close, no slippage model
- The simulator uses `monthly_returns_clean.parquet` directly. Pick at T,
  realise return = month T to T+1 (close-to-close).
- 10 bps cost model is a flat round-trip charge per ticker that changes
  between baskets. Not scaled to ADV, not a slippage model.
- For monthly strategies on S&P 500 names this is acceptable; for any new
  candidate at higher frequency or smaller-cap universe this needs
  upgrading (next-day-open or VWAP fill, ADV-scaled slippage).
- **Note**: end-of-month close has nuanced "last 30 minutes" liquidity
  characteristics. A more honest fill is "next-day open" or VWAP.

### ✓ No fundamentals release-lag bug (because no fundamentals)
- Price-only strategy. So no period-end vs filing-date lag risk.

### ✓ Regime gate is PIT-clean
- Uses SPY's own price features at asof T only (`spy_ret_21d`,
  `spy_mom_6_1`, `spy_dsma200`, `spy_below_200_streak`, `spy_mom_12_1`).
  No outside data, no future data.

### ✓ One bug found and fixed in v6
- v3 stored `dd_from_52wh` as a positive magnitude
  (`backtester.py:246: pack.add("dd_from_52wh", -pullback_252)`); v3's
  `regime_strict_dd` branch tested `dd <= -0.10` and could never fire
  → it was a no-op. v3-deployed used the `tight` gate which doesn't
  reference this field, so deployed numbers are unaffected. v6 corrects
  the sign on load. **No action needed.**

### ✓ Deterministic / reproducible
- HistGBM uses `random_state` set in `ml_strategy.py`.
- Engine is pure-pandas / numpy, no hidden randomness.
- v6 `run_baseline.py` reproduces v3 numbers exactly:
  `cagr_full=0.39774062, sharpe=0.95536375, max_dd=-0.49828619`.

## Verdict

**Engine is honest enough to extend, with two known caveats:**

1. Survivorship: the bias overlay (MC delisting at α=4%/yr) is reasonable
   but a proper delisted-with-final-return dataset would harden any
   claim. **Surface to user before final validation.**
2. Execution: month-end close fills + flat 10 bps work for monthly S&P
   500 strategies. Any new candidate at higher frequency or smaller-cap
   universe needs an upgraded fill+slippage model. **Surface if scope
   expands.**

No blocking leakage was found. Walk-forward is correctly embargoed.
Targets do not leak into training. PIT membership is correctly
constructed. The 42.80% WF mean CAGR figure is honestly produced under
the documented assumptions.
