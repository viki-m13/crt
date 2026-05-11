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

### ⚠ Survivorship — partly handled, now empirically measured (May 2026)
- Inclusion: `monthly_returns_clean.parquet` has 1,833 tickers including
  many that delisted (`delisted_panel.parquet` exists). Good.
- Returns from delisted names: tickers with bad/incomplete months get
  masked by `bad_month_cells_mask.parquet`. Returns sourced from
  Yahoo Finance via `yfinance` (in `extend_history.py`).
- **Gap, quantified**: Yahoo's coverage of delisted names was incomplete.
  The original v2 panel had only **611 of the 985 unique PIT S&P 500
  tickers** (51% coverage in 2003, rising to 96% in 2025). The 374
  missing names were mostly acquired or bankrupt companies whose tickers
  Yahoo retired.

#### Augmented PIT panel + empirical validation (May 2026 work)

See [`data/sp500_pit/`](../data/sp500_pit/README.md) and
[`experiments/monthly_dca/v5/spx_pit/REPORT.md`](../experiments/monthly_dca/v5/spx_pit/REPORT.md)
for full detail.

Backfilled **161 acquired/renamed large-caps** from FNSPID (Hugging Face
`Zihan1004/FNSPID`, CC BY-NC) + yfinance, date-validated against PIT
membership to filter ticker-reuse traps. Coverage lifts to **72% in
2003 → 99.7% in 2025**. 213 OTC bankruptcy-Q tickers (AAMRQ, LEHMQ, etc.)
remain unreachable on free data.

End-to-end pipeline re-run on augmented panel (Phase 1-5d in
[REPORT.md](../experiments/monthly_dca/v5/spx_pit/REPORT.md)):
re-extracted daily prices → resampled to monthly clean → recomputed all
79 features → retrained GBM walk-forward → regenerated Chronos
forecasts → re-ran v3-winner AND v5-winner backtests.

**Empirical PIT-corrected numbers** (NaN-on-acquisition treated as 0%
cash payout, not -100%, since the majority of the backfilled names were
acquired-at-premium, not bankrupt):

|                    | Original (biased) | Augmented (PIT) |        Δ |
|--------------------|------------------:|----------------:|---------:|
| **v5 WF mean CAGR**|        **47.16%** |      **32.68%** | **-14.5pp** |
| v5 Full CAGR       |            43.86% |          32.92% |  -10.9pp |
| v5 WF beats SPY    |             10/10 |            8/10 |       -2 |
| v3 WF mean CAGR    |            42.80% |          25.78% |  -17.0pp |
| v3 Full CAGR       |            39.77% |          31.81% |   -8.0pp |

The deployed strategies' 43-47% WF mean CAGR overstates the
PIT-honest number by **14-17pp**. Corrected WF mean is **~26-33% —
still strong, still beats SPY by ~19pp/yr on average, still positive
in 10/10 (v5) and 8/10 (v3) splits**.

**Chronos filter helps**: v5 loses LESS to the PIT correction than v3
(-14.5pp vs -17.0pp). Chronos genuinely protects against picking the
worst acquired/delisted names.

The MC bias overlay (`v3_winner_bias_sensitivity.csv`, α∈{0..20}%/yr,
28.6% CAGR at α=4%) was a reasonable proxy but TOO PESSIMISTIC at
α=4% — the empirical PIT-corrected number lands at 32% (v5) / 26%
(v3 WF mean), notably higher than the MC's 28.6%.

- **The 213 unreachable bankruptcy-Q tickers are the residual gap.**
  These would shave the corrected numbers further (perhaps another
  3-5pp on WF mean). Closing this gap requires CRSP / Sharadar (paid).

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

1. ~~Survivorship: the bias overlay (MC delisting at α=4%/yr) is reasonable
   but a proper delisted-with-final-return dataset would harden any
   claim.~~ **Largely addressed (May 2026):** empirical PIT-corrected
   numbers via the augmented panel — see
   [`data/sp500_pit/`](../data/sp500_pit/README.md). Honest deployed-v5
   WF mean = **32.7% CAGR** (vs original 47.2% claim). 213 OTC bankruptcy
   tickers remain unreachable on free data (residual ~3-5pp uncertainty).
2. Execution: month-end close fills + flat 10 bps work for monthly S&P
   500 strategies. Any new candidate at higher frequency or smaller-cap
   universe needs an upgraded fill+slippage model. **Surface if scope
   expands.**

No blocking leakage was found. Walk-forward is correctly embargoed.
Targets do not leak into training. PIT membership is correctly
constructed. The headline 42.80% (v3) / 47.16% (v5) WF mean CAGR figures
are honestly produced *under the original data assumptions*; the
PIT-corrected numbers per the May 2026 augmentation are 25.78% (v3) and
**32.68% (v5)**.
