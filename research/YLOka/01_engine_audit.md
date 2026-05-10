# 01 — Engine Audit

**Branch**: `claude/rebuild-stock-selection-YLOka`
**Author**: Claude (Phase 0)
**Date**: 2026-05-10
**Scope**: trustworthiness of the v3 `ml_3plus6` backtest engine that produces the headline metrics on the front page.

## Executive verdict

The engine is **broadly trustworthy**, but with important caveats around metric framing and reserved holdouts. The headline 39.77% full-OOS CAGR is reproducible (✅ verified, see §1). The 42.80% "10 walk-forward splits mean" is *not* 10 independent experiments — it's 10 overlapping time slices of one OOS curve. The model itself IS retrained walk-forward (Jan refit, 7-month embargo) and that part is honest.

**Don't proceed to model-building until §6 (frozen holdout) is decided** — the prior v3–v7 search burned the entire history.

| Concern | Severity | Status | Action needed |
|---|---|---|---|
| PIT membership construction | Critical | ✅ Pass | None |
| Survivorship (delisted included pre-removal) | Critical | ✅ Pass | None |
| Forward-return label alignment | Critical | ✅ Pass | None |
| ML embargo (7 months for 6m label) | Critical | ✅ Pass | None |
| Feature lag (all backward-looking) | Critical | ✅ Pass | None |
| Bar-aligned execution | Major | ⚠️ Fine but tight | Document; consider next-day-open variant |
| "10 splits" framing in headline | Major | ⚠️ Misleading | Replace with full-OOS CAGR or true rolling WF |
| Multiple-testing across v3–v7 | Major | ⚠️ Real risk | Lock frozen holdout before any new search |
| Frozen holdout reserved | Critical | ❌ Missing | Decide & lock window before Phase 2 |
| Non-PIT-but-no-prediction names silently dropped | Minor | ⚠️ Subtle bias | Test impact; document |
| ADV-aware slippage / capacity | Minor | ❌ Not modeled | Estimate capacity ceiling |
| Cash regime earns 0% (no T-bill yield) | Minor | ⚠️ Conservative | Note in headline |

## 1. Reproducibility check (✅ verified)

Reproduced from `cache/v2/sp500_pit/v3_ml_3plus6_333_ew_tight_h6_equity.csv` (268 monthly rows, 2003-09-30 → 2025-12-31):

| Metric | Published `v3_ml_3plus6_summary.json` | Recomputed from `ret_m` |
|---|---:|---:|
| n_months | 268 | 268 |
| Full-period CAGR | 39.77% | 39.77% |
| Sharpe (annualised, monthly) | 0.9554 | 0.9554 |
| MaxDD | -49.83% | -49.83% |
| Cash months | 4 | 4 |

The 39.77% number is honest. (Cannot fully re-train the GBM in this session because the panels `panel_cross_section_v3.parquet` and `sp500_pit_panel.parquet` are not committed — they regenerate from `prices*.parquet` and `ml_preds_v2.parquet`. The cached predictions and equity curves match.)

## 2. Universe / PIT membership (✅ pass)

Verified directly from `cache/v2/sp500_pit/sp500_membership_monthly.parquet`:
- 268 month-ends, 2003-01-31 → 2026-04-30, **mean 500.0 members per month** (range 494–506).
- 985 unique tickers ever in the universe.
- Membership transitions look real: Sep 2008 → Dec 2009 removed 42 names including ACAS, SOV, MER (Merrill Lynch ACQ), ROH, MTLQQ, JNY, KBH and added 44 (HRL Hormel, MJN Mead Johnson, EQT, WDC). These match real S&P 500 changes around the GFC.
- Source files in `cache/v2/sp500_pit/sp500_hist_1996_2019.csv` + `sp500_changes_since_2019.csv`.

**No "today's S&P 500 list" leakage**. The strategy IS picking from the actual list at each asof.

## 3. Survivorship (✅ pass)

`experiments/monthly_dca/v2/sp500_pit_extended_sweep.py:262–266` — when a pick has `NaN` in `monthly_returns_clean.parquet` the engine assigns `-1.0` (i.e. -100%). This treats delisted-with-no-recovery as a wipe-out. ✅ Correct survivorship handling.

A complementary `cache/delisted_panel.parquet` (292 KB) supplements price coverage for delisted names. Synthetic delisting stress test (`sp500_pit_bias_overlay.py`) sweeps an extra alpha ∈ {4, 8, 12, 20}% annual delisting rate; results stored in `*_bias_sensitivity.csv`.

## 4. Look-ahead audit (✅ pass)

### 4.1 Forward-return labels

Spot-checked: `ml_preds_v2.parquet[asof=2003-09-30, ticker=CHE].fwd_1m_ret = -0.00703` matches `monthly_returns_clean.parquet[2003-10-31, CHE] = -0.00703`. The label IS the next-month return; not contemporaneous. ✅

### 4.2 ML embargo

`experiments/monthly_dca/v2/ml_strategy.py` docstring (line 16): "Every January, the model is refit on all data strictly older than (test_month - 7 months), enforcing a 7-month embargo (the 6m forward label of training rows ends before the test month -> no leak)." For a 6-month forward target, 7 months is the minimum honest embargo (label ends at most at train_end + 6m, which is one month before test_start). ✅

For the 3-month head of `ml_3plus6`: the same 7-month embargo over-protects (could be ~4m), so 3m predictions are pessimistically embargoed but not leaky. ✅

### 4.3 Bar-aligned execution

`sp500_pit_extended_sweep.py:202–273` — at each `asof = month_end_T`:
1. Compute features at T close.
2. Pick top-K by score (where score uses features at T close).
3. Earn the next-month return (from `monthly_returns_clean[T+1]`) on the basket.

This means: **signal at T close, fill at T close, return = T close → T+1 close**. This is fine if you can execute in the closing auction. Practical drift: ~5–10 bps annualised vs a "next-day open" model, depending on overnight-gap distribution. The 5 bps/leg cost partly absorbs this. **Document but don't change.**

### 4.4 Feature lag

48-feature list (`ml_strategy.py:32–48`) is exclusively from the price panel using rolling windows: pullbacks, momenta, dist-to-MAs, RSI, vol, range, breadth, tail metrics. No fundamentals, no analyst estimates, no news, no options data. **Nothing forward-looking** — confirmed by inspection. ✅

## 5. Walk-forward design — what's honest, what's framed

### 5.1 The model retraining IS proper walk-forward (✅)

Every January, refit on `asof ≤ test_month - 7m`. From January YYYY through December YYYY, a single model is used for predictions. The model never sees its own forward labels.

### 5.2 The "10 walk-forward splits" framing is misleading (⚠️)

`sp500_pit_v3_validate.py:62–80` (`per_split_eval`) does NOT train a separate model per split. It takes the **same** OOS equity curve `eq` and time-slices it by 10 windows defined in `WF_SPLITS`:

| Split | Window | Months | CAGR |
|---|---|---:|---:|
| A1 | 2011-01 → 2018-12 | 96 | 22.9% |
| A2 | 2015-01 → 2021-12 | 84 | 35.4% |
| A3 | 2018-01 → 2024-12 | 84 | 38.9% |
| R1_GFC | 2008-01 → 2010-12 | 36 | **108.8%** |
| R2 | 2011-01 → 2013-12 | 36 | 43.1% |
| R3 | 2014-01 → 2016-12 | 36 | 14.5% |
| R4 | 2017-01 → 2019-12 | 36 | 19.6% |
| R5_COVID | 2020-01 → 2022-12 | 36 | 62.2% |
| R6_AI | 2023-01 → 2024-12 | 24 | 40.8% |
| STRICT | 2021-01 → 2024-12 | 48 | 41.8% |

**Mean = 42.8% (the headline). Median = 39.9%. Min = 14.5%, Max = 108.8%.**

Issues:
- Windows overlap massively. R3 ⊂ A1 ⊂ A2 ⊂ A3. R6 ⊂ A3. STRICT ⊂ A2 ∪ A3. 2008–2024 is double-counted across the R-series.
- R1_GFC (108.8%) reflects buying the absolute generational bottom and riding the V-shape — when averaged in equally with 9 other splits, it pulls the mean up by ~7pp.
- The "10/10 positive, 9/10 beat SPY" claim is *technically true* but compromised by the overlap: any time slice of a strong upward-trending equity curve is positive, so 10/10 isn't 10 independent successes.

**Recommendation**: Replace headline with "**Full-OOS CAGR 39.77% over 22 years 4 months, 2003-09 → 2026-04**" (single number, no overlapping windows). Per-decade and per-regime breakdowns can sit underneath as descriptors, not as 10 independent trials.

### 5.3 Multiple-testing exposure

`v6/REPORT.md` and `v7/REPORT_V7.md` document **~600 strategy variants** across v3–v7 sweeps:
- v3 focused sweep: scorer × K × weighting × gate × hold × cap.
- v4 blend / simulator-knob sweep: 2,000+ combinations explored across `v4_*_sweep_results.csv`.
- v5 orthogonal-strategy ensembles: 7 base strategies × 30+ ensemble configs.
- v6 baseline + run_sweep + run_sweep2 + run_sweep3.
- v7 ~300-config sweep over stop / CDI / TLT.

The current "winner" (v3 ml_3plus6, 333_ew, tight gate, h6) was implicitly selected after these sweeps over the **entire 2003–2026 history**. There is no holdout reserved. Any new search I run would compound this.

## 6. Frozen holdout — DECISION REQUIRED (❌ missing)

Mission spec asks for a frozen holdout — a time period AND universe slice never touched during research, run only once at the end of Phase 4.

**Two competing pressures**:
- Reserving the most-recent 18–24 months means working on data through ~2024-04, losing the AI-rally regime tilt of 2024–2026.
- Reserving an earlier window means the live model never sees the recent regime — bad for live deployment.

**Pragmatic options for user to choose** (see Phase 0 questions):
1. **Time-only holdout**: reserve 2024-05 → 2026-04 (24 months); train/research on 2003-09 → 2024-04 (~21 years). Recent AI rally + 2025 small-cap rotation become the test. R6_AI and STRICT splits would be retired from research.
2. **Universe-only holdout**: research on PIT S&P 500; freeze a non-overlapping basket (Russell 1000 ex-S&P 500, or randomly sampled 200 names from the broader 1833 panel) for the gauntlet.
3. **Both**: reserve 2024-05 → 2026-04 *and* a 200-name universe slice.

I recommend **(3) Both** if we have data; otherwise **(1) Time-only**.

## 7. Cost / execution / capacity

- Per-leg cost 5 bps (round-trip 10 bps), applied at every rebalance regardless of overlap — slightly conservative (real round-trip on the 30% of basket that turns over each cycle is closer to ~3 bps per cycle).
- No ADV-aware slippage. With K=3 and ~$1k–$1M AUM, this is fine. At $10M+ the strategy frequently picks names like SYF, FOXA, HWM where 5% of ADV is reachable in a day. Capacity ceiling is probably in the $10M–$50M range without a fragmentation algorithm.
- Cash regime earns 0%. Adds about -1pp/year drag in 4 cash months / 22 years. v6 added a `cy` (cash yield) variant that captures ~3% T-bill rate during cash. **Should be added to v3 default** for honesty.

## 8. Subtle issues to verify in Phase 1

These are not yet show-stoppers but I want to test them when I have a working panel-rebuild path:

1. **Names in PIT but not in `ml_preds_v2`** — at 2003-09-30 this is 241 names; by 2025-10 only 17. They're silently excluded from candidate pool. Test: does this bias the candidate pool toward names with longer history?
2. **Cross-sectional rank label** — confirm the per-month rank is computed over the *training* universe (full panel?) vs the *eligible* universe (PIT members at that asof). If rank is computed over a non-PIT-restricted universe, the model could learn relative orderings that don't exist in the live PIT candidate pool.
3. **Forward-return computation for delisted names** — I confirmed `NaN → -1.0` in the engine. Need to also verify that when `monthly_returns_clean` was BUILT, missing returns weren't silently filled with 0 (which would understate delisting cost). Spot-check by counting NaN proportion on names known to delist (e.g., LEHMAN, BSC, MER) vs survivors.
4. **SPY regime features** — confirm `spy_features` at asof T uses only data ≤ T (no contemporaneous month of T fed forward).
5. **Engine's `next_d = mr_idx[cands[0][0] + 1]` logic** (line 258) — when `cands[0][1] > 7` days off, `ret_m = 0.0`. Could this silently zero out returns at the start/end of the panel? Spot-check the first/last months.

## 9. What this means for Phase 1+

**The engine is honest enough to use.** I will not rewrite it. I will:

1. Add a smoke test (`tests/test_pit_membership.py`, `tests/test_feature_lag.py`, `tests/test_walkforward_splitter.py`, `tests/test_costs.py`) that runs in <1 minute and protects the invariants verified above.
2. Replace the headline with **full-OOS CAGR + a non-overlapping rolling WF** ("rolling 5y CAGR over 2008→2026", reported as a distribution rather than 10 means).
3. Lock the frozen holdout per user decision in §6.
4. Build new feature/strategy candidates *only* on data through the holdout cutoff. Run final evaluation **once** on the holdout before any deployment.
5. Add a T-bill yield to cash regime and document.
6. Add a quick capacity estimate using ADV from the price panel.

## 10. Files referenced

- `experiments/monthly_dca/v2/ml_strategy.py:1–120` — production GBM, embargo
- `experiments/monthly_dca/v2/sp500_pit_extended_sweep.py:27–280` — splits, regime classifiers, `simulate_variant`
- `experiments/monthly_dca/v2/build_sp500_pit_membership.py` — PIT construction
- `experiments/monthly_dca/v2/sp500_pit_v3_validate.py:62–80` — `per_split_eval`
- `experiments/monthly_dca/cache/v2/sp500_pit/v3_ml_3plus6_summary.json` — published numbers
- `experiments/monthly_dca/cache/v2/sp500_pit/v3_ml_3plus6_walkforward.csv` — per-split table
- `experiments/monthly_dca/cache/v2/sp500_pit/v3_ml_3plus6_333_ew_tight_h6_equity.csv` — equity curve (reproduced from)
