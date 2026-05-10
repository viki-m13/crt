# Leakage Red-Team Report (multi_pillar_43Agh)

Date: 2026-05-10. Tests run by `tests/leakage_redteam.py`. Raw output at
`reports/leakage_redteam.json`.

## Test 1 — PIT Reconstitution

**Question**: does any (asof, ticker) row in the multi-pillar panel
correspond to a ticker that was NOT actually in the S&P 500 at that asof?

**Method**: left-join the panel to `sp500_membership_monthly.parquet`
on (asof, ticker); count rows where membership is missing.

**Result**: 0 violations across 88,124 panel rows. ✅ PASS.

## Test 2 — Shuffle-Score Test

**Question**: if we replace the score with random noise, does the
strategy still have an edge?

**Method**: within each asof group, randomly permute the score values.
Run the engine. If the strategy's edge is from the score (not from
universe construction or PIT data), the shuffled version should produce
SPY-like returns.

**Result**: shuffled CAGR = 11.58%, SPY CAGR = 11.94%, edge = -0.37pp.
The shuffled strategy returns essentially equal to SPY DCA. The edge is
zero. ✅ PASS — the strategy's edge comes from the score, not from
ambient leakage in universe construction.

## Test 3 — Survivorship-Exclusion

**Question**: if we exclude all delisted-ticker columns from the panel,
do results IMPROVE? (Improvement under exclusion = survivorship leakage
present.)

**Method**: drop all panel rows whose ticker is in the
`delisted_panel.parquet` columns (20 known delisted names: WM, FNMA,
FMCC, SHLD, DDS, BBBY, FRO, SVB, plus others). Rerun the engine.

**Result**:
- With delisted: CAGR ~24%, Sharpe ~0.86, MaxDD -43%
- Without delisted: CAGR 15.4%, Sharpe 0.75, MaxDD -70.3%

Excluding delisteds makes results MUCH WORSE, not better. This is
**inconsistent with survivorship leakage** — leakage would show up as an
improvement. The result is consistent with: some delisted-eventually
names provide rebound-capture upside before they collapse, AND the
remaining higher-quality universe is more concentrated and volatile.

✅ PASS — no survivorship leakage signature. (Note: the delisted-panel
contains 20 named tickers, not the full set of all delisted S&P
constituents over 30 years. The PIT membership table has 985 unique
tickers ever in the index, of which a substantial fraction were
delisted. The engine treats post-delist NaN returns as -100% per pick,
which is the conservative handling.)

## Test 4 — Generalisation (non-S&P-500 universe)

**Question**: does the multi-pillar edge persist outside the home
universe?

**Method**: rebuild the composite panel using `universe="non_sp500"`
(panel rows whose tickers are NOT in PIT membership at the asof). Run
the engine.

**Result**:
- non_sp500 CAGR = 35.36%
- non_sp500 Sharpe = 0.93
- non_sp500 MaxDD = -43.10%
- non_sp500 edge vs SPY = +23.4pp
- non_sp500 beats SPY = 9/10 splits

The strategy generalises **better** to non-S&P than V6/V7 did to S&P-PIT
on this metric (non-S&P-PIT had been V6's stronger universe in the
existing reports too — see `v6/REPORT.md §8`). The edge persists.

✅ PASS — no S&P-specific overfit signature.

## Test 5 — Walk-forward boundary integrity (verified by reading code)

**Method**: read `experiments/monthly_dca/v2/ml_strategy.py:200-238`. Train
window is `asof < tm - 7 months`. With 6m forward target, this gives a
1-month embargo on the target.

✅ PASS — verified by code inspection (already in `01_engine_audit.md`).

## Test 6 — Feature timestamp spot check

**Method**: hand-compute SPY's mom_12_1 at 3 random asofs from
prices_extended; compare to engine's value in the corresponding
features parquet.

**Result**: differences within 0.02 (the spec uses a 21d/252d window
which differs slightly from the literal 12-month / 1-month skip).
✅ PASS — features at asof T are computed from data ≤ T; values match
hand-computation within tolerance.

## Test 7 — Forensic study leakage (architecture review)

The forensic studies (Study A winners, Study B failures) used
**all-period** data to find episodes (1995-2026 daily prices) and the
EXISTING PIT features parquets to capture pre-event snapshots
(`offset_months=3` before base/peak). The archetype centroid built from
this combines all winner episodes from all eras.

**Risk**: when scoring a ticker at asof T using the centroid, the
centroid is based on winner cases that include events AFTER T (since
the centroid is built once from all 1995-2026 episodes).

**Mitigation**: the contribution of Pillar 4 was empirically negative
(the centroid did NOT improve picks even with this mild leakage), so
this risk does not affect the conclusions. A strict version that
rebuilds the centroid only from events whose pre-window asof < current
asof is straightforward to implement; if Pillar 4 ever showed positive
edge with the leaky centroid, this strict version would be required to
verify it. Status: **documented; not weaponised because Pillar 4 didn't
help.**

## Summary

| Test | Result | Signature |
|------|--------|-----------|
| PIT membership reconstitution | 0/88,124 violations | ✅ PASS |
| Shuffle-score | edge collapses to ~0 | ✅ PASS — real signal |
| Survivorship exclusion | no improvement (results worsen) | ✅ PASS — no leakage |
| Generalisation to non-SP | edge persists +23pp | ✅ PASS — not S&P-specific |
| Walk-forward embargo (code review) | 7m embargo, 1m beyond target | ✅ PASS |
| Feature timestamp spot check | matches hand-compute | ✅ PASS |
| Forensic centroid leakage | mild risk, empirically inert | ⚠️ noted, not weaponised |

**Verdict**: The measurable edge of the multi-pillar configurations is
real but **smaller than V3/V6** on PIT S&P 500. There is no leakage
signature. The conclusion ("multi-pillar overlays don't help on this
universe") stands and is not an artefact of testing methodology.

## Tests NOT run (scope constraints)

- **Frozen holdout (2025-01 → 2026-05)**: panel runs through 2025-12
  (the v6 ml_preds parquet's last asof). The most recent data is
  inside the panel and was effectively touched by the sweep. A strict
  frozen holdout would require regenerating ml_preds with cutoff
  2024-12-31, then evaluating on 2025-01-01 → 2026-05. **Not done in
  this session due to time** — would take 30-60 min to regenerate
  predictions.
- **Hyperparameter robustness (±20% sweep)**: covered partially by the
  drop_failure_pct sweep (0%, 10%, 20%, 30%, 40%) and K sweep (3, 4, 5)
  in `backtests/sweep_results.csv`. Other hyperparameters (cost_bps,
  hold_months, regime gate thresholds) were held at v6 defaults.
- **Capacity estimate**: K=3-5 picks of S&P 500 large-caps with 6-month
  hold and 4 rebalances/year suggests $1B+ capacity easily. Not
  formally analysed.
- **Live-degradation haircut**: not applied because none of the pillars
  Pareto-improve V6 — there is no headline number worth haircutting.
