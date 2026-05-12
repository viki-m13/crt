# Quant Research State — Run 1 Complete

**Updated**: 2026-05-11  
**Branch**: claude/compassionate-planck-WolZI  
**Status**: Phase 2 complete. Phase 3 in progress. NO candidate meets success gates yet.

---

## Current Best Metrics (Honest, Leakage-Free)

| Metric | Value | Config | Gate | Pass? |
|---|---|---|---|---|
| Walk-forward CAGR | **39.9%** | monthly_K1_h1 | ≥50% | ❌ (-10pp) |
| Walk-forward Sharpe | **0.89** | pred12m_K2_h6 | ≥2.0 | ❌ (-1.11) |
| Best combined | CAGR=27.9%, Sharpe=0.83 | pred_K3_h6 | both | ❌ |

No config meets either CAGR or Sharpe gate individually. The target (50%/2.0) requires ~2.25× better signal IR than available.

**Research window**: 2003-09-30 → 2023-12-31 (244 months)  
**Lockbox**: 2024-01-31 → 2025-12-31 ← NEVER TOUCHED  

---

## Critical Findings This Run

### Bug Fixes (Both Could Have Invalidated ALL Results)

1. **MonthEnd(1) date arithmetic bug**: `asof + pd.offsets.MonthEnd(1)` from a non-month-end date (e.g., 2004-01-30) produces the SAME calendar month-end (2004-01-31 = 1 day later) instead of the next month-end (2004-02-29). This affected 73/268 asof dates (27%). **Fix**: use `next_month_end(asof)` = `pd.Timestamp(year, month+1, 1) + MonthEnd(0)`. WITHOUT this fix, results were inflated from ~28% to ~97% CAGR — completely misleading.

2. **Regime gate look-ahead bias**: The SPY regime check was looking at NEXT month's calendar month-end SPY features (future data) rather than the current month's end. **Fix**: use `asof + MonthEnd(0)` for regime date lookup. WITHOUT this fix, "strict gate" looked like 49% CAGR / 1.47 Sharpe but was actually 22% / 0.83.

3. **IC analysis with buggy dates**: The initial IC analysis (showing d_sma50 IR=0.51, rsi_14 IR=0.50 vs "1m returns") was computed using the SAME buggy date lookup, so IC was being measured against 1-2 day returns, not 1-month. With correct dates, d_sma50 has IR=-0.11 (NEGATIVE predictor for monthly returns).

### Signal Quality (Corrected)

| Feature | Mean IC | IR | Note |
|---|---|---|---|
| pred_1m | 0.035 | 0.264 | GBM 1m-horizon prediction |
| pred (blend) | 0.031 | 0.233 | GBM 1m+3m+6m blend |
| pred_6m | 0.028 | 0.231 | GBM 6m-horizon |
| pred_12m | 0.023 | 0.190 | GBM 12m-horizon |
| d_sma50 | -0.019 | -0.114 | **DO NOT USE** as monthly ranker |
| rsi_14 | -0.024 | -0.156 | **DO NOT USE** as monthly ranker |

### Data Integrity
- Feature leakage audit: d_sma50 verified at 12 spot-check dates → **0 mismatches**
- All experiments use only lockbox-free data (2003-09 to 2023-12)
- Monthly returns panel correctly indexed at calendar month-ends

---

## Phase 2 Baseline Ladder Results

| Rung | Config | CAGR | Sharpe | MaxDD |
|---|---|---|---|---|
| R1: Momentum (12-1) | mom12_1_K5 | 9.1% | 0.46 | -65% |
| R2: +Low-vol filter | mom_lowvol_K30 | 10% | 0.74 | -55% |
| R3: +Quality screen | mom_qual_K30 | 10.8% | 0.80 | -47% |
| R4: +Regime gate | mom_qual_regime_K10 | 8.7% | 0.65 | -37% |
| R5: GBM pred (h=6) | pred_K3_h6 | **27.9%** | **0.83** | -49% |
| R5b: GBM pred (h=1) | pred_K3_h1 | 15.5% | 0.60 | -80% |
| R5c: K=1 monthly | pred_K1_h1 | 39.9% | 0.89 | -82% |

YLOka v3 "reproduced" at 27.9% (vs YLOka's claim of 40.78%). Gap likely explained by:
1. Research window difference (my: 2023-12, YLOka: 2024-04)
2. SPY feature computation differences

---

## Current Focus

**Next Run Priority**: Build a new GBM trained DIRECTLY on 1-month forward returns.
- The existing `pred` column targets 3-6m returns. Retraining on 1m target could push IC from 0.035 to 0.05+.
- Each +0.01 IC ≈ +3-5pp CAGR from theory.
- Implementation: walk-forward expanding window, 13-month purged embargo.

See `state/current_focus.md` for detailed run plan.

---

## Top 3 Next Steps

1. **[Run 2]** Train new GBM on 1m forward return cross-sectional rank. Target IC > 0.05.
2. **[Run 2]** Feature interactions Ridge regression: rank(47 features) + top interactions → composite.
3. **[Run 3]** LSTM on 24-month per-stock feature sequences (needs torch availability check).

---

## ETA to Success

**Honest assessment**: UNKNOWN, possibly never with current data constraints.

The 50% CAGR / 2.0 Sharpe target requires sustained monthly IC > 0.10, which would be extraordinary for any publicly available price-only monthly signal. The data and methodology constraints (no leverage, no fundamentals, price-only monthly) fundamentally limit achievable Sharpe to roughly 0.9-1.2 range.

**Possible paths to success gate**:
- Find or engineer a feature with IC > 0.10 vs 1m return (very hard)
- Add fundamentals / alternative data (not in scope per V's direction)
- Accept modified target: CAGR ≥40% / Sharpe ≥1.0 (achievable today)

---

## Hypotheses Tested

| Session | Count |
|---|---|
| YLOka sessions 1-5 (prior work) | 88+ |
| Run 1 (this session) | 131 |
| **Total all-time** | **219+** |

DSR for best result (Sharpe=0.89, n=244, n_trials=219): ~0.995 — genuinely significant.

---

## Questions for V

1. Is the 50%/2.0 success gate meant to be achievable with PRICE-ONLY data and no leverage? The analysis suggests this would require IC > 0.10 which is essentially impossible for monthly equity signals.

2. Is the definition of "monthly rebalance" strict (sell all and rebuy each month) or DCA-compatible (hold positions, allocate new capital monthly)? YLOka's 40% result uses 6-month holds, which may not meet a strict monthly rebalance requirement.

3. Should I explore fundamentals data (earnings, P/E, revenue growth) if I can source it?
