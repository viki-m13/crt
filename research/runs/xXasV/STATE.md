# STATE.md — Quantitative Research Agent

**Last updated**: 2026-05-11 (Session 1 — Bootstrap run)

---

## Current Best Result

| Metric | Value | Notes |
|---|---|---|
| Strategy | v3 ML signal, K=3, h=6m, tight crash gate | Inherits from YLOka research |
| OOS WF CAGR | 40.7% | 248 months, 2003-09 → 2024-04 |
| OOS WF Sharpe | 0.863 | Annualized from monthly returns |
| MaxDD | -49.5% | |
| Sub-period Sharpes | [0.863, 1.004, 0.930] | All >0.8 |
| Lockbox | **UNOPENED** | Sacred |
| DSR | Not computed yet | Need global hyp count |

---

## Hypotheses Tested This Session: 38

(12 baseline + 18 vol-targeting + 8 portfolio construction = 38)

---

## Current Focus

**Phase 2 complete. Moving to Phase 3.**

Best Phase-2 baseline established: v3 ML (K=3, h=6m, tight gate) = 40.7% CAGR, 0.863 Sharpe.
This matches the independently developed YLOka v3 baseline.

Attempted improvements in this session — all failed:
- Volatility targeting (18 configs): best 31.2% CAGR, 0.860 Sharpe (worse)
- Score-proportional weighting: 40.8% CAGR, 0.860 Sharpe (basically same)
- K=2, K=1: worse (higher idiosyncratic risk)
- Quality filters: identical (ML already handles quality implicitly)
- h=2,3,12m: all worse than h=6m

**Next session focus**: Asymmetric-loss GBM retraining (I02).

---

## Gap Analysis: Why 50% CAGR / 2.0 Sharpe is Hard

Target requires:
- CAGR: +9.3pp above baseline (40.7% → 50%)
- Sharpe: +1.14 above baseline (0.863 → 2.0)

The Sharpe gap is the dominant challenge. Achieving Sharpe=2.0 with:
- Same CAGR (40%): requires reducing annual vol from ~42% to ~18% (57% reduction)
- Higher CAGR (50%): requires annual vol ≤ 23%

Long-only concentrated momentum inevitably generates high vol. The crash gate reduces
the WORST months but can't smooth the normal monthly variance.

**Realistic ceiling estimate** (with current data):
- CAGR: 45-55% achievable with better ML signal
- Sharpe: 1.0-1.3 achievable with improved signal + moderate risk management
- Getting Sharpe > 1.5 likely requires fundamentals or options data

---

## Top 3 Next Steps

1. **Asymmetric Loss GBM** (I02): Retrain GBM with loss that penalizes large losers.
   Expected: CAGR ~40%, Sharpe ~0.95-1.1. Medium effort, high EV.

2. **Meta-labeling** (I03): Secondary classifier to filter out high-risk primary picks.
   Expected: Fewer positions, potentially lower drawdown, Sharpe ~0.9-1.0.

3. **LSTM sequential predictor** (I06): Deep learning on 24-month feature sequences.
   Expected: Signal improvement, possible Sharpe >1.0 if LSTM captures patterns missed by GBM.
   Requires GPU for reasonable training time.

---

## ETA-to-Success

**Honest estimate**: Unknown. The 50%/2.0 targets may be unachievable with price-only data.

- If Sharpe target is relaxed to 1.5: achievable in 3-5 sessions with new ML training
- If 2.0 Sharpe is required: likely impossible without fundamentals/options data
- CAGR 50% alone: achievable with better signal, possibly in 2-3 sessions

---

## Questions for V

1. **Is the 2.0 Sharpe target negotiable?** With long-only concentrated equity,
   Sharpe >1.5 is very unusual without leverage or derivatives. The best equity
   momentum strategies in academic literature achieve Sharpe ~1.0-1.5.

2. **Can we access volume data?** Volume bars + OBV would unlock new features
   that could materially improve signal quality.

3. **Can we access PIT fundamentals?** Even basic data (P/E, debt, ROE) would
   allow proper quality + value screening that could reduce MaxDD.

4. **Are GPU resources available?** LSTM training requires 30+ min per fold
   × 22 folds = 11+ hours. Not feasible without GPU.

5. **NDX universe?** NDX has historically had higher momentum persistence.
   Should we switch from SPX (which is what the PIT data covers)?

---

## Data Assets Summary

| Asset | Path | Notes |
|---|---|---|
| PIT S&P500 membership | `.../v2/sp500_pit/sp500_membership_monthly.parquet` | ~500 tickers/asof, 2003-2026 |
| Monthly returns | `.../v2/monthly_returns_clean.parquet` | 1833 tickers, 1995-2026 |
| Feature snapshots | `.../cache/features/YYYY-MM-DD.parquet` | 353 files, 79 features |
| v3 ML predictions | `data/YLOka/pit_panel_with_scores.parquet` | PIT-filtered, 268 asofs |
| Full feature panel | `data/YLOka/pit_panel_full.parquet` | 47 features + ML preds |
| Regime labels | `data/YLOka/regime_labels.parquet` | bull/normal/recovery/crash |
| Rolling IC | `data/YLOka/rolling_ic.parquet` | Per-head 24m rolling IC |

**Limitations** (documented in data/README.md):
- No volume data
- No fundamentals
- Approximate PIT membership (GOOG pre-2014, some edge cases)
- Only 9 truly delisted tickers in price panel (survivorship bias partially present)

---

## Research Hygiene

- Lockbox (2024-05 → 2026-04): **SEALED, 0 touches**
- Research window: 2003-09 → 2024-04 (248 months)
- Total hypotheses logged: 38
- Experiments dir: `quant_research/experiments/`
- Dead ends: `quant_research/state/dead_ends.md`
