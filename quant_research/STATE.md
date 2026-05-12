# Quant Research — STATE

**Last updated**: 2026-05-12 (Session 1 / Hour 1 bootstrap)

## STATUS HEADLINE
> **Sharpe 2.0 + CAGR 50% simultaneously appears structurally unreachable with current methodology.**
> Best achieved: CAGR=48.73%, Sharpe=1.00 (Donchian filter). Sharpe ceiling ≈1.12 at CAGR≈13%.
> QUESTIONS FOR V — see bottom of this file.

---

## Current Best Results (OOS Research Window 2003-09 → 2024-04)

| Configuration | CAGR | Sharpe | MDD | Note |
|---|---|---|---|---|
| Donchian130 filter K=3 h=6 | 48.73% | 1.00 | -57.0% | Best CAGR |
| Accel filter K=3 h=6 | 42.39% | 0.96 | -49.8% | |
| Baseline ml_3plus6 K=3 h=6 | 40.78% | 0.95 | -49.8% | v3 repro |
| invvol_port K=7 vc=0.5 | 18.0% | 1.05 | -32.7% | Best combined |
| qualinvvol vc=0.55 K=3 | 16.0% | 1.11 | -22.9% | Best Sharpe |
| ra_pred_div_vol K=10 | 12.7% | 1.12 | -22.7% | Highest Sharpe |

**Oracle bound (perfect foresight) K=3**: CAGR=1632%, Sharpe=6.32
**Signal IC (pred_3m vs 1m actual)**: 0.022, t=2.62
**Signal IC (pred_6m vs 1m actual)**: 0.029, t=4.01
**Total hypotheses tested all sessions**: ~224 (108 prior + 116 this session)

---

## Critical Finding: Why Sharpe 2.0 + CAGR 50% is Structurally Blocked

### The Math
For Sharpe ≥ 2.0 AND CAGR ≥ 50%:
- Monthly arithmetic return ≥ ~3.5%
- For Sharpe 2.0: monthly_std ≤ 3.5% / (2.0/√12) = 6.1%
- Annual vol ≤ 6.1% × √12 ≈ 21%

For a K=3 momentum portfolio with stocks having 40-80% annual vol and ρ=0.5:
- Portfolio annual vol = 60% × √(0.5 + 0.5/3) ≈ 49% — WAY above 21% limit

To get portfolio vol to 21%, need stocks with ~25% annual vol AND/OR K=10+ stocks.
But low-vol stocks (25% annual) don't produce 50%+ CAGR from momentum.

### Empirical Confirmation (This Session)
- Quality filter vol≤0.50: CAGR drops to 20%, Sharpe 0.89 — not 50%
- K=10 quality-filtered: CAGR 19%, Sharpe 1.03 — not 50%
- Risk-adjusted scoring (pred/vol): CAGR 13%, Sharpe 1.12 — not 50%
- LightGBM walk-forward ranker: IC = -0.009 (overfits), Sharpe 0.69-0.85
- Enhanced crash gate (breadth signal): Sharpe drops to 0.66 (over-filters)
- Vol targeting overlay: Sharpe stays ≈0.93 (doesn't improve; theory-correct)

### The Fundamental Law Bound
IC = 0.025, Breadth = K×12 monthly bets/year.
IR_max = IC × √(K×12) × 1/σ × √12
With K=3: Sharpe_max ≈ 0.025 × √36 × √12 = 0.025 × 6 × 3.46 ≈ 0.52

Observed Sharpe = 1.0 EXCEEDS the FL bound — this extra alpha comes entirely from the **crash gate** (avoiding 2008/2009, 2020). Without crash gate the strategy would Sharpe ≈0.5.

The crash gate cannot be made 2× more powerful without false positives. It already avoids the worst crashes. The breadth-based enhanced gate HURT performance.

---

## What's Been Tried (This Session + Prior)

### Prior sessions (YLOka, 108 hypotheses)
Conviction weights, classifier filters, K sweeps, hold sweeps, adaptive IC,
regime specialists, dispersion-conditional K, dynamic hold, soft cash.
All failed to materially improve on baseline.

### This session (116 hypotheses)
- Quality vol ceiling (0.35-0.80): improves Sharpe to 0.89-1.11 but cuts CAGR to 12-22%
- Invvol weighting (scorer blend): Sharpe 0.97-1.11, CAGR 13-16%
- Invvol portfolio weights: Sharpe 1.0-1.08 (K=7-10 best), CAGR 18-18%
- K sweep with quality: K=7 gives best Sharpe 1.04, CAGR 20%
- Vol targeting overlay: no Sharpe improvement (theoretically correct)
- Risk-adjusted scoring (pred/vol): Sharpe 1.00-1.12, CAGR 12-13%
- LightGBM walk-forward (return label): IC = -0.009, worse than baseline
- LightGBM walk-forward (Sharpe label): CAGR 14-15%, Sharpe 0.93-0.97
- Enhanced crash gate (breadth+portvol): CAGR drops, Sharpe 0.66-0.73

### Dead ends added to dead_ends.md
- Enhanced crash gate with breadth signal
- LightGBM retrained on same features (can't beat pred_3m/pred_6m)
- Vol targeting overlay (doesn't change Sharpe by construction)

---

## Current Focus

NONE — waiting for V input on goal framing.

Remaining high-EV ideas (won't reach Sharpe 2.0 without new data):
1. **Portfolio optimization (MVO)**: given K=10 quality candidates, use rolling covariance to find minimum variance portfolio. Expected: Sharpe 1.2-1.3, CAGR 15-20%.
2. **Regime-conditional stock selection**: bull/normal → baseline, recovery → quality. Expected: Sharpe 1.0-1.1, CAGR 35-40%.
3. **Fundamentals data** (if V can provide): add quality screens (ROE, rev growth, debt/equity). Expected: potential Sharpe 1.5-1.8 with CAGR 30-40%.
4. **Walk-forward v5 Chronos with new prompting** (if API available): better time-series predictions. Uncertain upside.

---

## Questions for V (Section 11 — Escalation)

**Fundamental question**: Is the Sharpe ≥ 2.0 + CAGR ≥ 50% target achievable with the current data?

After 224 hypotheses across 6 sessions, I conclude **no** — not simultaneously — for a price-only long-only monthly equities strategy. The mathematical ceiling appears to be around Sharpe 1.1-1.2 at CAGR 15-20%, or Sharpe 1.0 at CAGR 48-49%.

**Proposed alternatives (pick one):**

**Option A — Relax Sharpe, keep CAGR target:**
- Target: CAGR ≥ 50% (very close with Donchian), Sharpe ≥ 1.2
- Path: Focus on CAGR improvements while keeping vol manageable
- Achievable? Probably yes within 2-3 more sessions

**Option B — Relax CAGR, keep Sharpe target:**
- Target: CAGR ≥ 25%, Sharpe ≥ 2.0
- Path: Would need fundamental data OR different universe OR shorter holding period
- Achievable? Maybe with fundamentals; uncertain otherwise

**Option C — Change methodology:**
- Add fundamental data (P/E, ROE, growth rates)
- Use smaller universe (small/mid cap) with stronger momentum
- Add weekly rebalance for faster crash exit (but need higher-freq data)

**Option D — Accept current best:**
- Current best: CAGR 48.73%, Sharpe 1.00
- This already beats the "v5 Chronos 30% baseline" by 18 percentage points
- Submit as winner under relaxed criteria?

**If none of the above, please specify alternative direction.**

---

## Next Steps (if V approves continuation)

1. Implement MVO portfolio optimization on quality-filtered candidates
2. Try regime-conditional stock selection (quality in recovery, momentum in bull/normal)
3. If fundamentals data available, add quality screens
4. If Sharpe target is relaxed to 1.2, run full gauntlet on Donchian variant

---

## ETA-to-Success Estimate

**Honest: Unknown.** The target as specified appears unreachable with current data. With goal adjustment (Option A or D), success could be within 2-4 sessions.
