# Dual-benchmark DCA consistency — beat SPY-DCA *and* QQQ-DCA honestly

**Date:** 2026-06-15 · `improve_consistency_dca.py`

**User goal:** keep CAGR high, maximize the fraction of rolling monthly-DCA
windows (1/3/5/10y, money-weighted) where the strategy beats **both**
SPY-DCA and QQQ-DCA, and add an **off-switch** that prevents the deep
drawdowns. PIT S&P 500 only.

## New objective + anti-overfit discipline

Prior repo work only ever measured DCA-win vs **SPY-DCA**. This adds the
harder **dual** bar (beat SPY-DCA AND QQQ-DCA in the same window) and a
frozen-holdout split (DESIGN ≤2015-12 | HOLDOUT ≥2016-01, evaluated once).
The harness reproduces the deployed E2 stream from the canonical
reproducible sim (`improve_sim_v2`/`improve_pick_v3`; WIN1 repro CAGR 50.9%
vs 51.9%, MaxDD −65.8% exact).

## Baseline — deployed E2 on the new objective (the missing number)

| horizon | win vs SPY | win vs QQQ | **win BOTH** | worst MOIC vs SPY/QQQ |
|---|--:|--:|--:|--:|
| 1y  | 80% | 79% | **77%** | 0.57 / 0.55 |
| 3y  | 96% | 92% | **92%** | 0.64 / 0.59 |
| 5y  | 99% | 99% | **99%** | 0.83 / 0.79 |
| 10y | 100% | 100% | **100%** | 2.5 / 1.9 |

E2 already wins ~99–100% of 5y/10y windows vs both. **The only real
weakness is short horizons** (1y 77%, 3y 92%) and the worst-1y floor (~0.55).

## Finding 1 — off-switches do NOT improve the objective (honest negative)

Pre-registered exposure-overlay menu (causal; park in cash unless noted):

| variant | CAGR | MaxDD | Sharpe | win-both 1y/3y/5y/10y |
|---|--:|--:|--:|--:|
| **E2 baseline** | **56.0%** | −55.9% | 1.10 | **77/92/99/100** |
| + SPY crash off-switch | 56.0% | −55.9% | 1.10 | 77/92/99/100 (no-op) |
| + basket-DD breaker −25% (cash) | 30.6% | −47.5% | 0.96 | 60/58/67/78 |
| + basket-DD breaker −30% (cash) | 35.7% | −47.5% | 1.03 | 63/68/84/99 |
| + vol-target 30% | 37.2% | −55.9% | 1.15 | 74/80/89/100 |
| [diag] basket-DD −30%, park=beta | 42.7% | −63.8% | 1.12 | 69/75/96/100 |

1. **The external SPY off-switch is a no-op** — E2's internal
   `classify_regime_tight` gate already captures the SPY crash signal.
2. **Cash off-switches lower the drawdown and lift the worst-1y floor
   (0.55→0.75) but collapse CAGR and win-rate** — this strategy's CAGR is
   front-loaded into the post-crash V-recovery (+270% in 2009), so every
   month parked in cash loses to the rising benchmark.
3. **Parking in market-beta** recovers some CAGR and the floor but does
   *not* prevent the drawdown (beta also crashes) and still lowers 1y/3y
   win-rate. No overlay beats baseline on the objective.

**Conclusion:** "prevent the drawdown" and "keep CAGR high" are in direct
tension here — the −56% floor is the 2008 GFC and avoiding it forfeits the
recovery that is the edge. The deployed E2 is already the win-rate optimum
among off-switch overlays. The losing windows are dominated by
**idiosyncratic 2-stock entry-timing luck in normal markets**, not crashes.

## Next iteration (pre-registered)

**Time-diversification (staggered monthly entry).** The repo's 2024
diagnosis (`TIMING_LUCK.md`) showed staggering swings a bad-luck year from
−25pp to +56pp by removing entry-date luck — *without* going to cash, so
market exposure (and CAGR) is preserved. Hypothesis: a 6-tranche staggered
E2 lifts 1y win-both from 77% and 3y from 92% with ≤3pp CAGR cost. To be
implemented as a correct calendar-tranche sim (not stream-shifting, which
`IMPROVE_FINDINGS.md` Phase 5 flagged as a methodological artifact),
developed on DESIGN ≤2015 and scored once on the frozen HOLDOUT.

**Honest expectation:** a long-only concentrated S&P-500 equity strategy
has a structural ceiling on 1y win-rate vs QQQ in tech-melt-up years; the
realistic honest target is ~85% (1y) / ~97% (3y), not 100%.

## Files
- `improve_consistency_dca.py` — dual-benchmark + frozen-holdout harness, off-switch menu
- `augmented/improve_consistency_dca.json` — raw results
