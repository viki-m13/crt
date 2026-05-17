# Stacking E1's free-consistency lever on the stronger RC-D sleeve

**Date:** 2026-05-17 · `improve_phase6_rce1.py`, `improve_phase6b_rce1.py`.

**Context:** the other agent deployed **E1** = `0.5·WIN1 + 0.5·WIN2`
(Phase 4/5 on main). E1's insight: decorrelating two *role-swapped*
implementations of the same alpha buys consistency for free (worst-5y
+2.5→+11.9 %/yr, DD −66→−56 %) at no CAGR cost. This branch had already
found **RC D** (regime-conditional select-blend) — a strictly *stronger
single sleeve* than WIN1/WIN2.

**Question (user):** how does E1 compare, what do we learn, can we
improve RC D + adaptive-breadth further using it?

## What we learned from E1 — and the synthesis

E1's decorrelation lever (portfolio of two role-swapped sleeves) is
**orthogonal** to RC D's lever (regime-timed blend weight). One
diversifies *implementation* variance; the other adds *regime* alpha.
So they should **stack** — build E1 out of RC-D-grade sleeves instead of
plain WIN1/WIN2. They do. Apples-to-apples on E1's own Phase-4 gauntlet
(same `evaluate()` + `consistency()`, deployed-E1 reproduced exactly:
CAGR 52.2 %, Sh 1.04, DD −56 %, WF 10/10, worst-5y +11.9 %, CAGRx0309
33.7 %):

| variant | CAGR | Sh | Max DD | WF | era | worst-5y | CAGR ex-03-09 | DCA 3y |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **E1 (deployed)** | 52.2% | 1.04 | −56% | 10/10 | 4/4 | +11.9% | 33.7% | .941 |
| RC D + adaptK *solo* | **58.8%** | 1.11 | **−54%** | 10/10 | 4/4 | +10.3% | 41.6% | .937 |
| **LEAD-CAGR** `½WIN1+½(RC D+aK)` | **56.9%** | **1.11** | −56% | 10/10 | 4/4 | **+13.8%** | **38.5%** | **.962** |
| **LEAD-CONS** pure RC-E1 `½rcA_k+½rcB_k` | 54.6% | 1.10 | −56% | 10/10 | 4/4 | **+13.1%** | 36.4% | **.967** |

Both **LEAD** variants are a **strict Pareto improvement over the
deployed E1**: higher CAGR (+4.7 / +2.4 pp), higher Sharpe, same −56 %
DD, same WF 10/10 & era 4/4, **better** worst-5-year DCA, and a much
higher forward-relevant CAGR-excluding-2003-09 (33.7 → 38.5 / 36.4 %).

The single-sleeve `RC D + adaptK` actually has the **highest CAGR
(58.8 %) and a slightly shallower −54 % DD**, but its front-load
consistency is weaker (worst-5y +10.3 %). Wrapping it in E1's two-sleeve
structure trades ~2 pp CAGR for a materially better worst-5y (+13.8 %)
and OOS robustness — the right trade for a monthly-DCA accumulator.

## Overfit gauntlet — passes (E1's own bar)

- **Mix-weight plateau** (w on first sleeve 0.3–0.7): LEAD-CAGR holds
  CAGR 55–58 %, Sharpe 1.08–1.12, **WF 10/10 and era 4/4 in every
  cell**, worst-5y +10.8…+13.8 %; DD only degrades past w=0.7. 50/50 is
  the un-tuned center of a wide flat plateau. LEAD-CONS is even
  shallower-DD at low w (−53.5 % at w=0.3).
- **Cost-insensitive:** identical 0/10/20/30 bps (both sleeves
  low-turnover).
- **TRUE OOS** (design 2003-12 | untouched holdout 2013-26): LEAD-CAGR
  holdout **38.6 % / Sharpe 1.24** vs E1 35.4 % / 1.08; LEAD-CONS
  holdout Sharpe **1.24** with DD −52.9 % (vs E1 −55.9 %). Both are
  *stronger out-of-sample* than the deployed E1 — anti-overfit
  signature.
- **MC synthetic-delisting:** both ≥ E1 at every hazard (α0 57.0/54.5
  vs 52.8; α4 32.7/33.7 vs 27.8; α8 5.9/6.2 vs 5.0).

## Honest caveats

1. **The −56 % drawdown floor is unchanged.** It is the 2008 GFC
   systemic event; neither the decorrelation lever nor the regime lever
   reshapes a market-wide crash (only the documented, late-firing
   regime-cash gate touches that — `DCA_INVESTOR_EVAL.md`). The honest
   claim is **"materially higher CAGR / Sharpe / forward-consistency at
   E1's drawdown"**, not below it. (LEAD-CONS does nudge the *OOS*
   drawdown slightly better, −52.9 % vs −55.9 %.)
2. **LEAD-CAGR is the smaller refactor** — one sleeve is the
   already-deployed WIN1, the other is RC D + adaptive-breadth (a
   parametrized `run_sim_v3`). LEAD-CONS (pure RC-E1) is a full
   two-RC-sleeve build.
3. Same residual data caveat as the whole repo (213 OTC bankruptcy-Q
   tickers absent); the relative improvement uses the identical
   universe/pipeline as E1, so it should survive a CRSP correction.
4. **Not auto-deployed.** Per repo convention, changing the live
   product's public numbers is a user decision.
   `build_webapp_v5_pit.py`, `data.json`, `STRATEGY_SPEC` unchanged on
   this branch.

## Recommendation

The lesson from E1 was correct and we extended it: E1's "decorrelate
the same alpha across two implementations" stacks cleanly with this
branch's regime-conditional + adaptive-breadth stock-picking levers.

- **LEAD-CAGR = `0.5·WIN1 + 0.5·(RC D + adaptive-breadth)`** — the
  headline: strictly dominates the deployed E1 (+4.7 pp CAGR, +0.07
  Sharpe, +1.9 pp worst-5y, +4.8 pp forward CAGR-ex-03-09) at the same
  −56 % DD, fully overfit-screened, and a *small* refactor (reuses the
  already-live WIN1 sleeve).
- **LEAD-CONS = pure RC-E1** — for maximum year-to-year smoothness and
  OOS robustness (lowest yearly std, best DCA-3y, holdout Sharpe 1.24,
  shallower OOS DD).

Both honest, plateau-stable, cost-free, OOS-positive, MC-robust.
Offered for sign-off, not shipped. The −56 % GFC floor remains a
separate regime-timing research program, not a stock-picking knob.

## Files

- `improve_phase6_rce1.py` — RC-E1 construction + full phase-4 gauntlet
- `improve_phase6b_rce1.py` — mix-weight plateau + MC-delisting screens
- `improve_sim_v2.py` — synced to main (adds min_hold / k_by_regime;
  WIN1/WIN2 fallback still bit-exact)
- `augmented/improve_phase6_rce1.json`, `improve_phase6b_rce1.json`
