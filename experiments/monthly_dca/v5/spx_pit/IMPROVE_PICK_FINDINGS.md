# Stock-picking levers for consistency — conviction-adaptive breadth (POSITIVE-on-consistency, honest CAGR wall)

**Date:** 2026-05-17 · `improve_pick_v3.py`, `improve_pick_run.py`,
`improve_pick_validate.py`.

**Ask (user):** push CAGR higher, *more consistently every year*, with
*less drawdown*, for a **monthly-DCA** investor — think new, focus on
*stock picking* (selection), don't re-tread overlays.

## What was tried — four genuinely new SELECTION levers

All causal, all on the bit-exact production stream (max|Δ ret_m| vs the
WIN1 v2 sim = **0.00**), gauntlet = full WF/era/DCA + a new per-year
consistency block (yearly-CAGR std, worst calendar year, neg years).
None of these are overlays; every one changes *which stocks are held*.

| lever | idea | result |
|---|---|---|
| **conviction-adaptive breadth** | wide cross-sectional score gap = real conviction → hold K=2; bunched scores = picker guessing (the bad years) → widen to K=3 | **consistency win** (below) |
| decorrelated 2nd pick | pick #1 by score, #2 = best-scored with trailing-12m corr < ρ to #1 | non-monotonic in ρ (0.4→43%, 0.5→51%, 0.6→47%): a fragile peak, **rejected** |
| falling-knife screen | drop candidates in the pool's bottom-q 1m return AND still below their 3m | pure CAGR/WF bleed, **rejected** |
| blend trigger **and** blend select | the combo the prior phase only tested separately | −5pp CAGR, no DD help, **rejected** |

## The finding — conviction-adaptive breadth

At each rebalance, after the identical scorer + Chronos filter, measure
conviction = `score[top] − median(pool score)`. Wide gap → the picker
has a genuine standout, concentrate **K=2**. Bunched scores → the picker
is effectively guessing (historically the years it picks badly) → widen
to **K=3** so one bad low-conviction name can't dominate the year.

| variant | CAGR | Sharpe | Max DD | WF | era | **yr-CAGR std** | **worst yr** | DCA10y |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| WIN1 (deployed) | **51.9%** | **1.01** | −65.8% | 9/10 | 4/4 | 3.21 | −34.1% | 1.00 |
| **adaptive-breadth** | 48.3% | 0.99 | −65.8% | **10/10** | 4/4 | **2.48** | **−27.2%** | 1.00 |

It is **not a CAGR improvement** — it costs ~3.6pp of lump-sum CAGR.
It *is* a robust **consistency** improvement, which is precisely the
stated DCA goal:

- **Year-to-year dispersion −23%** (3.21 → 2.48 yearly-CAGR std).
- **Worst calendar year +6.9pp** (−34.1% → −27.2%).
- **Walk-forward 9/10 → 10/10** beats SPY.
- DCA 10y win stays **100%**; DCA 1y win 0.768 → 0.776.

## Overfit screens — all pass (the WIN1-grade battery)

- **Plateau, not a peak.** yr-std **2.48** and worst-year **−27.2%** are
  *literally identical* across conv thresholds 0.06–0.10 and a wide
  0.07/0.20 config. The only break is the degenerate `mid4/hi4`
  (too much breadth → 37.5% CAGR). This is a wide plateau on the
  consistency axis, the opposite of a fragile fit.
- **Cost-insensitive.** Identical at 0/10/20/30 bps (adaptive K adds no
  turnover — it only changes basket *width* at an existing rebalance).
- **TRUE OOS (design 2003-12 | untouched holdout 2013-26):**

  | | design CAGR/Sh | holdout CAGR/Sh/DD |
  |---|--:|--:|
  | WIN1 | 78.0% / 1.11 | 34.7% / 1.04 / **−59.0%** |
  | adaptive-breadth | 73.6% / 1.09 | 31.6% / **1.06** / **−51.6%** |

  Less in-sample design CAGR, **higher** holdout Sharpe, **8pp
  shallower holdout drawdown** — the anti-overfit signature the repo
  accepted for WIN1 itself.

## Honest caveats

1. **The headline −66% max DD does not move.** It is the 2008
   market-wide crash; widening from 2 to 3 names cannot escape a
   systemic event, and the worst 1y DCA MOIC (0.47×) is unchanged —
   it's the same GFC-terminal window documented in `DCA_INVESTOR_EVAL.md`.
   Only the (already-documented, late-firing) regime-cash gate touches
   that tail; this lever attacks the *interim-year* ride, not the GFC
   trough.
2. **The CAGR wall is real and is reported, not curve-fit away.** No
   pure stock-picking lever tried here (adaptive breadth, decorrelated
   2nd pick, falling-knife, blend×blend) beats WIN1 on CAGR. This
   independently re-confirms the repo's Phase-11/B verdict from the
   *selection* side: there is one price-only large-cap alpha and you
   cannot pick your way past its ceiling — you trade some CAGR for
   consistency, you do not get both for free.
3. **WIN2 (`select=blend`) remains the genuine DD lever** (−54.5% vs
   −65.8%), at its own CAGR/worst-year cost. AK and blend-select are
   *alternatives*, not additive — `AK + sel=blend` is the worst of both
   (−63% DD, −34.8% worst yr). Pick one axis to optimise.
4. **Not auto-deployed.** Per repo convention
   (`NOVEL_V9_PROD_FINDINGS.md`, `IMPROVE_FINDINGS.md`), changing the
   live product's public numbers is a user decision.
   `build_webapp_v5_pit.py`, `data.json`, `STRATEGY_SPEC` are unchanged.

## Recommendation

For a **monthly-DCA accumulator** whose felt experience is the
year-to-year ride (not a single lump-sum drawdown), **conviction-adaptive
breadth** is the strongest, most defensible *consistency* improvement
found: −23% yearly dispersion, +6.9pp worst year, WF 10/10, 100% 10y
DCA win retained, cost-free, plateau-stable, OOS-positive — for ~3.6pp
of headline CAGR. If the priority is purely the deepest-drawdown
number, WIN2 (`select=blend`) is the lever instead. Both are honest,
validated, and offered for sign-off — not shipped. The "higher CAGR
*and* less drawdown *and* more consistent, all at once" target is not
attainable from this single price-only alpha; that wall is now
demonstrated from the stock-picking side too.

## Files

- `improve_pick_v3.py` — extended bit-faithful sim (4 new selection
  levers; reproduces WIN1 exactly at defaults)
- `improve_pick_run.py` — lever sweep + per-year consistency block
- `improve_pick_validate.py` — plateau / cost / OOS / DCA gauntlet
- `augmented/improve_pick_v3.json`, `augmented/improve_pick_validate.json`
  — raw results
