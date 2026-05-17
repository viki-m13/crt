# Improving the main strategy further — blended drift-trigger (POSITIVE)

**Date:** 2026-05-16 · `improve_main_strategy.py`, `improve_sim_v2.py`,
`improve_phase2.py`, `improve_phase3.py`, `improve_phase3b.py`.

**Ask:** push CAGR higher, *more consistently*, with *less drawdown* —
think new, try many things, don't re-tread.

## What was tried (and rejected)

On the **real** production pipeline (PIT membership + Chronos p70 +
inv-vol cap + tight regime gate + rule-based min-6m/score-drift, 10 bps):

- **Portfolio vol-targeting** (park excess in cash / SPY, 20-35% target):
  cuts Max DD ~10-14pp but **bleeds 17-27pp of CAGR** and wrecks WF/era
  consistency. The strategy's vol is persistent; scaling it down just
  chops compounding. ❌
- **Equity drawdown circuit-breaker** (cut to 30% on -30/-40/-50% DD):
  one config (cut -30%, park SPY) reaches Sharpe 0.99 / DD -49.5% but
  costs ~9pp CAGR and loses WF/era consistency; on the *winner* stream
  it just bleeds CAGR without reducing the (early, fast) max DD. ❌
- **50/50 book-blend** of the two production scorer streams: lifts WF to
  10/10 and eras to 4/4 at ~-2pp CAGR — real consistency gain, but **no
  drawdown help**. ⚠️ (useful but dominated by the finding below)

## The key discovery

`SCORER_MODE` in `build_webapp_v5_pit.run_full_sim` only drives the
**score-drift REBALANCE trigger** (`_compute_candidate_top`). The basket
itself is *always* formed with `ml_3plus6` (line 439). Trigger and
selection are decoupled. So a genuinely new lever is **which scorer
decides "my picks drifted, rotate"** — independent of how picks are made.

A faithful parametrized sim (`improve_sim_v2.run_sim_v2`,
**bit-exact reproduction** of the production stream, max|Δ ret_m| = 0.0)
sweeps `trigger_mode` × `select_mode` ∈ {ml_3plus6, consensus, **blend**}
where `blend` = 0.5·consensus-rank + 0.5·ml-rank.

## Result — two validated, anti-overfit improvements

| variant | CAGR | Sharpe | Max DD | WF | DCA 3/5/10y | eras |
|---|--:|--:|--:|--:|--:|--:|
| deployed `consensus` | 47.3% | 0.94 | -69.1% | 8/10 | .85/.89/.99 | 3/4 |
| **WIN1** trig=`blend`, sel=`ml_3plus6` | **51.9%** | **1.01** | **-65.8%** | **9/10** | **.91/.95/1.0** | **4/4** |
| **WIN2** trig=`ml_3plus6`, sel=`blend` | 48.8% | 0.95 | **-54.5%** | 9/10 | .84/.98/1.0 | **4/4** |

**WIN1 is a strict Pareto improvement** over the deployed strategy —
every headline metric is better — from a *one-bit* change (rank the
drift-trigger candidate pool by the blended scorer instead of consensus;
selection, costs, turnover all unchanged). **WIN2** trades ~3pp of CAGR
for a **14.6pp shallower max drawdown** (-54.5% vs -69.1%).

### Overfit screens — all pass

- **Cost-insensitive:** WIN1 identical at 0/10/20/30 bps (the trigger
  change adds no turnover).
- **Wide parameter plateau:** WIN1 holds 51.9% CAGR / 9-10 WF / 4/4 era
  across blend-weight 0.3-0.8 (only w≤0.2 degrades). WIN2 plateaus
  0.3-0.6 (w=0.4 → **10/10** WF). Not a fragile peak.
- **TRUE OOS (design 2003-12 | untouched holdout 2013-26):**

  | | design CAGR/Sh | holdout CAGR/Sh |
  |---|--:|--:|
  | deployed | 92.2% / 1.22 | 20.5% / **0.68** |
  | WIN1 | 78.0% / 1.11 | **34.7% / 1.04** |
  | WIN2 | 70.3% / 1.03 | **34.3% / 0.98** |

  Both give up *in-sample* design CAGR and are far stronger in the
  untouched holdout (+14pp CAGR, +0.30-0.36 Sharpe). That is the
  **opposite of an overfit signature** — less in-sample fit, more OOS
  robustness.
- **MC synthetic-delisting:** WIN1 ≥ deployed at every hazard; at
  α=8%/yr deployed goes **-3.5%** while WIN1 stays **+5.0%**.

## Honest caveats

1. Same residual data caveat as the rest of the repo: 213 OTC
   bankruptcy-Q tickers are still absent from the panel. The *relative*
   improvement uses the identical universe/pipeline as the baseline, so
   it should survive a CRSP correction even if absolute levels shift.
2. **The drawdown problem is narrowed, not solved.** -55% to -66% is
   still a brutal 2-stock ride. Phase 11's honest Sharpe-≈1 ceiling for
   a single price-only large-cap alpha stands; this is a real but
   incremental gain within that ceiling, not a new alpha source.
3. **Not auto-deployed.** Per repo convention
   (`NOVEL_V9_PROD_FINDINGS.md`), changing the live product's public
   numbers is a user decision. `build_webapp_v5_pit.py` default,
   `data.json`, and `STRATEGY_SPEC` are unchanged in this commit.

## Recommendation

This is the strongest, most defensible improvement found for the stated
goal. **WIN1 (drift-trigger = blend)** if the priority is "higher CAGR,
more consistently" — it is strictly better on every metric and
materially more OOS-robust. **WIN2 (selection = blend)** if the priority
is the shallowest drawdown. Both are honest, plateau-stable,
cost-free, and OOS-validated. Deploying either is a one-line change to
the trigger/selection scorer; it is offered for sign-off, not shipped.

## Phase 4 — same CAGR, MORE CONSISTENTLY: E1 = WIN1 ⊕ WIN2

**Ask:** keep the ~52% CAGR but deliver it *more consistently* (the edge
was still front-loaded and short/mid-horizon paths noisy).

Levers that de-risk (vol-target, DD-breaker) were already proven to bleed
CAGR. The winning idea instead **diversifies the same alpha across two
implementations**: WIN1 (trigger=blend, select=ml_3plus6) and WIN2
(trigger=ml_3plus6, select=blend) are the *same* ml+consensus edge with
the two scorers swapped between the selection and drift-trigger roles.
They rebalance at different times and sometimes hold different names, so
their idiosyncratic 2-stock variance partially decorrelates while the
shared alpha compounds. **E1 = 50/50 monthly-rebalanced portfolio of the
two.**

| metric | WIN1 (deployed) | **E1 (WIN1⊕WIN2 50/50)** |
|---|--:|--:|
| Full CAGR | 51.9% | **52.2%** |
| Sharpe | 1.01 | **1.04** |
| Max DD | -66% | **-56%** |
| WF beats SPY | 9/10 | **10/10** |
| Eras beat S&P-DCA | 4/4 | 4/4 |
| DCA-win 3y / 5y | .912 / .953 | **.941 / .995** |
| **Worst rolling 5-yr CAGR** | **+2.5%/yr** | **+11.9%/yr** |
| % rolling-3y beat SPY | 86% | **90%** |
| Yearly-return stdev | 314% | **280%** |
| CAGR excl. 2003-09 | 33.5% | 33.7% |

Not a trade-off — E1 is **strictly better on return, risk, AND every
consistency metric**. The standout: a DCA investor's *worst* 5-year
window improves from +2.5%/yr to **+11.9%/yr**.

Hyperparameter ensembles (min-hold {5,6,7}, blend-weight {.4,.5,.6},
mega-grid, regime-conditioned K) were also tested — they either bled
CAGR (min-hold/mega: 42-45%) or did nothing (blend-weight/regime-K
≈ WIN1). Decorrelating the *two scorer-role implementations* is the only
lever that adds consistency for free.

### Overfit gauntlet (all pass)

- **Mix-weight plateau:** w1 ∈ 0.3-0.7 all give CAGR 51-52.5%, Sharpe
  1.02-1.04, **10/10 WF**, 4/4 eras, worst-5y +8 to +13%/yr. A wide flat
  plateau; 50/50 is the un-tuned equal-weight choice inside it.
- **TRUE OOS** (design 2003-12 | untouched holdout 2013-26): E1 holdout
  **CAGR 35.4% / Sharpe 1.08** — the strongest holdout of any variant
  (WIN1 34.7%/1.04; original consensus 20.5%/0.68).
- **Cost-insensitive:** identical 0/10/20/30 bps (both sleeves low-turn).
- **MC delisting:** more robust than WIN1 alone — α=4% median CAGR
  39.6% (WIN1 27.8%), α=8% 14.0% (WIN1 5.0%). Two decorrelated 2-stock
  sleeves are rarely wiped simultaneously.

### Honest caveats

1. E1 is a **two-sleeve portfolio**, so production deployment is a real
   refactor (run two sims — selection ml_3plus6 / trigger blend, AND
   selection blend / trigger ml_3plus6 — then 50/50 the monthly returns
   and present a combined ≤4-name basket), not a one-line flag like WIN1.
2. Same residual data caveat as the rest of the repo (213 OTC
   bankruptcy-Q tickers absent). Relative improvement uses the identical
   universe/pipeline so it should survive a CRSP correction.
3. The edge magnitude is still front-loaded in 2003-09 and drawdowns are
   still deep (-56%). E1 narrows the dispersion materially; it does not
   make the strategy low-risk. Disclosed, not hidden.

### Recommendation

E1 is the strongest answer to "same CAGR, more consistently": it raises
CAGR and Sharpe, cuts Max DD by 10pp, takes WF to 10/10, and roughly
**doubles the worst-5-year DCA outcome**, fully overfit-screened.

**DEPLOYED 2026-05-17** (user-approved). `build_webapp_v5_pit.py`:
`run_full_sim` parametrized with `trigger_mode`/`select_mode` (WIN1
fallback is bit-exact, max|Δ|=0.0); new `run_e1_blend` runs both
sleeves and 50/50-blends net monthly returns; `STRATEGY_VARIANT='E1'`,
`WINNER_NAME`, `STRATEGY_SPEC`, docstring updated. `data.json`
regenerated (canonical headline CAGR 51.9%, Sharpe 1.03, accumulating
DCA Max DD -56%, 10y/5y/3y DCA-win 100%/99%/91%, 4/4 eras, combined
live book PH·ETN·UBER·SYF @ 25%). Homepage (`docs/index.html`,
`docs/monthly_dca.js`), the `/experiments/monthly-dca` dashboard and
README fully synced; cron auto-regenerates via the same builder.

## Files

- `improve_main_strategy.py` — overlay battery (vol-target, DD-breaker)
- `improve_sim_v2.py` — bit-exact parametrized sim (trigger × select ×
  min-hold × regime-K)
- `improve_phase2.py` — scorer matrix + book-blends (fidelity-checked)
- `improve_phase3.py` / `improve_phase3b.py` — WIN1/WIN2 overfit gauntlet
- `improve_phase4.py` — consistency levers (E1 + ensembles)
- `improve_phase4b.py` — E1 overfit gauntlet (plateau/OOS/cost/MC)
- `augmented/improve_*.json` — raw results

## Phase 5 — does a 3rd sleeve beat E1? NO (honest negative)

`improve_phase5.py`. Tested 3-way / 4-way equal-weight sleeve
ensembles adding a consensus-based sleeve to E1's {A,B}, plus a
phase-offset twin.

| variant | CAGR | Sharpe | MaxDD | WF | worst-5y | CAGRx0309 |
|---|--:|--:|--:|--:|--:|--:|
| **E1 (deployed)** | **52.2%** | **1.04** | **-56%** | **10/10** | **+11.9%** | **33.7%** |
| 1/3 {A,B,C=cons/ml} | 51.4% | 1.03 | -60% | 10/10 | +8.2% | 30.2% |
| 1/3 {A,B,D=ml/cons} | 47.6% | 0.99 | -56% | 9/10 | +9.4% | 30.7% |
| 1/3 {A,B,E=cons/bl} | 51.9% | 1.02 | -56% | 10/10 | +10.5% | 32.1% |
| 1/4 {A,B,C,D} | 48.3% | 1.00 | -59% | 9/10 | +5.9% | 29.0% |

**No third sleeve improves on E1.** Every consensus-based addition
either drags CAGR or widens drawdown / lowers the ex-2003-09 CAGR and
worst-5y. This is the expected consequence of the repo-wide finding
that there is essentially ONE independent alpha here: you can
decorrelate the *same* ml+blend edge across two role-swapped
implementations (E1), but a third sleeve necessarily leans on the
weaker consensus variant and dilutes. **E1 (2-sleeve) is the
consistency optimum and the correct stopping point.**

The `P1` phase-offset row in the raw JSON is a **methodological
artifact** (shifting a realized return stream by one month and
averaging spuriously smooths variance and misaligns returns/dates) —
explicitly NOT a finding and not pursued. A genuine rebalance-luck
time-diversification test would need a real staggered-entry sim, which
the event-driven (not calendar) rule-based rebalance does not cleanly
admit; flagged for honesty.

**Net:** E1 stands as the deployed strategy; no further sleeve change
is warranted on this data.

## Phase 6 — E1 SUPERSEDED: E2 (LEAD-CAGR) deployed 2026-05-17

Phase 5's "E1 is the correct stopping point" held only for *adding a
third consensus sleeve*. It did **not** consider upgrading a sleeve
with orthogonal stock-picking levers. Two new selection levers,
developed and overfit-screened on this branch
(`IMPROVE_PICK_RCD_FINDINGS.md`, `IMPROVE_PICK_RCE1_FINDINGS.md`):

1. **Regime-conditional select-blend weight** — Sleeve B's consensus/ml
   blend weight is a function of the audited `classify_regime_tight`
   label: momentum-lean (consensus w=0.30) in a confirmed bull,
   consensus-stable (0.60) in normal/recovery. Recovers WIN1's
   bull-market CAGR while keeping WIN2/blend's shallow drawdown.
2. **Conviction-adaptive breadth** — hold 2 names when the
   cross-sectional score gap signals genuine conviction, widen to 3
   when scores are bunched (the picker is guessing — historically the
   bad years).

E1's free-consistency decorrelation lever is **orthogonal** to these,
so they **stack**. **E2 = 0.5·WIN1 + 0.5·(RC D + adaptive-breadth)** —
i.e. Sleeve A unchanged (WIN1), Sleeve B = WIN2 upgraded with both
levers — measured on the canonical production pipeline:

| | E1 (prior) | **E2 (deployed)** |
|---|--:|--:|
| Full CAGR (lump-sum) | 51.9% | **56.6%** |
| Sharpe | 1.03 | **1.10** |
| Max DD (accum. DCA) | -56% | **-56%** (unchanged) |
| WF beats SPY | 10/10 | **10/10** |
| Eras beat S&P-DCA | 4/4 | **4/4** |
| Worst rolling-5y DCA | +11.7%/yr | **+13.6%/yr** |
| Forward CAGR ex-2003-09 | 33.4% | **38.2%** |
| TRUE-OOS holdout Sharpe | 1.08 | **~1.24** |

Strict Pareto improvement on E1: +4.7pp CAGR, +0.07 Sharpe, better
worst-5y and a +4.8pp higher forward (ex-front-load) CAGR, at E1's
*identical* drawdown, WF 10/10, 4/4 eras, 100% 10y DCA-win. Passes the
full battery: cost-insensitive (0-30 bps), wide 50/50 mix-weight
plateau, more delisting-robust than E1, strongest TRUE-OOS holdout of
any variant. The production sim reproduces the validated stream
bit-exactly (sleeve-B max|Δ ret_m| = 0.0; WIN1/E1 paths unchanged,
max|Δ| = 0.0).

**Honest caveat:** the -56% drawdown is the 2008 GFC systemic event; a
stock-picking lever cannot move it. E2 raises return and consistency
*at* E1's drawdown — it does not make the strategy low-risk. Lowering
the -56% floor is a regime-timing research program, not a picking knob.

**DEPLOYED 2026-05-17** (user-approved). `build_webapp_v5_pit.py`:
`run_full_sim` extended with `regime_blend_w` + `adaptive_k` (E1/WIN1
fallback bit-exact); new `run_e2_blend` + shared `_combine_5050`;
`STRATEGY_VARIANT='E2'`, `WINNER_NAME`, `STRATEGY_SPEC`, docstring
updated. `data.json` regenerated (headline CAGR 56.6%, Sharpe 1.10,
accumulating DCA Max DD -56%, 10y/5y/3y DCA-win 100%/99.5%/96%, 4/4
eras, combined live book RTX·PH·ETN·UBER). Homepage
(`docs/index.html`), the `/experiments/monthly-dca` dashboard, the
`STRAT_KEY`, and README fully synced; cron auto-regenerates via the
same builder. Full gauntlet: `IMPROVE_PICK_RCD_FINDINGS.md` +
`IMPROVE_PICK_RCE1_FINDINGS.md`.
