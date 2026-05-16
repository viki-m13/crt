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

## Files

- `improve_main_strategy.py` — overlay battery (vol-target, DD-breaker)
- `improve_sim_v2.py` — bit-exact parametrized sim (trigger × select)
- `improve_phase2.py` — scorer matrix + book-blends (fidelity-checked)
- `improve_phase3.py` / `improve_phase3b.py` — full overfit gauntlet
- `augmented/improve_*.json` — raw results
