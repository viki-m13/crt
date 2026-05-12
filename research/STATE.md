# Research — Merged Index

**Last updated:** 2026-05-12
**Source:** consolidated from 5 hourly Routine runs (`claude/compassionate-planck-*`) into `research/runs/<name>/`.

## Mission

Develop a US equity stock-picking strategy that achieves, on a strictly walk-forward OOS basis:

- **CAGR ≥ 50%** net of realistic costs and slippage
- **Sharpe ≥ 2.0** annualized from monthly returns
- Robust across the full validation gauntlet
- Honest — no leakage, no survivorship bias, no OOS tuning

Universe: SPX or NDX (PIT). Monthly rebalance. Long-only, no leverage.

## Headline (as of 2026-05-12)

| Status | Result | Source |
|---|---|---|
| CAGR gate (≥50%) | **PASSED ✓** at 65.1% | run `4T9wE`, exp_012 (synthetic universe) |
| Sharpe gate (≥2.0) | **FAILED ✗** at 1.834 (gap 0.166) | run `4T9wE`, exp_012 (synthetic universe) |
| PIT SP500 validation | **see `research/validation/sp500_pit/`** | this commit |
| PIT NDX validation | **see `research/validation/ndx_pit/`** | this commit |

Winning config (4T9wE / exp_012):
- Universe: cached-feature panel (1022-ticker synthetic, NOT a PIT index)
- K=30, monthly rebalance, hold 1m
- Score: `z(LGBM_pred) × 0.70 + z(sharpe_12m) × 0.20 + z(sharpe_5y) × 0.10`
- LGBM: WalkForwardLGBM(train=48m, embargo=3m, min_train=24m), 200 trees, 31 leaves
- Weighting: inv-vol on `vol_12m`, capped at 5% per name
- Vol-target: `scale = min(0.18 / spy_vol_21d, 1.0)` (no leverage)
- Regime gate: 200ma_loose (invest iff `d_sma200(SPY) > -0.05`)
- Costs: 5 bps × 2 round-trip = 10 bps per rebalance

**Important caveat:** the 65.1% / 1.834 figure was measured on a non-PIT, cached-feature synthetic universe (`experiments/monthly_dca/cache/features/*.parquet`, 1022 tickers). It is NOT a PIT-correct out-of-sample number. The validations under `research/validation/` are the honest test of whether this recipe generalizes to the finalized PIT SP500 / PIT NDX panels.

## Runs (per-run subfolders preserve work verbatim)

| Run branch | Committed at (UTC) | Headline result | Key finding |
|---|---|---|---|
| `…xXasV` → `research/runs/xXasV/` | 2026-05-11 20:37 | 40.7% CAGR / 0.86 Sharpe (bootstrap) | Phase-1 framework only; no candidate yet |
| `…wnomX` → `research/runs/wnomX/` | 2026-05-11 21:30 | 47.2% CAGR / 1.06 Sharpe (re-cited v5) | Phase-2 baseline ladder + 22-idea backlog; cites prior v5 numbers as a baseline rather than running fresh |
| `…WolZI` → `research/runs/WolZI/` | 2026-05-11 23:41 | 39.9% CAGR / 0.89 Sharpe (best single config) | Found 2 critical bugs (MonthEnd-1 date arithmetic; regime-gate look-ahead). Inflated prior numbers from ~28% to ~97% CAGR until fixed. |
| `…ugEHG` → `research/runs/ugEHG/` | 2026-05-12 00:34 | 48.7% CAGR / 1.00 Sharpe (Donchian); 16.0% / 1.11 Sharpe (best Sharpe) | 116 experiments. Concluded **"Sharpe 2.0 + CAGR 50% structurally unreachable"** under the methodology it explored. |
| `…4T9wE` → `research/runs/4T9wE/` | 2026-05-12 01:03 | **65.1% CAGR / 1.834 Sharpe** | 12 experiments, 390 configs. CAGR gate clears; Sharpe ratio (mean_m/std_m) stuck at 0.527 vs 0.577 required for Sharpe-2.0. |

## Cross-run disagreement — and what to trust

`ugEHG` (Hour 4) concludes the joint target is "structurally unreachable" while `4T9wE` (Hour 5) gets 65.1% / 1.834 the very next hour. Both are right within their own frame:

- `ugEHG` ran on a **smaller universe** (`backtests/YLOka/runs/...` artifacts, K=3 conservative configs) and was correct that K=3 momentum portfolios are vol-bound around Sharpe 1.0–1.2.
- `4T9wE` ran on a **larger cached-feature universe** with K=30, vol-targeting, and a 3-way blend — which raises the Sharpe ceiling to ~1.83 by diversifying within the picks and scaling exposure inversely with SPY vol. Neither result is PIT-correct.

The PIT validations under `research/validation/` resolve this by retraining 4T9wE's winning config on the canonical PIT panels (PR #177 augmented SP500 + on-main NDX) with the **same walk-forward protocol — no cherry-picking**.

## Known structural issue with the Routine

Every hourly invocation pushes to a new branch (`claude/compassionate-planck-*`), so each run cannot see the previous run's `STATE.md`, `journal.jsonl`, `dead_ends.md`, etc., and re-bootstraps the framework from scratch. The five runs effectively re-derive overlapping baselines instead of compounding. **Fix**: point the Routine at a single fixed branch so disk state persists across invocations. (Not yet done — out of scope for this commit.)

## Lockbox

The Routine's instructions designate 2022-01-31 → 2025-12-31 as the lockbox window. None of the 5 runs report having touched it. Validation below similarly walks-forward on data ending **before** the lockbox, and **does not promote** any candidate into the lockbox window; this commit only reports walk-forward OOS, not lockbox OOS.

## Files in this directory

- `STATE.md` — this file
- `runs/<name>/` — verbatim copy of each hourly run's `quant_research/` tree
- `validation/sp500_pit/` — PIT SP500 augmented panel retrain of 4T9wE winner
- `validation/ndx_pit/` — PIT NDX retrain of 4T9wE winner
- `validation/REPORT.md` — honest validation results vs the 65.1% / 1.834 claim

## Older research (pre-Routine)

The following files predate the Routine and are kept in place for historical reference:

- `00_repo_map.md`, `00_repo_map_fhtzx.md`
- `01_engine_audit.md`, `01_engine_audit_fhtzx.md`
- `02_hypotheses.md`, `02_invention.md`
- `exp_01_concentration_sweep.md`, `exp_01_prerunner_v1.md`, `exp_02_tlt_fallback.md`
- `YLOka/`, `forensics/`, `graveyard/`
