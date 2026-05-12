# Quant Research — Current State

**Last updated:** 2026-05-12 (Run 1 — First bootstrap run)
**Branch:** claude/compassionate-planck-ExCgK

---

## Headline

**Bootstrap complete. Engine verified to 99% parity with v3 ground truth. Ready for candidate experimentation.**

---

## Benchmark Table (Ground Truth from Prior Work)

| Strategy | CAGR | Sharpe | WF Mean CAGR | WF Min CAGR | Notes |
|---|---|---|---|---|---|
| SPY buy-hold | 11.9% | 0.78 | — | — | Reference |
| **v3 ML baseline** | **39.8%** | **0.955** | **42.8%** | **14.5%** | Ground truth (v6/run_baseline.py) |
| v5 + Chronos-bolt-tiny filter | 44.8% | 1.04 | 45.9% | 17.0% | Best prior no-leverage result |
| v8 (k=2, h18, 2.5× leverage) | ~94% | 1.09 | 93.8% | — | VIOLATES constraints (leverage) |
| **Our engine parity check** | **39.4%** | **0.949** | — | — | Within 0.4pp of v3 ground truth ✓ |

**Factor ladder (our engine, OOS 2008-2024):**
| Rung | CAGR | Sharpe | Notes |
|---|---|---|---|
| R1: 12-1 momentum | 10.0% | 0.527 | Weak alone |
| R2: Mom + low-vol | 8.5% | 0.538 | Vol filter hurts |
| R3: Mom + quality | 12.5% | 0.795 | **Best factor rung** |
| R4: + regime gate (200dma) | 8.8% | 0.689 | Gate hurts here |
| R5: OLS composite | -0.6% | 0.089 | Unstable weights — DEAD |

---

## Current Focus (Next Run)

**Experiment 002: k=2 concentration sweep + IC-weighted regime gate**

Goal: Find a configuration ≥ 45% WF CAGR, ≥ 1.2 Sharpe without leverage.
Approach: sweep k ∈ {2,3} × hold_months ∈ {3,6} × weighting ∈ {ew, invvol} (8 variants).
Then add IC-gate filter.

---

## Top 3 Next Steps

1. **Run exp_002**: k=2 concentration + IC-weighted gate. Expected to push WF CAGR toward 50%.
2. **Regime-conditional model** (exp_003): Separate models for bull/crash/recovery regimes. 
   Highest theoretical alpha lift.
3. **Chronos ensemble** (exp_004): Expand v5 Chronos from filter to score blending.

---

## Honest Assessment

### Achievable vs target
- **CAGR ≥ 50% target**: Plausible with k=2 concentration + better model. v8 hit 50%+ WF 
  at k=1 (but was fragile). k=2 with v3 signal might sustain it.
- **Sharpe ≥ 2.0 target**: Very likely impossible in long-only monthly equities WITHOUT leverage. 
  Theoretical analysis: for 50% CAGR, need monthly std ≤ 5.96%. Current v3 portfolio std ≈ 10.3%.
  Best achievable without leverage: probably 1.2-1.5 Sharpe.
  **RECOMMENDATION FOR V:** Consider relaxing Sharpe target to 1.5 (still exceptional for 
  long-only equity) or allowing some form of crash protection (hedged via puts or cash-heavy 
  crash regime). Sharpe 2.0 would require market-neutral components.

### Data notes
- Universe: S&P 500 PIT membership 2003-2026 (280 monthly snapshots, ~500 members each)
- Feature data: 79 features per stock, 353 monthly snapshots (1997-2026)
- Predictions: ml_preds_v2.parquet (the correct v3 source, 2003-09 to 2025-12)
- Lockbox: last 24 months = 2024-02 to 2026-05 (UNTOUCHED, 0 touches logged)
- No fundamentals in dataset (price-only). Quality proxies via price-based metrics.
- Survivorship note: 9 truly delisted tickers, MC overlay α=4% gives bias-corrected CAGR -8pp

### Questions for V
1. **Sharpe 2.0 target**: Is this a hard requirement or aspirational? Theoretical ceiling 
   for long-only monthly equities is ~1.5 without leverage. Should we allow crash hedging 
   (go to T-bills/GLD/TLT during crash) while holding the "no derivatives" constraint?
2. **Lockbox definition**: The code uses last 24 months = 2024-02 to 2026-05. Is 2024-02 
   the right start? The last date in the data is 2026-05-07.
3. **Transaction costs**: We use 10 bps round-trip (same as v3). For monthly rebalance of 
   large-cap S&P 500 names, is 5 bps (CLAUDE.md floor) more appropriate?

---

## ETA to Success Gate Pass

**Honest estimate: Unknown.** Given the Sharpe 2.0 theoretical impossibility without leverage,
if the constraint is truly hard: never (long-only monthly equity cannot achieve 2.0 Sharpe at
50% CAGR). If relaxed to Sharpe 1.5: potentially 4-8 runs from now (2-3 weeks at hourly cron).

---

## Total Hypotheses Tested (for DSR calculation)
- This run: 8
- Cumulative: 8
- DSR deflation will apply once we have a candidate: negligible at this count.
