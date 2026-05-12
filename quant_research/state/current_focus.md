# Current Focus

**Updated:** 2026-05-12 (Run 1 — Bootstrap)

## This Hour's Plan (completed)

1. ✅ Bootstrap directory structure
2. ✅ Data integrity checks
3. ✅ Baseline ladder (rungs 1-5)
4. ✅ Engine parity verification vs v3 (39.4% CAGR, 0.949 Sharpe — within 0.4pp of ground truth)
5. ✅ Ideas backlog (20 ideas)
6. ✅ STATE.md + journal

## Next Run Plan

**Priority: k=2 concentration sweep + IC-weighted regime gate**

Target: find a configuration that pushes WF CAGR toward 50% while keeping Sharpe ≥ 1.2.

Steps:
1. Run 8-variant sweep: k ∈ {2,3} × hold_months ∈ {3,6} × weighting ∈ {ew, invvol}
   - Use OOS window 2008-01 to 2024-01 (consistent with our engine)
   - Fix: v3 ML signal (ml_preds_v2, 3plus6 blend)
   - Record: CAGR, Sharpe, MaxDD, WF splits
2. IC-weighted gate: add ic_filter to regime (invested only when rolling_ic_6m ≥ 0.03)
3. If k=2 gives > 45% WF CAGR → run k=2 parameter perturbation test
4. Report which approach gives best risk-adjusted results

**Time budget:** ~45 min execution, ~15 min analysis

## Key Constraint Reminder
- No leverage
- Monthly rebalance at month-end
- Long-only, SPX universe
- Cost floor 5 bps (using 10 bps for v3-comparable)
- Lockbox: 2024-02 to 2026-05 is SEALED (do not simulate past 2024-01-31)
