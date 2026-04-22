# Step 50 — Time-decay and regime-adaptive exits

**Q:** Should TP/SL/hold be dynamic — shrink targets as trades age, widen time-stops in bear markets?

**Verdict:** Marginal. Static TP10/SL15/TS252 already captures the edge. Regime tricks nudge CAGR a few bps but shift drawdowns into the wrong periods.

## Control replication
- Step41 winner (TP10/SL15/TS252, single-slot, CAP5+SMA12M top-1): **CAGR +11.13%, MDD 48.0%, Calmar 0.232, WR 70.5%**, N=122. Matches production.
- SPY DCA baseline: CAGR +6.63%, MDD 35.9%, Sharpe 1.11.

## Top per variant

| Variant | Best config | CAGR | MDD | Calmar | WR |
|---|---|---|---|---|---|
| Control | TP10/SL15/TS252 | 11.13% | 48.0% | 0.232 | 70.5% |
| **A** TP decay | `A_step_10_5` (10→5% at 126d) | **11.57%** | 47.1% | 0.246 | 71.0% |
| **B** 200dma TS | `bull189_bear504` | 11.13% | 48.0% | 0.232 | 70.5% |
| **C** 200dma TP | `bull10_bear20` | **11.70%** | 47.7% | 0.246 | 65.1% |
| **D** Vol-regime TS | `vol25_bull252_bear504` | 11.13% | 48.0% | 0.232 | 70.5% |
| **E** Combined | `A + B` | 11.57% | 47.1% | 0.246 | 71.0% |

## Observations

1. **B and D are no-ops.** Time-stops almost never bind — most trades exit at TP within 30 days. Extending bear-hold 252→504 changes nothing; shortening bull-hold to 63/126 hurts slightly by cutting slow winners.
2. **A (TP decay) is neutral-to-slightly-positive.** A single step 10→5% at 126d adds 44bps CAGR, trims MDD ~1pt. Deeper/faster decays (→2%, →0%, breakeven) hurt — they exit trades at losses that would have later hit 10%.
3. **C (regime TP) has the only real asymmetry — but it's a trap.**

## Crisis-period analysis

Strategy equity period returns (and window MDD):

| Strategy | GFC 2007-10→2009-06 | COVID 2020 | 2022 Bear |
|---|---|---|---|
| SPY DCA | +46.1% (35.9%) | +19.2% (33.9%) | -17.9% (23.9%) |
| **Control** | **+94.2% (42.9%)** | +8.9% (30.8%) | +82.8% (14.2%) |
| A_step_10_5 | +94.2% | +8.4% | +82.6% |
| B_bull189_bear504 | +94.2% | +8.9% | +82.8% |
| **C_bull10_bear20** | **+0.1% (47.7%)** | +16.0% | +35.5% |
| C_bull10_bear15 | +51.4% | +17.7% | +62.5% |
| E_combo (A+B) | +94.2% | +8.4% | +82.6% |

**The CAGR leader (C_bull10_bear20) is a disaster in the GFC** — it gives up ~94pts of GFC return vs control by raising TP to 20% in bear regimes. Trades enter bear, don't hit the taller target, and ride the crash back down. Full-window CAGR still looks good because post-2010 bull-regime trades dominate, but it is the *opposite* of what a macro-aware risk manager should do. **Widening TP in bear is the wrong direction: in crashes, take what you can.**

## Recommendation

**Keep the static TP10/SL15/TS252 production config.**

- **Reject** regime-aware TP widening. The post-2009 CAGR bump is an artifact of three non-crisis bear regimes (2011, late-2018, 2022); it inverts catastrophically in the one true crash (GFC).
- **Reject** regime-aware time-stops (B and D). Trades exit at TP long before any time-stop binds, so the knob has no effect (and when it does, it hurts).
- **TP time-decay (`A_step_10_5`, 10→5% at 126 bars)** is a marginal improvement: +44bps CAGR, -0.9pts MDD, +0.5pts WR, and it is directionally sound (asking for less as a reversion ages is sensible). It behaves identically to control in all three crisis windows. If you must adopt one adaptive rule, this is it — but 44bps inside a single-path backtest is within noise.

Bottom line: the strategy already extracts most of its edge via TP=10%. The 48% MDD is a **sizing** problem (single-slot concentration), not an exit-rule problem. For MDD reduction, multi-slot (step43) is the lever.
