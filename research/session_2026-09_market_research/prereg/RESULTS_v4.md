# BATCH 1 RESULTS (PREREG4, 12 ideas) — 2026-09-04
Panel: 29,961 stock-days, 20 symbols, 1,509 days, 2018-01-02..2023-12-29.
TEST 2024-01-01..2026-07-31 NEVER READ. Median Corwin-Schultz spread 2.72 bp.

## Cross-sectional ideas (per-day Spearman IC, day-clustered t)
 id  target  TRAIN IC      t  |  VALID IC      t
 I1  T_on    +0.0514  +5.06  |  +0.0631  +4.19   overnight reversal of last-30m move
 I2  T_on    +0.0501  +4.58  |  +0.0440  +2.63   close-vs-VWAP dislocation
 I3  T_cc    -0.0124  -1.30  |  -0.0103  -0.75   semivariance asymmetry
 I4  T_cc    -0.0155  -1.67  |  -0.0093  -0.66   efficiency-ratio trend continuation
 I11 T_on    +0.0219  +2.49  |  +0.0108  +0.85   1-min return skewness
 I12 T_cc    -0.0024  -0.29  |  -0.0126  -1.01   range-compression breakout

## Conditional ideas
 I5 Amihud as amplifier of I1: VALID spread -0.043 (WRONG SIGN: illiquid names
    are WEAKER, not stronger). TRAIN spread -0.010. Hypothesis backwards.
 I6 Trade-count surprise gate: TRAIN spread +0.022, VALID spread -0.024.
    SIGN FLIPS across splits -> dead.
 I7 Dispersion regime gate on I1: TRAIN low/high 0.046/0.057; VALID 0.037/0.089.
    Direction consistent (high dispersion stronger) but VALID low-disp t=1.85.

## Intraday ideas
 I8 Volume clock beats wall clock: |IC| 0.370 vs 0.330 TRAIN (t+25.1),
    0.370 vs 0.330 VALID (t+15.5). Replicates. NOT YET MONETIZED.
 I9 Slow band (12/48) vs fast (3/12), horizon-matched non-overlapping trades:
    BOTH bands have gross P&L indistinguishable from zero.
      VALID fast gross +0.07 bp/day (t+0.02), slow -4.72 bp/day (t-1.52)
      cost 14.8 bp/day -> NET -14.8 / -19.6 bp/day. edge/cost = +0.005 / -0.318.
 I10 Reversion IC by hour: -0.67 (09:30-11:00), -0.46 (11:00-13:00),
    -0.46 (13:00-15:00). Monotone decay, replicates TRAIN->VALID.

## THE CENTRAL FINDING OF THIS BATCH
The intraday reversion IC is REAL and survives de-overlapping
(non-overlapping IC -0.217 TRAIN / -0.234 VALID, t -36 / -24) yet monetizes
to ZERO gross. Harness verified: perfect-foresight signal returns
+299.6 bp/day (t+56.2), random returns +6.0 (t+1.4), the real signal +0.1 (t+0.02).
=> A large, replicating, de-overlapped IC is NOT evidence of tradable edge.
This retracts the earlier "edge/cost 3.41 at k=120" claim, which was computed
from IC and an assumed spread rather than from a simulated trade.

# BATCH 1 FINAL (controls returned)
 C-A shuffle null: I1 real|t|4.19 vs null-99th 2.39 PASS; I2 2.63 vs 2.54 PASS
     (note: the null's own 99th pct is 2.39-2.54, so a raw |t|>2 bar is too lenient)
 C-B orthogonality: I2 residualized on I1 -> t +2.92 TRAIN, +1.65 VALID. I2 adds
     nothing beyond I1 out of sample. corr(I1,I2)=+0.37.
 I1 ECONOMICS: XS quintile L/S, close->next open, round-trip spread both legs
     TRAIN gross +11.99 bp/day (t+2.40) cost 7.39 NET +4.59 (t+0.92) e/c 1.62
     VALID gross  +6.41 bp/day (t+1.18) cost 8.33 NET -1.92 (t-0.35) e/c 0.77
 HOLM across 12: only I8 (15.47) and I1 (4.19) clear. I2 (2.63 vs 2.81) fails.
 => BATCH 1: 12 ideas, 0 tradable. I1 statistically real, economically underwater.

# BATCH 2 (PREREG5) — judged on NET P&L, not IC
 I13 volume-clock reversion MONETIZED: VALID gross +2.13 (t+0.93) cost 12.37
     NET -10.24 (t-4.47) e/c 0.17.  => batch 1's IC survivor does NOT pay.
 id   VALID gross(t)      cost   NET(t)        e/c   verdict
 I14  +16.73 (+2.05)      8.99   +7.74(+0.95)  1.86  best candidate, not significant
 I15   +1.65 (+0.43)      8.16   -6.52(-1.69)  0.20  dead
 I16   +3.35 (+1.98)      8.90   -5.55(-3.30)  0.38  dead (gross real, cost kills)
 I17   +6.41 (+1.18)      8.33   -1.92(-0.35)  0.77  dead
 I18   +4.61 (+0.94)      7.95   -3.33(-0.68)  0.58  dead
 I20   +4.45 (+0.94)      6.83   -2.39(-0.51)  0.65  dead
 I21   -2.73 (-0.24)      8.34  -11.07(-0.96) -0.33  dead, sign flips
 I22   +7.91 (+0.91)      8.63   -0.72(-0.08)  0.92  dead
 I23   +6.83 (+1.25)      8.33   -1.50(-0.27)  0.82  dead
 I24   +3.49 (+0.64)      4.95   -1.46(-0.27)  0.70  dead (cheap names, weak signal)
 => BATCH 2: 12 ideas, 0 pass all three criteria.

# I14 detail (only survivor of the economic bar)
 HIGH dispersion days: TRAIN e/c 1.84, VALID e/c 1.86 (gross t +1.76 / +2.05)
 LOW  dispersion days: TRAIN e/c 1.31, VALID e/c -1.10 (sign flips)
 Consistent contrast, but net t +0.95 on VALID fails the Holm bar. NOT a pass.

# CUMULATIVE: 24 ideas pre-registered and tested. 0 tradable. 1 candidate (I14).

# PREREG6 — THE ARROW OF TIME (ATI). N0 FAILED AS SPECIFIED.
Ordinal-pattern irreversibility, 29,961 stock-days, 20 syms, 2018-2023.
Null = phase-randomized surrogate (IDENTICAL power spectrum, time-reversible
by construction), 20 per stock-day.

 ATI_raw          0.09738
 surrogate        0.06567      -> +48.3% excess
 excess, day-clustered t  +80.7 over 1,509 days
 mean per-stock-day z     +1.015   (pre-registered bar was +3)  -> N0 FAIL
 per-day SNR 0.59; 20-day causal pooling only reaches 1.11 (sd 1.710 -> 0.916,
 not the 0.38 iid pooling would give) => the residual spread is a PERSISTENT
 SYMBOL effect, not day-level signal.

## What is nonetheless established
Markets ARE time-irreversible, emphatically (t +80.7), and this is INVISIBLE to
every statistic in v1-v3 by construction (|FFT|^2 discards phase).
It is a SYMBOL-level property, stable and strongly differentiated:
  XLE 2.72, NVDA 2.14, NFLX 2.00, EEM 1.83 ... SPY 0.34, MSFT 0.20, AAPL 0.15, META 0.09
  (every symbol t = +2.8 to +58.9)
Economically coherent: the most efficiently arbitraged mega-caps are nearly
time-reversible; sector/EM/flow-driven names are markedly irreversible.

## Why it still is not a strategy
- Day-level conditioning (the actual proposal, H2) is not measurable: SNR 0.59.
  Dead as pre-registered. H1-H5 NOT RUN.
- Symbol-level H1 threat is live: spearman(ATI, 1-min skew) = +0.556. Skewness
  is itself a time-reversal asymmetry, so part of this is definitional.
  R^2 on (log dollar vol, rv, spread) = 0.263, so 74% is not liquidity -- but
  the skew overlap is untested and would have to be projected out.
- 20 symbols is far too thin a cross-section to trade a symbol-level score.
  Testing it properly needs hundreds of names = a data-acquisition problem.

## RUNNING TOTAL: 33 hypotheses pre-registered and tested. 0 tradable.
   1 candidate (I14, e/c 1.86 but net t +0.95). TEST 2024-2026 STILL SEALED.
