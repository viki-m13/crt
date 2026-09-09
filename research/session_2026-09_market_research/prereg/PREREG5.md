# PRE-REGISTRATION v5 — BATCH 2, 12 IDEAS, JUDGED ON MONEY NOT IC
Frozen 2026-09-04 before any v5 result is computed.

## Debt
Trials this session: 3 (v1-v3) + 12 (batch 1) = 15. This batch adds 12 -> 27.
TEST 2024-01-01..2026-07-31 still LOCKED, never read.

## Methodological change forced by batch 1
Batch 1 proved a de-overlapped IC of -0.23 (t -24) monetizes to +0.07 bp/day
(t +0.02). IC IS NO LONGER AN ACCEPTANCE CRITERION. Every idea below is judged
by SIMULATED NET P&L after measured Corwin-Schultz spread, day-clustered t.
IC may be reported as diagnostic only.

## Primary statistic (identical for all 12)
Cross-sectional quintile long/short, equal weight, held over the stated horizon.
gross = mean(top quintile) - mean(bottom quintile), per day.
cost  = sum of both legs' round-trip Corwin-Schultz spread.
net   = gross - cost.  t = day-clustered on net.
PASS requires ALL of: (a) net t > Holm threshold across the 12 on VALID,
(b) same sign on TRAIN, (c) edge/cost = gross/cost > 1.0 on VALID.

## THE 12 IDEAS (no substitutions)
 I13 Volume-clock reversion MONETIZED: trade at volume-bar boundaries, hold to
     matched horizon. (batch 1's only IC survivor -- does it pay?)
 I14 I1 restricted to high cross-sectional dispersion days (I7 x I1)
 I15 Opening gap (open vs prev close) -> first-30-min return
 I16 Opening-range fade: first-30-min return -> last-30-min return
 I17 Market-neutral I1: residualize ret_last30 on SPY's ret_last30 first
 I18 Close-auction pressure: last-5-min return (not last-30) -> overnight
 I19 Cross-asset overnight: SPY last-30m predicts names' overnight beyond own
 I20 Vol-scaled I1: divide signal by that day's realized vol before ranking
 I21 Day-of-week / turn-of-month conditioning on I1
 I22 Two-day short-term reversal -> next-day close-to-close
 I23 Volume-clock version of the OVERNIGHT signal (last 30m measured in volume
     time, not wall time)
 I24 Liquidity timing: I1 traded only in the cheapest spread tercile per symbol

## Controls (required, same as batch 1)
 C-A shuffle null on any idea that passes
 C-B orthogonality vs I1 -- any survivor must add beyond plain overnight reversal
 C-C harness check: oracle signal must return strongly positive in the same sim
