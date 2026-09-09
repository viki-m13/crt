# PRE-REGISTRATION v11 — DOES THE FLOW-FADE SURVIVE MAKER ADVERSE SELECTION?
Frozen 2026-09-05 before any v11 number is computed. Trial 38.

## What is being tested
PREREG10 established a REAL, replicating, non-artifact effect: heavy taker BUY
flow predicts negative forward returns (VALID -1.31 bp t-3.96 at k=15, and it
survives the t+1 bounce control). It is 6-12x too small for taker execution.
A maker fill does not pay the spread, it EARNS price improvement. Question:
does the edge survive the adverse selection that comes with being filled?

## Why this is not the pt-7/pt-8 maker thesis this account already killed
That one ASSUMED a fill rate (0.50) and applied a heuristic adverse bias.
Here fills are SIMULATED AGAINST THE ACTUAL 1-MIN HIGH/LOW PATH: an order at
level L is filled only if the market actually traded through L. Adverse
selection is therefore MEASURED, not modeled.

## Construction (causal)
At each event t in the top flow quintile (heavy taker BUY -> we want to be short):
  post a limit SELL at L = c_t * (1 + delta),  delta in {5, 10, 20} bp
  fill if max(high) over the next W=5 minutes >= L; fill price = L
  QUEUE HAIRCUT: require high >= L strictly above, not merely touching
  on fill at minute f: hold k in {15, 30} minutes, exit TAKER at close(f+k)
  fees: maker 2 bp on entry, taker 5 bp on exit (Binance perp VIP0)
Mirror the whole thing for the bottom flow quintile (heavy taker SELL -> buy).

## Pre-registered tests
 A1 ADVERSE SELECTION, measured: forward return conditional on FILL vs the
    unconditional forward return for the same events. The gap IS adverse
    selection. Report it in bp; it is the number that killed the earlier thesis.
 A2 NET P&L per FILLED trade, after 2 bp maker + 5 bp taker. Must be > 0.
 A3 THE CONTROL THAT DECIDES IT: run the identical maker simulation on RANDOM
    events (no flow condition). Plain market-making also earns delta. The flow
    version must beat the random version, day-clustered t > 3 on VALID.
    If it does not, the signal adds nothing and this is just market-making.
 A4 same sign on TRAIN and VALID; effect in >= 6 of 8 symbols.
 A5 fill rate must be reported. A strategy that fills 2% of the time is not a
    strategy at this size.

## Kill
A2 <= 0, or A3 fails, or A4 fails -> the flow-fade is not monetizable as a
maker either. Report, close the venue, move on. No re-tuning of delta/W/k
after seeing results -- the grids above are fixed now.
