# PREREG11 — MAKER EXECUTION OF THE FLOW-FADE.  KILLED.
Fills simulated against ACTUAL 1-min high/low paths (strict through-fill), not
an assumed fill rate. 8 Binance perps, 2023-2025. TEST 2026 locked.

## A1 — THE RESULT WORTH KEEPING: adverse selection scales 1:1 with the offset
 post limit SELL delta bp above market:
 delta   fill rate   move if UNFILTERED   move CONDITIONAL ON FILL   adverse selection
   5 bp     0.70            +5.2 bp              -0.6 bp                 -6.2 bp
  10 bp     0.52           +10.6 bp              -0.4 bp                -11.0 bp
  20 bp     0.29           +20.6 bp              -0.4 bp                -21.0 bp
Conditional on being filled, the subsequent move is ~-1 bp REGARDLESS of how far
out you post. The price improvement delta is competed away EXACTLY, one for one.
Posting further from the market buys nothing: you simply get filled only when
the market is about to run through you.

## A2 — net P&L per filled trade: -6.97 to -7.97 bp in EVERY config
Gross is ~-1 bp; the 7 bp of fees (2 maker + 5 taker) is the whole loss.
TRAIN and VALID agree to within 1 bp across all 6 delta x hold cells.

## A3 — the deciding control: flow-conditioned vs RANDOM events
flow beats random by +0.41 to +1.41 bp. TRAIN t up to +3.59, but
VALID t = +0.67 .. +2.20 -- NONE clears the pre-registered t>3 bar.
The flow signal adds a genuine sliver to market-making, too small to matter.

## Verdict
A2 fails (negative in all 12 cells), A3 fails on VALID. Per PREREG11: the
flow-fade is NOT monetizable as a maker either. Venue closed for this signal.
This also explains the account's earlier pt-7/pt-8 maker failure mechanistically:
it is not that fills were mismodelled, it is that maker price improvement is
exactly and completely offset by adverse selection.

## RUNNING TOTAL: 38 hypotheses. 0 tradable.
