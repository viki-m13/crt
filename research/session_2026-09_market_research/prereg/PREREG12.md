# PRE-REGISTRATION v12 — ADVERSE-SELECTION EFFICIENCY (ASE)
Frozen 2026-09-05 before any ASE number is computed. Trial 39.

## The invention
PREREG11 measured, on Binance majors, that maker price improvement is competed
away EXACTLY: post at +delta, and the fill-conditional move is ~-1 bp for
delta = 5, 10 and 20 bp. That is an EQUILIBRIUM (makers compete until the
spread just compensates adverse selection), not a quirk.

Define      ASE = adverse_selection / delta
  ASE = 1  -> price improvement fully competed away. No maker edge. (Binance majors)
  ASE < 1  -> liquidity provision is UNDER-COMPENSATED for its risk; the maker
              keeps delta*(1-ASE) per filled trade as a STRUCTURAL rent.
ASE is a property of a VENUE's competitive state, not a forecast. It predicts
nothing. This has not to my knowledge been published or used as a screen.

## Why this and not a 40th signal
38 prediction hypotheses failed against a cost wall. This inverts the target:
it looks for where the cost wall itself is mispriced. The mechanism (capital
and infrastructure have not shown up in a venue) persists -- it is not competed
away by being known. The account's only live-validated winner (Polymarket
weather fade) is exactly an under-competed venue, so the prior is good here.

## Measurement (identical across venues, matched 15-min bars)
At each bar t: post limit SELL at L = c_t*(1+delta) and limit BUY at c_t*(1-delta),
  delta in {10, 25, 50} bp.
Fill: within the next W=1 bar, sell fills if high > L (strictly through),
      buy fills if low < L. Queue haircut = strict inequality.
On fill, hold k in {1, 2, 4} bars, exit at close.
  filled_move    = realized P&L of that position, in bp (before fees)
  uncond_move    = same P&L computed over ALL bars, ignoring the fill filter
  adverse_sel    = uncond_move - filled_move
  ASE            = adverse_sel / delta
Venues: Binance perps (1m->15m, 8) = CALIBRATION NULL, expected ASE ~1.
        US equities (1m->15m, 20) = highly competed control.
        Builder-DEX tokenized equities (native 15m, ~53) = the candidate venue.

## Pre-registered tests
 E1 CALIBRATION: Binance majors must reproduce ASE ~ 1.0 (0.85-1.15). If the
    scanner does not recover the known null, it is broken and nothing else counts.
 E2 RANKING: report ASE per asset and per venue class.
 E3 Any asset with ASE < 0.8 must ALSO show positive net maker P&L after that
    venue's real fees, or the ASE gap is not economically real.
 E4 STABILITY: split each asset's history in half; ASE must have the same sign
    of deviation from 1.0 in both halves.
 E5 SAMPLE HONESTY: the builder DEX has ~5,000 bars (~2 months). Any finding
    there is PRELIMINARY by construction and must be labelled so. It cannot be
    called validated on this sample regardless of t-stat.

## Kill
E1 fails -> scanner broken, discard everything.
No venue with ASE < 0.8 -> the equilibrium holds everywhere I can see, report
that as the finding and stop claiming a maker opportunity exists.
