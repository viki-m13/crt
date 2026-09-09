# PREREG7 — WIDE UNIVERSE (breadth 20 -> ~330), survivorship-free PIT S&P 500
Panel 800 syms x 2,666 days 2016-2026; TEST 2024+ excluded at load.
PIT members/day: mean 332 (min 293, max 378). After $20M trailing ADV: 329/day.

## Headline: the candidate did NOT survive breadth
 test  split  ndays  gross bp/day      t   ann Sharpe
 W1    TRAIN   1501        -1.54   -1.07      -0.44    intraday reversal -> overnight
 W1    VALID    500        +0.92   +0.43      +0.30
 W2    TRAIN    750        -5.15   -1.88      -1.09    ^ on high-dispersion days (= I14)
 W2    VALID    381        +0.31   +0.12      +0.09
 W3    TRAIN   1501        +6.51   +2.90      +1.19    overnight reversal -> intraday
 W3    VALID    501        +7.65   +1.82      +1.29

I14 -- the only one of 33 hypotheses to clear edge/cost>1.0 on both splits at
N=20 -- is FLAT-TO-NEGATIVE at N=330 (W2: TRAIN -5.15, VALID +0.31).
=> The 20-name result was not a breadth artifact hiding a real edge. It was noise.

## W3 surfaced instead, and carries two suspected fatal flaws
 (1) NOT EXECUTABLE: signal uses open_D, entry is AT open_D -> simultaneous.
 (2) SHARED-PRICE ARTIFACT: signal -(open/prev_close) and target (close/open)
     both contain open_D with opposite sign. Noise in the open print manufactures
     exactly this "reversal" (bid-ask bounce).
20-symbol intraday check (entry AT open vs 09:35): TRAIN -2.04 -> -4.94 bp,
VALID -3.51 -> -7.42 bp. Negative at both entries and WORSE when executable.
Wide-universe controls running: (a) target that does not share open_D,
(b) liquidity terciles, (c) pre-registered breadth curve.

## CONTROL 1 — the shared-price artifact is confirmed, and it is ENORMOUS
 target close/open      (shares open_D)      VALID  +7.65 bp/day  t +1.82
 target close/prev_close(shares close_{D-1}) VALID -190.59 bp/day t -37.48
Both targets share a price with the signal, in OPPOSITE directions, and the
measured "effect" flips sign and explodes to -190 bp. Bid-ask bounce in the
shared price dominates by ~25x. The +7.65 bp W3 is a residue of that mechanism,
not an effect. W3 is DEAD.

## CONTROL 2 — concentrated in the noisiest names, as bounce predicts
 least liquid third VALID +9.14 bp (t+2.40) > most liquid third +8.00 (t+1.53)

## BREADTH CURVE — MY OWN PIVOT RATIONALE IS REFUTED
 N=20  t +1.27 | N=50 +1.56 | N=100 +1.40 | N=200 +1.67 | N=400 +1.82 | full +1.82
Going 20x wider (20 -> 400 names) raised t by 1.43x. Independent breadth would
give sqrt(20) = 4.5x. Daily cross-sectional equity returns are dominated by a
common factor, so EFFECTIVE breadth is roughly 2x, not 20x.
=> The 33 earlier rejections were NOT breadth-limited. They were honest.
   Widening the universe does not rescue them. This was worth knowing and it
   closes the line rather than opening one.

## FINAL: 34 hypotheses pre-registered and tested. 0 tradable. 0 candidates.
   TEST 2024-01-01..2026-08-11 NEVER READ.
