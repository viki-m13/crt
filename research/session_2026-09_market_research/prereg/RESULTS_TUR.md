# PREREG8/9 — THERMODYNAMIC CEILING ON ALPHA (Sharpe_step <= sqrt(sigma/2))
Novel construction: treat a strategy's P&L as a thermodynamic current, apply the
Barato-Seifert Thermodynamic Uncertainty Relation, and BOUND alpha from the price
path alone instead of searching for it. sigma = coarse-grained entropy production
from ordinal-pattern KL(forward || reversed), de-biased with phase-randomized
surrogates. 20 symbols, ~600k 1-min returns each, 2018-2023.

## v8 was VOID by its own control (reported, not buried)
T4 failed: surrogate/real oracle = 0.949. Design error, not a market fact --
a phase-randomized surrogate PRESERVES the power spectrum, hence preserves the
linear autocorrelation (bounce) the oracle exploits. Wrong null for the quantity.
sigma (bound side) unaffected: irreversibility is a phase property.

## v9 corrected: OOS oracle + IID-shuffle null
 T4'  IID shuffle 0.00115 vs real OOS 0.00963, ratio 0.120        PASS
 T1'  violations of oracle_OOS <= sqrt(sigma/2):  0 / 20          PASS
 T2'  median oracle_OOS / bound = 0.120 (bound ~8x above achieved) PASS
 T3'  spearman(bound, oracle_OOS)          = +0.164               FAIL (need +0.5)
      spearman(bound, NON-BOUNCE achieved) = -0.394               FAIL, WRONG SIGN

## Verdict
The INEQUALITY SURVIVES: never violated in 20/20, with only ~8x headroom, so it
is a real and non-trivial constraint.
The SCREEN FAILS: the bound does not rank assets by achievable edge -- it ranks
them BACKWARDS. High-sigma names (XLE .067, NVDA .044, NFLX .040) have zero or
negative non-bounce OOS edge; low-sigma names (META .001, AAPL .002, SPY .003)
hold the small positive residue.

## Why (the interesting part)
Entropy production measures information ARRIVAL. Information that is arriving and
being correctly incorporated is exactly what you CANNOT trade against -- it is
permanent impact, already in the price. So dissipation is real, is measurable,
and is anti-correlated with exploitability. The physics bound is sound; the
economic reading I hoped for is the opposite of true.

## sigma is nonetheless a real, stable, 65x-ranging asset property
XLE .0669 > NVDA .0442 > NFLX .0400 > EEM .0360 ... SPY .0030 > AAPL .0020 > META .0010
(same ordering as the ATI measurement, t = +2.8 to +58.9 per symbol)

## RUNNING TOTAL: 36 hypotheses. 0 tradable. TEST windows never read.
