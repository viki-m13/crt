# PREREG12 — ADVERSE-SELECTION EFFICIENCY (ASE) SCANNER
684 asset-configs, 69 assets, 3 venue classes, matched 15-min bars.
Fills simulated against actual high/low with strict through-fill.

## E1 CALIBRATION — PASSED. The instrument works.
Binance perps recover the known competitive equilibrium exactly:
  median ASE = 1.005   (delta=10bp: 1.052, 25bp: 0.995, 50bp: 0.992; std 0.013-0.033)
Maker price improvement is competed away one-for-one. This is a clean,
reproducible measurement of a market-making equilibrium.

## E2 — NO VENUE IS UNDER-COMPETED. The hypothesis fails.
  venue          n    median ASE
  binance-perp   72      1.005     <- at equilibrium
  builder-dex   432      1.175     <- WORSE than equilibrium
  us-equity     180      1.213     <- worst
ASE > 1 means adverse selection EXCEEDS the offset: naive liquidity provision
at these offsets loses more than the price improvement pays. The opposite of
the opportunity I predicted. Gross P&L per filled trade, before fees:
  binance -0.09 bp | builder-dex -4.29 bp | us-equity -4.35 bp (medians)

## The tail, and why it does not rescue it
Only 3 of 684 cells fall below ASE 0.8, all builder-DEX (JPY delta=10, and
JP225/AAPL at delta=50). They are STABLE across both halves (E4 pass) and show
+2.5 to +9.8 bp gross. But the liquidity control kills them:
  ASE<0.9 cells: 15 trades/bar, 11.1% zero-volume bars, 23.4% flat bars
  ASE>=0.9 cells: 35 trades/bar,  2.1% zero-volume bars,  4.9% flat bars
  spearman(ASE, trades per bar) = -0.352
The sub-0.8 readings sit in the thinnest, stalest series in the dataset. When
23% of bars have high == low and 11% have no volume, "high > L" is not evidence
a limit order would have filled -- the OHLC may be a mark/oracle print. I cannot
separate "under-competed venue" from "stale prices break my fill model" on
5,000 bars. Per E5 this venue was PRELIMINARY by construction anyway.

## Verdict
Per the pre-registered kill criterion: no venue with ASE < 0.8 ->
THE MAKER EQUILIBRIUM HOLDS EVERYWHERE I CAN MEASURE. Report and stop claiming
a maker opportunity exists. The invention is falsified as a profit thesis.

## What survives, and it is not nothing
The SCANNER is validated (E1 = 1.005 against a known null) and portable to any
venue with OHLC. It measures, in one number, whether liquidity provision in a
venue is fairly compensated. Its correct use is forward: screen NEW venues as
they launch, before capital arrives. The builder DEX is ~2 months old -- the
right move there is forward collection, not more mining of 5,000 bars.

## RUNNING TOTAL: 39 hypotheses. 0 tradable.
