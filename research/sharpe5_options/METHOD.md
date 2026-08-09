# The invented method: an orthogonal three-sleeve book

**What it is.** Not a better option strategy — a portfolio of three
*structurally different* bets, each individually mediocre, combined because
one of them is genuinely uncorrelated with the others. It is the only
construction in ~330 tested configurations that beats buying the index.

| | Sharpe | CAGR | Max DD |
|---|---:|---:|---:|
| SPY buy & hold | 0.84 | 14.5% | −34.2% |
| **Combined book, levered to SPY's volatility** | **0.98** | **17.3%** | **−31.0%** |
| Combined book, unlevered | 0.98 | 6.9% | −13.1% |
| — development 2019–2024 | 0.91 | 6.5% | −13.1% |
| — **holdout 2025–2026** | **1.24** | 8.3% | **−5.4%** |

---

## The three sleeves

### A — Vol-targeted equity, sized by *implied* vol (Sharpe 0.90)

Hold SPY, scale exposure to `median(IV) / IV_today`, capped at 2×, using an
expanding-window median so the target is always past-only.

The point is the *choice of vol estimate*. Implied vol is the market's
forward-looking forecast; trailing realized vol only learns about risk after it
arrives. Sized on implied vol the sleeve returns Sharpe 0.90 versus 0.79 on
trailing realized, and it is the most stable thing measured anywhere in this
project — dev 0.93, holdout 0.85, essentially no decay.

### B — Skew-ranked stock long-short, market-neutral (Sharpe 0.44)

Rank names by 25-delta option skew; go long the top quintile, short the bottom,
equal weight, hold ~8 observations. **Signal from options, position in stock.**

This is the sleeve that makes the book work, and not because it is good. Option
skew predicts forward stock returns with t=+5.6, monotonically increasing in
horizon — a real effect. Expressing it in stock rather than options replaces a
100–500bp option spread with a 1–5bp stock spread. Its own Sharpe is only 0.44,
but its correlation to the other two sleeves is **−0.03 and 0.00**.

### C — Long-dated wide put credit spreads (Sharpe 0.59)

Sell a 4%-OTM put, buy a 24%-OTM put, ~75 days to expiry, hold to expiry, never
manage. Liquid names only, requiring a positive worst-side credit.

The tenor and width are the whole point. The bid-ask toll is roughly **fixed per
leg in dollars**, while premium scales with √tenor and risk scales with width.
Moving from 30d/5%-wide to 75d/20%-wide roughly triples the premium collected
for the same toll and dilutes that toll over 4× the risk capital. It is the only
option structure tested that is profitable while *crossing* the spread.

---

## Why it works: orthogonality, not alpha

Weekly correlation matrix:

| | A | B | C |
|---|---:|---:|---:|
| **A** | 1.00 | −0.03 | 0.46 |
| **B** | −0.03 | 1.00 | 0.00 |
| **C** | 0.46 | 0.00 | 1.00 |

Sleeve B is orthogonal to both others. Combining k uncorrelated sleeves of
similar quality scales the Sharpe by roughly √k, and that — not any individual
edge — is what carries the book from 0.90 to 0.98 while cutting drawdown from
−23.7% to −13.1%. **A mediocre uncorrelated return stream is worth more than a
good correlated one.**

Weights are **inverse-volatility, fitted on development data only, then
frozen**: A 0.251, B 0.646, C 0.103. No return-based optimization anywhere.

---

## What has to be true for this to work

**≤5bp round-trip stock execution.** The whole edge over SPY is cost-sensitive:

| Stock cost | Sleeve B | Combined book |
|---|---:|---:|
| 2 bp | +0.44 | **+0.98** |
| 5 bp | +0.34 | +0.92 |
| **10 bp** | +0.16 | **+0.82 — below SPY** |

At 10bp the advantage is gone. Mega-cap spreads are typically 1–2bp, so this is
achievable, but it is a live operational requirement rather than a modelling
convenience.

**2.54× leverage** to reach SPY's volatility. Unlevered the book returns 6.9%.
The Sharpe advantage is real; the return advantage only appears when levered,
and leverage brings margin calls and financing costs this study does not model.

---

## Honest limits

- **~330 configurations were tried.** The Deflated Sharpe Ratio prices that in
  and is near zero. This survived out of sample, but so did earlier candidates
  that later failed.
- **7.5 years, 382 weekly observations, one holdout of 1.6 years.** The holdout
  Sharpe of 1.24 is encouraging and thin.
- **Sleeve C's −47% standalone drawdown** is real; it only looks tame at a 10%
  weight.
- **Nowhere near Sharpe 5.** The measured ceiling for this market is ~0.39 for a
  single short-vol book, and diversification lifted the combination to ~1.0.
  Reaching 5 needs roughly 100× the breadth this market offers.
- **Not tested live.** Adverse selection, borrow costs on the short leg, and
  capacity are all unmodelled.

## Reproducing

```bash
python3 study18_invent.py     # the combination and the implied-correlation gate
python3 study16_where.py      # why 75d/20% is the right option structure
python3 study10_optsig_stock.py   # the skew -> stock signal
python3 study11_timing.py     # vol targeting on implied vs realized
```
