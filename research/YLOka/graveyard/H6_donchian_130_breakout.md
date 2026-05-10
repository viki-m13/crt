# H6 — Donchian-130 breakout filter — KILLED (sample-of-1 win)

**Hypothesis**: from Concretum/Mulvaney trend research, intersect (top-K ML score) ∩ (price within 5% of 130d high) to capture durable trends. If filter empties basket, fall back to top-K unfiltered.

**Surface result**:
| | CAGR | Sharpe | MaxDD |
|---|---:|---:|---:|
| baseline | 40.78% | 0.953 | -49.83% |
| Donchian-130 | **48.73%** | 1.003 | -56.99% |

Looks like +7.95 pp CAGR, +0.05 Sharpe — a big "win".

**Diagnosis — why this is sample-of-1, not alpha**:

| year | base | Donchian | diff (pp) |
|---|---:|---:|---:|
| 2003 | 8.0% | 7.4% | -0.6 |
| 2004 | 27.7% | 29.7% | +2.0 |
| 2005 | 21.3% | 18.0% | -3.2 |
| 2006 | 26.4% | 28.0% | +1.6 |
| 2007 | 9.0% | -22.8% | **-31.8** |
| 2008 | -17.5% | -24.8% | -7.3 |
| 2009 | 625.5% | 625.5% | 0.0 |
| 2010 | 52.0% | 40.3% | -11.7 |
| 2011 | 21.5% | 29.4% | +8.0 |
| 2012 | 34.0% | 85.5% | +51.5 |
| 2013 | 80.1% | 71.4% | -8.7 |
| 2014 | 7.1% | 0.7% | -6.4 |
| 2015 | 3.4% | 2.6% | -0.8 |
| 2016 | 35.5% | 272.6% | **+237.1** |
| 2017 | 44.8% | 29.1% | -15.7 |
| 2018 | -18.4% | -19.8% | -1.4 |
| 2019 | 44.8% | 41.0% | -3.9 |
| 2020 | 109.6% | 109.6% | 0.0 |
| 2021 | 65.8% | 57.3% | -8.5 |
| 2022 | 22.8% | 53.1% | +30.4 |
| 2023 | 89.9% | 147.2% | +57.2 |
| 2024 | -8.3% | 1.8% | +10.1 |

- The 2016 single-year diff of **+237 pp** alone moves the 22-year compounded CAGR by +5-6 pp. Without 2016, lift is ~2 pp.
- 2016 mechanism: Donchian filter dropped the basket from K=3 → K=1 (only NVDA passed the within-5%-of-130d-high filter during late 2016). NVDA returned ~227% in 2016. A 100%-NVDA basket year vs an equal-weight 3-name basket explains the +237 pp.
- Average pick count: 2.17 (Donchian) vs 2.95 (baseline). The filter is silently shrinking K, which conflates basket-concentration effect with breakout-signal effect.
- MaxDD got worse (-57% vs -50%): smaller K → more concentration risk, no downside protection.
- 4/22 years beat baseline; median yearly diff 0.0%; mean +2.84 pp dragged by 2016.

**Don't repeat**: this is portfolio concentration disguised as a signal filter. If we wanted to test "concentrate on the strongest breakout", we should hold K constant at 1 with the Donchian filter — that's already exp_10 (K=1) which gave CAGR 28.6%, MaxDD -80%. So the filter doesn't help on its own; it only "wins" when it accidentally concentrates in a year where the lone surviving pick happens to be a multibagger.

**Real test would be**: hold K=3 fixed, replace the filter with a soft tilt (score += w × proximity_to_130d_high). That decouples the concentration effect from the signal effect. Deferred to a future session.
