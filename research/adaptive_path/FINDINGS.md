# Can we be over 90% sure a stock is higher 30+ sessions later?

**No — and the honest system says so by staying silent.** Built and run, the
90% evidence gate never opened once, at any horizon, on any date. Not at 30
sessions, not at 126, not at 252.

That is the designed behaviour rather than a failure, and the rest of this
document is what was measured on the way there, including the one result that
reverses an intuition worth reversing.

## The yardstick

On a point-in-time panel of 10,991 US tickers, 1990–2026, **including the
4,725 that stopped trading** (`research/uptrend/data.py`):

| horizon | P(higher), strict | delisting = loss | gap |
|---|---|---|---|
| 30 sessions | 51.4% | 51.0% | 0.3 |
| 63 | 53.9% | 52.9% | 1.0 |
| 126 | 55.2% | 53.0% | 2.2 |
| 252 | 57.0% | 52.5% | 4.5 |
| 504 | 59.7% | 50.3% | **9.4** |

"Strict" drops the observations where the stock stopped trading mid-horizon.
That is the number almost everyone quotes and it is the optimistic one. At two
years it is worth **9.4 points** — a survivor-only universe would hand back
most of any edge a model appeared to find.

The cheapest possible conditioning — above a rising 200-day average, realised
vol under 20%, market above its own average — takes 252-session P(higher) from
57.0% to **70.3%**. Any model has to beat that, not the raw base rate.

## The engine

`research/adaptive_path/engine.py`, built on the MEAP protocol supplied by the
user. Three experts (base rate, regularised logistic, gradient boosting) mixed
by exponentially-decayed recent loss with a fixed-share floor; the mixture
recalibrated online against **matured** forecasts only; a threshold chosen by a
conservative Wilson lower bound counted in non-overlapping horizon blocks,
Bonferroni-corrected across every horizon × threshold pair; and a conformal
path radius sized so the **whole trajectory** — every daily close, not sampled
checkpoints — lands inside at the target rate.

Nothing is fitted once and frozen. The one thing it will not do is speak
without evidence.

### What it produced

| horizon | forecasts | dates | base rate | max p_cal reached | gate opened |
|---|---|---|---|---|---|
| 30 | 76,566 | 256 | 52.7% | < 0.60 | **0 of 256** |
| 126 | 75,387 | 252 | 54.7% | < 0.70 | **0 of 252** |
| 252 | 73,568 | 246 | 54.7% | < 0.70 | **0 of 246** |

The gate could not open because the model never produced a probability near
90% in the first place. Its calibration is genuinely good — mean absolute gap
between predicted and realised across deciles is **0.036** at 30 sessions —
it is simply calibrated to the truth, which is that this is close to a coin
flip.

### The trap we nearly reported

At 126 and 252 sessions the *lowest*-probability bucket had the *highest*
precision (73.2% against a 54.7% base rate) — an apparent inversion worth a
paper. It is one event. The entire bucket at 126 sessions is **12 dates, all
in 2009**: the model saw crash-shaped features, said "unlikely", and the
rebound happened. Clustered by date it is a single observation and worth
nothing. This is exactly what date-clustering exists to catch.

## The frontier: what *does* reach 90%

Two dials raise P(higher) without forecasting anything — the horizon, and
diversification. Measured across the same panel under the best cheap filter,
`research/uptrend/frontier.py`:

**P(equal-weight basket is higher), delisted held at last traded price**

| horizon | N=1 | N=5 | N=10 | N=20 | N=50 |
|---|---|---|---|---|---|
| 30 | 59.4% | 66.2% | 68.6% | 70.6% | 72.4% |
| 126 | 69.0% | 77.1% | 79.4% | 81.1% | 82.3% |
| 252 | 72.5% | 80.6% | 83.2% | 84.8% | 86.3% |
| 756 | 76.6% | 85.2% | 87.5% | 88.9% | **90.2%** |

One cell clears 90%: a 50-name basket held three years. Two things then take
it away.

**Score a delisted name as a total loss and the whole table caps at ~74%:**

| horizon | N=1 | N=10 | N=50 |
|---|---|---|---|
| 252 | 69.9% | 70.3% | 70.9% |
| 756 | 70.6% | 72.9% | 73.9% |

**And the worst calendar year is brutal — in the direction nobody expects:**

| horizon | N=1 | N=5 | N=10 | N=20 | N=50 |
|---|---|---|---|---|---|
| 252 | 31.6% | 18.5% | 12.4% | 7.7% | 4.6% |
| 504 | 12.8% | 3.0% | 1.1% | 1.0% | **0.3%** |

**Diversification raises the average hit rate and destroys the worst one.**
Spreading across 50 names removes the idiosyncratic upside that lets a single
stock rise while the market falls; it does nothing about the market falling.
Enter at the wrong moment in 2007 and a one-name bet was higher two years
later 12.8% of the time, while the 50-name basket managed 0.3%. The
better-looking average and the worse tail are the same fact.

**Nothing in the grid clears 90% on the average, the pessimistic delisting
treatment, and the worst year simultaneously.**

## What is actually shippable

Not a 90% badge. What exists and works:

1. **A calibrated probability.** At 30 sessions the model's stated probability
   is within 0.036 of the realised rate. "58% likely higher in 30 sessions" is
   a true sentence; "90% likely" is not one we can write.
2. **A gate that stays shut.** Silence is the normal output and an empty day is
   a correct answer. The value is in what it refuses to say.
3. **The frontier table.** Horizon and diversification are the two dials that
   genuinely move the odds, they are not forecasts, and the worst-year row is
   the price.

## Reproduce

```
python research/uptrend/data.py                 # build the PIT panel
python research/uptrend/baserates.py            # the yardstick
python tests/test_adaptive_path.py              # 29 leakage/denominator checks
python research/adaptive_path/run.py --horizon 30
python research/adaptive_path/run.py --horizon 252
python research/uptrend/frontier.py             # horizon x diversification
```

## Corrections made to the supplied protocol

- **The recency decay was inert.** `date_weights` normalised decayed weights by
  their per-date *sum*; every row on a date shares that date's age, so the
  decay cancelled exactly and every date came out weighted 1.0 regardless of
  age. `expert_half_life_sessions` did nothing. Dividing by the per-date
  *count* keeps the intended equalisation and lets the decay survive
  (verified: 16× ratio at four half-lives).
- **Market features were duplicated per ticker** — seven of twenty-two
  matrices were one value per date broadcast across 11,000 columns, 2.8 GB of
  copies that OOM-killed the full-panel run at 14 GB.
- **`min_names_per_date` was hardcoded at 50**, which silently made the engine
  untestable on any small panel.
- **Delisting**, which the protocol did not need on a survivor universe and
  this one does: a name that stops trading mid-horizon is labelled *not
  higher* rather than dropped.
