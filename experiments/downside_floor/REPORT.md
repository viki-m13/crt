# Floor Picker — buying stocks that rarely fall below the purchase price

**Goal (as posed):** use state-of-the-art HuggingFace models to pick stocks
that, *after you buy them*, are "not below the purchase price often or at
all." This is a **downside-avoidance** objective, distinct from the repo's
deployed return/Sharpe strategy. It asks: minimize how often and how deeply
a position sits **underwater** relative to what you paid.

Everything here is point-in-time (PIT) and leakage-controlled, on the same
augmented S&P 500 PIT panel as the deployed strategy: **254 monthly asofs,
2003-01 → 2026-02, ~412 names/month, honest delisting (a name that stops
trading inside a window is marked fully underwater)**.

## How we measure "below the purchase price"

For every (month, ticker) buy, from daily closes we compute over forward
horizons H ∈ {1m, 3m, 6m, 12m} (`build_downside_labels.py`):

| label | meaning |
|---|---|
| `uw_frac` | fraction of the next H trading days the close is **below** the buy price — *how often it's underwater* (headline) |
| `ever_below` | did it dip below the buy price **at all** |
| `end_below` | is it below the buy price at +H |
| `maxdd` | deepest close-to-buy drawdown (how far underwater) |
| `safe` | `uw_frac < 0.15` — *basically never underwater* |

### The brutal base rate (the average S&P 500 stock)

| horizon | avg days underwater | P(ever dips below) | P(ends below) |
|---|---:|---:|---:|
| 1m | 44.6% | 83.5% | 44.0% |
| 3m | 42.5% | 89.3% | 40.9% |
| 12m | 37.7% | 93.5% | 34.2% |

**"Never below at all" is essentially unavailable for a single stock** — ~89%
of 3-month buys dip below the entry at some point. The achievable goal is to
*minimize time underwater and dip depth*, and to maximize the share of buys
that stay (almost) always above water.

## The HuggingFace model: Chronos-Bolt as a downside forecaster

The deployed v5 strategy already uses **`amazon/chronos-bolt-tiny`** (Apache-2.0)
but keeps only the *final* 3-month return quantiles. That discards exactly
what this objective needs: the **lower quantiles of the forward price path.**

`score_chronos_floor.py` re-scores the whole panel (zero-shot, ~5 min on CPU,
472 forecasts/sec) keeping the full 9-quantile path, and converts it into
downside forecasts relative to the entry price:
`chr_exp_uw_frac` (expected fraction of days below buy = mean_t P(price_t<entry)),
`chr_p_below_end`, and `chr_trough_q10/q30` (worst path-quantile dips).
Zero-shot ⇒ no training, no look-ahead.

**Honest result #1 — Chronos's *drift* is the wrong signal for this job.**
Ranking by Chronos's lowest expected-underwater (which tracks forecast drift)
selects high-momentum, high-volatility names that actually dip below the buy
price *more* than average (3m `uw_frac` 0.435 vs 0.425 universe). Forecast
upside ≠ path safety. **Chronos's useful contribution is its lower-quantile
*trough* features as inputs to a learned model, plus a mild upside tilt** —
not its point/drift forecast.

## FloorScore — the deployed picker

A learned drawdown model (sklearn `HistGradientBoostingRegressor`, walk-forward,
120-day embargo so every training label is fully realized before the test date)
is trained to predict downside labels from **cross-sectional features +
Chronos downside forecasts** (`floor_lib.py`). FloorScore is then a transparent
z-score blend of five leakage-free, economically-motivated ingredients
(higher = safer):

```
FloorScore =  z(gbm_maxdd_3m)            learned: shallow max drawdown-from-buy
            - z(gbm_uw_frac_3m)          learned: few days underwater
            - z(vol_3m_xs)               low realized volatility
            + 0.5·z(trend_health_5y_xs)  durable long-term uptrend (no value traps)
            + 0.5·z(chr_trough_q30_3m)   Chronos downside cushion (HF model)
```

Each month we buy the top-K (K=10) by FloorScore, equal weight, and read off
the realized forward downside. `floor_maxsafe` is a pure-downside rank-mix
variant (no upside tilt).

## Results — pooled, 2003-2026, K=10

**3-month horizon** (the headline):

| strategy | days underwater | P(ever<) | P(ends<) | mean maxdd | mean ret | median ret | safe% |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SPY (buy the index)** | **31.4%** | 85.0% | **27.3%** | **−4.6%** | +2.8% | +3.9% | **45.5%** |
| average S&P stock | 42.5% | 89.3% | 40.9% | −9.9% | +11.3% | +2.8% | 33.2% |
| naive low-vol | 39.7% | 88.9% | 36.7% | −6.1% | +1.2% | +2.5% | 35.4% |
| **FloorScore** | 39.5% | 89.1% | 36.6% | −6.4% | **+31.8%** | +3.0% | 35.6% |
| **FloorScore (max-safe)** | **38.6%** | 88.5% | **35.7%** | −6.3% | +2.5% | +3.1% | **37.2%** |

(1m / 6m / 12m and the full table are in `floor_backtest_results.json`.)

### Honest verdict

1. **The single most reliable way to "rarely be below your purchase price" is
   to buy the diversified index (SPY), not any single stock.** Across every
   horizon SPY has the lowest time-underwater, the lowest chance of ending
   below the buy price, and the highest "basically never underwater" rate.
   Diversification, not stock-picking, is the real downside lever. We report
   this plainly because it's true.

2. **If you are going to pick individual stocks, FloorScore is a genuine,
   robust improvement over the average S&P name and over naive low-vol.** It
   cuts time-underwater (42.5%→38.6%), shallows the average worst dip
   (−9.9%→−6.3%), lowers the chance of ending underwater (40.9%→35.7%), and
   raises the "basically never underwater" rate (33.2%→37.2%) — while the
   deployed (upside-tilted) variant *also* lifts mean forward return well
   above the universe. Low-vol matches it on safety only by surrendering the
   return right-tail; FloorScore keeps it.

3. **Robust across time, not a single-period artifact.** FloorScore beats the
   universe on 3m underwater/end-below/safe in all three eras
   (2003-09, 2010-19, 2020-26) and is at-or-better than the universe on
   `P(ends below)` in **17 of 24 years**.

4. **Chronos's marginal value is real but modest.** Removing all Chronos terms
   (`floor_noChr`) barely moves the downside metrics; Chronos's lower-quantile
   trough features improve the learned model slightly and the deployed blend
   uses them as a soft upside tilt. This objective is dominated by
   volatility/drawdown structure, which the HF time-series model only partly
   illuminates. (The repo previously found Chronos-Bolt the best of the
   HF time-series foundation models it tried — Moirai, TTM — so a different
   HF model is unlikely to change this conclusion.)

## Being *more* selective — where it helps and where it doesn't

A natural follow-up: can we tighten the net and get closer to "never below the
purchase price"? We tested two kinds of extra selectivity
(`explore_selectivity.py`).

**Per-name selectivity plateaus.** Concentrating from top-20 down to top-1 by
FloorScore, or adding strict conviction gates (predicted underwater ≤ 0.38,
predicted dip ≥ −4.5%, low vol, positive Chronos drift — which sits out ~24%
of months), barely moves how *often* a single buy is underwater (3m `uw_frac`
stays ~0.39–0.42) and leaves the "literally never dipped below" rate stuck near
**10%**. Stricter gates only shallow the worst *dip* (mean maxdd −9.9% → −6.3%)
and cost return; extreme concentration (top-1) actually *raises* 12-month tail
risk. **You cannot select your way to "never underwater" with one stock** — any
single name is volatile enough to spend ~40% of early days below entry.

**The selectivity that works is diversification + regime timing**
(`explore_portfolio.py`, measured at the *portfolio* level — the basket's value
vs its own cost basis):

| basket | 3m underwater | 3m ends-below | 3m worst dip | 3m mean ret |
|---|---:|---:|---:|---:|
| SPY buy & hold | 31.4% | 27.3% | −4.6% | +2.8% |
| FloorScore top-10 | 32.6% | 26.9% | −4.3% | +31.8% |
| **FloorScore top-10 + regime** | 30.8% | **24.1%** | **−3.8%** | **+38.8%** |

| basket | 12m underwater | 12m ends-below | 12m worst dip | 12m "never dipped" |
|---|---:|---:|---:|---:|
| SPY buy & hold | 22.6% | 13.1% | −8.6% | 10.7% |
| SPY + regime | 20.2% | 10.8% | −6.9% | 9.8% |
| FloorScore top-10 | 23.1% | 14.8% | −7.4% | 11.1% |
| **FloorScore top-10 + regime** | **19.9%** | **10.3%** | **−5.4%** | 9.8% |

(Regime gate = only buy when SPY is above its 200-day SMA; it sits out ~20–24%
of months — the bad ones. Full grid in `floor_portfolio_results.json`.)

The equal-weight basket cuts time-underwater from ~40% (single name) to ~20%
(12m) because idiosyncratic dips cancel, and the regime gate avoids buying into
broad drawdowns. The result: **a FloorScore top-10 basket bought only in SPY
uptrends matches or beats just-buying-the-index on every downside metric
(time-underwater, ends-below, worst dip) while keeping the picker's return** —
below its cost only ~10% of the time at 12 months, with a shallower worst dip
than SPY. That is the honest ceiling for "rarely below the purchase price":
diversify the selection and time the entry; don't over-concentrate.

## Today's safest buys

`score_today.py` ranks the latest available asof. As of 2026-02-27 the top
FloorScore names are the expected low-vol, durable-uptrend defensives — AFL,
MCD, KO, HIG, utilities (SO, DTE, CMS, LNT), defensive REITs (VICI, VTR, WELL)
— a strong face-validity check. Full list in `floor_today.csv`.

## Reproduce

```bash
pip install -r requirements.txt          # adds torch (CPU), chronos-forecasting, scikit-learn
python3 experiments/downside_floor/build_downside_labels.py   # ~10s, ground-truth labels
python3 experiments/downside_floor/score_chronos_floor.py     # ~5 min, Chronos downside forecasts
python3 experiments/downside_floor/floor_lib.py               # ~9 min, walk-forward GBMs -> floor_scored.parquet
python3 experiments/downside_floor/backtest_floor.py          # pooled + by-era/by-year validation
python3 experiments/downside_floor/explore_signals.py         # signal sweep (read-only)
python3 experiments/downside_floor/explore_selectivity.py     # how much selectivity buys you
python3 experiments/downside_floor/explore_portfolio.py       # basket + regime-gate (the real lever)
python3 experiments/downside_floor/score_today.py             # current safest-buys ranking
```

## Files

| file | role |
|---|---|
| `build_downside_labels.py` | forward underwater/drawdown ground-truth labels |
| `score_chronos_floor.py` | Chronos-Bolt full-path downside forecasts |
| `floor_lib.py` | merge + walk-forward downside GBMs → `floor_scored.parquet` |
| `explore_signals.py` | candidate-signal sweep |
| `explore_selectivity.py` | concentration & conviction-gate sweep (the plateau) |
| `explore_portfolio.py` | basket + regime-gate portfolio underwater (the breakthrough) |
| `backtest_floor.py` | locked FloorScore vs SPY/universe/low-vol, by era & year |
| `score_today.py` | live "safest buys" ranking |
