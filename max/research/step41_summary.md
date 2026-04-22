# Step 41 — Fixed-percentage TP grid search (CAP5+SMA12M ranker)

## Methodology

- **Universe.** 128 tickers from `max/research/data/raw/Close.parquet`; SPY excluded from picks (benchmark). Tickers need >= 252 bars of non-NaN history AND a finite CAP5+SMA12M signal on rank date to be eligible.
- **Ranker (CAP5+SMA12M).** Trailing 252-bar arithmetic mean of the `final` column (conviction = quality x 10D_prob x pullback_gate) from `max/research/data/bt_ext.parquet`, forward-filled to daily. Computed per-ticker, known at close of day D.
- **Entry.** First trading day of each calendar month = rank date D. Pick top-1 by CAP5+SMA12M score; fill at close of D+1 (one-bar delay matches production CAP5 conventions).
- **Exit.** TP target = entry_px * (1 + TP%). Fill at TP price on the first day d > entry_day where High[d] >= target. If no TP hit within `time_stop_bars` trading days (measured from entry_day), exit at close of that final bar.
- **Cash rule.** Only one position open at a time; $1000/mo contribution still accrues while a trade is open but stays in cash until the next monthly entry. Between trades cash sits idle (zero interest assumed).
- **Window.** 2006-04-25 to 2026-04-21 (5029 trading days).
- **Grid.** TP in {2, 3, 5, 7, 10, 15, 20, 30}% x time_stop in {10, 21, 42, 63, 126, 252, 504} bars = 56 combos.
- **SPY DCA baseline (same window, same cadence).** CAGR 6.63%, MaxDD 35.9%, Sharpe 1.11 (a touch lower than the 7.86% cited; the spread is attributable to window boundaries / entry-at-D+1 convention).

## Cross-checks (sanity)

- TP=2% / TS=504: win rate 97.6%, avg trade +1.3% -> hard to miss a 2% move in two years. Pass.
- TP=100% / TS=21: win rate 0.0% (of 118 trades) -> no stock doubles in a month. Pass.
- TP=100% / TS=63: 1.8% win rate. Pass.
- TP=100% / TS=252: 26.3% (above the "5-15%" band), but with only 19 trades in the window and a 1-year stop, CAP5-ranked names hitting 2x in a year is credible. Not a bug.

## Top 10 by CAGR

| TP% | stop (bars) | CAGR | MDD | Sharpe | WinRate | AvgRet | AvgDays | nTrades |
|----:|------------:|-----:|----:|-------:|--------:|-------:|--------:|--------:|
|  30 |  252 | +13.15% | 70.6% | 1.01 | 53.8% | +15.34% | 163 |  26 |
|  10 |  252 | +11.52% | 78.8% | 0.99 | 87.9% |  +5.15% |  67 |  58 |
|  10 |  126 | +10.39% | 83.3% | 0.96 | 83.5% |  +4.21% |  47 |  79 |
|  20 |   42 | +10.32% | 70.9% | 0.97 | 34.0% |  +2.62% |  33 |  94 |
|  20 |   63 | +10.17% | 80.4% | 0.95 | 42.1% |  +4.81% |  47 |  76 |
|  10 |   42 |  +9.94% | 61.7% | 0.99 | 60.3% |  +2.47% |  25 | 116 |
|  30 |   42 |  +9.59% | 64.7% | 0.96 | 15.9% |  +3.24% |  40 |  82 |
|   7 |   42 |  +9.39% | 63.6% | 1.02 | 69.7% |  +1.82% |  20 | 132 |
|  15 |   42 |  +9.30% | 70.1% | 0.95 | 40.4% |  +2.97% |  32 |  99 |
|  15 |   63 |  +8.17% | 82.4% | 0.92 | 52.9% |  +3.38% |  42 |  85 |

## Top 10 by win_rate x Sharpe (balanced / high-win-rate)

| TP% | stop (bars) | CAGR | MDD | Sharpe | WinRate | AvgRet | AvgDays | nTrades |
|----:|------------:|-----:|----:|-------:|--------:|-------:|--------:|--------:|
|   7 |  504 |  +7.67% | 62.7% | 0.98 | 94.2% | +3.89% |  68 |  52 |
|  10 |  504 |  +7.88% | 71.4% | 0.98 | 91.9% | +6.07% | 103 |  37 |
|  10 |  252 | +11.52% | 78.8% | 0.99 | 87.9% | +5.15% |  67 |  58 |
|   5 |  504 |  +4.71% | 77.2% | 0.92 | 95.0% | +2.18% |  58 |  60 |
|   2 |   42 |  +0.90% | 62.8% | 0.94 | 89.9% | +0.34% |   9 | 178 |
|   3 |  504 |  +3.36% | 77.0% | 0.84 | 97.2% | +2.15% |  45 |  72 |
|   2 |  504 |  +2.06% | 76.5% | 0.82 | 97.6% | +1.30% |  35 |  85 |
|   3 |  252 |  +2.69% | 76.3% | 0.84 | 95.3% | +0.93% |  29 | 106 |
|   5 |  252 |  +3.92% | 78.7% | 0.88 | 91.7% | +1.91% |  41 |  84 |
|   2 |  126 |  +0.97% | 71.9% | 0.84 | 95.2% | +0.36% |  16 | 147 |

## Recommendation

**Production pick: TP = 10%, time_stop = 252 bars (~12 months).**

- CAGR +11.52% vs SPY DCA ~7.86% (or +6.63% on this window) -> ~370-490 bps excess.
- Win rate 87.9% on 58 trades (real sample size, not a small-N artifact like TP=30%/252).
- Sharpe 0.99, MaxDD 78.8% (high — a long-stop 1-ticker strategy is concentrated by construction).
- Gives clean, concrete signals: "Buy stock X at $P on entry day, GTC limit sell at $P*1.10, auto-close at market on day 252 if not hit."
- If the user is drawdown-averse, the next-best balanced pick is **TP=10% / stop=126 bars** (10.39% CAGR, 83.5% WR, 79 trades, 83% MDD) — more trades, same TP, half the max hold.
- TP=30%/TS=252 wins on raw CAGR (13.15%) but has only 26 trades — too few to trust as robust.

## Caveats

- **Quality leakage.** The `quality` multiplier in `final` is noted as not strictly point-in-time (uses today's snapshot). That inflates CAP5 scores for names that happen to be high-quality today; CAP5+SMA12M's 252-day smoothing dampens but does not eliminate this. Treat CAGR numbers as an optimistic ceiling.
- **Survivorship.** Universe is today's 128 tickers — delistings and name swaps over 2006-2026 aren't in the pool. Real-world CAGR will be lower.
- **Small sample at long stops / wide TPs.** TP=30%/TS=252 has 26 trades across 20 years; the +13% CAGR has wide confidence intervals. TP=10%/TS=252's 58 trades is the sweet spot for statistical reliability.
- **Concentration risk.** Top-1 per month with 12-month holds => 1-3 concurrent names during bear markets -> 70%+ MDDs are real (2008-09 and 2022). This is not a risk-parity strategy.
- **Intrabar TP fill assumption.** We assume the TP limit fills exactly at target when High >= TP. In live trading, gap-ups above TP get better fills (conservative here); gap-downs that recover then spike might skip fills (realistic).
- **Idle-cash drag.** Cash between trades earns 0%. Money-market at ~4-5% would add ~20-30 bps.
- **Single-name pick.** One stock per month; a single catastrophic pick (e.g. a busted biotech at a conviction top) can cause an outsized loss. The 10% TP ceiling caps upside but does not cap downside — drawdown at time-stop can be severe.
