# Peer Breadth (Candidate 18) — full reproduction package

Everything behind https://milkmantrades.com/peer-breadth.html: the rule, the code, every signal
and trade, the frozen research run it all came from, and the independent audit of this package
(`AUDIT.md`, 2026-09-07) that this revision answers.

**Source of the idea:** my overnight AI research loop (ideate → test → independent QC → register)
over 465 recorded crypto expressions on 2026-09-06/07. It registered 31 candidates; this is the
one I judged most promising. The hypothesis: a peer-market transition toward broad participation
may persist in the coins already moving with it. Any mistakes in the backtest are mine.

## The rule (fully mechanical, no discretion)

Universe: 20 Binance USDT-margined perpetuals — 1000PEPE, AAVE, ADA, AVAX, BCH, BNB, BTC, DOGE,
ETH, FIL, LINK, LTC, NEAR, SOL, SUI, TRX, UNI, WLD, XRP, ZEC. Daily bars, UTC days. The universe
was selected in August 2026 and applied to every window, including 2025 → (see "survivorship").

**SIGNAL** — evaluated at each daily close, for each coin separately:
1. Peers = the other 19 coins. A peer is *eligible* only if it has a complete 5-day return at
   both today and yesterday; at least 16 of the 19 must be eligible (max(8, ceil(0.8×19))).
   Both shares below use that same eligible set, so a listing or gap cannot manufacture a
   transition.
2. Positive share = fraction of eligible peers whose 5-day close-to-close return is > 0.
   **Long** when today's share ≥ 70% and yesterday's < 70%, the coin's own 5-day return is > 0,
   and today's close is above yesterday's close. **Short** is the exact mirror (negative share,
   own return < 0, close below yesterday's). Zero-return peers count in the denominator only.
3. Stop distance = 2.5 × ATR(14) — the 14-bar **simple mean of true range** (not Wilder) — as a
   fraction of the close, floored at 0.5%. If it would be wider than 20% the signal is skipped.

**TRADE**
- Enter at the open of the minute after the signal (00:01 UTC; the package's daily bars use the
  daily open). Stop at entry ∓ stop distance; target at 3× the stop distance (3R); scheduled
  exit 20,160 minutes after entry (the open of the 14th day) if neither level is reached.
- One open position per coin: a signal in a coin that is already held is skipped.
- Engine fills: stop/target levels checked on minute OHLC, a target needs strict penetration,
  known opening-gap exits are processed before new entries. The package checks daily high/low
  with stop-before-target inside one bar (conservative).

**ACCOUNT** ($100,000 research account, unlevered)
- Each eligible signal wants 0.25% of open-time equity, measured to the stop and including a
  20 bp cost reserve. All signals admitted on the same day form one batch and are scaled
  pro-rata so open initial risk never exceeds 1.5% of equity and gross notional never exceeds
  2× equity — the cap is enforced at admission, not continuously. Quantity is floored to 1e-8;
  orders below $100 notional are not placed. So $250 is the *maximum* single-signal risk on the
  starting account; the median reserved risk was $104 and the median filled position $777
  (0.73% of entry equity). A full-size trade at the median 14% stop would be ~$1,769.
- Cost 20 bp round trip (10 bp per side = 9 bp fee + 1 bp adverse slippage), debited at entry
  and exit, plus the actual archived 8-hour funding settled while holding. 12 bp and 30 bp
  scenarios are in `reference/`.

## Run it yourself (four commands, ~5 minutes, pandas + numpy; scipy for 04)

```
python3 01_download_data.py                              # Binance public archive, no key -> data/
python3 02_peer_breadth_backtest.py                      # continuous account from 2020 -> out/
python3 02_peer_breadth_backtest.py --start 2025-01-01 --out out_fresh   # fresh $100K from 2025
python3 03_compare_to_reference.py                       # like-for-like vs the frozen research runs
python3 04_quant_checks.py                               # t-stat, CI, Sharpe/Sortino/Calmar, MC, deflated Sharpe
```

What `03` prints (already run here; the four `out*/` files ship in the zip):

```
signals 2025 onward: reference 1313, package 1313, matching keys 1313
stop distance within 1e-06: 1310 of 1313  (three BCH bars differ by one tick between Binance's daily and minute-built closes)

fresh $100K account from 2025      research (minute engine, minute DD)      package (daily bars, daily-close DD)
2025 -> Aug 2026                   +16.74%  n 468  PF 2.14  DD -2.83%        +17.24%  n 467  PF 2.20  DD -2.33%
2026 YTD                           +10.48%  n 225  PF 2.67  DD -2.40%        +10.90%  n 224  PF 2.82  DD -1.84%
latest 90 days                      +3.93%  n  80  PF 3.45  DD -1.89%         +3.93%  n  80  PF 3.44  DD -1.65%
continuous account from 2020
2020 -> Aug 2026                   +41.52%  n 1403 PF 1.66  DD -4.64%        +38.71%  n 1382 PF 1.63  DD -4.39%
```

The research engine's own daily-close drawdown on the fresh account is −2.33%, i.e. the package
reproduces the drawdown exactly at daily resolution; the −2.83% on the page is the same account
marked every minute. Remaining differences are fill resolution (daily high/low vs minute walk)
and the funding approximation on intraday-exit days. `03` exits non-zero if the signal sets
differ, so a broken download or an edited rule fails loudly.

## Files

| File | What it is |
|---|---|
| `01_download_data.py` | Pulls daily klines + funding for the 20 perps from `data.binance.vision` (monthly zips, 2019-09 → last complete month, 12 parallel fetches). Writes `data/<SYM>_1d.csv`, `data/<SYM>_funding.csv`. |
| `02_peer_breadth_backtest.py` | Readable re-implementation: signal generation per the rule, then the account in the engine's order of operations (funding settlement → known open-time exits → admission on open-time equity with one slot per coin and pro-rata batch sizing → intraday protection → close marks). Every constant is named at the top. Writes `signals.csv`, `trades.csv`, `equity_daily.csv`, `stats.json`. |
| `03_compare_to_reference.py` | One-to-one signal comparison with a stop-distance tolerance, plus the like-for-like account tables above. |
| `04_quant_checks.py` | Regenerates every quant figure on the page from `reference/` with stated conventions (Sortino with MAR 0, Pearson kurtosis, Calmar over daily-close DD, seeded Monte Carlo, three Deflated Sharpe trial counts). Writes `quant_checks.json`. |
| `out/`, `out_fresh/` | The two package runs as shipped (continuous 2020 → and fresh 2025 →). |
| `reference/config.json` | The frozen research configuration `G24-T07_breadth_daily_v1`. |
| `reference/peer-breadth-signals-2025.csv`, `-2020.csv` | Every raw research signal. `signal_ts` is the bar CLOSE (next midnight UTC); the package stamps the bar's own date. |
| `reference/peer-breadth-trades-2025.csv`, `-2020.csv` | The research trade ledgers: fresh $100K account from Jan 2025 (477 rows incl. 9 open at cutoff) and the continuous 2020 → account (1,412 rows). |
| `reference/peer-breadth-equity-daily-*.csv` | Daily marked equity of both research accounts. |
| `reference/metrics-*.json` | Every window statistic, at 12/20/30 bp and for the 2020 → account. Drawdowns there are minute marks. |
| `reference/library_sharpes_2025.json` | The 2025 → Sharpe of the 31 library candidates (25 expose the field) used by the Deflated Sharpe calculation. |
| `reference/qc-reconciliation-20bp.json`, `qc-signal-check.json` | The independent QC agent's reconciliation of the research run (PASS). |
| `AUDIT.md` | The independent audit of the first version of this package and page; this revision addresses its blockers. |
| `research_engine/` | The exact frozen minute-bar code (`engine.py`, `run_campaign.py`, `signals_trend.py`, `reader.py`, `prepare_data.py`, `SPEC.md`), for auditing. It is **not** a turnkey minute-level reproduction: that needs the minute arrays (~2 GB via `prepare_data.py`) and a driver/catalog that are not included. |

## Headline numbers (research run, 20 bp + funding, unlevered)

```
Jan 2025 -> Aug 2026 ($100K fresh account)
  468 natural exits, 54.5% win (95% CI 49.9-59.1%), PF 2.14
  +16.7% on the account, max drawdown -2.83% on minute marks (-2.33% on daily closes)
  avg +$35/trade, t = 5.75 (p = 2e-8)
  Sharpe 1.77, Sortino 3.43 (MAR 0), Calmar 4.2 (CAGR 9.8% / daily-close DD 2.33%)
  delete best 1 / 5 / 10 natural trades: +$15.6K / +$13.6K / +$11.8K (all still positive)
  18 of 20 coins profitable (NEAR, TRX negative); 94% time in market; max gross exposure 21.8%
  median hold 14 days (318 of 468 exits are the scheduled time exit)
2020 -> Aug 2026 (continuous account)
  1,403 exits, 51.4% win, PF 1.66, +41.5%, max DD -4.64% (minute marks); 6 of 7 years positive
  by year: 2020 -0.8, 2021 +6.6, 2022 +4.4, 2023 +5.2, 2024 +4.9, 2025 +5.2, 2026 YTD +10.3
Neighbors (daily, 65% / 75% threshold): +16.9% / +17.5% since 2025. The two 4-hour settings tried
lost money since 2025 (-11.6% / -1.8%) and are slightly positive in the last 90 days.
```

## The overfitting question (read this)

- **Deflated Sharpe ratio** (Bailey & López de Prado) depends on what you count as a trial.
  Against the 25 of 31 library candidates that expose a comparable Sharpe (σ 0.51): hurdle
  Sharpe 1.02, deflated probability **0.87**. Within this expression's own 5 tested settings:
  **0.74** (fresh) / **0.44** (2020 →). Against all 465 recorded expressions with the Sharpe
  spread the search actually produced (σ 2.27 from 285 run files, including junk screens and
  history replays): hurdle 6.9, probability ~0. None of these is a pass/fail criterion; all three
  are regenerated by `04_quant_checks.py`.
- **Monte Carlo drawdown** on the card/page reorders the 468 closed-trade net P&Ls 5,000 times
  (numpy `default_rng(0)`): median −$858, 95th percentile −$1,322, worst −$2,384. That is a
  dollar drawdown of the trade sequence, not an account-mark drawdown.
- **PBO / CSCV** needs every trial's return series and is queued.
- **Survivorship:** the 20 coins were selected in August 2026 and applied to every window.
  Dead or delisted coins are not in the set; historical membership was not reconstructed and the
  size or direction of the bias was not measured.
- **No forward evidence yet.** Rules frozen 2026-09-07; passive paper tracking starts here.

## Data

Binance USDT-margined perpetual futures from Binance's public archive
(`https://data.binance.vision/data/futures/um/monthly/...`): daily klines and funding rates for
the package, 1-minute klines for the research engine. UTC. The research arrays end 2026-09-01
00:00 UTC exclusive and `02` is capped at the same `END_DATE`; `01` downloads through the last
complete month, so later data lands in the CSVs but is not simulated until you change `END_DATE`.

Educational only, not financial advice. Test it yourself before you trade it.
