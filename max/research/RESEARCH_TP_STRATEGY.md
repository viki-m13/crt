# Take-Profit Strategy — Research Summary

**Goal**: A practical monthly-pick strategy — each month the system names one stock to buy, with an explicit buy price and a sell target. Must (a) beat monthly DCA into SPY by a wide margin, (b) deliver a predictable win rate, (c) produce concrete buy-at-P / sell-at-T signals a retail investor can place as limit orders.

**Prior production**: CAP5+SMA12M hold-forever (+18.31% CAGR 20Y, no sells). Its edge disappears when rotated (step39 research).

## TL;DR

**Ship: TP=+10%, Stop-loss=−15%, Time-stop=252 trading days (~12 months). Rank monthly by CAP5+SMA12M, pick top-1 stock.**

| Metric | This strategy | SPY DCA | Prod CAP5 (hold-forever) |
|---|---:|---:|---:|
| CAGR (2006–2026, 20Y) | **+11.13%** | +6.63% | +18.31% |
| Excess over SPY DCA | **+4.50pp** | — | +11.68pp |
| Max drawdown | 48.0% | 35.9% | 47.5% |
| Sharpe | 1.08 | 1.11 | 1.35 |
| Calmar (CAGR/MDD) | **0.232** | 0.185 | 0.39 |
| Trades over 20Y | 122 | — | — |
| Gross win rate (trade P&L > 0) | **70.5%** | — | — |
| Avg winner | +10.0% (TP) | — | — |
| Avg loser | −15.0% (SL) | — | — |

### Marketable framing

> **"Every month we pick one stock. Limit sell at entry × 1.10 (GTC). Stop-loss at entry × 0.85 (GTC). Close at market if neither hits within 12 months. Over 20 years of backtest: 70% of trades hit the +10% target; 11.1% annualized return — 4.5 percentage points ahead of monthly DCA into SPY."**

## Research path

Conducted in 6 parallel research tracks (agents) + in-tree extensions. All code in `max/research/step4{1-4}*.py`, all results in `max/research/step4{1-4}*_results.json`, per-step summaries in `step4{1-4}*_summary.md`.

### Step 41 — Fixed-% TP grid (CAP5+SMA12M ranker)

Swept TP ∈ {2, 3, 5, 7, 10, 15, 20, 30}% × time_stop ∈ {10, 21, 42, 63, 126, 252, 504} bars. 56 combos. Single open position at a time; $1000/mo contributions accumulate into cash while a trade is open.

- **SPY DCA baseline**: +6.63% CAGR, 35.9% MDD, Sharpe 1.11.
- **TP=30% × 252d**: +13.15% CAGR but only 26 trades (small-sample, 53.8% WR).
- **TP=10% × 252d (winner)**: +11.52% CAGR, 78.8% MDD, 87.9% TP hit-rate, 58 trades.
- **TP=7% × 504d**: 94.2% WR, +7.67% CAGR — very high predictability but only edges SPY.

Sweet spot for CAGR × win-rate × sample-size: **TP=10% × 252d**.

### Step 42 — Volatility-adaptive TP (ATR / σ / quantile)

Swept ATR-based (k × ATR14/price), σ-based (k × realized-60d), and quantile-based (Q% of ticker's 3Y forward-return distribution). 80 combos.

- All vol-adaptive configs **underperform fixed-%**. Best vol-adaptive: quantile q=80 × ts=252 at +0.77% CAGR.
- Root cause: vol-scaled TPs tend to be small → fire often for small $ → cash re-enters idle cycle → capital never stays deployed.
- Also: CAP5+SMA12M only ranks from a ~27-ticker subset each month, so per-ticker vol equalization is moot — the ranker already prunes the vol extremes.
- **Conclusion: use fixed 10%. No meaningful gain from adaptive targets.**

### Step 43 — Multi-slot, regime gate, trailing stop

Tested concurrent positions (2/3/5/10 slots), SPY > 200dma regime gate, trailing stops.

- **Multi-slot hurts CAGR**: 2-slot → +9.0%, 3-slot → +8.6%, 5-slot → +7.6%, 10-slot → +3.8%. Diversification works but at a cost; single-pick concentrates the CAP5+SMA12M edge.
- **SPY>200dma regime gate costs ~3pp CAGR** for marginal MDD improvement. Skipping the 2008/2020/2022 crashes means missing the post-crash bounces.
- **Trailing stops mostly hurt**: premature exits on volatility.

### Step 44 — Single-slot + stop-loss sweep

Added hard stop-loss to the step41 winner. The breakthrough:

| Variant | CAGR | MDD | GrossWR | Calmar | avg_loser |
|---|---:|---:|---:|---:|---:|
| step41 winner (no stop) | +11.52% | 78.8% | 89.7% | 0.146 | **−35.64%** |
| **+ SL −15% (★)** | **+11.13%** | **48.0%** | **70.5%** | **0.232** | **−15.0%** |
| + SL −20% | +8.30% | 65.3% | 73.8% | 0.127 | −19.7% |
| + SL −25% | +8.61% | 63.8% | 77.7% | 0.135 | −24.3% |
| + SL −30% | +9.98% | 62.5% | 83.3% | 0.160 | −28.7% |

The −15% stop caps the tail. Each loss is now exactly −15%; each win is exactly +10%. Positive expected value with 2.4× more wins than losses in frequency. Sharpe rises, Calmar improves by 59%, CAGR barely dents.

## Why this beats the alternatives

| Approach | CAGR | WR | MDD | Signal clarity |
|---|---:|---:|---:|---|
| SPY DCA | 6.6% | n/a | 36% | Trivial but no alpha |
| CAP5+SMA12M hold-forever | 18.3% | n/a | 47% | Buy every month, no sell — not "retail practical" |
| **TP strategy (this)** | **11.1%** | **70%** | **48%** | **"Buy X at $P, sell at $T"** ✓ |
| Fixed 5%/60d TP | 0.9% | 79% | 81% | Too tight; net loses to idle cash |
| Fixed 10%/60d TP | 9.9% | 60% | 62% | OK but lower WR |
| Multi-slot (3) TP 10%/252d | 8.6% | 84% | 60% | Diversified but expensive in CAGR |
| Vol-adaptive TP (ATR/σ) | <1% | varies | low | Mostly idle |

The TP strategy is the **only** config that cleanly delivers all three user goals: (a) meaningfully beats SPY DCA, (b) actionable win rate, (c) concrete buy/sell prices.

## Honest caveats

1. **Quality-factor leakage**: the `quality` multiplier in CAP5 conviction is not strictly point-in-time (uses today's value at all historical dates). This biases CAGR *upward*. Real-world deployment should expect lower numbers; treat +11.1% as an optimistic ceiling.
2. **Survivorship**: 128-ticker universe = today's survivors. Delisted names don't exist in the backtest. True universe-level CAGR would be lower.
3. **Slippage**: TP fills assume limit fills exactly when intraday high ≥ target. SL fills assume stop-market fills at the stop price. Real fills can be worse in fast-moving markets (gaps in crises especially hurt stop fills).
4. **Idle cash drag**: Between trades, cash earns 0%. A money-market-fund bucket at 4% would add ~20–30 bps/yr.
5. **Single-name concentration**: One pick at a time. A fraud or sudden regulatory action is a bigger risk than a diversified strategy.
6. **Crisis-sample size**: 20 years includes 2008, 2020, 2022. Only 3 major crises. Future crises may behave differently.
7. **Win rate drops with stop**: 87.9% (no stop) → 70.5% (with −15% stop). The trade-off is capped losses vs higher WR. We choose capped losses — they make the strategy practical and the Sharpe/Calmar improvement is worth the lower WR headline.

## What ships to the webapp

1. **Scanner (`daily_scan_max.py`)**: today's top-ranked stock + entry-reference price (last close) + TP price (close × 1.10) + SL price (close × 0.85) + time-stop date (today + 252 trading days).
2. **Webapp "This month's pick"**: single-pick card displays buy/TP/SL prices as a natural limit-order recipe. Existing rank-weighted picks table remains (for users who prefer CAP5 hold-forever DCA).
3. **New "Take-Profit Strategy" backtest card**: loads bt_series, simulates the TP strategy in-browser, shows 20Y equity curve, metrics table, and trade log.
4. **Copy updates throughout**: hero, rules, validation card reflect the TP approach alongside the existing CAP5 story.
