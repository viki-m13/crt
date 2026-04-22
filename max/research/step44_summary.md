# Step 44 — Single-slot TP+stop-loss optimization

## Objective

Preserve step41's CAGR winner (TP=10%, time_stop=252d, +11.52% CAGR, 87.9% TP-hit rate) while cutting the 78.8% max drawdown that makes it practically untradeable for a retail monthly-picker strategy.

## Headline result

| Config | CAGR | MDD | Sharpe | Calmar | GrossWR | N | avg_winner | avg_loser |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY DCA baseline | +6.63% | 35.9% | 1.11 | 0.185 | n/a | n/a | n/a | n/a |
| TP10/TS252 (step41 winner) | +11.52% | 78.8% | 0.99 | 0.146 | 89.7% | 58 | +9.86% | −35.64% |
| **TP10/TS252/SL15 (★)** | **+11.13%** | **48.0%** | **1.08** | **0.232** | **70.5%** | **122** | **+10.00%** | **−15.00%** |
| TP10/TS252/SL25 | +8.61% | 63.8% | 1.00 | 0.135 | 77.7% | 103 | +10.00% | −24.34% |
| TP10/TS252/SL30 | +9.98% | 62.5% | 1.01 | 0.160 | 83.3% | 90 | +10.00% | −28.65% |
| TP10/TS252/Trail20 | +9.73% | 56.6% | 1.04 | 0.172 | 72.3% | 112 | +10.00% | −16.98% |
| TP12/TS378 | +11.34% | 72.1% | 1.04 | 0.157 | 87.0% | 46 | +12.00% | −25.44% |

## Recommendation

**Production pick: TP=10%, time_stop=252 bars (~12 months), stop_loss=−15%.**

- CAGR +11.13% vs SPY DCA +6.63% → **+4.50pp excess**.
- MDD 48.0% vs baseline 78.8% → **shaved 30pp off drawdown** for a trivial 0.39pp CAGR cost.
- Sharpe 1.08 ≈ SPY's 1.11; **Calmar 0.232 vs SPY's 0.185 — 25% better risk-adj**.
- 122 trades in 20Y (~6/yr) — 2× more than baseline's 58 because stops cycle capital faster.
- 70.5% gross win rate. Each win = +10% exactly (limit fills at TP); each loss = −15% exactly (stop fills at SL). **Asymmetric payoff: 1:1.5 win/loss magnitude but 2.4:1 win/loss frequency → positive EV with predictable geometry.**

### Why not the raw step41 winner?
- +11.52% CAGR looks better, but 89.7% WR is inflated by 7 time-stops at avg −35.64% — when it loses it loses BIG. That's 12% of trades taking ~35% haircuts, which drives the 78.8% MDD.
- Stop-loss trades some headline win rate for capped losses. The CAGR cost is trivial.

### Marketable framing
> "Every month we pick one stock. Sell target: +10% (limit order, GTC). Hard stop-loss: −15%. If neither hits within 12 months, close at market. 70% of trades hit the +10% target. Backtest 2006–2026: 11.1% CAGR vs SPY DCA's 6.6%."

## Caveats

- **Quality leakage**: the `quality` multiplier inside `final` (from bt_ext.parquet) isn't strictly point-in-time — it uses today's snapshot at all historical dates. This biases CAGR upward. Real-world will be lower; treat these numbers as an optimistic ceiling.
- **Survivorship**: 128 tickers = today's survivors. Names that delisted 2008–2020 aren't in the pool. Real-world underperforms.
- **Slippage**: TP fills assume a limit order at target price fills exactly when intraday high ≥ target. Stop-loss fills assume stop-market fills at exactly the stop price. Both are simplifications; real fills can be worse in fast-moving markets (stops especially gap through in crises).
- **Idle cash drag**: between trades, cash earns 0%. Money-market at 4% would add ~20–30 bps.
- **Single-name risk**: one pick at a time. Catastrophic single-name events (fraud, regulatory action) can produce a −15% stop in one day, no warning.
- **Single regime**: backtest includes 2008 GFC, 2020 COVID, 2022 bear — but the sample of crises is small (n=3). Future crises may behave differently.
