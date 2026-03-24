# Crypto TMD-ARC Experiment Program

## Overview

This is an autonomous experiment loop for developing a profitable crypto trading strategy
based on the Temporal Momentum Dispersion with Adaptive Regime Cascade (TMD-ARC) framework.

Following Karpathy's autoresearch paradigm:
- **prepare.py** is FIXED (data download, feature computation, evaluation harness)
- **train.py** is MODIFIABLE (strategy parameters and logic)
- Results tracked in **results/results.tsv**

## Objective

Achieve a **Sharpe ratio > 5** on the out-of-sample test period (2023-2026) for
cryptocurrency markets, using the TMD-ARC signal framework adapted for crypto.

## Rules

1. Only modify `train.py` — never modify `prepare.py`
2. All features must use ONLY past data (no lookahead bias)
3. Transaction costs are fixed at 15 bps per trade (crypto exchange fees)
4. Data splits are fixed (train: 2018-2021, valid: 2022-2023, test: 2023-2026)
5. 90-day buffers between splits prevent feature leakage
6. BTC-USD is the market benchmark (analogous to SPY for stocks)
7. Universe: ~50 liquid cryptocurrencies available on Yahoo Finance

## Execution Loop

1. Modify train.py with experimental hypothesis
2. Run `python train.py`
3. Check sharpe metric in output
4. Log results to results/results.tsv
5. Keep if improved, revert if not
6. Repeat with new ideas

## Key Crypto Adaptations

- 365 trading days/year (24/7 markets)
- Higher base volatility → adjusted thresholds
- BTC as market leader for cascade analysis
- Shorter cascade lag (crypto info propagates faster)
- Higher transaction costs (15 bps vs 10 bps for stocks)
