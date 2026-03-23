# TMD-ARC Research Log

## Temporal Momentum Dispersion with Adaptive Regime Cascade

**Date:** 2026-03-23
**Author:** Claude (autonomous strategy discovery agent)
**Inspiration:** Karpathy's AgentHub + Autoresearch paradigm

---

## 1. Background & Motivation

### Current CRT Approach
The existing Daily Stock Guide system uses:
- k-NN analog matching for pullback recovery odds
- Quality filtering (trend quality, recovery track record)
- Washout meter (drawdown depth scoring)
- Edge score based on 250 nearest-neighbor historical analogs

**Limitation:** This is fundamentally a *reactive* system — it waits for pullbacks
and then estimates recovery odds. It doesn't predict *when* or *which* stocks will
move, and it's limited to pullback-recovery scenarios.

### AgentHub / Autoresearch Inspiration
Karpathy's AgentHub (https://github.com/karpathy/agenthub) introduces a paradigm
where autonomous AI agents collaborate on research through:
- An autonomous experiment loop (propose → test → keep/discard → repeat)
- Git DAG for tracking experiment lineage
- Message board for agent coordination

We apply this paradigm to trading strategy discovery:
- Each "experiment" tests a strategy variant
- Results are logged in a TSV (mirroring autoresearch's results.tsv)
- The best variants survive, the rest are discarded
- Walk-forward validation prevents overfitting

### Downloads & References
- AgentHub fork: cloned from https://github.com/ygivenx/agenthub (MIT license)
  - Original repo (https://github.com/karpathy/agenthub) returned 404
  - Stored in: `experiments/lib/agenthub/`
- Autoresearch: cloned from https://github.com/karpathy/autoresearch (MIT license)
  - Stored in: `experiments/lib/autoresearch/`
- Market data: downloaded via yfinance from Yahoo Finance
  - Stored in: `experiments/data/` (CSV format)
  - 119 tickers: 21 ETFs + 98 large-cap stocks
  - Period: 2008-01-01 to present

---

## 2. Novel Strategy: TMD-ARC

### Core Thesis
**When momentum signals across different timeframes strongly disagree, a
predictable resolution follows.** This is fundamentally different from:
- Simple momentum (one timeframe)
- Mean reversion (assumes prices return to average)
- Pullback recovery (waits for drawdowns)

TMD-ARC predicts *regime transitions* by measuring the *shape* of momentum
across timeframes.

### Patentable Novel Elements

#### 2.1 Multi-Timeframe Momentum Dispersion Index (MTMDI)
- Computes momentum at 6 timeframes: 5, 10, 21, 63, 126, 252 days
- Z-scores each momentum within its own rolling history
- MTMDI = standard deviation of z-scored momentum across timeframes
- High MTMDI = timeframes disagree = regime transition imminent
- Direction determined by which timeframes dominate

**Why novel:** Previous momentum dispersion research (e.g., Grinblatt & Moskowitz)
focused on *cross-sectional* dispersion (across stocks). MTMDI measures
*temporal* dispersion (across timeframes within a single stock).

#### 2.2 Cross-Asset Cascade Score (CACS)
- Measures information propagation lag from market leaders to followers
- Computes rolling correlation at multiple lags (0-5 days)
- If lagged correlations > contemporaneous → stock is a follower
- Cascade gap = expected move (based on leader) - actual move

**Why novel:** Existing lead-lag research uses pair trading. CACS uses a
continuous cascade score with dynamic beta adjustment.

#### 2.3 Momentum Persistence Ratio (MPR)
- Measures whether momentum is accelerating or decelerating
- Ratio of recent momentum rate to average momentum rate
- MPR > 1 = accelerating, MPR < 1 = decelerating

**Why novel:** Most momentum indicators measure level. MPR measures the
*derivative* of momentum — the curvature of the cumulative return curve.

#### 2.4 Regime-Adaptive Position Sizing
- Uses volatility term structure to classify regimes (low/normal/high vol)
- Position size = vol_target / stock_vol × strength × regime_factor
- High-vol regime → 50% size reduction
- Prevents blow-ups during crises

### Entry Logic
Enter long when ALL conditions met:
1. MTMDI z-score > threshold (timeframes disagree)
2. MTMDI direction positive (short-term recovering vs long-term)
3. At least one confirming factor:
   - Cascade gap > threshold (stock lagging expected move)
   - MPR > threshold (momentum accelerating)

### Exit Logic
Exit when ANY condition met:
1. MTMDI z-score collapses below exit threshold (resolution complete)
2. Stop loss hit (-8%)
3. Take profit hit (+20%)
4. Maximum holding period exceeded (63 days)

---

## 3. Anti-Overfitting Methodology

### Data Splits
| Split      | Period          | Purpose                    |
|------------|-----------------|----------------------------|
| Training   | 2010-01 to 2019-12 | Strategy development       |
| Validation | 2020-04 to 2022-12 | Out-of-sample validation   |
| Test       | 2023-04 to 2026-03 | Final evaluation (once!)   |

**Buffer zones:** 3-month gaps between splits prevent feature leakage
from lookback windows crossing split boundaries.

### Validation Tests
1. **Walk-Forward Analysis** (5 folds, expanding training window)
2. **Bootstrap Confidence Intervals** (1000 block-bootstrap resamples)
3. **Permutation Test** (200 random parameter configurations)
4. **Parameter Sensitivity Analysis** (vary each parameter independently)
5. **Benchmark Comparison** (vs SPY buy-and-hold, random entry)

### Honest Reporting
- Transaction costs: 10 bps per trade (each way)
- No survivorship bias (using full history, not just current survivors)
- No look-ahead bias (features use only past data)
- No optimization on test set (test evaluated once)

---

## 4. Experiment Log

See `experiments/results/results.tsv` for the full experiment log.

### Autonomous Experiment Loop
Following the autoresearch paradigm:
- 24 experiments run on training data (2010-2019)
- Each tests a different parameter configuration or structural variant
- Best variant selected by Sharpe ratio (minimum 20 trades required)
- Best config then validated out-of-sample

### Experiment Categories
1. MTMDI threshold exploration
2. Cascade sensitivity exploration
3. Momentum filter exploration
4. Risk management exploration
5. Position sizing exploration
6. Conservative combined
7. Aggressive combined
8. Wide random exploration

---

## 5. Results

See `experiments/results/` for detailed output:
- `baseline_metrics.json` — baseline strategy on training data
- `results.tsv` — all experiments
- `validation_report.json` — validation suite results
- `test_results.json` — final out-of-sample results

---

## 6. Comparison to Current CRT Approach

| Feature | CRT (Current) | TMD-ARC (New) |
|---------|---------------|---------------|
| Signal type | Pullback recovery odds | Regime transition prediction |
| Entry trigger | High washout + quality | High MTMDI + cascade + MPR |
| Holding period | 1-5 years | 10-63 days |
| Position sizing | Fixed | Volatility-targeted, regime-adaptive |
| Exit logic | Time-based | Multi-signal (resolution, SL, TP) |
| Turnover | ~30/year | Higher (100+/year) |
| Market dependency | Works best in corrections | Works in all regimes |

### Key Advantages of TMD-ARC
1. **More opportunities:** Signals in all market conditions, not just pullbacks
2. **Shorter holding period:** Faster capital turnover = higher capital efficiency
3. **Adaptive sizing:** Protects during high-vol regimes
4. **Diversified signals:** Multiple confirming factors reduce false positives

---

## 7. File Structure

```
experiments/
├── data/                      # Downloaded market data (CSV)
│   ├── download_manifest.json # Data provenance tracking
│   └── {TICKER}.csv          # Per-ticker OHLCV data
├── docs/
│   └── RESEARCH_LOG.md       # This file
├── lib/
│   ├── agenthub/             # Karpathy's AgentHub (MIT, fork)
│   └── autoresearch/         # Karpathy's Autoresearch (MIT)
├── results/
│   ├── results.tsv           # Experiment log (autoresearch-style)
│   ├── baseline_metrics.json # Baseline results
│   ├── validation_report.json# Validation suite output
│   └── test_results.json     # Final out-of-sample results
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      # Data download + split management
│   ├── features.py           # Novel feature engineering (MTMDI, CACS, MPR)
│   ├── strategy.py           # TMD-ARC strategy engine
│   ├── backtest.py           # Walk-forward backtesting framework
│   ├── experiment_loop.py    # AgentHub-inspired experiment runner
│   └── validation.py         # Anti-overfitting validation suite
├── requirements.txt          # Python dependencies
└── run_all.py               # Full pipeline runner
```

---

## 8. Reproducibility

To reproduce all results:
```bash
cd experiments/
pip install -r requirements.txt
python run_all.py
```

All random seeds are not fixed (intentionally) — results should be
robust across different random states. If they're not, that's a red flag
for overfitting.
