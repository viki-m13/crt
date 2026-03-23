# TMD-ARC Strategy Research

This is an autonomous strategy research project, modeled directly on
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, evaluation. Do not modify.
   - `train.py` — the file you modify. Strategy parameters, signal logic, position sizing.
4. **Verify data exists**: Check that `data/` contains CSV files with market data. If not, run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs a backtest on historical market data. The strategy script
runs for all trading days in the training period (2010-2019). You launch it as:
`python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  strategy parameters, signal logic, entry/exit conditions, position sizing,
  feature engineering, regime detection, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation,
  data loading, feature computation, and data split constants.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_strategy` function in `prepare.py`
  is the ground truth metric.

**The goal: get the highest Sharpe ratio on the training period
while maintaining robustness (low max drawdown, reasonable trade count).**

Since the data is fixed, you don't need to worry about download time.
Everything is fair game: change the entry logic, the exit logic, the
features, the position sizing. The only constraint is that the code
runs without crashing.

**Evaluation metric**: Sharpe ratio (annualized, after 10bps transaction costs)
**Secondary metrics**: CAGR, Max Drawdown, Win Rate, Number of Trades

## Output format

The script prints a summary like this:

```
---
sharpe:          2.839
cagr:            32.68%
max_drawdown:    -7.52%
n_trades:        2901
win_rate:        52.60%
profit_factor:   1.30
avg_hold_days:   9.8
```

You can extract the key metric:
```
grep "^sharpe:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 5 columns:

```
commit	sharpe	max_dd	status	description
```

1. git commit hash (short, 7 chars)
2. sharpe ratio achieved — use 0.000 for crashes
3. max drawdown (e.g. -7.5) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1`
5. Read out the results: `grep "^sharpe:\|^max_drawdown:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log`
7. Record the results in the tsv
8. If sharpe improved (higher), you "advance" the branch, keeping the commit
9. If sharpe is equal or worse, you git reset back to where you started

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask
the human if you should continue. The human might be asleep. You are autonomous.
