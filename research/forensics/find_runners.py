"""Identify historical 3x-in-12-months runners from the price panel.

For each ticker, scan all month-ends T and check whether the stock
≥3x'd in any 252-trading-day window starting at T.  We then capture
metadata about the run (when it started, how long it took, peak return,
peak date) and return a DataFrame of run events.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import load_panel, load_feature_months


def find_runs(panel: pd.DataFrame, mult: float = 3.0, lookahead_days: int = 252,
              min_history_days: int = 252) -> pd.DataFrame:
    """Return DataFrame of (ticker, start_date, peak_date, peak_ret, days_to_peak).

    A 'run' starts at month-end T if max(price[T..T+lookahead]) / price[T] >= mult.
    We deduplicate overlapping runs by keeping only the first start in any
    rolling 252-day window per ticker.
    """
    months = load_feature_months()
    panel_idx = panel.index
    # Map month -> panel pos
    month_pos = {}
    for m in months:
        p = panel_idx.searchsorted(m)
        if p >= len(panel_idx):
            continue
        if panel_idx[p] != m:
            p = max(0, p - 1)
        month_pos[m] = p

    runs = []
    for tkr in panel.columns:
        s = panel[tkr].to_numpy()
        last_run_pos = -10**9  # to dedupe overlapping runs
        for m in months:
            pos = month_pos.get(m)
            if pos is None or pos < min_history_days:
                continue
            entry = s[pos]
            if not np.isfinite(entry) or entry <= 0:
                continue
            end = min(pos + lookahead_days, len(s) - 1)
            window = s[pos:end + 1]
            mask = np.isfinite(window)
            if not mask.any():
                continue
            valid = window[mask]
            if valid.max() / entry >= mult:
                # Find peak position within window
                argmax_local = int(np.argmax(np.where(mask, window, -np.inf)))
                peak_pos = pos + argmax_local
                if pos - last_run_pos < lookahead_days:
                    # Overlapping with previous run, skip
                    continue
                runs.append({
                    "ticker": tkr,
                    "start_date": m,
                    "start_pos": pos,
                    "start_px": float(entry),
                    "peak_date": panel_idx[peak_pos],
                    "peak_pos": int(peak_pos),
                    "peak_px": float(window[argmax_local]),
                    "peak_ret": float(window[argmax_local] / entry - 1.0),
                    "days_to_peak": int(argmax_local),
                })
                last_run_pos = pos
    return pd.DataFrame(runs)


def find_non_runners(panel: pd.DataFrame, runs: pd.DataFrame,
                      lookahead_days: int = 252,
                      min_history_days: int = 252,
                      sample_per_year: int = 50,
                      seed: int = 0) -> pd.DataFrame:
    """Sample 'non-runners' — month-end stock-states whose 252d max return < 1.5x.

    Used to build a contrastive pre-runner-vs-non-runner dataset.
    """
    rng = np.random.default_rng(seed)
    months = load_feature_months()
    panel_idx = panel.index
    runs_set = set(zip(runs["ticker"], runs["start_date"]))

    candidates = []
    for tkr in panel.columns:
        s = panel[tkr].to_numpy()
        for m in months:
            if (tkr, m) in runs_set:
                continue
            p = panel_idx.searchsorted(m)
            if p >= len(panel_idx):
                continue
            if panel_idx[p] != m:
                p = max(0, p - 1)
            if p < min_history_days:
                continue
            entry = s[p]
            if not np.isfinite(entry) or entry <= 0:
                continue
            end = min(p + lookahead_days, len(s) - 1)
            window = s[p:end + 1]
            mask = np.isfinite(window)
            if not mask.any():
                continue
            mx = window[mask].max() / entry
            if mx < 1.5:
                candidates.append({
                    "ticker": tkr,
                    "start_date": m,
                    "start_pos": p,
                    "start_px": float(entry),
                    "max_252d_return": float(mx - 1.0),
                })
    df = pd.DataFrame(candidates)
    if df.empty:
        return df
    # Stratified sample by year
    df = df.assign(_year=df["start_date"].dt.year)
    sampled = df.groupby("_year", group_keys=False).apply(
        lambda g: g.sample(min(len(g), sample_per_year), random_state=seed)
    )
    if "_year" in sampled.columns:
        sampled = sampled.drop(columns=["_year"])
    return sampled.reset_index(drop=True)


def main():
    panel = load_panel()
    print(f"Panel: {panel.shape}, {panel.index.min()} -> {panel.index.max()}")

    print("Finding 3x-in-12mo runners...")
    runs = find_runs(panel, mult=3.0, lookahead_days=252)
    runs = runs.sort_values(["ticker", "start_date"]).reset_index(drop=True)
    out = Path(__file__).parent / "runs_3x_12m.parquet"
    runs.to_parquet(out)
    print(f"Found {len(runs)} 3x-in-12mo runs across {runs['ticker'].nunique()} tickers")
    print(f"Saved to {out}")
    print()
    print("Top 20 by peak_ret:")
    print(runs.sort_values("peak_ret", ascending=False).head(20).to_string(index=False))
    print()
    print("Year distribution:")
    print(runs["start_date"].dt.year.value_counts().sort_index())

    print()
    print("Sampling non-runners (50 per year)...")
    non = find_non_runners(panel, runs, sample_per_year=50)
    out2 = Path(__file__).parent / "non_runners_sample.parquet"
    non.to_parquet(out2)
    print(f"Sampled {len(non)} non-runner stock-months. Saved to {out2}")


if __name__ == "__main__":
    main()
