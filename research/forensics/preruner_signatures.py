"""For each runner and non-runner stock-month, fetch the ~67 cached features
at start_date.  Compare distributions.  This is the forensic step:
'what does the pre-runner footprint look like, compared to a typical stock?'
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import load_features

ROOT = Path(__file__).resolve().parents[2]
FORENSICS = ROOT / "research" / "forensics"


def build_dataset(runs_path: Path, non_runners_path: Path, out_path: Path,
                   max_per_class: int | None = None) -> pd.DataFrame:
    runs = pd.read_parquet(runs_path)
    non = pd.read_parquet(non_runners_path)
    if max_per_class:
        runs = runs.sample(min(len(runs), max_per_class), random_state=0)
        non = non.sample(min(len(non), max_per_class), random_state=0)
    runs = runs.assign(label=1)[["ticker", "start_date", "label"]]
    non = non.assign(label=0)[["ticker", "start_date", "label"]]
    all_events = pd.concat([runs, non], ignore_index=True)
    print(f"Total events: {len(all_events)} ({len(runs)} runners, {len(non)} non-runners)")

    # For each event, look up features at start_date
    rows = []
    cache = {}
    for i, row in all_events.iterrows():
        d = row["start_date"]
        if d not in cache:
            try:
                cache[d] = load_features(d)
            except Exception:
                cache[d] = None
        feats = cache[d]
        if feats is None or row["ticker"] not in feats.index:
            continue
        rec = feats.loc[row["ticker"]].to_dict()
        rec["ticker"] = row["ticker"]
        rec["start_date"] = d
        rec["label"] = row["label"]
        rows.append(rec)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(all_events)}")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"Saved {len(df)} feature rows to {out_path}")
    return df


def compare_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature: compare runner vs non-runner medians + AUC."""
    feature_cols = [c for c in df.columns if c not in ("ticker", "start_date", "label", "price")]
    df = df.dropna(subset=["label"])
    runners = df[df.label == 1]
    non = df[df.label == 0]
    rows = []
    for c in feature_cols:
        r = runners[c].astype(float).dropna()
        n = non[c].astype(float).dropna()
        if len(r) < 30 or len(n) < 30:
            continue
        # Mann-Whitney AUC (rough): compare ranks
        all_vals = np.concatenate([r.values, n.values])
        labels = np.concatenate([np.ones(len(r)), np.zeros(len(n))])
        order = np.argsort(all_vals)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(all_vals) + 1)
        # AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
        n_pos = len(r); n_neg = len(n)
        sum_ranks_pos = float(ranks[labels == 1].sum())
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        rows.append({
            "feature": c,
            "n_runners": int(len(r)),
            "n_non": int(len(n)),
            "median_runner": float(r.median()),
            "median_non": float(n.median()),
            "auc": float(auc),
            "lift_median": float(r.median() - n.median()),
        })
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def main():
    runs_path = FORENSICS / "runs_3x_12m.parquet"
    non_path = FORENSICS / "non_runners_sample.parquet"
    out_path = FORENSICS / "preruner_features.parquet"
    df = build_dataset(runs_path, non_path, out_path)

    cmp = compare_distributions(df)
    cmp.to_csv(FORENSICS / "feature_auc_table.csv", index=False)
    print()
    print("Top 25 features by AUC (closer to 1 = stronger separator favoring runners):")
    print(cmp.head(25).to_string(index=False))
    print()
    print("Bottom 10 features (lowest AUC favors non-runners):")
    print(cmp.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
