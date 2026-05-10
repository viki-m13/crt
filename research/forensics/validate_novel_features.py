"""For each runner / non-runner event-date, compute the novel features
and report AUC.  Tests whether the novel signals add information beyond
what's already in the cached feature set."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import load_panel
from strategy.features.novel_features import compute_all_novel

ROOT = Path(__file__).resolve().parents[2]
FORENSICS = ROOT / "research" / "forensics"


def main(sample_per_class: int = 500):
    runs = pd.read_parquet(FORENSICS / "runs_3x_12m.parquet")
    non = pd.read_parquet(FORENSICS / "non_runners_sample.parquet")
    runs = runs.sample(min(len(runs), sample_per_class), random_state=0)
    non = non.sample(min(len(non), sample_per_class), random_state=0)
    runs = runs.assign(label=1)[["ticker", "start_date", "label"]]
    non = non.assign(label=0)[["ticker", "start_date", "label"]]
    events = pd.concat([runs, non], ignore_index=True)
    events = events.sort_values("start_date")

    panel = load_panel()
    rows = []
    cache = None
    cache_date = None
    for i, row in events.iterrows():
        d = row["start_date"]
        if cache_date != d:
            try:
                cache = compute_all_novel(panel, d)
            except Exception:
                cache = None
            cache_date = d
        if cache is None or row["ticker"] not in cache.index:
            continue
        rec = cache.loc[row["ticker"]].to_dict()
        rec["ticker"] = row["ticker"]
        rec["start_date"] = d
        rec["label"] = row["label"]
        rows.append(rec)
        if len(rows) % 200 == 0:
            print(f"  {len(rows)} events processed")
    df = pd.DataFrame(rows)
    df.to_parquet(FORENSICS / "novel_features_dataset.parquet")
    print(f"Saved {len(df)} feature rows.")

    feature_cols = [c for c in df.columns if c not in ("ticker", "start_date", "label")]
    print()
    print(f"{'feature':25s} {'n_run':>7s} {'n_non':>7s} {'med_run':>10s} {'med_non':>10s} {'AUC':>6s}")
    print("-" * 70)
    rows = []
    for c in feature_cols:
        r = df[df.label == 1][c].astype(float).dropna()
        n = df[df.label == 0][c].astype(float).dropna()
        if len(r) < 30 or len(n) < 30:
            continue
        all_vals = np.concatenate([r.values, n.values])
        labels = np.concatenate([np.ones(len(r)), np.zeros(len(n))])
        order = np.argsort(all_vals)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(all_vals) + 1)
        n_pos = len(r); n_neg = len(n)
        sum_ranks_pos = float(ranks[labels == 1].sum())
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        rows.append({"feature": c, "n_run": len(r), "n_non": len(n),
                     "med_run": float(r.median()), "med_non": float(n.median()),
                     "auc": float(auc)})
    out = pd.DataFrame(rows).sort_values("auc", ascending=False, key=lambda s: s.abs() - 0.5 + 0)
    # sort by |auc - 0.5|
    out["sep"] = (out["auc"] - 0.5).abs()
    out = out.sort_values("sep", ascending=False).drop(columns=["sep"])
    out.to_csv(FORENSICS / "novel_feature_auc.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
