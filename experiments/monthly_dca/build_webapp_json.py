"""Build a single JSON data file consumed by experiments/docs/monthly-dca/.

Includes:
  - live picks for the latest cached month-end (top 5/10 for several strategies)
  - walk-forward aggregate results
  - year-by-year breakdown for the recommended strategy
  - oracle ceiling
  - full backtest summary (sweep)
  - per-strategy summary stats
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_score import (
    BENCH_EXCLUDED,
    load_features_long,
    load_panel,
)
from experiments.monthly_dca.strategies_fast import (
    pullback_in_winner,
    quality_pullback,
    explosive_winners,
    dual_momentum,
)


OUT = Path(__file__).resolve().parents[2] / "experiments" / "docs" / "monthly-dca"
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT / "data.json"


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        if not np.isfinite(f):
            return None
        return f
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if pd.isna(x):
        return None
    return x


def live_picks(panel: pd.DataFrame, asof: pd.Timestamp, fn, top_k: int) -> list[dict]:
    feats = load_features_long().loc[asof]
    feats = feats.copy()
    feats.index = feats.index.get_level_values("ticker") if hasattr(feats.index, "get_level_values") else feats.index
    s = fn(feats).dropna()
    s = s[~s.index.isin(BENCH_EXCLUDED)]
    top = s.sort_values(ascending=False).head(top_k)
    out = []
    for tkr, score in top.items():
        row = feats.loc[tkr]
        out.append({
            "ticker": tkr,
            "score": float(score),
            "price": to_jsonable(row.get("price")),
            "pullback_1y": to_jsonable(row.get("pullback_1y")),
            "trend_health_5y": to_jsonable(row.get("trend_health_5y")),
            "recovery_rate": to_jsonable(row.get("recovery_rate")),
            "rsi_14": to_jsonable(row.get("rsi_14")),
            "mom_12_1": to_jsonable(row.get("mom_12_1")),
            "mom_3y": to_jsonable(row.get("mom_3y")),
            "d_sma200": to_jsonable(row.get("d_sma200")),
        })
    return out


def main() -> None:
    panel = load_panel()
    cache = Path(__file__).resolve().parent / "cache"

    # Latest cached month-end
    feats = load_features_long()
    asofs = sorted(feats.index.get_level_values("asof").unique())
    latest = asofs[-1]

    # Live picks
    picks_pin_5 = live_picks(panel, latest, pullback_in_winner, 5)
    picks_pin_10 = live_picks(panel, latest, pullback_in_winner, 10)
    picks_qp_5 = live_picks(panel, latest, quality_pullback, 5)
    picks_ew_5 = live_picks(panel, latest, explosive_winners, 5)
    picks_dm_5 = live_picks(panel, latest, dual_momentum, 5)

    # Walk-forward aggregate
    wf = pd.read_csv(cache / "wf_aggregate.csv")
    # only keep robust (TRAIN-top20 in >=4/8 splits) and sort by mean test CAGR
    wf_robust = wf[wf["n_splits_in_train_top20"] >= 4].sort_values("mean_test_cagr", ascending=False)

    # Per-split top picks for clarity
    splits = []
    for split_csv in sorted((cache).glob("wf_*_train.csv")):
        name = split_csv.name.replace("wf_", "").replace("_train.csv", "")
        train = pd.read_csv(split_csv)
        test_csv = cache / f"wf_{name}_test.csv"
        if not test_csv.exists():
            continue
        test = pd.read_csv(test_csv)
        train_top5 = train.sort_values("cagr_dca_portfolio", ascending=False).head(5)
        train_top5_keys = set(train_top5["key"])
        test_match = test[test["key"].isin(train_top5_keys)]
        splits.append({
            "name": name,
            "train_top5": [
                {
                    "key": r["key"],
                    "n_picks": int(r["n_picks"]),
                    "win_rate": float(r["win_rate"]),
                    "cagr": float(r["cagr_dca_portfolio"]),
                    "spy_cagr": float(r["cagr_spy_dca"]),
                    "edge": float(r["edge_vs_spy_dca"]),
                }
                for _, r in train_top5.iterrows()
            ],
            "test_same_configs": [
                {
                    "key": r["key"],
                    "n_picks": int(r["n_picks"]),
                    "win_rate": float(r["win_rate"]),
                    "cagr": float(r["cagr_dca_portfolio"]),
                    "spy_cagr": float(r["cagr_spy_dca"]),
                    "edge": float(r["edge_vs_spy_dca"]),
                }
                for _, r in test_match.iterrows()
            ],
        })

    # Year-by-year for the recommended strategy
    yb_pin_k1 = pd.read_csv(cache / "yb_pullback_in_winner_k1.csv")
    yb_qp_k3 = pd.read_csv(cache / "yb_quality_pullback_k1.csv")
    yb_pin_k5 = pd.read_csv(cache / "yb_pullback_in_winner_k5.csv")

    # Oracle
    oracle = pd.read_csv(cache / "oracle.csv")

    # Pick log for the recommended strategy (full history)
    pick_log = pd.read_csv(cache / "picks_full_pullback_in_winner_k1.csv")
    pick_log_records = pick_log.assign(asof=pd.to_datetime(pick_log["asof"]).dt.strftime("%Y-%m-%d"))[[
        "asof", "ticker", "score", "price", "pullback_1y", "trend_health_5y",
        "recovery_rate", "ret__hold_forever", "ret__fixed_3y", "ret__fixed_5y",
    ]].to_dict(orient="records")

    # Top sweep
    sweep = pd.read_csv(cache / "sweep_v1.csv")
    sweep_top = sweep.sort_values("cagr_dca_portfolio", ascending=False).head(40)

    out = {
        "as_of": str(latest.date()),
        "panel": {
            "n_tickers": int(panel.shape[1]),
            "first_date": str(panel.index.min().date()),
            "last_date": str(panel.index.max().date()),
        },
        "spy_dca_cagr": float(yb_pin_k1["cagr_dca_spy"].mean()),
        "live_picks": {
            "pullback_in_winner_top5": picks_pin_5,
            "pullback_in_winner_top10": picks_pin_10,
            "quality_pullback_top5": picks_qp_5,
            "explosive_winners_top5": picks_ew_5,
            "dual_momentum_top5": picks_dm_5,
        },
        "walk_forward_aggregate": [
            {
                "key": r["key"],
                "n_splits_in_train_top20": int(r["n_splits_in_train_top20"]),
                "n_splits_with_test_data": int(r["n_splits_with_test_data"]),
                "mean_test_cagr": float(r["mean_test_cagr"]),
                "median_test_cagr": float(r["median_test_cagr"]),
                "min_test_cagr": float(r["min_test_cagr"]),
                "max_test_cagr": float(r["max_test_cagr"]),
                "mean_test_edge": float(r["mean_test_edge"]),
                "min_test_edge": float(r["min_test_edge"]),
                "mean_test_win": float(r["mean_test_win"]),
            }
            for _, r in wf_robust.iterrows()
        ],
        "splits": splits,
        "year_by_year": {
            "pullback_in_winner_k1": yb_pin_k1.to_dict(orient="records"),
            "quality_pullback_k1": yb_qp_k3.to_dict(orient="records"),
            "pullback_in_winner_k5": yb_pin_k5.to_dict(orient="records"),
        },
        "oracle": oracle.to_dict(orient="records"),
        "pick_log": pick_log_records,
        "sweep_top40": sweep_top.to_dict(orient="records"),
    }

    out = to_jsonable(out)

    # Pretty-print with stable key order
    with open(DATA_OUT, "w") as f:
        json.dump(out, f, indent=1, default=str)
    size_kb = DATA_OUT.stat().st_size / 1024
    print(f"Wrote {DATA_OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
