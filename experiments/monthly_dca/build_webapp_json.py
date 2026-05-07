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


def growth_curve(panel: pd.DataFrame, picks_csv: pd.DataFrame, ticker_col: str = "ticker",
                 asof_col: str = "asof", entry_col: str = "price") -> list[dict]:
    """Monthly snapshot of (cumulative invested, strategy value, SPY DCA value).

    For every month-end T from the first pick date through eval, sum:
      - strategy value = sum over picks made at asof <= T of (price[ticker, T] / entry_price)
      - spy value     = sum over picks made at asof <= T of (price[SPY, T] / price[SPY, asof])
      - invested      = count of picks made at asof <= T

    Each pick contributes $1 at its asof. We treat the basket as held forever.
    """
    picks = picks_csv.copy()
    picks[asof_col] = pd.to_datetime(picks[asof_col])
    me = month_end_dates(panel.index)
    me = me[me >= picks[asof_col].min()]
    spy = panel["SPY"]

    # Pre-extract entry prices for SPY at each pick's asof
    spy_at_pick = []
    for asof_t in picks[asof_col]:
        pos = panel.index.searchsorted(asof_t)
        spy_at_pick.append(float(spy.iloc[pos]) if pos < len(spy) else float("nan"))
    picks["_spy_entry"] = spy_at_pick

    out: list[dict] = []
    for d in me:
        # Picks made on or before d
        sub = picks[picks[asof_col] <= d]
        if sub.empty:
            continue
        # Strategy value
        strat_val = 0.0
        for _, p in sub.iterrows():
            t = p[ticker_col]
            entry = float(p[entry_col])
            if t not in panel.columns or entry == 0 or not np.isfinite(entry):
                strat_val += 1.0  # neutral
                continue
            # Price at d (or last available before d)
            s = panel[t].loc[panel.index <= d].dropna()
            if s.empty:
                strat_val += 0.0
                continue
            cur = float(s.iloc[-1])
            strat_val += cur / entry
        # SPY DCA value (same dates, $1 each)
        spy_val = 0.0
        for _, p in sub.iterrows():
            entry = p["_spy_entry"]
            if not np.isfinite(entry) or entry == 0:
                continue
            cur = float(spy.loc[spy.index <= d].dropna().iloc[-1])
            spy_val += cur / entry
        invested = float(len(sub))
        out.append({
            "date": str(d.date()),
            "invested": invested,
            "strat_value": float(strat_val),
            "spy_value": float(spy_val),
        })
    return out


from experiments.monthly_dca.backtester import month_end_dates


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

    # Single "pick of the month" — top-1 from pullback_in_winner with full feature snapshot
    pick_of_month = picks_pin_5[0] if picks_pin_5 else None

    # Growth curve over the full backtest window (k=1 hold_forever, the recommended config)
    pick_log_full = pd.read_csv(cache / "picks_full_pullback_in_winner_k1.csv")
    growth = growth_curve(panel, pick_log_full)

    # Headline backtest stats from k=1 hold_forever (full window)
    pin_summary_path = cache / "summary_pullback_in_winner_k1.json"
    headline = {}
    if pin_summary_path.exists():
        with open(pin_summary_path) as f:
            ps = json.load(f)
        s = ps.get("stats", {})
        headline = {
            "n_picks": int(s.get("n", 0)),
            "win_rate_raw": float(s.get("win_rate", 0)),
            "win_rate_bias_corr": float(s.get("win_rate_bias_corr_median") or 0),
            "cagr_raw": float(s.get("cagr_dca", 0)),
            "cagr_bias_corr": float(s.get("cagr_dca_bias_corr_median") or 0),
            "cagr_spy_dca": float(s.get("cagr_spy_dca", 0)),
            "edge": float(s.get("edge", 0)),
        }

    # Survivorship-bias study (random baseline, sensitivity, etc.)
    surv_path = cache / "survivorship_summary.json"
    survivorship = None
    if surv_path.exists():
        with open(surv_path) as f:
            survivorship = json.load(f)

    out = {
        "as_of": str(latest.date()),
        "panel": {
            "n_tickers": int(panel.shape[1]),
            "first_date": str(panel.index.min().date()),
            "last_date": str(panel.index.max().date()),
        },
        "spy_dca_cagr": float(yb_pin_k1["cagr_dca_spy"].mean()),
        "headline": headline,
        "pick_of_month": pick_of_month,
        "growth": growth,
        "survivorship": survivorship,
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
