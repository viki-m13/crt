"""Phase 8b: K=2 v5 generalization sweep across alternate universes.

The deployed v5 winner config (K=2, Chronos q=0.45, hold=6, cap=0.40,
tight regime) was tuned on PIT S&P 500. This script tests whether
the picker's edge persists when the eligible universe is widened or
shifted.

Universes:
  1. PIT_SP500_augmented      The deployed universe (985 PIT tickers,
                                 baseline 49.39% WF mean).
  2. broader_augmented        Full augmented panel — ~1964 tickers
                                 (PIT members + non-PIT US stocks).
  3. non_sp500_augmented      Augmented panel MINUS PIT members
                                 (~1205 non-S&P-500 names).
  4. random_500_seed{1,2,3}   Three random 500-ticker subsets.

Each test runs the SAME v5 picker (ml_3plus6 + Chronos p70 q=0.45 +
top-2 + invvol cap=0.4 + tight gate + 6m hold). The only change is
which set of tickers is eligible at each rebalance.

Input: augmented/ml_preds.parquet, augmented/ml_preds_chronos_broader.parquet
       (broader Chronos must be regenerated first via
       score_chronos_broader_aug.py).

Output:
  augmented/v5_k2_generalize.csv     side-by-side metrics per universe
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_v5_aug import (  # noqa: E402
    AUG, PIT, EXCLUDE, COST_BPS, WF_SPLITS,
    classify_regime_tight, load_spy_features, calc_invvol_weights,
)

K = 2
CHR_Q = 0.45
HOLD = 6
CAP = 0.40


def universe_pit_sp500(asof, members_g, ml_tickers_at):
    return members_g.get(asof, set()) & ml_tickers_at


def universe_broader(asof, members_g, ml_tickers_at):
    return ml_tickers_at  # everything with a prediction


def universe_non_pit(asof, members_g, ml_tickers_at):
    return ml_tickers_at - members_g.get(asof, set())


def make_random_subset_fn(seed: int, n: int = 500):
    """Returns a universe_fn that picks n random tickers consistent across asofs."""
    rng = np.random.default_rng(seed)

    def fn(asof, members_g, ml_tickers_at):
        if not hasattr(fn, "_sample"):
            # Defer sample initialization until we have a full universe
            return None
        return ml_tickers_at & fn._sample
    fn._seed = seed
    fn._n = n
    fn._sample = None
    fn._rng = rng
    return fn


def run_v5_universe(universe_fn, panel, ml, chr_, spy, mr, members_g,
                    universe_name: str,
                    k=K, chr_q=CHR_Q, hold=HOLD, cap=CAP) -> dict:
    cf = COST_BPS / 1e4
    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    ml_by_asof = {a: g for a, g in ml.groupby("asof")}
    chr_by_asof = {a: g for a, g in chr_.groupby("asof")}

    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]

    # For random subsets: build sample from full universe of predicted tickers
    if hasattr(universe_fn, "_sample") and universe_fn._sample is None:
        all_predicted = set(ml["ticker"].unique())
        universe_fn._sample = set(universe_fn._rng.choice(
            list(all_predicted), size=min(universe_fn._n, len(all_predicted)),
            replace=False))

    cur_picks = []; cur_weights = np.array([])
    cash = False; held_for = 0; equity = 1.0
    rows = []
    n_eligible_per_month = []

    for i, m in enumerate(months):
        regime = classify_regime_tight(spy.loc[m].to_dict() if m in spy.index else {})
        do_reb = (i == 0) or (held_for >= hold) or (cash != (regime == "crash"))
        ret_m = 0.0
        if not cash and cur_picks:
            mr_pos = mr.index.searchsorted(m)
            if mr_pos + 1 < len(mr.index):
                next_d = mr.index[mr_pos + 1]
                pick_rets = [0.0 if pd.isna(mr.at[next_d, tk]) else float(mr.at[next_d, tk])
                              for tk in cur_picks if tk in mr.columns]
                if len(pick_rets) == len(cur_weights):
                    ret_m = float((np.array(pick_rets) * cur_weights).sum())
                    equity *= (1 + ret_m)

        if do_reb:
            equity *= (1 - cf)
            if regime == "crash":
                cur_picks = []; cur_weights = np.array([]); cash = True
            else:
                sub_ml = ml_by_asof.get(m)
                sub_chr = chr_by_asof.get(m)
                if sub_ml is None:
                    cur_picks = []; cur_weights = np.array([])
                else:
                    ml_tickers_at = set(sub_ml["ticker"])
                    universe = universe_fn(m, members_g, ml_tickers_at)
                    if not universe:
                        cur_picks = []; cur_weights = np.array([])
                    else:
                        sub = sub_ml[sub_ml["ticker"].isin(universe)].copy()
                        sub = sub[~sub["ticker"].isin(EXCLUDE)]
                        sub = sub.dropna(subset=["ml_score"])
                        if chr_q > 0 and sub_chr is not None and not sub_chr.empty:
                            sub = sub.merge(sub_chr[["ticker", "chronos_p70_3m"]],
                                            on="ticker", how="left")
                            sub = sub.dropna(subset=["chronos_p70_3m"])
                            sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)
                            sub = sub[sub["chr_p70_rk"] >= chr_q]
                        n_eligible_per_month.append(len(sub))
                        sub = sub.sort_values("ml_score", ascending=False)
                        top = sub.head(k)
                        if len(top) < k:
                            cur_picks = []; cur_weights = np.array([])
                        else:
                            cur_picks = top["ticker"].tolist()
                            cur_weights = calc_invvol_weights(cur_picks, mr, m, cap=cap)
                cash = False
            held_for = 0
        else:
            held_for += 1
        rows.append({"date": m, "regime": regime, "equity": equity, "ret_m": ret_m,
                     "cash": cash})

    eq = pd.DataFrame(rows)
    n = len(eq)
    cagr_full = (eq["equity"].iloc[-1]) ** (12 / n) - 1
    r = eq["ret_m"].astype(float)
    sharpe = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
    peak = eq["equity"].cummax()
    mdd = float(((eq["equity"] - peak) / peak).min())

    spy_ret = mr["SPY"].dropna()
    spy_full = (1 + spy_ret.loc[eq["date"].iloc[0]:eq["date"].iloc[-1]]).prod() ** (12 / n) - 1
    next_months = pd.DatetimeIndex(eq["date"]) + pd.offsets.MonthEnd(1)
    spy_aligned = [float(spy_ret.loc[nxt]) if nxt in spy_ret.index else 0.0 for nxt in next_months]
    spy_df = pd.DataFrame({"date": eq["date"], "spy_ret_m": spy_aligned})

    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        rr = e["ret_m"].astype(float)
        ec = (1 + rr).cumprod()
        cagr_v = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        s = spy_df[(spy_df["date"] >= lo) & (spy_df["date"] <= hi)]
        sr = s["spy_ret_m"].astype(float); sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        wf_rows.append({"cagr": cagr_v, "spy_cagr": scgr})
    wf = pd.DataFrame(wf_rows)

    return dict(
        universe=universe_name,
        n_pool_mean=int(np.mean(n_eligible_per_month)) if n_eligible_per_month else 0,
        n_pool_max=int(np.max(n_eligible_per_month)) if n_eligible_per_month else 0,
        cagr_full=float(cagr_full),
        spy_cagr_full=float(spy_full),
        edge_full_pp=float((cagr_full - spy_full) * 100),
        sharpe=float(sharpe),
        max_dd=float(mdd),
        wf_mean_cagr=float(wf["cagr"].mean()),
        wf_median_cagr=float(wf["cagr"].median()),
        wf_min_cagr=float(wf["cagr"].min()),
        wf_max_cagr=float(wf["cagr"].max()),
        wf_mean_edge_pp=float((wf["cagr"] - wf["spy_cagr"]).mean() * 100),
        wf_n_positive=int((wf["cagr"] > 0).sum()),
        wf_n_beats_spy=int((wf["cagr"] > wf["spy_cagr"]).sum()),
        wf_n_splits=int(len(wf)),
    )


def main():
    t0 = time.time()
    print("Loading augmented data ...")
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    ml = pd.read_parquet(AUG / "ml_preds.parquet")
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml["ml_score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2

    # Use the BROADER chronos preds (covers full augmented universe).
    # Fall back to PIT-only if broader not yet generated.
    chr_broad_path = AUG / "ml_preds_chronos_broader.parquet"
    if chr_broad_path.exists():
        chr_ = pd.read_parquet(chr_broad_path)[["asof", "ticker", "chronos_p70_3m"]]
        print(f"  using broader Chronos: {chr_['ticker'].nunique()} tickers")
    else:
        chr_ = pd.read_parquet(AUG / "ml_preds_chronos.parquet")[["asof", "ticker", "chronos_p70_3m"]]
        print(f"  using PIT-only Chronos: {chr_['ticker'].nunique()} tickers (broader not yet generated)")
    chr_["asof"] = pd.to_datetime(chr_["asof"])

    spy = load_spy_features()
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").fillna(0.0)
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()

    universes = [
        ("PIT_SP500_augmented", universe_pit_sp500),
        ("broader_augmented", universe_broader),
        ("non_sp500_augmented", universe_non_pit),
        ("random_500_seed1", make_random_subset_fn(seed=1)),
        ("random_500_seed2", make_random_subset_fn(seed=2)),
        ("random_500_seed3", make_random_subset_fn(seed=3)),
    ]

    results = []
    for name, fn in universes:
        elapsed = time.time() - t0
        print(f"\n[{elapsed:.0f}s] Running K=2 v5 on universe: {name}")
        r = run_v5_universe(fn, panel, ml, chr_, spy, mr, members_g, name)
        print(f"  pool: mean {r['n_pool_mean']}, max {r['n_pool_max']}")
        print(f"  Full CAGR {r['cagr_full']*100:>6.2f}%   "
              f"WF mean {r['wf_mean_cagr']*100:>6.2f}%   "
              f"Sharpe {r['sharpe']:>5.2f}   "
              f"MaxDD {r['max_dd']*100:>6.1f}%   "
              f"beats {r['wf_n_beats_spy']}/{r['wf_n_splits']}")
        results.append(r)

    df = pd.DataFrame(results)
    out_path = AUG / "v5_k2_generalize.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print("\n=== K=2 v5 generalization summary ===")
    cols = ["universe", "n_pool_mean", "cagr_full", "wf_mean_cagr",
            "wf_mean_edge_pp", "sharpe", "max_dd", "wf_n_beats_spy"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
