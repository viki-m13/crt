"""Run the deployed v3 strategy on the ETF universes.

Pipeline (mirrors the production v3 PIT-S&P-500 setup):

  1. Load the price panel for the chosen universe (incl. SPY).
  2. Compute the same monthly features used by v3 (compute_features +
     compute_extras from experiments/monthly_dca/*).
  3. Build the (asof, ticker) cross-section with multi-horizon forward returns.
  4. Walk-forward fit a HistGradientBoostingRegressor on cross-sectional
     rank targets at horizons (1m, 3m, 6m), retraining each January with a
     7-month embargo. Score = mean of (pred_3m, pred_6m).  This reproduces
     the deployed v3 ML scorer ("ml_3plus6").
  5. Apply the v3 winner config: K=3 EW, tight regime gate, hold=6m,
     cost=10bps. The regime gate uses *SPY's* features at each month-end
     (same as production).
  6. Evaluate full-window CAGR/Sharpe/MaxDD and 10-split walk-forward
     summary, comparing to SPY buy-and-hold over the same window.

Outputs (per universe):
  results/<universe>_equity.csv      — month-end equity curve
  results/<universe>_picks.csv       — picks per rebalance
  results/<universe>_yearly.csv      — year-by-year vs SPY
  results/<universe>_walkforward.csv — 10-split WF table
  results/<universe>_summary.json    — top-line metrics

Run:
  python3 experiments/etf_v3_test_O0MtP/run_v3.py {broad|levered|combined}
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / "cache"
RESULTS = HERE / "results"
for d in (CACHE, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from experiments.monthly_dca.backtester import compute_features, month_end_dates  # noqa: E402
from experiments.monthly_dca.extra_features import compute_extras  # noqa: E402

# v3 regime gate (tight) — exactly the deployed gate
from experiments.monthly_dca.v2.sp500_pit_extended_sweep import (  # noqa: E402
    classify_regime_tight,
)

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402


# ---------------------------------------------------------------------------
# v3 winner spec — frozen from production (`build_webapp_v3_pit.py`)
# ---------------------------------------------------------------------------
WINNER = {
    "scorer": "ml_3plus6",
    "k_normal": 3,
    "k_recovery": 3,
    "k_bull": 3,
    "weighting": "ew",
    "regime_gate": "tight",
    "hold_months": 6,
    "cost_bps": 10.0,
}


# Walk-forward splits — mirror the production v3 PIT splits where data exists
WF_SPLITS = [
    ("A1", "2011-01-01", "2018-12-31"),
    ("A2", "2015-01-01", "2021-12-31"),
    ("A3", "2018-01-01", "2024-12-31"),
    ("R1_GFC", "2008-01-01", "2010-12-31"),
    ("R2", "2011-01-01", "2013-12-31"),
    ("R3", "2014-01-01", "2016-12-31"),
    ("R4", "2017-01-01", "2019-12-31"),
    ("R5_COVID", "2020-01-01", "2022-12-31"),
    ("R6_AI", "2023-01-01", "2024-12-31"),
    ("STRICT", "2021-01-01", "2024-12-31"),
]


# ---------------------------------------------------------------------------
def load_panel(universe: str) -> pd.DataFrame:
    p = DATA / f"prices_{universe}.parquet"
    panel = pd.read_parquet(p)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    return panel


# ---------------------------------------------------------------------------
def build_features(universe: str, panel: pd.DataFrame, start: str = "2003-01-01") -> pd.DataFrame:
    """Return cross-section panel: (asof, ticker) -> features + fwd returns."""
    out = CACHE / f"feat_{universe}.parquet"
    if out.exists():
        print(f"  [{universe}] reusing cached features {out.name}")
        return pd.read_parquet(out)

    months = month_end_dates(panel.index)
    months = months[(months >= pd.Timestamp(start)) & (months <= pd.Timestamp("2099-01-01"))]
    print(f"  [{universe}] computing features for {len(months)} months...")

    # Build monthly clean prices for fwd returns
    monthly = panel.resample("ME").last()

    rows = []
    t0 = time.time()
    for k, m in enumerate(months):
        try:
            base = compute_features(panel, m, min_history=252).df()
        except Exception:
            continue
        try:
            extras = compute_extras(panel, m)
        except Exception:
            extras = pd.DataFrame()
        # Merge extras
        if not extras.empty:
            base = base.join(extras, how="left", rsuffix="_x")
            base = base.loc[:, ~base.columns.str.endswith("_x")]
        base["asof"] = m
        rows.append(base)
        if (k + 1) % 24 == 0 or k == len(months) - 1:
            print(f"    [{k+1}/{len(months)}] {m.date()}  features cols={base.shape[1]}", flush=True)
    big = pd.concat(rows, axis=0, ignore_index=False)
    big.index.name = "ticker"

    # Forward returns at h=1,3,6,12 months
    print(f"  [{universe}] adding forward returns...", flush=True)
    big = big.reset_index()
    fwd = {}
    for h in (1, 3, 6, 12):
        col = f"fwd_{h}m_ret"
        fwd_vals = []
        for asof, sub in big.groupby("asof"):
            asof = pd.Timestamp(asof)
            # find month-end in monthly index nearest asof
            pos = monthly.index.searchsorted(asof)
            cands = []
            for j in (pos - 1, pos):
                if 0 <= j < len(monthly.index):
                    cands.append((j, abs((monthly.index[j] - asof).days)))
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7:
                fwd_vals.append(pd.Series(np.nan, index=sub.index))
                continue
            pos1 = cands[0][0]
            if pos1 + h >= len(monthly.index):
                fwd_vals.append(pd.Series(np.nan, index=sub.index))
                continue
            p1 = monthly.iloc[pos1]
            ph = monthly.iloc[pos1 + h]
            cap = 2.0 * h
            r = (ph / p1 - 1).clip(lower=-1.0, upper=cap)
            # Delisting -> -1
            end_pos = min(pos1 + h + 6, len(monthly.index) - 1)
            future_window = monthly.iloc[pos1 + h: end_pos + 1]
            any_future = future_window.notna().any()
            p1_valid = p1.notna()
            ph_nan = ph.isna()
            delist = p1_valid & ph_nan & ~any_future.reindex(monthly.columns, fill_value=False)
            r[delist] = -1.0
            sub_fwd = r.reindex(sub["ticker"]).values
            fwd_vals.append(pd.Series(sub_fwd, index=sub.index))
        big[col] = pd.concat(fwd_vals).sort_index()

    big = big.set_index(["asof", "ticker"]).sort_index()
    big.to_parquet(out, compression="zstd")
    print(f"  [{universe}] wrote {out.name} shape={big.shape}", flush=True)
    return big


# ---------------------------------------------------------------------------
# Walk-forward ML training (matches v2/ml_strategy.fit_walkforward, with
# a smaller training-rows threshold so smaller universes still train).
# ---------------------------------------------------------------------------
def fit_walkforward(
    big: pd.DataFrame,
    target_horizons=(1, 3, 6),
    train_start: pd.Timestamp = pd.Timestamp("2003-01-01"),
    embargo_months: int = 7,
    min_train_rows: int = 1000,
) -> pd.DataFrame:
    big = big.reset_index().copy()
    big["asof"] = pd.to_datetime(big["asof"])

    excluded = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}
    big = big[~big["ticker"].isin(excluded)].copy()

    target_cols = [f"rank_target_{h}m" for h in target_horizons]
    fwd_cols = [f"fwd_{h}m_ret" for h in target_horizons]

    feat_cols_raw = [c for c in big.columns
                     if c not in ("asof", "ticker") and not c.startswith("fwd_")
                     and not c.startswith("rank_target_")]

    # Compute rank targets per month
    for h, tc, fc in zip(target_horizons, target_cols, fwd_cols):
        big[tc] = big.groupby("asof")[fc].rank(pct=True)

    # Cross-sectional rank-z for features
    print(f"  cross-sectional ranking {len(feat_cols_raw)} features...", flush=True)
    for c in feat_cols_raw:
        big[c + "_xs"] = big.groupby("asof")[c].transform(lambda x: (x.rank(pct=True) - 0.5) * 2)
    xs_features = [c + "_xs" for c in feat_cols_raw]

    months = sorted(big["asof"].unique())
    last_trained_year = None
    models = {}
    all_preds = []

    model_kwargs = dict(
        max_iter=200, learning_rate=0.05, max_depth=5,
        min_samples_leaf=50, l2_regularization=1.0,
    )

    for tm_raw in months:
        tm = pd.Timestamp(tm_raw)
        if tm < train_start:
            continue
        do_retrain = (not models) or (tm.month == 1 and (last_trained_year is None or last_trained_year < tm.year))
        if do_retrain:
            cutoff = tm - pd.DateOffset(months=embargo_months)
            train = big[big["asof"] < cutoff]
            if len(train) >= min_train_rows:
                trained = {}
                for h, tc in zip(target_horizons, target_cols):
                    m = train[tc].notna()
                    if m.sum() < int(min_train_rows * 0.5):
                        continue
                    Xt = train.loc[m, xs_features].values
                    yt = train.loc[m, tc].values
                    mdl = HistGradientBoostingRegressor(**model_kwargs)
                    mdl.fit(Xt, yt)
                    trained[h] = mdl
                if trained:
                    models = trained
                    last_trained_year = tm.year
                    print(f"    retrain @ {tm.date()}  train_rows={len(train):,}", flush=True)

        if not models:
            continue
        test = big[big["asof"] == tm_raw]
        if len(test) == 0:
            continue
        Xtest = test[xs_features].values
        per_h = {h: models[h].predict(Xtest) for h in models}
        pred_avg = np.mean(list(per_h.values()), axis=0)
        rows = test[["asof", "ticker"]].copy()
        rows["pred"] = pred_avg
        for h, p in per_h.items():
            rows[f"pred_{h}m"] = p
        all_preds.append(rows)

    if not all_preds:
        return pd.DataFrame(columns=["asof", "ticker", "pred"])
    return pd.concat(all_preds, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# v3 strategy simulator — replicates simulate_variant from
# experiments/monthly_dca/v2/sp500_pit_extended_sweep.py
# ---------------------------------------------------------------------------
def get_spy_features_per_month(big: pd.DataFrame) -> pd.DataFrame:
    """Slice SPY features per month from the cross-section panel."""
    if isinstance(big.index, pd.MultiIndex):
        flat = big.reset_index()
    else:
        flat = big.copy()
    spy = flat[flat["ticker"] == "SPY"].copy()
    if spy.empty:
        raise RuntimeError("SPY missing from feature panel — needed for regime gate")
    keep = {
        "asof": "asof",
        "d_sma200": "spy_dsma200",
        "rsi_14": "spy_rsi14",
        "mom_12_1": "spy_mom_12_1",
        "mom_6_1": "spy_mom_6_1",
        "ret_21d": "spy_ret_21d",
        "max_below_200_streak": "spy_below_200_streak",
        "dd_from_52wh": "spy_dd_from_52wh",
    }
    out = spy[[c for c in keep if c in spy.columns]].rename(columns=keep)
    out = out.set_index("asof").sort_index()
    return out


def simulate(
    panel_prices: pd.DataFrame,
    preds: pd.DataFrame,
    spy_features: pd.DataFrame,
    cost_bps: float = 10.0,
    k: int = 3,
    hold_months: int = 6,
) -> pd.DataFrame:
    """Compound the v3 ml_3plus6 K=3 EW tight h=6 strategy."""
    monthly = panel_prices.resample("ME").last()
    mret = monthly.pct_change().clip(lower=-1.0, upper=2.0)
    mret_idx = mret.index

    excluded = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}
    p = preds[~preds["ticker"].isin(excluded)].dropna(subset=["pred"]).copy()
    by_asof = {pd.Timestamp(d): g.sort_values("pred", ascending=False) for d, g in p.groupby("asof")}
    months = sorted(by_asof.keys())

    cf = cost_bps / 10000.0
    equity = 1.0
    cur_picks: list[str] = []
    cur_w = np.array([])
    held_for = 0
    cash = False
    rows = []

    for i, m in enumerate(months):
        do_reb = (i == 0) or (held_for >= hold_months) or cash
        if do_reb:
            if m in spy_features.index:
                s = spy_features.loc[m].to_dict()
            else:
                s = {}
            regime = classify_regime_tight(s)
            sub = by_asof[m]
            if regime == "crash":
                cur_picks, cur_w, cash = [], np.array([]), True
                held_for = 0
            elif len(sub) < k:
                cur_picks, cur_w, cash = [], np.array([]), True
                held_for = 0
            else:
                top = sub.head(k)
                cur_picks = top["ticker"].tolist()
                cur_w = np.ones(k) / k
                cash = False
                held_for = 0
        else:
            regime = "hold"

        # Compute next-month return for current picks
        pos = mret_idx.searchsorted(m)
        cands = [(j, abs((mret_idx[j] - m).days)) for j in (pos - 1, pos) if 0 <= j < len(mret_idx)]
        cands.sort(key=lambda x: x[1])
        if cash or not cur_picks or not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mret_idx):
            ret_m = 0.0
        else:
            next_d = mret_idx[cands[0][0] + 1]
            picks_r = []
            for tk in cur_picks:
                if tk in mret.columns:
                    rv = mret.at[next_d, tk]
                    picks_r.append(-1.0 if pd.isna(rv) else float(rv))
                else:
                    picks_r.append(-1.0)
            ret_m = float((np.asarray(picks_r) * cur_w).sum())

        if not cash and cur_picks:
            equity *= (1 + ret_m) * (1 - cf if do_reb else 1.0)
        held_for += 1
        rows.append({
            "date": m,
            "equity": equity,
            "ret_m": ret_m,
            "regime": "cash" if cash else regime,
            "n_picks": len(cur_picks),
            "picks": ",".join(cur_picks),
            "rebalance": int(do_reb),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _cagr(eq: pd.Series) -> float:
    if len(eq) < 2:
        return 0.0
    return float(eq.iloc[-1] ** (12.0 / len(eq)) - 1.0)


def _sharpe_m(r: pd.Series) -> float:
    rs = r.dropna()
    if len(rs) < 2 or rs.std() == 0:
        return 0.0
    return float((rs.mean() / rs.std()) * np.sqrt(12))


def _max_dd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    if len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def evaluate_run(eq_df: pd.DataFrame, panel_prices: pd.DataFrame) -> dict:
    """Return summary stats vs SPY buy-and-hold."""
    monthly = panel_prices.resample("ME").last()
    spy_m = monthly["SPY"].pct_change()
    eq_df = eq_df.copy()
    eq_df["date"] = pd.to_datetime(eq_df["date"])

    # Strategy series
    r = eq_df["ret_m"].astype(float)
    strat_cagr = _cagr((1 + r).cumprod()) if len(r) else 0.0
    strat_sh = _sharpe_m(r)
    strat_mdd = _max_dd(r)

    # SPY series aligned to next-month after each rebalance month
    aligned = []
    for d in eq_df["date"]:
        pos = monthly.index.searchsorted(pd.Timestamp(d))
        cands = [(j, abs((monthly.index[j] - d).days)) for j in (pos - 1, pos) if 0 <= j < len(monthly.index)]
        cands.sort(key=lambda x: x[1])
        if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(monthly.index):
            aligned.append(0.0)
            continue
        nxt = monthly.index[cands[0][0] + 1]
        v = spy_m.get(nxt, 0.0)
        aligned.append(0.0 if pd.isna(v) else float(v))
    spy_s = pd.Series(aligned, index=eq_df["date"].values)
    spy_cagr = _cagr((1 + spy_s).cumprod())
    spy_sh = _sharpe_m(spy_s)
    spy_mdd = _max_dd(spy_s)

    # Walk-forward splits
    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
        m = (eq_df["date"] >= lo_ts) & (eq_df["date"] <= hi_ts)
        if m.sum() < 12:
            continue
        rs = eq_df.loc[m, "ret_m"].astype(float)
        cv = _cagr((1 + rs).cumprod())
        sm = (eq_df["date"] >= lo_ts) & (eq_df["date"] <= hi_ts)
        spy_window = spy_s[sm.values]
        scgr = _cagr((1 + spy_window).cumprod())
        wf_rows.append({
            "split": split, "from": lo, "to": hi, "n_m": int(m.sum()),
            "cagr": cv, "spy_cagr": scgr, "edge_pp": (cv - scgr) * 100,
            "sharpe": _sharpe_m(rs), "max_dd": _max_dd(rs),
            "n_cash": int((eq_df.loc[m, "regime"] == "cash").sum()),
        })
    wf = pd.DataFrame(wf_rows)
    return {
        "n_months": int(len(r)),
        "first_month": str(eq_df["date"].min().date()) if len(eq_df) else None,
        "last_month": str(eq_df["date"].max().date()) if len(eq_df) else None,
        "cagr_full": strat_cagr, "spy_cagr_full": spy_cagr,
        "edge_full_pp": (strat_cagr - spy_cagr) * 100,
        "sharpe": strat_sh, "spy_sharpe": spy_sh,
        "max_dd": strat_mdd, "spy_max_dd": spy_mdd,
        "n_cash": int((eq_df["regime"] == "cash").sum()),
        "n_rebalances": int(eq_df["rebalance"].sum()),
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else None,
        "wf_median_cagr": float(wf["cagr"].median()) if len(wf) else None,
        "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else None,
        "wf_max_cagr": float(wf["cagr"].max()) if len(wf) else None,
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else None,
        "wf_n_pos": int((wf["cagr"] > 0).sum()) if len(wf) else None,
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else None,
        "wf_n_splits": int(len(wf)),
        "wf_table": wf.to_dict("records"),
    }


# ---------------------------------------------------------------------------
def yearly_table(eq_df: pd.DataFrame, panel_prices: pd.DataFrame) -> pd.DataFrame:
    monthly = panel_prices.resample("ME").last()
    spy_m = monthly["SPY"].pct_change()
    eq = eq_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq["year"] = eq["date"].dt.year
    yr = eq.groupby("year")["ret_m"].apply(lambda x: ((1 + x).prod() - 1)).rename("strat_year_ret")
    spy_aligned = []
    for d in eq["date"]:
        pos = monthly.index.searchsorted(pd.Timestamp(d))
        cands = [(j, abs((monthly.index[j] - d).days)) for j in (pos - 1, pos) if 0 <= j < len(monthly.index)]
        cands.sort(key=lambda x: x[1])
        if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(monthly.index):
            spy_aligned.append(0.0)
            continue
        nxt = monthly.index[cands[0][0] + 1]
        v = spy_m.get(nxt, 0.0)
        spy_aligned.append(0.0 if pd.isna(v) else float(v))
    spy_s = pd.Series(spy_aligned, index=eq["date"].values)
    eq2 = eq.copy()
    eq2["spy_ret"] = spy_s.values
    syr = eq2.groupby("year")["spy_ret"].apply(lambda x: ((1 + x).prod() - 1)).rename("spy_year_ret")
    out = yr.to_frame().join(syr.to_frame(), how="left")
    out["edge_pp"] = (out["strat_year_ret"] - out["spy_year_ret"]) * 100
    return out.reset_index()


# ---------------------------------------------------------------------------
def run_universe(name: str) -> dict:
    print(f"\n=== run_universe: {name} ===")
    panel = load_panel(name)
    print(f"  panel shape={panel.shape}  range={panel.index.min().date()} → {panel.index.max().date()}")

    # Build features (cached)
    big = build_features(name, panel)

    # Walk-forward ML
    print(f"  fitting walk-forward ML on {name}...")
    preds = fit_walkforward(big, train_start=pd.Timestamp("2005-01-01"))
    preds_path = CACHE / f"preds_{name}.parquet"
    preds.to_parquet(preds_path, compression="zstd")
    print(f"  preds shape={preds.shape}  range={preds['asof'].min()} → {preds['asof'].max()}")

    # SPY features for regime gate
    spy_feats = get_spy_features_per_month(big)

    # Simulate v3 strategy
    eq = simulate(panel, preds, spy_feats, cost_bps=WINNER["cost_bps"],
                  k=WINNER["k_normal"], hold_months=WINNER["hold_months"])
    eq.to_csv(RESULTS / f"{name}_equity.csv", index=False)

    # Picks per rebalance
    picks = eq[eq["rebalance"] == 1][["date", "regime", "n_picks", "picks"]].copy()
    picks.to_csv(RESULTS / f"{name}_picks.csv", index=False)

    # Yearly + WF
    yr = yearly_table(eq, panel)
    yr.to_csv(RESULTS / f"{name}_yearly.csv", index=False)

    summary = evaluate_run(eq, panel)
    summary["universe"] = name
    summary["n_tickers_universe"] = int(panel.shape[1])
    summary["winner_spec"] = WINNER

    with open(RESULTS / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    pd.DataFrame(summary["wf_table"]).to_csv(RESULTS / f"{name}_walkforward.csv", index=False)

    print(f"  -> CAGR={summary['cagr_full']*100:.2f}%  SPY={summary['spy_cagr_full']*100:.2f}%  "
          f"edge={summary['edge_full_pp']:+.2f}pp  MDD={summary['max_dd']*100:.2f}%  "
          f"Sharpe={summary['sharpe']:.2f}")
    if summary["wf_mean_cagr"] is not None:
        print(f"  WF mean CAGR={summary['wf_mean_cagr']*100:.2f}%  +{summary['wf_mean_edge_pp']:+.2f}pp  "
              f"({summary['wf_n_beats_spy']}/{summary['wf_n_splits']} beat SPY)")
    return summary


def main():
    args = sys.argv[1:]
    targets = args if args else ["broad", "levered", "combined"]
    out = {}
    for u in targets:
        out[u] = run_universe(u)
    # Write top-level comparison
    cmp_rows = []
    for u, s in out.items():
        cmp_rows.append({
            "universe": u,
            "n_tickers": s["n_tickers_universe"],
            "n_months": s["n_months"],
            "first_month": s["first_month"],
            "last_month": s["last_month"],
            "cagr_full_pct": s["cagr_full"] * 100,
            "spy_cagr_pct": s["spy_cagr_full"] * 100,
            "edge_full_pp": s["edge_full_pp"],
            "sharpe": s["sharpe"],
            "max_dd_pct": s["max_dd"] * 100,
            "n_cash_months": s["n_cash"],
            "n_rebalances": s["n_rebalances"],
            "wf_mean_cagr_pct": (s["wf_mean_cagr"] or 0) * 100,
            "wf_mean_edge_pp": s["wf_mean_edge_pp"],
            "wf_n_beats_spy": s["wf_n_beats_spy"],
            "wf_n_splits": s["wf_n_splits"],
        })
    pd.DataFrame(cmp_rows).to_csv(RESULTS / "comparison.csv", index=False)
    print("\n=== comparison.csv ===")
    print(pd.DataFrame(cmp_rows).to_string(index=False))


if __name__ == "__main__":
    main()
