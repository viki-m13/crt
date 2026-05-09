"""Point-in-time S&P 500 backtest with the ML model RE-TRAINED on the
S&P 500 PIT universe only.

This is the academically-cleanest evaluation of the v2 strategy on a true PIT
S&P 500 universe:

  1. At each month-end T:
       - Determine the set of S&P 500 constituents on T
       - From cache/features/{T}.parquet, take only their feature rows
       - Cross-sectionally rank-transform the 67 features within the S&P 500
         universe at T (regime-free, scale-free)
       - Compute fwd 1m, 3m, 6m returns from the price panel
       - Cross-sectionally rank-transform the forward returns

  2. Walk-forward fit a HistGradientBoostingRegressor for each forward horizon
     (annual retrain, 7-month embargo). Predictions are averaged across horizons.

  3. Apply the same v2 'tight' regime gate (cash on crash; top-15 normal /
     top-7 bull / top-7 recovery), equal-weight within picks, 10bp/month cost.

  4. Walk-forward 10-split TEST CAGR / Sharpe / DD.
  5. Year-by-year, drawdowns, regime distribution.

Outputs to cache/v2/sp500_pit/sp500_pit_retrain_*.csv,.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
PIT.mkdir(parents=True, exist_ok=True)
FEATURES_DIR = CACHE / "features"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}


# ---------------------------------------------------------------------------
def classify_regime(s: dict) -> str:
    """v2 'tight' regime classifier — same as production."""
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


# ---------------------------------------------------------------------------
def load_spy_features() -> pd.DataFrame:
    rows = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


# ---------------------------------------------------------------------------
def build_pit_panel(members: pd.DataFrame, monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """Build the cross-section panel restricted to PIT S&P 500 members.

    Returns a DataFrame indexed by (asof, ticker) with cross-sectional
    rank-transformed features (suffix _xs), forward returns (1m, 3m, 6m),
    and forward-rank targets (rank_target_{h}m).
    """
    members = members.copy()
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set)

    panel_chunks = []
    feature_cols: list[str] | None = None
    feature_files = {pd.Timestamp(p.stem): p for p in FEATURES_DIR.glob("*.parquet")}

    asofs = sorted(set(members_g.index) & set(feature_files.keys()))
    print(f"  Building panel for {len(asofs)} months ({asofs[0].date()}..{asofs[-1].date()})")

    for d in asofs:
        feat = pd.read_parquet(feature_files[d])
        if feature_cols is None:
            feature_cols = list(feat.columns)
        sp = members_g[d]
        feat = feat[feat.index.isin(sp)]
        feat = feat[~feat.index.isin(EXCLUDE)]
        if len(feat) < 50:
            continue
        # Cross-sectionally rank within the S&P 500 universe at this asof
        for c in feature_cols:
            r = feat[c].rank(pct=True)
            feat[c + "_xs"] = (r - 0.5) * 2

        feat = feat.reset_index().rename(columns={"index": "ticker"})
        feat["asof"] = d
        panel_chunks.append(feat)
    panel = pd.concat(panel_chunks, axis=0, ignore_index=True)
    panel = panel[["asof", "ticker"] + [c + "_xs" for c in feature_cols] + feature_cols]
    print(f"  Panel built: {panel.shape}")
    print(f"  unique tickers: {panel['ticker'].nunique()}")

    # Compute forward returns from monthly_returns
    mr_dates = monthly_returns.index.sort_values()
    asof_idx = pd.DatetimeIndex(asofs)

    # Map asof to position in monthly returns; fwd_h is product of returns h months ahead
    print("  Computing forward returns (1m, 3m, 6m)...")

    # Precompute log-cumulative returns to make h-month forwards a difference
    log_mr = np.log1p(monthly_returns.fillna(0)).cumsum()
    # log_mr.iloc[t+h] - log_mr.iloc[t] = log of (1+r_{t+1})*...*(1+r_{t+h})

    asof_to_pos = {}
    for d in asof_idx:
        # find position in mr_dates closest to d
        pos = mr_dates.searchsorted(d)
        cand = []
        for j in (pos - 1, pos):
            if 0 <= j < len(mr_dates):
                cand.append((j, abs((mr_dates[j] - d).days)))
        cand.sort(key=lambda x: x[1])
        if cand and cand[0][1] <= 7:
            asof_to_pos[d] = cand[0][0]

    fwds = {1: [], 3: [], 6: []}
    for h in (1, 3, 6):
        for _, row in panel[["asof", "ticker"]].iterrows():
            d = row["asof"]
            tk = row["ticker"]
            pos = asof_to_pos.get(d, None)
            if pos is None or pos + h >= len(mr_dates) or tk not in monthly_returns.columns:
                fwds[h].append(np.nan)
                continue
            d0 = mr_dates[pos]
            dh = mr_dates[pos + h]
            try:
                lr0 = log_mr.at[d0, tk]
                lrh = log_mr.at[dh, tk]
            except KeyError:
                fwds[h].append(np.nan)
                continue
            if pd.isna(lr0) or pd.isna(lrh):
                fwds[h].append(np.nan)
                continue
            fwds[h].append(np.expm1(lrh - lr0))

    for h in (1, 3, 6):
        panel[f"fwd_{h}m_ret"] = fwds[h]

    # Compute rank targets per (asof, h) — rank within S&P 500 members at month T
    for h in (1, 3, 6):
        panel[f"rank_target_{h}m"] = panel.groupby("asof")[f"fwd_{h}m_ret"].rank(pct=True)

    print(f"  Panel ready: {panel.shape}")
    return panel


# ---------------------------------------------------------------------------
def fit_walkforward(
    panel: pd.DataFrame,
    embargo_months: int = 7,
    train_start: pd.Timestamp = pd.Timestamp("2003-01-01"),
    train_end: pd.Timestamp = pd.Timestamp("2025-12-31"),
) -> pd.DataFrame:
    """Walk-forward fit a multi-horizon GBM and return predictions per (asof, ticker).

    Mirrors the production v2 walk-forward fitter, but with the panel restricted
    to S&P 500 PIT members.
    """
    target_horizons = (1, 3, 6)
    target_cols = [f"rank_target_{h}m" for h in target_horizons]
    xs_cols = [c for c in panel.columns if c.endswith("_xs")]

    months = sorted(panel["asof"].unique())
    last_trained_year = None
    models: dict[int, HistGradientBoostingRegressor] = {}
    all_preds = []

    model_kwargs = dict(
        max_iter=300, learning_rate=0.04, max_depth=6,
        min_samples_leaf=200, l2_regularization=1.0,
    )

    for tm in months:
        tm = pd.Timestamp(tm)
        if tm < train_start or tm > train_end:
            continue
        do_retrain = (not models) or (tm.month == 1 and last_trained_year != tm.year)
        if do_retrain:
            cutoff = tm - pd.DateOffset(months=embargo_months)
            train = panel[panel["asof"] < cutoff]
            if len(train) < 5000:
                continue
            for h, tc in zip(target_horizons, target_cols):
                m = train[tc].notna()
                if m.sum() < 1000:
                    continue
                Xt = train.loc[m, xs_cols].values
                yt = train.loc[m, tc].values
                mdl = HistGradientBoostingRegressor(**model_kwargs)
                mdl.fit(Xt, yt)
                models[h] = mdl
            last_trained_year = tm.year
            print(f"    Retrained {tm.date()} (train rows={len(train)})")
        if not models:
            continue
        test = panel[panel["asof"] == tm]
        if len(test) == 0:
            continue
        Xtest = test[xs_cols].values
        per_horizon = {h: models[h].predict(Xtest) for h in target_horizons if h in models}
        if not per_horizon:
            continue
        pred_avg = np.mean(list(per_horizon.values()), axis=0)
        out = test[["asof", "ticker", "fwd_1m_ret"]].copy()
        out["pred"] = pred_avg
        for h, p in per_horizon.items():
            out[f"pred_{h}m"] = p
        all_preds.append(out)

    return pd.concat(all_preds, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
def build_outputs(preds: pd.DataFrame, spy: pd.DataFrame,
                  k_normal=15, k_recovery=7, k_bull=7) -> list:
    months = sorted(preds["asof"].unique())
    outs = []
    for m in months:
        m = pd.Timestamp(m)
        if m not in spy.index:
            continue
        regime = classify_regime(spy.loc[m].to_dict())
        sub = preds[preds["asof"] == m].sort_values("pred", ascending=False)
        n_eligible = len(sub)
        if regime == "crash":
            outs.append({"asof": m, "picks": [], "weights": np.array([]),
                         "cash": True, "regime": regime, "n_eligible": n_eligible})
            continue
        k = {"recovery": k_recovery, "bull": k_bull, "normal": k_normal}[regime]
        top = sub.head(k)
        if len(top) < k:
            outs.append({"asof": m, "picks": [], "weights": np.array([]),
                         "cash": True, "regime": regime, "n_eligible": n_eligible})
            continue
        weights = np.ones(k) / k
        outs.append({"asof": m, "picks": top["ticker"].tolist(), "weights": weights,
                     "cash": False, "regime": regime, "n_eligible": n_eligible})
    return outs


def _nearest_pos(idx, target, tol_days=7):
    pos = idx.searchsorted(target)
    cands = []
    for j in (pos - 1, pos):
        if 0 <= j < len(idx):
            cands.append((j, abs((idx[j] - target).days)))
    cands.sort(key=lambda x: x[1])
    return cands[0][0] if cands and cands[0][1] <= tol_days else None


def simulate(outs, monthly_returns, cost_bps: float = 10.0, starting_cash: float = 1.0):
    equity = starting_cash
    cost_factor = cost_bps / 10000.0
    rows = []
    for o in outs:
        if o["cash"] or len(o["picks"]) == 0:
            ret_m = 0.0
        else:
            pos1 = _nearest_pos(monthly_returns.index, o["asof"])
            if pos1 is None or pos1 + 1 >= len(monthly_returns.index):
                ret_m = 0.0
            else:
                next_d = monthly_returns.index[pos1 + 1]
                pick_rets = []
                for tk in o["picks"]:
                    if tk in monthly_returns.columns:
                        r = monthly_returns.at[next_d, tk]
                        pick_rets.append(-1.0 if pd.isna(r) else float(r))
                    else:
                        pick_rets.append(-1.0)
                pick_rets = np.array(pick_rets)
                ret_m = float((pick_rets * o["weights"]).sum())
        if not o["cash"] and len(o["picks"]) > 0:
            equity *= (1 + ret_m) * (1 - cost_factor)
        rows.append({"date": o["asof"], "equity": equity, "ret_m": ret_m,
                     "regime": o["regime"], "n_picks": len(o["picks"]),
                     "n_eligible": o["n_eligible"],
                     "picks": ",".join(o["picks"])})
    return pd.DataFrame(rows)


def cagr_from(eq: pd.Series, start_cash: float = 1.0) -> float:
    if len(eq) == 0:
        return 0.0
    n = len(eq)
    years = max(n / 12.0, 1 / 12.0)
    return (eq.iloc[-1] / start_cash) ** (1.0 / years) - 1.0


def sharpe_monthly(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(12)


def max_drawdown(eq: pd.Series, dates: pd.DatetimeIndex):
    s = pd.Series(eq.values, index=dates)
    peak = s.cummax()
    dd = (s - peak) / peak
    end = dd.idxmin()
    start = s.loc[:end].idxmax()
    return float(dd.min()), start, end


# ---------------------------------------------------------------------------
def walk_forward_splits():
    return [
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


def walk_forward_eval(eq: pd.DataFrame, spy_aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, lo, hi in walk_forward_splits():
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        ret = e["ret_m"].astype(float)
        eqc = (1 + ret).cumprod()
        cagr_v = (eqc.iloc[-1]) ** (12.0 / len(eqc)) - 1
        sh = sharpe_monthly(ret)
        mdd, _, _ = max_drawdown(eqc, pd.DatetimeIndex(e["date"]))
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        spy_ret = spy["spy_ret_m"].astype(float)
        spy_eq = (1 + spy_ret).cumprod()
        spy_cagr = (spy_eq.iloc[-1]) ** (12.0 / len(spy_eq)) - 1 if len(spy_eq) else 0.0
        rows.append({
            "split": name, "from": lo.date(), "to": hi.date(),
            "n_months": len(e), "cagr": cagr_v, "spy_cagr": spy_cagr,
            "edge_pp": (cagr_v - spy_cagr) * 100,
            "sharpe": sh, "max_dd": mdd,
            "n_cash_months": int((e["regime"] == "crash").sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    print("=== Loading inputs ===")
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    print(f"  PIT members: {len(members)}")

    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    print(f"  monthly returns: {monthly_returns.shape}")

    spy = load_spy_features()
    print(f"  SPY features: {spy.shape}")

    panel_path = PIT / "sp500_pit_panel.parquet"
    if panel_path.exists():
        print(f"=== Loading cached panel from {panel_path} ===")
        panel = pd.read_parquet(panel_path)
    else:
        print("\n=== Building PIT cross-section panel ===")
        t0 = time.time()
        panel = build_pit_panel(members, monthly_returns)
        panel.to_parquet(panel_path, index=False)
        print(f"  built in {time.time()-t0:.1f}s, saved to {panel_path}")

    print("\n=== Walk-forward fitting GBM (annual retrain, 7m embargo) ===")
    preds = fit_walkforward(panel)
    preds.to_parquet(PIT / "sp500_pit_retrain_preds.parquet", index=False)
    print(f"  preds: {preds.shape}, asofs={preds['asof'].nunique()}")

    print("\n=== Building strategy outputs (regime gate + S&P 500 picks) ===")
    outs = build_outputs(preds, spy)
    print(f"  {len(outs)} months")

    print("\n=== Simulating ===")
    eq = simulate(outs, monthly_returns, cost_bps=10.0, starting_cash=1.0)
    eq.to_csv(PIT / "sp500_pit_retrain_equity.csv", index=False)
    print(f"  final equity: ${eq['equity'].iloc[-1]:.2f}")
    cgr = cagr_from(eq["equity"])
    sh = sharpe_monthly(eq["ret_m"])
    mdd, dd_s, dd_e = max_drawdown(eq["equity"], pd.DatetimeIndex(eq["date"]))
    print(f"  CAGR: {cgr*100:.2f}%, Sharpe: {sh:.2f}, MaxDD: {mdd*100:.2f}%")

    # SPY benchmark
    spy_ret = monthly_returns["SPY"]
    eq_dates = pd.to_datetime(eq["date"])
    next_month = eq_dates + pd.offsets.MonthEnd(1)
    spy_aligned = pd.DataFrame({
        "date": eq_dates,
        "spy_ret_m": [float(spy_ret.loc[nxt]) if nxt in spy_ret.index else 0.0
                      for nxt in next_month],
    })
    spy_eq = (1 + spy_aligned["spy_ret_m"]).cumprod()
    spy_cagr = spy_eq.iloc[-1] ** (12.0 / len(spy_eq)) - 1
    print(f"  SPY buy-and-hold CAGR (same window): {spy_cagr*100:.2f}%")
    edge_pp = (cgr - spy_cagr) * 100
    print(f"  Edge: {edge_pp:+.2f}pp")

    # Year-by-year
    eq["year"] = eq["date"].dt.year
    yr = eq.groupby("year")["ret_m"].apply(lambda x: ((1 + x).prod() - 1)).rename("year_ret")
    spy_aligned["year"] = pd.to_datetime(spy_aligned["date"]).dt.year
    spy_yr = spy_aligned.groupby("year")["spy_ret_m"].apply(lambda x: ((1 + x).prod() - 1)).rename("spy_year_ret")
    yr_combined = yr.to_frame().join(spy_yr.to_frame(), how="left")
    yr_combined["edge_pp"] = (yr_combined["year_ret"] - yr_combined["spy_year_ret"]) * 100
    yr_combined.to_csv(PIT / "sp500_pit_retrain_yearly.csv")
    print("\n[year-by-year] (%)")
    print((yr_combined * pd.Series({"year_ret": 100, "spy_year_ret": 100, "edge_pp": 1})).round(1).to_string())

    # WF
    wf = walk_forward_eval(eq, spy_aligned)
    wf.to_csv(PIT / "sp500_pit_retrain_walkforward.csv", index=False)
    print("\n[walk-forward]")
    print(wf.round(3).to_string(index=False))

    # Drawdowns
    eq_idx = pd.Series(eq["equity"].values, index=pd.DatetimeIndex(eq["date"]))
    peak = eq_idx.cummax()
    dd_curve = (eq_idx - peak) / peak
    eq["dd_from_peak"] = dd_curve.values

    # Regime distribution
    eq["year"] = eq["date"].dt.year
    reg = eq.groupby(["year", "regime"]).size().unstack(fill_value=0)
    reg.to_csv(PIT / "sp500_pit_retrain_regimes.csv")
    print("\n[regime distribution by year]")
    print(reg.to_string())

    summary = {
        "as_of": str(eq["date"].max().date()),
        "n_months": int(len(eq)),
        "starting_cash": 1.0,
        "final_equity": float(eq["equity"].iloc[-1]),
        "cagr": float(cgr),
        "spy_buyhold_cagr": float(spy_cagr),
        "edge_vs_spy_pp": float(edge_pp),
        "sharpe_monthly_annl": float(sh),
        "max_drawdown": float(mdd),
        "max_dd_start": str(pd.Timestamp(dd_s).date()),
        "max_dd_trough": str(pd.Timestamp(dd_e).date()),
        "n_cash_months": int((eq["regime"] == "crash").sum()),
        "n_normal_months": int((eq["regime"] == "normal").sum()),
        "n_bull_months": int((eq["regime"] == "bull").sum()),
        "n_recovery_months": int((eq["regime"] == "recovery").sum()),
        "wf_mean_cagr": float(wf["cagr"].mean()),
        "wf_median_cagr": float(wf["cagr"].median()),
        "wf_min_cagr": float(wf["cagr"].min()),
        "wf_max_cagr": float(wf["cagr"].max()),
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()),
        "wf_n_positive": int((wf["cagr"] > 0).sum()),
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()),
        "wf_n_splits": int(len(wf)),
    }
    (PIT / "sp500_pit_retrain_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[summary]")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
