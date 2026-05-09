"""Filter-only point-in-time S&P 500 backtest of the v2 ML strategy.

This script takes the existing cross-universe ML predictions (`ml_preds_v2.parquet`,
trained on the broader 1,811-ticker universe) and applies the regime gate exactly
as v2-ml-apex does, but **restricts the picking universe at each month-end T to
the set of stocks that were S&P 500 constituents on T**.

It produces:
  - cache/v2/sp500_pit/sp500_pit_filter_equity.csv  (full equity curve)
  - cache/v2/sp500_pit/sp500_pit_filter_picks.csv   (per-month picks)
  - cache/v2/sp500_pit/sp500_pit_filter_summary.json
  - cache/v2/sp500_pit/sp500_pit_filter_yearly.csv
  - cache/v2/sp500_pit/sp500_pit_filter_walkforward.csv
  - cache/v2/sp500_pit/sp500_pit_filter_drawdowns.csv
  - cache/v2/sp500_pit/sp500_pit_filter_regimes.csv

Same regime gate as the live strategy (mode='tight'), same K's
(K_normal=15, K_recovery=7, K_bull=7, cash on crash), same 10bp/month cost,
same equal-weighting within picks, same walk-forward split definitions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
PIT.mkdir(parents=True, exist_ok=True)

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


def load_spy_features() -> pd.DataFrame:
    """Read each feature file and pull the SPY row."""
    rows = []
    for f in sorted((CACHE / "features").glob("*.parquet")):
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
@dataclass
class StratOutput:
    asof: pd.Timestamp
    picks: list
    weights: np.ndarray
    cash: bool
    regime: str
    n_eligible: int


def build_outputs(
    preds: pd.DataFrame,
    spy: pd.DataFrame,
    members: pd.DataFrame,
    k_normal: int = 15, k_recovery: int = 7, k_bull: int = 7,
) -> list:
    """Apply regime gate + S&P-500 filter to predictions."""
    members_g = members.groupby("asof")["ticker"].apply(set)
    months = sorted(preds["asof"].unique())
    outs = []
    for m in months:
        m = pd.Timestamp(m)
        if m not in spy.index:
            continue
        regime = classify_regime(spy.loc[m].to_dict())

        sub = preds[preds["asof"] == m].copy()
        sub = sub[~sub["ticker"].isin(EXCLUDE)]

        sp_set = members_g.get(m, set())
        sub_pit = sub[sub["ticker"].isin(sp_set)].sort_values("pred", ascending=False)

        if regime == "crash":
            outs.append(StratOutput(m, [], np.array([]), True, regime, len(sub_pit)))
            continue
        k = {"recovery": k_recovery, "bull": k_bull, "normal": k_normal}[regime]
        top = sub_pit.head(k)
        if len(top) < k:
            outs.append(StratOutput(m, [], np.array([]), True, regime, len(sub_pit)))
            continue
        weights = np.ones(k) / k
        outs.append(StratOutput(m, top["ticker"].tolist(), weights, False, regime, len(sub_pit)))
    return outs


def _nearest_pos(idx, target, tol_days=7):
    pos = idx.searchsorted(target)
    cands = []
    for j in (pos - 1, pos):
        if 0 <= j < len(idx):
            cands.append((j, abs((idx[j] - target).days)))
    cands.sort(key=lambda x: x[1])
    if not cands or cands[0][1] > tol_days:
        return None
    return cands[0][0]


def simulate(outs, monthly_returns, cost_bps: float = 10.0, starting_cash: float = 1.0):
    equity = starting_cash
    cost_factor = cost_bps / 10000.0
    rows = []
    for o in outs:
        if o.cash or len(o.picks) == 0:
            ret_m = 0.0
        else:
            pos1 = _nearest_pos(monthly_returns.index, o.asof)
            if pos1 is None or pos1 + 1 >= len(monthly_returns.index):
                ret_m = 0.0
            else:
                next_d = monthly_returns.index[pos1 + 1]
                pick_rets = []
                for tk in o.picks:
                    if tk in monthly_returns.columns:
                        r = monthly_returns.at[next_d, tk]
                        pick_rets.append(-1.0 if pd.isna(r) else float(r))
                    else:
                        pick_rets.append(-1.0)
                pick_rets = np.array(pick_rets)
                ret_m = float((pick_rets * o.weights).sum())
        if not o.cash and len(o.picks) > 0:
            equity *= (1 + ret_m) * (1 - cost_factor)
        rows.append({"date": o.asof, "equity": equity, "ret_m": ret_m,
                     "regime": o.regime, "n_picks": len(o.picks),
                     "n_eligible": o.n_eligible,
                     "picks": ",".join(o.picks),
                     })
    return pd.DataFrame(rows)


def cagr_from(equity: pd.Series, start_cash: float = 1.0) -> float:
    if len(equity) == 0:
        return 0.0
    n = len(equity)
    years = max(n / 12.0, 1 / 12.0)
    return (equity.iloc[-1] / start_cash) ** (1.0 / years) - 1.0


def sharpe_monthly(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(12)


def max_drawdown(equity: pd.Series, date_index: pd.DatetimeIndex | None = None) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    if len(equity) == 0:
        return 0.0, pd.NaT, pd.NaT
    if date_index is not None:
        equity = pd.Series(equity.values, index=date_index)
    peak = equity.cummax()
    dd = (equity - peak) / peak
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    return float(dd.min()), start, end


# ---------------------------------------------------------------------------
def yearly_returns(eq: pd.DataFrame) -> pd.DataFrame:
    eq = eq.copy()
    eq["year"] = pd.to_datetime(eq["date"]).dt.year
    out = eq.groupby("year")["ret_m"].apply(lambda x: ((1 + x).prod() - 1)).rename("year_ret")
    return out.to_frame()


def spy_yearly(spy_ret: pd.Series, eq: pd.DataFrame) -> pd.Series:
    eq_dates = pd.to_datetime(eq["date"])
    next_month = eq_dates + pd.offsets.MonthEnd(1)
    spy_eq_aligned = []
    for i, d in enumerate(eq_dates):
        # Strategy returns are realised in the *next* month after o.asof.
        # SPY benchmark return at the same month: spy_ret on the next month-end.
        nxt = next_month.iloc[i]
        # find nearest spy_ret date
        if nxt in spy_ret.index:
            spy_eq_aligned.append((d, float(spy_ret.loc[nxt])))
        else:
            spy_eq_aligned.append((d, 0.0))
    s = pd.DataFrame(spy_eq_aligned, columns=["date", "spy_ret_m"])
    s["year"] = pd.to_datetime(s["date"]).dt.year
    return s.groupby("year")["spy_ret_m"].apply(lambda x: ((1 + x).prod() - 1))


# ---------------------------------------------------------------------------
def walk_forward_splits():
    """Same 10 splits as v2 walk_forward_validate.py."""
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


def walk_forward(eq: pd.DataFrame, spy_yr_aligned: pd.DataFrame) -> pd.DataFrame:
    """Re-aggregate the equity curve over each WF window."""
    rows = []
    for name, lo, hi in walk_forward_splits():
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        # Build window equity from monthly returns in window
        ret = e["ret_m"].astype(float)
        eq_curve = (1 + ret).cumprod()
        cagr_v = (eq_curve.iloc[-1]) ** (12.0 / len(eq_curve)) - 1
        sh = sharpe_monthly(ret)
        mdd, dd_s, dd_e = max_drawdown(eq_curve)
        # SPY same window
        spy = spy_yr_aligned[(spy_yr_aligned["date"] >= lo) & (spy_yr_aligned["date"] <= hi)]
        spy_ret = spy["spy_ret_m"].astype(float)
        spy_eq = (1 + spy_ret).cumprod()
        spy_cagr = (spy_eq.iloc[-1]) ** (12.0 / len(spy_eq)) - 1 if len(spy_eq) else 0.0
        rows.append({
            "split": name, "from": lo.date(), "to": hi.date(),
            "n_months": len(e),
            "cagr": cagr_v, "spy_cagr": spy_cagr, "edge_pp": (cagr_v - spy_cagr) * 100,
            "sharpe": sh, "max_dd": mdd,
            "n_cash_months": int((e["regime"] == "crash").sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    print("=== Loading inputs ===")
    preds = pd.read_parquet(V2 / "ml_preds_v2.parquet")
    preds["asof"] = pd.to_datetime(preds["asof"])
    print(f"  preds: {len(preds)} rows, {preds['asof'].nunique()} months, "
          f"{preds['asof'].min().date()} -> {preds['asof'].max().date()}")

    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    print(f"  PIT members: {len(members)}, {members['asof'].nunique()} months")

    spy = load_spy_features()
    print(f"  SPY features: {spy.shape}")

    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    print(f"  monthly returns: {monthly_returns.shape}")

    print("\n=== Building strategy outputs (regime gate + SP500 PIT filter) ===")
    outs = build_outputs(preds, spy, members)
    print(f"  {len(outs)} months")
    regimes = pd.Series([o.regime for o in outs]).value_counts()
    print(f"  regime distribution:\n{regimes.to_string()}")

    print("\n=== Simulating ===")
    eq = simulate(outs, monthly_returns, cost_bps=10.0, starting_cash=1.0)
    eq.to_csv(PIT / "sp500_pit_filter_equity.csv", index=False)
    print(f"  months: {len(eq)}")
    print(f"  final equity: ${eq['equity'].iloc[-1]:.2f}")
    cgr = cagr_from(eq["equity"])
    sh = sharpe_monthly(eq["ret_m"])
    mdd, dd_s, dd_e = max_drawdown(eq["equity"], pd.DatetimeIndex(eq["date"]))
    print(f"  CAGR: {cgr*100:.2f}%")
    print(f"  Sharpe: {sh:.2f}")
    print(f"  MaxDD: {mdd*100:.2f}% from {pd.Timestamp(dd_s).date()} to {pd.Timestamp(dd_e).date()}")

    # SPY benchmark over same window
    spy_ret = monthly_returns["SPY"]
    yr = yearly_returns(eq)
    spy_yr = spy_yearly(spy_ret, eq)
    yr_combined = yr.join(spy_yr.rename("spy_year_ret"), how="left")
    yr_combined["edge_pp"] = (yr_combined["year_ret"] - yr_combined["spy_year_ret"]) * 100
    yr_combined.to_csv(PIT / "sp500_pit_filter_yearly.csv")
    print("\n[year-by-year]")
    print((yr_combined * 100).round(1).to_string())

    # SPY DCA / buy-and-hold benchmarks over the full window
    eq_dates = pd.to_datetime(eq["date"])
    next_month = eq_dates + pd.offsets.MonthEnd(1)
    spy_aligned = []
    for d, nxt in zip(eq_dates, next_month):
        spy_aligned.append({"date": d, "spy_ret_m": float(spy_ret.loc[nxt]) if nxt in spy_ret.index else 0.0})
    spy_aligned = pd.DataFrame(spy_aligned)
    spy_eq = (1 + spy_aligned["spy_ret_m"]).cumprod()
    spy_cagr = spy_eq.iloc[-1] ** (12.0 / len(spy_eq)) - 1
    print(f"\n[SPY buy-and-hold over same window] CAGR={spy_cagr*100:.2f}%")
    edge_pp = (cgr - spy_cagr) * 100
    print(f"[Edge vs SPY buy-and-hold]              {edge_pp:+.2f}pp")

    # Walk-forward
    wf = walk_forward(eq, spy_aligned)
    wf.to_csv(PIT / "sp500_pit_filter_walkforward.csv", index=False)
    print("\n[walk-forward] (10 splits)")
    print(wf.round(3).to_string(index=False))

    # Drawdown report
    dd_curve = eq["equity"].cummax()
    eq["dd_from_peak"] = (eq["equity"] - dd_curve) / dd_curve
    dd_episodes = []
    in_dd = False
    start = None
    for i, r in eq.iterrows():
        if not in_dd and r["dd_from_peak"] < -0.05:
            in_dd = True; start = r["date"]
            depth = r["dd_from_peak"]; trough = r["date"]
        elif in_dd:
            if r["dd_from_peak"] < depth:
                depth = r["dd_from_peak"]; trough = r["date"]
            if r["dd_from_peak"] >= -0.001:
                dd_episodes.append({"start": start, "trough": trough, "end": r["date"],
                                    "depth_pct": float(depth*100)})
                in_dd = False
    if in_dd:
        dd_episodes.append({"start": start, "trough": trough, "end": eq["date"].iloc[-1],
                            "depth_pct": float(depth*100)})
    dd_df = pd.DataFrame(dd_episodes).sort_values("depth_pct").head(10)
    dd_df.to_csv(PIT / "sp500_pit_filter_drawdowns.csv", index=False)
    print("\n[top drawdowns]")
    print(dd_df.round(2).to_string(index=False))

    # Regime distribution
    eq["year"] = eq["date"].dt.year
    reg = eq.groupby(["year", "regime"]).size().unstack(fill_value=0)
    reg.to_csv(PIT / "sp500_pit_filter_regimes.csv")

    # Coverage (S&P 500 members in panel) for transparency
    cov = members.merge(preds.assign(in_panel=1)[["asof", "ticker", "in_panel"]],
                        on=["asof", "ticker"], how="left")
    cov["in_panel"] = cov["in_panel"].fillna(0).astype(int)
    coverage_yr = cov.groupby(cov["asof"].dt.year)["in_panel"].mean()
    coverage_yr.to_csv(PIT / "sp500_pit_filter_coverage.csv")
    print("\n[panel coverage of S&P 500 members per year (mean)]")
    print((coverage_yr * 100).round(1).to_string())

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
        "panel_coverage_2003_first": float(coverage_yr.iloc[0]),
        "panel_coverage_2024_last": float(coverage_yr.iloc[-1]),
    }
    (PIT / "sp500_pit_filter_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[summary]")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
