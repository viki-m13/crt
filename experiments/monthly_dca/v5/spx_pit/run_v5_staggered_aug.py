"""Phase 6: staggered v5 on the augmented PIT panel — rebalance-timing-luck mitigation.

CONTEXT
=======
Earlier work (experiments/monthly_dca/v5/validations/REPORT_2024_diagnosis.md)
showed that the deployed v5's 2024 underperformance was a 2-out-of-12
timing artifact: the lump-sum Jan-31 and Jul-31 rebalance dates happened to
land on the year's worst picking moments. The other 10 monthly entries
delivered +3.7pp edge on average.

The validated mitigation is **6 staggered tranches**: at every month-end,
contribute fresh capital to a NEW basket and hold each tranche for 6 months
(so 6 baskets are active at any time, started in each of the last 6
months). This converts a single twice-yearly rebalance into 12 monthly
rebalances of 1/6th of the capital, averaging out timing luck.

This script runs the staggered scheme on the AUGMENTED PIT panel
(`prices_extended_pit.parquet` + retrained ml_preds + regenerated Chronos)
and compares to the single-tranche deployed-v5 result.

PARAMETERS — match deployed v5 winner exactly
==============================================
  scorer:          ml_3plus6  (avg pred_3m + pred_6m)
  Chronos filter:  p70 rank >= 0.45
  K:               3 picks
  weighting:       inverse-vol (12m) capped at 40% per pick
  hold:            6 months per tranche
  regime gate:     tight (crash -> cash; recovery / bull / normal -> K=3)
  cost:            10 bps round-trip on basket changes

Inputs (augmented):
  augmented/sp500_pit_panel.parquet
  augmented/ml_preds.parquet
  augmented/ml_preds_chronos.parquet
  augmented/monthly_returns_clean.parquet
  augmented/monthly_prices_clean.parquet
  augmented/features/*.parquet            (SPY regime features)

Outputs (also in augmented/):
  v5_staggered_summary.json               headline metrics for staggered v5
  v5_staggered_walkforward.csv            10-split WF for staggered
  v5_staggered_yearly.csv                 year-by-year strategy vs SPY
  v5_staggered_equity.csv                 NAV over time
  v5_staggered_tranches.csv               per-tranche entry/exit log
  v5_staggered_vs_lump.json               side-by-side comparison
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
AUG = PIT / "augmented"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}

CHRONOS_FILTER_Q = 0.45
CAP_PER_PICK = 0.40
HOLD_MONTHS = 6
K_PICKS = 3
COST_BPS = 10.0

WF_SPLITS = [
    ("A1",       "2011-01-01", "2018-12-31"),
    ("A2",       "2015-01-01", "2021-12-31"),
    ("A3",       "2018-01-01", "2024-12-31"),
    ("R1_GFC",   "2008-01-01", "2010-12-31"),
    ("R2",       "2011-01-01", "2013-12-31"),
    ("R3",       "2014-01-01", "2016-12-31"),
    ("R4",       "2017-01-01", "2019-12-31"),
    ("R5_COVID", "2020-01-01", "2022-12-31"),
    ("R6_AI",    "2023-01-01", "2024-12-31"),
    ("STRICT",   "2021-01-01", "2024-12-31"),
]


def classify_regime_tight(s: dict) -> str:
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
    rows = []
    for f in sorted((AUG / "features").glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


def calc_invvol_weights(tickers: list, monthly_returns: pd.DataFrame,
                        asof: pd.Timestamp, cap: float = CAP_PER_PICK) -> np.ndarray:
    """1/vol_1y weights, capped at `cap`, renormalised."""
    mr_idx = monthly_returns.index
    pos = mr_idx.searchsorted(asof)
    if pos == 0:
        return np.ones(len(tickers)) / len(tickers)
    window = monthly_returns.iloc[max(0, pos - 12): pos]
    vols = []
    for tk in tickers:
        if tk in window.columns:
            v = window[tk].dropna().std()
            vols.append(float(v) if v and not np.isnan(v) and v > 0 else np.nan)
        else:
            vols.append(np.nan)
    vols = np.array(vols)
    if np.all(np.isnan(vols)) or np.all(vols == 0):
        return np.ones(len(tickers)) / len(tickers)
    inv = np.where(np.isnan(vols) | (vols == 0), 1e-9, 1.0 / vols)
    w = inv / inv.sum()
    if cap < 1.0:
        for _ in range(8):
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            under = ~over
            if not under.any():
                break
            w[under] += excess * w[under] / w[under].sum()
        w = w / w.sum()
    return w


def pick_v5(m: pd.Timestamp, panel_by_asof, ml_by_asof, chr_by_asof,
            members_g, monthly_returns) -> tuple[list, np.ndarray]:
    """Return (picks, weights) for v5 at month-end `m`. Empty if no eligible."""
    sub_panel = panel_by_asof.get(m)
    sub_ml = ml_by_asof.get(m)
    sub_chr = chr_by_asof.get(m)
    if sub_panel is None or sub_ml is None:
        return [], np.array([])
    sp_set = members_g.get(m, set())
    sub = sub_panel[sub_panel["ticker"].isin(sp_set)]
    sub = sub[~sub["ticker"].isin(EXCLUDE)]
    sub = sub.merge(sub_ml[["ticker", "ml_score"]], on="ticker", how="left")
    sub = sub.dropna(subset=["ml_score"])
    if sub_chr is not None and not sub_chr.empty:
        sub = sub.merge(sub_chr[["ticker", "chronos_p70_3m"]], on="ticker", how="left")
        sub = sub.dropna(subset=["chronos_p70_3m"])
        sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)
        sub = sub[sub["chr_p70_rk"] >= CHRONOS_FILTER_Q]
    sub = sub.sort_values("ml_score", ascending=False)
    top = sub.head(K_PICKS)
    if len(top) < K_PICKS:
        return [], np.array([])
    picks = top["ticker"].tolist()
    weights = calc_invvol_weights(picks, monthly_returns, m, cap=CAP_PER_PICK)
    return picks, weights


def _next_month_returns(monthly_returns: pd.DataFrame, m: pd.Timestamp) -> pd.Series:
    """Returns at the month-end RIGHT AFTER `m` (i.e. the realized 1m return)."""
    mr_idx = monthly_returns.index
    pos = mr_idx.searchsorted(m)
    if pos + 1 >= len(mr_idx):
        return pd.Series(dtype=float)
    next_d = mr_idx[pos + 1]
    return monthly_returns.loc[next_d]


def run_staggered(panel, ml, chr_, monthly_returns, monthly_prices,
                   members_g, spy) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """6-tranche staggered v5. Returns (equity_log, tranche_log, yearly).

    Cash-flow design (mirrors experiments/monthly_dca/v5/validations/run_staggered_dca.py):

      Each month-end m:
        1. Mark all open tranches to current month's prices -> t['value']
        2. Close any tranche that has held HOLD_MONTHS or longer ->
           realised cash = t['value'] * (1 - cf), added to `proceeds`.
        3. capital_to_deploy = deposit (1.0) + proceeds + cash_pool_carried_forward
        4. If regime == 'crash' or no picks form -> stash capital_to_deploy
           in cash_pool to carry to next month. Otherwise open a NEW tranche
           with capital_to_deploy * (1 - cf), allocated by invvol weights.
        5. NAV = sum(open tranche values) + cash_pool.

    The previous-month cash_pool is correctly carried forward across months
    (the bug that this fixes was dropping it).
    """
    cf = COST_BPS / 1e4
    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    ml_by_asof = {a: g for a, g in ml.groupby("asof")}
    chr_by_asof = {a: g for a, g in chr_.groupby("asof")}

    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]
    print(f"  simulation months: {len(months)} ({months[0].date()}..{months[-1].date()})")

    active: list[dict] = []
    tranche_log: list[dict] = []
    equity_log: list[dict] = []
    deposit = 1.0
    cash_pool = 0.0
    cum_deposits = 0.0

    for i, m in enumerate(months):
        regime = classify_regime_tight(spy.loc[m].to_dict() if m in spy.index else {})

        # 1) Mark active tranches to current month's prices
        for t in active:
            t["value"] = _tranche_value(t, monthly_prices, m)

        # 2) Close tranches that have held HOLD_MONTHS or longer
        keep = []
        proceeds = 0.0
        for t in active:
            months_held = (m.year - t["entry_date"].year) * 12 + (m.month - t["entry_date"].month)
            if months_held >= HOLD_MONTHS:
                realised = t["value"] * (1 - cf)
                proceeds += realised
                tranche_log.append({
                    "entry_date": t["entry_date"], "exit_date": m,
                    "entered_with": t["notional_at_entry"], "exited_with": realised,
                    "return_pct": 100.0 * (realised / t["notional_at_entry"] - 1.0),
                    "picks": ",".join(t["picks"]),
                    "regime_at_entry": t["regime_at_entry"],
                })
            else:
                keep.append(t)
        active = keep

        # 3) Form deploy capital — carries previous cash_pool + this-month
        #    deposit + matured proceeds.
        capital_to_deploy = deposit + proceeds + cash_pool
        cum_deposits += deposit
        cash_pool = 0.0  # zeroed before this month's reform decision

        # 4) Try to form a new tranche
        new_capital_invested = False
        if regime == "crash":
            cash_pool = capital_to_deploy  # keep capital in cash through crash months
        else:
            picks, weights = pick_v5(m, panel_by_asof, ml_by_asof, chr_by_asof,
                                     members_g, monthly_returns)
            if not picks:
                cash_pool = capital_to_deploy
            else:
                spent = capital_to_deploy * (1 - cf)
                units = {}
                for tk, w in zip(picks, weights):
                    if tk in monthly_prices.columns and m in monthly_prices.index:
                        px = float(monthly_prices.at[m, tk])
                        if not pd.isna(px) and px > 0:
                            units[tk] = (spent * w) / px
                if units:
                    active.append({
                        "entry_date": m, "picks": picks,
                        "weights": weights, "units": units,
                        "notional_at_entry": capital_to_deploy,
                        "value": spent,
                        "regime_at_entry": regime,
                    })
                    new_capital_invested = True
                else:
                    cash_pool = capital_to_deploy

        # 5) Aggregate equity
        nav_invested = sum(t["value"] for t in active)
        nav_total = nav_invested + cash_pool

        equity_log.append({
            "date": m, "regime": regime,
            "n_active_tranches": len(active),
            "nav_invested": nav_invested,
            "cash_pool": cash_pool,
            "nav_total": nav_total,
            "cum_deposits": cum_deposits,
            "new_capital_invested": new_capital_invested,
        })

    eq = pd.DataFrame(equity_log)
    tranches = pd.DataFrame(tranche_log)
    return eq, tranches


def _tranche_value(t, monthly_prices, m):
    """Mark-to-market a tranche at month-end m using each ticker's units * price.
    Falls back to last-known price if current month is NaN."""
    val = 0.0
    for tk, units in t["units"].items():
        if tk in monthly_prices.columns:
            if m in monthly_prices.index:
                px = monthly_prices.at[m, tk]
            else:
                px = np.nan
            if pd.isna(px):
                # Forward-fill from last available price
                col = monthly_prices[tk]
                ffill = col.loc[:m].dropna()
                px = float(ffill.iloc[-1]) if len(ffill) > 0 else 0.0
            val += float(px) * units
    return val


def compute_returns_from_equity(eq: pd.DataFrame, monthly_returns: pd.DataFrame) -> pd.Series:
    """Compute per-month return of the deployed capital.

    Convention: deposit is added at the START of each month and immediately
    invested (less cost). NAV_t_after_deposit = NAV_t-1 + deposit_t.
    ret_m = NAV_t_end / NAV_t_after_deposit - 1
    """
    deposits = eq.index.size and 1.0
    rets = []
    prev_nav_end = 0.0
    for _, row in eq.iterrows():
        nav_after_deposit = prev_nav_end + 1.0
        ret = (row["nav_total"] - nav_after_deposit) / max(nav_after_deposit, 1e-9)
        rets.append(ret)
        prev_nav_end = row["nav_total"]
    return pd.Series(rets, index=eq["date"])


def yearly_table(eq: pd.DataFrame, monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """Year-by-year strategy vs SPY edge for the staggered approach."""
    rets = compute_returns_from_equity(eq, monthly_returns)
    spy_ret = monthly_returns["SPY"]
    df = pd.DataFrame({"date": rets.index, "ret_m": rets.values})
    df["year"] = pd.to_datetime(df["date"]).dt.year
    # Strategy: monthly returns of the deployed-capital NAV
    yr_strat = df.groupby("year")["ret_m"].apply(lambda r: (1 + r).prod() - 1)
    # SPY benchmark: realized in the SAME month-of-return (so SPY ret at
    # next-month-end aligned to strategy month-end m). For yearly aggregation
    # this is approximately the calendar-year SPY return.
    spy_year_rets = []
    for yr in yr_strat.index:
        spy_mask = (spy_ret.index.year == yr)
        s = spy_ret[spy_mask].dropna()
        spy_year_rets.append((1 + s).prod() - 1 if len(s) else 0.0)
    yr_df = pd.DataFrame({
        "year": yr_strat.index,
        "strategy_ret": yr_strat.values,
        "spy_ret": spy_year_rets,
    })
    yr_df["edge_pp"] = (yr_df["strategy_ret"] - yr_df["spy_ret"]) * 100
    return yr_df


def walkforward_table(eq: pd.DataFrame, monthly_returns: pd.DataFrame) -> pd.DataFrame:
    rets = compute_returns_from_equity(eq, monthly_returns)
    spy_ret = monthly_returns["SPY"]
    rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        mask = (rets.index >= lo) & (rets.index <= hi)
        r = rets[mask].astype(float)
        if len(r) == 0:
            continue
        ec = (1 + r).cumprod()
        cagr_v = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        sh = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
        peak = ec.cummax()
        mdd = float(((ec - peak) / peak).min())
        spy_mask = (spy_ret.index >= lo) & (spy_ret.index <= hi)
        s = spy_ret[spy_mask].dropna()
        sc = (1 + s).cumprod()
        spy_cagr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1 if len(sc) else 0.0
        rows.append({
            "split": split, "from": str(lo.date()), "to": str(hi.date()),
            "n_m": int(len(r)),
            "cagr": float(cagr_v), "spy_cagr": float(spy_cagr),
            "edge_pp": float((cagr_v - spy_cagr) * 100),
            "sharpe": float(sh), "max_dd": float(mdd),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 64)
    print("Phase 6: deployed v5-winner with 6-tranche stagger on augmented PIT")
    print("=" * 64)

    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    print(f"[1] panel: {panel.shape}")

    ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml["ml_score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2

    chr_ = pd.read_parquet(AUG / "ml_preds_chronos.parquet")[["asof", "ticker", "chronos_p70_3m"]]
    chr_["asof"] = pd.to_datetime(chr_["asof"])

    spy_features = load_spy_features()
    monthly_returns = pd.read_parquet(AUG / "monthly_returns_clean.parquet")
    monthly_prices = pd.read_parquet(AUG / "monthly_prices_clean.parquet")
    if not isinstance(monthly_returns.index, pd.DatetimeIndex):
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_prices.index = pd.to_datetime(monthly_prices.index)
    # NaN fill so genuine acquisition NaN doesn't book -100% in tranche mark-to-market.
    monthly_returns = monthly_returns.fillna(0.0)

    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()

    eq, tranches = run_staggered(panel, ml, chr_, monthly_returns, monthly_prices,
                                  members_g, spy_features)

    eq.to_csv(AUG / "v5_staggered_equity.csv", index=False)
    if len(tranches):
        tranches.to_csv(AUG / "v5_staggered_tranches.csv", index=False)

    yr = yearly_table(eq, monthly_returns)
    yr.to_csv(AUG / "v5_staggered_yearly.csv", index=False)

    wf = walkforward_table(eq, monthly_returns)
    wf.to_csv(AUG / "v5_staggered_walkforward.csv", index=False)

    # Aggregate headline. Use deployed-capital return series (NAV vs deposits)
    rets = compute_returns_from_equity(eq, monthly_returns)
    n_months = len(rets)
    cagr_full = (1 + rets).prod() ** (12.0 / n_months) - 1
    spy_full = (1 + monthly_returns["SPY"].loc[rets.index[0]:rets.index[-1]].dropna()).prod() ** (12.0 / n_months) - 1
    sharpe = (rets.mean() / max(rets.std(), 1e-9)) * np.sqrt(12)
    ec = (1 + rets).cumprod()
    peak = ec.cummax()
    mdd = float(((ec - peak) / peak).min())

    print(f"\n[staggered headline]")
    print(f"  n_months: {n_months}")
    print(f"  cagr_full: {cagr_full:.4f}  (SPY: {spy_full:.4f},  edge: {(cagr_full - spy_full)*100:+.2f}pp)")
    print(f"  sharpe: {sharpe:.4f}")
    print(f"  max_dd: {mdd:.4f}")
    print(f"  n_cash_months (regime=crash): {int((eq['regime'] == 'crash').sum())}")
    print(f"  n_tranches_closed: {len(tranches)}")
    print(f"  WF: {wf.to_string(index=False)}")

    print(f"\n[yearly]")
    print(yr.assign(strategy_pct=(yr['strategy_ret']*100).round(1),
                    spy_pct=(yr['spy_ret']*100).round(1),
                    edge_pp_r=yr['edge_pp'].round(1)
                    )[["year","strategy_pct","spy_pct","edge_pp_r"]].to_string(index=False))

    summary = {
        "variant_name": "v5_staggered_6tranche_winner",
        "panel": "augmented_PIT",
        "n_months": int(n_months),
        "cagr_full": float(cagr_full),
        "spy_cagr_full": float(spy_full),
        "edge_full_pp": float((cagr_full - spy_full) * 100),
        "sharpe": float(sharpe),
        "max_dd": float(mdd),
        "n_cash_months": int((eq["regime"] == "crash").sum()),
        "n_tranches_closed": int(len(tranches)),
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else None,
        "wf_median_cagr": float(wf["cagr"].median()) if len(wf) else None,
        "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else None,
        "wf_max_cagr": float(wf["cagr"].max()) if len(wf) else None,
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else None,
        "wf_n_positive": int((wf["cagr"] > 0).sum()) if len(wf) else 0,
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else 0,
        "wf_n_splits": int(len(wf)),
    }
    (AUG / "v5_staggered_summary.json").write_text(json.dumps(summary, indent=2))

    # Side-by-side vs single-tranche (the apples-to-apples comparator)
    lump = json.loads((AUG / "v5_winner_summary.json").read_text())
    cmp = {
        "panel": "augmented_PIT",
        "lump_sum_v5":  {k: lump[k] for k in
                          ["cagr_full", "wf_mean_cagr", "sharpe", "max_dd",
                           "wf_n_beats_spy", "wf_n_splits"]},
        "staggered_v5": {k: summary[k] for k in
                          ["cagr_full", "wf_mean_cagr", "sharpe", "max_dd",
                           "wf_n_beats_spy", "wf_n_splits"]},
    }
    (AUG / "v5_staggered_vs_lump.json").write_text(json.dumps(cmp, indent=2))

    print(f"\n[saved]")
    print(f"  {AUG / 'v5_staggered_summary.json'}")
    print(f"  {AUG / 'v5_staggered_walkforward.csv'}")
    print(f"  {AUG / 'v5_staggered_yearly.csv'}")
    print(f"  {AUG / 'v5_staggered_equity.csv'}")
    print(f"  {AUG / 'v5_staggered_tranches.csv'}")
    print(f"  {AUG / 'v5_staggered_vs_lump.json'}")


if __name__ == "__main__":
    main()
