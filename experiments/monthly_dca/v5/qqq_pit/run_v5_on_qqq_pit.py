"""Run the v5 strategy (Chronos p70 filter + GBM + tight gate + inv-vol +
6m hold) honestly on the PIT Nasdaq-100 universe.

This is an out-of-sample generalisation test: the GBM and Chronos models
were trained / pre-trained without seeing the NDX cohort as a label.
The PIT-NDX membership data comes from jmccarrell/n100tickers (accurate
coverage from 2015-01-01).

Outputs:
  experiments/monthly_dca/v5/qqq_pit/qqq_pit_equity.csv
  experiments/monthly_dca/v5/qqq_pit/qqq_pit_trades.csv
  experiments/monthly_dca/v5/qqq_pit/qqq_pit_walkforward.csv
  experiments/monthly_dca/v5/qqq_pit/qqq_pit_report.json
"""
from __future__ import annotations
import json
import sys
import urllib.request
import ssl
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
QQQ_DIR = ROOT / "experiments" / "monthly_dca" / "v5" / "qqq_pit"
CACHE_V2 = ROOT / "experiments" / "monthly_dca" / "cache" / "v2"
SP500_PIT = CACHE_V2 / "sp500_pit"

# Strategy parameters (v5 winner)
CHRONOS_FILTER_Q = 0.45
CAP_PER_PICK = 0.40
HOLD_MONTHS = 6
K_PICKS = 3
COST_BPS = 10.0


def calc_invvol_weights(picks: list[str], monthly_returns: pd.DataFrame,
                         asof: pd.Timestamp, cap: float = CAP_PER_PICK) -> np.ndarray:
    """Inverse 1-year volatility weights, capped per pick."""
    idx = monthly_returns.index.searchsorted(asof) - 1
    if idx < 12:
        return np.ones(len(picks)) / len(picks)
    window = monthly_returns.iloc[max(0, idx - 11): idx + 1]
    vols = []
    for tk in picks:
        if tk in window.columns:
            v = window[tk].std()
            vols.append(v if pd.notna(v) and v > 0 else 0.10)
        else:
            vols.append(0.10)
    inv = 1.0 / np.array(vols)
    w = inv / inv.sum()
    # Cap and renormalize until no over-cap remains
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        if (~over).any():
            w[~over] += excess * w[~over] / w[~over].sum()
        else:
            break
    return w


def regime_tight(spy_now: dict) -> str:
    """Tight crash gate (mirrors v5 build script)."""
    ret_21d = spy_now.get("spy_ret_21d", 0.0)
    mom_6m = spy_now.get("spy_mom_6_1", 0.0)
    if ret_21d <= -0.08:
        return "crash"
    if mom_6m <= -0.05 and ret_21d <= -0.03:
        return "crash"
    mom_12m = spy_now.get("spy_mom_12_1", 0.0)
    dsma200 = spy_now.get("spy_dsma200", 0.0)
    if mom_12m >= 0.10 and dsma200 > 0:
        return "bull"
    return "normal"


def fetch_qqq_prices() -> pd.Series:
    """Fetch QQQ ETF monthly closes via yfinance. Returns month-end series."""
    import yfinance as yf
    d = yf.download("QQQ", start="2014-01-01", end="2026-06-15",
                    auto_adjust=True, progress=False)
    if d is None or d.empty:
        raise RuntimeError("yfinance returned empty for QQQ")
    close = d["Close"] if "Close" in d.columns else d["Adj Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    m = close.resample("ME").last().dropna()
    return m


def fetch_spy_prices() -> pd.Series:
    """SPY for crash gate features."""
    import yfinance as yf
    d = yf.download("SPY", start="2014-01-01", end="2026-06-15",
                    auto_adjust=True, progress=False)
    close = d["Close"] if "Close" in d.columns else d["Adj Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def build_spy_features(spy_daily: pd.Series) -> pd.DataFrame:
    """Compute the SPY features needed by the regime gate, at each month-end."""
    df = pd.DataFrame({"close": spy_daily})
    df["ret_21d"] = df["close"].pct_change(21)
    df["ret_126d"] = df["close"].pct_change(126)
    df["ret_252d"] = df["close"].pct_change(252)
    df["sma200"] = df["close"].rolling(200).mean()
    df["dsma200"] = df["close"] / df["sma200"] - 1.0
    mon = df.resample("ME").last()
    out = pd.DataFrame({
        "spy_ret_21d": mon["ret_21d"],
        "spy_mom_6_1": mon["ret_126d"],
        "spy_mom_12_1": mon["ret_252d"],
        "spy_dsma200": mon["dsma200"],
    })
    return out


def cagr_from_equity(eq: pd.Series, years: float) -> float:
    if years <= 0 or eq.iloc[-1] <= 0:
        return float("nan")
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)


def max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1).min())


def sharpe_annual(monthly_rets: pd.Series) -> float:
    if monthly_rets.std() == 0:
        return float("nan")
    return float(monthly_rets.mean() / monthly_rets.std() * np.sqrt(12))


def run_sim(members_g: dict, mret: pd.DataFrame, mp: pd.DataFrame,
            ml_preds: pd.DataFrame, chronos_at: dict,
            spy_features: pd.DataFrame,
            start: pd.Timestamp, end: pd.Timestamp,
            qqq_monthly: pd.Series) -> dict:
    """Run the v5 simulator on the given window."""
    months = sorted(m for m in mret.index
                    if start <= m <= end and m in members_g)
    cur_picks: list[str] = []
    cur_weights = np.array([])
    last_rebalance = None
    held = 0
    cash = False
    basket_id = 0
    equity = 1.0
    cf = COST_BPS / 1e4

    rows = []
    trades = []
    open_trades = []
    n_cash_months = 0

    for i, m in enumerate(months):
        spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime = regime_tight(spy_now)
        do_reb = (i == 0) or (held >= HOLD_MONTHS) or cash

        if do_reb:
            # Book exits
            if cur_picks and last_rebalance is not None:
                for trade in open_trades:
                    tk = trade["ticker"]
                    entry_px = trade.get("entry_px")
                    exit_px = float(mp.at[m, tk]) if (tk in mp.columns and m in mp.index
                                                       and pd.notna(mp.at[m, tk])) else None
                    trade["exit_date"] = str(m.date())
                    trade["exit_px"] = exit_px
                    trade["return"] = ((exit_px / entry_px - 1)
                                       if (entry_px and exit_px) else None)
                    trade["status"] = "exited"
                    trades.append(trade)
                open_trades = []

            if regime == "crash":
                cur_picks, cur_weights, cash = [], np.array([]), True
                held = 0
                n_cash_months += 1
            else:
                eligible = members_g.get(m, set())
                sub = ml_preds[ml_preds["asof"] == m].copy()
                sub = sub[sub["ticker"].isin(eligible)]
                if len(sub) == 0:
                    cur_picks, cur_weights, cash = [], np.array([]), True
                else:
                    sub["score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2
                    # Chronos filter — cross-sectional rank within NDX cohort
                    if m in chronos_at:
                        cr = chronos_at[m]
                        sub["chr"] = sub["ticker"].map(cr)
                        sub["chr_rk"] = sub["chr"].rank(pct=True)
                        sub = sub[sub["chr_rk"] >= CHRONOS_FILTER_Q]
                    if len(sub) < K_PICKS:
                        cur_picks, cur_weights, cash = [], np.array([]), True
                    else:
                        top = sub.sort_values("score", ascending=False).head(K_PICKS)
                        cur_picks = top["ticker"].tolist()
                        cur_weights = calc_invvol_weights(cur_picks, mret, m,
                                                          cap=CAP_PER_PICK)
                        cash = False
                        last_rebalance = m
                        held = 0
                        basket_id += 1
                        for tk in cur_picks:
                            entry_px = float(mp.at[m, tk]) if (tk in mp.columns
                                                                 and m in mp.index
                                                                 and pd.notna(mp.at[m, tk])) else None
                            open_trades.append({
                                "asof": str(m.date()),
                                "ticker": tk,
                                "entry_date": str(m.date()),
                                "entry_px": entry_px,
                                "basket_id": basket_id,
                                "status": "open",
                            })

        # Compute this month's return
        if cash or not cur_picks:
            ret_m = 0.0
        else:
            rets = []
            for tk, w in zip(cur_picks, cur_weights):
                r = float(mret.at[m, tk]) if (tk in mret.columns and m in mret.index
                                              and pd.notna(mret.at[m, tk])) else 0.0
                rets.append(w * r)
            ret_m = sum(rets)
            if i > 0 and not do_reb:
                pass  # mid-cycle, no cost
            elif i > 0:
                ret_m -= cf  # turnover cost on rebalance
        equity *= (1 + ret_m)
        held += 1
        rows.append({"date": str(m.date()), "ret_m": ret_m, "equity": equity,
                     "regime": regime, "n_picks": len(cur_picks),
                     "picks": ",".join(cur_picks)})

    # Build equity curve dataframe
    eq = pd.DataFrame(rows).set_index("date")
    eq.index = pd.to_datetime(eq.index)
    # QQQ benchmark over the same months
    qqq_eq = qqq_monthly.reindex(eq.index, method="ffill")
    qqq_eq = qqq_eq / qqq_eq.iloc[0]
    eq["qqq_eq"] = qqq_eq.values

    n_months = len(eq)
    years = n_months / 12.0
    strat_cagr = (eq["equity"].iloc[-1]) ** (1 / years) - 1 if years > 0 else float("nan")
    qqq_cagr = (eq["qqq_eq"].iloc[-1]) ** (1 / years) - 1 if years > 0 else float("nan")
    sharpe = sharpe_annual(eq["ret_m"])
    dd = max_dd(eq["equity"])

    return {
        "n_months": n_months,
        "years": years,
        "strat_cagr": strat_cagr,
        "qqq_cagr": qqq_cagr,
        "edge_pp": (strat_cagr - qqq_cagr) * 100,
        "sharpe": sharpe,
        "max_dd": dd,
        "n_cash_months": n_cash_months,
        "n_baskets": basket_id,
        "final_equity_strat": float(eq["equity"].iloc[-1]),
        "final_equity_qqq": float(eq["qqq_eq"].iloc[-1]),
        "equity_curve": eq.reset_index(),
        "trades": trades,
    }


def main():
    QQQ_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("v5 strategy on PIT Nasdaq-100 — honest backtest")
    print("=" * 60)

    # 1. Membership
    print("\n[1/6] Loading PIT NDX membership...")
    mem = pd.read_parquet(QQQ_DIR / "ndx_pit_membership_monthly.parquet")
    members_g = {asof: set(g["ticker"].tolist())
                 for asof, g in mem.groupby("asof")}
    print(f"     → {len(members_g)} months, median pool "
          f"{int(np.median([len(v) for v in members_g.values()]))}")

    # 2. Prices/returns (NDX-restricted panel built by backfill_ndx_tickers.py)
    print("\n[2/6] Loading NDX prices & returns...")
    mp = pd.read_parquet(QQQ_DIR / "ndx_monthly_prices.parquet")
    mret = pd.read_parquet(QQQ_DIR / "ndx_monthly_returns.parquet")
    if not isinstance(mp.index, pd.DatetimeIndex):
        mp.index = pd.to_datetime(mp.index)
    if not isinstance(mret.index, pd.DatetimeIndex):
        mret.index = pd.to_datetime(mret.index)
    print(f"     → prices {mp.shape}, returns {mret.shape}")

    # 3. ML predictions (use existing v2 GBM trained on broader 1833 panel)
    print("\n[3/6] Loading ML preds (GBM 3m+6m) and Chronos predictions...")
    ml_all = pd.read_parquet(CACHE_V2 / "ml_preds_v2.parquet")
    ml_all["asof"] = pd.to_datetime(ml_all["asof"])
    ml_ndx = ml_all[ml_all["ticker"].isin(mem["ticker"].unique())].copy()
    print(f"     → ML preds NDX-restricted: {ml_ndx.shape}, "
          f"asof range {ml_ndx['asof'].min().date()} → {ml_ndx['asof'].max().date()}")

    ch_all = pd.read_parquet(SP500_PIT / "ml_preds_chronos_broader.parquet")
    ch_all["asof"] = pd.to_datetime(ch_all["asof"])
    ch_ndx = ch_all[ch_all["ticker"].isin(mem["ticker"].unique())].copy()
    print(f"     → Chronos preds NDX-restricted: {ch_ndx.shape}, "
          f"asof range {ch_ndx['asof'].min().date()} → {ch_ndx['asof'].max().date()}")
    # Build dict {asof -> {ticker -> chronos_p70_3m}}
    chronos_at: dict = {}
    for asof, g in ch_ndx.groupby("asof"):
        chronos_at[asof] = dict(zip(g["ticker"], g["chronos_p70_3m"]))

    # 4. SPY for the crash gate
    print("\n[4/6] Fetching SPY for crash gate...")
    spy_daily = fetch_spy_prices()
    spy_feat = build_spy_features(spy_daily)
    print(f"     → SPY monthly features: {spy_feat.shape}")

    # 5. QQQ benchmark
    print("\n[5/6] Fetching QQQ benchmark...")
    qqq_monthly = fetch_qqq_prices()
    print(f"     → QQQ monthly closes: {qqq_monthly.shape}, "
          f"{qqq_monthly.index.min().date()} → {qqq_monthly.index.max().date()}")

    # 6. Run sim
    print("\n[6/6] Running v5 strategy on PIT NDX 2015-2026...")
    asofs = sorted(members_g.keys())
    start = max(asofs[0], pd.Timestamp("2015-01-31"))
    end = min(asofs[-1], mret.index.max(), spy_feat.index.max())
    # First ML preds asof must be available; trim to common
    ml_min = ml_ndx["asof"].min()
    start = max(start, ml_min)
    print(f"     → Window: {start.date()} → {end.date()}")

    full = run_sim(members_g, mret, mp, ml_ndx, chronos_at, spy_feat,
                    start, end, qqq_monthly)
    print(f"\n  FULL WINDOW ({full['years']:.1f}y):")
    print(f"    Strategy CAGR: {full['strat_cagr']*100:.2f}%")
    print(f"    QQQ CAGR:      {full['qqq_cagr']*100:.2f}%")
    print(f"    Edge:          {full['edge_pp']:+.2f}pp")
    print(f"    Sharpe:        {full['sharpe']:.2f}")
    print(f"    MaxDD:         {full['max_dd']*100:.1f}%")
    print(f"    Baskets:       {full['n_baskets']}")
    print(f"    Cash months:   {full['n_cash_months']}/{full['n_months']}")
    print(f"    $1 → strat ${full['final_equity_strat']:.2f}, "
          f"QQQ ${full['final_equity_qqq']:.2f}")

    # Walk-forward: run on the post-2015 v5 splits
    # (R3 2014-16, R4 2017-19, R5 2020-22 COVID, R6 2023-25 AI rally, STRICT 2022-25)
    wf_splits = [
        ("R3 2015-16", "2015-01-31", "2016-12-31"),
        ("R4 2017-19", "2017-01-31", "2019-12-31"),
        ("R5 COVID 2020-22", "2020-01-31", "2022-12-31"),
        ("R6 AI 2023-25", "2023-01-31", "2025-12-31"),
        ("STRICT 2022-25", "2022-01-31", "2025-12-31"),
        ("Post-COVID 2020-26", "2020-01-31", "2026-04-30"),
    ]
    print("\nWalk-forward splits (post-2015 only — PIT NDX coverage starts 2015):")
    wf_rows = []
    for name, s, e in wf_splits:
        sp = pd.Timestamp(s)
        ep = min(pd.Timestamp(e), end)
        if sp >= ep:
            continue
        r = run_sim(members_g, mret, mp, ml_ndx, chronos_at, spy_feat,
                    sp, ep, qqq_monthly)
        wf_rows.append({
            "split": name,
            "from": str(sp.date()),
            "to": str(ep.date()),
            "n_months": r["n_months"],
            "strat_cagr_pct": r["strat_cagr"] * 100,
            "qqq_cagr_pct": r["qqq_cagr"] * 100,
            "edge_pp": r["edge_pp"],
            "sharpe": r["sharpe"],
            "max_dd_pct": r["max_dd"] * 100,
            "n_cash_months": r["n_cash_months"],
        })
        print(f"  {name:24s} {sp.date()}→{ep.date()} "
              f"strat={r['strat_cagr']*100:7.2f}% "
              f"qqq={r['qqq_cagr']*100:7.2f}% "
              f"edge={r['edge_pp']:+7.2f}pp "
              f"sharpe={r['sharpe']:.2f} dd={r['max_dd']*100:6.1f}%")

    # Save outputs
    full["equity_curve"].to_csv(QQQ_DIR / "qqq_pit_equity.csv", index=False)
    pd.DataFrame(full["trades"]).to_csv(QQQ_DIR / "qqq_pit_trades.csv", index=False)
    pd.DataFrame(wf_rows).to_csv(QQQ_DIR / "qqq_pit_walkforward.csv", index=False)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "universe": "PIT Nasdaq-100 (jmccarrell/n100tickers, 2015+)",
        "n_unique_tickers": int(mem["ticker"].nunique()),
        "n_panel_resolvable": int(sum(1 for t in mem["ticker"].unique() if t in mp.columns)),
        "median_pool_size": int(np.median([len(v) for v in members_g.values()])),
        "strategy": "v5_ml_3plus6_chronos_p70_k3_invvol_cap0.4_h6_tight",
        "window": {"start": str(start.date()), "end": str(end.date()),
                    "years": full["years"]},
        "full_window": {k: v for k, v in full.items()
                         if k not in ("equity_curve", "trades")},
        "walk_forward": wf_rows,
    }
    with open(QQQ_DIR / "qqq_pit_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved:")
    print(f"  {QQQ_DIR / 'qqq_pit_equity.csv'}")
    print(f"  {QQQ_DIR / 'qqq_pit_trades.csv'}")
    print(f"  {QQQ_DIR / 'qqq_pit_walkforward.csv'}")
    print(f"  {QQQ_DIR / 'qqq_pit_report.json'}")


if __name__ == "__main__":
    main()
