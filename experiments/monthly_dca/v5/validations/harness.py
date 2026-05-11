"""Shared validation harness for v5 strategy variants.

Provides:
  - Common data loaders (membership, prices, returns, ml_preds, chronos preds,
    SPY features, IVV-derived sector map).
  - A pluggable simulator: pass a `pick_fn` callback that returns picks
    + weights for a given month, and the harness handles equity, costs,
    rebalancing, regime gate, walk-forward aggregation, DCA-CAGR metric,
    year-by-year.
  - 10-split walk-forward identical to v5 production
    (A1/A2/A3/R1_GFC/R2/R3/R4/R5_COVID/R6_AI/STRICT).

A variant is a function:
    pick_fn(asof, eligible_pool: set, ctx) -> (picks: list[str], weights: np.ndarray)
where `ctx` is a dict with everything the variant might need
(ml_preds_v2, ml_preds_v6, chronos_at_asof, regime, sectors, monthly_returns_so_far, ...).

Returning (picks=[], weights=[]) means "no valid basket — hold cash".
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
SP500_PIT = CACHE / "v2" / "sp500_pit"
V5_DIR = ROOT / "experiments" / "monthly_dca" / "v5"

# v5 winner constants
CHRONOS_FILTER_Q = 0.45
CAP_PER_PICK = 0.40
HOLD_MONTHS = 6
K_PICKS = 3
COST_BPS = 10.0

# 10-split walk-forward (identical to v5 production)
WALK_FORWARD_SPLITS = [
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

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}


# ============================================================
#  Data loaders (single load for all variants)
# ============================================================
@dataclass
class HarnessData:
    members_g: dict          # asof -> set of tickers eligible (PIT SP500)
    mret: pd.DataFrame       # monthly returns, wide (date index, ticker cols)
    mp: pd.DataFrame         # monthly prices, wide
    ml_v2: pd.DataFrame      # ml_preds_v2 (asof, ticker, pred_3m, pred_6m, pred)
    ml_v6: pd.DataFrame      # ml_preds_v6
    ml_pattern: pd.DataFrame # ml_preds_pattern_sim
    ml_ttm: pd.DataFrame     # ml_preds_ttm
    ml_vertical: pd.DataFrame
    chronos: dict            # asof -> {ticker: p70} (cohort-agnostic, raw)
    spy_features: pd.DataFrame  # monthly index, columns include spy_ret_21d, spy_mom_*
    sector_map: dict         # ticker -> GICS sector (best-effort, from IVV)
    asofs: list              # sorted month-ends covering the full range


def load_all() -> HarnessData:
    print("Loading harness data...")
    mem = pd.read_parquet(SP500_PIT / "sp500_membership_monthly.parquet")
    mem["asof"] = pd.to_datetime(mem["asof"])
    members_g = {asof: set(g["ticker"].tolist())
                 for asof, g in mem.groupby("asof")}
    print(f"  membership: {len(members_g)} month-ends, "
          f"{mem['ticker'].nunique()} unique tickers")

    mret = pd.read_parquet(CACHE / "v2" / "monthly_returns_clean.parquet")
    mp = pd.read_parquet(CACHE / "v2" / "monthly_prices_clean.parquet")
    mret.index = pd.to_datetime(mret.index)
    mp.index = pd.to_datetime(mp.index)
    print(f"  prices (calendar ME): {mp.shape}; returns (calendar ME): {mret.shape}")
    # IMPORTANT: mret/mp use CALENDAR month-ends (2003-05-31). The mem,
    # ml_preds, spy_features all use TRADING-DAY month-ends (2003-05-30
    # Friday). Reindex to the trading-day asof grid so every component
    # aligns. Within ±7 days we accept the nearest calendar ME's value.
    trading_asofs = sorted(set(mem["asof"].unique()))
    def _reindex_fuzzy(df: pd.DataFrame, target_idx: list) -> pd.DataFrame:
        out = pd.DataFrame(index=target_idx, columns=df.columns)
        src_idx = df.index.values
        for t in target_idx:
            pos = df.index.searchsorted(t)
            cands = []
            for j in (pos - 1, pos):
                if 0 <= j < len(src_idx):
                    cands.append((j, abs((src_idx[j] - t.to_numpy()) / np.timedelta64(1, "D"))))
            cands.sort(key=lambda x: x[1])
            if cands and cands[0][1] <= 7:
                out.loc[t] = df.iloc[cands[0][0]]
        return out.astype(df.dtypes.to_dict())
    mret = _reindex_fuzzy(mret, trading_asofs)
    mp = _reindex_fuzzy(mp, trading_asofs)
    print(f"  prices (TD ME): {mp.shape}; returns (TD ME): {mret.shape}")

    ml_v2 = pd.read_parquet(CACHE / "v2" / "ml_preds_v2.parquet")
    ml_v2["asof"] = pd.to_datetime(ml_v2["asof"])
    print(f"  ml_v2: {ml_v2.shape}, {ml_v2['asof'].min().date()} → {ml_v2['asof'].max().date()}")

    def _try(name):
        path = SP500_PIT / f"ml_preds_{name}.parquet"
        try:
            df = pd.read_parquet(path)
            df["asof"] = pd.to_datetime(df["asof"])
            print(f"  ml_{name}: {df.shape}")
            return df
        except Exception as e:
            print(f"  ml_{name}: MISSING ({e})")
            return pd.DataFrame()
    ml_v6 = _try("v6")
    ml_pattern = _try("pattern_sim")
    ml_ttm = _try("ttm")
    ml_vertical = _try("vertical")

    ch = pd.read_parquet(SP500_PIT / "ml_preds_chronos.parquet")
    ch["asof"] = pd.to_datetime(ch["asof"])
    chronos = {asof: dict(zip(g["ticker"], g["chronos_p70_3m"]))
               for asof, g in ch.groupby("asof")}
    print(f"  chronos: {ch.shape}, asofs={len(chronos)}")

    # SPY features for crash gate
    spy_features = _build_spy_features(mp)
    print(f"  spy_features: {spy_features.shape}")

    # Sector map (IVV current holdings — best-effort for current S&P 500 names)
    ivv = pd.read_csv(V5_DIR / "ivv_holdings_latest.csv")
    sector_map = dict(zip(ivv["ticker"], ivv["sector"]))
    print(f"  sector_map: {len(sector_map)} tickers")

    asofs = sorted(set(mem["asof"].unique()))
    return HarnessData(members_g, mret, mp, ml_v2, ml_v6, ml_pattern, ml_ttm,
                       ml_vertical, chronos, spy_features, sector_map, asofs)


def _build_spy_features(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Load SPY features identically to v5 production: per-asof feature parquets
    in cache/features/. These contain the DAILY-based 21d return, 200-day SMA
    distance, 12-1 momentum, 6-1 momentum, RSI14 and below-200dma streak."""
    feat_dir = CACHE / "features"
    rows = []
    for f in sorted(feat_dir.glob("*.parquet")):
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


def classify_regime_tight(s: dict) -> str:
    """v5 production regime gate — uses streak + 21d + 6m + mom_12_1 + dsma200.

    Mirrors classify_regime_tight in build_webapp_v5_pit.py."""
    r21 = s.get("spy_ret_21d", 0.0) or 0.0
    r6m = s.get("spy_mom_6_1", 0.0) or 0.0
    streak = s.get("spy_below_200_streak", 0.0) or 0.0
    dsma = s.get("spy_dsma200", 0.0) or 0.0
    if r21 <= -0.08:
        return "crash"
    if r6m <= -0.05 and r21 <= -0.03:
        return "crash"
    if streak >= 40 and dsma >= 0:
        # recovery — keep K_PICKS=3 in v5 winner config
        return "recovery"
    m12 = s.get("spy_mom_12_1", 0.0) or 0.0
    if m12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


# ============================================================
#  Weighting helpers
# ============================================================
def invvol_weights(picks: list[str], monthly_returns: pd.DataFrame,
                    asof: pd.Timestamp, cap: float = CAP_PER_PICK) -> np.ndarray:
    if not picks:
        return np.array([])
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


# ============================================================
#  Simulator (pluggable pick_fn)
# ============================================================
def run_sim(data: HarnessData,
            pick_fn: Callable,
            start: pd.Timestamp = pd.Timestamp("2003-09-30"),
            end: Optional[pd.Timestamp] = None,
            hold_months: int = HOLD_MONTHS,
            cost_bps: float = COST_BPS,
            cash_overlay_fn: Optional[Callable] = None,
            ) -> dict:
    """Run the v5-style simulator with a pluggable pick function.

    Returns dict with:
      equity_log:  list of {date, ret_m, equity, regime, picks, weights, cash}
      trade_log:   list of closed trades
      n_baskets, n_cash_months, n_months
    """
    cf = cost_bps / 1e4
    if end is None:
        end = data.spy_features.index.max()

    asofs = [m for m in data.asofs
             if start <= m <= end
             and m in data.spy_features.index
             and m in data.mret.index
             and m in data.members_g]
    asofs = sorted(asofs)

    cur_picks: list[str] = []
    cur_weights = np.array([])
    held = 0
    cash = False
    equity = 1.0
    basket_id = 0
    n_cash_months = 0
    last_rebalance: Optional[pd.Timestamp] = None

    log = []
    trade_log = []
    open_trades = []

    for i, m in enumerate(asofs):
        spy_now = data.spy_features.loc[m].to_dict() if m in data.spy_features.index else {}
        regime = classify_regime_tight(spy_now)
        do_reb = (i == 0) or (held >= hold_months) or cash

        if do_reb:
            # Book exits
            for tr in open_trades:
                tk = tr["ticker"]
                exit_px = (float(data.mp.at[m, tk])
                            if (tk in data.mp.columns and m in data.mp.index
                                and pd.notna(data.mp.at[m, tk]))
                            else None)
                tr["exit_date"] = str(m.date())
                tr["exit_px"] = exit_px
                tr["return"] = ((exit_px / tr["entry_px"] - 1)
                                if (tr.get("entry_px") and exit_px) else None)
                tr["status"] = "exited"
                trade_log.append(tr)
            open_trades = []

            if regime == "crash":
                cur_picks, cur_weights, cash = [], np.array([]), True
                held = 0
                n_cash_months += 1
            else:
                eligible = data.members_g.get(m, set()) - EXCLUDE
                picks, weights = pick_fn(m, eligible, data, regime)
                if not picks or len(picks) < 1:
                    cur_picks, cur_weights, cash = [], np.array([]), True
                else:
                    cur_picks = list(picks)
                    cur_weights = np.array(weights, dtype=float)
                    if cur_weights.sum() == 0:
                        cur_weights = np.ones(len(cur_picks)) / len(cur_picks)
                    else:
                        cur_weights = cur_weights / cur_weights.sum()
                    cash = False
                    last_rebalance = m
                    held = 0
                    basket_id += 1
                    for tk in cur_picks:
                        ep = (float(data.mp.at[m, tk])
                              if (tk in data.mp.columns and m in data.mp.index
                                  and pd.notna(data.mp.at[m, tk]))
                              else None)
                        open_trades.append({
                            "ticker": tk, "entry_date": str(m.date()),
                            "entry_px": ep, "basket_id": basket_id,
                            "status": "open",
                        })

        # This month's return
        if cash or not cur_picks:
            ret_m = 0.0
        else:
            r = 0.0
            for tk, w in zip(cur_picks, cur_weights):
                rt = (float(data.mret.at[m, tk])
                      if (tk in data.mret.columns and m in data.mret.index
                          and pd.notna(data.mret.at[m, tk]))
                      else 0.0)
                r += w * rt
            ret_m = r
            if i > 0 and do_reb:
                ret_m -= cf

        # Optional cash overlay (e.g. variant 1: SPY sleeve when dispersion low)
        if cash_overlay_fn is not None and not cash and cur_picks:
            overlay_w, overlay_ret = cash_overlay_fn(m, regime, data)
            # blend: (1-overlay_w) * strat + overlay_w * spy
            ret_m = (1 - overlay_w) * ret_m + overlay_w * overlay_ret

        equity *= (1 + ret_m)
        held += 1
        log.append({
            "date": str(m.date()), "ret_m": ret_m, "equity": equity,
            "regime": regime, "cash": cash,
            "picks": ",".join(cur_picks) if cur_picks else "",
            "weights": ",".join(f"{w:.3f}" for w in cur_weights) if len(cur_weights) else "",
            "n_picks": len(cur_picks),
        })

    return {
        "log": log,
        "trades": trade_log,
        "n_baskets": basket_id,
        "n_cash_months": n_cash_months,
        "n_months": len(asofs),
    }


# ============================================================
#  Metrics
# ============================================================
def lump_sum_cagr(log: list) -> float:
    n_months = len(log)
    if n_months < 2:
        return float("nan")
    years = n_months / 12.0
    return log[-1]["equity"] ** (1.0 / years) - 1


def dca_cagr(log: list) -> float:
    """DCA-into-strategy CAGR: deposit $1 each month, value via strategy returns.

    Implementation: at each month-end, deposit $1. Then apply that month's
    return to the entire pot (deposit happens at start, return is realized
    end-of-month). Final value / total deposits is the multiple; CAGR is
    money-weighted (XIRR-style approximation using equal monthly deposits).
    """
    if len(log) < 2:
        return float("nan")
    n = len(log)
    pot = 0.0
    deposits = 0.0
    for r in log:
        pot += 1.0           # deposit at start of month
        deposits += 1.0
        pot *= (1 + r["ret_m"])  # apply month's return
    # Money-weighted IRR via XIRR approximation (Newton on monthly rate)
    # Cashflows: -1 each month-end, then +pot at the final month-end
    # Equivalent: 0 = sum_{i=0..n-1} -1*(1+r)^{(n-1-i)} + pot
    # Solve for r monthly. Use bisection in [-0.5, +0.5] monthly.
    def npv(r_m):
        s = 0.0
        for i in range(n):
            s += -(1 + r_m) ** (n - 1 - i)
        return s + pot
    # NPV(lo) > 0 (positive return needed), NPV(hi) < 0 (too aggressive)
    lo, hi = -0.5, 0.5
    if npv(lo) < 0 or npv(hi) > 0:
        return float("nan")
    for _ in range(80):
        mid = (lo + hi) / 2
        v = npv(mid)
        if abs(v) < 1e-8:
            break
        # NPV decreases as r_m increases; if v > 0, root is to the right
        if v > 0:
            lo = mid
        else:
            hi = mid
    r_m = (lo + hi) / 2
    return (1 + r_m) ** 12 - 1


def sharpe_ann(log: list) -> float:
    rets = np.array([r["ret_m"] for r in log])
    if rets.std() == 0:
        return float("nan")
    return float(rets.mean() / rets.std() * np.sqrt(12))


def max_dd(log: list) -> float:
    eq = np.array([r["equity"] for r in log])
    if len(eq) < 2:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    return float((eq / peaks - 1).min())


def wf_aggregate(log: list, data: HarnessData) -> list[dict]:
    """Re-aggregate the full-window equity log into the 10 walk-forward splits.
    Reports per-split strategy CAGR, SPY CAGR, edge, Sharpe, MaxDD."""
    df = pd.DataFrame(log)
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for name, lo_s, hi_s in WALK_FORWARD_SPLITS:
        lo, hi = pd.Timestamp(lo_s), pd.Timestamp(hi_s)
        sub = df[(df["date"] >= lo) & (df["date"] <= hi)]
        if len(sub) == 0:
            continue
        # Window equity from monthly returns
        ret = sub["ret_m"].astype(float).values
        eq = np.cumprod(1 + ret)
        cagr_v = eq[-1] ** (12 / len(eq)) - 1
        rmean, rstd = ret.mean(), ret.std()
        sh = float(rmean / rstd * np.sqrt(12)) if rstd > 0 else float("nan")
        peaks = np.maximum.accumulate(eq)
        mdd = float((eq / peaks - 1).min())
        # SPY same window
        if "SPY" in data.mret.columns:
            spy = data.mret.loc[lo:hi, "SPY"].dropna().values
            spy_eq = np.cumprod(1 + spy) if len(spy) else np.array([1.0])
            spy_cagr = spy_eq[-1] ** (12 / max(len(spy_eq), 1)) - 1
        else:
            spy_cagr = float("nan")
        rows.append({
            "split": name,
            "from": str(lo.date()),
            "to": str(hi.date()),
            "n_months": int(len(sub)),
            "cagr_pct": cagr_v * 100,
            "spy_cagr_pct": spy_cagr * 100,
            "edge_pp": (cagr_v - spy_cagr) * 100,
            "sharpe": sh,
            "max_dd_pct": mdd * 100,
            "n_cash_months": int((sub["cash"]).sum()) if "cash" in sub else 0,
        })
    return rows


def year_by_year(log: list, data: HarnessData) -> list[dict]:
    df = pd.DataFrame(log)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    out = []
    for yr, g in df.groupby("year"):
        rets = g["ret_m"].astype(float)
        strat_ret = float((1 + rets).prod() - 1)
        if "SPY" in data.mret.columns:
            sm = data.mret.loc[g["date"].min():g["date"].max(), "SPY"].dropna()
            spy_ret = float((1 + sm).prod() - 1) if len(sm) else float("nan")
        else:
            spy_ret = float("nan")
        out.append({
            "year": int(yr),
            "n_months": len(g),
            "strat_ret_pct": strat_ret * 100,
            "spy_ret_pct": spy_ret * 100,
            "edge_pp": (strat_ret - spy_ret) * 100,
        })
    return out


# ============================================================
#  Standard "v5 winner" pick function
# ============================================================
def pick_v5_baseline(asof: pd.Timestamp, eligible: set, data: HarnessData,
                      regime: str) -> tuple[list[str], list[float]]:
    """v5 production picker: GBM 3m+6m mean, gated by Chronos p70 q=0.45 rank,
    top-3 by score, inv-vol cap 0.40."""
    sub = data.ml_v2[data.ml_v2["asof"] == asof].copy()
    sub = sub[sub["ticker"].isin(eligible)]
    if len(sub) == 0:
        return [], []
    sub["score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2
    # Chronos filter
    ch = data.chronos.get(asof, {})
    if ch:
        sub["chr"] = sub["ticker"].map(ch)
        sub["chr_rk"] = sub["chr"].rank(pct=True)
        sub = sub[sub["chr_rk"] >= CHRONOS_FILTER_Q]
    if len(sub) < K_PICKS:
        return [], []
    top = sub.sort_values("score", ascending=False).head(K_PICKS)
    picks = top["ticker"].tolist()
    weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
    return picks, list(weights)


def evaluate(data: HarnessData, pick_fn: Callable, variant_name: str,
              cash_overlay_fn: Optional[Callable] = None,
              **sim_kwargs) -> dict:
    """Run sim, compute headline + WF + year-by-year."""
    sim = run_sim(data, pick_fn, cash_overlay_fn=cash_overlay_fn, **sim_kwargs)
    log = sim["log"]
    cagr = lump_sum_cagr(log)
    dca = dca_cagr(log)
    sh = sharpe_ann(log)
    mdd = max_dd(log)
    # SPY benchmark over same window
    df = pd.DataFrame(log)
    df["date"] = pd.to_datetime(df["date"])
    spy_eq = np.cumprod(1 + data.mret["SPY"].reindex(df["date"]).fillna(0).values)
    spy_cagr_full = spy_eq[-1] ** (12 / len(spy_eq)) - 1 if len(spy_eq) else float("nan")
    # DCA SPY for fair comparison: deposit $1 each month into SPY, compute IRR
    spy_log = [{"ret_m": float(data.mret.at[pd.Timestamp(r["date"]), "SPY"])
                if (pd.Timestamp(r["date"]) in data.mret.index
                    and pd.notna(data.mret.at[pd.Timestamp(r["date"]), "SPY"]))
                else 0.0} for r in log]
    spy_dca = dca_cagr(spy_log)
    wf = wf_aggregate(log, data)
    wf_mean = float(np.mean([r["cagr_pct"] for r in wf])) if wf else float("nan")
    wf_min = float(np.min([r["cagr_pct"] for r in wf])) if wf else float("nan")
    wf_max = float(np.max([r["cagr_pct"] for r in wf])) if wf else float("nan")
    wf_edge_mean = float(np.mean([r["edge_pp"] for r in wf])) if wf else float("nan")
    wf_beat = int(np.sum([1 for r in wf if r["edge_pp"] > 0]))
    wf_pos = int(np.sum([1 for r in wf if r["cagr_pct"] > 0]))
    yby = year_by_year(log, data)
    return {
        "variant": variant_name,
        "n_months": sim["n_months"],
        "n_baskets": sim["n_baskets"],
        "n_cash_months": sim["n_cash_months"],
        "cagr_lump_sum_pct": cagr * 100,
        "cagr_dca_pct": dca * 100,
        "spy_lump_sum_pct": spy_cagr_full * 100,
        "spy_dca_pct": spy_dca * 100,
        "edge_lump_sum_pp": (cagr - spy_cagr_full) * 100,
        "edge_dca_pp": (dca - spy_dca) * 100,
        "sharpe": sh,
        "max_dd_pct": mdd * 100,
        "wf_mean_pct": wf_mean,
        "wf_min_pct": wf_min,
        "wf_max_pct": wf_max,
        "wf_mean_edge_pp": wf_edge_mean,
        "wf_n_beat_spy": wf_beat,
        "wf_n_positive": wf_pos,
        "walk_forward": wf,
        "year_by_year": yby,
        "log": log,
    }
