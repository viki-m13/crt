"""Weekly simulator. Mirrors v6 lib_engine.simulate but at weekly cadence.

Honest assumptions:
- Returns: Friday-close to Friday-close, computed from prices_extended.
- Cost: cost_bps applied per round-trip on each ticker that changes
  between consecutive baskets (i.e. only if we sell+buy).
- Crash gate uses SPY weekly features (PIT) recomputed at weekly asofs.
- Crash fallback: 'cash' or 'tlt' (single-ticker allocation).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
WEEKLY_CACHE = Path(__file__).resolve().parent / "cache"
PRICES_PATH = ROOT / "experiments" / "monthly_dca" / "cache" / "prices_extended.parquet"


# ---------------------------------------------------------------------------
# Regime gates (same logic as v6 monthly, but using weekly SPY features)
# ---------------------------------------------------------------------------
def regime_safer(s: dict) -> str:
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    dd52 = s.get("spy_dd_from_52wh", 0.0)
    if r21 <= -0.06 or (r6m <= -0.05 and r21 <= -0.02) or dd52 <= -0.08:
        return "crash"
    if dsma < 0 or r6m < 0:
        return "warning"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def regime_tight(s: dict) -> str:
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


def regime_strict_dd(s: dict) -> str:
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    dd52 = s.get("spy_dd_from_52wh", 0.0)
    rsi14 = s.get("spy_rsi14", 50.0)
    if r21 <= -0.06 or (r6m <= -0.05 and r21 <= -0.02) or dd52 <= -0.10 or (dsma < -0.05 and rsi14 < 45):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


REGIMES = {
    "safer": regime_safer,
    "tight": regime_tight,
    "strict_dd": regime_strict_dd,
}


@dataclass
class WConfig:
    name: str = "weekly_v8"
    regime_gate: str = "safer"
    k_normal: int = 1
    k_recovery: int = 1
    k_bull: int = 1
    weighting: str = "ew"
    hold_weeks: int = 1            # rebalance every N weeks
    cost_bps: float = 10.0          # round-trip per changed ticker
    crash_fallback: str = "tlt"     # 'cash' | 'tlt' | 'spy'
    fallback_ticker: str = "TLT"
    half_cash_warning: bool = False
    cash_yield_yr: float = 0.03


def _build_weekly_returns():
    """Friday-close to Friday-close return panel for all tickers.

    Honest NaN handling: returns are NaN when we don't have two consecutive
    valid weekly closes. We do NOT treat short data gaps (1-2 missing weeks
    inside an otherwise-valid history) as -100% — that's a data hiccup, not
    a real delisting. We forward-fill prices for up to 2 weeks before
    computing returns so genuine delistings still register but data hiccups
    don't artificially destroy a single-pick weekly portfolio.
    """
    px = pd.read_parquet(PRICES_PATH)
    feat = pd.read_parquet(WEEKLY_CACHE / "features_weekly.parquet",
                            columns=["asof"])
    weekly_asofs = pd.DatetimeIndex(sorted(feat["asof"].unique()))
    weekly_px = px.reindex(weekly_asofs).ffill(limit=2)
    weekly_ret = weekly_px.pct_change()
    return weekly_px, weekly_ret


def _load_score_panel(universe: str = "sp500_pit") -> pd.DataFrame:
    preds = pd.read_parquet(WEEKLY_CACHE / "weekly_preds.parquet")
    preds["asof"] = pd.to_datetime(preds["asof"])
    if universe == "sp500_pit":
        mem = pd.read_parquet(WEEKLY_CACHE / "sp500_membership_weekly.parquet")
        mem["asof"] = pd.to_datetime(mem["asof"])
        preds = preds.merge(mem, on=["asof", "ticker"], how="inner")
    return preds


def simulate(cfg: WConfig,
             score_panel: pd.DataFrame,
             weekly_returns: pd.DataFrame,
             spy_features: pd.DataFrame,
             starting_cash: float = 1.0) -> pd.DataFrame:
    cls = REGIMES[cfg.regime_gate]
    cf = cfg.cost_bps / 10000.0
    cash_step = (1 + cfg.cash_yield_yr) ** (1 / 52) - 1 if cfg.cash_yield_yr > 0 else 0.0

    by_asof = {pd.Timestamp(d): g.copy() for d, g in score_panel.groupby("asof")}
    weeks = sorted(by_asof.keys())
    wr_idx = weekly_returns.index

    equity = starting_cash
    cur_picks: list[str] = []
    cur_weights = np.array([])
    held_for = 0
    in_cash = False
    rows = []

    for i, m in enumerate(weeks):
        spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime = cls(spy_now)

        do_reb = (i == 0) or (held_for >= cfg.hold_weeks) or in_cash

        if do_reb:
            if regime == "crash":
                if cfg.crash_fallback in ("tlt", "spy") and cfg.fallback_ticker in weekly_returns.columns:
                    new_picks = [cfg.fallback_ticker]
                    new_weights = np.array([1.0])
                    in_cash = False
                else:
                    new_picks = []
                    new_weights = np.array([])
                    in_cash = True
            else:
                k = {"recovery": cfg.k_recovery,
                     "bull": cfg.k_bull,
                     "warning": cfg.k_normal,
                     "normal": cfg.k_normal}.get(regime, cfg.k_normal)
                sub = by_asof.get(m, pd.DataFrame())
                if len(sub) < k:
                    new_picks = []
                    new_weights = np.array([])
                    in_cash = True
                else:
                    top = sub.sort_values("pred", ascending=False).head(k)
                    new_picks = top["ticker"].tolist()
                    new_weights = np.ones(k) / k
                    in_cash = False
                    if cfg.half_cash_warning and regime == "warning":
                        new_weights = new_weights * 0.5

            # Apply round-trip cost on changed tickers
            changed = set(cur_picks) ^ set(new_picks)
            cost = cf * len(changed) * 1.0  # one round-trip per changed name
            cur_picks = new_picks
            cur_weights = new_weights
            held_for = 0
            equity *= max(0.0, 1 - cost)
        # Apply weekly return on the held basket (next-week return).
        # NaN return = data missing this week; treat as 0% (hold position
        # without P&L update) rather than -100% — see _build_weekly_returns
        # for honest NaN policy. Genuine delisting registers when a ticker's
        # price drops to zero / persistently NaN beyond the ffill window.
        if in_cash or len(cur_picks) == 0:
            ret_w = cash_step
        else:
            pos1 = wr_idx.searchsorted(m)
            if pos1 + 1 >= len(wr_idx):
                ret_w = 0.0
            else:
                next_d = wr_idx[pos1 + 1]
                pick_rets = []
                for tk in cur_picks:
                    if tk in weekly_returns.columns:
                        rr = weekly_returns.at[next_d, tk]
                        pick_rets.append(0.0 if pd.isna(rr) else float(rr))
                    else:
                        pick_rets.append(0.0)
                pick_rets = np.array(pick_rets)
                gross = float(cur_weights.sum())
                ret_w = float((pick_rets * cur_weights).sum()) + (1.0 - gross) * cash_step

        equity *= (1 + ret_w)
        held_for += 1

        rows.append({
            "date": m, "equity": equity, "ret_w": ret_w,
            "regime": regime if not in_cash else "cash",
            "n_picks": len(cur_picks),
            "picks": ",".join(cur_picks),
            "gross": float(cur_weights.sum()) if len(cur_weights) else 0.0,
        })
    return pd.DataFrame(rows)


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


def cagr_weekly(ret: pd.Series) -> float:
    if len(ret) == 0:
        return 0.0
    eq = (1 + ret.fillna(0)).cumprod()
    return float(eq.iloc[-1] ** (52.0 / len(eq)) - 1)


def sharpe_weekly(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(52))


def maxdd_weekly(ret: pd.Series) -> float:
    eq = (1 + ret.fillna(0)).cumprod()
    if len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def evaluate(eq: pd.DataFrame, spy_aln: pd.DataFrame, name: str = "") -> dict:
    ret = eq["ret_w"].astype(float)
    cgr = cagr_weekly(ret)
    sh = sharpe_weekly(ret)
    mdd = maxdd_weekly(ret)
    n_cash = int((eq["regime"] == "cash").sum())
    wf = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)]
        if len(e) == 0:
            continue
        r = e["ret_w"].astype(float)
        spy_e = spy_aln[(spy_aln["date"] >= lo) & (spy_aln["date"] <= hi)]
        sr = spy_e["spy_ret_w"].astype(float)
        wf.append({
            "split": split,
            "cagr": cagr_weekly(r),
            "sharpe": sharpe_weekly(r),
            "max_dd": maxdd_weekly(r),
            "spy_cagr": cagr_weekly(sr),
            "edge_pp": (cagr_weekly(r) - cagr_weekly(sr)) * 100,
        })
    wf = pd.DataFrame(wf)
    spy_full_cagr = cagr_weekly(spy_aln["spy_ret_w"].astype(float))
    return {
        "name": name,
        "cagr_full": float(cgr),
        "spy_cagr_full": float(spy_full_cagr),
        "edge_full_pp": float((cgr - spy_full_cagr) * 100),
        "sharpe": float(sh),
        "max_dd": float(mdd),
        "n_cash": n_cash,
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else 0.0,
        "wf_median_cagr": float(wf["cagr"].median()) if len(wf) else 0.0,
        "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else 0.0,
        "wf_max_cagr": float(wf["cagr"].max()) if len(wf) else 0.0,
        "wf_mean_sharpe": float(wf["sharpe"].mean()) if len(wf) else 0.0,
        "wf_min_sharpe": float(wf["sharpe"].min()) if len(wf) else 0.0,
        "wf_mean_dd": float(wf["max_dd"].mean()) if len(wf) else 0.0,
        "wf_min_dd": float(wf["max_dd"].min()) if len(wf) else 0.0,
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else 0.0,
        "wf_n_pos": int((wf["cagr"] > 0).sum()) if len(wf) else 0,
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else 0,
        "wf_n_splits": int(len(wf)),
    }


def build_spy_aligned(eq: pd.DataFrame, weekly_returns: pd.DataFrame) -> pd.DataFrame:
    """SPY return aligned to next-week (matches the simulator's t->t+1 fill)."""
    eq_dates = pd.DatetimeIndex(eq["date"])
    rows = []
    wr_idx = weekly_returns.index
    for d in eq_dates:
        pos = wr_idx.searchsorted(d)
        if pos + 1 < len(wr_idx) and "SPY" in weekly_returns.columns:
            r = weekly_returns.at[wr_idx[pos + 1], "SPY"]
            rows.append(0.0 if pd.isna(r) else float(r))
        else:
            rows.append(0.0)
    return pd.DataFrame({"date": eq_dates, "spy_ret_w": rows})
