"""
Core walk-forward backtest engine.

Design:
- Monthly rebalance or hold_months cadence.
- Universe: passed in via score_panel (caller applies PIT filter externally).
- Costs: cost_bps_per_leg per side on turnover.
- Regime gate: simple SPY 10m MA (default) OR tight crash detector (v3 replica).
- Cash earns cash_yield_annual / 12 per month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path

REPO = Path("/home/user/crt")
PRICES_MONTHLY = REPO / "experiments/monthly_dca/cache/v2/monthly_returns_clean.parquet"
PRICES_DAILY   = REPO / "experiments/monthly_dca/cache/prices_extended.parquet"
PIT_SCORES     = REPO / "data/YLOka/pit_panel_with_scores.parquet"
PIT_FULL       = REPO / "data/YLOka/pit_panel_full.parquet"
ML_PREDS_V2    = REPO / "experiments/monthly_dca/cache/v2/ml_preds_v2.parquet"
FEATURES_DIR   = REPO / "experiments/monthly_dca/cache/features"

_monthly_returns: Optional[pd.DataFrame] = None
_spy_features: Optional[pd.DataFrame] = None


def get_monthly_returns() -> pd.DataFrame:
    global _monthly_returns
    if _monthly_returns is None:
        _monthly_returns = pd.read_parquet(PRICES_MONTHLY)
        _monthly_returns.index = pd.to_datetime(_monthly_returns.index)
    return _monthly_returns


def get_spy_features() -> pd.DataFrame:
    """
    Load SPY row from each cached monthly feature snapshot.
    Columns: spy_dsma200, spy_mom_12_1, spy_mom_6_1, spy_ret_21d,
             spy_below_200_streak, spy_dd_from_52wh, spy_vol_12m.
    Index: asof (month-end Timestamp).
    Matches YLOka harness load_spy_features() exactly.
    """
    global _spy_features
    if _spy_features is not None:
        return _spy_features
    rows = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
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
            "spy_dd_from_52wh": float(spy.get("dd_from_52wh", 0.0)),
            "spy_vol_12m": float(spy.get("vol_12m", spy.get("vol_1y", 0.15))),
        })
    _spy_features = pd.DataFrame(rows).set_index("asof")
    return _spy_features


def regime_tight(s: dict) -> str:
    """
    Exact replica of YLOka harness regime_tight.
    Returns 'crash' | 'recovery' | 'bull' | 'normal'.
    """
    r21    = s.get("spy_ret_21d", 0.0)
    r6m    = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma   = s.get("spy_dsma200", 0.0)
    mom12  = s.get("spy_mom_12_1", 0.0)
    if pd.isna(r21):
        return "normal"
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def get_regime_at(asof: pd.Timestamp, spy_feats: pd.DataFrame,
                  gate_type: str = "tight") -> str:
    """Return regime label for the given asof, using the nearest available spy features."""
    idx = spy_feats.index[spy_feats.index <= asof]
    if len(idx) == 0:
        return "normal"
    s = spy_feats.loc[idx[-1]].to_dict()
    if gate_type == "tight":
        return regime_tight(s)
    # simple MA gate (fallback)
    dsma = s.get("spy_dsma200", 0.0)
    return "crash" if dsma < -0.05 else "normal"


def compute_turnover_cost(prev_w: dict, new_w: dict, cost_bps: float = 5.0) -> float:
    all_t = set(prev_w) | set(new_w)
    turnover = sum(abs(new_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in all_t)
    return turnover * (cost_bps / 10000.0)


def load_pit_scores_panel() -> pd.DataFrame:
    df = pd.read_parquet(PIT_SCORES)
    df["asof"] = pd.to_datetime(df["asof"])
    return df


def load_pit_full_panel() -> pd.DataFrame:
    df = pd.read_parquet(PIT_FULL)
    df["asof"] = pd.to_datetime(df["asof"])
    return df


@dataclass
class BacktestConfig:
    name: str
    score_fn: Callable[[pd.DataFrame], pd.Series] = field(default=None)
    K: int = 10
    weighting: str = "ew"           # ew | invvol | score_proportional
    cost_bps_per_leg: float = 5.0
    regime_gate: str = "none"       # "none" | "tight" | "ma10"
    hold_months: int = 1
    cash_yield_annual: float = 0.04
    start: str = "2003-09-30"
    end: str = "2024-04-30"


def run_backtest(cfg: BacktestConfig,
                 score_panel: pd.DataFrame,
                 verbose: bool = False) -> dict:
    rets = get_monthly_returns()
    spy_feats = get_spy_features() if cfg.regime_gate != "none" else None

    start = pd.Timestamp(cfg.start)
    end   = pd.Timestamp(cfg.end)
    asofs = sorted(score_panel["asof"].unique())
    asofs = [a for a in asofs if start <= a <= end]

    monthly_rets = []
    dates = []
    portfolio_weights: dict = {}
    held_for = 0
    cur_w: dict = {}
    in_cash = False

    for i, asof in enumerate(asofs):
        # Regime
        if cfg.regime_gate != "none" and spy_feats is not None:
            regime = get_regime_at(asof, spy_feats, cfg.regime_gate)
        else:
            regime = "normal"

        is_crash = (regime == "crash")
        # Mirror YLOka: do_reb = (i==0) or (held_for >= hold) or PREVIOUS_cash_state
        # Using in_cash from PREVIOUS iteration forces immediate re-entry after crash ends.
        do_rebalance = (i == 0) or (held_for >= cfg.hold_months) or in_cash

        if do_rebalance:
            if is_crash:
                new_w = {}
                in_cash = True
            else:
                grp = score_panel[score_panel["asof"] == asof].copy()
                if cfg.score_fn is None or len(grp) == 0:
                    new_w = {}
                    in_cash = True
                else:
                    scores = cfg.score_fn(grp)
                    scores = scores.dropna()
                    if len(scores) == 0:
                        new_w = {}
                        in_cash = True
                    else:
                        top = scores.nlargest(cfg.K)
                        tickers = top.index.tolist()
                        n = len(tickers)

                        if cfg.weighting == "ew":
                            new_w = {t: 1.0 / n for t in tickers}
                        elif cfg.weighting == "score_proportional":
                            s = top.clip(lower=0)
                            s_sum = s.sum()
                            new_w = ({t: float(s[t] / s_sum) for t in tickers}
                                     if s_sum > 0 else {t: 1.0 / n for t in tickers})
                        elif cfg.weighting == "invvol":
                            gi = grp.set_index("ticker")
                            if "vol_1y" in gi.columns:
                                vols = gi.reindex(tickers)["vol_1y"].clip(lower=0.05)
                                iv = 1.0 / vols
                                iv_sum = iv.sum()
                                new_w = {t: float(iv[t] / iv_sum) for t in tickers}
                            else:
                                new_w = {t: 1.0 / n for t in tickers}
                        else:
                            new_w = {t: 1.0 / n for t in tickers}

                        in_cash = False

            cost = compute_turnover_cost(portfolio_weights, new_w, cfg.cost_bps_per_leg)
            cur_w = new_w
            portfolio_weights = new_w
            held_for = 1
        else:
            cost = 0.0
            held_for += 1

        # Compute next-month gross return.
        # PIT asofs may be last TRADING day; returns indexed at CALENDAR month-end.
        # Snap asof to nearest return date (within 7 days), then take the NEXT month.
        pos = rets.index.searchsorted(asof)
        # Check both pos-1 and pos for proximity
        snap_pos = None
        for candidate_pos in [pos - 1, pos]:
            if 0 <= candidate_pos < len(rets.index):
                if abs((rets.index[candidate_pos] - asof).days) <= 7:
                    snap_pos = candidate_pos
                    break
        if snap_pos is None or snap_pos + 1 >= len(rets.index):
            port_gross = 0.0
        elif in_cash and len(cur_w) == 0:
            port_gross = cfg.cash_yield_annual / 12.0
        else:
            nm = rets.index[snap_pos + 1]
            port_gross = sum(
                w * float(rets.loc[nm, t])
                for t, w in cur_w.items()
                if t in rets.columns and not pd.isna(rets.loc[nm, t])
            )

        monthly_rets.append(port_gross - cost)
        dates.append(asof)

        if verbose:
            picks_str = ",".join(list(cur_w.keys())[:3]) + ("..." if len(cur_w) > 3 else "")
            st = f"[{regime}] {picks_str if not in_cash else 'CASH'}"
            print(f"{asof.date()} {st} gross={port_gross:.3f} cost={cost:.5f}")

    ret_series = pd.Series(monthly_rets, index=pd.DatetimeIndex(dates), name=cfg.name)
    equity = (1 + ret_series).cumprod()
    n = len(ret_series)

    if n < 12:
        return {"name": cfg.name, "cagr": 0.0, "sharpe": 0.0, "maxdd": 0.0,
                "n_months": n, "returns": ret_series, "equity": equity}

    cagr = float(equity.iloc[-1] ** (12.0 / n) - 1)
    excess = ret_series - (cfg.cash_yield_annual / 12.0)
    sharpe = float(excess.mean() / excess.std() * np.sqrt(12)) if excess.std() > 0 else 0.0
    roll_max = equity.cummax()
    maxdd = float((equity / roll_max - 1).min())

    chunk = n // 3
    sub_sharpes = []
    for k in range(3):
        sl = ret_series.iloc[k * chunk:(k + 1) * chunk]
        ex = sl - (cfg.cash_yield_annual / 12.0)
        ss = float(ex.mean() / ex.std() * np.sqrt(12)) if ex.std() > 0 else 0.0
        sub_sharpes.append(round(ss, 3))

    return {
        "name": cfg.name,
        "cagr": cagr,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "n_months": n,
        "sub_sharpes": sub_sharpes,
        "win_rate": float((ret_series > 0).mean()),
        "returns": ret_series,
        "equity": equity,
    }


def summary_str(r: dict) -> str:
    ss = r.get("sub_sharpes", [0, 0, 0])
    return (f"[{r['name']}] CAGR={r['cagr']:.1%} Sharpe={r['sharpe']:.3f} "
            f"MaxDD={r['maxdd']:.1%} Sub={ss} N={r['n_months']}mo")
