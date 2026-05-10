"""
v6 — Risk-controlled enhancement of the v3 deployed strategy.

Goal: keep CAGR (>= v3) while improving Sharpe and reducing MaxDD on the same
honest PIT S&P 500 walk-forward. No leakage: at any month-end T we use only
features and signals computed strictly from data with index <= T.

This module exposes a clean simulator wrapper used by all v6 experiments
so the same engine is shared (no dual codepaths, no silent divergence).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
FEATURES_DIR = CACHE / "features"

EXCLUDE_TICKERS = {
    "SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
    "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
    "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD",
}


# ---------------------------------------------------------------------------
# Regime gates
# ---------------------------------------------------------------------------
def regime_tight(s: dict) -> str:
    """v3 deployed regime gate (PIT)."""
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
    """Earlier crash trigger using SPY drawdown from 52w-high."""
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    dd52 = s.get("spy_dd_from_52wh", 0.0)
    rsi14 = s.get("spy_rsi14", 50.0)
    # Trigger crash a touch earlier — DD-from-52w-high or 21d weakness
    if r21 <= -0.06 or (r6m <= -0.05 and r21 <= -0.02) or dd52 <= -0.10 or (dsma < -0.05 and rsi14 < 45):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def regime_safer(s: dict) -> str:
    """Conservative: deeper de-risk + half-cash on warning."""
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    dd52 = s.get("spy_dd_from_52wh", 0.0)
    if r21 <= -0.06 or (r6m <= -0.05 and r21 <= -0.02) or dd52 <= -0.08:
        return "crash"
    if dsma < 0 or r6m < 0:
        return "warning"  # half-exposure
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def regime_faber(s: dict) -> str:
    """Classic Faber-style 200dma + 12m mom rule with mild buffer."""
    dsma = s.get("spy_dsma200", 0.0)
    r21 = s.get("spy_ret_21d", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    if dsma < -0.02:
        return "crash"
    if streak >= 30 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def regime_faber_lite(s: dict) -> str:
    """Faber-200dma + r21 confirmation. Slightly more permissive."""
    dsma = s.get("spy_dsma200", 0.0)
    r21 = s.get("spy_ret_21d", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    if dsma < -0.02 and r21 < 0:
        return "crash"
    if streak >= 30 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def regime_combo(s: dict) -> str:
    """Combine v3 'tight' rule with Faber 200dma backstop."""
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dd52 = s.get("spy_dd_from_52wh", 0.0)
    # v3 tight crash conditions OR SPY below 200dma in stress OR DD > 12% from peak
    if (r21 <= -0.08
            or (r6m <= -0.05 and r21 <= -0.03)
            or (dsma < -0.02 and r21 < 0)
            or (dd52 <= -0.12 and r21 < 0)):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


REGIMES = {
    "tight": regime_tight,
    "strict_dd": regime_strict_dd,
    "safer": regime_safer,
    "faber": regime_faber,
    "faber_lite": regime_faber_lite,
    "combo": regime_combo,
}


# ---------------------------------------------------------------------------
# SPY feature loader (PIT — every value at asof T uses only data <= T)
# ---------------------------------------------------------------------------
def load_spy_features() -> pd.DataFrame:
    rows = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        # NOTE: features.dd_from_52wh is stored as a POSITIVE magnitude
        # (e.g. 0.30 = 30% below 52w high). We convert to a SIGNED value
        # (-0.30) so all downstream logic can use `dd <= -X` consistently.
        dd_pos = float(spy.get("dd_from_52wh", 0.0))
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
            "spy_dd_from_52wh": -abs(dd_pos),
            "spy_vol_1y": float(spy.get("vol_1y", 0.15)),
        })
    return pd.DataFrame(rows).set_index("asof")


def load_score_panel(scorer: str = "ml_3plus6", universe: str = "sp500_pit",
                     attach_pullback: bool = False) -> pd.DataFrame:
    """Return DataFrame with columns asof, ticker, score, vol_1y, [pullback_1y]."""
    ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")
    ml["asof"] = pd.to_datetime(ml["asof"])
    if scorer == "ml_3plus6":
        ml["score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2
    elif scorer == "ml_filter":
        ml["score"] = ml["pred"]
    elif scorer == "ml_h6":
        ml["score"] = ml["pred_6m"]
    elif scorer == "ml_h3":
        ml["score"] = ml["pred_3m"]
    elif scorer == "ml_3plus6plus1":
        ml["score"] = (ml["pred_1m"] + ml["pred_3m"] + ml["pred_6m"]) / 3
    else:
        raise ValueError(scorer)

    # Restrict to PIT S&P 500
    if universe == "sp500_pit":
        mem = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
        mem["asof"] = pd.to_datetime(mem["asof"])
        ml = ml.merge(mem, on=["asof", "ticker"], how="inner")
    elif universe == "broader":
        # use ml as-is (1811 tickers)
        pass
    elif universe == "non_sp500":
        mem = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
        mem["asof"] = pd.to_datetime(mem["asof"])
        mem["in_sp500"] = True
        ml = ml.merge(mem, on=["asof", "ticker"], how="left")
        ml = ml[ml["in_sp500"].isna()].drop(columns=["in_sp500"])
    else:
        raise ValueError(universe)

    ml = ml[~ml["ticker"].isin(EXCLUDE_TICKERS)]
    ml = ml.dropna(subset=["score"])

    # Attach ticker vol_1y and 6m mom (for sizing experiments)
    feat_cache = {}
    asofs = sorted(ml["asof"].unique())

    def _feat(asof, ticker, col):
        if asof not in feat_cache:
            f = FEATURES_DIR / f"{pd.Timestamp(asof).date()}.parquet"
            if not f.exists():
                feat_cache[asof] = None
                return np.nan
            feat_cache[asof] = pd.read_parquet(f)
        df = feat_cache[asof]
        if df is None or ticker not in df.index:
            return np.nan
        try:
            return float(df.loc[ticker, col])
        except KeyError:
            return np.nan

    # Pre-merge vol_1y for sizing (if present in features)
    vol_rows = []
    for asof in asofs:
        f = FEATURES_DIR / f"{pd.Timestamp(asof).date()}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        if "vol_1y" not in df.columns:
            continue
        vr = df["vol_1y"].rename("vol_1y").to_frame()
        vr["asof"] = pd.Timestamp(asof)
        vr["ticker"] = vr.index
        vol_rows.append(vr.reset_index(drop=True))
    if vol_rows:
        vols = pd.concat(vol_rows, ignore_index=True)
        ml = ml.merge(vols, on=["asof", "ticker"], how="left")
    else:
        ml["vol_1y"] = np.nan

    extras = []
    if attach_pullback:
        extras.append("pullback_1y")
        extras.append("mom_12_1")
        extras.append("trend_health_5y")
        feat_rows = []
        for asof in asofs:
            f = FEATURES_DIR / f"{pd.Timestamp(asof).date()}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f)
            cols = [c for c in ("pullback_1y", "mom_12_1", "trend_health_5y") if c in df.columns]
            if not cols:
                continue
            pr = df[cols].copy()
            pr["asof"] = pd.Timestamp(asof)
            pr["ticker"] = pr.index
            feat_rows.append(pr.reset_index(drop=True))
        if feat_rows:
            extras_df = pd.concat(feat_rows, ignore_index=True)
            ml = ml.merge(extras_df, on=["asof", "ticker"], how="left")

    # Cross-sectional vol rank (used for vol_penalty)
    if "vol_1y" in ml.columns:
        ml["vol_rank"] = ml.groupby("asof")["vol_1y"].rank(pct=True)
        extras.append("vol_rank")
    return ml[["asof", "ticker", "score", "vol_1y"] + extras]


# ---------------------------------------------------------------------------
# Configuration & simulator
# ---------------------------------------------------------------------------
@dataclass
class V6Config:
    name: str = "v3_baseline"
    scorer: str = "ml_3plus6"
    universe: str = "sp500_pit"
    regime_gate: str = "tight"
    k_normal: int = 3
    k_recovery: int = 3
    k_bull: int = 3
    weighting: str = "ew"            # ew | invvol | conv | softmax
    hold_months: int = 6
    cost_bps: float = 10.0
    cap_per_pick: float = 1.0
    # New v6 risk controls
    vol_target_yr: float = 0.0       # 0.0 = off; otherwise scale gross by min(target/portvol_est, 1)
    half_cash_warning: bool = False  # if regime "warning", deploy half capital
    cash_yield_yr: float = 0.0       # bills yield while in cash (annualised)
    drawdown_de_risk: float = 0.0    # if >0, halve gross when running dd worse than this
    # Smoothing the regime score: count of consecutive crash signals required
    crash_persist: int = 1           # 1 = current behaviour
    # Trailing stop on portfolio: if running dd from peak <= -X, go to cash at next month-end
    trailing_stop: float = 0.0       # 0 disabled
    # SPY drawdown-based continuous gross scaling: when SPY dd_from_52wh <= -X,
    # scale gross by 1 + dd_x/X*0.5 (clamped to [0.5, 1.0])
    spy_dd_scale: float = 0.0        # 0 disabled
    spy_dd_floor: float = 0.5        # min gross when spy_dd_scale active
    # Sticky cash: after a trailing-stop or crash, require N consecutive normal months to re-enter
    cash_sticky: int = 0             # 0 = re-enter immediately when regime clears
    # Score-blend with vol: penalise high-vol picks at score level
    vol_penalty: float = 0.0         # 0 disabled; otherwise score - vol_penalty * vol_xs
    # Apply gross scaling MONTHLY (between rebalances) — picks unchanged but
    # exposure adjusts to fresh SPY conditions every month.
    monthly_exposure: bool = False
    # Filter picks whose own pullback_1y is <= -X (default 0 = off).
    # Useful to avoid 'falling knife' picks in 2008-style crashes.
    pullback_filter: float = 0.0
    # Reset trailing-stop peak on re-entry (correct vs sticky)
    ts_reset_on_reentry: bool = True
    # Fallback during crash: 'cash' | 'spy' | 'tlt' (treasuries via long bond proxy)
    # When in crash, instead of cash, allocate to a fallback ticker.
    crash_fallback: str = "cash"
    fallback_ticker: str = "SPY"
    # Pick-momentum filter: require pick's mom_12_1 >= -X (cuts death-spiral picks).
    # 0.0 = off. Practical defaults 0.30 (drop picks down >30% on 12m mom).
    min_pick_mom: float = 0.0
    # Smart re-entry from crash: require these conditions before deploying again
    smart_reentry: bool = False  # require dsma200 > -0.02 AND ret_21d > 0
    # Quality blend: post-multiply score by quality factor (trend_health rank
    # after winsor). 0.0 = pure ML score; 1.0 = quality-only weighted.
    quality_blend: float = 0.0


def _nearest_pos(idx: pd.DatetimeIndex, target: pd.Timestamp, tol_days: int = 7) -> Optional[int]:
    pos = idx.searchsorted(target)
    cands = []
    for j in (pos - 1, pos):
        if 0 <= j < len(idx):
            cands.append((j, abs((idx[j] - target).days)))
    cands.sort(key=lambda x: x[1])
    if cands and cands[0][1] <= tol_days:
        return cands[0][0]
    return None


def simulate(cfg: V6Config,
             score_panel: pd.DataFrame,
             monthly_returns: pd.DataFrame,
             spy_features: pd.DataFrame,
             starting_cash: float = 1.0) -> pd.DataFrame:
    """Pure-cash compounding simulator with v6 risk controls."""
    cls = REGIMES[cfg.regime_gate]
    cf = cfg.cost_bps / 10000.0
    cash_step = (1 + cfg.cash_yield_yr) ** (1 / 12) - 1 if cfg.cash_yield_yr > 0 else 0.0

    by_asof = {pd.Timestamp(d): g.copy() for d, g in score_panel.groupby("asof")}
    months = sorted(by_asof.keys())
    mr_idx = monthly_returns.index

    equity = starting_cash
    cur_picks: list[str] = []
    cur_unscaled_weights = np.array([])  # raw EW/invvol weights (before gross scaling)
    cur_weights = np.array([])           # actually applied weights (post-scale)
    held_for = 0
    in_cash = False
    crash_streak = 0
    normal_streak_after_cash = 0
    peak_equity = equity
    rows = []

    for i, m in enumerate(months):
        spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime = cls(spy_now)
        if regime == "crash":
            crash_streak += 1
            normal_streak_after_cash = 0
        else:
            crash_streak = 0
            if in_cash:
                normal_streak_after_cash += 1

        # Effective regime — require persistence
        eff_regime = regime
        if regime == "crash" and crash_streak < cfg.crash_persist:
            eff_regime = "normal"  # don't act on crash until persistent

        # Sticky cash: must wait N normal months before re-entering
        sticky_block = (in_cash and cfg.cash_sticky > 0
                        and normal_streak_after_cash < cfg.cash_sticky
                        and eff_regime != "crash")
        # Smart re-entry: require dsma200 > -0.02 AND ret_21d > 0
        if cfg.smart_reentry and in_cash and eff_regime != "crash":
            d2 = float(spy_now.get("spy_dsma200", 0.0))
            r21 = float(spy_now.get("spy_ret_21d", 0.0))
            if d2 < -0.02 or r21 < 0:
                sticky_block = True

        # Trailing stop: portfolio drawdown from running peak
        running_dd = (equity / peak_equity) - 1.0 if peak_equity > 0 else 0.0
        ts_triggered = (cfg.trailing_stop > 0 and running_dd <= -cfg.trailing_stop)

        do_reb = (i == 0) or (held_for >= cfg.hold_months) or in_cash or ts_triggered

        if do_reb:
            if eff_regime == "crash" or ts_triggered or sticky_block:
                if cfg.crash_fallback in ("spy", "tlt") and cfg.fallback_ticker in monthly_returns.columns:
                    # Allocate 100% to fallback ticker — diversified beta vs cash
                    cur_picks = [cfg.fallback_ticker]
                    cur_unscaled_weights = np.array([1.0])
                    cur_weights = np.array([1.0])
                    in_cash = False
                    gross = 1.0
                    held_for = 0
                    if cfg.ts_reset_on_reentry and ts_triggered:
                        peak_equity = equity
                else:
                    cur_picks, cur_weights, cur_unscaled_weights = [], np.array([]), np.array([])
                    in_cash = True
                    held_for = 0
                    gross = 0.0
                    if cfg.ts_reset_on_reentry and ts_triggered:
                        peak_equity = equity  # reset trailing-stop peak so re-entry can occur
            else:
                # leaving cash for active position — reset sticky counter & ts peak
                normal_streak_after_cash = 0
                if cfg.ts_reset_on_reentry and in_cash:
                    peak_equity = equity
                k = {
                    "recovery": cfg.k_recovery,
                    "bull": cfg.k_bull,
                    "normal": cfg.k_normal,
                    "warning": cfg.k_normal,
                }[eff_regime]
                sub = by_asof.get(m, pd.DataFrame())
                # Optional pullback filter: drop picks deeply below 1Y high
                if cfg.pullback_filter > 0 and "pullback_1y" in sub.columns:
                    sub = sub[(sub["pullback_1y"].isna()) | (sub["pullback_1y"] >= -cfg.pullback_filter)]
                # Optional mom filter: drop picks whose 12m mom is too negative
                if cfg.min_pick_mom > 0 and "mom_12_1" in sub.columns:
                    sub = sub[(sub["mom_12_1"].isna()) | (sub["mom_12_1"] >= -cfg.min_pick_mom)]
                # Optional vol penalty: blend score with negative vol_rank
                if cfg.vol_penalty > 0 and "vol_rank" in sub.columns:
                    sub = sub.copy()
                    score_z = (sub["score"] - sub["score"].mean()) / max(sub["score"].std(), 1e-9)
                    sub["score"] = score_z - cfg.vol_penalty * (sub["vol_rank"] - 0.5) * 2
                # Optional quality blend: multiply score by trend_health_5y rank
                if cfg.quality_blend > 0 and "trend_health_5y" in sub.columns:
                    sub = sub.copy()
                    th = sub["trend_health_5y"].fillna(sub["trend_health_5y"].median())
                    th_rank = th.rank(pct=True)
                    s = (sub["score"] - sub["score"].min()) + 1e-9
                    sub["score"] = s * (1 - cfg.quality_blend) + s * th_rank * cfg.quality_blend
                if len(sub) < k:
                    cur_picks, cur_weights, cur_unscaled_weights = [], np.array([]), np.array([])
                    in_cash = True
                    gross = 0.0
                else:
                    top = sub.sort_values("score", ascending=False).head(k)
                    cur_picks = top["ticker"].tolist()
                    if cfg.weighting == "ew":
                        w = np.ones(k) / k
                    elif cfg.weighting == "invvol":
                        vv = top["vol_1y"].values
                        vv = np.where(np.isnan(vv) | (vv <= 0), 0.4, vv)
                        invv = 1.0 / vv
                        w = invv / invv.sum()
                    elif cfg.weighting == "conv":
                        s = top["score"].values
                        shifted = s - s.min() + 1e-6
                        w = shifted / shifted.sum()
                    elif cfg.weighting == "softmax":
                        s = top["score"].values
                        ss = (s - s.mean()) / max(s.std(), 1e-9)
                        ws = np.exp(2.0 * ss)
                        w = ws / ws.sum()
                    else:
                        w = np.ones(k) / k
                    if cfg.cap_per_pick < 1.0:
                        w = np.minimum(w, cfg.cap_per_pick)
                        w = w / w.sum()
                    cur_unscaled_weights = w.copy()
                    cur_weights = w
                    in_cash = False

                    # Volatility targeting on the basket
                    gross = 1.0
                    if cfg.vol_target_yr > 0:
                        vv = top["vol_1y"].values
                        vv = np.where(np.isnan(vv) | (vv <= 0), 0.4, vv)
                        # Rough basket vol (assume avg correlation 0.5 across picks)
                        n = len(vv)
                        avg_corr = 0.5
                        # var(p) = sum w_i^2 v_i^2 + sum_{i!=j} w_i w_j corr v_i v_j
                        var_diag = float(np.sum((w * vv) ** 2))
                        var_cross = 0.0
                        for ii in range(n):
                            for jj in range(n):
                                if ii != jj:
                                    var_cross += w[ii] * w[jj] * avg_corr * vv[ii] * vv[jj]
                        port_vol = float(np.sqrt(max(var_diag + var_cross, 1e-9)))
                        gross = float(min(cfg.vol_target_yr / port_vol, 1.0))
                    if cfg.half_cash_warning and eff_regime == "warning":
                        gross *= 0.5
                    if cfg.drawdown_de_risk > 0 and running_dd <= -cfg.drawdown_de_risk:
                        gross *= 0.5
                    # SPY-DD-based continuous scaling
                    if cfg.spy_dd_scale > 0:
                        dd52 = float(spy_now.get("spy_dd_from_52wh", 0.0))
                        if dd52 < 0:
                            f = max(cfg.spy_dd_floor, 1.0 + (dd52 / cfg.spy_dd_scale) * (1.0 - cfg.spy_dd_floor))
                            gross *= f
                    cur_weights = cur_unscaled_weights * gross
            held_for = 0
        else:
            # Not a rebalance month, but monthly_exposure may adjust gross.
            if cfg.monthly_exposure and not in_cash and len(cur_unscaled_weights) > 0:
                gross = 1.0
                if cfg.spy_dd_scale > 0:
                    dd52 = float(spy_now.get("spy_dd_from_52wh", 0.0))
                    if dd52 < 0:
                        f = max(cfg.spy_dd_floor, 1.0 + (dd52 / cfg.spy_dd_scale) * (1.0 - cfg.spy_dd_floor))
                        gross *= f
                if cfg.drawdown_de_risk > 0 and running_dd <= -cfg.drawdown_de_risk:
                    gross *= 0.5
                cur_weights = cur_unscaled_weights * gross

        # Apply month return on the held basket
        if in_cash or len(cur_picks) == 0:
            ret_m = cash_step  # earn cash yield only
        else:
            pos1 = _nearest_pos(mr_idx, m)
            if pos1 is None or pos1 + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[pos1 + 1]
                pick_rets = []
                for tk in cur_picks:
                    if tk in monthly_returns.columns:
                        rr = monthly_returns.at[next_d, tk]
                        pick_rets.append(-1.0 if pd.isna(rr) else float(rr))
                    else:
                        pick_rets.append(-1.0)
                pick_rets = np.array(pick_rets)
                gross_used = float(cur_weights.sum())
                # Equity ret = sum(w_i * r_i) on risk + (1-gross)*cash
                ret_m = float((pick_rets * cur_weights).sum()) + (1.0 - gross_used) * cash_step

        if not in_cash and len(cur_picks) > 0:
            if do_reb:
                equity *= (1 + ret_m) * (1 - cf * float(cur_weights.sum()))
            else:
                equity *= (1 + ret_m)
        else:
            equity *= (1 + ret_m)
        held_for += 1
        peak_equity = max(peak_equity, equity)

        rows.append({
            "date": m, "equity": equity, "ret_m": ret_m,
            "regime": eff_regime if not in_cash else "cash",
            "n_picks": len(cur_picks),
            "gross": float(cur_weights.sum()) if len(cur_weights) else 0.0,
            "picks": ",".join(cur_picks),
            "weights_csv": ",".join(f"{w:.4f}" for w in cur_weights) if len(cur_weights) else "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Walk-forward + metrics
# ---------------------------------------------------------------------------
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


def cagr_monthly(ret: pd.Series) -> float:
    if len(ret) == 0:
        return 0.0
    eq = (1 + ret.fillna(0)).cumprod()
    return float(eq.iloc[-1] ** (12.0 / len(eq)) - 1)


def sharpe_monthly(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(12))


def maxdd_monthly(ret: pd.Series) -> float:
    eq = (1 + ret.fillna(0)).cumprod()
    if len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def evaluate(eq: pd.DataFrame, spy_aligned: pd.DataFrame, name: str = "") -> dict:
    ret = eq["ret_m"].astype(float)
    cgr = cagr_monthly(ret)
    sh = sharpe_monthly(ret)
    mdd = maxdd_monthly(ret)
    n_cash = int((eq["regime"] == "cash").sum())

    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)]
        if len(e) == 0:
            continue
        r = e["ret_m"].astype(float)
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        sr = spy["spy_ret_m"].astype(float)
        wf_rows.append({
            "split": split,
            "cagr": cagr_monthly(r),
            "sharpe": sharpe_monthly(r),
            "max_dd": maxdd_monthly(r),
            "spy_cagr": cagr_monthly(sr),
            "edge_pp": (cagr_monthly(r) - cagr_monthly(sr)) * 100,
        })
    wf = pd.DataFrame(wf_rows)
    spy_full_cagr = cagr_monthly(spy_aligned["spy_ret_m"].astype(float))
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
        "wf_min_dd": float(wf["max_dd"].min()) if len(wf) else 0.0,  # min => deepest DD
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else 0.0,
        "wf_n_pos": int((wf["cagr"] > 0).sum()) if len(wf) else 0,
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else 0,
        "wf_n_splits": int(len(wf)),
    }


def build_spy_aligned(eq: pd.DataFrame, monthly_returns: pd.DataFrame) -> pd.DataFrame:
    eq_dates = pd.DatetimeIndex(eq["date"])
    next_month = eq_dates + pd.offsets.MonthEnd(1)
    return pd.DataFrame({
        "date": eq_dates,
        "spy_ret_m": [
            float(monthly_returns["SPY"].loc[nxt]) if nxt in monthly_returns["SPY"].index else 0.0
            for nxt in next_month
        ],
    })
