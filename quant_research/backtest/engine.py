"""
Core walk-forward backtest engine.

Design principles:
- PIT-clean: universe at date T uses only PIT membership for T
- No look-ahead: features at T use only data ≤ T close
- Realistic costs: round-trip floor 5 bps + optional ADV-scaled slippage
- Deterministic and cacheable

Usage:
    from backtest.engine import run_wf
    results = run_wf(score_fn, config)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Literal
import pathlib, warnings

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = pathlib.Path(__file__).parents[2]
CACHE = ROOT / "experiments/monthly_dca/cache"
PIT_PATH = CACHE / "v2/sp500_pit/sp500_membership_monthly.parquet"
# Pre-built monthly returns with ME index (calendar month-end)
MONTHLY_RET_PATH = CACHE / "v2/monthly_returns_clean.parquet"
FEAT_DIR = CACHE / "features"


# ── Data loading (cached) ─────────────────────────────────────────────────────

_monthly_returns: pd.DataFrame | None = None
_pit_lookup: dict | None = None
_feat_cache: dict = {}


def _load_monthly_prices() -> pd.DataFrame:
    """Returns monthly returns (ME-indexed)."""
    return _load_monthly_returns()


def _load_monthly_returns() -> pd.DataFrame:
    global _monthly_returns
    if _monthly_returns is None:
        _monthly_returns = pd.read_parquet(MONTHLY_RET_PATH)
    return _monthly_returns


def _nearest_me_pos(mr_index: pd.DatetimeIndex, asof: pd.Timestamp) -> int | None:
    """Map a BME date to the nearest ME position in monthly returns index."""
    pos = mr_index.searchsorted(asof)
    if pos >= len(mr_index):
        pos = len(mr_index) - 1
    # Allow up to 5 calendar days offset between BME and ME
    if abs((mr_index[pos] - asof).days) <= 5:
        return pos
    if pos > 0 and abs((mr_index[pos - 1] - asof).days) <= 5:
        return pos - 1
    return None


def _load_pit() -> dict[pd.Timestamp, set[str]]:
    """Return {month_end: {ticker, ...}} mapping."""
    global _pit_lookup
    if _pit_lookup is None:
        pit = pd.read_parquet(PIT_PATH)
        _pit_lookup = {}
        for asof, grp in pit.groupby("asof"):
            _pit_lookup[pd.Timestamp(asof)] = set(grp["ticker"].tolist())
    return _pit_lookup


def _load_features(asof: pd.Timestamp) -> pd.DataFrame | None:
    key = asof.strftime("%Y-%m-%d")
    if key in _feat_cache:
        return _feat_cache[key]
    path = FEAT_DIR / f"{key}.parquet"
    if not path.exists():
        # Try nearest available
        files = sorted(FEAT_DIR.glob("*.parquet"))
        dates = [pd.Timestamp(f.stem) for f in files]
        idx = np.searchsorted(dates, asof)
        if idx == 0 or idx >= len(dates):
            return None
        # Use the most recent snapshot ≤ asof
        idx = idx - 1
        if dates[idx] > asof:
            return None
        path = files[idx]
    df = pd.read_parquet(path)
    _feat_cache[key] = df
    return df


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    # Universe
    universe: Literal["sp500_pit", "broader"] = "sp500_pit"

    # Selection
    k: int = 5
    weighting: Literal["ew", "invvol"] = "ew"
    vol_col: str = "vol_1y"

    # Costs: round-trip per turnover name (bps)
    cost_bps: float = 5.0

    # Holding: months before forced rebalance (1 = monthly)
    hold_months: int = 1

    # Regime gate: "tight" (v3), "safer" (v8), "always" (no gate)
    regime_gate: str = "tight"

    # Cash yield (annual, credited during cash months)
    cash_yield_yr: float = 0.045

    # Walk-forward windows
    train_years: int = 5   # training window for any ML model
    test_years: int = 3    # each OOS test window
    min_train_years: int = 3

    # Lockbox: last N months sealed
    lockbox_months: int = 24


# ── Regime gate ───────────────────────────────────────────────────────────────

_spy_feat_cache: dict[pd.Timestamp, dict] = {}


def _load_spy_features(asof: pd.Timestamp) -> dict:
    """Extract SPY regime features from the feature snapshot at asof."""
    if asof in _spy_feat_cache:
        return _spy_feat_cache[asof]
    feats = _load_features(asof)
    if feats is None or "SPY" not in feats.index:
        result = {}
    else:
        spy = feats.loc["SPY"]
        dd_pos = float(spy.get("dd_from_52wh", 0.0))
        result = {
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
            "spy_dd_from_52wh": -abs(dd_pos),
        }
    _spy_feat_cache[asof] = result
    return result


def _regime_tight(s: dict) -> str:
    """v3 deployed regime gate — exact replica of v6/lib_engine.py."""
    r21    = s.get("spy_ret_21d", 0.0)
    r6m    = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma   = s.get("spy_dsma200", 0.0)
    mom12  = s.get("spy_mom_12_1", 0.0)
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def _regime_safer(s: dict) -> str:
    """Earlier crash trigger (v8 'safer' gate)."""
    r21   = s.get("spy_ret_21d", 0.0)
    r6m   = s.get("spy_mom_6_1", 0.0)
    dsma  = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    if r21 <= -0.06 or (r6m <= -0.04 and r21 <= -0.02) or (dsma < -0.05 and r21 < 0):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def _is_in_market(asof: pd.Timestamp, gate: str) -> bool:
    """Return True if the regime gate says we should be invested."""
    s = _load_spy_features(asof)
    if not s:
        return True  # default to invested if no data
    if gate == "tight":
        return _regime_tight(s) != "crash"
    if gate == "safer":
        return _regime_safer(s) != "crash"
    if gate == "always":
        return True
    # Default: tight
    return _regime_tight(s) != "crash"


# ── Main backtest ─────────────────────────────────────────────────────────────

def simulate(
    score_fn: Callable[[pd.DataFrame, pd.Timestamp], pd.Series],
    cfg: BacktestConfig,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run a full simulation.

    score_fn(features_df, asof) → pd.Series[ticker → score]
        Higher score = better. Receives PIT-filtered feature snapshot.

    Returns a DataFrame with columns:
        date, ret, spy_ret, in_cash, n_picks, cost, picks
    """
    mr = _load_monthly_returns()
    mr_idx = mr.index
    pit = _load_pit()

    # Date range (use PIT BME dates)
    all_pit_dates = sorted(pit.keys())
    if start_date:
        start_date = pd.Timestamp(start_date)
        all_pit_dates = [d for d in all_pit_dates if d >= start_date]
    if end_date:
        end_date = pd.Timestamp(end_date)
        all_pit_dates = [d for d in all_pit_dates if d <= end_date]

    monthly_cash_yield = (1 + cfg.cash_yield_yr) ** (1 / 12) - 1

    records = []
    last_picks: list[str] = []
    feats_pit_last: pd.DataFrame | None = None
    months_held = 0
    in_cash = False

    for i, asof in enumerate(all_pit_dates):
        # Map asof (BME) to its position in monthly returns (ME)
        pos = _nearest_me_pos(mr_idx, asof)
        if pos is None or pos + 1 >= len(mr_idx):
            continue

        # Return is realized the NEXT month (pos → pos+1)
        next_me = mr_idx[pos + 1]
        spy_ret = float(mr.loc[next_me, "SPY"]) if "SPY" in mr.columns else 0.0

        # Rebalance decision (mirrors v6 lib_engine logic):
        # - First month, OR held long enough, OR currently in cash (trying to re-enter)
        should_rebalance = (i == 0) or (months_held >= cfg.hold_months) or in_cash

        if should_rebalance:
            # Evaluate regime only at rebalance (including when in_cash → re-entry check)
            regime_ok = (cfg.regime_gate == "always") or _is_in_market(asof, cfg.regime_gate)

            if not regime_ok:
                # Go/stay in cash
                in_cash = True
                last_picks = []
                feats_pit_last = None
                months_held = 0
                records.append({
                    "date": asof, "ret": monthly_cash_yield,
                    "spy_ret": spy_ret, "in_cash": True,
                    "n_picks": 0, "cost": 0.0, "picks": [],
                })
                continue

            # Regime OK: attempt to select stocks
            feats = _load_features(asof)
            if feats is None:
                in_cash = True
                records.append({
                    "date": asof, "ret": monthly_cash_yield,
                    "spy_ret": spy_ret, "in_cash": True,
                    "n_picks": 0, "cost": 0.0, "picks": [],
                })
                last_picks = []
                months_held = 0
                continue

            # PIT filter
            pit_members = pit[asof]
            universe_tickers = (
                pit_members if cfg.universe == "sp500_pit"
                else set(feats.index.tolist())
            )
            avail = feats.index.intersection(universe_tickers)
            feats_pit = feats.loc[avail]
            feats_pit_last = feats_pit

            # Score
            scores = score_fn(feats_pit, asof)
            scores = scores.dropna()
            if len(scores) == 0:
                in_cash = True
                records.append({
                    "date": asof, "ret": monthly_cash_yield,
                    "spy_ret": spy_ret, "in_cash": True,
                    "n_picks": 0, "cost": 0.0, "picks": [],
                })
                last_picks = []
                months_held = 0
                continue

            # Top-K
            k = min(cfg.k, len(scores))
            picks = scores.nlargest(k).index.tolist()

            # Cost: turnover vs prior picks
            new_picks = set(picks) - set(last_picks)
            dropped = set(last_picks) - set(picks)
            turnover_names = len(new_picks) + len(dropped)
            cost = (cfg.cost_bps / 10000) * (turnover_names / max(k, 1))

            last_picks = picks
            in_cash = False
            months_held = 0
        else:
            picks = last_picks
            feats_pit = feats_pit_last
            cost = 0.0

        months_held += 1

        # Weights
        weights = _compute_weights(picks, feats_pit, cfg.weighting, cfg.vol_col)

        # Portfolio return: realized at next_me
        port_ret = 0.0
        for t in picks:
            if t in mr.columns:
                raw = mr.loc[next_me, t]
                r = float(raw) if not (raw is None or (isinstance(raw, float) and np.isnan(raw))) else 0.0
            else:
                r = 0.0
            port_ret += weights.get(t, 0.0) * r
        net_ret = port_ret - cost

        records.append({
            "date": asof, "ret": net_ret,
            "spy_ret": spy_ret, "in_cash": False,
            "n_picks": len(picks), "cost": cost, "picks": picks,
        })

    return pd.DataFrame(records)


def _period_return(monthly_ret: pd.DataFrame, ticker: str,
                   from_date: pd.Timestamp, to_date: pd.Timestamp) -> float:
    if ticker not in monthly_ret.columns:
        return 0.0
    s = monthly_ret[ticker]
    mask = (s.index > from_date) & (s.index <= to_date)
    vals = s[mask].dropna()
    if len(vals) == 0:
        return 0.0
    return float(np.prod(1 + vals) - 1)


def _compute_weights(picks: list[str], feats: pd.DataFrame | None,
                     scheme: str, vol_col: str) -> dict[str, float]:
    if not picks:
        return {}
    if scheme == "ew" or feats is None:
        w = 1.0 / len(picks)
        return {t: w for t in picks}
    if scheme == "invvol":
        vols = []
        for t in picks:
            if t in feats.index and vol_col in feats.columns:
                v = float(feats.loc[t, vol_col])
                vols.append(max(v, 0.01))
            else:
                vols.append(0.20)
        inv = [1.0 / v for v in vols]
        total = sum(inv)
        return {t: inv[i] / total for i, t in enumerate(picks)}
    return {t: 1.0 / len(picks) for t in picks}


def _portfolio_return(monthly_ret: pd.DataFrame, picks: list[str],
                      weights: dict[str, float],
                      from_date: pd.Timestamp, to_date: pd.Timestamp) -> float:
    total = 0.0
    for t in picks:
        r = _period_return(monthly_ret, t, from_date, to_date)
        total += weights.get(t, 0.0) * r
    return total


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(sim: pd.DataFrame, label: str = "") -> dict:
    """Compute key performance metrics from simulation output."""
    rets = sim["ret"].fillna(0.0).values
    spy  = sim["spy_ret"].fillna(0.0).values

    n_months = len(rets)
    if n_months < 3:
        return {"label": label, "cagr": 0.0, "spy_cagr": 0.0, "edge_pp": 0.0,
                "sharpe": 0.0, "max_dd": 0.0, "n_months": n_months, "cash_pct": 0.0}

    # CAGR
    total_ret = float(np.prod(1 + rets))
    cagr = float(total_ret ** (12 / n_months) - 1) if total_ret > 0 else -1.0

    spy_total = float(np.prod(1 + spy))
    spy_cagr = float(spy_total ** (12 / n_months) - 1) if spy_total > 0 else -1.0

    # Sharpe (annualized from monthly, no risk-free subtraction)
    mean_r = float(np.mean(rets))
    std_r  = float(np.std(rets, ddof=1)) if len(rets) > 1 else 1.0
    sharpe = mean_r / std_r * np.sqrt(12) if std_r > 1e-8 else 0.0

    # MaxDD
    equity = np.cumprod(1 + np.clip(rets, -0.9999, None))
    roll_max = np.maximum.accumulate(equity)
    dd = equity / roll_max - 1
    max_dd = float(dd.min())

    cash_pct = float(sim["in_cash"].mean())

    return {
        "label": label,
        "cagr": round(cagr, 4),
        "spy_cagr": round(spy_cagr, 4),
        "edge_pp": round((cagr - spy_cagr) * 100, 2),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "n_months": n_months,
        "cash_pct": round(cash_pct, 3),
    }


def run_walk_forward(
    score_fn: Callable,
    cfg: BacktestConfig,
    oos_start: pd.Timestamp | str = "2008-01-31",
    oos_end: pd.Timestamp | str = "2024-01-31",  # exclude lockbox (last 24m)
    window_years: int = 5,
    step_years: int = 2,
) -> dict:
    """Run rolling walk-forward and return aggregate metrics."""
    oos_start = pd.Timestamp(oos_start)
    oos_end   = pd.Timestamp(oos_end)

    splits = []
    t = oos_start
    pit = _load_pit()
    all_dates = sorted(pit.keys())

    while t <= oos_end:
        test_end = min(t + pd.DateOffset(years=window_years), oos_end)
        sim = simulate(score_fn, cfg, start_date=t, end_date=test_end)
        if len(sim) >= 12:  # require at least 12 months for meaningful metrics
            m = compute_metrics(sim, label=f"{t.year}-{test_end.year}")
            splits.append(m)
        t = t + pd.DateOffset(years=step_years)

    if not splits:
        return {}

    # Also compute full OOS
    sim_full = simulate(score_fn, cfg, start_date=oos_start, end_date=oos_end)
    full = compute_metrics(sim_full, label="full_oos")

    cagtrs = [s["cagr"] for s in splits]
    sharpes = [s["sharpe"] for s in splits]

    return {
        "full": full,
        "splits": splits,
        "wf_mean_cagr": round(float(np.mean(cagtrs)), 4),
        "wf_min_cagr":  round(float(np.min(cagtrs)), 4),
        "wf_mean_sharpe": round(float(np.mean(sharpes)), 4),
        "wf_min_sharpe":  round(float(np.min(sharpes)), 4),
        "wf_n_pos": sum(1 for c in cagtrs if c > 0),
        "wf_n_splits": len(splits),
    }
