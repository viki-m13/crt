"""
Clean walk-forward backtest engine for monthly-rebalance long-only strategy.
Designed for honest research with full Sharpe/CAGR/DD metrics.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Optional

FEAT_DIR = Path("/home/user/crt/experiments/monthly_dca/cache/features")
PRICE_PATH = Path("/home/user/crt/experiments/monthly_dca/cache/prices_extended.parquet")
PIT_PANEL_PATH = Path("/home/user/crt/data/YLOka/pit_panel_full.parquet")
EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}

# ---------------------------------------------------------------------------
# Data loading (cached globally)
# ---------------------------------------------------------------------------
_PRICES: Optional[pd.DataFrame] = None
_MONTHLY: Optional[pd.DataFrame] = None
_FEAT_DATES: Optional[list] = None
_FEAT_CACHE: dict = {}
_SPY_REGIME_CACHE: Optional[pd.DataFrame] = None
_PIT_PANEL: Optional[pd.DataFrame] = None


def get_prices() -> pd.DataFrame:
    global _PRICES
    if _PRICES is None:
        _PRICES = pd.read_parquet(PRICE_PATH)
    return _PRICES


def get_monthly_prices() -> pd.DataFrame:
    global _MONTHLY
    if _MONTHLY is None:
        p = get_prices()
        _MONTHLY = p.resample("ME").last().ffill(limit=5)
    return _MONTHLY


def get_feat_dates() -> list:
    global _FEAT_DATES
    if _FEAT_DATES is None:
        files = sorted(glob.glob(str(FEAT_DIR / "*.parquet")))
        _FEAT_DATES = [pd.Timestamp(Path(f).stem) for f in files]
    return _FEAT_DATES


def load_features(date: pd.Timestamp) -> pd.DataFrame:
    if date not in _FEAT_CACHE:
        path = FEAT_DIR / f"{date.strftime('%Y-%m-%d')}.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df = df[~df.index.isin(EXCLUDE)]
        _FEAT_CACHE[date] = df
    return _FEAT_CACHE[date]


def get_spy_regime_df() -> pd.DataFrame:
    """Precompute SPY regime indicators from price data (point-in-time)."""
    global _SPY_REGIME_CACHE
    if _SPY_REGIME_CACHE is not None:
        return _SPY_REGIME_CACHE

    prices = get_prices()
    if "SPY" not in prices.columns:
        _SPY_REGIME_CACHE = pd.DataFrame()
        return _SPY_REGIME_CACHE

    spy = prices["SPY"].dropna()
    # Rolling indicators
    sma200 = spy.rolling(200, min_periods=100).mean()
    sma50 = spy.rolling(50, min_periods=25).mean()

    # Momentum
    def lag_ret(n):
        return spy.pct_change(n)

    mom_6m = lag_ret(126)   # ~6 months
    mom_3m = lag_ret(63)    # ~3 months
    mom_12m = lag_ret(252)  # ~12 months

    # RSI-14 (approximate)
    delta = spy.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    # Realized vol (21-day)
    vol_21 = spy.pct_change().rolling(21).std() * np.sqrt(252)

    regime_df = pd.DataFrame({
        "d_sma200": (spy - sma200) / sma200,
        "d_sma50": (spy - sma50) / sma50,
        "mom_6_1": mom_6m,
        "mom_3": mom_3m,
        "mom_12_1": mom_12m,
        "rsi_14": rsi,
        "vol_21d": vol_21,
    }, index=spy.index)

    # Resample to month-end (use last trading day of month)
    _SPY_REGIME_CACHE = regime_df.resample("ME").last()
    return _SPY_REGIME_CACHE


def get_spy_stats_at(date: pd.Timestamp) -> dict:
    """Get SPY regime stats at a given date."""
    regime_df = get_spy_regime_df()
    if regime_df.empty:
        return {}
    # Find the nearest date <= date
    idx = regime_df.index.searchsorted(date, side="right") - 1
    if idx < 0:
        return {}
    row = regime_df.iloc[idx]
    return row.to_dict()


def get_pit_panel() -> pd.DataFrame:
    global _PIT_PANEL
    if _PIT_PANEL is None:
        _PIT_PANEL = pd.read_parquet(PIT_PANEL_PATH)
        _PIT_PANEL["asof"] = pd.to_datetime(_PIT_PANEL["asof"])
    return _PIT_PANEL


def get_pit_scores_at(date: pd.Timestamp) -> pd.DataFrame:
    """Get ML predictions from the PIT panel for a given date."""
    panel = get_pit_panel()
    # Find closest available date
    avail_dates = panel["asof"].unique()
    avail_dates = sorted(avail_dates)
    idx = pd.Series(avail_dates).searchsorted(date, side="right") - 1
    if idx < 0:
        return pd.DataFrame()
    closest = avail_dates[idx]
    if abs((closest - date).days) > 35:
        return pd.DataFrame()
    subset = panel[panel["asof"] == closest].set_index("ticker")
    return subset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    monthly_rets: pd.Series,
    rf: float = 0.0,
) -> dict:
    """Compute annualized metrics from monthly return series (as decimals)."""
    if len(monthly_rets) < 6:
        return {}
    excess = monthly_rets - rf / 12.0
    ann_ret = (1 + monthly_rets).prod() ** (12 / len(monthly_rets)) - 1
    ann_vol = monthly_rets.std() * np.sqrt(12)
    sharpe = excess.mean() / monthly_rets.std() * np.sqrt(12) if monthly_rets.std() > 0 else 0.0
    cum = (1 + monthly_rets).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = dd.min()
    downside = monthly_rets[monthly_rets < 0].std()
    sortino = monthly_rets.mean() / downside * np.sqrt(12) if downside > 0 else np.inf
    win_rate = (monthly_rets > 0).mean()

    return {
        "cagr": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "n_months": len(monthly_rets),
        "mean_m": monthly_rets.mean(),
        "std_m": monthly_rets.std(),
    }


# ---------------------------------------------------------------------------
# Regime functions (use SPY price data, not features)
# ---------------------------------------------------------------------------
def make_regime_fn(name: str):
    """Factory for regime functions based on price-derived SPY stats."""

    def regime_fn(date: pd.Timestamp, feats: pd.DataFrame) -> bool:
        stats = get_spy_stats_at(date)
        if not stats:
            return True

        d200 = stats.get("d_sma200", 0.0)
        mom6 = stats.get("mom_6_1", 0.0)
        mom3 = stats.get("mom_3", 0.0)
        rsi = stats.get("rsi_14", 50.0)
        vol = stats.get("vol_21d", 0.15)

        if name == "200ma_loose":
            return bool(d200 > -0.05)
        elif name == "200ma":
            return bool(d200 > -0.02)
        elif name == "200ma_strict":
            return bool(d200 > 0.0 and mom6 > -0.05)
        elif name == "conservative":
            return bool(d200 > 0.0 and (mom6 > 0 or mom3 > 0.02))
        elif name == "aggressive":
            # Avoid only deep bear markets
            return bool(not (d200 < -0.10 and rsi < 35))
        else:
            return True

    return regime_fn


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------
def run_backtest(
    score_fn: Callable[[pd.DataFrame], pd.Series],
    start: str = "2003-01-31",
    end: str = "2021-12-31",
    top_k: int = 10,
    weighting: str = "ew",          # "ew" | "inv_vol" | "vol_score"
    cost_bps: float = 5.0,
    regime_fn=None,                  # fn(date, feats) -> bool  OR  None
    min_price: float = 1.0,         # skip penny stocks
) -> tuple[pd.DataFrame, dict]:
    """
    Run a monthly-rebalance long-only backtest.
    Returns: equity_df (with ret_m column) and metrics dict.
    """
    monthly_px = get_monthly_prices()
    feat_dates = get_feat_dates()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    dates = [d for d in feat_dates if start_ts <= d <= end_ts]
    if len(dates) < 6:
        return pd.DataFrame(), {}

    cost = cost_bps / 10_000.0
    records = []

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]

        feats = load_features(date)
        if feats.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0,
                            "regime_ok": False, "picks": ""})
            continue

        # Regime gate
        if regime_fn is not None and not regime_fn(date, feats):
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0,
                            "regime_ok": False, "picks": ""})
            continue

        # Set date context for ML-based score functions
        try:
            from features.signals import set_date_context
            set_date_context(date)
        except ImportError:
            pass

        scores = score_fn(feats).dropna()
        scores = scores[~scores.index.isin(EXCLUDE)]
        if scores.empty:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0,
                            "regime_ok": True, "picks": ""})
            continue

        top = scores.sort_values(ascending=False).head(top_k)
        tickers = top.index.tolist()

        # Get prices — use nearest available date
        d0_idx = min(monthly_px.index.searchsorted(date, side="right"), len(monthly_px.index) - 1)
        if d0_idx > 0 and monthly_px.index[d0_idx] > date:
            d0_idx -= 1
        d1_idx = min(monthly_px.index.searchsorted(next_date, side="right"), len(monthly_px.index) - 1)
        if d1_idx > 0 and monthly_px.index[d1_idx] > next_date:
            d1_idx -= 1
        d0 = monthly_px.index[d0_idx]
        d1 = monthly_px.index[d1_idx]

        p0_row = monthly_px.loc[d0]
        p1_row = monthly_px.loc[d1]

        # Filter to valid picks
        common = [
            t for t in tickers
            if t in monthly_px.columns
            and np.isfinite(p0_row.get(t, np.nan)) and p0_row.get(t, 0) >= min_price
            and np.isfinite(p1_row.get(t, np.nan)) and p1_row.get(t, 0) >= min_price
        ]
        if not common:
            records.append({"date": date, "ret_m": 0.0, "n_picks": 0,
                            "regime_ok": True, "picks": ""})
            continue

        # Weights
        if weighting == "inv_vol":
            vols = []
            for t in common:
                if t in feats.index and "vol_12m" in feats.columns:
                    v = feats.loc[t, "vol_12m"]
                    vols.append(max(float(v), 0.05) if np.isfinite(v) else 0.20)
                else:
                    vols.append(0.20)
            inv_v = 1.0 / np.array(vols)
            weights = inv_v / inv_v.sum()
        elif weighting == "vol_score":
            vols = []
            for t in common:
                if t in feats.index and "vol_12m" in feats.columns:
                    v = feats.loc[t, "vol_12m"]
                    vols.append(max(float(v), 0.05) if np.isfinite(v) else 0.20)
                else:
                    vols.append(0.20)
            sc = np.array([float(top.get(t, 0.0)) for t in common])
            sc = sc - sc.min() + 1e-6
            inv_v = 1.0 / np.array(vols)
            raw_w = sc * inv_v
            weights = raw_w / raw_w.sum()
        else:
            weights = np.ones(len(common)) / len(common)

        # Portfolio return, net of round-trip costs
        rets = np.array([(p1_row[t] - p0_row[t]) / p0_row[t] for t in common])
        port_ret = float((weights * rets).sum())
        port_ret_net = port_ret - 2 * cost

        records.append({
            "date": date,
            "ret_m": port_ret_net,
            "n_picks": len(common),
            "regime_ok": True,
            "picks": ",".join(common),
        })

    if not records:
        return pd.DataFrame(), {}

    df = pd.DataFrame(records).set_index("date")
    metrics = compute_metrics(df["ret_m"])
    return df, metrics


# ---------------------------------------------------------------------------
# Walk-forward engine (for stateless score functions)
# ---------------------------------------------------------------------------
def walk_forward(
    score_fn: Callable[[pd.DataFrame], pd.Series],
    train_years: int = 4,
    test_years: int = 2,
    full_start: str = "2003-01-31",
    full_end: str = "2021-12-31",
    top_k: int = 10,
    weighting: str = "ew",
    cost_bps: float = 5.0,
    regime_fn=None,
) -> tuple[pd.DataFrame, dict]:
    """
    Walk-forward validation with non-overlapping test windows.
    For stateless score functions, this is just temporal out-of-sample.
    """
    feat_dates = get_feat_dates()
    start_ts = pd.Timestamp(full_start)
    end_ts = pd.Timestamp(full_end)
    dates = [d for d in feat_dates if start_ts <= d <= end_ts]

    all_oos_rets = []
    splits = []
    test_months = test_years * 12
    train_months = train_years * 12
    test_start_idx = train_months

    while test_start_idx < len(dates):
        test_end_idx = min(test_start_idx + test_months, len(dates) - 1)
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]

        oos_df, oos_metrics = run_backtest(
            score_fn=score_fn,
            start=test_start.strftime("%Y-%m-%d"),
            end=test_end.strftime("%Y-%m-%d"),
            top_k=top_k,
            weighting=weighting,
            cost_bps=cost_bps,
            regime_fn=regime_fn,
        )
        if not oos_df.empty and oos_metrics:
            all_oos_rets.append(oos_df["ret_m"])
            splits.append({"test_start": test_start, "test_end": test_end, **oos_metrics})

        test_start_idx += test_months

    if not all_oos_rets:
        return pd.DataFrame(), {}

    combined_rets = pd.concat(all_oos_rets).sort_index()
    combined_metrics = compute_metrics(combined_rets)
    combined_metrics["n_splits"] = len(splits)
    combined_metrics["splits"] = splits
    split_sharpes = [s.get("sharpe", 0.0) for s in splits]
    combined_metrics["sharpe_min"] = min(split_sharpes) if split_sharpes else 0.0
    combined_metrics["sharpe_mean"] = float(np.mean(split_sharpes)) if split_sharpes else 0.0
    combined_metrics["sharpe_std"] = float(np.std(split_sharpes)) if split_sharpes else 0.0

    combined_df = combined_rets.to_frame("ret_m")
    return combined_df, combined_metrics
