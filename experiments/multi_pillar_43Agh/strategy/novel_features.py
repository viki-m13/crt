"""Pillar 3 — Novel mathematical features.

Implements four practical novel features that are not in the existing
67-feature panel. Each is computed strictly from daily price data
≤ asof (PIT-correct).

  1. tail_shape_gpd_60d : GPD shape parameter on left tail of last 252-day returns.
                          Higher = heavier left tail = more failure-prone.
  2. price_persistence_60d : Hurst-like exponent / persistence of price autocorrelation
                              over last 60 days. < 0.5 = mean-reverting, > 0.5 = trending.
  3. spy_corr_60d        : Rolling 60-day correlation of daily returns with SPY.
                          Higher = more market-coupled.
  4. spy_te_proxy        : Granger-causality proxy: regression of stock_return[t]
                          on SPY_return[t-1..t-5] R^2 minus on stock_return[t-1..t-5] R^2.
                          Positive = SPY leads stock; negative = stock leads.

Output: per-asof parquet at experiments/multi_pillar_43Agh/data/novel_features/{asof}.parquet
columns: tail_shape_gpd, price_persistence, spy_corr_60d, spy_te_proxy

Run from repo root.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
FEATURES_DIR = CACHE / "features"
OUT = ROOT / "experiments" / "multi_pillar_43Agh" / "data" / "novel_features"
OUT.mkdir(parents=True, exist_ok=True)


def gpd_shape(neg_rets: np.ndarray, threshold_q: float = 0.10) -> float:
    """Fit GPD to the left tail (returns below the threshold_q quantile, sign-flipped).
    Returns the shape parameter (xi). NaN if insufficient data.
    """
    if len(neg_rets) < 50:
        return np.nan
    losses = -neg_rets[neg_rets < 0]  # positive losses
    if len(losses) < 20:
        return np.nan
    thr = np.quantile(losses, 1 - threshold_q)
    excess = losses[losses > thr] - thr
    if len(excess) < 8:
        return np.nan
    try:
        c, loc, scale = stats.genpareto.fit(excess, floc=0)
        return float(c)
    except Exception:
        return np.nan


def price_persistence(rets: np.ndarray) -> float:
    """Hurst-ish exponent via rescaled-range-like: lag-1 autocorrelation
    transformed to [0, 1]. > 0.5 = trending; < 0.5 = mean-reverting.
    """
    if len(rets) < 20:
        return np.nan
    rets = rets[~np.isnan(rets)]
    if len(rets) < 20:
        return np.nan
    a = float(np.corrcoef(rets[:-1], rets[1:])[0, 1])
    return 0.5 + 0.5 * np.clip(a, -1, 1)


def te_proxy(stock_ret: np.ndarray, spy_ret: np.ndarray, lag: int = 5) -> float:
    """Approximate transfer entropy direction: SPY → stock minus stock → SPY.
    Positive = SPY leads. Computed via R² of lag-1..lag-5 cross-regression.
    """
    if len(stock_ret) < lag + 30 or len(spy_ret) < lag + 30:
        return np.nan
    n = min(len(stock_ret), len(spy_ret))
    s = stock_ret[-n:]
    p = spy_ret[-n:]
    s = s[~np.isnan(s) & ~np.isnan(p)]
    p = p[~np.isnan(stock_ret[-n:]) & ~np.isnan(p)]
    if len(s) < lag + 30:
        return np.nan
    try:
        # SPY(t-1..t-lag) → stock(t)
        X1 = np.column_stack([p[lag - i - 1: -i - 1] for i in range(lag)])
        y = s[lag:]
        if len(X1) != len(y):
            return np.nan
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        pred = X1 @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2_spy_to_stk = 1 - ss_res / ss_tot
        # stock(t-1..t-lag) → SPY(t)
        X2 = np.column_stack([s[lag - i - 1: -i - 1] for i in range(lag)])
        y2 = p[lag:]
        beta2, *_ = np.linalg.lstsq(X2, y2, rcond=None)
        pred2 = X2 @ beta2
        ss_res2 = float(np.sum((y2 - pred2) ** 2))
        ss_tot2 = float(np.sum((y2 - y2.mean()) ** 2)) + 1e-12
        r2_stk_to_spy = 1 - ss_res2 / ss_tot2
        return float(r2_spy_to_stk - r2_stk_to_spy)
    except Exception:
        return np.nan


def compute_at_asof(asof: pd.Timestamp, prices: pd.DataFrame,
                    window: int = 252) -> pd.DataFrame:
    """For all tickers, compute the 4 novel features at asof using last `window`
    trading days of daily prices."""
    if asof not in prices.index:
        # use nearest <= asof
        avail = prices.index[prices.index <= asof]
        if len(avail) == 0:
            return pd.DataFrame()
        asof_use = avail.max()
    else:
        asof_use = asof
    pos = int(prices.index.get_loc(asof_use))
    lo = max(0, pos - window + 1)
    block = prices.iloc[lo: pos + 1]
    if len(block) < 60:
        return pd.DataFrame()
    if "SPY" not in block.columns:
        return pd.DataFrame()
    spy_ret = block["SPY"].pct_change().values

    out_rows = []
    for tk in block.columns:
        s = block[tk].dropna()
        if len(s) < 60:
            continue
        # Use the slice aligned to spy
        rets = block[tk].pct_change().values
        # Hurst
        ph = price_persistence(rets[~np.isnan(rets)][-120:])
        # Tail
        ts = gpd_shape(rets[~np.isnan(rets)])
        # Corr
        # Align
        mask = ~np.isnan(rets) & ~np.isnan(spy_ret)
        n_overlap = int(mask.sum())
        if n_overlap >= 60:
            r1 = rets[mask][-60:]
            r2 = spy_ret[mask][-60:]
            corr = float(np.corrcoef(r1, r2)[0, 1])
        else:
            corr = np.nan
        # TE
        te = te_proxy(rets[mask], spy_ret[mask], lag=5) if mask.sum() >= 80 else np.nan
        out_rows.append({"ticker": tk, "tail_shape_gpd": ts,
                         "price_persistence": ph, "spy_corr_60d": corr,
                         "spy_te_proxy": te})
    return pd.DataFrame(out_rows).set_index("ticker")


def build_panel(asofs: list[pd.Timestamp], force: bool = False) -> None:
    prices = pd.read_parquet(CACHE / "prices_extended.parquet")
    n_built = 0
    for ao in asofs:
        out_f = OUT / f"{pd.Timestamp(ao).date()}.parquet"
        if out_f.exists() and not force:
            continue
        df = compute_at_asof(pd.Timestamp(ao), prices)
        if len(df) > 0:
            df.to_parquet(out_f)
            n_built += 1
    print(f"  built {n_built} novel-feature parquets")


if __name__ == "__main__":
    # Build for all month-end asofs in the existing features panel
    asofs = sorted(pd.Timestamp(p.stem) for p in FEATURES_DIR.glob("*.parquet"))
    print(f"building novel features for {len(asofs)} asofs ...")
    build_panel(asofs)
