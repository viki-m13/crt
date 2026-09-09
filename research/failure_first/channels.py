"""Seven independent ways a "this stock will be higher" call goes wrong.

The premise being tested: instead of modelling P(rises) directly, model each
distinct way the call FAILS, and buy only when every channel is quiet. The
appeal is that a conjunction can reach selectivity a single classifier cannot
— our earlier model never emitted a probability above 0.70 at any horizon,
because one model averaging over all risks has nowhere to put extreme
confidence.

The channels, each a risk score in [0,1] where 1 is dangerous:

  market      the whole market falls and takes everything with it. The
              dominant channel, and the one that makes the other six
              correlated whether we like it or not.
  trend       the stock itself is in a downtrend — below or under a falling
              long average, deep in its own drawdown.
  volatility  dispersion alone makes a down outcome likely. High vol widens
              both tails, and the left one is what ends the call.
  tail        gap risk: this name has a recent history of violent single-day
              moves, so a quiet average hides a fat left tail.
  stretch     overextension. A name far above its own trend in volatility
              units has a mean-reversion problem, not a momentum one.
  liquidity   thin dollar volume and a low price. The exit is the risk here,
              and it is invisible in a close-only backtest.
  distress    proximity to the lows, sustained underperformance — the shape
              names have before they stop trading altogether. This is the
              channel the delisted 4,725 tickers exist to inform.

EVERY SCORE IS A CROSS-SECTIONAL RANK COMPUTED WITHIN A SINGLE DATE, from
data available on that date. Ranking within a date is what makes the channels
comparable across regimes without a lookback threshold fitted on the future.
The one exception is `market`, which is a single time series and therefore
uses an EXPANDING-window percentile — only its own past.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ("market", "trend", "volatility", "tail", "stretch",
            "liquidity", "distress")


def _rank(a: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per row, NaN-safe. 0 = lowest value."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    for i in range(a.shape[0]):
        row = a[i]
        ok = np.isfinite(row)
        m = int(ok.sum())
        if m < 20:
            continue
        order = np.argsort(np.argsort(row[ok]))
        out[i, ok] = order / max(m - 1, 1)
    return out


def _expanding_pct(s: pd.Series, min_periods: int = 500) -> np.ndarray:
    """Percentile of each value within its OWN past only. No look-ahead."""
    return s.expanding(min_periods).apply(
        lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan,
        raw=True).to_numpy(dtype=np.float32)


def build(px: pd.DataFrame, vol: pd.DataFrame | None = None) -> dict:
    """Point-in-time risk score per channel, shape (bars, tickers)."""
    arr = px.to_numpy(dtype=np.float32)
    lp = np.log(np.where(arr > 0, arr, np.nan)).astype(np.float32)
    F = pd.DataFrame(lp, index=px.index, columns=px.columns)
    r1 = F.diff()
    out: dict[str, np.ndarray] = {}

    sma200 = F.rolling(200, min_periods=150).mean()
    hi252 = F.rolling(252, min_periods=120).max()
    lo252 = F.rolling(252, min_periods=120).min()
    v63 = r1.rolling(63, min_periods=40).std() * np.sqrt(252)

    # --- market: one series, expanding percentile of its own history ------
    mret = r1.mean(axis=1)
    mlvl = mret.cumsum()
    m_gap = (mlvl - mlvl.rolling(200, min_periods=150).mean())
    m_vol = mret.rolling(63, min_periods=40).std() * np.sqrt(252)
    m_dd = mlvl - mlvl.rolling(252, min_periods=120).max()
    breadth = F.gt(sma200).mean(axis=1)
    risk_m = (_expanding_pct(-m_gap) + _expanding_pct(m_vol)
              + _expanding_pct(-m_dd) + _expanding_pct(-breadth)) / 4.0
    out["market"] = np.repeat(risk_m[:, None], arr.shape[1], axis=1)

    # --- the six cross-sectional channels ---------------------------------
    sma_slope = sma200.diff(21)
    out["trend"] = _rank(-((F - sma200) + sma_slope).to_numpy(np.float32))
    out["volatility"] = _rank(v63.to_numpy(np.float32))
    worst1 = r1.rolling(252, min_periods=120).min()
    worst5 = (F - F.shift(5)).rolling(252, min_periods=120).min()
    out["tail"] = _rank(-(worst1 + worst5).to_numpy(np.float32))
    out["stretch"] = _rank(((F - sma200) / v63.replace(0, np.nan)
                            ).to_numpy(np.float32))

    if vol is not None:
        dv = (pd.DataFrame(vol.to_numpy(dtype=np.float32),
                           index=px.index, columns=px.columns) * px)
        adv = dv.rolling(63, min_periods=40).median()
        out["liquidity"] = _rank(-(np.log(adv.clip(lower=1))
                                   + np.log(px.clip(lower=0.01))
                                   ).to_numpy(np.float32))
    else:
        out["liquidity"] = _rank(-np.log(np.clip(arr, 0.01, None)))

    rng = (hi252 - lo252).replace(0, np.nan)
    pos = (F - lo252) / rng                       # 0 = at the 52w low
    rel = (F - F.shift(252)).sub((F - F.shift(252)).mean(axis=1), axis=0)
    out["distress"] = _rank(-(pos + rel).to_numpy(np.float32))
    return out


def gate(scores: dict, q: float) -> np.ndarray:
    """BUY where EVERY channel is below its q-th safest level. The conjunction."""
    m = np.ones(next(iter(scores.values())).shape, dtype=bool)
    for c in CHANNELS:
        s = scores[c]
        m &= np.isfinite(s) & (s <= q)
    return m


def worst_channel(scores: dict) -> np.ndarray:
    """The max risk across channels — the conjunction expressed as a score.

    Thresholding this at q must be exactly `gate(scores, q)`, which makes it
    the right object to compare against a single channel at equal coverage.

    It uses plain `max`, not `nanmax`, and that is the whole point: nanmax
    ignores missing channels, so a name with an unmeasurable risk would come
    back as safe as the rest of its channels allow — a free pass through the
    exact gate it should fail. `gate` requires every channel finite, and one
    NaN here must therefore poison the result. This is the same defect that
    once produced 110 phantom muni signals on a calm day.
    """
    st = np.stack([scores[c] for c in CHANNELS])
    return np.max(st, axis=0)


def orthogonalise(scores: dict, sample_every: int = 7) -> dict:
    """Re-express each channel as the part of it the others do not explain.

    The premise of the failure-first idea is that the channels are separate
    risks. Measured, they are not: trend and stretch correlate -0.91 (a stock
    safe in an uptrend is by construction overextended) and trend and distress
    +0.87. Seven channels behave like about three.

    This gives the idea its strongest form. Each channel is regressed on all
    the others and replaced by its residual, re-ranked within each date, so
    the seven really are close to independent and the conjunction is doing
    seven separate jobs. If it still fails to beat the best single channel
    after this, the failure is in the premise rather than in the encoding.

    The regression is fitted on a strided sample of ALL history at once. That
    is a deliberate in-sample advantage handed to the idea under test: if it
    cannot win with it, it certainly cannot win without it.
    """
    keys = list(CHANNELS)
    st = np.stack([scores[c] for c in keys])          # (C, T, N)
    flat = st.reshape(len(keys), -1).T                # (T*N, C)
    samp = flat[::sample_every]
    samp = samp[np.isfinite(samp).all(axis=1)]
    out: dict[str, np.ndarray] = {}
    for j, c in enumerate(keys):
        others = [i for i in range(len(keys)) if i != j]
        X = np.c_[np.ones(len(samp)), samp[:, others]]
        beta, *_ = np.linalg.lstsq(X, samp[:, j], rcond=None)
        Xf = np.c_[np.ones(len(flat)), flat[:, others]]
        resid = flat[:, j] - Xf @ beta
        out[c] = _rank(resid.reshape(st.shape[1], st.shape[2]).astype(np.float32))
    return out
