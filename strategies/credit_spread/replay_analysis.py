"""Analysis toolkit for the point-in-time replay rows.

Loads results/replay_rows.csv.gz into numpy columns, attaches
vectorized credit-spread pricing (same model as pricing.py, plus an
extra-conservative variant priced at bare realized vol with commissions),
and evaluates candidate publication gates.

Protocol: gates are TUNED on rows with publish date <= TRAIN_END and
verified once on the untouched holdout (publish date > TRAIN_END),
which contains the entire window where the legacy engine took its 36
live losses.
"""
from __future__ import annotations

import csv
import gzip
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS_PATH = os.path.join(HERE, "results", "replay_rows.csv.gz")

TRAIN_END = "2024-12-31"

# Commissions per share for a 2-leg vertical opened once (expires
# worthless when OTM, so no closing cost): 2 x $0.66 per contract.
COMMISSION_PER_SHARE = 0.0132

WIDTH_PCT = 0.05
NET_BID_ASK_FLOOR = 0.05
NET_BID_ASK_FRAC = 0.10


def load_rows(path: str = ROWS_PATH) -> dict[str, np.ndarray]:
    cols: dict[str, list] = {}
    with gzip.open(path, "rt") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        for hcol in header:
            cols[hcol] = []
        for row in rd:
            for k, v in zip(header, row):
                cols[k].append(v)
    out: dict[str, np.ndarray] = {}
    out["date"] = np.array(cols["date"])
    out["ticker"] = np.array(cols["ticker"])
    out["side"] = np.array(cols["side"])
    out["variant"] = np.array(cols["variant"])
    out["expiry_date"] = np.array(cols["expiry_date"])
    for k in ("horizon", "cal_days", "win", "n_pooled"):
        out[k] = np.array(cols[k], dtype=np.int64)
    for k in ("spot", "hist_max", "buffer", "strike"):
        out[k] = np.array(cols[k], dtype=np.float64)
    for k in ("close_at_expiry", "move", "sigma60"):
        out[k] = np.array([v if v != "" else "nan" for v in cols[k]], dtype=np.float64)
    return out


# ----------------------- vectorized pricing -----------------------


def _ncdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _ncdf_fast(x: np.ndarray) -> np.ndarray:
    # erf via numpy-compatible approximation-free route: use scipy if
    # present, else vectorized math.erf (slower but fine for ~1M rows).
    try:
        from scipy.special import erf  # type: ignore
        return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))
    except Exception:
        return _ncdf(x)


def _bs(side_is_put: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray,
        sigma: np.ndarray) -> np.ndarray:
    """Black-Scholes price, r=0. Vectorized; intrinsic where degenerate."""
    ok = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    Ts = np.where(ok, T, 1.0)
    sg = np.where(ok, sigma, 1.0)
    d1 = (np.log(S / np.where(K > 0, K, 1.0)) + 0.5 * sg * sg * Ts) / (sg * np.sqrt(Ts))
    d2 = d1 - sg * np.sqrt(Ts)
    put = K * _ncdf_fast(-d2) - S * _ncdf_fast(-d1)
    call = S * _ncdf_fast(d1) - K * _ncdf_fast(d2)
    px = np.where(side_is_put, put, call)
    intrinsic = np.where(side_is_put, np.maximum(K - S, 0.0), np.maximum(S - K, 0.0))
    return np.where(ok, px, intrinsic)


def tenor_haircut_vec(T: np.ndarray) -> np.ndarray:
    return np.select(
        [T < 0.10, T < 0.25, T < 0.50, T < 1.00],
        [0.80, 0.72, 0.65, 0.58],
        default=0.50,
    )


def attach_pricing(R: dict[str, np.ndarray], iv_mult: float, alpha_put: float = 0.20,
                   alpha_call: float = 0.05, prefix: str = "") -> None:
    """Adds {prefix}mid, {prefix}fill, {prefix}ror columns (per-share)."""
    is_put = R["side"] == "put"
    S = R["spot"]
    T = R["cal_days"] / 365.0
    sigma = np.where(np.isfinite(R["sigma60"]), R["sigma60"], np.nan)
    atm = sigma * iv_mult
    b = R["buffer"]
    K_s = np.where(is_put, S * (1 - b), S * (1 + b))
    width = S * WIDTH_PCT
    K_l = np.where(is_put, K_s - width, K_s + width)
    # smile
    sqT = np.sqrt(np.maximum(T, 1e-9))
    x_s = np.log(S / np.maximum(K_s, 1e-9)) / sqT
    x_l = np.log(S / np.maximum(K_l, 1e-9)) / sqT
    iv_s = atm * (1 + np.where(is_put, alpha_put * np.maximum(0, x_s),
                               alpha_call * np.maximum(0, -x_s)))
    iv_l = atm * (1 + np.where(is_put, alpha_put * np.maximum(0, x_l),
                               alpha_call * np.maximum(0, -x_l)))
    p_s = _bs(is_put, S, K_s, T, iv_s)
    p_l = _bs(is_put, S, K_l, T, iv_l)
    mid = np.maximum(p_s - p_l, 0.0)
    bid_ask = np.maximum(NET_BID_ASK_FLOOR, NET_BID_ASK_FRAC * mid)
    fill = np.minimum(np.maximum(mid - bid_ask / 2.0, 0.0), mid * tenor_haircut_vec(T))
    fill = np.where(K_l > 0, fill, 0.0)  # long-put strike must be positive
    net = fill - COMMISSION_PER_SHARE
    ror = net / np.maximum(width - fill, 0.01)
    R[prefix + "mid"] = mid
    R[prefix + "fill"] = fill
    R[prefix + "net"] = net
    R[prefix + "ror"] = ror
    R[prefix + "width"] = width


# ----------------------- gate evaluation -----------------------


def summarize(R: dict[str, np.ndarray], mask: np.ndarray, label: str = "",
              by: str | None = None) -> dict:
    resolved = mask & (R["win"] >= 0)
    wins = int((R["win"][resolved] == 1).sum())
    losses = int((R["win"][resolved] == 0).sum())
    pub = int(mask.sum())
    out = {
        "label": label, "published": pub, "resolved": wins + losses,
        "wins": wins, "losses": losses,
        "win_rate": wins / (wins + losses) if wins + losses else None,
    }
    if by:
        keys = np.unique(R[by][mask])
        out["by_" + by] = {}
        for k in keys:
            sub = mask & (R[by] == k)
            rs = sub & (R["win"] >= 0)
            w = int((R["win"][rs] == 1).sum()); l = int((R["win"][rs] == 0).sum())
            out["by_" + by][str(k)] = {"published": int(sub.sum()), "wins": w, "losses": l}
    return out


def print_summary(s: dict) -> None:
    wr = f"{s['win_rate']*100:.4f}%" if s["win_rate"] is not None else "n/a"
    print(f"{s['label']:<40s} pub={s['published']:>8d} resolved={s['resolved']:>8d} "
          f"losses={s['losses']:>5d} wr={wr}")


if __name__ == "__main__":
    R = load_rows()
    n = len(R["date"])
    print(f"rows: {n}")
    attach_pricing(R, iv_mult=1.30)                  # realistic (matches pricing.py)
    attach_pricing(R, iv_mult=1.00, prefix="cons_")  # conservative IV floor
    all_mask = np.ones(n, dtype=bool)
    print_summary(summarize(R, all_mask, "ALL (legacy engine, no gates)"))
    train = R["date"] <= TRAIN_END
    print_summary(summarize(R, train, "train <= " + TRAIN_END))
    print_summary(summarize(R, ~train, "holdout  > " + TRAIN_END))
