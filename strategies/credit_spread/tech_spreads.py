"""Technically-timed vertical credit spreads — signal logic ported from
whchien/ai-trader (github.com/whchien/ai-trader, GPL-3.0), adapted to
close-only data and used to TIME spread placement instead of stock
entries.

Signals (per ticker, causal, close-based):
  sma_cross   CrossSMAStrategy(5/37): golden cross -> bull, death -> bear
  bbands      BBandsStrategy(20,2): close < lower band -> bull (mean
              reversion), close > upper band -> bear
  rsi_bb      RsiBollingerBandsStrategy: RSI14<30 & close<=lower -> bull;
              RSI14>70 | close>=upper -> bear
  momentum    MomentumStrategy: momentum(14)>0 turning positive with
              close>SMA50 -> bull; close<SMA50 with momentum<0 -> bear
  donchian    TurtleTradingStrategy (close-based Donchian): close breaks
              20d high -> bull; close breaks 10d low -> bear

Trade wrapper: on a signal-flip day, sell a vertical credit spread on
the signal side — bull -> put spread below spot, bear -> call spread
above — with buffer k * sigma60_daily * sqrt(h), width 5% of spot,
expiry snapped DOWN to a covered standard expiry, conservative fills
(BS + smile at 1.3x realized IV, tenor haircut, bid-ask floor,
commissions). Optional early profit-take: resting buyback at TP_FRAC of
the entry credit, executed at the NEXT session's modeled cost.

Baseline for the alpha test: unconditional entries (every Monday) at
the same (k, h) — the signal only has timing alpha if its win rate /
expectancy beats the unconditional one at the same distance.

Protocol: design 2008-2018, single frozen validation 2019-2026.
Universe: optionable, >=10y listed. Uses the full-history panel
(CS_DATA_DIR=cache_full).

Run:  CS_DATA_DIR=$PWD/cache_full python3 tech_spreads.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from common import (
    covered_options_expiry, list_tickers, load_series, _rolling_mean,
)
from pricing import (
    COMMISSION_PER_SHARE, bs_call, bs_put, expected_fill_credit,
    iv_at_strike,
)

HERE = os.path.dirname(os.path.abspath(__file__))
WIDTH_PCT = 0.05
DESIGN_END = np.datetime64("2018-12-31")
START = np.datetime64("2008-01-02")


# ----------------------- indicators (close-only, causal) -----------------------

def rolling_std(x, n):
    m = _rolling_mean(x, n)
    m2 = _rolling_mean(x * x, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def rsi(close, n=14):
    out = np.full(len(close), np.nan)
    d = np.diff(close)
    up = np.maximum(d, 0.0)
    dn = np.maximum(-d, 0.0)
    if len(close) <= n:
        return out
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, len(close)):
        if i > n:
            au = (au * (n - 1) + up[i - 1]) / n
            ad = (ad * (n - 1) + dn[i - 1]) / n
        out[i] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    return out


def sigma60(close):
    lr = np.concatenate(([0.0], np.diff(np.log(close))))
    s = rolling_std(lr, 60)
    return s * np.sqrt(252.0)


def rolling_max(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = x[i - n + 1:i + 1].max()
    return out


def rolling_min(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = x[i - n + 1:i + 1].min()
    return out


def signals(close):
    """Return dict of {name: (bull_entry_bool, bear_entry_bool)} arrays.
    Entry = signal-flip day (first day the condition becomes true)."""
    n = len(close)
    def flips(cond):
        c = np.asarray(cond, bool)
        f = np.zeros(n, bool)
        f[1:] = c[1:] & ~c[:-1]
        return f

    out = {}
    f5, s37 = _rolling_mean(close, 5), _rolling_mean(close, 37)
    out["sma_cross"] = (flips(f5 > s37), flips(f5 < s37))

    mid = _rolling_mean(close, 20)
    sd = rolling_std(close, 20)
    lower, upper = mid - 2 * sd, mid + 2 * sd
    out["bbands"] = (flips(close < lower), flips(close > upper))

    r = rsi(close, 14)
    out["rsi_bb"] = (flips((r < 30) & (close <= lower)),
                     flips((r > 70) | (close >= upper)))

    sma50 = _rolling_mean(close, 50)
    mom = np.full(n, np.nan)
    mom[14:] = close[14:] - close[:-14]
    out["momentum"] = (flips((mom > 0) & (close > sma50)),
                       flips((mom < 0) & (close < sma50)))

    hi20 = rolling_max(close, 20)
    lo10 = rolling_min(close, 10)
    prev_hi = np.concatenate(([np.nan], hi20[:-1]))
    prev_lo = np.concatenate(([np.nan], lo10[:-1]))
    out["donchian"] = (flips(close > prev_hi), flips(close < prev_lo))

    # unconditional baseline: every 5th session, both directions
    base = np.zeros(n, bool)
    base[::5] = True
    out["baseline"] = (base.copy(), base.copy())
    return out


# ----------------------- spread pricing -----------------------

def spread_entry(side, S, b, h_cal_days, sig):
    """Conservative net entry credit at buffer b; None if untradeable."""
    Ks = S * (1 - b) if side == "put" else S * (1 + b)
    Kl = Ks - S * WIDTH_PCT if side == "put" else Ks + S * WIDTH_PCT
    if Kl <= 0:
        return None
    T = max(h_cal_days, 1) / 365.0
    atm = sig * 1.30
    ivs = iv_at_strike(S, Ks, T, atm, side)
    ivl = iv_at_strike(S, Kl, T, atm, side)
    mid = max((bs_put(S, Ks, T, ivs) - bs_put(S, Kl, T, ivl)) if side == "put"
              else (bs_call(S, Ks, T, ivs) - bs_call(S, Kl, T, ivl)), 0.0)
    fill, _ = expected_fill_credit(mid, T)
    net = fill - COMMISSION_PER_SHARE
    if net < 0.05:
        return None
    return Ks, Kl, net


def spread_cost(side, S, Ks, Kl, T, sig):
    """Cost to buy the spread back (per share), modeled."""
    if T <= 0:
        intr = max(Ks - S, 0) if side == "put" else max(S - Ks, 0)
        return min(intr, abs(Ks - Kl))
    atm = sig * 1.30
    ivs = iv_at_strike(S, Ks, T, atm, side)
    ivl = iv_at_strike(S, Kl, T, atm, side)
    v = ((bs_put(S, Ks, T, ivs) - bs_put(S, Kl, T, ivl)) if side == "put"
         else (bs_call(S, Ks, T, ivs) - bs_call(S, Kl, T, ivl)))
    return max(v, 0.0)


# ----------------------- backtest -----------------------

def run(k_sigma: float, horizon: int, tp_frac: float | None,
        tickers: list[str], optionable: dict) -> dict:
    """Returns {signal_name: {window: stats}} for one (k, h, tp) cell."""
    stats: dict = {}
    expiry_cache: dict = {}
    for t in tickers:
        ts = load_series(t)
        if ts is None or not optionable.get(t, False):
            continue
        close, dates = ts.close, ts.dates
        if (dates[-1] - dates[0]).astype(int) < 3652:
            continue
        n = len(close)
        sig = sigma60(close)
        sigs = signals(close)
        i0 = int(np.searchsorted(dates, START))
        for name, (bull, bear) in sigs.items():
            for side, entry_days in (("put", bull), ("call", bear)):
                idxs = np.where(entry_days[i0:])[0] + i0
                for j in idxs:
                    if not np.isfinite(sig[j]) or sig[j] <= 0:
                        continue
                    dkey = (str(dates[j]), horizon)
                    if dkey not in expiry_cache:
                        expiry_cache[dkey] = covered_options_expiry(str(dates[j]), horizon)
                    snap = expiry_cache[dkey]
                    if snap is None:
                        continue
                    exp_iso, _kind, cal_days, _sess = snap
                    exp_d = np.datetime64(exp_iso)
                    if dates[-1] < exp_d:
                        continue
                    b = k_sigma * (sig[j] / np.sqrt(252.0)) * np.sqrt(horizon)
                    ent = spread_entry(side, close[j], b, cal_days, sig[j])
                    if ent is None:
                        continue
                    Ks, Kl, net = ent
                    width = close[j] * WIDTH_PCT
                    # walk to expiry with optional TP
                    ke = int(np.searchsorted(dates, exp_d, side="right")) - 1
                    pnl = None
                    if tp_frac is not None:
                        thr = tp_frac * (net + COMMISSION_PER_SHARE)
                        for m in range(j + 1, ke):
                            Trem = max((exp_d - dates[m]).astype(int), 0) / 365.0
                            v = spread_cost(side, close[m], Ks, Kl, Trem, sig[j])
                            if v + max(0.05, 0.10 * v) / 2.0 <= thr:
                                pnl = (net - thr - 2 * COMMISSION_PER_SHARE) * 100.0
                                break
                    if pnl is None:
                        S_T = close[ke]
                        intr = (min(max(Ks - S_T, 0), width) if side == "put"
                                else min(max(S_T - Ks, 0), width))
                        pnl = (net - intr) * 100.0
                    window = "design" if dates[j] <= DESIGN_END else "validation"
                    d = stats.setdefault(name, {}).setdefault(window, {
                        "n": 0, "wins": 0, "otm": 0, "pnl": 0.0, "risk": 0.0})
                    d["n"] += 1
                    d["wins"] += pnl > 0
                    d["pnl"] += pnl
                    d["risk"] += (width - net) * 100.0
    return stats


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    with open(os.path.join(HERE, "results", "optionable.json")) as fh:
        optionable = json.load(fh)["optionable"]

    k = float(os.environ.get("TS_K", "1.5"))
    h = int(os.environ.get("TS_H", "14"))
    tp = os.environ.get("TS_TP")
    tp_frac = float(tp) if tp else None
    t0 = time.time()
    stats = run(k, h, tp_frac, tickers, optionable)
    print(f"k={k} h={h} tp={tp_frac}  ({time.time()-t0:.0f}s)")
    print(f"{'signal':<10} {'window':<10} {'trades':>7} {'win%':>7} "
          f"{'pnl$':>10} {'ror/trade':>9}")
    for name in ("baseline", "sma_cross", "bbands", "rsi_bb", "momentum", "donchian"):
        for w in ("design", "validation"):
            d = stats.get(name, {}).get(w)
            if not d or not d["n"]:
                continue
            ror = d["pnl"] / d["risk"] * 100 if d["risk"] else 0
            print(f"{name:<10} {w:<10} {d['n']:>7} "
                  f"{100*d['wins']/d['n']:>6.2f}% {d['pnl']:>10.0f} {ror:>8.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
