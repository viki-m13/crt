#!/usr/bin/env python3
"""Study 10: option-implied signals, expressed in STOCK.

The binding constraint in every previous study was the option bid-ask spread:
100-500 bps per round trip, against a gross edge of similar size. But nothing
forces the trade to be expressed in options. If option prices carry information
about the underlying, the position can be taken in the stock, where the spread
is 1-5 bps on liquid names — a 50-100x cost reduction — and the holding period
can be days instead of a month, which multiplies breadth.

That attacks both terms of IR = IC x sqrt(BR) at once.

SIGNALS (all point-in-time, from the chain only):
  iv_spread  put-call parity deviation: vega-weighted mean of (call_iv-put_iv)
             over matched strikes. Calls rich relative to puts = bullish.
  skew25     25-delta put IV minus 25-delta call IV.
  term       front ATM IV minus back ATM IV.
  d_iv       change in front ATM IV over the past ~week.
  ivrv       front ATM IV minus EWMA realized vol.
  cp_oi_prox (unavailable: no volume/OI in this dataset)

LOOKAHEAD CONTROL: signal is computed at observation t; the position is entered
at t+1 and held to t+1+h. The signal snapshot and the entry price therefore
never come from the same quote set, so a stale or noisy quote at t cannot
manufacture return through the entry price. This matters because both the
signals and the spot series are derived from the same option quotes.

COSTS: stock round trip charged at 10 bps (5 bps per side) — deliberately
conservative for names of this liquidity, where 1-2 bps is typical.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
SIG_PQ = os.path.join(HERE, "cache", "optsig.parquet")
STOCK_BPS_RT = 10.0     # round-trip stock cost in bps (conservative)

LIQUID = set("SPY DIA AAPL MSFT NVDA AMZN GOOGL META TSLA AMD MU QCOM NFLX BA "
             "XOM JPM BAC C GM F INTC CSCO PYPL AVGO ORCL CRM ADBE TXN COST "
             "WMT DIS CAT GE PFE CVX WFC MS GS V MA UNH LLY HD PEP KO MRK "
             "ABBV TMO ACN LIN DHR NEE PM RTX HON UNP LOW SBUX GILD MDT "
             "AMAT LRCX KLAC ADI MRVL PANW NOW SNOW UBER ABNB COIN PLTR "
             "SMCI ON WDC ROKU DASH".split())


def build_signals():
    """One row per (date, symbol) with option-implied signals."""
    dates = E.available_dates()
    rows = []
    for di, day in enumerate(dates):
        ch = E.load_chain(day)
        ch = ch[(ch.bid > 0) & (ch.ask > 0) & (ch.vol > 0.02) & (ch.vol < 4.0)]
        if ch.empty:
            continue
        spots = E.parity_spots(E.load_chain(day))
        for sym, g in ch.groupby("act_symbol"):
            if sym not in LIQUID:
                continue
            s = spots.get(sym)
            if s is None or not np.isfinite(s):
                continue
            front = g[(g.dte >= 10) & (g.dte <= 60)]
            if front.empty:
                continue
            exp = front.expiration.min()
            ge = front[front.expiration == exp]
            dte = float(ge.dte.iloc[0])

            calls = ge[ge.call_put == "Call"].set_index("strike")
            puts = ge[ge.call_put == "Put"].set_index("strike")
            ks = calls.index.intersection(puts.index)
            if len(ks) < 3:
                continue
            # --- parity-deviation IV spread, vega-weighted, near the money
            m = np.abs(np.log(np.array([float(k) for k in ks]) / s))
            sel = [k for k, mm in zip(ks, m) if mm <= 0.15]
            if len(sel) < 2:
                sel = list(ks[np.argsort(m)[:4]])
            cv = calls.loc[sel, "vol"].astype(float)
            pv = puts.loc[sel, "vol"].astype(float)
            wv = (calls.loc[sel, "vega"].astype(float)
                  + puts.loc[sel, "vega"].astype(float)).clip(lower=1e-4)
            iv_spread = float(np.average(cv - pv, weights=wv))

            # --- ATM IV, skew, term
            allk = np.array([float(k) for k in ge.strike])
            atm = ge.iloc[np.argsort(np.abs(allk - s))[:4]]
            atm_iv = float(np.average(atm.vol, weights=np.maximum(atm.vega, 1e-4)))
            p25 = ge[(ge.call_put == "Put") & (ge.delta.abs().between(0.15, 0.35))]
            c25 = ge[(ge.call_put == "Call") & (ge.delta.between(0.15, 0.35))]
            skew25 = np.nan
            if len(p25) and len(c25):
                skew25 = float(p25.loc[(p25.delta.abs() - .25).abs().idxmin()].vol
                               - c25.loc[(c25.delta - .25).abs().idxmin()].vol)
            back = g[g.dte > 60]
            term = np.nan
            if not back.empty:
                gb = back[back.expiration == back.expiration.min()]
                kb = np.array([float(k) for k in gb.strike])
                if len(kb):
                    ab = gb.iloc[np.argsort(np.abs(kb - s))[:4]]
                    term = atm_iv - float(np.average(ab.vol,
                                                     weights=np.maximum(ab.vega, 1e-4)))
            rows.append((day, sym, s, iv_spread, atm_iv, skew25, term, dte))
        if (di + 1) % 200 == 0:
            print(f"optsig {di+1}/{len(dates)} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows, columns=["date", "act_symbol", "spot", "iv_spread",
                                     "atm_iv", "skew25", "term", "dte"])
    df.to_parquet(SIG_PQ, index=False)
    print("signals:", df.shape)
    return df


def analyse(df: pd.DataFrame, horizons=(1, 2, 3, 5, 8)):
    df = df.copy()
    df["date"] = pd.to_datetime(df.date)
    df = df.sort_values(["act_symbol", "date"])
    g = df.groupby("act_symbol")
    df["d_iv"] = g.atm_iv.transform(lambda s: s - s.shift(3))
    df["rv"] = g.spot.transform(
        lambda s: np.log(s / s.shift(1)).rolling(20, min_periods=8).std() * math.sqrt(252 / 2.3))
    df["ivrv"] = df.atm_iv - df.rv
    df["mom"] = g.spot.transform(lambda s: s / s.shift(24) - 1)
    df["rev"] = g.spot.transform(lambda s: s / s.shift(3) - 1)

    # entry at t+1, exit at t+1+h  (signal snapshot never prices the entry)
    for h in horizons:
        df[f"fwd{h}"] = g.spot.transform(lambda s: s.shift(-1 - h) / s.shift(-1) - 1)

    sigs = ["iv_spread", "skew25", "term", "d_iv", "ivrv", "mom", "rev"]
    print("\n" + "=" * 74)
    print("IC of option-implied signals vs FORWARD STOCK RETURNS (entry t+1)")
    print("=" * 74)
    print(f"{'signal':>10} " + "".join(f"{'h='+str(h):>14}" for h in horizons))
    best = {}
    for sig in sigs:
        line = f"{sig:>10} "
        for h in horizons:
            ics = []
            for _, gg in df.groupby("date"):
                x = gg[[sig, f"fwd{h}"]].dropna()
                if len(x) >= 15 and x[sig].nunique() > 5:
                    ic = spearmanr(x[sig], x[f"fwd{h}"]).statistic
                    if np.isfinite(ic):
                        ics.append(ic)
            if len(ics) < 100:
                line += f"{'-':>14}"
                continue
            a = np.array(ics)
            t = a.mean() / (a.std() / math.sqrt(len(a)))
            line += f"{a.mean():+.4f}({t:+.1f})".rjust(14)
            best[(sig, h)] = (a.mean(), t, len(a))
        print(line)
    return df, best


def portfolio(df, sig, h, q=0.2, dev_end="2024-12-31", flip=False,
              label="", costs_bps=STOCK_BPS_RT):
    """Cross-sectional long-short, market-neutral, equal weight, hold h obs.

    Overlapping cohorts: h independent books each rebalancing every h days, so
    every observation date contributes 1/h of capital. Costs charged on the
    turnover of each cohort's entry and exit.
    """
    d = df.dropna(subset=[sig, f"fwd{h}"]).copy()
    rows = []
    for dt, g in d.groupby("date"):
        if len(g) < 20:
            continue
        x = g[sig] * (-1 if flip else 1)
        lo, hi = x.quantile(q), x.quantile(1 - q)
        shorts = g[x <= lo][f"fwd{h}"]
        longs = g[x >= hi][f"fwd{h}"]
        if len(longs) < 3 or len(shorts) < 3:
            continue
        gross = 0.5 * (longs.mean() - shorts.mean())
        net = gross - costs_bps / 1e4          # one full round trip per cohort
        rows.append((dt, gross, net, len(longs) + len(shorts)))
    if len(rows) < 60:
        return None
    r = pd.DataFrame(rows, columns=["date", "gross", "net", "n"]).set_index("date")
    # each date starts a cohort held h obs; portfolio return per obs = mean of
    # the h cohorts currently live => scale variance appropriately by using the
    # per-cohort series annualized on its own cadence
    obs_per_year = len(r) / max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    rounds = obs_per_year / h
    out = {}
    for col in ("gross", "net"):
        mu, sd = r[col].mean(), r[col].std()
        out[col] = mu / sd * math.sqrt(rounds) if sd > 0 else np.nan
        out[col + "_mean"] = mu
    dev = r[r.index <= dev_end]
    hold = r[r.index > dev_end]
    for nm, seg in (("dev", dev), ("hold", hold)):
        if len(seg) > 30 and seg.net.std() > 0:
            out[nm] = seg.net.mean() / seg.net.std() * math.sqrt(rounds)
        else:
            out[nm] = np.nan
    out["n_dates"] = len(r)
    out["rounds"] = rounds
    print(f"  {label:<34} grossSh={out['gross']:+.2f} netSh={out['net']:+.2f} "
          f"| dev={out['dev']:+.2f} hold={out['hold']:+.2f} "
          f"| meanNet={out['net_mean']*1e4:+.1f}bp n={len(r)}")
    return out


def main():
    if os.path.exists(SIG_PQ):
        df = pd.read_parquet(SIG_PQ)
        print(f"loaded cached signals {df.shape}")
    else:
        df = build_signals()
    df, best = analyse(df)
    df.to_parquet(os.path.join(HERE, "cache", "optsig_full.parquet"), index=False)

    print("\n" + "=" * 74)
    print("LONG-SHORT STOCK PORTFOLIOS (market-neutral, 10bp round-trip costs)")
    print("=" * 74)
    ranked = sorted(best.items(), key=lambda kv: -abs(kv[1][1]))[:8]
    for (sig, h), (ic, t, n) in ranked:
        portfolio(df, sig, h, flip=(ic < 0),
                  label=f"{sig} h={h} ({'flip' if ic < 0 else 'long'})")


if __name__ == "__main__":
    main()
