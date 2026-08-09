#!/usr/bin/env python3
"""Study 13: stock-price effects CONDITIONED on option-implied state.

Study 10 asks whether option signals predict stock returns on their own. This
asks a different and often more productive question: do well-known stock-price
effects work *better* in some option-implied regimes than others?

The intuition is that implied vol and skew describe how much disagreement and
how much crash-fear is priced into a name. Short-term reversal, for instance,
is compensation for providing liquidity — it should pay more where liquidity is
scarce and uncertainty is high, which the option market prices in real time and
a trailing realized-vol estimate learns only after the fact.

Tested interactions (all expressed in STOCK, entry at t+1, 10bp round trip):
  reversal x IV level        does 3-day reversal pay more in high-IV names?
  reversal x IV innovation   ... more when IV just jumped?
  reversal x skew            ... more where crash fear is priced?
  momentum x IV level        does 8-week momentum survive better in calm names?

A real interaction shows monotonic behaviour across conditioning buckets, not
just one strong cell. Buckets are cross-sectional ranks per date, so they are
point-in-time by construction.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
COST_BPS = 10.0


def load():
    p = os.path.join(HERE, "cache", "optsig_full.parquet")
    if not os.path.exists(p):
        raise SystemExit("run study10 first (needs cache/optsig_full.parquet)")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df.date)
    return df.sort_values(["act_symbol", "date"])


def bucketed_ic(df, base_sig, cond_sig, h, nb=3, flip_base=True):
    """IC of base_sig vs forward return, within buckets of cond_sig."""
    d = df.dropna(subset=[base_sig, cond_sig, f"fwd{h}"]).copy()
    d["cb"] = d.groupby("date")[cond_sig].transform(
        lambda s: pd.qcut(s.rank(method="first"), nb, labels=False)
        if s.notna().sum() >= nb * 5 else np.nan)
    out = []
    for b in range(nb):
        g = d[d.cb == b]
        ics = []
        for _, gg in g.groupby("date"):
            x = gg[[base_sig, f"fwd{h}"]].dropna()
            if len(x) >= 8 and x[base_sig].nunique() > 4:
                ic = spearmanr(x[base_sig], x[f"fwd{h}"]).statistic
                if np.isfinite(ic):
                    ics.append(ic)
        if len(ics) < 80:
            out.append((b, np.nan, np.nan, len(ics)))
            continue
        a = np.array(ics)
        out.append((b, a.mean(), a.mean() / (a.std() / math.sqrt(len(a))), len(a)))
    return out


def ls_return(df, base_sig, h, cond_sig=None, cond_bucket=None, nb=3,
              q=0.2, flip=True):
    """Long-short stock book, optionally restricted to one conditioning bucket."""
    d = df.dropna(subset=[base_sig, f"fwd{h}"]).copy()
    if cond_sig is not None:
        d = d.dropna(subset=[cond_sig])
        d["cb"] = d.groupby("date")[cond_sig].transform(
            lambda s: pd.qcut(s.rank(method="first"), nb, labels=False)
            if s.notna().sum() >= nb * 5 else np.nan)
        d = d[d.cb == cond_bucket]
    rows = []
    for dt, g in d.groupby("date"):
        if len(g) < 12:
            continue
        x = g[base_sig] * (-1 if flip else 1)
        lo, hi = x.quantile(q), x.quantile(1 - q)
        L, S = g[x >= hi][f"fwd{h}"], g[x <= lo][f"fwd{h}"]
        if len(L) < 2 or len(S) < 2:
            continue
        rows.append((dt, 0.5 * (L.mean() - S.mean()) - COST_BPS / 1e4))
    if len(rows) < 60:
        return None
    r = pd.Series(dict(rows)).sort_index()
    opy = len(r) / max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    rounds = opy / h
    return {"sharpe": r.mean() / r.std() * math.sqrt(rounds) if r.std() > 0 else np.nan,
            "mean_bp": r.mean() * 1e4, "n": len(r), "rounds": rounds,
            "series": r}


def main():
    df = load()
    print("=" * 76)
    print("STUDY 13 — stock effects conditioned on option-implied state")
    print("=" * 76)

    pairs = [("rev", "atm_iv", "3-day reversal x IV level"),
             ("rev", "d_iv", "3-day reversal x IV innovation"),
             ("rev", "skew25", "3-day reversal x skew"),
             ("rev", "iv_spread", "3-day reversal x parity deviation"),
             ("mom", "atm_iv", "8-week momentum x IV level")]
    for h in (3, 5):
        print(f"\n--- horizon h={h} obs ---")
        for base, cond, lab in pairs:
            if base not in df.columns or cond not in df.columns:
                continue
            res = bucketed_ic(df, base, cond, h)
            cells = "  ".join(
                f"b{b}:{ic:+.4f}({t:+.1f})" if np.isfinite(ic) else f"b{b}:  --"
                for b, ic, t, n in res)
            print(f"  {lab:<38} {cells}")

    print("\n" + "=" * 76)
    print("LONG-SHORT BOOKS in the most favourable conditioning bucket")
    print("=" * 76)
    for base, cond, lab in pairs:
        if base not in df.columns or cond not in df.columns:
            continue
        for h in (3, 5):
            res = bucketed_ic(df, base, cond, h)
            valid = [(b, ic, t) for b, ic, t, n in res if np.isfinite(ic)]
            if not valid:
                continue
            b_best = max(valid, key=lambda x: abs(x[1]))[0]
            ic_best = dict((b, ic) for b, ic, t in valid)[b_best]
            out = ls_return(df, base, h, cond, b_best, flip=(ic_best < 0))
            if out:
                dev = out["series"][out["series"].index <= "2024-12-31"]
                hold = out["series"][out["series"].index > "2024-12-31"]
                ds = dev.mean() / dev.std() * math.sqrt(out["rounds"]) if dev.std() > 0 else np.nan
                hs = hold.mean() / hold.std() * math.sqrt(out["rounds"]) if len(hold) > 20 and hold.std() > 0 else np.nan
                print(f"  {lab:<34} h={h} bucket{b_best}: "
                      f"Sharpe={out['sharpe']:+.2f} (dev {ds:+.2f} / hold {hs:+.2f}) "
                      f"mean={out['mean_bp']:+.1f}bp n={out['n']}")


if __name__ == "__main__":
    main()
