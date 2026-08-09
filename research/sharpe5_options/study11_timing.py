#!/usr/bin/env python3
"""Study 11: option-implied MARKET TIMING and vol-targeting of a stock position.

Study 10 is cross-sectional (which stock). This is time-series (how much
exposure, when) and is therefore a different bet that should decorrelate from
both the cross-sectional book and the short-premium sleeves.

Two mechanisms, both expressed purely in stock:

  1. VOL TARGETING. Scale exposure to SPY by 1/sigma, where sigma comes from
     the option market (ATM implied) rather than trailing realized. Implied vol
     is forward-looking, so it should reprice risk faster than a trailing
     window. Documented to raise Sharpe for equity exposure; the question here
     is whether the option-implied version beats the realized-vol version.

  2. SIGNAL TIMING. Scale or flip exposure on option-implied state: the
     variance risk premium (IV - RV), the term-structure slope, and the skew.

Benchmark is always buy-and-hold SPY over the identical window, because a
timing overlay that merely tracks the market has added nothing. Costs are
charged on turnover at 5 bps per unit of exposure traded.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COST_BPS = 5.0


def perf(r: pd.Series, rounds_per_year: float, label: str, turnover=None):
    if len(r) < 40 or r.std() == 0:
        return None
    sh = r.mean() / r.std() * math.sqrt(rounds_per_year)
    cum = float((1 + r).prod())
    yrs = len(r) / rounds_per_year
    cagr = cum ** (1 / max(yrs, 1e-9)) - 1
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    print(f"  {label:<38} Sharpe={sh:+.2f} CAGR={cagr:+.1%} maxDD={dd:+.1%}"
          + (f" turn={turnover:.1f}x/yr" if turnover else ""))
    return {"sharpe": sh, "cagr": cagr, "dd": dd}


def main():
    f = pd.read_parquet(os.path.join(HERE, "cache", "features.parquet"))
    f["date"] = pd.to_datetime(f.date)
    spy = f[f.act_symbol == "SPY"].sort_values("date").reset_index(drop=True)
    spy["ret"] = spy.spot.pct_change().shift(-1)          # t -> t+1 return
    spy["iv"] = spy.iv_front
    spy["rv"] = spy.rv_ewma
    spy["vrp"] = spy.iv - spy.rv
    spy["term"] = spy.iv_front - spy.iv_back
    spy = spy.dropna(subset=["ret", "iv"])
    n = len(spy)
    obs_per_year = n / max((spy.date.iloc[-1] - spy.date.iloc[0]).days / 365.25, 1e-9)
    print(f"SPY observations: {n}  ({obs_per_year:.0f}/yr, "
          f"{spy.date.iloc[0].date()} -> {spy.date.iloc[-1].date()})")

    print("\n" + "=" * 74)
    print("BENCHMARK")
    print("=" * 74)
    perf(spy.ret, obs_per_year, "buy & hold SPY")

    print("\n" + "=" * 74)
    print("1) VOL TARGETING — exposure = target_vol / sigma, capped at 2x")
    print("=" * 74)
    for src, nm in [("iv", "option-implied ATM IV"), ("rv", "trailing realized")]:
        sig = spy[src].replace(0, np.nan)
        # target = median sigma so average exposure is ~1x; all past-only
        tgt = sig.expanding(min_periods=40).median()
        expo = (tgt / sig).clip(0, 2.0).shift(1)          # decided before the return
        r = (expo * spy.ret).dropna()
        turn = float(expo.diff().abs().mean() * obs_per_year)
        r = r - expo.diff().abs().fillna(0).reindex(r.index) * COST_BPS / 1e4
        perf(r, obs_per_year, f"vol-target on {nm}", turn)

    print("\n" + "=" * 74)
    print("2) SIGNAL TIMING — long SPY only when option state is favourable")
    print("=" * 74)
    # all thresholds are expanding-window quantiles: past data only
    for nm, raw, rule in [
        ("VRP > 0 (IV above realized)", spy.vrp, lambda s, q: s > 0),
        ("VRP above its median", spy.vrp, lambda s, q: s > q(0.5)),
        ("term in contango (front<back)", spy.term, lambda s, q: s < 0),
        ("IV below its 80th pct", spy.iv, lambda s, q: s < q(0.8)),
        ("IV below its median", spy.iv, lambda s, q: s < q(0.5)),
    ]:
        def qf(p, raw=raw):
            return raw.expanding(min_periods=40).quantile(p)
        mask = rule(raw, qf)
        expo = mask.astype(float).shift(1)
        r = (expo * spy.ret).dropna()
        turn = float(expo.diff().abs().mean() * obs_per_year)
        r = r - expo.diff().abs().fillna(0).reindex(r.index) * COST_BPS / 1e4
        perf(r, obs_per_year, nm, turn)

    print("\n" + "=" * 74)
    print("3) COMBINED — vol-target sizing x signal gate")
    print("=" * 74)
    sig = spy.iv.replace(0, np.nan)
    tgt = sig.expanding(min_periods=40).median()
    base = (tgt / sig).clip(0, 2.0)
    for nm, raw, rule in [
        ("vol-target x VRP>0", spy.vrp, lambda s: s > 0),
        ("vol-target x contango", spy.term, lambda s: s < 0),
    ]:
        expo = (base * rule(raw).astype(float)).shift(1)
        r = (expo * spy.ret).dropna()
        turn = float(expo.diff().abs().mean() * obs_per_year)
        r = r - expo.diff().abs().fillna(0).reindex(r.index) * COST_BPS / 1e4
        perf(r, obs_per_year, nm, turn)

    print("\n" + "=" * 74)
    print("DEV / HOLDOUT split on the best few")
    print("=" * 74)
    dev_end = pd.Timestamp("2024-12-31")
    for nm, expo in [
        ("vol-target(IV)", (tgt / sig).clip(0, 2.0).shift(1)),
        ("vol-target x VRP>0", (base * (spy.vrp > 0).astype(float)).shift(1)),
    ]:
        r = (expo * spy.ret)
        r.index = spy.date
        for seg, lab in ((r[r.index <= dev_end], "dev"), (r[r.index > dev_end], "holdout")):
            seg = seg.dropna()
            if len(seg) > 40 and seg.std() > 0:
                print(f"  {nm:<24} {lab:<8} Sharpe="
                      f"{seg.mean()/seg.std()*math.sqrt(obs_per_year):+.2f} (n={len(seg)})")


if __name__ == "__main__":
    main()
