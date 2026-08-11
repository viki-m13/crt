#!/usr/bin/env python3
"""Study 29: HIGH credit/width credit spreads — the user's spec.

Spec, verbatim intent: "selling credit spreads, making a hundred dollar bet to
make a profit of two hundred or three hundred dollars ... at the very least a
hundred dollar bet to make fifty dollars."

Translation: risk = width - credit, reward = credit.  credit/risk >= 0.5
means credit/width >= 1/3; credit/risk = 2-3x means credit/width = 2/3..3/4,
which forces the short strike IN the money.  By put-call parity a 5%-ITM put
credit spread IS a long OTM call spread: the position is a leveraged bullish
bet.  The candidate edge is physical drift (equity risk premium) vs the
risk-neutral drift the option is priced on — which is beta, not alpha, unless
proven otherwise.  Study 15's quintile table showed the highest-cw quintile as
the only profitable one; study 25's control showed ITM-call-equivalents were
bull-market beta.  This study settles it with:

  1. screening sweep across short-strike offsets (-5% OTM .. +5% ITM)
  2. honest ladder portfolio on RISK-based capital (capital = max loss)
  3. beta control: regress ladder weekly returns on SPY weekly returns
  4. dev (<=2024-12-31) / holdout (2025+) split
  5. ITM bid-ask width check (the fills are worst-side throughout)

All fills worst-side: sell at bid, buy at ask.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import engine as E
from structures import settle

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE: dict = {}
DEV_END = pd.Timestamp("2024-12-31")


def load_panel():
    sp = pd.read_parquet(os.path.join(HERE, "cache", "spots.parquet"))
    panel = sp.pivot(index="date", columns="act_symbol", values="spot").sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


def build(off, width, dte_lo, dte_hi, panel, sym="SPY"):
    """Put credit spread cohorts. off = short strike offset vs spot
    (+ = ITM, - = OTM). width in fraction of spot. Worst-side fills."""
    key = (off, width, dte_lo, dte_hi, sym)
    if key in CACHE:
        return CACHE[key]
    hist = panel[sym].dropna()

    def se_of(exp):
        ts = pd.Timestamp(exp)
        b = hist[hist.index <= ts]
        a = hist[hist.index > ts]
        if len(b) and (ts - b.index[-1]).days <= 4:
            return float(b.iloc[-1])
        if len(a) and (a.index[0] - ts).days <= 4:
            return float(a.iloc[0])
        return None

    rows = []
    for day in E.available_dates():
        ch = E.load_chain(day)
        g = ch[(ch.act_symbol == sym) & (ch.dte >= dte_lo) & (ch.dte <= dte_hi)
               & (ch.call_put == "Put")]
        ts = pd.Timestamp(day)
        if g.empty or ts not in panel.index:
            continue
        s = panel.at[ts, sym]
        if not np.isfinite(s):
            continue
        exp = g.expiration.min()
        ge = g[g.expiration == exp]
        se = se_of(exp)
        if se is None:
            continue
        sh, lg = ge[ge.bid > 0], ge[ge.ask > 0]
        if sh.empty or lg.empty:
            continue
        sl = sh.loc[(sh.strike - s * (1 + off)).abs().idxmin()]
        wl = lg.loc[(lg.strike - s * (1 + off - width)).abs().idxmin()]
        ks, kl = float(sl.strike), float(wl.strike)
        if kl >= ks:
            continue
        w = ks - kl
        credit = float(sl.bid) - float(wl.ask)          # worst side both legs
        if credit <= 0:
            continue
        risk = w - credit
        if risk <= 0.005 * s:                            # ~free money = bad quote
            continue
        loss = settle(se, ks, "Put") - settle(se, kl, "Put")
        # bid-ask width of the SHORT (possibly ITM) leg, as a fraction of spot
        ba = (float(sl.ask) - float(sl.bid)) / s if float(sl.ask) > 0 else np.nan
        rows.append((day, exp, s, se, ks, kl, w, credit, risk, loss, ba))
    d = pd.DataFrame(rows, columns=["date", "exp", "spot", "se", "ks", "kl",
                                    "w", "credit", "risk", "loss", "ba_short"])
    if len(d):
        d["date"] = pd.to_datetime(d.date)
        d["ret_risk"] = (d.credit - d.loss) / d.risk     # return on max loss
        d["dte"] = (pd.to_datetime(d.exp) - d.date).dt.days
    CACHE[key] = d
    return d


def uptrend_mask(d, panel, sym="SPY", trading_days=200):
    """spot >= its ~200-trading-day SMA, computed on the chain-date grid."""
    hist = panel[sym].dropna()
    obs_per_year = len(hist) / max((hist.index[-1] - hist.index[0]).days / 365.25, 1e-9)
    win = max(10, int(round(obs_per_year * trading_days / 252)))
    sma = hist.rolling(win, min_periods=win // 2).mean()
    ok = (hist >= sma)
    return d.date.map(ok).fillna(False).values, win


def screen_row(d, label):
    if d is None or len(d) < 30:
        print(f"  {label:<26} insufficient")
        return
    cw = (d.credit / d.w).mean()
    cr = (d.credit / d.risk).mean()
    be = (d.risk / d.w).mean()               # breakeven win prob = risk/width
    win = (d.ret_risk > 0).mean()
    mu = d.ret_risk.mean()
    ann = (1 + mu) ** (365.25 / d.dte.mean()) - 1 if mu > -1 else np.nan
    print(f"  {label:<26} n={len(d):>4} c/w={cw:.3f} credit:risk={cr:4.2f}x "
          f"BE={be:5.1%} win={win:5.1%} edge={win-be:+5.1%} "
          f"ret/risk={mu:+.4f} ann*={ann:+7.1%}")


def ladder(d, rung_eq=0.05, cap=0.60, cadence="W", mask=None):
    """Weekly-rung ladder on RISK-based capital: each rung commits
    rung_eq*equity of MAX-LOSS capital; per-unit return = (credit-loss)/risk,
    floored at -1 by construction.  Skipped rungs (gate off / cap full) leave
    capital idle — no return imputed."""
    if d is None or len(d) < 40:
        return None
    d = d.sort_values("date").copy()
    if mask is not None:
        d["gate"] = mask
    else:
        d["gate"] = True
    d["per"] = d.date.dt.to_period(cadence)
    d = d.groupby("per", as_index=False).first()
    equity, open_r, curve = 1.0, [], []
    for r in d.itertuples(index=False):
        still = []
        for exp, capital, rr in open_r:
            if exp <= r.date.strftime("%Y-%m-%d"):
                equity += capital * rr
            else:
                still.append((exp, capital, rr))
        open_r = still
        at_risk = sum(c for _, c, _ in open_r)
        want = rung_eq * equity
        if r.gate and want > 0 and at_risk + want <= cap * equity and equity > 0:
            open_r.append((r.exp, want, float(r.ret_risk)))
        curve.append((r.date, equity))
    for exp, capital, rr in open_r:
        equity += capital * rr
    eq = pd.Series(dict(curve)).sort_index()
    if len(eq) < 30 or (eq <= 0).any():
        return {"ruin": True, "eq": eq}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    rr_ = eq.pct_change().dropna()
    ppy = len(eq) / max(yrs, 1e-9)
    return {"cagr": float(eq.iloc[-1]) ** (1 / max(yrs, 1e-9)) - 1,
            "sharpe": rr_.mean() / rr_.std() * math.sqrt(ppy) if rr_.std() > 0 else np.nan,
            "dd": float((eq / eq.cummax() - 1).min()),
            "eq": eq, "n": len(d), "ruin": False}


def show(r, label):
    if r is None:
        print(f"  {label:<44} insufficient")
        return
    if r.get("ruin"):
        print(f"  {label:<44} RUIN")
        return
    wy = (1 + r["eq"].pct_change().dropna()).groupby(
        r["eq"].pct_change().dropna().index.year).prod() - 1
    print(f"  {label:<44} CAGR={r['cagr']:+7.1%} Sharpe={r['sharpe']:+5.2f} "
          f"maxDD={r['dd']:+7.1%} worstYr={wy.min():+.1%}")


def beta_control(r, panel, label):
    """Regress ladder weekly returns on SPY weekly returns: how much of the
    CAGR is just levered SPY?  alpha reported with Newey-West-free naive t —
    conservative reading: |t|<2 means no evidence of alpha."""
    if r is None or r.get("ruin"):
        return
    eq = r["eq"]
    spy = panel["SPY"].dropna()
    lw = eq.resample("W-FRI").last().pct_change().dropna()
    sw = spy.resample("W-FRI").last().pct_change().dropna()
    both = pd.DataFrame({"l": lw, "s": sw}).dropna()
    if len(both) < 40:
        return
    b = both.l.cov(both.s) / both.s.var()
    a = both.l.mean() - b * both.s.mean()
    resid = both.l - (a + b * both.s)
    se = resid.std() / math.sqrt(len(both))
    t = a / se if se > 0 else np.nan
    print(f"    {label}: beta={b:+.2f}  alpha={a*52:+.1%}/yr (t={t:+.2f})  "
          f"corr={both.l.corr(both.s):+.2f}  n={len(both)}w")


def main():
    panel = load_panel()
    print("=" * 96)
    print("STUDY 29 — high credit/width credit spreads (the 'risk $100 to make $200' spec)")
    print("=" * 96)

    print("\n--- 1. screening sweep: SPY put credit spreads, 5% wide, 45-75 DTE, worst-side ---")
    print("    (ann* compounds mean per-trade ret at own tenor; ignores overlap/capital — "
          "screening only)")
    ds = {}
    for off in (-0.05, -0.02, 0.00, 0.02, 0.05):
        d = build(off, 0.05, 45, 75, panel)
        ds[off] = d
        lab = f"short {'ITM' if off > 0 else 'OTM' if off < 0 else 'ATM'} {abs(off):.0%}"
        screen_row(d, lab)

    print("\n--- 2. ITM liquidity: bid-ask width of the short leg (fraction of spot) ---")
    for off, d in ds.items():
        if d is None or not len(d):
            continue
        print(f"  off={off:+.0%}: median={d.ba_short.median():.3%}  "
              f"p90={d.ba_short.quantile(0.9):.3%}  "
              f"credit/spot={(d.credit/d.spot).median():.3%}")

    print("\n--- 3. honest ladder: weekly rungs, 5% of equity AT RISK per rung, 60% cap ---")
    results = {}
    for off in (0.02, 0.05):
        d = ds[off]
        mask, win = uptrend_mask(d, panel)
        for gate, m in (("always", None), (f"uptrend(sma{win})", mask)):
            r = ladder(d, mask=m)
            results[(off, gate)] = r
            show(r, f"ITM {off:.0%}, gate={gate}")
            beta_control(r, panel, "beta control")

    # OTM comparator at the same risk accounting
    d = ds[-0.05]
    r = ladder(d)
    show(r, "OTM 5% comparator, gate=always")
    beta_control(r, panel, "beta control")

    print("\n--- 4. dev/holdout split (dev <= 2024-12-31) ---")
    for off in (0.02, 0.05):
        d = ds[off]
        mask, win = uptrend_mask(d, panel)
        for gate, m in (("always", None), ("uptrend", mask)):
            for lab, cut in (("dev", d.date <= DEV_END), ("holdout", d.date > DEV_END)):
                dd_ = d[cut.values].copy()
                mm = m[cut.values] if m is not None else None
                r = ladder(dd_, mask=mm)
                show(r, f"ITM {off:.0%} {gate:<8} {lab}")

    print("\n--- 5. yearly ladder returns, ITM 5%, gate=always ---")
    r = results[(0.05, "always")]
    if r and not r.get("ruin"):
        rr = r["eq"].pct_change().dropna()
        wy = (1 + rr).groupby(rr.index.year).prod() - 1
        for y, v in wy.items():
            print(f"  {y}: {v:+7.1%}")

    print("\nVerdict criteria (pre-stated): the spec is only a STRATEGY rather than")
    print("levered beta if (a) ladder alpha t>2 vs SPY, (b) holdout does not")
    print("collapse, (c) worst year survivable at the sized cap.")


if __name__ == "__main__":
    main()
