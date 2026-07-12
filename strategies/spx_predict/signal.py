"""Live daily signal for the SPX/SPY directional options strategy.

PRICING MODEL v2 (VALIDATION.md §10) — the premium model is audited
against real-market features that v1 ignored, each of which materially
changed the numbers:
  carry — forward F = S*exp(r*T) with historical 3m T-bill rates (the
          panel is TOTAL-RETURN SPY, so the self-consistent carry is r).
  skew  — IV(K) = s_atm * (1 + BETA*ln(F/K)): OTM puts trade rich, OTM
          calls trade cheap, as in the real SPX surface.
  blend — ATM IV from blended variance (w*rv60^2 + (1-w)*rvbar^2, rvbar
          = point-in-time expanding mean): long-dated IV mean-reverts,
          so 2008 prices at ~46% (not 90%) and dead-calm at ~16%.
  slip  — 3% each way (LEAPS/quarter spreads are wider than fronts).
Under v2 the old call-heavy numbers fell hard (28%->10% CAGR) and the
frontier inverted: SELLING the skew-rich puts is the durable edge.

THE STRATEGY (selected on design <2016, confirmed on validation >=2016):
  PUT   — SPY put credit spread, sell -3% / buy -6%, 63 sessions
          (~3 months), HOLD TO EXPIRY. Weekly ladder (every 5th
          session), 3% of equity per rung, 60% total at-risk cap,
          200-dma regime filter. Design 25.2%/-31% DD; validation
          30.3%/-28%; robust to skew/IV/slippage perturbations (19-22%).
  CALL  — bull call spread +2%/+7%, 252 sessions, GTC sell at 80% of
          width, monthly ladder f=5%/cap 30%. The max-per-trade-ROR
          alternative: design 9.9%/-30%, validation 13.5%/-26%.

Honest limits: ~88-93% win, NOT 99%; all concurrent rungs lose together
in a crash (the cap bounds it); fills are modeled (v2 surface), though
SPY/SPX options are the most liquid listed.

Emits spx/docs/data/signal.json.
"""
from __future__ import annotations
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WEB_DATA = os.path.join(REPO, "spx", "docs", "data")
_LOCAL = os.path.join(HERE, "data", "SPY.json")
_SHARED = os.path.join(HERE, "..", "credit_spread", "cache_full", "SPY.json")
SPY_PATH = os.environ.get("SPX_SPY_PATH", _LOCAL if os.path.exists(_LOCAL) else _SHARED)

# ---- pricing model v2 ----
SLIP = 0.03        # per-side slippage on spread value
BETA = 1.0         # skew slope per unit ln(F/K)
W_BLEND = 0.30     # weight on current rv60^2 in the ATM variance blend
IVM = 1.15         # VRP multiplier on the blended base
REGIME_SMA = 200
SPLIT = np.datetime64("2016-01-01")

# 3m T-bill, annual average (decimal). Coarse but far better than r=0;
# extend yearly. Years missing default to the last entry.
TBILL = {1993: .030, 1994: .042, 1995: .055, 1996: .050, 1997: .051,
         1998: .048, 1999: .046, 2000: .058, 2001: .034, 2002: .016,
         2003: .010, 2004: .014, 2005: .032, 2006: .047, 2007: .044,
         2008: .014, 2009: .0015, 2010: .001, 2011: .0005, 2012: .001,
         2013: .0005, 2014: .0003, 2015: .0005, 2016: .003, 2017: .009,
         2018: .020, 2019: .021, 2020: .004, 2021: .0005, 2022: .020,
         2023: .051, 2024: .050, 2025: .043, 2026: .040}

# Frozen books. THE strategy is the put ladder; the call ladder is the
# max-per-trade-ROR alternative.
STRUCTURES = {
    "put":  {"kind": "put_spread", "label": "Put credit spread (the strategy)",
             "k1_off": -0.03, "k2_off": -0.06, "horizon": 63,
             "exit": "expiry", "exit_level": None,
             "every": 5, "f": 0.03, "cap": 0.60},
    "call": {"kind": "call_spread", "label": "Bull call spread (max ROR/trade)",
             "k1_off": 0.02, "k2_off": 0.07, "horizon": 252,
             "exit": "limit", "exit_level": 0.80,
             "every": 21, "f": 0.05, "cap": 0.30},
}


def _N(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call_F(F, K, T, s, r):
    if T <= 0 or s <= 0:
        return max(F - K, 0.0) * math.exp(-r * T)
    d1 = (math.log(F / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return math.exp(-r * T) * (F * _N(d1) - K * _N(d1 - s * math.sqrt(T)))


def bs_put_F(F, K, T, s, r):
    if T <= 0 or s <= 0:
        return max(K - F, 0.0) * math.exp(-r * T)
    d1 = (math.log(F / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return math.exp(-r * T) * (K * _N(-(d1 - s * math.sqrt(T))) - F * _N(-d1))


class Market:
    """SPY panel + the v2 pricing surface (carry, skew, blended IV)."""

    def __init__(self, path=SPY_PATH):
        with open(path) as fh:
            blob = json.load(fh)
        s = blob["series"]
        dates = np.array(s["dates"], dtype="datetime64[D]")
        px = np.array(s["prices"], float)
        m = (px > 0) & np.isfinite(px)
        self.dates, self.px = dates[m], px[m]
        n = len(self.px)
        lr = np.concatenate(([0.0], np.diff(np.log(self.px))))
        self.rv = np.full(n, np.nan)
        for i in range(60, n):
            self.rv[i] = np.std(lr[i - 59:i + 1], ddof=1) * math.sqrt(252)
        self.sma = np.full(n, np.nan)
        for i in range(REGIME_SMA, n):
            self.sma[i] = self.px[i - REGIME_SMA + 1:i + 1].mean()
        self.rvbar = np.full(n, np.nan)
        csum = 0.0; cnt = 0
        for i in range(60, n):
            csum += self.rv[i]; cnt += 1
            self.rvbar[i] = csum / cnt
        years = np.array([int(str(dt)[:4]) for dt in self.dates])
        last = max(TBILL)
        self.rate = np.array([TBILL.get(y, TBILL[last]) for y in years])
        self.n = n

    def regime_ok(self, i):
        return np.isfinite(self.sma[i]) and self.px[i] >= self.sma[i]

    def atm_iv(self, i):
        if not np.isfinite(self.rv[i]) or not np.isfinite(self.rvbar[i]):
            return float("nan")
        return IVM * math.sqrt(W_BLEND * self.rv[i] ** 2
                               + (1 - W_BLEND) * self.rvbar[i] ** 2)

    def spread_val(self, j, K1, K2, T, kind):
        """Mid value of the K1/K2 spread at session j, time-to-expiry T."""
        S = self.px[j]
        r = self.rate[j]
        F = S * math.exp(r * T)
        s_atm = self.atm_iv(j)
        if not np.isfinite(s_atm) or s_atm <= 0:
            return None

        def iv(K):
            return max(s_atm * (1.0 + BETA * math.log(F / K)), 0.03)
        if kind == "call_spread":
            return bs_call_F(F, K1, T, iv(K1), r) - bs_call_F(F, K2, T, iv(K2), r)
        return bs_put_F(F, K1, T, iv(K1), r) - bs_put_F(F, K2, T, iv(K2), r)


def _trade_path(mkt: Market, spec, i):
    """One spread entered at session i under the frozen rule.
    Returns ("closed", dict) | ("open", dict) | None."""
    n = mkt.n
    X = spec["horizon"]; T0 = X / 252.0
    kind = spec["kind"]; is_credit = kind == "put_spread"
    if not mkt.regime_ok(i):
        return None
    S = mkt.px[i]
    K1 = S * (1 + spec["k1_off"]); K2 = S * (1 + spec["k2_off"])
    width = abs(K2 - K1)
    v = mkt.spread_val(i, K1, K2, T0, kind)
    if v is None or v <= 0:
        return None
    if is_credit:
        credit = v * (1 - SLIP); risk = width - credit
        if credit <= 0 or risk <= 0:
            return None
        denom = risk; basis = credit
    else:
        debit = v * (1 + SLIP)
        denom = debit; basis = debit
        tgt = spec["exit_level"] * width

    jend = min(i + X, n - 1)
    pnl = None; exit_j = None; reason = None
    if not is_credit:  # GTC limit exit for the call book
        for j in range(i + 1, jend + 1):
            Tr = (i + X - j) / 252.0
            vj = mkt.spread_val(j, K1, K2, Tr, kind)
            if vj is not None and vj >= tgt:
                pnl = vj * (1 - SLIP) - debit; exit_j = j; reason = "gtc-limit"
                break
    if pnl is None and jend < n - 1:
        if is_credit:
            loss = min(max(K1 - mkt.px[jend], 0.0), width)
            pnl = credit - loss
        else:
            pnl = min(max(mkt.px[jend] - K1, 0.0), width) - debit
        exit_j = jend; reason = "expiry"
    if pnl is None:
        # still open at panel end
        Tr = max((i + X - (n - 1)) / 252.0, 1e-6)
        mark = mkt.spread_val(n - 1, K1, K2, Tr, kind) or 0.0
        cur = (credit - mark * (1 + SLIP)) if is_credit else (mark * (1 - SLIP) - basis)
        exp_date = mkt.dates[i] + np.timedelta64(int(round(X * 365 / 252)), "D")
        return ("open", {
            "entry_date": str(mkt.dates[i]), "spot_at_entry": round(float(S), 2),
            "k1": round(float(K1), 2), "k2": round(float(K2), 2),
            "width": round(float(width), 2),
            "entry_credit" if is_credit else "entry_debit": round(float(basis), 2),
            "current_spot": round(float(mkt.px[n - 1]), 2),
            "current_mark": round(float(mark), 2),
            "current_ror": round(float(cur / denom), 4),
            "max_ror": round(float((credit / risk) if is_credit
                                   else (width - basis) / basis), 4),
            "expiry_date": str(exp_date),
            "days_held": int(n - 1 - i), "days_to_expiry": int(X - (n - 1 - i)),
        })
    split_i = int(np.searchsorted(mkt.dates, SPLIT))
    return ("closed", {
        "entry_date": str(mkt.dates[i]), "exit_date": str(mkt.dates[exit_j]),
        "spot_at_entry": round(float(S), 2), "k1": round(float(K1), 2),
        "k2": round(float(K2), 2), "ror": round(float(pnl / denom), 4),
        "win": bool(pnl > 0), "hold_days": int(exit_j - i), "reason": reason,
        "is_val": bool(i >= split_i), "_exit_i": exit_j,
    })


def simulate_ladder(mkt: Market, spec, f=None, cap=None):
    """Ladder: a new rung every spec['every'] sessions (regime
    permitting), each risking f of current equity, total at-risk capped.
    Returns (closed_trades, open_positions, equity_curve)."""
    f = spec["f"] if f is None else f
    cap = spec["cap"] if cap is None else cap
    n = mkt.n
    entries = list(range(210, n, spec["every"]))
    trades = []; opens = []
    for i in entries:
        r = _trade_path(mkt, spec, i)
        if r is None:
            continue
        (trades if r[0] == "closed" else opens).append((i, r[1]))
    pending = {}
    eq = 1.0; at_risk = 0.0
    curve = []; closed = []; open_out = []
    ev = {i: ("closed", t) for i, t in trades}
    ev.update({i: ("open", t) for i, t in opens})
    start_i = entries[0] if entries else n
    curve.append([str(mkt.dates[start_i]), 1.0])
    for i in range(start_i, n):
        if i in pending:
            for t in pending.pop(i):
                eq *= (1 + t["f_used"] * t["ror"])
                at_risk -= t["f_used"]
                closed.append(t)
            curve.append([str(mkt.dates[i]), round(eq, 4)])
        if i in ev:
            r_kind, t = ev[i]
            fu = min(f, max(cap - at_risk, 0.0))
            if fu > 1e-9:
                t = dict(t); t["f_used"] = round(fu, 4)
                at_risk += fu
                if r_kind == "closed":
                    pending.setdefault(t["_exit_i"], []).append(t)
                else:
                    open_out.append(t)
    for t in closed:
        t.pop("_exit_i", None)
    return closed, open_out, curve


def backtest_full(mkt: Market, spec):
    """Every eligible entry day under the exact live rule (overlapping
    samples) — the robust headline win-rate/ROR."""
    split_i = int(np.searchsorted(mkt.dates, SPLIT))
    ror = []; win = []; hold = []; val = []
    for i in range(210, mkt.n - spec["horizon"]):
        r = _trade_path(mkt, spec, i)
        if r is None or r[0] != "closed":
            continue
        t = r[1]
        ror.append(t["ror"]); win.append(int(t["win"]))
        hold.append(t["hold_days"]); val.append(i >= split_i)
    return map(np.array, (ror, win, hold, val))


def curve_metrics(curve):
    if not curve or len(curve) < 2:
        return {"cagr": None, "maxdd": None}
    v = np.array([c[1] for c in curve])
    dd = float((v / np.maximum.accumulate(v) - 1).min())
    yrs = (np.datetime64(curve[-1][0]) - np.datetime64(curve[0][0])
           ).astype("timedelta64[D]").astype(int) / 365.25
    return {"cagr": round(float(v[-1] ** (1 / yrs) - 1), 4), "maxdd": round(dd, 4)}


def stats(mkt: Market, spec, ladder_trades):
    r, w, hold, val = backtest_full(mkt, spec)

    def sizing(mult):
        f = spec["f"] * mult; cap = spec["cap"] * mult
        _, _, curve = simulate_ladder(mkt, spec, f=f, cap=cap)
        m = curve_metrics(curve)
        return {"frac": round(f, 3), "cagr": m["cagr"], "maxdd": m["maxdd"]}
    return {
        "n": int(len(r)), "n_ladder": len(ladder_trades),
        "win_rate": round(float(w.mean()), 4),
        "win_rate_val": round(float(w[val].mean()), 4) if val.sum() else None,
        "mean_ror": round(float(r.mean()), 4),
        "median_ror": round(float(np.median(r)), 4),
        "mean_ror_val": round(float(r[val].mean()), 4) if val.sum() else None,
        "avg_hold_days": round(float(hold.mean()), 1),
        "worst_ror": round(float(r.min()), 4),
        "annualized_ror": round(float(r.mean() * 252 / hold.mean()), 4),
        "sizing": [sizing(m) for m in (0.67, 1.0, 1.5)],
    }


def ror_histogram(trades, edges=(-1.01, -0.5, 0.0, 0.5, 1.0, 1.4)):
    r = np.array([t["ror"] for t in trades])
    labels = ["≤−50%", "−50–0%", "0–50%", "50–100%", "≥100%"]
    counts = [int(((r >= edges[i]) & (r < edges[i + 1])).sum())
              for i in range(len(edges) - 1)]
    counts[-1] += int((r >= edges[-1]).sum())
    return [{"label": labels[i], "count": counts[i]} for i in range(len(labels))]


def examples(trades):
    wins = sorted([t for t in trades if t["win"]], key=lambda t: t["ror"])
    losses = sorted([t for t in trades if not t["win"]], key=lambda t: t["ror"])
    out = {}
    if wins:
        out["winner"] = wins[len(wins) // 2]
    if losses:
        out["loser"] = losses[0]
    return out


def spy_benchmark(mkt: Market, start_date):
    dates, px = mkt.dates, mkt.px
    i0 = int(np.searchsorted(dates, np.datetime64(start_date)))
    base = px[i0]
    pts = [[str(dates[i]), round(float(px[i] / base), 4)]
           for i in range(i0, len(px), 21)]
    if pts[-1][0] != str(dates[-1]):
        pts.append([str(dates[-1]), round(float(px[-1] / base), 4)])
    yrs = max((dates[-1] - dates[i0]).astype("timedelta64[D]").astype(int), 1) / 365.25
    eq = px[i0:] / base
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"curve": pts, "cagr": round(float((px[-1] / base) ** (1 / yrs) - 1), 4),
            "maxdd": round(dd, 4)}


def action(spec, open_positions, regime_ok, is_entry_day):
    if not regime_ok:
        return "STAND ASIDE — SPY below its 200-day average (no new rungs)"
    cadence = "weekly" if spec["every"] <= 5 else "monthly"
    if is_entry_day:
        return (f"ENTER — open this {cadence.rstrip('ly')}'s rung "
                f"({spec['f']*100:.0f}% of equity at risk)")
    n_open = len(open_positions)
    nxt = "next session" if spec["every"] <= 5 else "the first session of next month"
    return (f"HOLD — {n_open} rung{'s' if n_open != 1 else ''} open; "
            f"next {cadence} rung {nxt}")


def enter_today(mkt: Market, spec):
    i = mkt.n - 1
    S = mkt.px[i]; X = spec["horizon"]; T0 = X / 252.0
    kind = spec["kind"]; is_credit = kind == "put_spread"
    K1 = S * (1 + spec["k1_off"]); K2 = S * (1 + spec["k2_off"])
    width = abs(K2 - K1)
    v = mkt.spread_val(i, K1, K2, T0, kind) or 0.0
    exp_date = mkt.dates[i] + np.timedelta64(int(round(X * 365 / 252)), "D")
    iv = mkt.atm_iv(i)
    if is_credit:
        credit = v * (1 - SLIP); risk = width - credit
        return {"spot": round(float(S), 2), "sell_strike": round(float(K1), 2),
                "buy_strike": round(float(K2), 2), "width": round(float(width), 2),
                "est_credit": round(float(credit), 2),
                "max_ror": round(float(credit / risk), 4) if risk > 0 else None,
                "iv_proxy": round(float(iv), 4), "expiry_date": str(exp_date),
                "profit_target": "hold to expiry"}
    debit = v * (1 + SLIP)
    return {"spot": round(float(S), 2), "long_strike": round(float(K1), 2),
            "short_strike": round(float(K2), 2), "width": round(float(width), 2),
            "est_debit": round(float(debit), 2),
            "max_ror": round(float((width - debit) / debit), 4) if debit > 0 else None,
            "breakeven": round(float(K1 + debit), 2),
            "iv_proxy": round(float(iv), 4), "expiry_date": str(exp_date),
            "profit_target": f"GTC sell at {int(spec['exit_level']*100)}% of width"}


def main() -> int:
    mkt = Market()
    i_last = mkt.n - 1
    regime_ok = mkt.regime_ok(i_last)
    books = {}
    earliest = None
    for key, spec in STRUCTURES.items():
        trades, opens, curve = simulate_ladder(mkt, spec)
        if trades:
            e = trades[0]["entry_date"]
            earliest = e if earliest is None or e < earliest else earliest
        # entry day if the last session falls on the entry cadence grid
        entries = list(range(210, mkt.n, spec["every"]))
        is_entry_day = i_last in entries
        books[key] = {
            "label": spec["label"], "spec": spec,
            "today_action": action(spec, opens, regime_ok, is_entry_day),
            "open_positions": opens,
            "enter_today": enter_today(mkt, spec),
            "track_record": stats(mkt, spec, trades),
            "equity": curve,
            "equity_metrics": curve_metrics(curve),
            "ror_histogram": ror_histogram(trades),
            "examples": examples(trades),
            "recent_trades": list(reversed(trades))[:150],
        }
    strat_curve = books["put"]["equity"]
    out = {
        "as_of": str(mkt.dates[-1]), "spot": round(float(mkt.px[-1]), 2),
        "pricing_model": {"version": 2, "carry": "3m T-bill", "skew_beta": BETA,
                          "iv": f"{IVM} x sqrt({W_BLEND}*rv60^2 + {1-W_BLEND:.2f}*rvbar^2)",
                          "slippage": SLIP},
        "equity_sizing": STRUCTURES["put"]["f"],
        "ladder_cap": STRUCTURES["put"]["cap"],
        "cadence": "one new rung every week (put book) / month (call book)",
        "strategy_equity": dict(curve_metrics(strat_curve), curve=strat_curve),
        "spy_benchmark": spy_benchmark(mkt, earliest or str(mkt.dates[210])),
        "regime": {"sma200": round(float(mkt.sma[i_last]), 2)
                   if np.isfinite(mkt.sma[i_last]) else None,
                   "uptrend": bool(regime_ok),
                   "filter": "enter only when SPY >= its 200-day average"},
        "books": books,
        "note": ("Pricing model v2: carry (historical T-bill), volatility skew, "
                 "mean-reverting blended IV, 3% slippage — audited so the "
                 "backtest pays real-world premiums. ~88-93% accurate, NOT "
                 "99%; concurrent rungs lose together in a crash (the at-risk "
                 "cap bounds it)."),
    }
    os.makedirs(WEB_DATA, exist_ok=True)
    with open(os.path.join(WEB_DATA, "signal.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"as_of={out['as_of']} spot={out['spot']} regime_ok={regime_ok}")
    for key, bk in books.items():
        tr = bk["track_record"]; em = bk["equity_metrics"]
        print(f"  [{key}] {bk['today_action'][:34]:34s} n={tr['n']} "
              f"win={tr['win_rate']*100:.0f}%(val {(tr['win_rate_val'] or 0)*100:.0f}%) "
              f"ROR={tr['mean_ror']*100:+.0f}% | ladder CAGR={em['cagr']*100:.1f}% "
              f"DD={em['maxdd']*100:.0f}% open={len(bk['open_positions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
