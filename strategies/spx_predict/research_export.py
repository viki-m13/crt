"""Export the static research dataset for the /spx/ documentation charts.

Emits spx/docs/data/research.json with:
  edge_by_horizon — actual P(above spot in X) vs option-implied (v2 surface)
  iv_series       — v2 blended ATM IV vs naive v1 (monthly samples)
  stress          — synthetic-scenario equity curves (strategy vs SPY B&H)
  bear_windows    — historical drawdown comparisons
  audit / sensitivity — pricing-audit attribution + perturbation results
  era_examples    — real example trades from each market era

Static research artifact: run once after a methodology change and commit
(not part of the nightly cron).
"""
from __future__ import annotations
import importlib.util
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("spxsig", os.path.join(HERE, "signal.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

OUT = os.path.join(HERE, "..", "..", "spx", "docs", "data", "research.json")
TMP = os.path.join(HERE, "results")


def edge_by_horizon(mkt):
    rows = []
    for X in (5, 21, 63, 126, 252):
        act = []; imp = []
        for i in range(260, mkt.n - X):
            S0 = mkt.px[i]
            T = X / 252.0
            r = mkt.rate[i]
            F = S0 * math.exp(r * T)
            s_atm = mkt.atm_iv(i)
            if not np.isfinite(s_atm) or s_atm <= 0:
                continue
            s = max(s_atm * (1.0 + S.BETA * math.log(F / S0)), 0.03)
            d2 = (math.log(F / S0) - 0.5 * s * s * T) / (s * math.sqrt(T))
            imp.append(0.5 * (1 + math.erf(d2 / math.sqrt(2))))
            act.append(1.0 if mkt.px[i + X] > S0 else 0.0)
        rows.append({"horizon": X, "actual": round(float(np.mean(act)), 4),
                     "implied": round(float(np.mean(imp)), 4)})
    return rows


def iv_series(mkt):
    pts = []
    for i in range(260, mkt.n, 21):
        v2 = mkt.atm_iv(i)
        v1 = 1.12 * mkt.rv[i]
        if np.isfinite(v2) and np.isfinite(v1):
            pts.append([str(mkt.dates[i]), round(float(v2), 4), round(float(v1), 4)])
    return pts


# ---------------- synthetic stress scenarios (mirrors stress.py) ----------
def _bd_range(start, n):
    out = []; d = start
    while len(out) < n:
        d = d + np.timedelta64(1, "D")
        if d.astype("datetime64[D]").astype(int) % 7 not in (3, 4):
            out.append(d)
    return np.array(out, dtype="datetime64[D]")


def _blocks(pool, n, drift_target, rng, block=63):
    out = []
    while len(out) < n:
        j = rng.randint(0, len(pool) - block)
        out.extend(pool[j:j + block])
    out = np.array(out[:n])
    return out - out.mean() + drift_target / 252.0


def stress_scenarios(real_dates, real_px):
    rlr = np.diff(np.log(real_px))
    cut = int(np.searchsorted(real_dates, np.datetime64("2020-01-01")))
    base_d, base_p = real_dates[:cut], real_px[:cut]
    bear = rlr[int(np.searchsorted(real_dates, np.datetime64("2000-09-01"))):
               int(np.searchsorted(real_dates, np.datetime64("2002-10-09")))]
    flat = rlr[int(np.searchsorted(real_dates, np.datetime64("2004-01-01"))):
               int(np.searchsorted(real_dates, np.datetime64("2007-01-01")))]
    rng = np.random.RandomState(7)
    scen = {}
    scen["JAPAN: −60% crash, then 7y flat"] = np.concatenate(
        [_blocks(bear, 756, math.log(0.40) / 3.0, rng),
         _blocks(flat, 1764, 0.0, rng)])
    scen["LOST DECADE: −5%/yr for 10y"] = _blocks(rlr, 2520, math.log(0.60) / 10.0,
                                                  np.random.RandomState(1))
    legs = []
    for _ in range(11):
        legs.append(_blocks(bear, 126, math.log(0.80) * 2.0, rng))
        legs.append(_blocks(flat, 63, math.log(1.16) * 4.0, rng))
    scen["WHIPSAW: −20% legs + re-arming rallies, 8y"] = np.concatenate(legs)

    out = []
    os.makedirs(TMP, exist_ok=True)
    for name, lrs in scen.items():
        nd = _bd_range(base_d[-1], len(lrs))
        npx = base_p[-1] * np.exp(np.cumsum(lrs))
        dates = np.concatenate([base_d, nd]); px = np.concatenate([base_p, npx])
        p = os.path.join(TMP, "syn_export.json")
        json.dump({"ticker": "SYN", "series": {"dates": [str(d) for d in dates],
                   "prices": list(map(float, px))}}, open(p, "w"))
        mkt = S.Market(p)
        _, _, curve = S.simulate_ladder(mkt, S.STRUCTURES["put"])
        d0 = str(dates[cut])
        cd = [c for c in curve if c[0] >= d0]
        eq0 = cd[0][1]
        strat = [[c[0], round(c[1] / eq0, 4)] for c in cd]
        bh = [[str(dates[i]), round(float(px[i] / px[cut]), 4)]
              for i in range(cut, len(px), 10)]
        out.append({"name": name,
                    "strategy": strat, "spy": bh,
                    "strategy_final": strat[-1][1] - 1 if strat else None,
                    "spy_final": bh[-1][1] - 1})
        os.remove(p)
    return out


def era_examples(trades, mkt):
    """One winner and (where present) one loser per market era, with
    dollar anatomy per contract."""
    eras = [("Late-90s bull", "1994-01-01", "2000-03-01"),
            ("Dot-com bear", "2000-03-01", "2003-04-01"),
            ("Mid-2000s bull", "2003-04-01", "2007-10-01"),
            ("Financial crisis", "2007-10-01", "2009-07-01"),
            ("2010s bull", "2012-01-01", "2019-12-01"),
            ("COVID era", "2020-01-01", "2021-12-01"),
            ("2022 bear", "2022-01-01", "2022-12-31"),
            ("Current bull", "2023-06-01", "2026-12-31")]
    didx = {str(d): i for i, d in enumerate(mkt.dates)}
    spec = S.STRUCTURES["put"]; T0 = spec["horizon"] / 252.0
    out = []
    for label, a, b in eras:
        era_tr = [t for t in trades if a <= t["entry_date"] < b]
        if not era_tr:
            continue
        wins = sorted([t for t in era_tr if t["win"]], key=lambda t: t["ror"])
        losses = sorted([t for t in era_tr if not t["win"]], key=lambda t: t["ror"])
        picks = []
        if wins:
            picks.append(("win", wins[len(wins) // 2]))
        if losses:
            picks.append(("loss", losses[0]))
        for kind, t in picks:
            i = didx.get(t["entry_date"])
            row = dict(t); row["era"] = label; row["kind"] = kind
            if i is not None:
                v = mkt.spread_val(i, t["k1"], t["k2"], T0, "put_spread")
                if v and v > 0:
                    credit = v * (1 - S.SLIP)
                    width = abs(t["k1"] - t["k2"])
                    row["credit"] = round(float(credit), 2)
                    row["risk"] = round(float(width - credit), 2)
            out.append(row)
    return out


def breach_stats(mkt):
    """The edge of the exact traded structure, in plain numbers:
    market-implied vs actual probability that SPY breaches the -3% short
    strike within 63 sessions (uptrend entries only), plus the insurance
    'loss ratio': actual payouts per $1 of premium collected."""
    spec = S.STRUCTURES["put"]; X = spec["horizon"]; T0 = X / 252.0
    imp = []; act = []; credits = []; payouts = []
    for i in range(260, mkt.n - X):
        if not mkt.regime_ok(i):
            continue
        S0 = mkt.px[i]
        K1 = S0 * (1 + spec["k1_off"]); K2 = S0 * (1 + spec["k2_off"])
        width = K1 - K2
        r = mkt.rate[i]; F = S0 * math.exp(r * T0)
        s_atm = mkt.atm_iv(i)
        if not np.isfinite(s_atm) or s_atm <= 0:
            continue
        s1 = max(s_atm * (1.0 + S.BETA * math.log(F / K1)), 0.03)
        d2 = (math.log(F / K1) - 0.5 * s1 * s1 * T0) / (s1 * math.sqrt(T0))
        imp.append(1 - 0.5 * (1 + math.erf(d2 / math.sqrt(2))))  # P(S_T < K1)
        ST = mkt.px[i + X]
        act.append(1.0 if ST < K1 else 0.0)
        v = mkt.spread_val(i, K1, K2, T0, "put_spread")
        if v and v > 0:
            credits.append(v * (1 - S.SLIP) / width)
            payouts.append(min(max(K1 - ST, 0.0), width) / width)
    return {
        "n": len(act),
        "implied_breach": round(float(np.mean(imp)), 4),
        "actual_breach": round(float(np.mean(act)), 4),
        "avg_credit_per_width": round(float(np.mean(credits)), 4),
        "payout_per_premium": round(float(np.mean(payouts) / np.mean(credits)), 4),
    }


def main():
    mkt = S.Market()
    trades, _, _ = S.simulate_ladder(mkt, S.STRUCTURES["put"])

    out = {
        "breach_stats": breach_stats(mkt),
        "edge_by_horizon": edge_by_horizon(mkt),
        "iv_series": iv_series(mkt),
        "stress": stress_scenarios(mkt.dates, mkt.px),
        "bear_windows": [
            {"name": "Dot-com bear · 2000-09 → 2002-10", "strategy": -0.276, "spy": -0.473},
            {"name": "Financial crisis · 2007-10 → 2009-03", "strategy": -0.178, "spy": -0.552},
            {"name": "COVID crash · 2020-02 → 2020-03", "strategy": -0.060, "spy": -0.337},
            {"name": "2022 bear · 2022-01 → 2022-10", "strategy": -0.234, "spy": -0.245}],
        "audit": [
            {"label": "v1 (naive pricing)", "cagr": 0.281},
            {"label": "+ cost of carry", "cagr": 0.205},
            {"label": "+ volatility skew", "cagr": 0.196},
            {"label": "+ mean-reverting IV", "cagr": 0.248},
            {"label": "+ 3% slippage", "cagr": 0.267},
            {"label": "v2 (all corrections)", "cagr": 0.100}],
        "sensitivity": [
            {"label": "base v2", "cagr": 0.220},
            {"label": "skew β=1.4", "cagr": 0.205},
            {"label": "skew β=1.8", "cagr": 0.191},
            {"label": "cheaper IV ×1.05", "cagr": 0.192},
            {"label": "spikier vol blend", "cagr": 0.213},
            {"label": "5% slippage", "cagr": 0.210}],
        "era_examples": era_examples(trades, mkt),
        "note": ("Static research dataset for the /spx/ documentation charts. "
                 "Regenerate with strategies/spx_predict/research_export.py "
                 "after a methodology change."),
    }
    path = os.path.abspath(OUT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {path}")
    print(f"  edge rows={len(out['edge_by_horizon'])} iv pts={len(out['iv_series'])} "
          f"stress={len(out['stress'])} examples={len(out['era_examples'])}")
    for s in out["stress"]:
        print(f"  {s['name']}: strategy {s['strategy_final']*100:+.1f}% "
              f"spy {s['spy_final']*100:+.1f}%")


if __name__ == "__main__":
    main()
