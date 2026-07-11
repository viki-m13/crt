"""Live daily signal for the SPX/SPY directional options strategy.

Two frozen books harvest the same +40pp directional edge (physical
P(SPY up) ~80% vs option-implied ~46%) at opposite ends of the
ROR/accuracy frontier (see VALIDATION.md §7). Both trade SPY (a liquid,
cash-settled SPX proxy), 252-session (~1y) horizon, one position at a
time, GTC limit exits, re-enter next session on close.

  CALL  — bull CALL SPREAD, long ATM / short +5%. GTC sell the spread
          when it marks 80% of width. ~82% win, +60% ROR/trade. (max ROR)
  PUT   — short PUT SPREAD, -5% / -10%. GTC buy-back after capturing 50%
          of the credit. ~89% win (96% validation), +14% ROR. (max accuracy)

Pricing: Black-Scholes, IV = 60d realized * 1.12, r=0. Entry padded and
exit cut by 2% slippage. SPY/SPX options are the most liquid listed, so
modeled fills are realistic — but this is ~80-90% accurate, NOT 99%: the
ROR comes with real losing trades in bear markets.

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

HORIZON = 252
IV_MULT = 1.12
SLIP = 0.02
REGIME_SMA = 200      # only enter when SPY >= its 200-day average (free
                      # downside protection: sidesteps bear-market clusters,
                      # lifts win 81->87%, ROR +60->+70%, cuts DD -39->-28%)
SPLIT = np.datetime64("2016-01-01")

# Frozen structure specs.
STRUCTURES = {
    "call": {"kind": "call_spread", "label": "Bull call spread (max ROR)",
             "k1_off": 0.0, "k2_off": 0.05, "exit": "limit", "exit_level": 0.80},
    "put":  {"kind": "put_spread", "label": "Put credit spread (max accuracy)",
             "k1_off": -0.05, "k2_off": -0.10, "exit": "capture", "exit_level": 0.50},
}


def _N(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, s):
    if T <= 0 or s <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return S * _N(d1) - K * _N(d1 - s * math.sqrt(T))


def bs_put(S, K, T, s):
    if T <= 0 or s <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return K * _N(-(d1 - s * math.sqrt(T))) - S * _N(-d1)


def load():
    b = json.load(open(SPY_PATH))
    dates = np.array(b["series"]["dates"], dtype="datetime64[D]")
    px = np.array(b["series"]["prices"], float)
    m = (px > 0) & np.isfinite(px)
    dates, px = dates[m], px[m]
    lr = np.concatenate(([0.0], np.diff(np.log(px))))
    n = len(px)
    rv = np.full(n, np.nan)
    for i in range(60, n):
        rv[i] = np.std(lr[i - 59:i + 1], ddof=1) * math.sqrt(252)
    sma = np.full(n, np.nan)
    for i in range(REGIME_SMA, n):
        sma[i] = px[i - REGIME_SMA + 1:i + 1].mean()
    return dates, px, rv, sma


def _regime_ok(px_i, sma_i):
    """Enter only in an uptrend (spot at/above its 200-day average)."""
    return np.isfinite(sma_i) and px_i >= sma_i


def _val_at(px_j, K1, K2, Tr, sj, kind):
    if kind == "call_spread":
        return bs_call(px_j, K1, Tr, sj) - bs_call(px_j, K2, Tr, sj)
    return bs_put(px_j, K1, Tr, sj) - bs_put(px_j, K2, Tr, sj)


def simulate(dates, px, rv, sma, spec):
    """Sequential one-position-at-a-time for one structure."""
    n = len(px); T0 = HORIZON / 252.0
    kind = spec["kind"]; is_credit = kind == "put_spread"
    trades = []; open_pos = None; i = 60
    split_i = int(np.searchsorted(dates, SPLIT))
    while i < n:
        S = px[i]; s = rv[i] * IV_MULT
        if not np.isfinite(s) or s <= 0 or not _regime_ok(S, sma[i]):
            i += 1; continue
        K1 = S * (1 + spec["k1_off"]); K2 = S * (1 + spec["k2_off"])
        width = abs(K2 - K1)
        entry = _val_at(S, K1, K2, T0, s, kind)
        if is_credit:
            credit = entry * (1 - SLIP); risk = width - credit
            if credit <= 0 or risk <= 0:
                i += 1; continue
            cost_basis = credit; denom = risk
            buyback = (1 - spec["exit_level"]) * credit  # capture X% of credit
        else:
            debit = entry * (1 + SLIP)
            if debit <= 0:
                i += 1; continue
            cost_basis = debit; denom = debit
            tgt = spec["exit_level"] * width  # sell when mark >= X% of width

        jend = min(i + HORIZON, n - 1)
        pnl = None; exit_j = None; reason = None
        for j in range(i + 1, jend + 1):
            Tr = (i + HORIZON - j) / 252.0
            sj = rv[j] * IV_MULT if np.isfinite(rv[j]) else s
            v = _val_at(px[j], K1, K2, Tr, sj, kind)
            if is_credit:
                vb = v * (1 + SLIP)
                if vb <= buyback:
                    pnl = credit - vb; exit_j = j; reason = "gtc-profit"; break
            else:
                if v >= tgt:
                    pnl = v * (1 - SLIP) - debit; exit_j = j; reason = "gtc-limit"; break
        if pnl is None and jend < n - 1:
            if is_credit:
                loss = min(max(K1 - px[jend], 0.0), width); pnl = credit - loss
            else:
                pnl = min(max(px[jend] - K1, 0.0), width) - debit
            exit_j = jend; reason = "expiry"
        if pnl is None:
            # still open at panel end
            Tr = max((i + HORIZON - (n - 1)) / 252.0, 1e-6)
            sj = rv[n - 1] * IV_MULT if np.isfinite(rv[n - 1]) else s
            mark = _val_at(px[n - 1], K1, K2, Tr, sj, kind)
            cur_pnl = (credit - mark * (1 + SLIP)) if is_credit else (mark * (1 - SLIP) - debit)
            exp_date = dates[i] + np.timedelta64(int(round(HORIZON * 365 / 252)), "D")
            open_pos = {
                "entry_date": str(dates[i]), "spot_at_entry": round(float(S), 2),
                "k1": round(float(K1), 2), "k2": round(float(K2), 2),
                "width": round(float(width), 2),
                "entry_credit" if is_credit else "entry_debit": round(float(cost_basis), 2),
                "current_spot": round(float(px[n - 1]), 2),
                "current_mark": round(float(mark), 2),
                "current_ror": round(float(cur_pnl / denom), 4),
                "max_ror": round(float((credit / risk) if is_credit else (width - debit) / debit), 4),
                "expiry_date": str(exp_date),
                "days_held": int(n - 1 - i), "days_to_expiry": int(HORIZON - (n - 1 - i)),
            }
            break
        trades.append({
            "entry_date": str(dates[i]), "exit_date": str(dates[exit_j]),
            "spot_at_entry": round(float(S), 2), "k1": round(float(K1), 2),
            "k2": round(float(K2), 2), "ror": round(float(pnl / denom), 4),
            "win": bool(pnl > 0), "hold_days": int(exit_j - i), "reason": reason,
            "is_val": bool(i >= split_i),
        })
        i = exit_j + 1
    return trades, open_pos


def backtest_full(dates, px, rv, sma, spec):
    """Every eligible entry day, same exit rule — the robust headline
    win-rate/ROR (thousands of overlapping samples). Returns arrays."""
    n = len(px); T0 = HORIZON / 252.0; kind = spec["kind"]
    is_credit = kind == "put_spread"; split_i = int(np.searchsorted(dates, SPLIT))
    ror = []; win = []; hold = []; val = []
    for i in range(60, n - HORIZON):
        S = px[i]; s = rv[i] * IV_MULT
        if not np.isfinite(s) or s <= 0 or not _regime_ok(S, sma[i]):
            continue
        K1 = S * (1 + spec["k1_off"]); K2 = S * (1 + spec["k2_off"]); width = abs(K2 - K1)
        entry = _val_at(S, K1, K2, T0, s, kind)
        if is_credit:
            credit = entry * (1 - SLIP); risk = width - credit
            if credit <= 0 or risk <= 0:
                continue
            denom = risk; buyback = (1 - spec["exit_level"]) * credit
        else:
            debit = entry * (1 + SLIP)
            if debit <= 0:
                continue
            denom = debit; tgt = spec["exit_level"] * width
        pnl = None; hd = HORIZON
        for j in range(i + 1, i + HORIZON + 1):
            Tr = (i + HORIZON - j) / 252.0
            sj = rv[j] * IV_MULT if np.isfinite(rv[j]) else s
            v = _val_at(px[j], K1, K2, Tr, sj, kind)
            if is_credit:
                vb = v * (1 + SLIP)
                if vb <= buyback:
                    pnl = credit - vb; hd = j - i; break
            else:
                if v >= tgt:
                    pnl = v * (1 - SLIP) - debit; hd = j - i; break
        if pnl is None:
            if is_credit:
                pnl = credit - min(max(K1 - px[i + HORIZON], 0.0), width)
            else:
                pnl = min(max(px[i + HORIZON] - K1, 0.0), width) - debit
        ror.append(pnl / denom); win.append(int(pnl > 0)); hold.append(hd); val.append(i >= split_i)
    return (np.array(ror), np.array(win), np.array(hold), np.array(val))


def stats(dates, px, rv, sma, spec, seq_trades):
    r, w, hold, val = backtest_full(dates, px, rv, sma, spec)

    def sizing(frac):
        """Realistic compounding on the non-overlapping sequential book."""
        if not seq_trades:
            return {"frac": frac, "cagr": None, "maxdd": None}
        cap = 1.0; eq = [1.0]
        for t in seq_trades:
            cap *= (1 + frac * t["ror"]); eq.append(cap)
        eq = np.array(eq)
        dd = float((eq / np.maximum.accumulate(eq) - 1).min())
        yrs = (np.datetime64(seq_trades[-1]["exit_date"]) - np.datetime64(seq_trades[0]["entry_date"])
               ).astype("timedelta64[D]").astype(int) / 365.25
        return {"frac": frac, "cagr": round(cap ** (1 / yrs) - 1, 4), "maxdd": round(dd, 4)}
    return {
        "n": int(len(r)), "n_sequential": len(seq_trades),
        "win_rate": round(float(w.mean()), 4),
        "win_rate_val": round(float(w[val].mean()), 4) if val.sum() else None,
        "mean_ror": round(float(r.mean()), 4), "median_ror": round(float(np.median(r)), 4),
        "mean_ror_val": round(float(r[val].mean()), 4) if val.sum() else None,
        "avg_hold_days": round(float(hold.mean()), 1),
        "worst_ror": round(float(r.min()), 4),
        "annualized_ror": round(float(r.mean() * 252 / hold.mean()), 4),
        "sizing": [sizing(f) for f in (0.10, 0.15, 0.25)],
    }


REF_FRAC = 0.15   # reference sizing for the plotted equity curve


def equity_curve(seq_trades, frac=REF_FRAC):
    """Stepwise account equity at each trade exit (start 1.0)."""
    if not seq_trades:
        return []
    cap = 1.0
    pts = [[seq_trades[0]["entry_date"], 1.0]]
    for t in seq_trades:
        cap *= (1 + frac * t["ror"])
        pts.append([t["exit_date"], round(cap, 4)])
    return pts


def ror_histogram(seq_trades, edges=(-1.01, -0.5, 0.0, 0.5, 1.0, 1.4)):
    """Bin per-trade RORs for a distribution bar chart."""
    r = np.array([t["ror"] for t in seq_trades])
    labels = ["≤−50%", "−50–0%", "0–50%", "50–100%", "≥100%"]
    counts = [int(((r >= edges[i]) & (r < edges[i + 1])).sum()) for i in range(len(edges) - 1)]
    counts[-1] += int((r >= edges[-1]).sum())  # fold the +140% max into the top bin
    return [{"label": labels[i], "count": counts[i]} for i in range(len(labels))]


def examples(seq_trades):
    """A representative winner (median-ish win) and the worst loser."""
    wins = sorted([t for t in seq_trades if t["win"]], key=lambda t: t["ror"])
    losses = sorted([t for t in seq_trades if not t["win"]], key=lambda t: t["ror"])
    out = {}
    if wins:
        out["winner"] = wins[len(wins) // 2]
    if losses:
        out["loser"] = losses[0]
    return out


def spy_benchmark(dates, px, start_date):
    """SPY buy-and-hold, normalized to 1.0 at start_date, ~monthly points."""
    i0 = int(np.searchsorted(dates, np.datetime64(start_date)))
    base = px[i0]
    pts = []
    step = 21  # ~monthly
    for i in range(i0, len(px), step):
        pts.append([str(dates[i]), round(float(px[i] / base), 4)])
    if pts[-1][0] != str(dates[-1]):
        pts.append([str(dates[-1]), round(float(px[-1] / base), 4)])
    cagr = (px[-1] / base) ** (365.25 / max((dates[-1] - dates[i0]).astype("timedelta64[D]").astype(int), 1)) - 1
    eq = px[i0:] / base
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"curve": pts, "cagr": round(float(cagr), 4), "maxdd": round(dd, 4)}


def action(open_pos, spec, regime_ok):
    if open_pos is None:
        return "ENTER" if regime_ok else "STAND ASIDE — SPY below its 200-day average (downtrend)"
    is_credit = spec["kind"] == "put_spread"
    if is_credit:
        # GTC target = capture exit_level of the credit
        tgt_ror = spec["exit_level"] * open_pos["max_ror"]
    else:
        # GTC target = spread marks exit_level of width; ROR at that mark
        debit = open_pos["entry_debit"]
        tgt_ror = (spec["exit_level"] * open_pos["width"] - debit) / max(debit, 1e-9)
    if open_pos["current_ror"] >= tgt_ror:
        return "EXIT — GTC target reached, close and roll a fresh 1-year spread"
    return "HOLD — GTC limit working"


def enter_today(dates, px, rv, sma, spec):
    i = len(px) - 1; S = px[i]; s = rv[i] * IV_MULT; T0 = HORIZON / 252.0
    kind = spec["kind"]; is_credit = kind == "put_spread"
    K1 = S * (1 + spec["k1_off"]); K2 = S * (1 + spec["k2_off"]); width = abs(K2 - K1)
    entry = _val_at(S, K1, K2, T0, s, kind)
    exp_date = dates[i] + np.timedelta64(int(round(HORIZON * 365 / 252)), "D")
    if is_credit:
        credit = entry * (1 - SLIP); risk = width - credit
        return {"spot": round(float(S), 2), "sell_strike": round(float(K1), 2),
                "buy_strike": round(float(K2), 2), "width": round(float(width), 2),
                "est_credit": round(float(credit), 2), "max_ror": round(float(credit / risk), 4),
                "iv_proxy": round(float(s), 4), "expiry_date": str(exp_date),
                "profit_target": f"buy back at {int((1-spec['exit_level'])*100)}% of credit"}
    debit = entry * (1 + SLIP)
    return {"spot": round(float(S), 2), "long_strike": round(float(K1), 2),
            "short_strike": round(float(K2), 2), "width": round(float(width), 2),
            "est_debit": round(float(debit), 2), "max_ror": round(float((width - debit) / debit), 4),
            "breakeven": round(float(K1 + debit), 2), "iv_proxy": round(float(s), 4),
            "expiry_date": str(exp_date),
            "profit_target": f"sell spread at {int(spec['exit_level']*100)}% of width"}


def main() -> int:
    dates, px, rv, sma = load()
    regime_ok = _regime_ok(px[-1], sma[-1])
    books = {}
    earliest = None
    for key, spec in STRUCTURES.items():
        trades, open_pos = simulate(dates, px, rv, sma, spec)
        if trades:
            e = trades[0]["entry_date"]
            earliest = e if earliest is None or e < earliest else earliest
        books[key] = {
            "label": spec["label"], "spec": spec,
            "today_action": action(open_pos, spec, regime_ok),
            "open_position": open_pos,
            "enter_today": enter_today(dates, px, rv, sma, spec),
            "track_record": stats(dates, px, rv, sma, spec, trades),
            "equity": equity_curve(trades),
            "ror_histogram": ror_histogram(trades),
            "examples": examples(trades),
            "recent_trades": list(reversed(trades)),
        }
    out = {
        "as_of": str(dates[-1]), "spot": round(float(px[-1]), 2),
        "horizon_sessions": HORIZON,
        "equity_sizing": REF_FRAC,
        "spy_benchmark": spy_benchmark(dates, px, earliest or str(dates[60])),
        "regime": {"sma200": round(float(sma[-1]), 2) if np.isfinite(sma[-1]) else None,
                   "uptrend": bool(regime_ok),
                   "filter": "enter only when SPY >= its 200-day average"},
        "books": books,
        "note": ("Modeled Black-Scholes fills (IV = 60d realized x 1.12); SPY/SPX "
                 "options are the most liquid listed, so fills are realistic. "
                 "~80-90% accurate, NOT 99% — the ROR comes with real bear-market "
                 "losers. Per-trade ROR becomes portfolio CAGR only via position "
                 "sizing, which sets the drawdown."),
    }
    os.makedirs(WEB_DATA, exist_ok=True)
    with open(os.path.join(WEB_DATA, "signal.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"as_of={out['as_of']} spot={out['spot']}")
    for key, bk in books.items():
        tr = bk["track_record"]
        print(f"  [{key}] {bk['today_action'][:22]:22s} n={tr['n']} "
              f"win={tr['win_rate']*100:.0f}%(val {(tr['win_rate_val'] or 0)*100:.0f}%) "
              f"ROR={tr['mean_ror']*100:+.0f}% ann={tr['annualized_ror']*100:+.0f}% "
              f"hold={tr['avg_hold_days']:.0f}d")
        if bk["open_position"]:
            op = bk["open_position"]
            print(f"        OPEN {op['k1']}/{op['k2']} exp {op['expiry_date']} "
                  f"ROR {op['current_ror']*100:+.0f}% ({op['days_to_expiry']}d left)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
