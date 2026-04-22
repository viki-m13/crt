#!/usr/bin/env python3
"""Pre-compute the Dynamic TP strategy backtest on the webapp's embedded
bt_series and write results into full.json under `tp_backtest`.

Runs once per data refresh. The webapp reads the precomputed results and
renders immediately — no in-browser simulation.

Strategy mirrors max/docs/app.js :: simulateTakeProfit():
  - Monthly top-1 by CAP5+SMA12M (smoothed conviction).
  - Enter at next trading day's close.
  - TP = entry × (1 + clamp(7 × ATR14_pct, 0.05, 0.25)).
  - SL = entry × (1 − clamp(7 × ATR14_pct, 0.05, 0.12)).
  - Exit at first close ≥ TP or ≤ SL, else at close after 252 bars.
  - Cash idle between trades.
  - $1,000/mo contributions accrue while trade is open, deploy on next entry.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

OUT_DIR = os.path.join("max", "docs", "data")
FULL_PATH = os.path.join(OUT_DIR, "full.json")

DCA_MONTHLY = 1000.0
TP_ATR_K = 7.0
SL_ATR_K = 7.0
TP_CAP = 0.25
TP_FLOOR = 0.05
SL_CAP = 0.12
SL_FLOOR = 0.05
TS_BARS = 252


def clamp_frac(atr_pct, k, floor, cap):
    if atr_pct is None or not math.isfinite(atr_pct) or atr_pct <= 0:
        return floor
    d = k * atr_pct
    return max(floor, min(d, cap))


def month_first_indices(dates: list[str]) -> list[int]:
    out: list[int] = []
    prev = ""
    for i, d in enumerate(dates):
        ym = d[:7]
        if ym != prev:
            out.append(i)
            prev = ym
    return out


def max_end_date(bt_series: dict) -> str:
    m = ""
    for s in bt_series.values():
        ds = s.get("dates") or []
        if ds and ds[-1] > m:
            m = ds[-1]
    return m


def shift_date_days(iso: str, days: int) -> str:
    from datetime import date, timedelta
    y, mo, d = iso[:10].split("-")
    return (date(int(y), int(mo), int(d)) + timedelta(days=days)).isoformat()


def build_lookups(bt_series: dict) -> dict:
    """Return {allDates, priceLookup, scoreLookup, atrLookup, monthFirstIdx}."""
    max_end = max_end_date(bt_series)
    stale_cutoff = shift_date_days(max_end, -365) if max_end else ""

    price = {}
    score = {}
    atr = {}
    dates_set: set[str] = set()
    for tk, s in bt_series.items():
        ds = s.get("dates") or []
        ps = s.get("prices") or []
        smooth = s.get("final_smooth12m") or s.get("final") or []
        atrs = s.get("atr14_pct") or []
        if not ds or not ps:
            continue
        pm = {}
        fm = {}
        am = {}
        for i, d in enumerate(ds):
            try:
                pm[d] = float(ps[i]) if ps[i] is not None else None
                if i < len(smooth) and smooth[i] is not None:
                    fm[d] = float(smooth[i])
                if i < len(atrs) and atrs[i] is not None and atrs[i] != atrs[i]:  # NaN check
                    pass
                elif i < len(atrs) and atrs[i] is not None:
                    v = float(atrs[i])
                    if v > 0 and math.isfinite(v):
                        am[d] = v
            except (TypeError, ValueError):
                continue
        price[tk] = pm
        score[tk] = fm
        atr[tk] = am
        if ds[-1] >= stale_cutoff:
            for d in ds:
                dates_set.add(d)

    all_dates = sorted(dates_set)
    mfi = month_first_indices(all_dates)
    return {
        "allDates": all_dates,
        "priceLookup": price,
        "scoreLookup": score,
        "atrLookup": atr,
        "monthFirstIdx": mfi,
    }


def simulate(lookups: dict, universe: list[str]) -> dict:
    all_dates = lookups["allDates"]
    priceL = lookups["priceLookup"]
    scoreL = lookups["scoreLookup"]
    atrL = lookups["atrLookup"]
    mfi = lookups["monthFirstIdx"]
    n = len(all_dates)
    equity = [0.0] * n
    positions: list[dict] = []
    cash = 0.0
    total_invested = 0.0
    open_pos: dict | None = None

    for mi, di in enumerate(mfi):
        total_invested += DCA_MONTHLY
        cash += DCA_MONTHLY
        entry_idx = di + 1
        if entry_idx >= n:
            break
        rank_date = all_dates[di]
        entry_date = all_dates[entry_idx]

        if open_pos is None:
            best_tk = None
            best_score = -float("inf")
            for tk in universe:
                s = scoreL.get(tk, {}).get(rank_date)
                p = priceL.get(tk, {}).get(rank_date)
                if s is None or p is None or not math.isfinite(s) or not math.isfinite(p) or s <= 0 or p <= 0:
                    continue
                if s > best_score:
                    best_score = s
                    best_tk = tk
            if best_tk:
                entry_px = priceL.get(best_tk, {}).get(entry_date)
                if entry_px is not None and math.isfinite(entry_px) and entry_px > 0 and cash > 0:
                    atr_today = atrL.get(best_tk, {}).get(rank_date)
                    if atr_today is None:
                        atr_today = atrL.get(best_tk, {}).get(entry_date)
                    tp_frac = clamp_frac(atr_today, TP_ATR_K, TP_FLOOR, TP_CAP)
                    sl_frac = clamp_frac(atr_today, SL_ATR_K, SL_FLOOR, SL_CAP)
                    shares = cash / entry_px
                    open_pos = {
                        "tk": best_tk,
                        "entry_idx": entry_idx,
                        "entry_px": entry_px,
                        "shares": shares,
                        "cost": cash,
                        "tp_px": entry_px * (1 + tp_frac),
                        "sl_px": entry_px * (1 - sl_frac),
                        "tp_frac": tp_frac,
                        "sl_frac": sl_frac,
                        "atr_pct": atr_today,
                        "stop_idx": entry_idx + TS_BARS,
                    }
                    cash = 0.0

        next_di = mfi[mi + 1] if (mi + 1) < len(mfi) else n
        for d in range(di, next_di):
            if open_pos is not None and d > open_pos["entry_idx"]:
                tk = open_pos["tk"]
                px = priceL.get(tk, {}).get(all_dates[d])
                if px is not None and math.isfinite(px) and px > 0:
                    exit_reason = None
                    exit_px = None
                    if px >= open_pos["tp_px"]:
                        exit_reason = "tp"
                        exit_px = open_pos["tp_px"]
                    elif px <= open_pos["sl_px"]:
                        exit_reason = "sl"
                        exit_px = open_pos["sl_px"]
                    elif d >= open_pos["stop_idx"]:
                        exit_reason = "time"
                        exit_px = px
                    if exit_reason:
                        proceeds = open_pos["shares"] * exit_px
                        cash += proceeds
                        positions.append({
                            "tk": open_pos["tk"],
                            "entry_idx": open_pos["entry_idx"],
                            "exit_idx": d,
                            "entry_date": all_dates[open_pos["entry_idx"]],
                            "exit_date": all_dates[d],
                            "entry_px": float(open_pos["entry_px"]),
                            "exit_px": float(exit_px),
                            "cost": float(open_pos["cost"]),
                            "proceeds": float(proceeds),
                            "ret": float(proceeds / open_pos["cost"] - 1),
                            "days_held": d - open_pos["entry_idx"],
                            "reason": exit_reason,
                            "tp_frac": float(open_pos["tp_frac"]),
                            "sl_frac": float(open_pos["sl_frac"]),
                            "atr_pct": float(open_pos["atr_pct"]) if open_pos["atr_pct"] else None,
                        })
                        open_pos = None

            eq = cash
            if open_pos is not None:
                if d < open_pos["entry_idx"]:
                    eq += open_pos["cost"]
                else:
                    px = priceL.get(open_pos["tk"], {}).get(all_dates[d])
                    eq += (open_pos["shares"] * px) if (px is not None and math.isfinite(px)) else open_pos["cost"]
            equity[d] = eq

    return {
        "equity": equity,
        "positions": positions,
        "total_invested": total_invested,
        "dates": all_dates,
    }


def benchmark_spy(lookups: dict) -> dict:
    """Monthly DCA into SPY baseline. Entry at D+1 close, hold forever."""
    all_dates = lookups["allDates"]
    priceL = lookups["priceLookup"]
    mfi = lookups["monthFirstIdx"]
    n = len(all_dates)
    equity = [0.0] * n
    shares = 0.0
    total_invested = 0.0
    spy_px = priceL.get("SPY") or {}

    # Forward-fill last-known spy price for mark-to-market
    last_spy = None
    for mi, di in enumerate(mfi):
        entry_idx = di + 1
        if entry_idx >= n:
            break
        px = spy_px.get(all_dates[entry_idx])
        if px is None or not math.isfinite(px) or px <= 0:
            continue
        shares += DCA_MONTHLY / px
        total_invested += DCA_MONTHLY
        # Mark-to-market between entry and next contribution
    for i in range(n):
        px = spy_px.get(all_dates[i])
        if px is not None and math.isfinite(px) and px > 0:
            last_spy = px
        if last_spy is not None:
            equity[i] = shares * last_spy
        elif i > 0:
            equity[i] = equity[i - 1]

    # Re-run to recompute shares accumulation properly (order matters for CAGR)
    equity = [0.0] * n
    shares = 0.0
    total_invested = 0.0
    nxt = 0
    for i in range(n):
        # Process monthly contributions due today (on D+1 after each month-first)
        while nxt < len(mfi) and mfi[nxt] + 1 == i:
            px = spy_px.get(all_dates[i])
            if px is not None and math.isfinite(px) and px > 0:
                shares += DCA_MONTHLY / px
                total_invested += DCA_MONTHLY
            nxt += 1
        px = spy_px.get(all_dates[i])
        if px is not None and math.isfinite(px) and px > 0:
            equity[i] = shares * px
        elif i > 0:
            equity[i] = equity[i - 1]
    return {"equity": equity, "total_invested": total_invested, "dates": all_dates}


def metrics(equity: list[float], total_invested: float, dates: list[str]) -> dict:
    if not equity or total_invested <= 0:
        return {}
    # Find start of positive equity
    start_i = next((i for i, v in enumerate(equity) if v > 0), 0)
    eq = equity[start_i:]
    if not eq:
        return {}
    final = eq[-1]
    yrs = len(eq) / 252.0
    cagr = (final / total_invested) ** (1 / yrs) - 1 if yrs > 0 and final > 0 else -1.0

    # Daily returns
    ret = []
    for i in range(1, len(eq)):
        if eq[i - 1] > 0:
            ret.append(eq[i] / eq[i - 1] - 1)
    if ret:
        mean_r = sum(ret) / len(ret)
        var_r = sum((r - mean_r) ** 2 for r in ret) / len(ret)
        std_r = math.sqrt(var_r)
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    peak = 0.0
    maxdd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > maxdd:
                maxdd = dd
    calmar = cagr / maxdd if maxdd > 0 else 0.0

    return {
        "cagr": cagr,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "final_equity": final,
        "total_invested": total_invested,
        "window_start": dates[start_i] if start_i < len(dates) else None,
        "window_end": dates[-1] if dates else None,
        "years": yrs,
    }


def downsample_equity(equity: list[float], dates: list[str], target: int = 400) -> tuple[list[str], list[float], list[float]]:
    """Downsample equity curve to ~target points for efficient browser render."""
    n = len(equity)
    if n <= target:
        return dates, list(equity), []
    stride = max(1, n // target)
    out_dates = []
    out_eq = []
    for i in range(0, n, stride):
        out_dates.append(dates[i])
        out_eq.append(float(equity[i]))
    return out_dates, out_eq, []


def main() -> int:
    if not os.path.exists(FULL_PATH):
        print(f"[err] {FULL_PATH} not found", file=sys.stderr)
        return 1
    with open(FULL_PATH, "r") as f:
        doc = json.load(f)

    bt = doc.get("bt_series") or {}
    if not bt:
        print("[warn] no bt_series; skipping TP backtest precompute")
        return 0

    lookups = build_lookups(bt)
    # Universe: all tickers in bt_series, excluding SPY
    universe = [t for t in bt.keys() if t != "SPY"]
    print(f"Universe: {len(universe)} tickers, spine: {len(lookups['allDates'])} dates "
          f"({lookups['allDates'][0] if lookups['allDates'] else '—'} → "
          f"{lookups['allDates'][-1] if lookups['allDates'] else '—'})")

    sim = simulate(lookups, universe)
    bench = benchmark_spy(lookups)

    sim_metrics = metrics(sim["equity"], sim["total_invested"], sim["dates"])
    bench_metrics = metrics(bench["equity"], bench["total_invested"], bench["dates"])

    # Trade stats
    positions = sim["positions"]
    n_trades = len(positions)
    tp_hits = sum(1 for p in positions if p["reason"] == "tp")
    sl_hits = sum(1 for p in positions if p["reason"] == "sl")
    time_hits = sum(1 for p in positions if p["reason"] == "time")
    wins = [p for p in positions if p["ret"] > 0]
    losers = [p for p in positions if p["ret"] < 0]
    avg_win = sum(p["ret"] for p in wins) / len(wins) if wins else 0.0
    avg_loss = sum(p["ret"] for p in losers) / len(losers) if losers else 0.0
    gross_wr = len(wins) / n_trades if n_trades else 0.0
    days = [p["days_held"] for p in positions]
    avg_days = sum(days) / len(days) if days else 0.0
    days_sorted = sorted(days)
    med_days = days_sorted[len(days_sorted) // 2] if days_sorted else 0

    # Keep best/worst 8 trades for the log
    best = sorted(positions, key=lambda p: p["ret"], reverse=True)[:8]
    worst = sorted(positions, key=lambda p: p["ret"])[:8]

    print(f"TP strategy: CAGR {sim_metrics['cagr']*100:.2f}% MDD {sim_metrics['maxdd']*100:.1f}% "
          f"Sharpe {sim_metrics['sharpe']:.2f} Calmar {sim_metrics['calmar']:.3f}")
    print(f"  {n_trades} trades (TP {tp_hits} / SL {sl_hits} / time {time_hits})  gross WR {gross_wr*100:.1f}%")
    print(f"  avg winner +{avg_win*100:.2f}%  avg loser {avg_loss*100:.2f}%  median hold {med_days}d")
    print(f"SPY DCA baseline: CAGR {bench_metrics['cagr']*100:.2f}% MDD {bench_metrics['maxdd']*100:.1f}%")

    # Downsample curves for webapp
    strat_dates, strat_eq, _ = downsample_equity(sim["equity"], sim["dates"])
    bench_dates, bench_eq, _ = downsample_equity(bench["equity"], bench["dates"])

    doc["tp_backtest"] = {
        "spine_start": lookups["allDates"][0] if lookups["allDates"] else None,
        "spine_end": lookups["allDates"][-1] if lookups["allDates"] else None,
        "strategy": {
            "metrics": sim_metrics,
            "equity_curve": {"dates": strat_dates, "equity": strat_eq},
            "trade_counts": {
                "n_trades": n_trades,
                "tp_hits": tp_hits,
                "sl_hits": sl_hits,
                "time_hits": time_hits,
                "gross_wr": gross_wr,
                "avg_winner": avg_win,
                "avg_loser": avg_loss,
                "avg_days_held": avg_days,
                "median_days_held": med_days,
            },
            "best_trades": best,
            "worst_trades": worst,
        },
        "spy_dca": {
            "metrics": bench_metrics,
            "equity_curve": {"dates": bench_dates, "equity": bench_eq},
        },
        "model": {
            "tp_atr_k": TP_ATR_K,
            "sl_atr_k": SL_ATR_K,
            "tp_cap": TP_CAP,
            "tp_floor": TP_FLOOR,
            "sl_cap": SL_CAP,
            "sl_floor": SL_FLOOR,
            "time_stop_bars": TS_BARS,
            "dca_monthly": DCA_MONTHLY,
        },
    }

    with open(FULL_PATH, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"Wrote tp_backtest into {FULL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
