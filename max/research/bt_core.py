#!/usr/bin/env python3
"""Core backtest primitives for the stocks Top-Picks strategy.

Loads max/docs/data/full.json (per-ticker bt_series: dates, prices,
point-in-time final_score) and simulates monthly-DCA strategies. The
signal is computed at the close of each month-first trading day; entry
fills at the *next* day's close (proxy for next-day open, since the
live pipeline runs the model after close). No per-ticker "best horizon"
look-ahead: every run uses an explicit hold choice.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech", "ADBE": "Tech",
    "CRM": "Tech", "AMD": "Tech", "INTC": "Tech", "CSCO": "Tech", "TXN": "Tech",
    "AMAT": "Tech", "MU": "Tech", "NOW": "Tech", "PANW": "Tech", "CDNS": "Tech",
    "ARM": "Tech", "SMCI": "Tech",
    "GOOGL": "Comm", "META": "Comm", "NFLX": "Comm", "DIS": "Comm", "T": "Comm", "VZ": "Comm",
    "AMZN": "Disc", "TSLA": "Disc", "HD": "Disc", "MCD": "Disc", "NKE": "Disc",
    "SBUX": "Disc", "LOW": "Disc", "TJX": "Disc", "BKNG": "Disc", "GM": "Disc",
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "COST": "Staples",
    "WMT": "Staples", "PM": "Staples", "CL": "Staples",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "MPC": "Energy",
    "JPM": "Fin", "BAC": "Fin", "WFC": "Fin", "GS": "Fin", "MS": "Fin",
    "BLK": "Fin", "AXP": "Fin", "C": "Fin", "USB": "Fin", "PNC": "Fin",
    "COIN": "Fin", "SQ": "Fin",
    "UNH": "Health", "JNJ": "Health", "LLY": "Health", "PFE": "Health",
    "ABBV": "Health", "TMO": "Health", "ABT": "Health", "MRK": "Health",
    "AMGN": "Health", "GILD": "Health",
    "CAT": "Indust", "HON": "Indust", "UPS": "Indust", "BA": "Indust",
    "RTX": "Indust", "DE": "Indust", "GE": "Indust", "LMT": "Indust",
    "UNP": "Indust", "FDX": "Indust",
    "LIN": "Mat", "APD": "Mat", "FCX": "Mat", "NEM": "Mat", "NUE": "Mat",
    "AMT": "RE", "PLD": "RE", "CCI": "RE", "EQIX": "RE", "SPG": "RE",
    "NEE": "Util", "DUK": "Util", "SO": "Util", "D": "Util", "AEP": "Util",
    "MARA": "Crypto-linked",
}

DCA_MONTHLY = 1000.0
TRADING_DAYS_YR = 252


@dataclass
class MarketData:
    all_dates: list
    date_idx: dict
    prices: dict
    finals: dict
    month_first_idx: list
    bench_filled: dict
    stocks: list
    items_by_ticker: dict


def load_market(full_json_path: str, bench_tickers=("SPY",)) -> MarketData:
    with open(full_json_path) as f:
        d = json.load(f)
    bt = d["bt_series"]

    max_end = ""
    for s in bt.values():
        if s["dates"] and s["dates"][-1] > max_end:
            max_end = s["dates"][-1]
    stale_cutoff = (datetime.fromisoformat(max_end) - timedelta(days=365)).date().isoformat()

    date_set = set()
    for tk, s in bt.items():
        if s["dates"] and s["dates"][-1] >= stale_cutoff:
            date_set.update(s["dates"])
    all_dates = sorted(date_set)
    date_idx = {dd: i for i, dd in enumerate(all_dates)}

    n = len(all_dates)
    prices, finals = {}, {}
    for tk, s in bt.items():
        p = np.full(n, np.nan)
        f = np.full(n, np.nan)
        for i, dd in enumerate(s["dates"]):
            if dd in date_idx:
                ix = date_idx[dd]
                p[ix] = s["prices"][i]
                f[ix] = s["final"][i]
        prices[tk] = p
        finals[tk] = f

    bench_filled = {}
    for tk in bench_tickers:
        if tk not in bt:
            continue
        s = bt[tk]
        seed = 0.0
        if all_dates:
            for dd, vv in zip(s["dates"], s["prices"]):
                if dd <= all_dates[0] and math.isfinite(vv) and vv > 0:
                    seed = vv
        arr = np.zeros(n)
        last = seed
        for i in range(n):
            pm = prices[tk][i] if tk in prices else np.nan
            if math.isfinite(pm) and pm > 0:
                last = pm
            arr[i] = last
        bench_filled[tk] = arr

    month_first_idx = []
    prev_ym = ""
    for i, dd in enumerate(all_dates):
        ym = dd[:7]
        if ym != prev_ym:
            month_first_idx.append(i)
            prev_ym = ym

    stocks = [
        it["ticker"] for it in d["items"]
        if not it.get("is_crypto") and it["ticker"] not in set(bench_tickers)
        and it["ticker"] in bt
    ]
    items_by_ticker = {it["ticker"]: it for it in d["items"]}

    return MarketData(
        all_dates=all_dates, date_idx=date_idx, prices=prices, finals=finals,
        month_first_idx=month_first_idx, bench_filled=bench_filled,
        stocks=stocks, items_by_ticker=items_by_ticker,
    )


def first_valid_month_idx(md: MarketData, universe: list, min_tickers: int = 3) -> int:
    for m, di in enumerate(md.month_first_idx):
        n = 0
        for tk in universe:
            p, f = md.prices.get(tk), md.finals.get(tk)
            if p is None or f is None:
                continue
            if math.isfinite(p[di]) and p[di] > 0 and math.isfinite(f[di]) and f[di] > 0:
                n += 1
                if n >= min_tickers:
                    return m
    return 0


def spy_200sma(md: MarketData) -> np.ndarray:
    p = md.bench_filled.get("SPY")
    if p is None:
        return np.full(len(md.all_dates), np.nan)
    sma = np.full(len(p), np.nan)
    w = 200
    for i in range(len(p)):
        if i + 1 >= w:
            sma[i] = float(np.mean(p[i + 1 - w : i + 1]))
    return sma


@dataclass
class StrategyConfig:
    top_n: int = 5
    score_threshold: Optional[float] = None
    hold_days: int = 252
    weighting: str = "equal"
    regime_gate: bool = False
    regime_scale_down: float = 0.0
    sector_cap: Optional[int] = None
    trail_stop: Optional[float] = None
    dd_stop: Optional[float] = None
    min_score: float = 0.0
    entry_delay: int = 1
    start_month_idx: Optional[int] = None
    universe_filter: Optional[set] = None  # restrict universe to this set if provided
    label: str = "strategy"


@dataclass
class SimResult:
    equity: np.ndarray
    total_invested: float
    positions: list = field(default_factory=list)
    picks_by_month: list = field(default_factory=list)  # list of (date, [(tk,score)])


def simulate(md: MarketData, universe: list, cfg: StrategyConfig) -> SimResult:
    n = len(md.all_dates)
    equity = np.zeros(n)
    positions = []
    picks_by_month = []
    cash = 0.0
    total_invested = 0.0

    eff_universe = [t for t in universe if not cfg.universe_filter or t in cfg.universe_filter]
    if cfg.start_month_idx is None:
        cfg.start_month_idx = first_valid_month_idx(md, eff_universe, min_tickers=3)
    sma = spy_200sma(md) if cfg.regime_gate else None

    for m in range(cfg.start_month_idx, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        monthly = DCA_MONTHLY
        if cfg.regime_gate and sma is not None:
            spy_p = md.bench_filled["SPY"][di]
            if math.isfinite(sma[di]) and spy_p < sma[di]:
                monthly = DCA_MONTHLY * cfg.regime_scale_down
                if monthly <= 0:
                    continue

        cand = []
        for tk in eff_universe:
            f = md.finals.get(tk); p = md.prices.get(tk)
            if f is None or p is None:
                continue
            fv, pv = f[di], p[di]
            if not (math.isfinite(fv) and fv > cfg.min_score and math.isfinite(pv) and pv > 0):
                continue
            cand.append((tk, fv, pv))
        if not cand:
            continue
        cand.sort(key=lambda x: x[1], reverse=True)

        if cfg.score_threshold is not None:
            picks = [c for c in cand if c[1] >= cfg.score_threshold][: cfg.top_n]
            if not picks:
                picks = cand[:1]
        else:
            picks = cand[: cfg.top_n]

        if cfg.sector_cap is not None:
            per_sector = {}
            filtered = []
            for tk, fv, pv in picks:
                sec = SECTOR_MAP.get(tk, "Other")
                if per_sector.get(sec, 0) < cfg.sector_cap:
                    filtered.append((tk, fv, pv))
                    per_sector[sec] = per_sector.get(sec, 0) + 1
            i = len(picks)
            while len(filtered) < cfg.top_n and i < len(cand):
                tk, fv, pv = cand[i]; i += 1
                sec = SECTOR_MAP.get(tk, "Other")
                if per_sector.get(sec, 0) < cfg.sector_cap:
                    filtered.append((tk, fv, pv))
                    per_sector[sec] = per_sector.get(sec, 0) + 1
            picks = filtered

        if not picks:
            continue

        if cfg.weighting == "equal":
            weights = [1.0 / len(picks)] * len(picks)
        elif cfg.weighting == "rank":
            raw = [1.0 / (i + 1) for i in range(len(picks))]
            s = sum(raw); weights = [r / s for r in raw]
        elif cfg.weighting == "score":
            raw = [max(p[1], 0.0) for p in picks]
            s = sum(raw)
            weights = [r / s for r in raw] if s > 0 else [1.0 / len(picks)] * len(picks)
        else:
            raise ValueError(cfg.weighting)

        entry_idx = di + cfg.entry_delay
        if entry_idx >= n:
            continue
        adj_picks = []
        adj_weights = []
        for (tk, fv, _), w in zip(picks, weights):
            ep = md.prices.get(tk)
            if ep is None:
                continue
            px = ep[entry_idx]
            if math.isfinite(px) and px > 0:
                adj_picks.append((tk, fv, px))
                adj_weights.append(w)
        if not adj_picks:
            continue
        # Renormalize weights if any picks dropped
        sw = sum(adj_weights)
        if sw <= 0:
            continue
        adj_weights = [w / sw for w in adj_weights]

        total_invested += monthly
        picks_by_month.append((md.all_dates[di], [(tk, fv) for tk, fv, _ in adj_picks]))
        for (tk, fv, pv), w in zip(adj_picks, adj_weights):
            alloc = monthly * w
            positions.append({
                "tk": tk, "buy_idx": entry_idx, "sell_idx": entry_idx + cfg.hold_days,
                "shares": alloc / pv, "cost": alloc, "buy_price": pv,
                "sold": False, "sell_price": 0.0, "peak": pv,
            })

    for d in range(n):
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px) and px > 0:
                if px > pos["peak"]:
                    pos["peak"] = px
                if cfg.trail_stop is not None and px <= pos["peak"] * (1 - cfg.trail_stop):
                    cash += pos["shares"] * px; pos["sold"] = True; pos["sell_price"] = px
                    continue
                if cfg.dd_stop is not None and px <= pos["buy_price"] * (1 - cfg.dd_stop):
                    cash += pos["shares"] * px; pos["sold"] = True; pos["sell_price"] = px
                    continue
            if d >= pos["sell_idx"] and math.isfinite(px) and px > 0:
                cash += pos["shares"] * px; pos["sold"] = True; pos["sell_price"] = px

        open_val = 0.0
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash + open_val

    return SimResult(equity=equity, total_invested=total_invested,
                     positions=positions, picks_by_month=picks_by_month)


def simulate_benchmark(md: MarketData, bench_tickers: list, hold_days: int,
                       start_month_idx: int, entry_delay: int = 1) -> SimResult:
    n = len(md.all_dates)
    equity = np.zeros(n)
    positions = []
    cash = 0.0
    total_invested = 0.0
    for m in range(start_month_idx, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        entry_idx = di + entry_delay
        if entry_idx >= n:
            break
        avail = [t for t in bench_tickers
                 if md.bench_filled.get(t) is not None and md.bench_filled[t][entry_idx] > 0]
        if not avail:
            continue
        per = DCA_MONTHLY / len(avail)
        total_invested += DCA_MONTHLY
        for tk in avail:
            p = md.bench_filled[tk][entry_idx]
            positions.append({
                "tk": tk, "buy_idx": entry_idx, "sell_idx": entry_idx + hold_days,
                "shares": per / p, "cost": per, "buy_price": p,
                "sold": False, "sell_price": 0.0, "peak": p,
            })
    for d in range(n):
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            if d >= pos["sell_idx"]:
                px = md.bench_filled[pos["tk"]][min(d, n - 1)]
                if px > 0:
                    cash += pos["shares"] * px; pos["sold"] = True; pos["sell_price"] = px
        open_val = 0.0
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            px = md.bench_filled[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash + open_val
    return SimResult(equity=equity, total_invested=total_invested, positions=positions)


def compute_metrics(md: MarketData, eq: np.ndarray, total_invested: float) -> dict:
    if eq.size == 0 or total_invested <= 0:
        return {}
    final = float(eq[-1])
    monthly_rets = []
    for i in range(1, len(md.month_first_idx)):
        prev = eq[md.month_first_idx[i - 1]]
        cur = eq[md.month_first_idx[i]]
        if prev > 0:
            monthly_rets.append(cur / prev - 1)
    arr = np.array(monthly_rets)
    avg = arr.mean() if arr.size else 0.0
    std = arr.std() if arr.size else 0.0
    sharpe = (avg / std) * math.sqrt(12) if std > 0 else 0.0
    peak, maxdd = 0.0, 0.0
    for v in eq:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > maxdd:
                maxdd = dd
    total_return = (final - total_invested) / total_invested
    yrs = len(md.all_dates) / TRADING_DAYS_YR
    cagr = (final / total_invested) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    return {
        "final": final, "invested": total_invested, "total_return": total_return,
        "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "n_months": len(arr),
    }


def fmt_pct(x):
    return f"{x*100:+.2f}%" if x is not None else "—"


def fmt_metrics(m: dict) -> str:
    return (f"CAGR {fmt_pct(m['cagr'])}  TR {fmt_pct(m['total_return'])}  "
            f"Sharpe {m['sharpe']:.2f}  MaxDD {fmt_pct(-m['maxdd'])}  "
            f"Inv ${m['invested']:,.0f}  Final ${m['final']:,.0f}")


def load_and_prep(full_json="/home/user/crt/max/docs/data/full.json"):
    md = load_market(full_json)
    start_m = first_valid_month_idx(md, md.stocks, min_tickers=3)
    return md, start_m
