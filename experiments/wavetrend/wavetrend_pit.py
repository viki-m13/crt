"""WaveTrend pyramiding on the PIT-corrected S&P 500 / NDX panels.

This is a SEPARATE experiment from the deployed v5 strategy. It takes the
WaveTrend pyramiding idea from the repo-root `wavetrend` file and rebuilds it
honestly on the point-in-time data the repo already has:

  * Universe = PIT index membership (S&P 500 or Nasdaq-100), so no
    survivorship / look-ahead universe selection. The original file picked
    "underperformers vs SPY" using FULL-SAMPLE final equity -> that is a
    look-ahead bug; it is removed here.
  * Prices = experiments/monthly_dca/cache/v2/sp500_pit/prices_extended_pit.parquet
    (daily auto-adjusted close, 1994 tickers). The panel has CLOSE ONLY (no
    High/Low), so the Pine `ap = hlc3` is approximated by adjusted close.
    This is causal and is exactly the series we would trade on EOD.
  * Signals are causal (ewm / rolling only look backward). A trade signalled
    using prices through day i is executed at the close of day i+1.
  * A monthly return stream is produced so the strategy can be evaluated
    apples-to-apples against the deployed v5 sleeve framework.

Indicator math is identical to the Pine in the repo-root `wavetrend` file:
    ap  = close (proxy for hlc3)
    esa = ema(ap, n1)
    d   = ema(|ap-esa|, n1)
    ci  = (ap-esa) / (0.015 d)
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, sig_len)
    BUY  when crossover(wt1, wt2) and wt1 <= oversold and ticker is a member
    EXIT when RSI(period) > overbought, or membership lost, or delisted
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
QQQ = ROOT / "experiments" / "monthly_dca" / "v5" / "qqq_pit"
OUT = ROOT / "experiments" / "wavetrend"

INITIAL_CAPITAL = 1_000_000.0
POSITION_SIZE = 10_000.0  # $ per pyramid unit (faithful to the original)

# ETF / leveraged tickers that may sit in the price panel but are never
# legitimate single-name picks.
EXCLUDE = {
    "SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
    "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
    "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD",
}


@dataclass
class Params:
    wt_n1: int = 60
    wt_n2: int = 140
    wt_sig_len: int = 4
    wt_oversold: float = -60.0
    rsi_period: int = 252
    rsi_overbought: float = 70.0

    def clamp(self) -> "Params":
        return Params(
            wt_n1=int(np.clip(round(self.wt_n1), 5, 200)),
            wt_n2=int(np.clip(round(self.wt_n2), 5, 300)),
            wt_sig_len=int(np.clip(round(self.wt_sig_len), 2, 15)),
            wt_oversold=float(np.clip(self.wt_oversold, -120.0, -10.0)),
            rsi_period=int(np.clip(round(self.rsi_period), 10, 360)),
            rsi_overbought=float(np.clip(self.rsi_overbought, 52.0, 90.0)),
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_prices() -> pd.DataFrame:
    px = pd.read_parquet(PIT / "prices_extended_pit.parquet")
    px.index = pd.to_datetime(px.index)
    return px.sort_index()


def load_membership(universe: str) -> pd.DataFrame:
    """Return monthly (asof, ticker) PIT membership for 'sp500' or 'ndx'."""
    if universe == "sp500":
        mem = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    elif universe == "ndx":
        mem = pd.read_parquet(QQQ / "ndx_pit_membership_monthly.parquet")
    else:
        raise ValueError(universe)
    mem["asof"] = pd.to_datetime(mem["asof"])
    return mem


def build_daily_membership(mem: pd.DataFrame, dates: pd.DatetimeIndex,
                           tickers: list[str]) -> pd.DataFrame:
    """Boolean (dates x tickers): True when the ticker is an index member.

    Monthly membership is forward-filled to daily: a name is tradable from the
    month-end it first appears through the month-end it last appears.
    """
    monthly_dates = sorted(mem["asof"].unique())
    grp = mem.groupby("asof")["ticker"].apply(set).to_dict()
    mask_m = pd.DataFrame(False, index=pd.DatetimeIndex(monthly_dates),
                          columns=tickers)
    for d in monthly_dates:
        present = [t for t in grp[d] if t in mask_m.columns]
        mask_m.loc[d, present] = True
    daily = mask_m.reindex(mask_m.index.union(dates)).ffill().reindex(dates)
    return daily.fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Indicators (vectorised across the whole panel)
# ---------------------------------------------------------------------------

def compute_wavetrend(close: pd.DataFrame, n1: int, n2: int, sig_len: int):
    ap = close
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d.replace(0.0, np.nan))
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(window=sig_len, min_periods=sig_len).mean()
    return wt1, wt2


def compute_rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# Event-driven pyramiding simulation
# ---------------------------------------------------------------------------

def simulate(close: pd.DataFrame, member: pd.DataFrame, p: Params,
             cost_bps: float = 10.0):
    """Run the pyramiding sim. Returns (daily_equity, trades_df).

    Execution: a signal computed from prices through day i is acted on at the
    close of day i+1 (no same-bar look-ahead). Mark-to-market daily.
    """
    p = p.clamp()
    wt1, wt2 = compute_wavetrend(close, p.wt_n1, p.wt_n2, p.wt_sig_len)
    rsi = compute_rsi(close, p.rsi_period)

    prev1, prev2 = wt1.shift(1), wt2.shift(1)
    cross_up = (prev1 < prev2) & (wt1 >= wt2)
    buy_sig = (cross_up & (wt1 <= p.wt_oversold)).fillna(False)
    exit_sig = (rsi > p.rsi_overbought).fillna(False)

    tickers = list(close.columns)
    dates = close.index
    px = close.values
    bs = buy_sig.values
    xs = exit_sig.values
    mb = member.reindex(index=dates, columns=tickers).fillna(False).values
    col = {t: j for j, t in enumerate(tickers)}
    n = len(dates)
    cf = cost_bps / 10000.0

    cash = INITIAL_CAPITAL
    positions: dict[int, list] = {}  # col_idx -> list of {entry_price, shares, entry_date}
    last_valid = np.full(len(tickers), np.nan)
    trades = []
    equity = np.empty(n)

    for i in range(n):
        row_px = px[i]
        valid = ~np.isnan(row_px)
        last_valid[valid] = row_px[valid]

        # ---- act on day i-1 signals at today's close ----
        if i >= 1:
            sig_buy = np.where(bs[i - 1])[0]
            sig_exit = set(np.where(xs[i - 1])[0])

            # EXITS: rsi exit, or membership lost, or delisted (price gone)
            for j in list(positions.keys()):
                lots = positions[j]
                if not lots:
                    continue
                lost_member = not mb[i, j]
                delisted = not valid[j]
                rsi_exit = j in sig_exit
                if rsi_exit or lost_member or delisted:
                    exit_px = row_px[j] if valid[j] else last_valid[j]
                    if np.isnan(exit_px):
                        continue
                    for lot in lots:
                        proceeds = lot["shares"] * exit_px * (1 - cf)
                        cash += proceeds
                        trades.append({
                            "ticker": tickers[j],
                            "entry_date": lot["entry_date"],
                            "exit_date": dates[i],
                            "entry_price": lot["entry_price"],
                            "exit_price": float(exit_px),
                            "ret": exit_px / lot["entry_price"] - 1.0,
                            "reason": "rsi" if rsi_exit else
                                      ("member" if lost_member else "delist"),
                        })
                    positions[j] = []

            # ENTRIES: pyramiding, members only, cash-budget gated
            for j in sig_buy:
                if not mb[i, j] or not valid[j] or row_px[j] <= 0:
                    continue
                if cash >= POSITION_SIZE:
                    shares = POSITION_SIZE / row_px[j]
                    cash -= POSITION_SIZE * (1 + cf)
                    positions.setdefault(j, []).append({
                        "entry_date": dates[i],
                        "entry_price": float(row_px[j]),
                        "shares": shares,
                    })

        # ---- mark to market ----
        mv = 0.0
        for j, lots in positions.items():
            if not lots:
                continue
            pr = row_px[j] if valid[j] else last_valid[j]
            if np.isnan(pr):
                continue
            for lot in lots:
                mv += lot["shares"] * pr
        equity[i] = cash + mv

    eq = pd.Series(equity, index=dates, name="equity")
    tr = pd.DataFrame(trades)
    return eq, tr


# ---------------------------------------------------------------------------
# Metrics  (mirrors experiments/monthly_dca/v5/spx_pit/run_v5_winner_aug.py)
# ---------------------------------------------------------------------------

def monthly_returns_from_equity(eq: pd.Series) -> pd.Series:
    m = eq.resample("ME").last()
    return m.pct_change().dropna()


def cagr_monthly(ret: pd.Series) -> float:
    if len(ret) == 0:
        return 0.0
    return float((1 + ret).cumprod().iloc[-1] ** (12.0 / len(ret)) - 1)


def sharpe_monthly(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(12))


def max_dd_monthly(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    if len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def metrics_block(ret: pd.Series) -> dict:
    return {
        "cagr": cagr_monthly(ret),
        "vol": float(ret.std() * np.sqrt(12)) if len(ret) > 1 else 0.0,
        "sharpe": sharpe_monthly(ret),
        "mdd": max_dd_monthly(ret),
        "n": int(len(ret)),
    }


def win_rate(tr: pd.DataFrame) -> float:
    if tr is None or tr.empty:
        return 0.0
    return float((tr["ret"] > 0).mean())


# ===========================================================================
# "Never sell": WaveTrend is ONLY an entry signal. Buy and hold forever.
# ===========================================================================
#
# Each WaveTrend oversold-cross on a current index member adds one equal-$
# unit of that stock; the unit is NEVER sold. A name that delists / is
# acquired (price -> NaN) is frozen at its last traded price (the honest
# cash-payout-at-last-price convention used everywhere else in this repo)
# and simply stops contributing P&L thereafter -- it is not a forced loss.
#
# The reported return stream is the TIME-WEIGHTED return of the
# accumulating book (new contributions are removed from the daily return
# so it is scale-free and directly comparable to the v5 / SPY monthly
# streams). Fully vectorised -> fast enough for wide param search + grids.
#
# Few parameters (n1, n2, sig_len, oversold, optional own-trend SMA gate)
# on purpose: fewer knobs + a plateau check is the anti-overfit lever.

@dataclass
class HParams:
    wt_n1: int = 60
    wt_n2: int = 140
    wt_sig_len: int = 4
    wt_oversold: float = -60.0
    trend_sma: int = 0  # 0 = off; else only buy if close > SMA(trend_sma)

    def clamp(self) -> "HParams":
        return HParams(
            wt_n1=int(np.clip(round(self.wt_n1), 5, 200)),
            wt_n2=int(np.clip(round(self.wt_n2), 5, 300)),
            wt_sig_len=int(np.clip(round(self.wt_sig_len), 2, 15)),
            wt_oversold=float(np.clip(self.wt_oversold, -120.0, -10.0)),
            trend_sma=int(np.clip(round(self.trend_sma), 0, 250)),
        )


def simulate_hold_forever(close: pd.DataFrame, member: pd.DataFrame,
                          p: HParams, unit: float = 1.0):
    """Buy-and-hold-forever on WaveTrend entries. Never sells.

    Returns dict with:
      ret_d   : daily time-weighted return of the accumulating book
      equity  : scale-free time-weighted equity curve (cumprod(1+ret_d))
      n_entries : number of units bought
      entry_winrate : fraction of units above water at end of sample
      contrib_curve : cumulative $ deployed (for the finite-capital view)
    """
    p = p.clamp()
    wt1, wt2 = compute_wavetrend(close, p.wt_n1, p.wt_n2, p.wt_sig_len)
    prev1, prev2 = wt1.shift(1), wt2.shift(1)
    cross_up = (prev1 < prev2) & (wt1 >= wt2)
    buy_sig = (cross_up & (wt1 <= p.wt_oversold)).fillna(False)
    if p.trend_sma > 0:
        trend_ok = (close > close.rolling(p.trend_sma,
                                          min_periods=p.trend_sma).mean())
        buy_sig = buy_sig & trend_ok.fillna(False)

    tickers = list(close.columns)
    dates = close.index
    px = close.values
    bs = buy_sig.values
    mb = member.reindex(index=dates, columns=tickers).fillna(False).values
    nT = len(tickers)
    n = len(dates)

    shares = np.zeros(nT)
    last_px = np.full(nT, np.nan)
    entry_cost = np.zeros(nT)   # cumulative $ put into each name
    entry_shares_log = []       # (col, entry_price) per unit, for win-rate
    ret_d = np.zeros(n)
    contrib = np.zeros(n)
    prev_mv = 0.0

    for i in range(n):
        row = px[i]
        valid = ~np.isnan(row)
        last_px[valid] = row[valid]

        c_today = 0.0
        if i >= 1:
            for j in np.where(bs[i - 1])[0]:
                if mb[i, j] and valid[j] and row[j] > 0:
                    sh = unit / row[j]
                    shares[j] += sh
                    entry_cost[j] += unit
                    c_today += unit
                    entry_shares_log.append((j, row[j]))

        eff = np.where(valid, row, last_px)
        held = shares > 0
        mv = float(np.nansum(shares[held] * eff[held])) if held.any() else 0.0
        contrib[i] = c_today
        if prev_mv > 0:
            ret_d[i] = (mv - c_today) / prev_mv - 1.0
        prev_mv = mv

    ret_s = pd.Series(ret_d, index=dates)
    eq = (1.0 + ret_s).cumprod()

    # per-unit win rate at end of sample (never sold -> mark at last price)
    wins = tot = 0
    for j, ep in entry_shares_log:
        fp = last_px[j]
        if not np.isnan(fp) and ep > 0:
            tot += 1
            if fp > ep:
                wins += 1
    entry_wr = wins / tot if tot else 0.0

    return {
        "ret_d": ret_s,
        "equity": eq,
        "n_entries": len(entry_shares_log),
        "entry_winrate": entry_wr,
        "contrib_curve": pd.Series(np.cumsum(contrib), index=dates),
    }


# ===========================================================================
# WT-Bottom: enhanced "absolute bottom" detector + depth-pyramiding, no sell
# ===========================================================================
#
# Creative enhancements over plain WaveTrend (Part 3):
#
#  1. ROLLING-RANGE BOTTOM QUARTILE. A WaveTrend oversold-cross only counts
#     as a *bottom* if the stock is also in the bottom `q_bottom` of its own
#     trailing `roll_window` price range:
#         pos = (close - rollmin) / (rollmax - rollmin)   in [0,1]
#         require pos <= q_bottom        (e.g. <=0.25 = bottom quartile)
#     This rejects "oversold but still mid-range" momentum dips and keeps
#     only genuine relative lows.
#
#  2. TREND-REGIME GATE. Only buy when close > SMA(trend_sma) -- accumulate
#     dips *inside* an uptrend, where buy-and-hold works (Part 3 showed this
#     gate is the load-bearing, generalising ingredient).
#
#  3. WAVETREND CURL confirmation: wt1 must be turning up (1 bar).
#
#  4. DEPTH-SCALED PYRAMIDING. One base unit per distinct bottom; extra
#     units the deeper the drawdown from the rolling high:
#         extra = min(MAX_ADD, floor((-dd) / dd_step))
#     so deeper bottoms get more conviction -- but only within an uptrend.
#
#  5. RE-ARM. After a buy a name is disarmed until it climbs back above the
#     mid of its rolling range (pos > REARM), so each bottom is a distinct
#     event and we don't buy every day of a long oversold stretch.
#
# Never sells. Delisted/acquired names frozen at last price. Scale-free
# time-weighted return stream (same accounting as simulate_hold_forever).

MAX_ADD = 4      # cap on depth-scaled extra units (fixed, not searched)
REARM = 0.50     # rolling-range pos that re-arms a name (fixed)


@dataclass
class BParams:
    wt_n1: int = 60
    wt_n2: int = 60
    wt_sig_len: int = 4
    wt_oversold: float = -45.0
    roll_window: int = 120      # trailing window for the rolling range
    q_bottom: float = 0.25      # bottom-quartile threshold on rolling pos
    trend_sma: int = 0          # stock uptrend gate (0 = off)
    mkt_sma: int = 200          # SPY-regime uptrend gate (0 = off)
    dd_step: float = 0.10       # extra unit per this much drawdown depth

    def clamp(self) -> "BParams":
        return BParams(
            wt_n1=int(np.clip(round(self.wt_n1), 5, 200)),
            wt_n2=int(np.clip(round(self.wt_n2), 5, 300)),
            wt_sig_len=int(np.clip(round(self.wt_sig_len), 2, 15)),
            wt_oversold=float(np.clip(self.wt_oversold, -120.0, -5.0)),
            roll_window=int(np.clip(round(self.roll_window), 30, 504)),
            q_bottom=float(np.clip(self.q_bottom, 0.05, 0.6)),
            trend_sma=int(np.clip(round(self.trend_sma), 0, 250)),
            mkt_sma=int(np.clip(round(self.mkt_sma), 0, 250)),
            dd_step=float(np.clip(self.dd_step, 0.03, 0.5)),
        )


def simulate_bottom_accumulate(close: pd.DataFrame, member: pd.DataFrame,
                               p: BParams, spy: pd.Series | None = None,
                               unit: float = 1.0):
    """WT-Bottom accumulation. Never sells. Depth-scaled pyramiding."""
    p = p.clamp()
    wt1, wt2 = compute_wavetrend(close, p.wt_n1, p.wt_n2, p.wt_sig_len)
    prev1, prev2 = wt1.shift(1), wt2.shift(1)
    cross_up = (prev1 < prev2) & (wt1 >= wt2)
    curl = (wt1 > wt1.shift(1)).fillna(False)
    base_sig = (cross_up & (wt1 <= p.wt_oversold) & curl).fillna(False)

    rmax = close.rolling(p.roll_window, min_periods=p.roll_window // 2).max()
    rmin = close.rolling(p.roll_window, min_periods=p.roll_window // 2).min()
    span = (rmax - rmin)
    pos = ((close - rmin) / span.where(span > 0)).clip(0, 1)
    pos = pos.fillna(0.5)
    dd = (close / rmax - 1.0).fillna(0.0)            # <= 0
    if p.trend_sma > 0:
        trend_ok = (close > close.rolling(
            p.trend_sma, min_periods=p.trend_sma).mean()).fillna(False)
    else:
        trend_ok = pd.DataFrame(True, index=close.index,
                                columns=close.columns)

    tickers = list(close.columns)
    dates = close.index
    px = close.values
    bs = base_sig.values
    posv = pos.values
    ddv = dd.values
    tk_ok = trend_ok.reindex(index=dates, columns=tickers).fillna(False).values
    mb = member.reindex(index=dates, columns=tickers).fillna(False).values

    # SPY-regime gate: buy stock dips only when the market is trending up
    if spy is not None and p.mkt_sma > 0:
        sp = spy.reindex(dates).ffill()
        mkt_ok = (sp > sp.rolling(p.mkt_sma,
                                  min_periods=p.mkt_sma).mean()).fillna(False)
        mkt_ok = mkt_ok.values
    else:
        mkt_ok = np.ones(len(dates), dtype=bool)
    nT = len(tickers)
    n = len(dates)

    shares = np.zeros(nT)
    last_px = np.full(nT, np.nan)
    armed = np.ones(nT, dtype=bool)
    entry_log = []
    ret_d = np.zeros(n)
    contrib = np.zeros(n)
    prev_mv = 0.0

    for i in range(n):
        row = px[i]
        valid = ~np.isnan(row)
        last_px[valid] = row[valid]

        c_today = 0.0
        if i >= 1:
            s = i - 1
            for j in np.where(bs[s])[0]:
                if not (armed[j] and mb[i, j] and valid[j] and row[j] > 0):
                    continue
                if posv[s, j] > p.q_bottom or not tk_ok[s, j]:
                    continue
                if not mkt_ok[s]:
                    continue
                extra = min(MAX_ADD, int((-ddv[s, j]) / p.dd_step))
                nu = 1 + max(0, extra)
                amt = unit * nu
                shares[j] += amt / row[j]
                c_today += amt
                armed[j] = False
                entry_log.append((j, row[j], nu))
            # re-arm names that have climbed back out of the bottom
            armed[posv[i] > REARM] = True

        eff = np.where(valid, row, last_px)
        held = shares > 0
        mv = float(np.nansum(shares[held] * eff[held])) if held.any() else 0.0
        contrib[i] = c_today
        if prev_mv > 0:
            ret_d[i] = (mv - c_today) / prev_mv - 1.0
        prev_mv = mv

    ret_s = pd.Series(ret_d, index=dates)
    eq = (1.0 + ret_s).cumprod()
    wins = tot = 0
    units_total = 0
    for j, ep, nu in entry_log:
        units_total += nu
        fp = last_px[j]
        if not np.isnan(fp) and ep > 0:
            tot += 1
            if fp > ep:
                wins += 1
    return {
        "ret_d": ret_s,
        "equity": eq,
        "n_events": len(entry_log),
        "n_units": units_total,
        "entry_winrate": (wins / tot) if tot else 0.0,
        "contrib_curve": pd.Series(np.cumsum(contrib), index=dates),
    }


# ===========================================================================
# Filtered variant: creative trade filters to push win-rate up HONESTLY
# ===========================================================================
#
# A high win-rate is trivial if you allow "sell winners fast, hold losers
# forever" (that is the Result-1 trap). The honest lever is *trade
# selection*: enter only high-probability dips and bound each trade so the
# loss tail is small. Filters available (each 0/off-able):
#
#   trend_sma   only buy if close > its own SMA(trend_sma)  -> no falling knives
#   mkt_sma     only buy if SPY > SMA(mkt_sma)              -> no bear-market dips
#   rs_lookback only buy if stock's N-day return > SPY's    -> relative strength
#   confirm     require wt1 to have risen `confirm` bars    -> turn confirmation
#   profit_take per-lot exit at +profit_take                -> bank the bounce
#   stop_loss   per-lot exit at -stop_loss                  -> cap the loss tail
#   max_hold    per-lot time stop in trading days           -> no dead money
#
# RSI / membership-loss / delist still force a whole-ticker exit.

@dataclass
class FParams:
    wt_n1: int = 45
    wt_n2: int = 80
    wt_sig_len: int = 4
    wt_oversold: float = -50.0
    rsi_period: int = 200
    rsi_overbought: float = 75.0
    trend_sma: int = 0          # 0 = off
    mkt_sma: int = 0            # 0 = off
    rs_lookback: int = 0        # 0 = off
    confirm: int = 0            # 0 = off
    profit_take: float = 0.0    # 0 = off
    stop_loss: float = 0.0      # 0 = off (value is the positive loss fraction)
    max_hold: int = 0           # 0 = off
    no_pyramid: int = 0         # 1 = at most one open lot per ticker

    def clamp(self) -> "FParams":
        def i(v, lo, hi):
            return int(np.clip(round(v), lo, hi))
        return FParams(
            wt_n1=i(self.wt_n1, 5, 200),
            wt_n2=i(self.wt_n2, 5, 300),
            wt_sig_len=i(self.wt_sig_len, 2, 15),
            wt_oversold=float(np.clip(self.wt_oversold, -120.0, -10.0)),
            rsi_period=i(self.rsi_period, 10, 360),
            rsi_overbought=float(np.clip(self.rsi_overbought, 52.0, 92.0)),
            trend_sma=i(self.trend_sma, 0, 250),
            mkt_sma=i(self.mkt_sma, 0, 250),
            rs_lookback=i(self.rs_lookback, 0, 252),
            confirm=i(self.confirm, 0, 5),
            profit_take=float(np.clip(self.profit_take, 0.0, 1.0)),
            stop_loss=float(np.clip(self.stop_loss, 0.0, 0.9)),
            max_hold=i(self.max_hold, 0, 756),
            no_pyramid=i(self.no_pyramid, 0, 1),
        )


def simulate_filtered(close: pd.DataFrame, member: pd.DataFrame,
                      spy: pd.Series, p: FParams, cost_bps: float = 10.0):
    """Pyramiding sim with entry filters + per-lot risk exits.

    `spy` is the daily SPY close (for the market-trend and RS filters).
    Execution: signal through day i-1 acted on at close of day i (no
    same-bar look-ahead). Returns (daily_equity, trades_df).
    """
    p = p.clamp()
    wt1, wt2 = compute_wavetrend(close, p.wt_n1, p.wt_n2, p.wt_sig_len)
    rsi = compute_rsi(close, p.rsi_period)

    prev1, prev2 = wt1.shift(1), wt2.shift(1)
    cross_up = (prev1 < prev2) & (wt1 >= wt2)
    buy_sig = (cross_up & (wt1 <= p.wt_oversold)).fillna(False)
    if p.confirm > 0:
        rising = (wt1 > wt1.shift(1))
        for k in range(1, p.confirm):
            rising = rising & (wt1.shift(k) > wt1.shift(k + 1))
        buy_sig = buy_sig & rising.fillna(False)
    exit_sig = (rsi > p.rsi_overbought).fillna(False)

    tickers = list(close.columns)
    dates = close.index
    px = close.values
    bs = buy_sig.values
    xs = exit_sig.values
    mb = member.reindex(index=dates, columns=tickers).fillna(False).values

    # Entry-quality filter masks (all causal, shifted by 1 day at use)
    if p.trend_sma > 0:
        trend_ok = (close > close.rolling(p.trend_sma,
                                          min_periods=p.trend_sma).mean())
        trend_ok = trend_ok.fillna(False).values
    else:
        trend_ok = None
    spy_d = spy.reindex(dates).ffill()
    if p.mkt_sma > 0:
        mkt_ok = (spy_d > spy_d.rolling(p.mkt_sma,
                                        min_periods=p.mkt_sma).mean())
        mkt_ok = mkt_ok.fillna(False).values
    else:
        mkt_ok = None
    if p.rs_lookback > 0:
        stk_mom = close.pct_change(p.rs_lookback)
        spy_mom = spy_d.pct_change(p.rs_lookback)
        rs_ok = stk_mom.sub(spy_mom, axis=0).gt(0.0).fillna(False).values
    else:
        rs_ok = None

    n = len(dates)
    cf = cost_bps / 10000.0
    cash = INITIAL_CAPITAL
    positions: dict[int, list] = {}
    last_valid = np.full(len(tickers), np.nan)
    trades = []
    equity = np.empty(n)

    for i in range(n):
        row_px = px[i]
        valid = ~np.isnan(row_px)
        last_valid[valid] = row_px[valid]

        if i >= 1:
            s = i - 1
            sig_exit = set(np.where(xs[s])[0])
            sig_buy = np.where(bs[s])[0]

            for j in list(positions.keys()):
                lots = positions[j]
                if not lots:
                    continue
                lost_member = not mb[i, j]
                delisted = not valid[j]
                rsi_exit = j in sig_exit
                cur_px = row_px[j] if valid[j] else last_valid[j]
                if np.isnan(cur_px):
                    continue
                keep = []
                for lot in lots:
                    r = cur_px / lot["entry_price"] - 1.0
                    held = i - lot["entry_idx"]
                    reason = None
                    if delisted:
                        reason = "delist"
                    elif lost_member:
                        reason = "member"
                    elif p.stop_loss > 0 and r <= -p.stop_loss:
                        reason = "stop"
                    elif p.profit_take > 0 and r >= p.profit_take:
                        reason = "take"
                    elif p.max_hold > 0 and held >= p.max_hold:
                        reason = "time"
                    elif rsi_exit:
                        reason = "rsi"
                    if reason is None:
                        keep.append(lot)
                        continue
                    cash += lot["shares"] * cur_px * (1 - cf)
                    trades.append({
                        "ticker": tickers[j],
                        "entry_date": lot["entry_date"],
                        "exit_date": dates[i],
                        "entry_price": lot["entry_price"],
                        "exit_price": float(cur_px),
                        "ret": r, "reason": reason,
                    })
                positions[j] = keep

            for j in sig_buy:
                if not mb[i, j] or not valid[j] or row_px[j] <= 0:
                    continue
                if p.no_pyramid and positions.get(j):
                    continue
                if trend_ok is not None and not trend_ok[s, j]:
                    continue
                if mkt_ok is not None and not mkt_ok[s]:
                    continue
                if rs_ok is not None and not rs_ok[s, j]:
                    continue
                if cash >= POSITION_SIZE:
                    cash -= POSITION_SIZE * (1 + cf)
                    positions.setdefault(j, []).append({
                        "entry_date": dates[i],
                        "entry_idx": i,
                        "entry_price": float(row_px[j]),
                        "shares": POSITION_SIZE / row_px[j],
                    })

        mv = 0.0
        for j, lots in positions.items():
            if not lots:
                continue
            pr = row_px[j] if valid[j] else last_valid[j]
            if np.isnan(pr):
                continue
            for lot in lots:
                mv += lot["shares"] * pr
        equity[i] = cash + mv

    return pd.Series(equity, index=dates, name="equity"), pd.DataFrame(trades)
