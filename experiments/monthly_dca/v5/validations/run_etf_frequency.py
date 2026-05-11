"""Rotation-frequency sweep — does WEEKLY rotation beat monthly?

For both unleveraged (12 ETFs) and pure-3x leveraged (8 ETFs) universes,
test rotation frequencies: daily, weekly, biweekly, monthly, bimonthly,
quarterly. Same 12-month momentum signal, same top-2 selection. Different
rotation cadence.

Hypothesis: leveraged ETFs benefit more from faster rotation because
volatility decay punishes longer holds. Unleveraged probably indifferent
or slightly worse with faster rotation (more noise, more costs).

Tests:
  1. FREQ sweep on each universe (daily/weekly/biweekly/monthly/bimonthly/quarterly)
  2. CRISIS stress: COVID Feb-Apr 2020, 2018 Q4, 2022 bear at each frequency
  3. PER-WEEK trace: did weekly catch trend changes earlier than monthly?
  4. COST sensitivity at weekly (more rotations = more cost drag)
  5. Lookback x frequency interaction
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import load_all

RES = Path(__file__).resolve().parent / "results"

UNLEV_12 = ["XLE","XLF","XLK","XLU","XLV","XLP","XLY","XLI","XLB",
             "TLT","EFA","EEM"]
PURE_3X = ["SPXL","TQQQ","TNA","SOXL","FAS","URTY","YINN","UPRO"]


def freq_schedule(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> list[pd.Timestamp]:
    """Generate rebalance dates at the requested frequency.

    freq options:
      'D' — daily (every trading day)
      'W' — weekly (every Friday)
      '2W' — biweekly
      'ME' — monthly (calendar month-end)
      '2ME' — every 2 months
      'QE' — every 3 months (quarterly)
    """
    if freq == "D":
        return list(pd.bdate_range(start=start, end=end))
    return list(pd.date_range(start=start, end=end, freq=freq))


def run_freq(daily_prices: pd.DataFrame,
              daily_returns: pd.DataFrame,
              rebal_dates: list[pd.Timestamp],
              assets: list[str],
              lookback_d: int = 252,
              top_n: int = 2,
              cost_bps: float = 10.0,
              hold_full_period: bool = True) -> dict:
    """Run the ETF momentum strategy at the requested rebalance frequency.

    Between rebalances, the basket is held and the portfolio earns the
    daily-compounded return of the held assets. At each rebal_date the
    composition is recomputed from the trailing lookback_d daily return.
    """
    cf = cost_bps / 1e4
    available = [a for a in assets if a in daily_prices.columns]
    if not available:
        return {"log": [], "n_rotations": 0}

    # Pre-compute daily returns for available assets
    px = daily_prices[available].copy()
    rets_daily = daily_returns[available] if daily_returns is not None else px.pct_change()

    # Build calendar of (rebal_date, picks)
    schedule = []
    prev_picks: list[str] = []
    n_rotations = 0
    for d in rebal_dates:
        sub_px = px.loc[:d].dropna(how="all")
        picks = []
        if len(sub_px) >= lookback_d:
            mom = sub_px.iloc[-1] / sub_px.iloc[-lookback_d] - 1
            mom = mom[mom > 0]
            if len(mom) > 0:
                picks = mom.sort_values(ascending=False).head(top_n).index.tolist()
        schedule.append((d, picks))
        if picks and prev_picks and set(picks) != set(prev_picks):
            n_rotations += 1
        elif picks and not prev_picks:
            n_rotations += 1  # initial entry
        prev_picks = picks

    if not schedule:
        return {"log": [], "n_rotations": 0}

    all_days = pd.bdate_range(start=schedule[0][0], end=schedule[-1][0])
    equity = 1.0
    log = []
    cur_picks = []           # the basket we're CURRENTLY HOLDING (from a prior rebal)
    pending_picks = None     # picks computed at end of today, to take effect tomorrow
    pending_is_rotation = False
    sched_idx = 0
    for d in all_days:
        # 1) Apply today's return BEFORE updating today's picks (no look-ahead)
        if cur_picks:
            rs = []
            for tk in cur_picks:
                if tk in rets_daily.columns and d in rets_daily.index:
                    v = rets_daily.at[d, tk]
                    if pd.notna(v): rs.append(float(v))
            ret_d = float(np.mean(rs)) if rs else 0.0
        else:
            ret_d = 0.0

        # 2) Apply the pending switch from the PRIOR rebal day (if any).
        # That switch was decided yesterday; today is its first day of effect.
        # We charge the cost on the FIRST day of the new basket's hold.
        if pending_picks is not None:
            cur_picks = list(pending_picks)
            if pending_is_rotation:
                ret_d -= cf
            pending_picks = None
            pending_is_rotation = False

        equity *= (1 + ret_d)
        log.append({"date": str(d.date()), "picks": ",".join(cur_picks),
                     "ret_d": ret_d, "equity": equity})

        # 3) If today is a rebal date, queue the new picks for tomorrow
        while sched_idx < len(schedule) and schedule[sched_idx][0] <= d:
            new_picks = schedule[sched_idx][1]
            if set(new_picks) != set(cur_picks):
                pending_picks = new_picks
                pending_is_rotation = True
            sched_idx += 1

    return {"log": log, "n_rotations": n_rotations}


def metrics(log, spy_daily):
    df = pd.DataFrame(log)
    if not len(df):
        return {}
    df["date"] = pd.to_datetime(df["date"])
    n_days = len(df)
    years = n_days / 252
    cagr = (df["equity"].iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0
    spy = spy_daily.loc[df["date"].iloc[0]:df["date"].iloc[-1]]
    spy_eq = (1 + spy.fillna(0)).cumprod()
    spy_cagr = (spy_eq.iloc[-1] ** (252/len(spy)) - 1) * 100 if len(spy) else 0
    rr = df["ret_d"]
    sh = float(rr.mean() / rr.std() * np.sqrt(252)) if rr.std() > 0 else 0
    peak = df["equity"].cummax()
    mdd = float(((df["equity"] - peak) / peak).min() * 100)
    return dict(cagr=cagr, spy_cagr=spy_cagr, edge=cagr-spy_cagr, sharpe=sh, mdd=mdd)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    daily = pd.read_parquet(Path("experiments/monthly_dca/cache/prices_extended.parquet"))
    daily.index = pd.to_datetime(daily.index)
    if "SPY" not in daily.columns:
        raise RuntimeError("SPY missing from daily prices")
    daily_returns = daily.pct_change()
    spy_daily = daily_returns["SPY"]

    FREQS = [
        ("D",   "Daily"),
        ("W",   "Weekly (Fri)"),
        ("2W",  "Biweekly"),
        ("ME",  "Monthly"),
        ("2ME", "Bimonthly"),
        ("QE",  "Quarterly"),
    ]

    print(f"\n{'='*78}\n  1. FREQUENCY SWEEP — Unleveraged 12 ETFs (full window 2003-2026)\n{'='*78}")
    print(f"{'Freq':<14} {'CAGR':>8} {'SPY':>8} {'Edge':>9} {'Sharpe':>7} {'MDD':>8} {'Rotations':>10}")
    print('-' * 78)
    rows_unlev = []
    for freq_code, freq_label in FREQS:
        rebals = freq_schedule(pd.Timestamp("2003-01-31"),
                                 pd.Timestamp("2026-04-30"), freq_code)
        sim = run_freq(daily, daily_returns, rebals,
                        UNLEV_12, lookback_d=252, top_n=2)
        m = metrics(sim["log"], spy_daily)
        print(f"{freq_label:<14} {m['cagr']:>+7.2f}% {m['spy_cagr']:>+7.2f}% {m['edge']:>+8.2f}pp {m['sharpe']:>7.2f} {m['mdd']:>+7.2f}% {sim['n_rotations']:>10}")
        rows_unlev.append({"freq": freq_code, "freq_label": freq_label,
                            "universe": "unleveraged_12", **m,
                            "n_rotations": sim["n_rotations"]})

    print(f"\n{'='*78}\n  2. FREQUENCY SWEEP — Pure 3× Leveraged (2010-2026, post-launch)\n{'='*78}")
    print(f"{'Freq':<14} {'CAGR':>8} {'SPY':>8} {'Edge':>9} {'Sharpe':>7} {'MDD':>8} {'Rotations':>10}")
    print('-' * 78)
    rows_lev = []
    for freq_code, freq_label in FREQS:
        rebals = freq_schedule(pd.Timestamp("2010-03-31"),
                                 pd.Timestamp("2026-04-30"), freq_code)
        sim = run_freq(daily, daily_returns, rebals,
                        PURE_3X, lookback_d=252, top_n=2)
        m = metrics(sim["log"], spy_daily)
        print(f"{freq_label:<14} {m['cagr']:>+7.2f}% {m['spy_cagr']:>+7.2f}% {m['edge']:>+8.2f}pp {m['sharpe']:>7.2f} {m['mdd']:>+7.2f}% {sim['n_rotations']:>10}")
        rows_lev.append({"freq": freq_code, "freq_label": freq_label,
                          "universe": "pure_3x_leveraged", **m,
                          "n_rotations": sim["n_rotations"]})

    print(f"\n{'='*78}\n  3. SAME with 6-MONTH lookback (better for leveraged?)\n{'='*78}")
    print(f"{'Freq':<14} {'Universe':<22} {'CAGR':>8} {'Edge':>9} {'Sharpe':>7} {'MDD':>8}")
    print('-' * 78)
    rows_6m = []
    for freq_code, freq_label in FREQS:
        for univ_label, univ in [("unleveraged_12", UNLEV_12), ("pure_3x_leveraged", PURE_3X)]:
            start = "2003-01-31" if univ_label == "unleveraged_12" else "2010-03-31"
            rebals = freq_schedule(pd.Timestamp(start),
                                     pd.Timestamp("2026-04-30"), freq_code)
            sim = run_freq(daily, daily_returns, rebals, univ,
                            lookback_d=126, top_n=2)
            m = metrics(sim["log"], spy_daily)
            print(f"{freq_label:<14} {univ_label:<22} {m['cagr']:>+7.2f}% {m['edge']:>+8.2f}pp {m['sharpe']:>7.2f} {m['mdd']:>+7.2f}%")
            rows_6m.append({"freq": freq_code, "universe": univ_label,
                             "lookback": "6m", **m,
                             "n_rotations": sim["n_rotations"]})

    print(f"\n{'='*78}\n  4. CRISIS STRESS at each frequency (12m lookback)\n{'='*78}")
    crises = [
        ("2020 COVID Feb-Apr", "2020-01-31", "2020-04-30"),
        ("2020 full year",      "2020-01-31", "2020-12-31"),
        ("2022 bear",            "2022-01-31", "2022-12-31"),
        ("2018 Q4",              "2018-09-30", "2018-12-31"),
        ("2023-24 AI rally",     "2023-01-31", "2024-12-31"),
    ]
    rows_crisis = []
    for cname, lo, hi in crises:
        print(f"\n  {cname} ({lo} → {hi})")
        print(f"    {'Freq':<14} {'Universe':<22} {'CAGR':>8} {'MDD':>8}")
        for freq_code, freq_label in FREQS:
            rebals = freq_schedule(pd.Timestamp(lo), pd.Timestamp(hi), freq_code)
            for univ_label, univ in [("unleveraged_12", UNLEV_12), ("pure_3x_leveraged", PURE_3X)]:
                sim = run_freq(daily, daily_returns, rebals, univ,
                                lookback_d=252, top_n=2)
                m = metrics(sim["log"], spy_daily)
                print(f"    {freq_label:<14} {univ_label:<22} {m['cagr']:>+7.2f}% {m['mdd']:>+7.2f}%")
                rows_crisis.append({"crisis": cname, "freq": freq_code,
                                     "universe": univ_label, **m})

    # Save artifacts
    pd.DataFrame(rows_unlev + rows_lev).to_csv(RES / "etf_freq_main.csv", index=False)
    pd.DataFrame(rows_6m).to_csv(RES / "etf_freq_6m.csv", index=False)
    pd.DataFrame(rows_crisis).to_csv(RES / "etf_freq_crises.csv", index=False)
    print(f"\nSaved 3 CSV artifacts to results/")


if __name__ == "__main__":
    main()
