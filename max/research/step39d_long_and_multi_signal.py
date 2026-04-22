"""Step 39d: long-window smoothing + smoothing other signals.

Step 39c found SMA 24M > SMA 12M > SMA 6M monotonically. Test:
  - SMA 30M, 36M, 48M — does improvement plateau or keep growing?
  - Smoothing `wash` instead of `final`
  - Smoothing `final_raw` (pre-gate score)
  - Double smoothing: smooth the smoothed score further
  - Does SMA 24M hold in tight rolling windows (3Y, 5Y)?
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bt_core import (simulate_benchmark, compute_metrics, StrategyConfig,
                     DCA_MONTHLY, first_valid_month_idx, SimResult)
from bt_core_ext import load_and_prep_ext


md, start_m = load_and_prep_ext()
cfg = StrategyConfig(start_month_idx=start_m, top_n=5, max_ticker_frac=0.05,
                     hold_days=5000, weighting="rank", entry_delay=1)

print(f"Loaded {len(md.stocks)} stocks, {len(md.all_dates)} dates")
print(f"Has washes: {md.washes is not None}  finals_raw: {md.finals_raw is not None}")


def simulate_custom_score(md, universe, cfg, score_fn):
    """Generic simulator — score_fn(tk, di) -> (ok, score, price).

    Replicates simulate_smoothed but lets score_fn decide entirely.
    """
    n = len(md.all_dates)
    equity = np.zeros(n)
    positions = []
    picks_by_month = []
    total_invested = 0.0
    cash = 0.0

    eff_universe = [t for t in universe]

    for m in range(cfg.start_month_idx, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        monthly = DCA_MONTHLY

        cand = []
        for tk in eff_universe:
            ok, sc, pv = score_fn(tk, di)
            if not ok:
                continue
            cand.append((tk, sc, pv))
        if not cand:
            continue
        cand.sort(key=lambda x: x[1], reverse=True)
        picks = cand[: cfg.top_n]

        if cfg.max_ticker_frac is not None:
            invested_after = total_invested + monthly
            cap_dollars = cfg.max_ticker_frac * invested_after
            by_tk = {}
            for pos in positions:
                by_tk[pos["tk"]] = by_tk.get(pos["tk"], 0.0) + pos["cost"]
            def _ok(tk): return by_tk.get(tk, 0.0) < cap_dollars
            filtered = [c for c in picks if _ok(c[0])]
            i = len(picks); seen = set(c[0] for c in filtered)
            while len(filtered) < cfg.top_n and i < len(cand):
                c = cand[i]; i += 1
                if c[0] in seen: continue
                if _ok(c[0]):
                    filtered.append(c); seen.add(c[0])
            picks = filtered
            if not picks: continue

        if cfg.weighting == "rank":
            raw = [1.0 / (i + 1) for i in range(len(picks))]
            s = sum(raw); weights = [r / s for r in raw]
        else:
            weights = [1.0 / len(picks)] * len(picks)

        entry_idx = di + cfg.entry_delay
        if entry_idx >= n: continue
        adj, adj_w = [], []
        for (tk, fv, _), w in zip(picks, weights):
            ep = md.prices.get(tk)
            if ep is None: continue
            px = ep[entry_idx]
            if math.isfinite(px) and px > 0:
                adj.append((tk, fv, px))
                adj_w.append(w)
        if not adj: continue
        sw = sum(adj_w)
        if sw <= 0: continue
        adj_w = [w / sw for w in adj_w]

        total_invested += monthly
        picks_by_month.append((md.all_dates[di], [(tk, fv) for tk, fv, _ in adj]))
        for (tk, fv, pv), w in zip(adj, adj_w):
            alloc = monthly * w
            positions.append({
                "tk": tk, "buy_idx": entry_idx, "sell_idx": entry_idx + cfg.hold_days,
                "shares": alloc / pv, "cost": alloc, "buy_price": pv,
                "sold": False, "sell_price": 0.0, "peak": pv,
            })

    for d in range(n):
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]: continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px) and px > 0 and px > pos["peak"]:
                pos["peak"] = px
            if d >= pos["sell_idx"] and math.isfinite(px) and px > 0:
                cash += pos["shares"] * px; pos["sold"] = True; pos["sell_price"] = px
        open_val = 0.0
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]: continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash + open_val
    return SimResult(equity=equity, total_invested=total_invested,
                     positions=positions, picks_by_month=picks_by_month)


def make_mean_score_fn(signal_dict, months):
    w_d = months * 21
    def fn(tk, di):
        arr = signal_dict.get(tk)
        px = md.prices.get(tk)
        if arr is None or px is None: return (False, 0.0, 0.0)
        pv = px[di]
        if not (math.isfinite(pv) and pv > 0): return (False, 0.0, 0.0)
        lo = max(0, di - w_d)
        vals = arr[lo:di + 1]
        valid = vals[np.isfinite(vals)]
        if len(valid) == 0: return (False, 0.0, 0.0)
        sc = float(np.mean(valid))
        if not (math.isfinite(sc) and sc > 0): return (False, 0.0, 0.0)
        return (True, sc, pv)
    return fn


def make_current_fn(signal_dict):
    def fn(tk, di):
        arr = signal_dict.get(tk)
        px = md.prices.get(tk)
        if arr is None or px is None: return (False, 0.0, 0.0)
        pv = px[di]
        sc = arr[di] if di < len(arr) else float("nan")
        if not (math.isfinite(pv) and pv > 0 and math.isfinite(sc) and sc > 0):
            return (False, 0.0, 0.0)
        return (True, float(sc), pv)
    return fn


def make_double_smooth_fn(signal_dict, m1, m2):
    """Smooth with window m1, then smooth that over window m2 (day-level)."""
    w1_d = m1 * 21
    w2_d = m2 * 21

    # Precompute first-pass smoothed series per ticker
    smoothed = {}
    for tk, arr in signal_dict.items():
        out = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            lo = max(0, i - w1_d)
            vals = arr[lo:i + 1]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                out[i] = float(np.mean(valid))
        smoothed[tk] = out

    return make_mean_score_fn(smoothed, m2)


# Long windows on `final`
print("\n" + "=" * 80)
print("## Long-window smoothing on `final`")
print("=" * 80)
for months in [0, 6, 12, 18, 24, 30, 36, 48, 60]:
    if months == 0:
        sf = make_current_fn(md.finals)
        label = f"current (incumbent)"
    else:
        sf = make_mean_score_fn(md.finals, months)
        label = f"SMA {months}M"
    r = simulate_custom_score(md, md.stocks, cfg, sf)
    m = compute_metrics(md, r.equity, r.total_invested)
    print(f"  {label:25s}  CAGR {m['cagr']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  "
          f"MaxDD {-m['maxdd']*100:+.2f}%  Final ${m['final']:11,.0f}")


# Smoothing alternative signals
if md.washes is not None and md.finals_raw is not None:
    print("\n## Smoothing alternative signals")
    for sig_name, sig_dict in [("final", md.finals),
                                ("final_raw", md.finals_raw),
                                ("wash", md.washes)]:
        print(f"\n  Signal = {sig_name}")
        for months in [0, 6, 12, 24]:
            if months == 0:
                sf = make_current_fn(sig_dict)
                lbl = "current"
            else:
                sf = make_mean_score_fn(sig_dict, months)
                lbl = f"SMA {months}M"
            r = simulate_custom_score(md, md.stocks, cfg, sf)
            m = compute_metrics(md, r.equity, r.total_invested)
            print(f"    {lbl:25s}  CAGR {m['cagr']*100:+6.2f}%  Sharpe {m['sharpe']:.2f}  "
                  f"Final ${m['final']:11,.0f}")


# Double-smoothing (smooth-of-smooth)
print("\n## Double smoothing (first pass × second pass)")
for m1, m2 in [(6, 6), (6, 12), (12, 12), (12, 24), (24, 12), (24, 24)]:
    sf = make_double_smooth_fn(md.finals, m1, m2)
    r = simulate_custom_score(md, md.stocks, cfg, sf)
    m = compute_metrics(md, r.equity, r.total_invested)
    print(f"  SMA({m1}M) → SMA({m2}M):  CAGR {m['cagr']*100:+6.2f}%  "
          f"Sharpe {m['sharpe']:.2f}  Final ${m['final']:11,.0f}")


# Tight rolling windows: 5Y, 3Y
def win_metric(eq, from_i, to_i):
    if from_i >= to_i or to_i > len(eq): return None
    start = max(eq[from_i], 0.01)
    end = eq[to_i - 1]
    if end <= 0: return None
    yrs = (to_i - from_i) / 252
    cagr = (end / start) ** (1 / yrs) - 1
    return {"cagr": cagr, "yrs": yrs}


print("\n## Rolling 5Y and 3Y windows (SMA 6/12/24M vs incumbent)")
labels_cache = {}
for months in [0, 6, 12, 24]:
    sf = make_current_fn(md.finals) if months == 0 else make_mean_score_fn(md.finals, months)
    r = simulate_custom_score(md, md.stocks, cfg, sf)
    labels_cache["incumbent" if months == 0 else f"SMA{months}M"] = r

for win_yrs in [5, 3]:
    WIN_D = win_yrs * 252
    print(f"\n  Rolling {win_yrs}Y windows (step 1Y):")
    header = f"  {'window':21s}  {'incumbent':>9s}"
    for tag in ["SMA6M", "SMA12M", "SMA24M"]:
        header += f"  {tag:>9s}"
    print(header)
    wins = {"SMA6M": 0, "SMA12M": 0, "SMA24M": 0}
    n = 0
    for from_i in range(0, len(md.all_dates) - WIN_D, 252):
        to_i = from_i + WIN_D
        d_from = md.all_dates[from_i][:7]
        d_to = md.all_dates[to_i - 1][:7]
        m_inc = win_metric(labels_cache["incumbent"].equity, from_i, to_i)
        if m_inc is None: continue
        row = f"  {d_from} → {d_to}  {m_inc['cagr']*100:+8.2f}%"
        for tag in ["SMA6M", "SMA12M", "SMA24M"]:
            m = win_metric(labels_cache[tag].equity, from_i, to_i)
            if m:
                row += f"  {m['cagr']*100:+8.2f}%"
                if m['cagr'] > m_inc['cagr']: wins[tag] += 1
            else:
                row += f"  {'—':>9s}"
        print(row)
        n += 1
    print(f"  Wins vs incumbent: SMA6M {wins['SMA6M']}/{n}  SMA12M {wins['SMA12M']}/{n}  SMA24M {wins['SMA24M']}/{n}")

print("\n## DONE")
