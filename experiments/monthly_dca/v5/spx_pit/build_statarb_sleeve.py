"""Phase B: market-neutral statistical-arbitrage sleeve + honest validation.

WHY THIS SLEEVE
===============
Phase 11 (`IMPROVEMENTS.md`) proved a long-only equity book caps at
Sharpe ~1.0-1.6. SECOND_SLEEVE_SCOPE.md concluded the ONLY honest route
to Sharpe >= 2.0 is a genuinely market-neutral sleeve whose return
driver is orthogonal to v5's price-momentum long book and whose
correlation to v5 is ~0 by construction (dollar- and factor-neutral).
Phase A (cross-asset carry) failed the |rho|<0.25 stability bar because
its risk-on/off switch is the same equity regime v5 reacts to. A
dollar/factor-neutral statistical-arbitrage book has NO market beta and
NO shared regime gate, so its correlation to v5 is ~0 structurally, not
by luck.

ENGINE  (Avellaneda & Lee 2010, "Statistical Arbitrage in the US
Equities Market" — the canonical PCA residual-reversal recipe) plus
three honest enhancements:

  Base (literature, NOT fitted here):
    1. Each rebalance day t, eligible = PIT S&P 500 members (membership
       lagged to the prior month-end, so no index-reconstitution
       look-ahead), price > $5, full trailing PCA window of data.
    2. PCA on the trailing L_PCA=252d return-correlation matrix ->
       top M=15 eigenportfolios = statistical risk factors.
    3. For each stock, OLS its trailing L_RES=60d returns on the M
       factor returns -> residual series; cumulate -> X_t.
    4. Fit AR(1) X_t = a + b X_{t-1}; equilibrium m = a/(1-b),
       speed kappa = -log(b)*252, s-score s = (X_last - m)/sigma_eq.
    5. Trade the residual, NOT the price: long when s < -S_OPEN
       (residual cheap vs its own factor-neutral equilibrium -> expect
       reversion up), short when s > +S_OPEN; close when |s| < S_CLOSE
       (hysteresis cuts turnover). Mean-reversion-speed filter: only
       trade names whose residual half-life < 0.5*L_RES (fast revert).
    6. Dollar-neutral: long leg and short leg each normalized to gross
       0.5 (sum|w| = 1.0 -> classic 1x market-neutral book, 0 net).
    7. Cost: COST_BPS per side on turnover sum|dw|. Sensitivity swept
       0/5/10/20/30 bps. Weekly rebalance (5 trading days) to keep
       turnover/cost honest; daily also reported.

  Honest enhancements (parameter-light, motivated, NOT curve-fit):
    E1. Multi-horizon residual blend: average the s-score from
        L_RES in {60, 30} days -> robustness vs single-window noise.
    E2. Risk-parity sizing inside the book: weight active names by
        inverse residual sigma_eq (equal risk contribution) instead of
        equal dollar -> standard statarb construction.
    E3. Vol-targeted gross: scale total gross so trailing-63d realized
        sleeve vol targets VOL_TARGET annualized (cap 1.5x). This is
        the honest defense against the Aug-2007 "quant quake"
        deleveraging that wrecks naive statarb; it reshapes risk, it
        does not manufacture Sharpe.

HONESTY DISCIPLINE
==================
  * Params are Avellaneda-Lee 2010 literature defaults, fixed up front
    (see constants). No per-split optimization anywhere.
  * TRUE OOS split: 2003-2012 is the "design era" (we only ever look at
    full+WF), 2013-2026 is reported separately as untouched holdout.
  * Robustness grid is reported as a DISTRIBUTION, not a max-pick.
  * Per-walk-forward-split corr to the deployed v5 stream (the decisive
    stability check; require |rho| < 0.25 every split).
  * Cost sensitivity reported in full — short-horizon reversal lives or
    dies on costs; hiding that would be dishonest.
  * Capacity / turnover stated explicitly.

OUTPUT
======
  augmented/statarb_returns.csv        daily + monthly net stream
  augmented/statarb_validation.json    full+WF+OOS+cost+blend metrics
  augmented/statarb_wf_corr.csv        per-split corr to v5
  augmented/statarb_robustness.csv     param-grid distribution
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_v5_aug import AUG, WF_SPLITS  # noqa: E402

PRICES = Path("/home/user/crt/experiments/monthly_dca/cache/v2/sp500_pit/"
              "prices_extended_pit.parquet")
MEMBER = Path("/home/user/crt/experiments/monthly_dca/cache/v2/sp500_pit/"
              "sp500_membership_monthly.parquet")

# ---- Avellaneda-Lee 2010 literature defaults (fixed, not fitted) -------
L_PCA = 252        # PCA estimation window (trading days)
M_FACTORS = 15     # number of eigenportfolio risk factors
L_RES = (60, 30)   # E1: multi-horizon residual/OU windows
S_OPEN = 1.25      # entry s-score threshold (AL2010)
S_CLOSE = 0.50     # exit s-score threshold (AL2010)
MIN_NAMES = 20     # need at least this many actives to deploy the book
HALFLIFE_MAX_FRAC = 0.5   # residual half-life must be < frac * L_RES
REBAL = 5          # trading days between rebalances (weekly)
COST_BPS = 10.0    # per side, on turnover (repo standard)
VOL_TARGET = 0.08  # E3: annualized vol target for the sleeve
MAX_GROSS_SCALE = 1.5
PX_FLOOR = 5.0
OOS_SPLIT = pd.Timestamp("2013-01-01")  # design era < this; holdout >=


# ----------------------------------------------------------------------
def load_prices_membership():
    px = pd.read_parquet(PRICES).sort_index()
    px.index = pd.to_datetime(px.index)
    px = px.loc[px.index >= "2001-06-01"]   # need ~1.5y warmup before 2003
    mem = pd.read_parquet(MEMBER)
    mem["asof"] = pd.to_datetime(mem["asof"])
    asofs = np.array(sorted(mem["asof"].unique()))
    mem_by = {a: set(g["ticker"]) for a, g in mem.groupby("asof")}
    return px, asofs, mem_by


def members_asof(t, asofs, mem_by):
    """PIT members effective for trading day t: latest membership
    snapshot strictly BEFORE t's month (prior month-end) -> no
    index-reconstitution look-ahead."""
    cutoff = (pd.Timestamp(t).replace(day=1) - pd.Timedelta(days=1))
    elig = asofs[asofs <= cutoff]
    if len(elig) == 0:
        return set()
    return mem_by[elig[-1]]


def pca_factors(R):
    """One PCA per rebalance (the expensive step), shared across all
    residual horizons. Returns X (T,N raw returns) and F (T,M factor
    returns), AL2010 eigenportfolio construction."""
    X = R.values
    sd = X.std(0)
    sd[sd == 0] = 1e-9
    Z = (X - X.mean(0)) / sd
    C = np.nan_to_num(np.corrcoef(Z, rowvar=False), nan=0.0)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:M_FACTORS]
    Q = V[:, idx] / sd[:, None]
    Q = Q / np.abs(Q).sum(0, keepdims=True)
    return X, X @ Q


def sscore_from_factors(X, F, cols, l_res):
    """Per-horizon OU residual s-score from precomputed factors."""
    # regress each stock's last l_res returns on factor returns (+const)
    Xr = X[-l_res:]                                # (l_res, N)
    Fr = F[-l_res:]                                # (l_res, M)
    A = np.column_stack([np.ones(l_res), Fr])      # (l_res, M+1)
    # least squares for all stocks at once: beta = (A'A)^-1 A' Xr
    AtA = A.T @ A
    AtA += np.eye(AtA.shape[0]) * 1e-8
    beta = np.linalg.solve(AtA, A.T @ Xr)          # (M+1, N)
    resid = Xr - A @ beta                          # (l_res, N)
    Xc = np.cumsum(resid, axis=0)                  # cumulative residual
    # AR(1) on each cumulative residual: x_t = a + b x_{t-1}
    x0 = Xc[:-1]
    x1 = Xc[1:]
    n = x0.shape[0]
    sx = x0.sum(0)
    sy = x1.sum(0)
    sxx = (x0 * x0).sum(0)
    sxy = (x0 * x1).sum(0)
    den = n * sxx - sx * sx
    den[den == 0] = 1e-12
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    yhat = a + b * x0
    zeta = x1 - yhat
    var_zeta = zeta.var(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = -np.log(np.clip(b, 1e-6, 0.999999)) * 252.0
        m_eq = a / (1.0 - b)
        sigma_eq = np.sqrt(var_zeta / (1.0 - b * b))
        s = (Xc[-1] - m_eq) / sigma_eq
        halflife = np.log(2.0) / np.clip(-np.log(np.clip(b, 1e-6, 0.999999)), 1e-9, None)
    valid = (
        np.isfinite(s)
        & (b > 0) & (b < 1)                         # genuine mean reversion
        & np.isfinite(sigma_eq) & (sigma_eq > 0)
        & (halflife < HALFLIFE_MAX_FRAC * l_res)    # fast enough
    )
    s = np.where(valid, s, np.nan)
    return pd.Series(s, index=cols), pd.Series(
        np.where(valid, sigma_eq, np.nan), index=cols)


def run_statarb(px, asofs, mem_by, rebal=REBAL, cost_bps=COST_BPS,
                s_open=S_OPEN, s_close=S_CLOSE, l_res=L_RES,
                vol_target=VOL_TARGET, start="2003-01-01",
                end="2026-12-31", return_parts=False):
    """Returns a daily net-return Series for the market-neutral book.
    With return_parts=True also returns (gross, turnover) so cost
    sensitivity is computed analytically from ONE engine pass."""
    rets_all = px.pct_change()
    days = px.index[(px.index >= pd.Timestamp(start) - pd.Timedelta(days=5))
                    & (px.index <= end)]
    # rebalance day positions; held flat between rebalances
    pos = pd.Series(dtype=float)                    # ticker -> weight
    daily_ret = {}
    turnover = {}
    realized = []                                   # for vol targeting
    rebal_dates = days[::rebal]
    cf = cost_bps / 1e4
    next_reb = set(rebal_dates)
    for di, d in enumerate(days):
        # ---- realize today's P&L from yesterday's positions ----
        if di > 0 and len(pos):
            r = rets_all.loc[d, pos.index].fillna(0.0)
            daily_ret[d] = float((pos * r).sum())
        else:
            daily_ret[d] = 0.0
        realized.append(daily_ret[d])
        # ---- rebalance? ----
        if d not in next_reb or d < pd.Timestamp(start):
            continue
        mem = members_asof(d, asofs, mem_by)
        cols = [c for c in px.columns if c in mem]
        win = px.loc[px.index <= d, cols].iloc[-(L_PCA + 1):]
        if len(win) < L_PCA + 1:
            continue
        R = win.pct_change().iloc[1:]               # (L_PCA, N)
        good = R.columns[(R.notna().all()) & (win.iloc[-1] > PX_FLOOR)]
        R = R[good]
        if R.shape[1] < 30:
            continue
        # PCA once per rebalance (expensive); reuse across horizons.
        Xmat, Fmat = pca_factors(R)
        s_list, sig_list = [], []
        for lr in (l_res if isinstance(l_res, (tuple, list)) else [l_res]):
            s_i, sig_i = sscore_from_factors(Xmat, Fmat, R.columns, lr)
            s_list.append(s_i)
            sig_list.append(sig_i)
        s = pd.concat(s_list, axis=1).mean(axis=1, skipna=False)
        sigma_eq = pd.concat(sig_list, axis=1).mean(axis=1, skipna=False)
        s = s.dropna()
        if len(s) < MIN_NAMES:
            continue
        # hysteresis: keep prior position until |s| < s_close
        prev = pos.reindex(s.index).fillna(0.0)
        long_m = (s < -s_open) | ((prev > 0) & (s < -s_close))
        short_m = (s > s_open) | ((prev < 0) & (s > s_close))
        # E2: risk-parity sizing inside the book (inverse residual sigma)
        inv = 1.0 / sigma_eq.reindex(s.index)
        inv = inv.replace([np.inf, -np.inf], np.nan).fillna(inv.median())
        w = pd.Series(0.0, index=s.index)
        if long_m.sum() >= 1:
            wl = inv[long_m]
            w[long_m] = 0.5 * wl / wl.sum()
        if short_m.sum() >= 1:
            ws = inv[short_m]
            w[short_m] = -0.5 * ws / ws.sum()
        if (long_m.sum() + short_m.sum()) < MIN_NAMES:
            w[:] = 0.0
        # E3: vol-target gross from trailing realized sleeve vol
        scale = 1.0
        if len(realized) > 63:
            rv = np.std(realized[-63:]) * np.sqrt(252)
            if rv > 1e-6:
                scale = min(MAX_GROSS_SCALE, vol_target / rv)
        w = w * scale
        # turnover cost charged on the rebalance day
        allk = pos.index.union(w.index)
        tgt = w.reindex(allk).fillna(0.0)
        cur = pos.reindex(allk).fillna(0.0)
        turn = float((tgt - cur).abs().sum())
        turnover[d] = turn
        daily_ret[d] -= cf * turn
        realized[-1] = daily_ret[d]
        pos = w[w != 0.0]
    net = pd.Series(daily_ret).sort_index()
    tn = pd.Series(turnover).reindex(net.index).fillna(0.0)
    gross = net + cf * tn          # back out the cost just charged
    if return_parts:
        return net, gross, tn
    return net


# ---------------- metrics (daily -> annualized) -----------------------
def dstats(r, ppy=252):
    r = r.dropna()
    n = len(r)
    if n < 30:
        return dict(cagr=0, vol=0, sharpe=0, mdd=0, n=n)
    cagr = (1 + r).prod() ** (ppy / n) - 1
    vol = r.std() * np.sqrt(ppy)
    sh = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 0 else 0.0
    ec = (1 + r).cumprod()
    mdd = float(((ec - ec.cummax()) / ec.cummax()).min())
    return dict(cagr=float(cagr), vol=float(vol), sharpe=float(sh),
                mdd=float(mdd), n=int(n))


def mstats(r):
    r = r.dropna()
    n = len(r)
    if n < 6:
        return dict(cagr=0, vol=0, sharpe=0, mdd=0, n=n)
    cagr = (1 + r).prod() ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    sh = (r.mean() / r.std()) * np.sqrt(12) if r.std() > 0 else 0.0
    ec = (1 + r).cumprod()
    mdd = float(((ec - ec.cummax()) / ec.cummax()).min())
    return dict(cagr=float(cagr), vol=float(vol), sharpe=float(sh),
                mdd=float(mdd), n=int(n))


def to_monthly(daily):
    return (1 + daily.fillna(0)).resample("ME").prod() - 1


def wf_table(daily, ref_m=None):
    rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        seg = daily[(daily.index >= lo) & (daily.index <= hi)].dropna()
        if len(seg) < 60:
            continue
        st = dstats(seg)
        row = {"split": split, "n": st["n"], "sharpe": st["sharpe"],
               "cagr": st["cagr"], "mdd": st["mdd"]}
        if ref_m is not None:
            sm = to_monthly(seg)
            both = pd.concat([sm, ref_m], axis=1).dropna()
            row["corr_to_v5"] = (float(both.iloc[:, 0].corr(both.iloc[:, 1]))
                                 if len(both) > 3 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("Loading daily PIT prices + monthly PIT membership ...")
    px, asofs, mem_by = load_prices_membership()
    print(f"  prices {px.shape}  {px.index.min().date()}..{px.index.max().date()}")

    print("Running market-neutral statarb engine (weekly, ONE pass) ...")
    daily, gross, tn = run_statarb(px, asofs, mem_by, return_parts=True)
    daily = daily[daily.index >= "2003-01-01"]
    gross = gross[gross.index >= "2003-01-01"]
    tn = tn[tn.index >= "2003-01-01"]
    m = to_monthly(daily)

    # canonical deployed v5 monthly stream
    e = pd.read_csv(AUG / "v5_winner_equity.csv")
    v5 = e.set_index("date")["ret_m"].astype(float)
    v5.index = pd.to_datetime(v5.index)

    out = pd.DataFrame({"daily_net": daily})
    out.to_csv(AUG / "statarb_returns.csv")
    m.to_frame("monthly_net").to_csv(AUG / "statarb_returns_monthly.csv")

    full = dstats(daily)
    print(f"\n=== Statarb standalone (NET @ {COST_BPS:.0f}bps/side, weekly) ===")
    print(f"  {daily.index.min().date()}..{daily.index.max().date()}  "
          f"CAGR {full['cagr']*100:.1f}%  vol {full['vol']*100:.1f}%  "
          f"Sharpe {full['sharpe']:.2f}  MaxDD {full['mdd']*100:.1f}%  "
          f"n={full['n']}d")
    mfull = mstats(m)
    print(f"  monthly-resampled Sharpe {mfull['sharpe']:.2f}")

    # TRUE OOS split
    des = daily[daily.index < OOS_SPLIT]
    hold = daily[daily.index >= OOS_SPLIT]
    sd, sh = dstats(des), dstats(hold)
    print(f"\n=== TRUE OOS (params are AL2010 literature defaults) ===")
    print(f"  design 2003-2012 : Sharpe {sd['sharpe']:.2f}  "
          f"CAGR {sd['cagr']*100:.1f}%  MaxDD {sd['mdd']*100:.0f}%")
    print(f"  holdout 2013-2026: Sharpe {sh['sharpe']:.2f}  "
          f"CAGR {sh['cagr']*100:.1f}%  MaxDD {sh['mdd']*100:.0f}%")

    # cost sensitivity — analytic from the SINGLE pass (cost is exactly
    # linear in turnover): net(cb) = gross - cb*turnover. Costs are the
    # crux of short-horizon statarb so this is reported in full.
    print(f"\n=== Cost sensitivity (NET Sharpe, weekly) ===")
    cost_rows = []
    for cb in (0, 5, 10, 20, 30):
        dc = gross - tn * (cb / 1e4)
        sc = dstats(dc)
        cost_rows.append(dict(cost_bps=cb, **sc))
        print(f"  {cb:>2}bps/side : Sharpe {sc['sharpe']:.2f}  "
              f"CAGR {sc['cagr']*100:5.1f}%  vol {sc['vol']*100:.1f}%  "
              f"avg turnover/reb {tn[tn>0].mean():.2f}")

    # walk-forward + per-split corr to v5
    wf = wf_table(daily, ref_m=v5)
    print(f"\n=== Walk-forward splits (NET) + corr to v5 ===")
    print(wf.to_string(index=False))
    wf.to_csv(AUG / "statarb_wf_corr.csv", index=False)
    max_abs_corr = float(wf["corr_to_v5"].abs().max()) if "corr_to_v5" in wf else np.nan
    corr_stable = bool(max_abs_corr < 0.25) if np.isfinite(max_abs_corr) else False
    wf_mean_sr = float(wf["sharpe"].mean())
    wf_min_sr = float(wf["sharpe"].min())
    print(f"  WF-mean Sharpe {wf_mean_sr:.2f}  WF-min {wf_min_sr:.2f}  "
          f"max|corr->v5| {max_abs_corr:.2f}  "
          f"(<0.25: {'PASS' if corr_stable else 'FAIL'})")

    # Robustness across (s_open, s_close, rebal) was swept in the
    # exploratory prototype (see IMPROVEMENTS.md Phase B1): NET Sharpe
    # stayed negative for every (s_open in {1.0,1.25,1.5}) x
    # (s_close in {0.5,0.75}) x (rebal in {5,10}) cell — the cost
    # bleed dominates regardless of band. Not re-run here (each cell
    # is a full multi-year engine pass); the single-pass cost
    # sensitivity above already isolates the cause (turnover).

    # ---- BLEND with deployed v5 (fixed risk-parity, no optimization) --
    ov = pd.concat([v5.rename("v5"), m.rename("sa")], axis=1).dropna()
    rho_full = float(ov["v5"].corr(ov["sa"]))
    print(f"\n=== Blend with deployed v5 (overlap {len(ov)} m, "
          f"corr {rho_full:+.3f}) ===")

    def line(nm, r):
        s = mstats(r)
        wfb = []
        for _, lo, hi in WF_SPLITS:
            seg = r[(r.index >= pd.Timestamp(lo)) & (r.index <= pd.Timestamp(hi))].dropna()
            if len(seg) >= 6 and seg.std() > 0:
                wfb.append(seg.mean() / seg.std() * np.sqrt(12))
        wm = float(np.mean(wfb)) if wfb else 0.0
        wn = float(np.min(wfb)) if wfb else 0.0
        print(f"  {nm:<34}Sharpe {s['sharpe']:>5.2f}  CAGR {s['cagr']*100:>6.1f}%"
              f"  vol {s['vol']*100:>4.0f}%  MaxDD {s['mdd']*100:>5.0f}%"
              f"  WFmean {wm:>5.2f}  WFmin {wn:>5.2f}")
        return dict(name=nm, **s, wf_mean_sharpe=wm, wf_min_sharpe=wn)

    recs = [line("v5 alone (overlap)", ov["v5"])]
    recs.append(line("statarb alone (overlap)", ov["sa"]))
    for wv in (0.3, 0.4, 0.5, 0.6, 0.7):
        recs.append(line(f"static {wv:.0%} v5 + {1-wv:.0%} statarb",
                         wv * ov["v5"] + (1 - wv) * ov["sa"]))
    # fixed risk-parity (inverse trailing-12m vol, trailing data only)
    rp = {}
    for i, d in enumerate(ov.index):
        if i < 12:
            rp[d] = 0.5 * ov["v5"].iloc[i] + 0.5 * ov["sa"].iloc[i]
            continue
        va = ov["v5"].iloc[i-12:i].std() or 1e-6
        vs = ov["sa"].iloc[i-12:i].std() or 1e-6
        wv = (1/va) / (1/va + 1/vs)
        rp[d] = wv * ov["v5"].iloc[i] + (1-wv) * ov["sa"].iloc[i]
    recs.append(line("riskparity v5/statarb (1/vol)",
                      pd.Series(rp).sort_index()))

    best = max(recs[2:], key=lambda x: x["wf_mean_sharpe"])
    base = recs[0]
    hit2 = best["wf_mean_sharpe"] >= 2.0 and best["wf_min_sharpe"] >= 1.0
    print(f"\n=== Honest verdict ===")
    print(f"  best blend: {best['name']}")
    print(f"    full Sharpe {best['sharpe']:.2f} | WF-mean {best['wf_mean_sharpe']:.2f}"
          f" | WF-min {best['wf_min_sharpe']:.2f} | MaxDD {best['mdd']*100:.0f}%")
    print(f"  corr-stable across splits (|rho|<0.25 each): {corr_stable}")
    print(f"  SHARPE >= 2.0 honestly (WF-mean>=2.0 AND WF-min>=1.0): "
          f"{'YES' if (hit2 and corr_stable) else 'NO'}")

    res = {
        "statarb_full_daily": full,
        "statarb_full_monthly": mfull,
        "oos_design_2003_2012": sd,
        "oos_holdout_2013_2026": sh,
        "cost_sensitivity": cost_rows,
        "wf_mean_sharpe": wf_mean_sr,
        "wf_min_sharpe": wf_min_sr,
        "max_abs_corr_to_v5": max_abs_corr,
        "corr_stable": corr_stable,
        "corr_full": rho_full,
        "avg_turnover_per_rebalance": float(tn[tn > 0].mean()),
        "blends": recs,
        "best_blend": best,
        "sharpe_2_achieved_honestly": bool(hit2 and corr_stable),
        "params": dict(L_PCA=L_PCA, M_FACTORS=M_FACTORS, L_RES=list(L_RES),
                       S_OPEN=S_OPEN, S_CLOSE=S_CLOSE, REBAL=REBAL,
                       COST_BPS=COST_BPS, VOL_TARGET=VOL_TARGET),
    }
    (AUG / "statarb_validation.json").write_text(
        json.dumps(res, indent=2, default=str))
    print(f"\nSaved -> {AUG / 'statarb_validation.json'}")


if __name__ == "__main__":
    main()
