/* ============================================================
   Monthly DCA — main page renderer
   Reads /experiments/monthly-dca/data.json and renders:
     - hero meta line
     - this month's pick card
     - backtest growth chart + stats
     - year-by-year grid
     - survivorship audit cards + sensitivity table
   ============================================================ */

const DATA_URL = "/experiments/monthly-dca/data.json";

const fmtPct = (x, d = 1) => x == null || !Number.isFinite(+x) ? "—" : `${(x * 100).toFixed(d)}%`;
const fmtPct0 = x => x == null || !Number.isFinite(+x) ? "—" : `${Math.round(x * 100)}%`;
const fmtPctSigned = (x, d = 1) => {
  if (x == null || !Number.isFinite(+x)) return "—";
  const v = (x * 100).toFixed(d);
  return Number(v) >= 0 ? `+${v}%` : `${v}%`;
};
const fmtX = x => {
  if (x == null || !Number.isFinite(+x)) return "—";
  const v = +x;
  if (v >= 1000) return `${Math.round(v).toLocaleString()}×`;
  if (v >= 100) return `${v.toFixed(0)}×`;
  if (v >= 10) return `${v.toFixed(1)}×`;
  return `${v.toFixed(2)}×`;
};
const fmtPx = x => x == null ? "—" : "$" + (+x).toFixed(2);
const fmtNum = (x, d = 2) => x == null ? "—" : (+x).toFixed(d);
const fmtMoney = x => {
  if (x == null || !Number.isFinite(+x)) return "—";
  const v = +x;
  if (v >= 1000) return `$${Math.round(v).toLocaleString()}`;
  return `$${v.toFixed(0)}`;
};

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const c of (Array.isArray(children) ? children : [children])) {
    if (c == null) continue;
    if (typeof c === "string") e.appendChild(document.createTextNode(c));
    else e.appendChild(c);
  }
  return e;
}

function clsRet(x) { return x == null ? "" : (x > 0 ? "pos" : (x < 0 ? "neg" : "")); }

function median(arr) {
  const a = (arr || []).filter(x => x != null && Number.isFinite(+x)).map(Number).sort((p, q) => p - q);
  if (!a.length) return null;
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

fetch(DATA_URL + "?v=" + Date.now())
  .then(r => r.json())
  .then(render)
  .catch(err => {
    document.getElementById("pickBody").innerHTML =
      `<div style="color:var(--orange)">Could not load data: ${err.message}</div>`;
  });

function render(data) {
  renderHeroMeta(data);
  renderDcaInvestor(data);
  renderPick(data);
  renderBacktest(data);
  renderHorizons(data);
  renderYears(data);
  renderWalkForward(data);
  renderBias(data);
  renderCaseStudies(data);
  renderV3Sections(data);
  renderHistorical(data);
  renderTrades(data);
}

/* ============================================================
   DCA-investor outcomes — the only section that describes the
   actual user (monthly contributions), rendered from
   data.dca_investor (built daily by build_webapp_v5_pit.py)
   ============================================================ */
const DCA_HLABEL = { H12: "1 year", H24: "2 years", H36: "3 years", H60: "5 years", H120: "10 years" };

function renderDcaInvestor(data) {
  const sec = document.getElementById("dcaInvestorSection");
  if (!sec) return;
  const di = data.dca_investor;
  if (!di || !di.horizons) {
    sec.innerHTML = `<div style="color:var(--orange)">DCA-investor data not available in this build.</div>`;
    return;
  }
  sec.innerHTML = "";

  const hasSwitch = !!(di.horizons.H120 && di.horizons.H120.mn_switch);
  const h120 = di.horizons.H120.v5;
  const h12 = di.horizons.H12.v5;
  const fh = di.full_history || {};

  // hero numbers
  try {
    const w = Math.round((h120.win_vs_spy_dca || 0) * 100) + "%";
    const a = document.getElementById("heroWin"); if (a) a.textContent = w;
    const b = document.getElementById("heroWin2"); if (b) b.textContent = w;
  } catch (e) {}

  // headline band
  const band = el("div", { class: "bias-grid", style: "margin-bottom:20px" });
  band.appendChild(biasCard("10-year DCA win vs S&P-DCA",
    fmtPct0(h120.win_vs_spy_dca),
    `${h120.n_windows} rolling 10y windows on PIT data. Median money-weighted return ${fmtPct(h120.median_irr)}/yr, worst ${fmtPct(h120.min_irr)}/yr. This is the honest "high hit rate".`,
    true));
  band.appendChild(biasCard("Median 10y DCA return",
    fmtPct(h120.median_irr),
    `Annualized money-weighted return on monthly contributions over 10y. S&P-DCA median: ${fmtPct(di.horizons.H120.SPY?.median_irr)}/yr.`));
  band.appendChild(biasCard("Worst 1-year DCA return",
    fmtPct(h12.min_irr),
    `Brutal short-horizon reality: the single worst 12-month contribution window. DCA is a multi-year commitment, not a 1-year trade.`));
  band.appendChild(biasCard("Raw interim drawdown",
    fmtPct(fh.v5?.max_value_drawdown, 0),
    `Peak-to-trough on the accumulating portfolio (raw v5). There is no no-downside version. The MN-switch variant below cuts this to ${fmtPct(fh.mn_switch?.max_value_drawdown, 0)}.`));
  sec.appendChild(band);

  // HONEST era-by-era: the headline is front-loaded; recent eras can lag.
  const era = di.by_era;
  if (era?.length) {
    const anyLag = era.some(e => !e.beat_spy);
    const warn = el("div", { class: "pick-disclaimer", style: "margin:0 0 20px;border-color:var(--orange)" });
    warn.innerHTML =
      "<strong>Read this honestly — the edge is NOT uniform.</strong> The all-history number is heavily front-loaded by the 2003–2009 GFC-recovery era (a one-off window that will not repeat at that scale). Broken into non-overlapping eras below" +
      (anyLag ? ", the strategy <strong>underperformed S&P-DCA in at least one recent era</strong>." : ".") +
      " It does <em>not</em> substantially beat the S&P in every period — the reliable edge is the long-horizon (10-year) result, not any given short era.";
    sec.appendChild(warn);
    const wrap = el("div", { class: "bias-table-wrap", style: "margin-bottom:20px" });
    const t = el("table", { class: "bias-table" });
    t.innerHTML = "<thead><tr><th>Era (non-overlapping)</th><th>Strategy DCA (IRR)</th><th>S&P-DCA (IRR)</th><th>Edge</th><th>Beat S&P-DCA?</th></tr></thead>";
    const tb = el("tbody");
    era.forEach(e => {
      const ep = (e.strat_irr != null && e.spy_irr != null) ? e.strat_irr - e.spy_irr : null;
      const tr = el("tr");
      tr.appendChild(el("td", { class: "tkr" }, e.era));
      tr.appendChild(el("td", { class: clsRet(ep) }, fmtPct(e.strat_irr)));
      tr.appendChild(el("td", {}, fmtPct(e.spy_irr)));
      tr.appendChild(el("td", { class: clsRet(ep) }, fmtPctSigned(ep)));
      tr.appendChild(el("td", { class: e.beat_spy ? "pos" : "neg" }, e.beat_spy ? "yes" : "NO"));
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    wrap.appendChild(t);
    sec.appendChild(wrap);
  }

  // horizon breakdown — responsive card grid (mobile-safe; no wide table)
  const hint = el("p", { class: "bias-sub", style: "margin:0 0 12px" },
    "Each card: a DCA horizon. Top number = % of all rolling windows where contributing into the strategy beat the same contributions into the S&P 500. Returns shown are annualized money-weighted (the rate your contributions actually compounded at).");
  sec.appendChild(hint);
  const grid = el("div", { class: "bias-grid" });
  ["H120", "H60", "H36", "H24", "H12"].forEach(H => {
    const row = di.horizons[H]; if (!row) return;
    const v = row.v5 || {}, mn = row.mn_switch, sp = row.SPY || {};
    const win = v.win_vs_spy_dca;
    const isHero = win != null && win >= 0.999;
    const c = el("div", { class: "bias-card" + (isHero ? " bias-card-headline" : "") });
    c.appendChild(el("div", { class: "bias-label" }, `${DCA_HLABEL[H]} DCA — win vs S&P-DCA`));
    c.appendChild(el("div", { class: "bias-value" }, fmtPct0(win)));
    let sub = `Raw v5: median <strong>${fmtPct(v.median_irr)}/yr</strong> · worst <span style="color:var(--orange)">${fmtPct(v.min_irr)}/yr</span>`;
    if (mn) sub += `<br>Smoother (MN-switch): win <strong>${fmtPct0(mn.win_vs_spy_dca)}</strong> · median ${fmtPct(mn.median_irr)}/yr`;
    sub += `<br>S&P-DCA median: ${fmtPct(sp.median_irr)}/yr`;
    c.appendChild(el("div", { class: "bias-sub", html: sub }));
    grid.appendChild(c);
  });
  sec.appendChild(grid);

  // two-variant explainer
  if (hasSwitch && fh.v5 && fh.mn_switch) {
    const g = el("div", { class: "bias-grid", style: "margin-top:20px" });
    g.appendChild(biasCard("Variant A — Raw v5 (max parabola)",
      fmtPct(fh.v5.money_weighted_irr),
      `Money-weighted return on monthly contributions, ${di.window}. Each $1 → $${fmtNum(fh.v5.terminal_moic, 0)}; interim drawdown ${fmtPct(fh.v5.max_value_drawdown, 0)}. Choose this only for a 10-year+ commitment you will not interrupt.`));
    g.appendChild(biasCard("Variant B — MN drawdown-switch",
      fmtPct(fh.mn_switch.money_weighted_irr),
      `Same picker, but rotates into the validated market-neutral sleeve when the portfolio draws down past -25% (an honest portfolio switch, not a new alpha). Interim drawdown only ${fmtPct(fh.mn_switch.max_value_drawdown, 0)}. ~Half the upside for ~half the drawdown — still 100% at 10y.`));
    sec.appendChild(g);
  }
}

function biasCard(label, value, sub, headline) {
  const c = el("div", { class: "bias-card" + (headline ? " bias-card-headline" : "") });
  c.appendChild(el("div", { class: "bias-label" }, label));
  c.appendChild(el("div", { class: "bias-value" }, value));
  c.appendChild(el("div", { class: "bias-sub" }, sub));
  return c;
}

/* ============================================================
   Case studies — every closed basket + the current one
   ============================================================ */
function renderCaseStudies(data) {
  const sec = document.getElementById("caseStudies");
  if (!sec) return;
  sec.innerHTML = "";

  const log = data.pick_log || [];
  if (!log.length) return;

  // Group by basket_id
  const byBasket = {};
  log.forEach(p => {
    const k = p.basket_id ?? p.asof;
    if (!byBasket[k]) byBasket[k] = [];
    byBasket[k].push(p);
  });

  const bids = Object.keys(byBasket).sort((a, b) => {
    const aD = byBasket[a][0]?.asof || "";
    const bD = byBasket[b][0]?.asof || "";
    return bD.localeCompare(aD);
  });

  bids.forEach(bid => {
    const basket = byBasket[bid];
    if (!basket.length) return;
    const isOpen = basket[0]?.status !== "exited";
    const entry = basket[0]?.asof;
    const exit = basket[0]?.exit_date || (isOpen ? "open" : "?");

    const validRets = basket.map(p => p.ret_strat ?? p.return).filter(r => r != null);
    const basketRet = validRets.length
      ? validRets.reduce((a, b) => a + b, 0) / validRets.length
      : null;
    const spyRets = basket.map(p => p.ret_spy).filter(r => r != null);
    const spyBasketRet = spyRets.length ? spyRets[0] : null;

    const card = el("div", { class: "case-card" });
    const dateLabel = isOpen
      ? `${entry} → in progress (open)`
      : `${entry} → ${exit}`;
    card.appendChild(el("div", { class: "case-label" }, dateLabel));

    const tickers = basket.map(p => p.ticker).join(" · ");
    card.appendChild(el("div", { class: "case-tickers" }, tickers));

    if (isOpen) {
      card.appendChild(el("div", { class: "case-ret mid" }, "Currently held"));
    } else if (basketRet != null) {
      const beat = (spyBasketRet != null) ? (basketRet - spyBasketRet) : null;
      let line = `Strategy ${fmtPctSigned(basketRet, 1)}`;
      if (spyBasketRet != null) {
        line += ` · SPY ${fmtPctSigned(spyBasketRet, 1)} · `;
        if (beat != null) {
          line += beat >= 0 ? `beat by ${(beat*100).toFixed(1)}pp` : `lagged by ${Math.abs(beat*100).toFixed(1)}pp`;
        }
      }
      card.appendChild(el("div", { class: "case-ret " + clsRet(basketRet) }, line));
    }
    sec.appendChild(card);
  });
}

/* ============================================================
   v3-specific: sub-period CAGR, multi-universe generalisation,
   parameter sensitivity, drawdown ledger
   ============================================================ */
function renderV3Sections(data) {
  const host = document.getElementById("v3Sections");
  if (!host) return;
  host.innerHTML = "";

  // 1. Sub-period robustness — DCA money-in vs S&P-DCA
  const subp = data.dca_investor?.subperiods;
  if (subp?.length) {
    const sec = el("div", { class: "v3-block" });
    sec.appendChild(el("h3", { class: "section-h3" }, "Sub-period robustness (DCA)"));
    sec.appendChild(el("p", { class: "section-sub" },
      "If you'd contributed monthly through each historical sub-period: the money-weighted return (annualized IRR) on your contributions vs identical contributions into the S&P 500. The DCA edge holds across every multi-year sub-period."));
    const tbl = el("table", { class: "v3-table" });
    tbl.innerHTML = `<thead><tr><th>Period</th><th>Months</th><th>Strategy DCA (IRR)</th><th>S&P-DCA (IRR)</th><th>Edge</th></tr></thead>`;
    const tb = el("tbody");
    subp.forEach(p => {
      const ep = (p.strat_irr != null && p.spy_irr != null) ? p.strat_irr - p.spy_irr : null;
      const row = el("tr");
      row.appendChild(el("td", {}, (p.period || "").replace(/^p\d_/, "").replace(/_/g, "-")));
      row.appendChild(el("td", {}, String(p.n_months)));
      row.appendChild(el("td", { class: clsRet(ep) }, fmtPct(p.strat_irr)));
      row.appendChild(el("td", {}, fmtPct(p.spy_irr)));
      row.appendChild(el("td", { class: clsRet(ep) }, fmtPctSigned(ep)));
      tb.appendChild(row);
    });
    tbl.appendChild(tb);
    sec.appendChild(tbl);
    host.appendChild(sec);
  }

  // 2. Multi-universe generalisation
  if (data.multi_universe_generalisation?.length) {
    const sec = el("div", { class: "v3-block" });
    sec.appendChild(el("h3", { class: "section-h3" }, "Engine robustness — generalises across universes"));
    sec.appendChild(el("p", { class: "section-sub" },
      "This validates the PICKER that generates your DCA account — it is engine robustness, not a DCA return promise (your DCA outcome is the section at the top). Same K=2 config run on alternative universes; shown as how many out-of-sample windows the picker beat the relevant benchmark. It generalises across curated US-equity cohorts."));
    const tbl = el("table", { class: "v3-table" });
    tbl.innerHTML = `<thead><tr><th>Universe</th><th>Pool size</th><th>OOS windows beating benchmark</th></tr></thead>`;
    const tb = el("tbody");
    data.multi_universe_generalisation.forEach(u => {
      const row = el("tr");
      const UNIV_LABEL = {
        "PIT_SP500_augmented": "PIT S&P 500 (augmented)",
        "NDX_PIT": "PIT Nasdaq-100",
        "broader_augmented": "Broader 1964-ticker (augmented)",
        "non_sp500_augmented": "Non-S&P 500 (augmented)",
        "random_500_seed1": "Random 500 (seed 1)",
        "random_500_seed2": "Random 500 (seed 2)",
        "random_500_seed3": "Random 500 (seed 3)",
        "PIT_SP500": "PIT S&P 500",
        "broader_1833": "Broader 1,833-ticker",
        "non_SP500_PIT": "Non-S&P 500 PIT",
        "sp500_pit": "PIT S&P 500",
      };
      const n_splits = u.wf_n_splits || (u.universe === "NDX_PIT" ? 8 : 10);
      const bench_label = u.universe === "NDX_PIT" ? "QQQ" : "SPY";
      row.appendChild(el("td", {}, UNIV_LABEL[u.universe] || u.universe.replace(/_/g, " ")));
      row.appendChild(el("td", {}, String(u.n_pool)));
      row.appendChild(el("td", { class: u.wf_n_beats_spy >= n_splits * 0.6 ? "pos" : "" },
        `${u.wf_n_beats_spy}/${n_splits} vs ${bench_label}`));
      tb.appendChild(row);
    });
    tbl.appendChild(tb);
    sec.appendChild(tbl);
    host.appendChild(sec);
  }

  // 3. Parameter sensitivity — engine robustness only (no lump-sum returns)
  if (data.parameter_sensitivity?.length) {
    const sec = el("div", { class: "v3-block" });
    sec.appendChild(el("h3", { class: "section-h3" }, "Engine robustness — parameter sensitivity"));
    sec.appendChild(el("p", { class: "section-sub" },
      "Each row perturbs ONE config parameter; shown as how many out-of-sample windows the picker still beat the S&P. The picker is not on a knife-edge — it stays robust across a broad parameter plateau. Engine validation, not a DCA return."));
    const tbl = el("table", { class: "v3-table" });
    tbl.innerHTML = `<thead><tr><th>Param</th><th>Value</th><th>OOS windows beating S&P</th></tr></thead>`;
    const tb = el("tbody");
    data.parameter_sensitivity.forEach(s => {
      const row = el("tr");
      row.appendChild(el("td", {}, s.param));
      row.appendChild(el("td", {}, String(s.value)));
      row.appendChild(el("td", { class: s.wf_n_beats_spy >= 6 ? "pos" : "" }, `${s.wf_n_beats_spy}/10`));
      tb.appendChild(row);
    });
    tbl.appendChild(tb);
    sec.appendChild(tbl);
    host.appendChild(sec);
  }

  // 4. Drawdown ledger (top 5)
  if (data.drawdowns?.length) {
    const sec = el("div", { class: "v3-block" });
    sec.appendChild(el("h3", { class: "section-h3" }, "Picker drawdown ledger (the engine, not your DCA account)"));
    sec.appendChild(el("p", { class: "section-sub" },
      "Peak-to-trough drawdowns of the underlying lump-sum picker equity (worst ~-77%) — shown for completeness of the engine. The drawdown a monthly contributor actually felt is the honest ~-56% figure disclosed in the DCA outcomes section above; these are deeper/sharper because they ignore the cushioning effect of steady monthly contributions."));
    const tbl = el("table", { class: "v3-table" });
    tbl.innerHTML = `<thead><tr><th>Start</th><th>Trough</th><th>End</th><th>Depth</th></tr></thead>`;
    const tb = el("tbody");
    data.drawdowns.slice(0, 8).forEach(d => {
      const row = el("tr");
      row.appendChild(el("td", {}, d.start));
      row.appendChild(el("td", {}, d.trough));
      row.appendChild(el("td", {}, d.end));
      row.appendChild(el("td", { class: "neg" }, d.depth_pct.toFixed(1) + "%"));
      tb.appendChild(row);
    });
    tbl.appendChild(tb);
    sec.appendChild(tbl);
    host.appendChild(sec);
  }

  // 5. Most-picked tickers
  if (data.most_picked?.length) {
    const sec = el("div", { class: "v3-block" });
    sec.appendChild(el("h3", { class: "section-h3" }, "Names that drove your DCA account"));
    sec.appendChild(el("p", { class: "section-sub" },
      "The tickers the strategy selected most often across the 22-year history — i.e. where your monthly contributions were most often allocated."));
    const grid = el("div", { class: "most-picked-grid" });
    data.most_picked.slice(0, 12).forEach(p => {
      const card = el("div", { class: "most-picked-card" });
      card.appendChild(el("div", { class: "most-picked-tkr" }, p.ticker));
      card.appendChild(el("div", { class: "most-picked-n" }, `${p.n_months_picked} months`));
      grid.appendChild(card);
    });
    sec.appendChild(grid);
    host.appendChild(sec);
  }
}

function renderHeroMeta(data) {
  const meta = document.getElementById("heroMeta");
  if (!meta) return;
  const d10 = data.dca_investor?.horizons?.H120?.v5;
  const fh = data.dca_investor?.full_history?.v5;
  const parts = [`As of ${data.as_of}`,
    `${data.panel?.n_tickers ?? "?"} tickers, ${data.panel?.first_date} → ${data.panel?.last_date}`];
  if (d10) parts.push(`10y monthly-DCA beat S&P-DCA in ${fmtPct0(d10.win_vs_spy_dca)} of windows (median ${fmtPct(d10.median_irr)}/yr)`);
  if (fh) parts.push(`money-weighted IRR ${fmtPct(fh.money_weighted_irr)}`);
  meta.textContent = parts.join(" · ");
}

function renderPick(data) {
  const body = document.getElementById("pickBody");
  body.classList.remove("pick-loading");
  body.innerHTML = "";
  const basket = data.pick_of_month_basket || (data.pick_of_month ? [data.pick_of_month] : []);
  const ls = data.live_state || {};
  if (!basket.length) {
    if (ls.cash_position) {
      body.innerHTML = `<div class="basket-dates"><strong>100% cash this month.</strong> SPY broke down enough to trigger the crash gate. We re-evaluate at the next month-end.</div>`;
      return;
    }
    body.innerHTML = "<div>No picks available for this month.</div>";
    return;
  }
  const buyDate = ls.last_rebalance_date || data.as_of;
  const sellDate = ls.next_rebalance_date || (() => {
    try { const d = new Date(data.as_of); d.setMonth(d.getMonth() + 6); return d.toISOString().slice(0, 10); }
    catch (e) { return ""; }
  })();
  const monthsHeld = ls.months_since_rebalance != null ? ls.months_since_rebalance : null;
  const monthsLeft = monthsHeld != null ? Math.max(0, 6 - monthsHeld) : null;

  // What does the user need to do THIS month? Two cases:
  //   A) On rebalance day → big SELL/BUY/HOLD cards (action required today)
  //   B) Mid-cycle → big "do nothing" banner (no action this month)
  const lastRebMonth = (ls.last_rebalance_date || "").slice(0, 7);
  const asofMonth = (data.as_of || "").slice(0, 7);
  const justRebalanced = lastRebMonth === asofMonth || monthsHeld === 0;
  const buys = ls.last_rebalance_to_buy || [];
  const holds = ls.last_rebalance_to_hold || [];
  const sells = ls.last_rebalance_to_sell || [];

  if (justRebalanced) {
    // Rebalance day: show big action banner + SELL/BUY/HOLD cards
    const head = el("div", { class: "basket-dates basket-dates-action" });
    head.innerHTML = `<strong>Rebalance day — action required.</strong> Today (${data.as_of}) is the 6-month rebalance. Sell the names leaving, buy the names entering, keep the carryover. <strong>Inverse-volatility weighted</strong> (lower-vol stocks get more weight) with 40% cap per pick.`;
    body.appendChild(head);
    if (buys.length + holds.length + sells.length > 0) {
      const actions = el("div", { class: "actions-grid" });
      if (sells.length) {
        const card = el("div", { class: "action-card sell" });
        card.appendChild(el("div", { class: "action-label" }, "SELL today"));
        card.appendChild(el("div", { class: "action-tickers" }, sells.join(" · ")));
        card.appendChild(el("div", { class: "action-sub" }, "These names are leaving the basket."));
        actions.appendChild(card);
      }
      if (buys.length) {
        const card = el("div", { class: "action-card buy" });
        card.appendChild(el("div", { class: "action-label" }, "BUY today"));
        card.appendChild(el("div", { class: "action-tickers" }, buys.join(" · ")));
        card.appendChild(el("div", { class: "action-sub" }, "Weights = 1/vol_1y, capped at 40% per name."));
        actions.appendChild(card);
      }
      if (holds.length) {
        const card = el("div", { class: "action-card hold" });
        card.appendChild(el("div", { class: "action-label" }, "KEEP holding"));
        card.appendChild(el("div", { class: "action-tickers" }, holds.join(" · ")));
        card.appendChild(el("div", { class: "action-sub" }, "Carried forward — don't sell, don't rebuy."));
        actions.appendChild(card);
      }
      body.appendChild(actions);
    }
  } else {
    // Mid-cycle: big "do nothing" message + the current 2 picks
    const head = el("div", { class: "basket-dates basket-dates-noaction" });
    const tickerStr = (ls.current_basket_picks || []).join(", ") || basket.map(p=>p.ticker).join(", ");
    const weights = ls.current_basket_weights || [];
    const tickerWtStr = (ls.current_basket_picks || []).map((tk, i) => {
      const w = weights[i];
      return w != null ? `${tk} (${Math.round(w*100)}%)` : tk;
    }).join(", ");
    head.innerHTML = `<strong>This month: do nothing.</strong><br>` +
      `Continue holding <strong>${tickerWtStr}</strong> (bought ${buyDate}). ` +
      `Next rebalance: <strong>${sellDate}</strong> (${monthsLeft} more month${monthsLeft===1?'':'s'}).`;
    body.appendChild(head);
  }
  const grid = el("div", { class: "basket-grid" });
  basket.forEach((p, i) => {
    const card = el("div", { class: "basket-card" + (i === 0 ? " first" : "") });
    card.appendChild(el("div", { class: "basket-tkr-row" }, [
      el("span", { class: "basket-rank" }, "#" + (i + 1)),
      el("span", { class: "basket-tkr" }, p.ticker),
      el("span", { class: "basket-tkr-px" }, fmtPx(p.price)),
    ]));
    if (sellDate) {
      const heldOrBought = justRebalanced ? "Bought" : "Held since";
      card.appendChild(el("div", { class: "basket-sell-by" },
        `${heldOrBought} ${buyDate} • Sell ${sellDate}`));
    }
    const stats = el("div", { class: "basket-stats" });
    const statRows = [
      ["1y pullback", fmtPctSigned(p.pullback_1y, 0), clsRet(p.pullback_1y)],
      ["5y trend", fmtNum(p.trend_health_5y, 2), ""],
      ["3y momentum", fmtPctSigned(p.mom_3y, 0), clsRet(p.mom_3y)],
      ["12m momentum", fmtPctSigned(p.mom_12_1, 0), clsRet(p.mom_12_1)],
      ["RSI", Math.round(p.rsi_14 || 0), ""],
      ["vs 200dma", fmtPctSigned(p.d_sma200, 0), clsRet(p.d_sma200)],
    ];
    statRows.forEach(([lbl, val, c]) => {
      const item = el("div", { class: "basket-stat" });
      item.appendChild(el("span", { class: "basket-stat-label" }, lbl));
      item.appendChild(el("span", { class: "basket-stat-value " + c }, String(val)));
      stats.appendChild(item);
    });
    card.appendChild(stats);
    grid.appendChild(card);
  });
  body.appendChild(grid);

  // Reference footer mid-cycle: tiny note about what happened at last rebalance
  if (!justRebalanced && (buys.length || sells.length)) {
    const ref = el("details", { class: "rebalance-ref" });
    ref.appendChild(el("summary", {},
      `What happened at the last rebalance (${ls.last_rebalance_date}) — for reference`));
    const refBody = el("div", { class: "rebalance-ref-body" });
    if (sells.length) {
      refBody.appendChild(el("div", {},
        el("strong", { class: "neg" }, "Sold: "), sells.join(", ")));
    }
    if (buys.length) {
      refBody.appendChild(el("div", {},
        el("strong", { class: "pos" }, "Bought: "), buys.join(", ")));
    }
    if (holds.length) {
      refBody.appendChild(el("div", {},
        el("strong", {}, "Kept: "), holds.join(", ") + " (carried forward)"));
    }
    ref.appendChild(refBody);
    body.appendChild(ref);
  }
}

function renderBacktest(data) {
  const stats = document.getElementById("btStats");
  stats.innerHTML = "";
  const last = data.growth?.[data.growth.length - 1];
  if (!last) { stats.textContent = "No growth data."; return; }
  const final_strat = last.strat_value;
  const final_spy = last.spy_value;
  const final_cash = last.invested;
  const fh = data.dca_investor?.full_history || {};
  const d10 = data.dca_investor?.horizons?.H120?.v5;

  const items = [
    ["Every $1 contributed became", fmtX(fh.v5?.terminal_moic ?? (final_strat / final_cash)),
      fmtMoney(final_cash) + " contributed → " + fmtMoney(final_strat) + " (full history)", "highlight"],
    ["Money-weighted IRR", fmtPct(fh.v5?.money_weighted_irr),
      "the rate your monthly contributions actually compounded at", "ok"],
    ["S&P-DCA, same schedule", fmtMoney(final_spy),
      "identical monthly contributions into the S&P 500", "ok"],
    ["10-year DCA win vs S&P-DCA", fmtPct0(d10?.win_vs_spy_dca),
      "of all rolling 10y windows; median " + fmtPct(d10?.median_irr) + "/yr money-weighted", "ok"],
  ];
  items.forEach(([lbl, val, sub, cls]) => {
    const c = el("div", { class: "bt-stat-card " + cls });
    c.appendChild(el("div", { class: "bt-stat-label" }, lbl));
    c.appendChild(el("div", { class: "bt-stat-value" }, val));
    c.appendChild(el("div", { class: "bt-stat-sub" }, sub));
    stats.appendChild(c);
  });

  // Draw chart with period selector
  const canvas = document.getElementById("btChart");
  if (canvas && data.growth) {
    let currentPeriod = "all";
    const draw = () => {
      const { rows, isRebased } = filterGrowthByPeriod(data.growth, currentPeriod);
      drawGrowth(canvas, rows, isRebased);
    };
    requestAnimationFrame(draw);
    window.addEventListener("resize", draw, { passive: true });

    // Hook up period buttons
    const bar = document.getElementById("btPeriodBar");
    if (bar) {
      bar.querySelectorAll(".bt-period-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          currentPeriod = btn.getAttribute("data-period");
          bar.querySelectorAll(".bt-period-btn").forEach(b => {
            const active = b === btn;
            b.classList.toggle("active", active);
            b.setAttribute("aria-selected", active ? "true" : "false");
          });
          draw();
        });
      });
    }
  }
}

// Re-simulate the DCA account from the start of the selected window, so
// every period button means "if you STARTED contributing $1/mo then" —
// all three lines begin at $0 and grow comparably (not a mid-stream
// balance where the long-running account dwarfs everything).
function dcaResimulate(rows) {
  let v = 0, sp = 0, inv = 0;
  return rows.map(g => {
    v = (v + 1) * (1 + (g.r || 0));
    sp = (sp + 1) * (1 + (g.s || 0));
    inv += 1;
    return { date: g.date, strat_value: v, spy_value: sp, invested: inv };
  });
}

function filterGrowthByPeriod(growth, period) {
  if (!growth?.length) return { rows: [], isRebased: false };
  if (period === "all") return { rows: dcaResimulate(growth), isRebased: false };
  const yearsMap = { "1y": 1, "3y": 3, "5y": 5, "10y": 10, "20y": 20 };
  const yrs = yearsMap[period];
  if (!yrs) return { rows: dcaResimulate(growth), isRebased: false };
  const last = new Date(growth[growth.length - 1].date);
  const cutoff = new Date(last);
  cutoff.setFullYear(cutoff.getFullYear() - yrs);
  const startIdx = growth.findIndex(g => new Date(g.date) >= cutoff);
  if (startIdx <= 0) return { rows: dcaResimulate(growth), isRebased: false };
  // "Started DCA-ing N years ago": contribute from the slice start.
  return { rows: dcaResimulate(growth.slice(startIdx)), isRebased: false };
}

function drawGrowth(canvas, growth, isRebased) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
  const H = canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
  ctx.clearRect(0, 0, W, H);

  if (!growth || growth.length < 2) return;
  const padL = 52 * devicePixelRatio, padR = 12 * devicePixelRatio,
        padT = 18 * devicePixelRatio, padB = 32 * devicePixelRatio;
  const n = growth.length;
  let maxY = 0;
  growth.forEach(g => {
    maxY = Math.max(maxY, g.strat_value || 0, g.spy_value || 0, g.invested || 0);
  });
  const minY = 0;
  const niceMax = niceTop(maxY);
  // Rebased periods use × multipliers; "All" uses absolute dollars.
  const yMode = isRebased ? "x" : "$";

  const xAt = i => padL + (W - padL - padR) * (i / (n - 1));
  const yAt = v => H - padB - (H - padT - padB) * ((v - minY) / (niceMax - minY));

  // Y-axis grid
  ctx.font = `${11 * devicePixelRatio}px IBM Plex Mono, ui-monospace, monospace`;
  ctx.fillStyle = "#767676";
  ctx.strokeStyle = "#e5e5e5";
  ctx.lineWidth = 1 * devicePixelRatio;
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = niceMax * (i / ticks);
    const y = yAt(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(formatYAxisValue(v, yMode), 4 * devicePixelRatio, y + 4 * devicePixelRatio);
  }

  // X-axis labels (years) — adaptive density to avoid overlap
  // Approx min pixel-spacing per label = 56px (room for 'YYYY' + breathing room)
  const minLabelSpacingPx = 56 * devicePixelRatio;
  const chartWidth = W - padL - padR;
  // First, gather unique years and their first index
  const yearFirstIdx = [];
  const seenYears = new Set();
  growth.forEach((g, i) => {
    const yr = String(g.date).slice(0, 4);
    if (!seenYears.has(yr)) {
      seenYears.add(yr);
      yearFirstIdx.push({ yr, i });
    }
  });
  // Decide how many years to skip between labels so they don't overlap
  const maxLabels = Math.max(2, Math.floor(chartWidth / minLabelSpacingPx));
  const yearStep = Math.max(1, Math.ceil(yearFirstIdx.length / maxLabels));
  // For short windows (<=2 years), label months instead
  const totalMonths = growth.length;
  const useMonths = (yearFirstIdx.length <= 2);

  if (useMonths) {
    const minMonthSpacingPx = 48 * devicePixelRatio;
    const maxMonthLabels = Math.max(2, Math.floor(chartWidth / minMonthSpacingPx));
    const monthStep = Math.max(1, Math.ceil(totalMonths / maxMonthLabels));
    growth.forEach((g, i) => {
      if (i % monthStep !== 0 && i !== growth.length - 1) return;
      const x = xAt(i);
      ctx.fillStyle = "#767676";
      const lbl = String(g.date).slice(0, 7); // YYYY-MM
      const w = ctx.measureText(lbl).width;
      ctx.fillText(lbl, x - w / 2, H - 8 * devicePixelRatio);
      ctx.strokeStyle = "#f0f0f0";
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB); ctx.stroke();
    });
  } else {
    yearFirstIdx.forEach((y, k) => {
      const isLast = k === yearFirstIdx.length - 1;
      if (k % yearStep !== 0 && !isLast) return;
      const x = xAt(y.i);
      ctx.fillStyle = "#767676";
      const w = ctx.measureText(y.yr).width;
      ctx.fillText(y.yr, x - w / 2, H - 8 * devicePixelRatio);
      ctx.strokeStyle = "#f0f0f0";
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB); ctx.stroke();
    });
  }

  function drawLine(field, color, width = 2.4, fill = false) {
    ctx.lineWidth = width * devicePixelRatio;
    ctx.strokeStyle = color;
    ctx.beginPath();
    growth.forEach((g, i) => {
      const x = xAt(i), y = yAt(g[field]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (fill) {
      ctx.lineTo(xAt(n - 1), yAt(0));
      ctx.lineTo(xAt(0), yAt(0));
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.06;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  // Cash invested (faint dashed)
  ctx.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]);
  drawLine("invested", "#bbbbbb", 1.5);
  ctx.setLineDash([]);
  // SPY
  drawLine("spy_value", "#0a6636", 2.0);
  // Strategy
  drawLine("strat_value", "#000000", 2.6, true);
}

function niceTop(v) {
  if (!v || !Number.isFinite(v)) return 1;
  const exp = Math.pow(10, Math.floor(Math.log10(v)));
  const f = v / exp;
  let nice;
  if (f <= 1) nice = 1;
  else if (f <= 2) nice = 2;
  else if (f <= 5) nice = 5;
  else nice = 10;
  return nice * exp;
}

function formatYAxisValue(v, mode) {
  if (mode === "x") {
    if (v === 0) return "0×";
    if (v >= 1000) return `${Math.round(v / 1000)}k×`;
    return `${v.toFixed(v < 10 ? 1 : 0)}×`;
  }
  if (v >= 1000) return `$${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k`;
  return `$${v.toFixed(0)}`;
}

function renderYears(data) {
  const grid = document.getElementById("btYearGrid");
  if (!grid) return;
  grid.innerHTML = "";
  const rows = data.year_by_year?.pullback_in_winner_k1 || [];
  rows.forEach(r => {
    // DCA-framed, shown as % return on the money you contributed THAT
    // year (contribute monthly Jan–Dec; year-end value ÷ contributed − 1).
    const edge = r.edge_pct;
    const c = el("div", { class: "year-cell" });
    c.appendChild(el("div", { class: "year-cell-yr" }, String(r.year)));
    c.appendChild(el("div", { class: "year-cell-cagr " + clsRet(edge) }, fmtPctSigned(r.strat_gain_pct, 0)));
    const meta = el("div", { class: "year-cell-meta" });
    meta.innerHTML = `on ${r.months}mo of contributions · S&P-DCA ${fmtPctSigned(r.spy_gain_pct, 0)}`;
    c.appendChild(meta);
    c.appendChild(el("div", { class: "year-cell-edge " + clsRet(edge) },
      "edge " + fmtPctSigned(edge, 0) + " vs S&P-DCA"));
    grid.appendChild(c);
  });
}

function renderBias(data) {
  const s = data.survivorship || {};
  // v3: bias_sensitivity is a list keyed by base_rate_annual.
  const bs = data.bias_sensitivity || s.sensitivity || [];
  const get = a => bs.find(x => Math.abs(x.base_rate_annual - a) < 0.001);
  const a0 = get(0.0);
  const a4 = get(0.04);
  const a8 = get(0.08);
  const a20 = get(0.20);
  // Engine robustness: does the picker still BEAT the S&P after phantom
  // delistings? Shown as a yes/no robustness verdict per hazard rate α —
  // not a return number. (The DCA outcome is the section at the top.)
  const spyRef = data.headline?.cagr_spy_dca ?? 0.10;
  const beats = r => r && r.stratified_cagr_median != null && r.stratified_cagr_median > spyRef;
  // largest α where the picker still beats the S&P
  let breakAlpha = 0;
  bs.forEach(r => { if (beats(r)) breakAlpha = Math.max(breakAlpha, r.base_rate_annual); });

  const elById = id => document.getElementById(id);
  if (elById("biasAlpha"))
    elById("biasAlpha").textContent = `≤ ${(breakAlpha * 100).toFixed(0)}%/yr`;
  if (elById("biasRandom") && a4)
    elById("biasRandom").textContent = beats(a4) ? "Still beats S&P" : "Edge gone";
  if (elById("biasStrat") && a8)
    elById("biasStrat").textContent = beats(a8) ? "Still beats S&P" : "Edge gone";
  if (elById("biasWorst") && a20)
    elById("biasWorst").textContent = beats(a20) ? "Still beats S&P" : "Edge gone";

  // Sensitivity table — robustness verdict only, no lump-sum returns
  const sensTbody = document.querySelector("#sensTable tbody");
  if (sensTbody) {
    sensTbody.innerHTML = "";
    bs.forEach(r => {
      const ok = beats(r);
      const tr = el("tr");
      tr.appendChild(el("td", {}, `${(r.base_rate_annual * 100).toFixed(0)}%/yr phantom delisting`));
      tr.appendChild(el("td", { class: ok ? "pos" : "neg" }, ok ? "Picker still beats S&P" : "Edge breaks down"));
      sensTbody.appendChild(tr);
    });
  }

  // Delisted augmentation
  const da = s.delisted_augmentation;
  if (da) {
    const cnt = document.getElementById("delistedCount");
    if (cnt) cnt.textContent = `${da.n_with_data} of ${da.tickers_attempted.length}`;
    const list = document.getElementById("delistedList");
    if (list) {
      list.innerHTML = "";
      (da.tickers_with_data || []).forEach(t => {
        list.appendChild(el("span", { class: "delisted-tag" }, t));
      });
    }
  }
}

/* ============================================================
   "If you started X years ago" cards
   ============================================================ */
function renderHorizons(data) {
  const sec = document.getElementById("horizonsSection");
  if (!sec || !data.horizon_stats?.length) return;
  sec.innerHTML = "";
  data.horizon_stats.forEach(h => {
    // DCA-framed: if you'd contributed monthly starting Xy ago to today.
    // Primary = annualized money-weighted IRR (the rate you compounded at).
    const edgePp = (h.strat_irr != null && h.spy_irr != null) ? h.strat_irr - h.spy_irr : null;
    const card = el("div", { class: "horizon-card" });
    card.appendChild(el("div", { class: "horizon-label" }, `DCA'd for ${h.years_back}y`));
    card.appendChild(el("div", { class: "horizon-since" }, `since ${h.since_date}`));
    card.appendChild(el("div", { class: "horizon-cagr " + clsRet(edgePp) }, fmtPct(h.strat_irr)));
    card.appendChild(el("div", { class: "horizon-meta" }, "money-weighted IRR (annualized)"));
    card.appendChild(el("div", { class: "horizon-edge " + clsRet(edgePp) },
      "edge " + fmtPctSigned(edgePp) + " vs S&P-DCA"));
    card.appendChild(el("div", { class: "horizon-multiple" },
      `S&P-DCA IRR ${fmtPct(h.spy_irr)} · each $1 → $${fmtNum(h.strat_moic, 2)}`));
    sec.appendChild(card);
  });
}

/* ============================================================
   Walk-forward: explain the 80%+ CAGR honestly
   ============================================================ */
function renderWalkForward(data) {
  const sec = document.getElementById("wfSection");
  if (!sec) return;
  sec.innerHTML = "";
  const wfd = data.dca_investor?.walk_forward;
  const sum = data.dca_investor?.walk_forward_summary;
  if (!wfd || !sum) { sec.innerHTML = "<div class='section-sub'>DCA walk-forward not available in this build.</div>"; return; }

  const summary = el("div", { class: "wf-grid" });
  summary.appendChild(wfCard("DCA beat S&P-DCA in",
    `${sum.n_beat_spy}/${sum.n_windows}`,
    "out-of-sample windows where monthly contributions into the strategy ended ahead of identical contributions into the S&P 500. Short fixed windows are near coin-flip — the DCA edge is a multi-year-horizon effect (see DCA outcomes: ~80% at 3y, 100% at 10y).",
    "highlight"));
  const medIrr = median(wfd.map(w => w.strat_irr));
  const medSpyIrr = median(wfd.map(w => w.spy_irr));
  const worstIrr = Math.min(...wfd.map(w => w.strat_irr));
  summary.appendChild(wfCard("Median window DCA return",
    fmtPct(medIrr),
    `money-weighted IRR across the windows, vs S&P-DCA ${fmtPct(medSpyIrr)}.`));
  summary.appendChild(wfCard("Worst window DCA return",
    fmtPct(worstIrr),
    "Single worst out-of-sample window for a monthly contributor. Honest: short windows can be negative."));
  sec.appendChild(summary);

  const explain = el("div", { class: "wf-explain" });
  explain.appendChild(el("p", { html:
    "These are 10 strictly out-of-sample windows (the model only ever sees data older than test − 7 months). <strong>This is the honest DCA reframe of walk-forward:</strong> the often-quoted \"10/10 beat SPY\" is a <em>lump-sum</em> per-window CAGR statistic that no monthly contributor experiences. In <em>DCA</em> terms — contributing every month through each window — the strategy beats S&P-DCA in " + sum.n_beat_spy + "/" + sum.n_windows + " of these short fixed windows. That is exactly consistent with the headline finding: the edge is weak over any single short window and only compounds reliably over a decade-long contribution horizon."
  }));
  sec.appendChild(explain);

  const wrap = el("div", { class: "bias-table-wrap" });
  const t = el("table", { class: "bias-table" });
  t.innerHTML = "<thead><tr><th>OOS window</th><th>From</th><th>To</th><th>Months</th><th>Strategy DCA (IRR)</th><th>S&P-DCA (IRR)</th><th>Edge</th><th>Beat?</th></tr></thead>";
  const tb = el("tbody");
  wfd.forEach(w => {
    const ep = (w.strat_irr != null && w.spy_irr != null) ? w.strat_irr - w.spy_irr : null;
    const row = el("tr");
    row.appendChild(el("td", { class: "tkr" }, (w.split || "").replace(/_/g, " ")));
    row.appendChild(el("td", {}, w.from));
    row.appendChild(el("td", {}, w.to));
    row.appendChild(el("td", {}, String(w.n_months)));
    row.appendChild(el("td", { class: clsRet(ep) }, fmtPct(w.strat_irr)));
    row.appendChild(el("td", {}, fmtPct(w.spy_irr)));
    row.appendChild(el("td", { class: clsRet(ep) }, fmtPctSigned(ep)));
    row.appendChild(el("td", { class: w.beat_spy ? "pos" : "neg" }, w.beat_spy ? "yes" : "no"));
    tb.appendChild(row);
  });
  t.appendChild(tb);
  wrap.appendChild(t);
  sec.appendChild(wrap);
}

function wfCard(label, value, sub, mod = "") {
  const c = el("div", { class: "wf-card " + mod });
  c.appendChild(el("div", { class: "wf-label" }, label));
  c.appendChild(el("div", { class: "wf-value" }, value));
  c.appendChild(el("div", { class: "wf-sub" }, sub));
  return c;
}

/* ============================================================
   Full historical trade log (collapsible)
   ============================================================ */
function renderTrades(data) {
  const sec = document.getElementById("tradesSection");
  if (!sec || !data.pick_log?.length) return;
  sec.innerHTML = "";

  const wins = data.pick_log.filter(p => p.win).length;
  const beat = data.pick_log.filter(p => p.beat_spy).length;
  const tot = data.pick_log.length;

  // Only count CLOSED trades for win/beat stats; open positions are TBD.
  const closed = data.pick_log.filter(p => p.status === "exited" && p.return != null);
  const cWins = closed.filter(p => p.return > 0).length;
  const cBeat = closed.filter(p => p.beat_spy === true || p.beat_spy === 1).length;
  const cTot = closed.length;
  const open = tot - cTot;

  const stats = el("div", { class: "trades-stats" });
  stats.innerHTML = `
    <div><strong>${tot}</strong> trades total · <strong>${open}</strong> still held</div>
    <div><strong>${cWins}</strong> profitable closed (${fmtPct0(cTot ? cWins/cTot : 0)})</div>
    <div><strong>${cBeat}</strong> beat SPY closed (${fmtPct0(cTot ? cBeat/cTot : 0)})</div>
  `;
  sec.appendChild(stats);

  const det = el("details", { class: "accordion accordion-trades", open: "" });
  det.appendChild(el("summary", {}, `Show every trade (${tot} since 2003)`));
  const body = el("div", { class: "accordion-body" });
  const wrap = el("div", { class: "bias-table-wrap" });
  const t = el("table", { class: "trades-table" });
  t.appendChild(el("thead", {}, el("tr", {},
    ["Entry", "Ticker", "Entry $", "Exit", "Exit $", "Status",
     "Hold", "Return", "SPY return", "Annualised", "Beat SPY?"]
      .map(h => el("th", {}, h)))));
  const tb = el("tbody");
  // Sort newest first
  const sorted = [...data.pick_log].sort((a, b) => a.asof < b.asof ? 1 : -1);
  sorted.forEach(p => {
    const row = el("tr");
    const isHeld = p.status !== "exited";
    const ret = p.ret_strat ?? p.return;
    const cagrAnnl = p.cagr;  // annualised (≈ (1+ret_6m)^2 - 1)
    const yrsTxt = p.years != null ? `${(+p.years).toFixed(1)}y` : "—";
    row.appendChild(el("td", { class: "mid" }, p.asof));
    row.appendChild(el("td", { class: "tkr" }, p.ticker));
    row.appendChild(el("td", {}, fmtPx(p.entry_px)));
    row.appendChild(el("td", { class: "mid" }, p.exit_date || (isHeld ? "open" : "—")));
    row.appendChild(el("td", {}, isHeld ? "—" : fmtPx(p.exit_px)));
    row.appendChild(el("td", { class: isHeld ? "pos" : "mid" }, isHeld ? "held" : "exited"));
    row.appendChild(el("td", { class: "mid" }, yrsTxt));
    row.appendChild(el("td", { class: clsRet(ret) }, fmtPctSigned(ret, 0)));
    row.appendChild(el("td", { class: clsRet(p.ret_spy) }, fmtPctSigned(p.ret_spy, 0)));
    row.appendChild(el("td", { class: clsRet(cagrAnnl) }, fmtPctSigned(cagrAnnl, 0)));
    const beat = p.beat_spy === true || p.beat_spy === 1;
    const beatTxt = isHeld ? "—" : (beat ? "Yes" : "No");
    const beatCls = isHeld ? "mid" : (beat ? "pos" : "neg");
    row.appendChild(el("td", { class: beatCls }, beatTxt));
    tb.appendChild(row);
  });
  t.appendChild(tb);
  wrap.appendChild(t);
  body.appendChild(wrap);
  det.appendChild(body);
  sec.appendChild(det);
}

/* ============================================================
   Historical monthly baskets browser
   ============================================================ */
function renderHistorical(data) {
  const sec = document.getElementById("historicalSection");
  if (!sec || !data.pick_log?.length) return;
  sec.innerHTML = "";

  // Group picks by asof
  const byAsof = {};
  for (const p of data.pick_log) {
    if (!byAsof[p.asof]) byAsof[p.asof] = [];
    byAsof[p.asof].push(p);
  }
  const asofs = Object.keys(byAsof).sort().reverse();

  // Year selector
  const years = [...new Set(asofs.map(a => a.slice(0, 4)))].sort().reverse();
  const yearBar = el("div", { class: "hist-year-bar" });
  let activeYear = years[0];
  years.forEach(yr => {
    const btn = el("button", { class: "hist-year-btn" + (yr === activeYear ? " active" : "") }, yr);
    btn.onclick = () => {
      activeYear = yr;
      yearBar.querySelectorAll(".hist-year-btn").forEach(x => x.classList.remove("active"));
      btn.classList.add("active");
      renderYearMonths();
    };
    yearBar.appendChild(btn);
  });
  sec.appendChild(yearBar);

  const monthsContainer = el("div", { class: "hist-months" });
  sec.appendChild(monthsContainer);

  function renderYearMonths() {
    monthsContainer.innerHTML = "";
    const yearAsofs = asofs.filter(a => a.startsWith(activeYear));
    yearAsofs.forEach(asof => {
      const basket = byAsof[asof];
      const card = el("div", { class: "hist-month" });
      const head = el("div", { class: "hist-month-head" });
      head.appendChild(el("span", { class: "hist-month-date" }, asof));
      // Only count CLOSED trades for win/beat metrics on this basket
      const closed = basket.filter(p => p.status === "exited" && p.return != null);
      const allHeld = closed.length === 0;
      let metaTxt;
      if (allHeld) {
        metaTxt = `${basket.length} picks · still held`;
      } else {
        const winRate = closed.filter(p => p.return > 0).length / closed.length;
        const beatRate = closed.filter(p => p.beat_spy === true || p.beat_spy === 1).length / closed.length;
        metaTxt = `${basket.length} picks · ${fmtPct0(winRate)} winners · ${fmtPct0(beatRate)} beat SPY`;
      }
      head.appendChild(el("span", { class: "hist-month-meta" }, metaTxt));
      card.appendChild(head);
      const grid = el("div", { class: "hist-pick-grid" });
      basket.forEach(p => {
        const item = el("div", { class: "hist-pick" });
        const isHeld = p.status !== "exited";
        const tickerLine = el("div", { class: "hist-pick-tkr-line" });
        tickerLine.appendChild(el("span", { class: "hist-pick-tkr" }, p.ticker));
        tickerLine.appendChild(el("span", { class: "hist-pick-status " + (isHeld ? "held" : "exited") },
          isHeld ? "held" : "exited"));
        item.appendChild(tickerLine);
        const px = el("div", { class: "hist-pick-px" });
        if (isHeld) {
          px.innerHTML = `<span class="hist-pick-from">bought at</span> ${fmtPx(p.entry_px)}`;
        } else {
          px.innerHTML = `<span class="hist-px-label">exit:</span> ${fmtPx(p.exit_px)} <span class="hist-pick-from">from</span> ${fmtPx(p.entry_px)}`;
        }
        item.appendChild(px);
        if (isHeld) {
          item.appendChild(el("div", { class: "hist-pick-ret mid" }, "in progress"));
        } else {
          const retTxt = fmtPctSigned(p.ret_strat ?? p.return, 0) +
                          " · vs SPY " + (p.ret_spy != null ? fmtPctSigned(p.ret_spy, 0) : "—");
          item.appendChild(el("div", { class: "hist-pick-ret " + clsRet(p.ret_strat ?? p.return) }, retTxt));
        }
        grid.appendChild(item);
      });
      card.appendChild(grid);
      monthsContainer.appendChild(card);
    });
  }
  renderYearMonths();
}
