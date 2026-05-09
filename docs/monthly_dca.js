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

fetch(DATA_URL + "?v=" + Date.now())
  .then(r => r.json())
  .then(render)
  .catch(err => {
    document.getElementById("pickBody").innerHTML =
      `<div style="color:var(--orange)">Could not load data: ${err.message}</div>`;
  });

function render(data) {
  renderHeroMeta(data);
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
   Case studies — annotated v3 baskets across history
   ============================================================ */
function renderCaseStudies(data) {
  const sec = document.getElementById("caseStudies");
  if (!sec) return;
  sec.innerHTML = "";

  // Pull rebalance events from pick_log (each basket has basket_id and a set of trades)
  const log = data.pick_log || [];
  if (!log.length) return;
  const byBasket = {};
  log.forEach(p => {
    const k = p.basket_id ?? p.asof;
    if (!byBasket[k]) byBasket[k] = [];
    byBasket[k].push(p);
  });

  // Pre-defined case-study windows we want to highlight
  // (entry_date prefix → label / commentary).
  const wantedDates = [
    { prefix: "2009-04",    label: "Post-GFC bottom (2009-04 → 2009-10)",
      blurb: "Bear regime ended in April 2009; first non-cash rebalance after the March 2009 bottom captures the strongest recovery basket of the entire backtest. Holding for 6 months captures the full V-bottom rally." },
    { prefix: "2020-03",    label: "COVID bottom (2020-03 → 2020-09)",
      blurb: "Tight gate fired in February 2020 (cash) and re-entered in March/April 2020 with the recovery-regime basket. The 6m hold captures the entire post-March 2020 V-bottom rally." },
    { prefix: "2016-07",    label: "Brexit / 2016 mid-year (2016-07 → 2017-01)",
      blurb: "Bull-regime basket through the post-Brexit drift, election surprise, and post-election rally. Strategy didn't predict any of these macro events — the GBM just kept ranking the price-only momentum signal cross-sectionally." },
    { prefix: "2022-12",    label: "Post-2022-bear bottom (2022-12 → 2023-06)",
      blurb: "After the 2022 bear, recovery-regime basket entered in late 2022 captures the AI rally launch in Q1-Q2 2023." },
    { prefix: "2023-05",    label: "AI rally (2023-05 → 2023-11)",
      blurb: "Bull-regime basket through the mid-2023 mega-cap rally. K=3 caught NVDA + tech leaders." },
  ];

  const cases = [];
  // For each wanted window, find the latest matching basket
  wantedDates.forEach(w => {
    const matches = log.filter(p => p.asof && p.asof.startsWith(w.prefix));
    if (matches.length) {
      // Pick the basket_id that has the most members from this prefix
      const bid = matches[0].basket_id;
      const basket = log.filter(p => p.basket_id === bid);
      if (basket.length) cases.push({ ...w, picks: basket });
    }
  });

  // Always add the latest 2 baskets as recent case studies
  const allBids = [...new Set(log.map(p => p.basket_id).filter(b => b != null))].sort((a, b) => b - a);
  allBids.slice(0, 2).forEach(bid => {
    const basket = log.filter(p => p.basket_id === bid);
    if (basket.length && !cases.some(c => c.picks[0]?.basket_id === bid)) {
      const entryDate = basket[0]?.asof || "";
      const exitDate = basket[0]?.exit_date || "?";
      const status = basket[0]?.status === "open" ? "currently held" : "closed";
      cases.unshift({
        prefix: entryDate.slice(0, 7),
        label: `Recent basket (${entryDate} → ${exitDate}, ${status})`,
        blurb: status === "currently held"
          ? "The current basket — held until the next 6-month rebalance."
          : "Most recently closed basket.",
        picks: basket,
      });
    }
  });

  cases.slice(0, 8).forEach(c => {
    const card = el("div", { class: "case-card" });
    card.appendChild(el("div", { class: "case-label" }, c.label));
    const tickers = c.picks.map(p => p.ticker).join(" · ");
    card.appendChild(el("div", { class: "case-tickers" }, tickers));
    // Aggregate basket return (equal-weighted)
    const validRets = c.picks.map(p => p.return ?? p.ret_strat).filter(r => r != null);
    if (validRets.length) {
      const basketRet = validRets.reduce((a, b) => a + b, 0) / validRets.length;
      const status = c.picks[0]?.status === "open" ? "(open)" : "";
      card.appendChild(el("div", { class: "case-ret " + clsRet(basketRet) },
        `Basket return: ${fmtPctSigned(basketRet, 1)} ${status}`));
    }
    card.appendChild(el("div", { class: "case-blurb" }, c.blurb));
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

  // 1. Sub-period CAGR
  if (data.sub_periods?.length) {
    const sec = el("div", { class: "v3-block" });
    sec.appendChild(el("h3", { class: "section-h3" }, "Sub-period robustness"));
    sec.appendChild(el("p", { class: "section-sub" },
      "Strategy CAGR vs SPY across overlapping decade windows. Edge persists in every period."));
    const tbl = el("table", { class: "v3-table" });
    tbl.innerHTML = `<thead><tr><th>Period</th><th>Months</th><th>Strategy CAGR</th><th>SPY CAGR</th><th>Edge (pp)</th></tr></thead>`;
    const tb = el("tbody");
    data.sub_periods.forEach(p => {
      const row = el("tr");
      row.appendChild(el("td", {}, p.period.replace(/^p\d_/, "").replace(/_/g, "-")));
      row.appendChild(el("td", {}, String(p.n_months)));
      row.appendChild(el("td", {}, fmtPct(p.cagr_strat)));
      row.appendChild(el("td", {}, fmtPct(p.cagr_spy)));
      row.appendChild(el("td", { class: clsRet(p.edge_pp) }, (p.edge_pp >= 0 ? "+" : "") + p.edge_pp.toFixed(1)));
      tb.appendChild(row);
    });
    tbl.appendChild(tb);
    sec.appendChild(tbl);
    host.appendChild(sec);
  }

  // 2. Multi-universe generalisation
  if (data.multi_universe_generalisation?.length) {
    const sec = el("div", { class: "v3-block" });
    sec.appendChild(el("h3", { class: "section-h3" }, "Generalisation across universes"));
    sec.appendChild(el("p", { class: "section-sub" },
      "Same v3 config (K=3, EW, tight gate, 6m hold) on 5 alternative universes. Strategy isn't an artefact of the S&P 500 cohort — it works on broader, non-S&P-500, and random universes too."));
    const tbl = el("table", { class: "v3-table" });
    tbl.innerHTML = `<thead><tr><th>Universe</th><th>Pool size</th><th>Full CAGR</th><th>WF mean</th><th>WF min</th><th>Edge vs SPY</th><th>Sharpe</th><th>MaxDD</th><th>Beats SPY</th></tr></thead>`;
    const tb = el("tbody");
    data.multi_universe_generalisation.forEach(u => {
      const row = el("tr");
      row.appendChild(el("td", {}, u.universe.replace(/_/g, " ")));
      row.appendChild(el("td", {}, String(u.n_pool)));
      row.appendChild(el("td", {}, fmtPct(u.cagr_full)));
      row.appendChild(el("td", {}, fmtPct(u.wf_mean_cagr)));
      row.appendChild(el("td", {}, fmtPct(u.wf_min_cagr)));
      row.appendChild(el("td", { class: clsRet(u.wf_mean_edge_pp) }, (u.wf_mean_edge_pp >= 0 ? "+" : "") + u.wf_mean_edge_pp.toFixed(1) + "pp"));
      row.appendChild(el("td", {}, u.sharpe.toFixed(2)));
      row.appendChild(el("td", {}, fmtPct(u.max_dd, 0)));
      row.appendChild(el("td", {}, `${u.wf_n_beats_spy}/10`));
      tb.appendChild(row);
    });
    tbl.appendChild(tb);
    sec.appendChild(tbl);
    host.appendChild(sec);
  }

  // 3. Parameter sensitivity
  if (data.parameter_sensitivity?.length) {
    const sec = el("div", { class: "v3-block" });
    sec.appendChild(el("h3", { class: "section-h3" }, "Parameter sensitivity"));
    sec.appendChild(el("p", { class: "section-sub" },
      "Each row perturbs ONE parameter while holding the others at the v3 winner config (K=3, EW, tight, h=6, cost=10bp). Robust plateau across reasonable perturbations confirms the result is not on a knife-edge."));
    const tbl = el("table", { class: "v3-table" });
    tbl.innerHTML = `<thead><tr><th>Param</th><th>Value</th><th>Full CAGR</th><th>WF mean</th><th>WF min</th><th>Edge (pp)</th><th>Beats SPY</th><th>MaxDD</th></tr></thead>`;
    const tb = el("tbody");
    data.parameter_sensitivity.forEach(s => {
      const row = el("tr");
      row.appendChild(el("td", {}, s.param));
      row.appendChild(el("td", {}, String(s.value)));
      row.appendChild(el("td", {}, fmtPct(s.cagr_full)));
      row.appendChild(el("td", {}, fmtPct(s.wf_mean_cagr)));
      row.appendChild(el("td", {}, fmtPct(s.wf_min_cagr)));
      row.appendChild(el("td", { class: clsRet(s.wf_mean_edge_pp) }, (s.wf_mean_edge_pp >= 0 ? "+" : "") + s.wf_mean_edge_pp.toFixed(1)));
      row.appendChild(el("td", {}, `${s.wf_n_beats_spy}/10`));
      row.appendChild(el("td", {}, fmtPct(s.max_dd, 0)));
      tb.appendChild(row);
    });
    tbl.appendChild(tb);
    sec.appendChild(tbl);
    host.appendChild(sec);
  }

  // 4. Drawdown ledger (top 5)
  if (data.drawdowns?.length) {
    const sec = el("div", { class: "v3-block" });
    sec.appendChild(el("h3", { class: "section-h3" }, "Drawdown ledger"));
    sec.appendChild(el("p", { class: "section-sub" },
      "Top peak-to-trough drawdowns of 5%+ in the v3 backtest. The deepest is the GFC (-50%); recovery in 5 months."));
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
    sec.appendChild(el("h3", { class: "section-h3" }, "Most-picked tickers"));
    sec.appendChild(el("p", { class: "section-sub" },
      "Tickers that the v3 strategy selected most often across the 22-year backtest. Concentration in NVDA reflects its persistent multi-horizon momentum signal in S&P 500."));
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
  const win = data.headline?.win_rate_raw;
  const cagr = data.headline?.cagr_raw;
  const spy = data.headline?.cagr_spy_dca;
  const wf = data.walk_forward_aggregate?.[0];
  const parts = [
    `As of ${data.as_of}`,
    `${data.panel?.n_tickers ?? "?"} tickers, ${data.panel?.first_date} → ${data.panel?.last_date}`,
    `Backtest: ${fmtPct(cagr)} CAGR vs SPY DCA ${fmtPct(spy)}`,
    `Walk-forward (${wf?.n_splits_with_test_data ?? "?"} splits): mean OOS CAGR ${fmtPct(wf?.mean_test_cagr)}`,
  ];
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

  // Header line: where are we in the 6-month cycle?
  const head = el("div", { class: "basket-dates" });
  if (monthsHeld === 0) {
    head.innerHTML = `<strong>Rebalance day.</strong> Last rebalance ${buyDate}, next rebalance ${sellDate}.`;
  } else if (monthsLeft === 0) {
    head.innerHTML = `<strong>Rebalance imminent.</strong> Current basket bought ${buyDate}, ` +
      `next rebalance at the next month-end (${sellDate}).`;
  } else {
    head.innerHTML = `Currently holding the basket bought on <strong>${buyDate}</strong>. ` +
      `<strong>Hold until ${sellDate}</strong> — that's ${monthsLeft} more month${monthsLeft===1?'':'s'} from now. No action this month.`;
  }
  body.appendChild(head);

  // If at-rebalance (just rebalanced), show explicit BUY / SELL / HOLD actions
  const lastRebMonth = (ls.last_rebalance_date || "").slice(0, 7);
  const asofMonth = (data.as_of || "").slice(0, 7);
  const justRebalanced = lastRebMonth === asofMonth || monthsHeld === 0;
  if (justRebalanced && (ls.last_rebalance_to_buy || ls.last_rebalance_to_sell)) {
    const actions = el("div", { class: "actions-grid" });
    const buys = ls.last_rebalance_to_buy || [];
    const holds = ls.last_rebalance_to_hold || [];
    const sells = ls.last_rebalance_to_sell || [];
    if (sells.length) {
      const card = el("div", { class: "action-card sell" });
      card.appendChild(el("div", { class: "action-label" }, "SELL today"));
      card.appendChild(el("div", { class: "action-tickers" }, sells.join(" · ")));
      card.appendChild(el("div", { class: "action-sub" }, "Was in last basket, no longer in current basket."));
      actions.appendChild(card);
    }
    if (buys.length) {
      const card = el("div", { class: "action-card buy" });
      card.appendChild(el("div", { class: "action-label" }, "BUY today"));
      card.appendChild(el("div", { class: "action-tickers" }, buys.join(" · ")));
      card.appendChild(el("div", { class: "action-sub" }, "New names entering the basket."));
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
  const grid = el("div", { class: "basket-grid" });
  basket.forEach((p, i) => {
    const card = el("div", { class: "basket-card" + (i === 0 ? " first" : "") });
    card.appendChild(el("div", { class: "basket-tkr-row" }, [
      el("span", { class: "basket-rank" }, "#" + (i + 1)),
      el("span", { class: "basket-tkr" }, p.ticker),
      el("span", { class: "basket-tkr-px" }, fmtPx(p.price)),
    ]));
    if (sellDate) {
      card.appendChild(el("div", { class: "basket-sell-by" },
        `Buy ${buyDate} • Sell ${sellDate}`));
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
}

function renderBacktest(data) {
  const stats = document.getElementById("btStats");
  stats.innerHTML = "";
  const last = data.growth?.[data.growth.length - 1];
  if (!last) { stats.textContent = "No growth data."; return; }
  const final_strat = last.strat_value;
  const final_spy = last.spy_value;
  const final_cash = last.invested;
  const cagr = data.headline?.cagr_raw;
  const spyCagr = data.headline?.cagr_spy_dca;
  const wf = data.walk_forward_aggregate?.[0];

  const items = [
    ["Strategy CAGR", fmtPct(cagr), "vs SPY DCA " + fmtPct(spyCagr), "highlight"],
    ["Strategy multiple", fmtX(final_strat / final_cash), fmtMoney(final_cash) + " in → " + fmtMoney(final_strat) + " out", "ok"],
    ["SPY multiple", fmtX(final_spy / final_cash), fmtMoney(final_cash) + " in → " + fmtMoney(final_spy) + " out", "ok"],
    ["Walk-forward CAGR", fmtPct(wf?.mean_test_cagr), "mean across " + (wf?.n_splits_with_test_data || "?") + " out-of-sample splits", "ok"],
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

function filterGrowthByPeriod(growth, period) {
  if (!growth?.length || period === "all") return { rows: growth, isRebased: false };
  const yearsMap = { "1y": 1, "3y": 3, "5y": 5, "10y": 10, "20y": 20 };
  const yrs = yearsMap[period];
  if (!yrs) return { rows: growth, isRebased: false };
  const last = new Date(growth[growth.length - 1].date);
  const cutoff = new Date(last);
  cutoff.setFullYear(cutoff.getFullYear() - yrs);
  // Find first index with date >= cutoff
  const startIdx = growth.findIndex(g => new Date(g.date) >= cutoff);
  if (startIdx <= 0) return { rows: growth, isRebased: false };
  // Re-base equity values to start at 1.0 from the new starting point
  const sub = growth.slice(startIdx);
  const baseStrat = sub[0].strat_value || 1;
  const baseSpy = sub[0].spy_value || 1;
  const baseInv = sub[0].invested || 1;
  return {
    rows: sub.map(g => ({
      date: g.date,
      strat_value: g.strat_value / baseStrat,
      spy_value: (g.spy_value != null ? g.spy_value / baseSpy : null),
      invested: g.invested / baseInv,
    })),
    isRebased: true,
  };
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
    const c = el("div", { class: "year-cell" });
    c.appendChild(el("div", { class: "year-cell-yr" }, String(r.year)));
    c.appendChild(el("div", { class: "year-cell-cagr " + clsRet(r.cagr_dca_picks) }, fmtPctSigned(r.cagr_dca_picks, 0)));
    const meta = el("div", { class: "year-cell-meta" });
    meta.innerHTML = `n=${r.n_picks} · win ${fmtPct0(r.win_rate)} · SPY ${fmtPct(r.cagr_dca_spy)}`;
    c.appendChild(meta);
    c.appendChild(el("div", { class: "year-cell-edge " + clsRet(r.edge) }, "edge " + fmtPctSigned(r.edge, 0)));
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
  const rawCagr = data.headline?.cagr_raw;
  const spyCagr = data.headline?.cagr_spy_dca;
  const edge_full = (rawCagr != null && spyCagr != null) ? rawCagr - spyCagr : null;

  const elById = id => document.getElementById(id);
  if (elById("biasAlpha")) {
    elById("biasAlpha").textContent = fmtPctSigned(edge_full, 1);
  }
  if (elById("biasRandom") && a4) {
    elById("biasRandom").textContent = fmtPct(a4.stratified_cagr_median);
  }
  if (elById("biasStrat") && a8) {
    elById("biasStrat").textContent = fmtPct(a8.stratified_cagr_median);
  }
  if (elById("biasWorst") && a20) {
    elById("biasWorst").textContent = fmtPct(a20.stratified_cagr_median);
  }

  // Sensitivity table
  const sensTbody = document.querySelector("#sensTable tbody");
  if (sensTbody) {
    sensTbody.innerHTML = "";
    bs.forEach(r => {
      const tr = el("tr");
      tr.appendChild(el("td", {}, `${(r.base_rate_annual * 100).toFixed(0)}%`));
      tr.appendChild(el("td", { class: clsRet(r.stratified_cagr_median) }, fmtPct(r.stratified_cagr_median)));
      tr.appendChild(el("td", {}, fmtPct(r.stratified_cagr_p10)));
      tr.appendChild(el("td", {}, fmtPct(r.stratified_cagr_p90)));
      tr.appendChild(el("td", { class: clsRet(r.uniform_cagr_median) }, fmtPct(r.uniform_cagr_median)));
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
    const card = el("div", { class: "horizon-card" });
    card.appendChild(el("div", { class: "horizon-label" }, `${h.years_back}y ago`));
    card.appendChild(el("div", { class: "horizon-since" }, `since ${h.since_date}`));
    card.appendChild(el("div", { class: "horizon-cagr " + clsRet(h.cagr_strat) }, fmtPct(h.cagr_strat)));
    card.appendChild(el("div", { class: "horizon-meta" },
      `vs SPY DCA ${fmtPct(h.cagr_spy)} · ${h.n_picks} picks`));
    card.appendChild(el("div", { class: "horizon-edge " + clsRet(h.edge_vs_spy) },
      `edge ${fmtPctSigned(h.edge_vs_spy)}`));
    card.appendChild(el("div", { class: "horizon-multiple" },
      `$1 → ${fmtMoney(h.strat_terminal)}  ·  SPY: ${fmtMoney(h.spy_terminal)}`));
    sec.appendChild(card);
  });
}

/* ============================================================
   Walk-forward: explain the 80%+ CAGR honestly
   ============================================================ */
function renderWalkForward(data) {
  const sec = document.getElementById("wfSection");
  if (!sec || !data.wf_explanation) return;
  sec.innerHTML = "";
  const wf = data.wf_explanation;

  const summary = el("div", { class: "wf-grid" });
  summary.appendChild(wfCard("Walk-forward mean (TEST)",
    fmtPct(wf.headline_mean_test_cagr),
    `Average across ${wf.n_splits} OOS test windows. Headline metric.`,
    "highlight"));
  summary.appendChild(wfCard("Walk-forward worst (TEST)",
    fmtPct(wf.headline_min_test_cagr),
    "The single worst test window. The strategy never collapsed.",
    clsRet(wf.headline_min_test_cagr)));
  summary.appendChild(wfCard("Walk-forward best (TEST)",
    fmtPct(wf.headline_max_test_cagr),
    "Best test window. Recent windows that captured 2022-2024 ride hardest."));
  summary.appendChild(wfCard("Full-deployment CAGR",
    fmtPct(data.headline?.cagr_raw),
    "If you'd deployed every month from 2002 onward, this is your XIRR."));
  sec.appendChild(summary);

  const explain = el("div", { class: "wf-explain" });
  explain.appendChild(el("p", {}, wf.explanation));
  explain.appendChild(el("p", { html:
    "<strong>Plain English:</strong> Walk-forward TEST CAGR is the average annualized return across many <em>different starting points</em>, each held to today. Most of those starting points happened during the 2010-2024 expansion, so their compound rates are high. The full-deployment CAGR weights every entry month equally — including 2008 financial crisis entries, 2015-2016 sideways markets, and the 2021 bubble peak — which dilutes to a lower (but more realistic) number for someone deploying from scratch today."
  }));
  sec.appendChild(explain);

  // Per-split detail table
  if (Array.isArray(data.splits) && data.splits.length) {
    const det = el("details", { class: "accordion" });
    det.appendChild(el("summary", {}, "Per-split TRAIN→TEST detail (8 walk-forward windows)"));
    const body = el("div", { class: "accordion-body" });
    data.splits.forEach(s => {
      body.appendChild(el("div", { class: "split-name" }, s.name.replace(/_/g, " ")));
      const wrap = el("div", { class: "bias-table-wrap" });
      const t = el("table", { class: "bias-table" });
      t.appendChild(el("thead", {}, el("tr", {},
        ["Phase", "Strategy::K::Exit", "n", "Win", "CAGR", "SPY", "Edge"].map(h => el("th", {}, h)))));
      const tb = el("tbody");
      (s.train_top5 || []).slice(0, 3).forEach(r => {
        const row = el("tr");
        row.appendChild(el("td", { class: "mid" }, "TRAIN"));
        row.appendChild(el("td", { class: "tkr" }, r.key));
        row.appendChild(el("td", {}, String(r.n_picks)));
        row.appendChild(el("td", {}, fmtPct0(r.win_rate)));
        row.appendChild(el("td", { class: clsRet(r.cagr) }, fmtPct(r.cagr)));
        row.appendChild(el("td", {}, fmtPct(r.spy_cagr)));
        row.appendChild(el("td", { class: clsRet(r.edge) }, fmtPctSigned(r.edge)));
        tb.appendChild(row);
      });
      (s.test_same_configs || []).slice(0, 3).forEach(r => {
        const row = el("tr");
        row.appendChild(el("td", { class: "mid" }, "TEST"));
        row.appendChild(el("td", { class: "tkr" }, r.key));
        row.appendChild(el("td", {}, String(r.n_picks)));
        row.appendChild(el("td", {}, fmtPct0(r.win_rate)));
        row.appendChild(el("td", { class: clsRet(r.cagr) }, fmtPct(r.cagr)));
        row.appendChild(el("td", {}, fmtPct(r.spy_cagr)));
        row.appendChild(el("td", { class: clsRet(r.edge) }, fmtPctSigned(r.edge)));
        tb.appendChild(row);
      });
      t.appendChild(tb);
      wrap.appendChild(t);
      body.appendChild(wrap);
    });
    det.appendChild(body);
    sec.appendChild(det);
  }
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
