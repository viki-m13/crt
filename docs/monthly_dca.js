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
const fmtX = x => x == null || !Number.isFinite(+x) ? "—" : `${(+x).toFixed(2)}×`;
const fmtPx = x => x == null ? "—" : "$" + (+x).toFixed(2);
const fmtNum = (x, d = 2) => x == null ? "—" : (+x).toFixed(d);

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
  renderTrades(data);
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
  if (!basket.length) {
    body.innerHTML = "<div>No picks available for this month.</div>";
    return;
  }
  const grid = el("div", { class: "basket-grid" });
  basket.forEach((p, i) => {
    const card = el("div", { class: "basket-card" + (i === 0 ? " first" : "") });
    card.appendChild(el("div", { class: "basket-tkr-row" }, [
      el("span", { class: "basket-rank" }, "#" + (i + 1)),
      el("span", { class: "basket-tkr" }, p.ticker),
      el("span", { class: "basket-tkr-px" }, fmtPx(p.price)),
    ]));
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
    ["Strategy multiple", fmtX(final_strat / final_cash), "$" + final_cash.toFixed(0) + " in → $" + final_strat.toFixed(0) + " out", "ok"],
    ["SPY multiple", fmtX(final_spy / final_cash), "$" + final_cash.toFixed(0) + " in → $" + final_spy.toFixed(0) + " out", "ok"],
    ["Walk-forward CAGR", fmtPct(wf?.mean_test_cagr), "mean across " + (wf?.n_splits_with_test_data || "?") + " out-of-sample splits", "ok"],
  ];
  items.forEach(([lbl, val, sub, cls]) => {
    const c = el("div", { class: "bt-stat-card " + cls });
    c.appendChild(el("div", { class: "bt-stat-label" }, lbl));
    c.appendChild(el("div", { class: "bt-stat-value" }, val));
    c.appendChild(el("div", { class: "bt-stat-sub" }, sub));
    stats.appendChild(c);
  });

  // Draw chart
  const canvas = document.getElementById("btChart");
  if (canvas && data.growth) {
    requestAnimationFrame(() => drawGrowth(canvas, data.growth));
    window.addEventListener("resize", () => drawGrowth(canvas, data.growth), { passive: true });
  }
}

function drawGrowth(canvas, growth) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
  const H = canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
  ctx.clearRect(0, 0, W, H);

  if (!growth || growth.length < 2) return;
  const padL = 56 * devicePixelRatio, padR = 16 * devicePixelRatio,
        padT = 18 * devicePixelRatio, padB = 32 * devicePixelRatio;
  const n = growth.length;
  let maxY = 0;
  growth.forEach(g => {
    maxY = Math.max(maxY, g.strat_value, g.spy_value, g.invested);
  });
  // Always include 0
  const minY = 0;
  // Round maxY up to nice number
  const niceMax = niceTop(maxY);

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
    ctx.fillText(`$${v.toFixed(0)}`, 4 * devicePixelRatio, y + 4 * devicePixelRatio);
  }

  // X-axis labels (years)
  const seenYears = new Set();
  growth.forEach((g, i) => {
    const yr = String(g.date).slice(0, 4);
    if (!seenYears.has(yr)) {
      seenYears.add(yr);
      const x = xAt(i);
      ctx.fillStyle = "#767676";
      ctx.fillText(yr, x - 12 * devicePixelRatio, H - 8 * devicePixelRatio);
      ctx.strokeStyle = "#f0f0f0";
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB); ctx.stroke();
    }
  });

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
  const s = data.survivorship;
  if (!s) return;

  const stratCagr = s.stratified_default_4pct?.cagr_dca_median;
  const stratCagrP10 = s.stratified_default_4pct?.cagr_dca_p10;
  const random = s.random_baseline_k1?.cagr_mean;
  const rawCagr = data.headline?.cagr_raw;
  const alpha = (rawCagr != null && random != null) ? rawCagr - random : null;
  const worst = (s.sensitivity || []).find(x => x.base_rate_annual >= 0.20);

  const elById = id => document.getElementById(id);
  if (elById("biasAlpha")) {
    elById("biasAlpha").textContent = fmtPctSigned(alpha, 1);
  }
  if (elById("biasRandom")) {
    elById("biasRandom").textContent = fmtPct(random);
  }
  if (elById("biasStrat")) {
    const txt = fmtPct(stratCagr) + (stratCagrP10 != null ? `  (p10 ${fmtPct(stratCagrP10)})` : "");
    elById("biasStrat").textContent = txt;
  }
  if (elById("biasWorst") && worst) {
    elById("biasWorst").textContent = fmtPct(worst.stratified_cagr_median);
  }

  // Sensitivity table
  const sensTbody = document.querySelector("#sensTable tbody");
  if (sensTbody) {
    sensTbody.innerHTML = "";
    (s.sensitivity || []).forEach(r => {
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
      `$1/mo → $${h.strat_terminal.toFixed(0)}  (SPY DCA: $${h.spy_terminal.toFixed(0)})`));
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

  const stats = el("div", { class: "trades-stats" });
  stats.innerHTML = `
    <div><strong>${tot}</strong> trades total</div>
    <div><strong>${wins}</strong> profitable (${fmtPct0(wins/tot)})</div>
    <div><strong>${beat}</strong> beat SPY held-to-today (${fmtPct0(beat/tot)})</div>
  `;
  sec.appendChild(stats);

  const det = el("details", { class: "accordion accordion-trades", open: "" });
  det.appendChild(el("summary", {}, "Show every monthly trade since 2002"));
  const body = el("div", { class: "accordion-body" });
  const wrap = el("div", { class: "bias-table-wrap" });
  const t = el("table", { class: "trades-table" });
  t.appendChild(el("thead", {}, el("tr", {},
    ["Asof", "Ticker", "Entry", "Current", "Years",
     "Strategy ret", "SPY ret", "Strategy CAGR", "SPY CAGR", "Beat SPY?"]
      .map(h => el("th", {}, h)))));
  const tb = el("tbody");
  // Sort newest first
  const sorted = [...data.pick_log].sort((a, b) => a.asof < b.asof ? 1 : -1);
  sorted.forEach(p => {
    const row = el("tr");
    row.appendChild(el("td", { class: "mid" }, p.asof));
    row.appendChild(el("td", { class: "tkr" }, p.ticker));
    row.appendChild(el("td", {}, fmtPx(p.entry_px)));
    row.appendChild(el("td", {}, fmtPx(p.current_px)));
    row.appendChild(el("td", { class: "mid" }, p.years_held?.toFixed(1) ?? "—"));
    row.appendChild(el("td", { class: clsRet(p.ret_strat) }, fmtPctSigned(p.ret_strat, 0)));
    row.appendChild(el("td", { class: clsRet(p.ret_spy) }, fmtPctSigned(p.ret_spy, 0)));
    row.appendChild(el("td", { class: clsRet(p.cagr_strat) }, fmtPctSigned(p.cagr_strat, 0)));
    row.appendChild(el("td", {}, fmtPctSigned(p.cagr_spy, 0)));
    row.appendChild(el("td", { class: p.beat_spy ? "pos" : "neg" }, p.beat_spy ? "Yes" : "No"));
    tb.appendChild(row);
  });
  t.appendChild(tb);
  wrap.appendChild(t);
  body.appendChild(wrap);
  det.appendChild(body);
  sec.appendChild(det);
}
