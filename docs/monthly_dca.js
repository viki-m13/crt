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
  renderYears(data);
  renderBias(data);
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
  const p = data.pick_of_month;
  if (!p) {
    body.innerHTML = "<div>No pick available for this month.</div>";
    return;
  }
  // Pretty narrative
  const why = [];
  if (p.pullback_1y != null) why.push(`down <strong>${fmtPctSigned(p.pullback_1y, 0)}</strong> from its 1-year high`);
  if (p.trend_health_5y != null) why.push(`<strong>${fmtPct0(p.trend_health_5y)}</strong> of the last 5 years above its 200-day moving average`);
  if (p.recovery_rate != null) why.push(`recovered <strong>${fmtPct0(p.recovery_rate)}</strong> of past similar drawdowns`);
  if (p.rsi_14 != null) why.push(`14-day RSI of <strong>${Math.round(p.rsi_14)}</strong> (not in freefall)`);

  const card = el("div", { class: "pick-card" });
  card.appendChild(el("div", { class: "pick-tkr-row" }, [
    el("span", { class: "pick-tkr" }, p.ticker),
    el("span", { class: "pick-tkr-px" }, fmtPx(p.price) + " at last close"),
  ]));
  card.appendChild(el("div", { class: "pick-rationale", html: "Why this pick: " + why.join(", ") + "." }));
  const stats = el("div", { class: "pick-stats" });
  const statRows = [
    ["1-year pullback", fmtPctSigned(p.pullback_1y, 0), clsRet(p.pullback_1y)],
    ["5-year trend health", fmtNum(p.trend_health_5y, 2), ""],
    ["3-year momentum", fmtPctSigned(p.mom_3y, 0), clsRet(p.mom_3y)],
    ["12-month momentum", fmtPctSigned(p.mom_12_1, 0), clsRet(p.mom_12_1)],
    ["RSI(14)", Math.round(p.rsi_14 || 0), ""],
    ["Distance from 200dma", fmtPctSigned(p.d_sma200, 0), clsRet(p.d_sma200)],
  ];
  statRows.forEach(([lbl, val, c]) => {
    const item = el("div", { class: "pick-stat" });
    item.appendChild(el("div", { class: "pick-stat-label" }, lbl));
    item.appendChild(el("div", { class: "pick-stat-value " + c }, String(val)));
    stats.appendChild(item);
  });
  card.appendChild(stats);
  body.appendChild(card);
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
