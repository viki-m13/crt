/* Max Strategy v9-max — 10D/30D/60D/3M/6M/1Y/3Y/5Y horizons
   Three sections: Top Recommendations, Stocks, Crypto. Each with its own
   backtest. Ranking in the backtest uses point-in-time final score (no
   look-ahead); the current listing uses the latest scan's analog probabilities. */

const DATA_URL = "./data/full.json";

const HORIZONS = [
  { id: "10D", prob: "prob_10d", median: "median_10d", downside: "downside_10d", days: 10,   label: "10 Days" },
  { id: "30D", prob: "prob_30d", median: "median_30d", downside: null,           days: 30,   label: "30 Days" },
  { id: "60D", prob: "prob_60d", median: "median_60d", downside: null,           days: 60,   label: "60 Days" },
  { id: "3M",  prob: "prob_3m",  median: "median_3m",  downside: null,           days: 63,   label: "3 Months" },
  { id: "6M",  prob: "prob_6m",  median: "median_6m",  downside: null,           days: 126,  label: "6 Months" },
  { id: "1Y",  prob: "prob_1y",  median: "median_1y",  downside: "downside_1y",  days: 252,  label: "1 Year" },
  { id: "3Y",  prob: "prob_3y",  median: "median_3y",  downside: null,           days: 756,  label: "3 Years" },
  { id: "5Y",  prob: "prob_5y",  median: "median_5y",  downside: null,           days: 1260, label: "5 Years" },
];
const HORIZON_BY_ID = Object.fromEntries(HORIZONS.map(h => [h.id, h]));

const SECTION_DEFAULT_HORIZON = { stocks: "1Y", crypto: "60D" };
const SECTION_BENCHMARK = { stocks: ["SPY"], crypto: ["BTC-USD"], top: ["SPY", "BTC-USD"] };
const DCA_MONTHLY = 1000;

let FULL = null;
let ITEMS_STOCKS = [];
let ITEMS_CRYPTO = [];
let BT = null; // built once from bt_series

/* ---------- util ---------- */
function byId(id) { return document.getElementById(id); }
function fmtPct(v) {
  if (v == null || !Number.isFinite(v)) return "—";
  return (v * 100).toFixed(1) + "%";
}
function fmtPctSigned(v) {
  if (v == null || !Number.isFinite(v)) return "—";
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
}
function fmtProb(v) {
  if (v == null || !Number.isFinite(v)) return "—";
  // v is 0-100 here
  return v.toFixed(0) + "%";
}
function fmtNum(v, d = 0) {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: d });
}
function horizonExists(items, hid) {
  const h = HORIZON_BY_ID[hid];
  return items.some(it => it[h.prob] != null && Number.isFinite(Number(it[h.prob])));
}

/* ---------- data load ---------- */
async function load() {
  try {
    const res = await fetch(DATA_URL + "?v=" + Date.now());
    FULL = await res.json();
  } catch (e) {
    byId("top-picks-listing").innerHTML = `<div class="footnote" style="color:#b00">Data load error: ${e.message}</div>`;
    return;
  }
  splitItems();
  updateHeader();
  renderTopPicks();
  setupHorizonTabs("stocks");
  setupHorizonTabs("crypto");
  renderSectionListing("stocks", SECTION_DEFAULT_HORIZON.stocks);
  renderSectionListing("crypto", SECTION_DEFAULT_HORIZON.crypto);
  setupBacktestTriggers();
}

function splitItems() {
  const items = FULL.items || [];
  ITEMS_STOCKS = items.filter(i => !i.is_crypto);
  ITEMS_CRYPTO = items.filter(i => i.is_crypto);
}

function updateHeader() {
  const stat = byId("statUniverse");
  if (stat) stat.textContent = (FULL.items || []).length.toLocaleString();
  const asOf = byId("asOf");
  if (asOf && FULL.as_of) {
    const d = new Date(FULL.as_of);
    if (!isNaN(d)) asOf.textContent = d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
    else asOf.textContent = String(FULL.as_of).slice(0, 10);
  }
}

/* ---------- horizon tabs ---------- */
function setupHorizonTabs(section) {
  const items = section === "stocks" ? ITEMS_STOCKS : ITEMS_CRYPTO;
  const container = document.querySelector(`.horizon-tabs[data-section="${section}"]`);
  if (!container) return;
  container.innerHTML = "";
  for (const h of HORIZONS) {
    const btn = document.createElement("button");
    btn.className = "btn-lite";
    btn.textContent = h.id;
    btn.dataset.horizon = h.id;
    const has = horizonExists(items, h.id);
    if (!has) btn.classList.add("disabled");
    if (h.id === SECTION_DEFAULT_HORIZON[section]) btn.classList.add("active");
    btn.addEventListener("click", () => {
      if (btn.classList.contains("disabled")) return;
      container.querySelectorAll(".btn-lite").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderSectionListing(section, h.id);
      // Re-run backtest if already open
      const detId = `${section}-backtest-section`;
      const det = byId(detId);
      if (det && det.open) runSectionBacktest(section);
    });
    container.appendChild(btn);
  }
}

/* ---------- listing rendering ---------- */
function renderSectionListing(section, horizonId) {
  const items = section === "stocks" ? ITEMS_STOCKS : ITEMS_CRYPTO;
  const h = HORIZON_BY_ID[horizonId];
  const probLabel = byId(`${section}-prob-label`);
  if (probLabel) probLabel.textContent = `${h.id} Prob`;

  const sorted = items.slice().filter(it => it[h.prob] != null && Number.isFinite(Number(it[h.prob])))
    .sort((a, b) => Number(b[h.prob]) - Number(a[h.prob]));
  const container = byId(`${section}-listing`);
  container.innerHTML = "";
  if (!sorted.length) {
    container.innerHTML = `<div class="footnote">No ${section} have probabilities at the ${h.id} horizon yet. Try a different horizon.</div>`;
    return;
  }
  for (const it of sorted) {
    container.appendChild(buildRow(it, h));
  }
}

function buildRow(item, h, opts) {
  const div = document.createElement("div");
  div.className = "max-ticker-row";
  const prob = item[h.prob];
  const med = item[h.median];
  const down = h.downside ? item[h.downside] : null;
  const hzBadge = opts && opts.showHorizon ? `<span class="row-horizon">${opts.showHorizon}</span>` : "";
  div.innerHTML = `
    <div class="row-ticker">${item.ticker}${hzBadge}</div>
    <div class="row-cell">${fmtProb(prob)}</div>
    <div class="row-cell">${fmtPctSigned(med)}</div>
    <div class="row-cell">${fmtPctSigned(down)}</div>
    <div class="row-cell">${fmtNum(item.washout_today)}</div>
    <div class="row-cell">${fmtNum(item.quality)}</div>
    <div class="row-cell max-col-cases">${fmtNum(item.n_analogs)}</div>
  `;
  return div;
}

/* ---------- Top Picks: pick each item's best horizon ----------
   We pick the horizon where the setup's edge over the universe at that
   horizon is greatest. "Edge" = (item prob) − (universe median prob at that
   horizon). This naturally produces a mix (some 30D, some 5Y) because each
   horizon has its own baseline. We then rank picks by edge × sqrt(n_analogs)
   to prefer well-evidenced edges, and require positive expected return. */
let UNIVERSE_BASELINES = null;
function computeUniverseBaselines(items) {
  const out = {};
  for (const h of HORIZONS) {
    const vals = [];
    for (const it of items) {
      const p = it[h.prob];
      if (p != null && Number.isFinite(Number(p))) vals.push(Number(p));
    }
    if (!vals.length) { out[h.id] = null; continue; }
    vals.sort((a, b) => a - b);
    out[h.id] = vals[Math.floor(vals.length / 2)];
  }
  return out;
}

function bestHorizonFor(item) {
  if (!UNIVERSE_BASELINES) return null;
  let best = null;
  for (const h of HORIZONS) {
    const p = item[h.prob];
    const m = item[h.median];
    const base = UNIVERSE_BASELINES[h.id];
    if (p == null || !Number.isFinite(Number(p)) || base == null) continue;
    const prob = Number(p);
    const med = Number.isFinite(Number(m)) ? Number(m) : 0;
    const expected = 1 + (prob / 100) * med;
    if (expected <= 1) continue; // require positive expected return
    const edge = prob - base; // percentage points above universe median
    const n = Number(item.n_analogs || 0);
    const evidence = Math.sqrt(Math.min(n, 500)) / Math.sqrt(500);
    const score = edge * evidence;
    if (!best || score > best.score) {
      best = { horizon: h, score, prob, median: med, edge, expected };
    }
  }
  return best;
}

function renderTopPicks() {
  UNIVERSE_BASELINES = computeUniverseBaselines(FULL.items || []);
  const scored = (FULL.items || [])
    .map(it => ({ it, best: bestHorizonFor(it) }))
    .filter(x => x.best != null && x.best.score > 0)
    .sort((a, b) => b.best.score - a.best.score)
    .slice(0, 15);

  const container = byId("top-picks-listing");
  container.innerHTML = "";
  if (!scored.length) {
    container.innerHTML = `<div class="footnote">No ranked picks available.</div>`;
    return;
  }
  for (const { it, best } of scored) {
    container.appendChild(buildRow(it, best.horizon, { showHorizon: best.horizon.id }));
  }
}

/* ---------- Backtest engine (point-in-time ranking) ---------- */
function buildBT() {
  if (BT) return BT;
  const bt_series = FULL.bt_series || {};
  const priceLookup = {}; // ticker -> Map(date -> price)
  const scoreLookup = {}; // ticker -> Map(date -> final)
  // Build date spine as UNION of every ticker's dates — avoids being broken by
  // any one ticker that happens to have stale/truncated history.
  const dateSet = new Set();
  for (const s of Object.values(bt_series)) {
    for (const d of s.dates) dateSet.add(d);
  }
  const allDates = Array.from(dateSet).sort();
  for (const [tk, s] of Object.entries(bt_series)) {
    const pm = new Map(), fm = new Map();
    for (let i = 0; i < s.dates.length; i++) {
      pm.set(s.dates[i], Number(s.prices[i]));
      fm.set(s.dates[i], Number(s.final[i]));
    }
    priceLookup[tk] = pm;
    scoreLookup[tk] = fm;
  }
  const monthFirstIdx = [];
  let prevYM = "";
  for (let i = 0; i < allDates.length; i++) {
    const ym = allDates[i].substring(0, 7);
    if (ym !== prevYM) { monthFirstIdx.push(i); prevYM = ym; }
  }
  BT = { allDates, priceLookup, scoreLookup, monthFirstIdx };
  return BT;
}

/* Simulate DCA: pick up to topN tickers from `universe` ranked by point-in-time
   final score; hold each for holdDays trading days; sell at market. */
function simulateDCA(universe, topN, holdDays, opts) {
  const { allDates, priceLookup, scoreLookup, monthFirstIdx } = buildBT();
  const equity = new Float64Array(allDates.length);
  const positions = [];
  let cash = 0;
  let totalInvested = 0;

  // Precompute next-trading-day index for each month-first
  for (let m = 0; m < monthFirstIdx.length; m++) {
    const dIdx = monthFirstIdx[m];
    const date = allDates[dIdx];
    // rank universe by point-in-time score
    const ranked = [];
    for (const tk of universe) {
      const s = scoreLookup[tk]?.get(date);
      const p = priceLookup[tk]?.get(date);
      if (s != null && Number.isFinite(s) && s > 0 && p != null && Number.isFinite(p) && p > 0) {
        ranked.push({ tk, s, p });
      }
    }
    if (!ranked.length) continue;
    ranked.sort((a, b) => b.s - a.s);
    const picks = ranked.slice(0, Math.min(topN, ranked.length));
    const perAsset = DCA_MONTHLY / picks.length;
    totalInvested += DCA_MONTHLY;
    for (const { tk, p } of picks) {
      positions.push({ tk, shares: perAsset / p, cost: perAsset, buyIdx: dIdx, sellIdx: dIdx + holdDays, sold: false, sellPrice: 0 });
    }
  }

  // Walk day-by-day to compute equity
  for (let d = 0; d < allDates.length; d++) {
    const date = allDates[d];
    // sell expired
    for (const pos of positions) {
      if (!pos.sold && d >= pos.sellIdx) {
        const sp = priceLookup[pos.tk]?.get(date);
        if (sp && sp > 0) { cash += pos.shares * sp; pos.sold = true; pos.sellPrice = sp; }
      }
    }
    let open = 0;
    for (const pos of positions) {
      if (pos.sold || d < pos.buyIdx) continue;
      const px = priceLookup[pos.tk]?.get(date);
      if (px && Number.isFinite(px)) open += pos.shares * px;
    }
    equity[d] = cash + open;
  }

  return { equity, totalInvested, positions };
}

function simulateBenchmarkDCA(benchTickers, holdDays) {
  const { allDates, priceLookup, monthFirstIdx } = buildBT();
  // Pre-build a per-ticker forward-filled price array aligned to allDates so that
  // benchmarks with stale-ending data (e.g. SPY series that cut off early) still
  // produce a meaningful equity curve rather than collapsing to zero.
  const filled = {};
  for (const tk of benchTickers) {
    const pm = priceLookup[tk];
    if (!pm) continue;
    const arr = new Float64Array(allDates.length);
    let last = 0;
    for (let i = 0; i < allDates.length; i++) {
      const v = pm.get(allDates[i]);
      if (v != null && Number.isFinite(v) && v > 0) last = v;
      arr[i] = last; // 0 before first real price
    }
    filled[tk] = arr;
  }
  const equity = new Float64Array(allDates.length);
  const positions = [];
  let cash = 0, totalInvested = 0;
  for (const dIdx of monthFirstIdx) {
    const avail = benchTickers.filter(t => filled[t] && filled[t][dIdx] > 0);
    if (!avail.length) continue;
    const per = DCA_MONTHLY / avail.length;
    totalInvested += DCA_MONTHLY;
    for (const tk of avail) {
      const p = filled[tk][dIdx];
      positions.push({ tk, shares: per / p, cost: per, buyIdx: dIdx, sellIdx: dIdx + holdDays, sold: false, sellPrice: 0 });
    }
  }
  for (let d = 0; d < allDates.length; d++) {
    for (const pos of positions) {
      if (!pos.sold && d >= pos.sellIdx) {
        const sp = filled[pos.tk]?.[Math.min(d, allDates.length - 1)];
        if (sp && sp > 0) { cash += pos.shares * sp; pos.sold = true; pos.sellPrice = sp; }
      }
    }
    let open = 0;
    for (const pos of positions) {
      if (pos.sold || d < pos.buyIdx) continue;
      const px = filled[pos.tk]?.[d];
      if (px && Number.isFinite(px)) open += pos.shares * px;
    }
    equity[d] = cash + open;
  }
  return { equity, totalInvested, positions };
}

/* ---------- Metrics ---------- */
function computeMetrics(allDates, eq, totalInvested) {
  if (!eq || eq.length === 0) return null;
  const final = eq[eq.length - 1];
  // compute monthly returns
  const monthFirst = [];
  let prevYM = "";
  for (let i = 0; i < allDates.length; i++) {
    const ym = allDates[i].substring(0, 7);
    if (ym !== prevYM) { monthFirst.push(i); prevYM = ym; }
  }
  const monthlyRets = [];
  for (let m = 1; m < monthFirst.length; m++) {
    const prev = eq[monthFirst[m - 1]];
    const cur = eq[monthFirst[m]];
    if (prev > 0) monthlyRets.push(cur / prev - 1);
  }
  const avg = monthlyRets.length ? monthlyRets.reduce((a, b) => a + b, 0) / monthlyRets.length : 0;
  const std = monthlyRets.length ? Math.sqrt(monthlyRets.reduce((a, b) => a + (b - avg) ** 2, 0) / monthlyRets.length) : 0;
  const sharpe = std > 0 ? (avg / std) * Math.sqrt(12) : 0;
  // max drawdown on equity
  let peak = 0, maxDD = 0;
  for (let i = 0; i < eq.length; i++) {
    if (eq[i] > peak) peak = eq[i];
    if (peak > 0) {
      const dd = (peak - eq[i]) / peak;
      if (dd > maxDD) maxDD = dd;
    }
  }
  const totalReturn = totalInvested > 0 ? (final - totalInvested) / totalInvested : 0;
  // CAGR using the time span
  const yrs = allDates.length / 252;
  const cagr = (totalInvested > 0 && yrs > 0) ? Math.pow(final / totalInvested, 1 / yrs) - 1 : 0;
  return { final, totalInvested, totalReturn, cagr, maxDD, sharpe, nMonths: monthlyRets.length };
}

/* ---------- Backtest render ---------- */
function renderBacktest(bodyId, results, benchTickers, holdDays, horizonLabel, opts) {
  const body = byId(bodyId);
  body.innerHTML = "";
  const { allDates } = buildBT();

  // Chart
  const wrap = document.createElement("div");
  wrap.className = "bt-chart-wrap";
  const canvas = document.createElement("canvas");
  canvas.className = "bt-canvas";
  canvas.style.width = "100%";
  canvas.style.height = "280px";
  wrap.appendChild(canvas);
  const legend = document.createElement("div");
  legend.className = "bt-legend";
  const colors = { 1: "#064e2b", 5: "#1a6dd1", 10: "#d4820e", bench: "#999" };
  const labels = { 1: "Top 1", 5: "Top 5", 10: "Top 10", bench: benchTickers.length > 1 ? "50/50 " + benchTickers.join("+") : benchTickers[0] };
  for (const k of [1, 5, 10, "bench"]) {
    const s = document.createElement("span");
    s.className = "bt-legend-item";
    s.innerHTML = `<span class="bt-swatch" style="display:inline-block;width:10px;height:10px;background:${colors[k]};margin-right:4px"></span>${labels[k]}`;
    legend.appendChild(s);
  }
  wrap.appendChild(legend);
  body.appendChild(wrap);
  requestAnimationFrame(() => drawEquityChart(canvas, allDates, results, colors));

  // Metrics table
  const rows = [
    ["Total Invested", r => "$" + r.metrics.totalInvested.toLocaleString()],
    ["Final Value",    r => "$" + Math.round(r.metrics.final).toLocaleString()],
    ["Total Return",   r => fmtPctSigned(r.metrics.totalReturn)],
    ["CAGR",           r => fmtPctSigned(r.metrics.cagr)],
    ["Max Drawdown",   r => fmtPctSigned(-r.metrics.maxDD)],
    ["Sharpe",         r => r.metrics.sharpe.toFixed(2)],
  ];
  const tbl = document.createElement("table");
  tbl.className = "outcomes-table bt-table";
  let html = `<thead><tr><th></th><th>Top 1</th><th>Top 5</th><th>Top 10</th><th>${labels.bench}</th></tr></thead><tbody>`;
  for (const [lbl, fn] of rows) {
    html += `<tr><td>${lbl}</td><td>${fn(results[1])}</td><td>${fn(results[5])}</td><td>${fn(results[10])}</td><td>${fn(results.bench)}</td></tr>`;
  }
  html += "</tbody>";
  tbl.innerHTML = html;
  const scroll = document.createElement("div");
  scroll.className = "table-scroll";
  scroll.appendChild(tbl);
  body.appendChild(scroll);

  const info = document.createElement("div");
  info.className = "footnote";
  info.textContent = `${horizonLabel} hold. Ranking uses the scanner's point-in-time opportunity score on each DCA date — no look-ahead. Not financial advice; past performance does not predict future results.`;
  body.appendChild(info);
}

function drawEquityChart(canvas, dates, results, colors) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width = Math.floor(canvas.clientWidth * dpr);
  const h = canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx.clearRect(0, 0, w, h);
  const series = [
    { eq: results[1].sim.equity, color: colors[1] },
    { eq: results[5].sim.equity, color: colors[5] },
    { eq: results[10].sim.equity, color: colors[10] },
    { eq: results.bench.sim.equity, color: colors.bench },
  ];
  let maxY = 1;
  for (const s of series) for (let i = 0; i < s.eq.length; i++) if (s.eq[i] > maxY) maxY = s.eq[i];
  maxY *= 1.05;
  ctx.strokeStyle = "#eee"; ctx.lineWidth = 1 * dpr;
  ctx.beginPath(); ctx.moveTo(0, h - 1); ctx.lineTo(w, h - 1); ctx.stroke();
  ctx.font = `${10 * dpr}px system-ui,sans-serif`;
  ctx.fillStyle = "#888";
  for (let g = 0; g <= 4; g++) {
    const y = (h - 1) - (g / 4) * (h - 20 * dpr);
    const v = (g / 4) * maxY;
    ctx.fillText("$" + Math.round(v).toLocaleString(), 4 * dpr, y - 2 * dpr);
  }
  for (const s of series) {
    if (!s.eq.length) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.8 * dpr;
    ctx.beginPath();
    for (let i = 0; i < s.eq.length; i++) {
      const x = (i / (s.eq.length - 1)) * w;
      const y = (h - 1) - (s.eq[i] / maxY) * (h - 20 * dpr);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

/* ---------- Backtest orchestrators ---------- */
function runBacktestFor(section, holdDays, universe, benchTickers) {
  const bt = buildBT();
  if (!bt.allDates || !bt.allDates.length) return null;
  const results = {};
  for (const n of [1, 5, 10]) {
    const sim = simulateDCA(universe, n, holdDays);
    const metrics = computeMetrics(bt.allDates, sim.equity, sim.totalInvested);
    results[n] = { sim, metrics };
  }
  const benchSim = simulateBenchmarkDCA(benchTickers, holdDays);
  results.bench = { sim: benchSim, metrics: computeMetrics(bt.allDates, benchSim.equity, benchSim.totalInvested) };
  return results;
}

function setupBacktestTriggers() {
  // Stocks
  const stocksDet = byId("stocks-backtest-section");
  let stocksLoaded = false;
  stocksDet?.addEventListener("toggle", () => {
    if (stocksDet.open && !stocksLoaded) { stocksLoaded = true; runSectionBacktest("stocks"); }
  });
  // Crypto
  const cryptoDet = byId("crypto-backtest-section");
  let cryptoLoaded = false;
  cryptoDet?.addEventListener("toggle", () => {
    if (cryptoDet.open && !cryptoLoaded) { cryptoLoaded = true; runSectionBacktest("crypto"); }
  });
  // Top picks
  const topDet = byId("toppicks-backtest-section");
  let topLoaded = false;
  topDet?.addEventListener("toggle", () => {
    if (topDet.open && !topLoaded) { topLoaded = true; runTopBacktest(currentTopHoldDays()); }
  });
  document.querySelectorAll('.bt-hold-tabs[data-backtest="top"] .btn-lite').forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll('.bt-hold-tabs[data-backtest="top"] .btn-lite').forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      if (topLoaded) runTopBacktest(Number(btn.dataset.hold));
    });
  });
}

function currentHorizon(section) {
  const active = document.querySelector(`.horizon-tabs[data-section="${section}"] .btn-lite.active`);
  return active ? HORIZON_BY_ID[active.dataset.horizon] : HORIZON_BY_ID[SECTION_DEFAULT_HORIZON[section]];
}

function currentTopHoldDays() {
  const active = document.querySelector('.bt-hold-tabs[data-backtest="top"] .btn-lite.active');
  return active ? Number(active.dataset.hold) : 252;
}

function runSectionBacktest(section) {
  const h = currentHorizon(section);
  const items = section === "stocks" ? ITEMS_STOCKS : ITEMS_CRYPTO;
  const universe = items.map(i => i.ticker);
  const benchTickers = SECTION_BENCHMARK[section];
  const bodyId = `${section}-backtest-body`;
  byId(bodyId).innerHTML = `<div class="footnote">Running backtest&hellip;</div>`;
  setTimeout(() => {
    const results = runBacktestFor(section, h.days, universe, benchTickers);
    if (!results) { byId(bodyId).innerHTML = `<div class="footnote">No backtest data available.</div>`; return; }
    renderBacktest(bodyId, results, benchTickers, h.days, h.label);
    renderBestHorizonNote(section);
  }, 10);
}

function runTopBacktest(holdDays) {
  const universe = (FULL.items || []).map(i => i.ticker);
  const benchTickers = SECTION_BENCHMARK.top;
  const bodyId = "toppicks-backtest-body";
  byId(bodyId).innerHTML = `<div class="footnote">Running backtest&hellip;</div>`;
  setTimeout(() => {
    const results = runBacktestFor("top", holdDays, universe, benchTickers);
    if (!results) { byId(bodyId).innerHTML = `<div class="footnote">No backtest data available.</div>`; return; }
    const label = holdDays + " Trading Days";
    renderBacktest(bodyId, results, benchTickers, holdDays, label);
  }, 10);
}

/* ---------- "Best horizon for section" ---------- */
function renderBestHorizonNote(section) {
  const note = byId(`${section}-best-horizon-note`);
  if (!note) return;
  note.textContent = "Calculating best horizon for this section…";
  // Compute top-5 DCA Sharpe across all horizons — lightweight but not instant.
  const items = section === "stocks" ? ITEMS_STOCKS : ITEMS_CRYPTO;
  const universe = items.map(i => i.ticker);
  const { allDates } = buildBT();
  let best = null;
  setTimeout(() => {
    for (const h of HORIZONS) {
      // only evaluate horizons where at least 10 items have data (else noise)
      const covered = items.filter(i => i[h.prob] != null).length;
      if (covered < 10) continue;
      const sim = simulateDCA(universe, 5, h.days);
      const m = computeMetrics(allDates, sim.equity, sim.totalInvested);
      if (!m) continue;
      if (!best || m.sharpe > best.m.sharpe) best = { h, m };
    }
    if (!best) { note.textContent = ""; return; }
    note.textContent = `Best horizon for ${section}: ${best.h.id} (Top-5 Sharpe ${best.m.sharpe.toFixed(2)}, CAGR ${fmtPctSigned(best.m.cagr)}).`;
  }, 50);
}

/* ---------- go ---------- */
load();
