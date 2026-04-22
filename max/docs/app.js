/* Max Strategy v9-max — 10D/30D/60D/3M/6M/1Y/3Y/5Y horizons
   Two sections: Top Recommendations, Stocks, Crypto. Each with its own
   backtest. Ranking in the backtest uses point-in-time final score (no
   look-ahead); the current listing uses the latest scan's analog probabilities. */

const DATA_URL = "./data/full.json";

const HORIZONS = [
  { id: "10D", prob: "prob_10d", median: "median_10d", downside: "downside_10d", days: 10,   label: "10 Days" },
  { id: "30D", prob: "prob_30d", median: "median_30d", downside: "downside_30d", days: 30,   label: "30 Days" },
  { id: "60D", prob: "prob_60d", median: "median_60d", downside: "downside_60d", days: 60,   label: "60 Days" },
  { id: "3M",  prob: "prob_3m",  median: "median_3m",  downside: "downside_3m",  days: 63,   label: "3 Months" },
  { id: "6M",  prob: "prob_6m",  median: "median_6m",  downside: "downside_6m",  days: 126,  label: "6 Months" },
  { id: "1Y",  prob: "prob_1y",  median: "median_1y",  downside: "downside_1y",  days: 252,  label: "1 Year" },
  { id: "3Y",  prob: "prob_3y",  median: "median_3y",  downside: "downside_3y",  days: 756,  label: "3 Years" },
  { id: "5Y",  prob: "prob_5y",  median: "median_5y",  downside: "downside_5y",  days: 1260, label: "5 Years" },
];
const HORIZON_BY_ID = Object.fromEntries(HORIZONS.map(h => [h.id, h]));

const SECTION_DEFAULT_HORIZON = { stocks: "1Y", crypto: "60D" };
const SECTION_BENCHMARK = { stocks: ["SPY"], crypto: ["BTC-USD"], topstocks: ["SPY"], topcrypto: ["BTC-USD"] };
const DCA_MONTHLY = 1000;

/* Backtest periods. `months` null = use the default firstValidMonthIdx start.
   Otherwise the window starts `months` calendar-months before the end of the
   spine. `half` picks the first or second half of the usable range. */
const PERIODS = [
  { id: "full", label: "Full",        months: null, half: null },
  { id: "5y",   label: "Last 5Y",     months: 60,   half: null },
  { id: "3y",   label: "Last 3Y",     months: 36,   half: null },
  { id: "1y",   label: "Last 1Y",     months: 12,   half: null },
  { id: "6m",   label: "Last 6M",     months: 6,    half: null },
  { id: "h1",   label: "First Half",  months: null, half: 1 },
  { id: "h2",   label: "Second Half", months: null, half: 2 },
];
const PERIOD_BY_KEY = Object.fromEntries(PERIODS.map(p => [p.id, p]));
const PERIOD_STATE = {}; // key (section or top-name) -> selected period id

let FULL = null;
let ITEMS_STOCKS = [];
let ITEMS_CRYPTO = [];
let BT = null; // built once from bt_series

function sectionItems(section) {
  if (section === "stocks" || section === "topstocks") return ITEMS_STOCKS;
  if (section === "crypto" || section === "topcrypto") return ITEMS_CRYPTO;
  return [];
}

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
function fmtPriceVal(v) {
  if (v == null || !Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs >= 1000) return "$" + v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 10)   return "$" + v.toFixed(2);
  if (abs >= 1)    return "$" + v.toFixed(3);
  if (abs >= 0.01) return "$" + v.toFixed(4);
  return "$" + v.toPrecision(3);
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
  // Defer heavy computations so the listings paint immediately.
  setTimeout(() => {
    renderThisMonthsPicks();
    renderConcentration();
  }, 30);
}

function splitItems() {
  const items = FULL.items || [];
  ITEMS_CRYPTO = items.filter(i => i.is_crypto);
  ITEMS_STOCKS = items.filter(i => !i.is_crypto);
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
  const items = sectionItems(section);
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
  const items = sectionItems(section);
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

function buildRow(item, h) {
  const div = document.createElement("div");
  div.className = "max-ticker-row";
  const prob = item[h.prob];
  const med = item[h.median];
  const down = h.downside ? item[h.downside] : null;
  div.innerHTML = `
    <div class="row-ticker">${item.ticker}</div>
    <div class="row-cell">${fmtPriceVal(item.last_price)}</div>
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
   We pick the horizon where the setup's edge over its asset class baseline
   at that horizon is greatest. "Edge" = (item prob) − (within-class median
   prob at that horizon). Stocks are baselined against stocks, crypto against
   crypto — so stocks aren't held to crypto's volatility-warped distribution
   and vice versa. We then rank picks by edge × sqrt(n_analogs) to prefer
   well-evidenced edges, and require positive expected return. */
const BASELINES = { stocks: null, crypto: null };
/* ticker -> preferred hold in trading days (built from today's bestHorizonFor
   across every item in a section). Used by the Per-Pick backtest so each
   position is sold at the horizon the scanner recommends for that ticker,
   mixing 30D, 60D, 1Y, 5Y etc. in a single DCA run. */
const TOP_HORIZONS = { topstocks: {}, topcrypto: {} };
function computeBaselines(items) {
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

function bestHorizonFor(item, baselines) {
  if (!baselines) return null;
  let best = null;
  for (const h of HORIZONS) {
    const p = item[h.prob];
    const m = item[h.median];
    const base = baselines[h.id];
    if (p == null || !Number.isFinite(Number(p)) || base == null) continue;
    const prob = Number(p);
    const med = Number.isFinite(Number(m)) ? Number(m) : 0;
    const expected = 1 + (prob / 100) * med;
    if (expected <= 1) continue;
    const edge = prob - base;
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
  BASELINES.stocks = computeBaselines(ITEMS_STOCKS);
  BASELINES.crypto = computeBaselines(ITEMS_CRYPTO);
  renderTopPicksFor("stocks");
  renderTopPicksFor("crypto");
  TOP_HORIZONS.topstocks = buildHorizonByTicker(ITEMS_STOCKS, BASELINES.stocks);
  TOP_HORIZONS.topcrypto = buildHorizonByTicker(ITEMS_CRYPTO, BASELINES.crypto);
}

function buildHorizonByTicker(items, baselines) {
  const map = {};
  for (const it of items) {
    const best = bestHorizonFor(it, baselines);
    if (best) map[it.ticker] = best.horizon.days;
  }
  return map;
}

function renderTopPicksFor(section) {
  const items = sectionItems(section);
  const baselines = BASELINES[section];
  // Stocks rank by the CAP5+SMA12M point-in-time opportunity score — the
  // trailing 12-month mean of the conviction series (step39 winner). If the
  // smoothed value is missing (new tickers without 12M of history), fall
  // back to the raw conviction, then final_score. Crypto keeps best-horizon
  // edge ranking since its backtest uses the per-pick engine.
  const rankByConviction = section === "stocks";
  const convScore = (it) => {
    const s = it.conviction_smooth12m;
    if (s != null && Number.isFinite(Number(s))) return Number(s);
    const c = it.conviction;
    if (c != null && Number.isFinite(Number(c))) return Number(c);
    const f = it.final_score;
    if (f != null && Number.isFinite(Number(f))) return Number(f);
    return null;
  };
  let scored;
  if (rankByConviction) {
    scored = items
      .map(it => ({ it, best: bestHorizonFor(it, baselines), score: convScore(it) }))
      .filter(x => x.best != null && x.score != null)
      .sort((a, b) => b.score - a.score)
      .slice(0, 15);
  } else {
    scored = items
      .map(it => ({ it, best: bestHorizonFor(it, baselines) }))
      .filter(x => x.best != null && x.best.score > 0)
      .sort((a, b) => b.best.score - a.best.score)
      .slice(0, 15);
  }
  const container = byId(`top-${section}-listing`);
  if (!container) return;
  container.innerHTML = "";
  if (!scored.length) {
    container.innerHTML = `<div class="footnote">No ranked ${section} picks available.</div>`;
    return;
  }
  for (const { it, best } of scored) {
    container.appendChild(buildRow(it, best.horizon));
  }
}

/* ---------- Backtest engine (point-in-time ranking) ----------

The Max CAP5 strategy ranks candidates by the trailing 12-month mean of the
conviction score (step39 research winner — SMA 12M, adopted as production
default). The scanner publishes both the raw conviction series (`final`)
and the smoothed series (`final_smooth12m`) in `bt_series[ticker]`. If the
smoothed series is missing (older JSON payloads), the raw series is used
so the webapp stays compatible with pre-v10 scans. */
function buildBT() {
  if (BT) return BT;
  const bt_series = FULL.bt_series || {};
  const priceLookup = {}; // ticker -> Map(date -> price)
  const scoreLookup = {}; // ticker -> Map(date -> final smoothed)
  const scoreRawLookup = {}; // ticker -> Map(date -> final raw)
  // Find the freshest latest-date across all tickers. Series that end more than
  // a year before that are "stale" and their dates get excluded from the spine —
  // otherwise a single abandoned ticker (e.g. SPY bt_series ending 2020)
  // creates a huge empty left-half in the chart and a fake cliff when its
  // positions become un-markable mid-simulation.
  let maxEndDate = "";
  for (const s of Object.values(bt_series)) {
    if (s.dates && s.dates.length) {
      const last = s.dates[s.dates.length - 1];
      if (last > maxEndDate) maxEndDate = last;
    }
  }
  const staleCutoff = shiftDateDays(maxEndDate, -365);
  const dateSet = new Set();
  for (const [tk, s] of Object.entries(bt_series)) {
    if (!s.dates || !s.dates.length) continue;
    const last = s.dates[s.dates.length - 1];
    // Always keep prices for lookups (benchmark forward-fill, etc) but only
    // contribute to the spine if the series is fresh.
    const pm = new Map(), fm = new Map(), fmRaw = new Map();
    const smooth = (s.final_smooth12m && s.final_smooth12m.length === s.dates.length)
      ? s.final_smooth12m
      : s.final;
    for (let i = 0; i < s.dates.length; i++) {
      pm.set(s.dates[i], Number(s.prices[i]));
      fm.set(s.dates[i], Number(smooth[i]));
      fmRaw.set(s.dates[i], Number(s.final[i]));
    }
    priceLookup[tk] = pm;
    scoreLookup[tk] = fm;
    scoreRawLookup[tk] = fmRaw;
    if (last >= staleCutoff) {
      for (const d of s.dates) dateSet.add(d);
    }
  }
  const allDates = Array.from(dateSet).sort();
  const monthFirstIdx = [];
  let prevYM = "";
  for (let i = 0; i < allDates.length; i++) {
    const ym = allDates[i].substring(0, 7);
    if (ym !== prevYM) { monthFirstIdx.push(i); prevYM = ym; }
  }
  BT = { allDates, priceLookup, scoreLookup, scoreRawLookup, monthFirstIdx };
  return BT;
}

function shiftDateDays(isoDate, days) {
  if (!isoDate) return "";
  const d = new Date(isoDate + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().substring(0, 10);
}

/* Find the first month-first index where at least `minTickers` entries in
   `universe` have a valid score+price. Used so both portfolio and benchmark
   DCA from the same start date instead of the benchmark getting a 5-year
   head start because SPY has earlier history than stocks. */
function firstValidMonthIdx(universe, minTickers) {
  const { allDates, priceLookup, scoreLookup, monthFirstIdx } = buildBT();
  for (let m = 0; m < monthFirstIdx.length; m++) {
    const date = allDates[monthFirstIdx[m]];
    let n = 0;
    for (const tk of universe) {
      const s = scoreLookup[tk]?.get(date);
      const p = priceLookup[tk]?.get(date);
      if (s != null && Number.isFinite(s) && s > 0 && p != null && Number.isFinite(p) && p > 0) {
        n++;
        if (n >= minTickers) return m;
      }
    }
  }
  return 0;
}

/* Resolve the user-selected period to {startMonthIdx, endMonthIdx}. End is
   exclusive; null means "to the end of the spine". */
function periodWindow(periodId, firstValid) {
  const { monthFirstIdx } = buildBT();
  const total = monthFirstIdx.length;
  const p = PERIOD_BY_KEY[periodId] || PERIOD_BY_KEY.full;
  if (p.half === 1) {
    const mid = firstValid + Math.floor((total - firstValid) / 2);
    return { startMonthIdx: firstValid, endMonthIdx: mid };
  }
  if (p.half === 2) {
    const mid = firstValid + Math.floor((total - firstValid) / 2);
    return { startMonthIdx: mid, endMonthIdx: total };
  }
  if (p.months == null) return { startMonthIdx: firstValid, endMonthIdx: total };
  const s = Math.max(firstValid, total - p.months);
  return { startMonthIdx: s, endMonthIdx: total };
}

function injectPeriodBar(bodyId, stateKey, onChange) {
  const body = byId(bodyId);
  if (!body) return;
  const parent = body.parentElement;
  let row = parent.querySelector(".bt-period-row");
  if (row) return;
  row = document.createElement("div");
  row.className = "bt-period-row";
  const cur = PERIOD_STATE[stateKey] || "full";
  row.innerHTML = `<span class="bt-period-label">Period</span>` +
    PERIODS.map(p => `<button type="button" class="btn-lite${p.id === cur ? " active" : ""}" data-period="${p.id}">${p.label}</button>`).join("");
  parent.insertBefore(row, body);
  row.addEventListener("click", (ev) => {
    const btn = ev.target.closest(".btn-lite");
    if (!btn) return;
    const pid = btn.dataset.period;
    if (PERIOD_STATE[stateKey] === pid) return;
    PERIOD_STATE[stateKey] = pid;
    for (const b of row.querySelectorAll(".btn-lite")) {
      b.classList.toggle("active", b.dataset.period === pid);
    }
    onChange();
  });
}

/* Simulate DCA: pick up to topN tickers from `universe` ranked by point-in-time
   final score; hold each for holdDays trading days; sell at market. */
function simulateDCA(universe, topN, holdDays, opts) {
  const { allDates, priceLookup, scoreLookup, monthFirstIdx } = buildBT();
  const equity = new Float64Array(allDates.length);
  const positions = [];
  let cash = 0;
  let totalInvested = 0;
  const startMonthIdx = (opts && opts.startMonthIdx != null) ? opts.startMonthIdx : 0;
  const endMonthIdx = (opts && opts.endMonthIdx != null) ? opts.endMonthIdx : monthFirstIdx.length;

  // Precompute next-trading-day index for each month-first
  for (let m = startMonthIdx; m < endMonthIdx; m++) {
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

function simulateBenchmarkDCA(benchTickers, holdDays, opts) {
  const { allDates, priceLookup, monthFirstIdx } = buildBT();
  const startMonthIdx = (opts && opts.startMonthIdx != null) ? opts.startMonthIdx : 0;
  // Pre-build a per-ticker forward-filled price array aligned to allDates so that
  // benchmarks with stale-ending data (e.g. SPY series that cut off early) still
  // produce a meaningful equity curve rather than collapsing to zero.
  // Seed `last` from the benchmark's most recent known price BEFORE the spine
  // starts, so a benchmark that ended before the fresh-spine window still
  // carries a valid starting price forward.
  const filled = {};
  for (const tk of benchTickers) {
    const pm = priceLookup[tk];
    if (!pm) continue;
    const spineStart = allDates[0] || "";
    let last = 0;
    for (const [d, v] of pm.entries()) {
      if (d <= spineStart && Number.isFinite(v) && v > 0 && d > "") last = v;
    }
    const arr = new Float64Array(allDates.length);
    for (let i = 0; i < allDates.length; i++) {
      const v = pm.get(allDates[i]);
      if (v != null && Number.isFinite(v) && v > 0) last = v;
      arr[i] = last;
    }
    filled[tk] = arr;
  }
  const equity = new Float64Array(allDates.length);
  const positions = [];
  let cash = 0, totalInvested = 0;
  const endMonthIdx = (opts && opts.endMonthIdx != null) ? opts.endMonthIdx : monthFirstIdx.length;
  for (let m = startMonthIdx; m < endMonthIdx; m++) {
    const dIdx = monthFirstIdx[m];
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

/* Max strategy DCA — point-in-time ranking, rank-weighted sizing, hold-forever,
   next-day-open entry, AND a 5% per-ticker concentration cap (Step 23 finalist).
   The cap drops any pick whose cumulative cost basis would exceed 5% of total
   invested capital, then backfills from the next-ranked candidate. On the 20Y
   extended spine it strictly beats the uncapped variant: +17.41% vs +17.12%
   CAGR (+9.55pp vs +9.26pp excess over SPY), lower MaxDD (46.15% vs 48.92%),
   higher Sharpe (1.34 vs 1.32), higher Calmar (0.38 vs 0.35), and — crucially
   — turns the recent-1Y window from +2.58pp to +8.00pp excess by stopping
   repeat-buying of value traps (ADBE, CRM, NOW) once they've accumulated.
   Jackknife: 0/96 drops go negative. Bootstrap: 0/200 random 50-ticker rosters
   go negative (median +8.17pp). See max/research/step20-23 for the sweep. */
const MAX_TICKER_FRAC = 0.05;
function simulateDCAMax(universe, topN, opts) {
  const { allDates, priceLookup, scoreLookup, monthFirstIdx } = buildBT();
  const equity = new Float64Array(allDates.length);
  const positions = [];
  let cash = 0, totalInvested = 0;
  const startMonthIdx = (opts && opts.startMonthIdx != null) ? opts.startMonthIdx : 0;
  const endMonthIdx = (opts && opts.endMonthIdx != null) ? opts.endMonthIdx : monthFirstIdx.length;
  const entryDelay = (opts && opts.entryDelay != null) ? opts.entryDelay : 1;
  const maxTickerFrac = (opts && opts.maxTickerFrac != null) ? opts.maxTickerFrac : MAX_TICKER_FRAC;
  const tkCost = new Map();
  for (let m = startMonthIdx; m < endMonthIdx; m++) {
    const dIdx = monthFirstIdx[m];
    const date = allDates[dIdx];
    const ranked = [];
    for (const tk of universe) {
      const s = scoreLookup[tk]?.get(date);
      const p = priceLookup[tk]?.get(date);
      if (s != null && Number.isFinite(s) && s > 0 && p != null && Number.isFinite(p) && p > 0) {
        ranked.push({ tk, s });
      }
    }
    if (!ranked.length) continue;
    ranked.sort((a, b) => b.s - a.s);
    // Apply per-ticker cap: drop candidates whose cumulative basis already >=
    // cap dollars, then take the top N of what's left. Cap is against the
    // post-this-month invested total to keep the fraction stable over time.
    const capDollars = maxTickerFrac > 0
      ? (totalInvested + DCA_MONTHLY) * maxTickerFrac
      : Infinity;
    const eligible = maxTickerFrac > 0
      ? ranked.filter(r => (tkCost.get(r.tk) || 0) < capDollars)
      : ranked;
    const picks = eligible.slice(0, Math.min(topN, eligible.length));
    // Fill at the next trading day's close (proxy for next-day open: live scan
    // runs after market close, so the earliest realistic execution is +1 bar).
    const entryIdx = dIdx + entryDelay;
    if (entryIdx >= allDates.length) continue;
    const entryDate = allDates[entryIdx];
    const adj = [];
    for (const { tk, s } of picks) {
      const px = priceLookup[tk]?.get(entryDate);
      if (px != null && Number.isFinite(px) && px > 0) adj.push({ tk, s, px });
    }
    if (!adj.length) continue;
    // Rank weights: 1/1, 1/2, 1/3, … then normalize.
    const raw = adj.map((_, i) => 1 / (i + 1));
    const sumRaw = raw.reduce((a, b) => a + b, 0);
    const weights = raw.map(r => r / sumRaw);
    totalInvested += DCA_MONTHLY;
    for (let i = 0; i < adj.length; i++) {
      const { tk, px } = adj[i];
      const alloc = DCA_MONTHLY * weights[i];
      tkCost.set(tk, (tkCost.get(tk) || 0) + alloc);
      // Hold-forever: sellIdx beyond the spine, so positions never schedule a sale.
      positions.push({ tk, shares: alloc / px, cost: alloc, buyIdx: entryIdx,
                       sellIdx: allDates.length + 1, sold: false, sellPrice: 0,
                       weight: weights[i] });
    }
  }
  for (let d = 0; d < allDates.length; d++) {
    const date = allDates[d];
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

/* Benchmark for the Max strategy: SPY DCA, hold-forever, same next-day-open
   entry. Capital schedule matches the strategy exactly so the curves are
   directly comparable. */
function simulateBenchmarkDCAMax(benchTickers, opts) {
  const { allDates, priceLookup, monthFirstIdx } = buildBT();
  const startMonthIdx = (opts && opts.startMonthIdx != null) ? opts.startMonthIdx : 0;
  const endMonthIdx = (opts && opts.endMonthIdx != null) ? opts.endMonthIdx : monthFirstIdx.length;
  const entryDelay = (opts && opts.entryDelay != null) ? opts.entryDelay : 1;
  const filled = {};
  for (const tk of benchTickers) {
    const pm = priceLookup[tk];
    if (!pm) continue;
    const spineStart = allDates[0] || "";
    let last = 0;
    for (const [d, v] of pm.entries()) {
      if (d <= spineStart && Number.isFinite(v) && v > 0 && d > "") last = v;
    }
    const arr = new Float64Array(allDates.length);
    for (let i = 0; i < allDates.length; i++) {
      const v = pm.get(allDates[i]);
      if (v != null && Number.isFinite(v) && v > 0) last = v;
      arr[i] = last;
    }
    filled[tk] = arr;
  }
  const equity = new Float64Array(allDates.length);
  const positions = [];
  let cash = 0, totalInvested = 0;
  for (let m = startMonthIdx; m < endMonthIdx; m++) {
    const dIdx = monthFirstIdx[m];
    const entryIdx = dIdx + entryDelay;
    if (entryIdx >= allDates.length) break;
    const avail = benchTickers.filter(t => filled[t] && filled[t][entryIdx] > 0);
    if (!avail.length) continue;
    const per = DCA_MONTHLY / avail.length;
    totalInvested += DCA_MONTHLY;
    for (const tk of avail) {
      const p = filled[tk][entryIdx];
      positions.push({ tk, shares: per / p, cost: per, buyIdx: entryIdx,
                       sellIdx: allDates.length + 1, sold: false, sellPrice: 0 });
    }
  }
  for (let d = 0; d < allDates.length; d++) {
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

/* Per-pick DCA: rank by point-in-time final score, but hold each pick for the
   trading-day horizon chosen by today's bestHorizonFor() for that ticker
   (30D, 60D, 1Y, 3Y, …). Tickers that didn't produce a best-horizon today
   fall back to `opts.fallbackHold`. Returns the per-month horizon schedule so
   the benchmark can mirror it. NOTE: this contains look-ahead because the
   per-ticker hold uses today's probabilities. Kept for crypto path only;
   stocks use simulateDCAMax above. */
function simulateDCAPerPick(universe, topN, horizonByTicker, opts) {
  const { allDates, priceLookup, scoreLookup, monthFirstIdx } = buildBT();
  const equity = new Float64Array(allDates.length);
  const positions = [];
  const pickHorizonsByMonth = [];
  let cash = 0, totalInvested = 0;
  const startMonthIdx = (opts && opts.startMonthIdx != null) ? opts.startMonthIdx : 0;
  const endMonthIdx = (opts && opts.endMonthIdx != null) ? opts.endMonthIdx : monthFirstIdx.length;
  const fallback = (opts && opts.fallbackHold != null) ? opts.fallbackHold : 252;

  for (let m = startMonthIdx; m < endMonthIdx; m++) {
    const dIdx = monthFirstIdx[m];
    const date = allDates[dIdx];
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
    const monthHorizons = [];
    for (const { tk, p } of picks) {
      const hold = horizonByTicker[tk] || fallback;
      monthHorizons.push(hold);
      positions.push({ tk, shares: perAsset / p, cost: perAsset, buyIdx: dIdx, sellIdx: dIdx + hold, sold: false, sellPrice: 0 });
    }
    pickHorizonsByMonth.push({ dIdx, horizons: monthHorizons });
  }

  for (let d = 0; d < allDates.length; d++) {
    const date = allDates[d];
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

  return { equity, totalInvested, positions, pickHorizonsByMonth };
}

/* Benchmark DCA that mirrors the portfolio's per-pick hold schedule: for every
   DCA month, open N benchmark positions (N = portfolio picks that month) with
   the exact same per-position hold lengths. Keeps capital timing identical
   so the comparison isn't distorted by different hold-profile cash drag. */
function simulateBenchmarkDCAPerPick(benchTickers, pickHorizonsByMonth, opts) {
  const { allDates, priceLookup } = buildBT();
  const filled = {};
  for (const tk of benchTickers) {
    const pm = priceLookup[tk];
    if (!pm) continue;
    const spineStart = allDates[0] || "";
    let last = 0;
    for (const [d, v] of pm.entries()) {
      if (d <= spineStart && Number.isFinite(v) && v > 0 && d > "") last = v;
    }
    const arr = new Float64Array(allDates.length);
    for (let i = 0; i < allDates.length; i++) {
      const v = pm.get(allDates[i]);
      if (v != null && Number.isFinite(v) && v > 0) last = v;
      arr[i] = last;
    }
    filled[tk] = arr;
  }
  const equity = new Float64Array(allDates.length);
  const positions = [];
  let cash = 0, totalInvested = 0;
  for (const { dIdx, horizons } of pickHorizonsByMonth) {
    const avail = benchTickers.filter(t => filled[t] && filled[t][dIdx] > 0);
    if (!avail.length || !horizons.length) continue;
    const perSlot = DCA_MONTHLY / horizons.length;
    const perLeg = perSlot / avail.length;
    totalInvested += DCA_MONTHLY;
    for (const hold of horizons) {
      for (const tk of avail) {
        const p = filled[tk][dIdx];
        positions.push({ tk, shares: perLeg / p, cost: perLeg, buyIdx: dIdx, sellIdx: dIdx + hold, sold: false, sellPrice: 0 });
      }
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
/* opts.fromIdx, opts.toIdx bound the measurement window on the day axis.
   Defaults to the full spine. totalInvested should already reflect only the
   window (the simulators take endMonthIdx for that). */
function computeMetrics(allDates, eq, totalInvested, opts) {
  if (!eq || eq.length === 0) return null;
  const from = (opts && opts.fromIdx != null) ? Math.max(0, opts.fromIdx) : 0;
  const to = (opts && opts.toIdx != null) ? Math.min(eq.length, opts.toIdx) : eq.length;
  if (to <= from) return null;
  const final = eq[to - 1];
  // compute monthly returns within window
  const monthFirst = [];
  let prevYM = "";
  for (let i = from; i < to; i++) {
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
  // max drawdown on equity within window
  let peak = 0, maxDD = 0;
  for (let i = from; i < to; i++) {
    if (eq[i] > peak) peak = eq[i];
    if (peak > 0) {
      const dd = (peak - eq[i]) / peak;
      if (dd > maxDD) maxDD = dd;
    }
  }
  const totalReturn = totalInvested > 0 ? (final - totalInvested) / totalInvested : 0;
  // CAGR over the window
  const yrs = (to - from) / 252;
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
  requestAnimationFrame(() => drawEquityChart(canvas, allDates, results, colors, results.window));

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

  // Last 10 completed transactions — show Top 1 (concentrated) and Top 5
  // (diversified) so users can see both the outlier-driven and the
  // smoothed-out position history.
  renderRecentTransactions(body, results[1].sim, allDates, "Last 10 Completed Transactions — Top 1 Portfolio");
  renderRecentTransactions(body, results[5].sim, allDates, "Last 10 Completed Transactions — Top 5 Portfolio");

  const info = document.createElement("div");
  info.className = "footnote";
  info.textContent = (holdDays == null)
    ? `${horizonLabel}. DCA buys $1,000 on the first trading day of each calendar month, ranks the universe by the scanner's point-in-time opportunity score, and allocates that month's cash to the top-N picks with rank weights (1/1, 1/2, 1/3, … normalized). A 5% per-ticker concentration cap drops any candidate whose cumulative cost basis has already reached 5% of total invested and backfills from the next-ranked name — this prevents repeat-loading into value traps. Fills at the close of the next trading day (the live scan runs after market close). Positions are held forever — no scheduled sell, no look-ahead. Not financial advice; past performance does not predict future results.`
    : `${horizonLabel} hold. DCA buys $1,000 on the first trading day of each calendar month, splitting the cash equally across that month's top-N ranked picks. Each position is sold at the close of its hold horizon. Ranking uses the scanner's point-in-time opportunity score on each DCA date — no look-ahead. Not financial advice; past performance does not predict future results.`;
  body.appendChild(info);
}

function renderRecentTransactions(body, sim, allDates, heading) {
  const closed = [];
  for (const p of sim.positions || []) {
    if (!p.sold || !(p.sellPrice > 0)) continue;
    if (p.buyIdx < 0 || p.buyIdx >= allDates.length) continue;
    const sellIdxClamped = Math.min(p.sellIdx, allDates.length - 1);
    const buyPrice = p.shares > 0 ? p.cost / p.shares : 0;
    const proceeds = p.shares * p.sellPrice;
    closed.push({
      ticker: p.tk,
      buyDate: allDates[p.buyIdx],
      sellDate: allDates[sellIdxClamped],
      holdDays: p.sellIdx - p.buyIdx,
      buyPrice, sellPrice: p.sellPrice,
      cost: p.cost, proceeds,
      returnPct: buyPrice > 0 ? (p.sellPrice - buyPrice) / buyPrice : 0,
    });
  }
  if (!closed.length) return;
  closed.sort((a, b) => b.sellDate.localeCompare(a.sellDate));
  const recent = closed.slice(0, 10);

  const hdr = document.createElement("div");
  hdr.className = "section-title";
  hdr.style.cssText = "margin-top: 20px; font-size: 12px; letter-spacing: .08em;";
  hdr.textContent = heading || "Last 10 Completed Transactions";
  body.appendChild(hdr);

  const tbl = document.createElement("table");
  tbl.className = "outcomes-table bt-table";
  let html = `<thead><tr>
    <th>Ticker</th><th>Hold</th>
    <th>Buy Date</th><th>Buy $</th>
    <th>Sell Date</th><th>Sell $</th>
    <th>Return</th><th>Invested</th><th>Proceeds</th>
  </tr></thead><tbody>`;
  const fmtPrice = v => v >= 100 ? v.toFixed(2) : v >= 1 ? v.toFixed(3) : v.toPrecision(3);
  for (const t of recent) {
    html += `<tr>
      <td>${t.ticker}</td>
      <td>${t.holdDays}d</td>
      <td>${t.buyDate}</td>
      <td>$${fmtPrice(t.buyPrice)}</td>
      <td>${t.sellDate}</td>
      <td>$${fmtPrice(t.sellPrice)}</td>
      <td style="color:${t.returnPct >= 0 ? '#064e2b' : '#b00020'}">${fmtPctSigned(t.returnPct)}</td>
      <td>$${Math.round(t.cost).toLocaleString()}</td>
      <td>$${Math.round(t.proceeds).toLocaleString()}</td>
    </tr>`;
  }
  html += "</tbody>";
  tbl.innerHTML = html;
  const scroll = document.createElement("div");
  scroll.className = "table-scroll";
  scroll.appendChild(tbl);
  body.appendChild(scroll);
}

function drawEquityChart(canvas, dates, results, colors, window_) {
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
  // Crop x-axis to the selected window; within that, trim the left edge to the
  // first index with non-zero equity so the chart doesn't open with flat zero.
  const len = dates.length;
  const winFrom = (window_ && window_.fromIdx != null) ? Math.max(0, window_.fromIdx) : 0;
  const winTo = (window_ && window_.toIdx != null) ? Math.min(len, window_.toIdx) : len;
  let startIdx = winFrom;
  outer: for (let i = winFrom; i < winTo; i++) {
    for (const s of series) if (s.eq[i] > 0) { startIdx = i; break outer; }
  }
  let maxY = 1;
  for (const s of series) for (let i = startIdx; i < Math.min(s.eq.length, winTo); i++) if (s.eq[i] > maxY) maxY = s.eq[i];
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
  const endIdx = Math.max(startIdx, winTo - 1);
  const startLbl = dates[startIdx] || "";
  const endLbl = dates[endIdx] || "";
  ctx.textAlign = "left";
  ctx.fillText(startLbl, 4 * dpr, h - 4 * dpr);
  ctx.textAlign = "right";
  ctx.fillText(endLbl, w - 4 * dpr, h - 4 * dpr);
  ctx.textAlign = "left";
  const span = Math.max(1, endIdx - startIdx);
  for (const s of series) {
    if (!s.eq.length) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.8 * dpr;
    ctx.beginPath();
    const stop = Math.min(s.eq.length, endIdx + 1);
    for (let i = startIdx; i < stop; i++) {
      const x = ((i - startIdx) / span) * w;
      const y = (h - 1) - (s.eq[i] / maxY) * (h - 20 * dpr);
      if (i === startIdx) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

/* ---------- Backtest orchestrators ---------- */
function runBacktestFor(section, holdDays, universe, benchTickers, periodId) {
  const bt = buildBT();
  if (!bt.allDates || !bt.allDates.length) return null;
  // Both portfolio and benchmark start DCA from the same month — the first
  // month the portfolio universe has enough tickers with real data. Otherwise
  // a benchmark with deeper history (e.g. SPY 2014+) gets a years-long head
  // start vs. stocks whose bt_series only begins in 2021.
  const firstValid = firstValidMonthIdx(universe, 3);
  const { startMonthIdx, endMonthIdx } = periodWindow(periodId, firstValid);
  const fromIdx = bt.monthFirstIdx[startMonthIdx] || 0;
  const toIdx = endMonthIdx < bt.monthFirstIdx.length
    ? bt.monthFirstIdx[endMonthIdx]
    : bt.allDates.length;
  const results = {};
  for (const n of [1, 5, 10]) {
    const sim = simulateDCA(universe, n, holdDays, { startMonthIdx, endMonthIdx });
    const metrics = computeMetrics(bt.allDates, sim.equity, sim.totalInvested, { fromIdx, toIdx });
    results[n] = { sim, metrics };
  }
  const benchSim = simulateBenchmarkDCA(benchTickers, holdDays, { startMonthIdx, endMonthIdx });
  results.bench = { sim: benchSim, metrics: computeMetrics(bt.allDates, benchSim.equity, benchSim.totalInvested, { fromIdx, toIdx }) };
  results.window = { fromIdx, toIdx };
  return results;
}

function setupBacktestTriggers() {
  const lazySection = (detId, section) => {
    const det = byId(detId);
    if (!det) return;
    let loaded = false;
    det.addEventListener("toggle", () => {
      if (det.open && !loaded) { loaded = true; runSectionBacktest(section); }
    });
  };
  const lazyTop = (detId, which) => {
    const det = byId(detId);
    if (!det) return;
    let loaded = false;
    det.addEventListener("toggle", () => {
      if (det.open && !loaded) { loaded = true; runTopBacktest(which); }
    });
  };
  lazySection("stocks-backtest-section", "stocks");
  lazySection("crypto-backtest-section", "crypto");
  lazyTop("topstocks-backtest-section", "topstocks");
  lazyTop("topcrypto-backtest-section", "topcrypto");
}

function currentHorizon(section) {
  const active = document.querySelector(`.horizon-tabs[data-section="${section}"] .btn-lite.active`);
  return active ? HORIZON_BY_ID[active.dataset.horizon] : HORIZON_BY_ID[SECTION_DEFAULT_HORIZON[section]];
}

function portfolioUniverse(section) {
  // Exclude benchmark tickers from the pickable universe so the portfolio
  // isn't just "buys SPY" by default — benchmarks are only for comparison.
  const items = sectionItems(section);
  const bench = new Set(SECTION_BENCHMARK[section] || []);
  return items.map(i => i.ticker).filter(t => !bench.has(t));
}

function runSectionBacktest(section) {
  const h = currentHorizon(section);
  const universe = portfolioUniverse(section);
  const benchTickers = SECTION_BENCHMARK[section];
  const bodyId = `${section}-backtest-body`;
  injectPeriodBar(bodyId, section, () => runSectionBacktest(section));
  const periodId = PERIOD_STATE[section] || "full";
  byId(bodyId).innerHTML = `<div class="footnote">Running backtest&hellip;</div>`;
  setTimeout(() => {
    const results = runBacktestFor(section, h.days, universe, benchTickers, periodId);
    if (!results) { byId(bodyId).innerHTML = `<div class="footnote">No backtest data available.</div>`; return; }
    renderBacktest(bodyId, results, benchTickers, h.days, h.label);
    renderBestHorizonNote(section);
  }, 10);
}

function runTopBacktest(which) {
  const universe = portfolioUniverse(which);
  const benchTickers = SECTION_BENCHMARK[which];
  const bodyId = `${which}-backtest-body`;
  injectPeriodBar(bodyId, which, () => runTopBacktest(which));
  const periodId = PERIOD_STATE[which] || "full";
  byId(bodyId).innerHTML = `<div class="footnote">Running backtest&hellip;</div>`;
  setTimeout(() => {
    // For stocks we use the Max strategy validated by the Python backtester
    // (rank-weighted, hold-forever, next-day-open entry — substantially beats
    // SPY DCA across both halves and 3 of 4 quartiles of the spine).
    // Crypto keeps the per-pick variant for now (still being tuned).
    const results = (which === "topstocks")
      ? runTopBacktestMax(which, universe, benchTickers, periodId)
      : runTopBacktestPerPick(which, universe, benchTickers, periodId);
    if (!results) { byId(bodyId).innerHTML = `<div class="footnote">No backtest data available.</div>`; return; }
    const label = (which === "topstocks") ? "Max (rank-weighted, 5% cap, hold-forever)" : "Per-Pick";
    renderBacktest(bodyId, results, benchTickers, null, label);
  }, 10);
}

/* Max strategy (stocks). Three changes vs Per-Pick:
   1. Rank-weighted sizing — top pick gets 1/1, 2nd 1/2, 3rd 1/3, … normalized.
      Tilts capital toward higher-conviction names without going all-in.
   2. Hold-forever — no scheduled sell. Removes look-ahead (the per-pick variant
      assigned each historical buy a hold length derived from TODAY's
      bestHorizonFor) and lets winners run.
   3. Next-day-open entry — the live scan runs after market close, so we book
      the fill at the close of the *next* trading day (a 1-bar delay). */
function runTopBacktestMax(which, universe, benchTickers, periodId) {
  const bt = buildBT();
  if (!bt.allDates || !bt.allDates.length) return null;
  const firstValid = firstValidMonthIdx(universe, 3);
  const { startMonthIdx, endMonthIdx } = periodWindow(periodId, firstValid);
  const fromIdx = bt.monthFirstIdx[startMonthIdx] || 0;
  const toIdx = endMonthIdx < bt.monthFirstIdx.length
    ? bt.monthFirstIdx[endMonthIdx]
    : bt.allDates.length;
  const results = {};
  for (const n of [1, 5, 10]) {
    const sim = simulateDCAMax(universe, n, { startMonthIdx, endMonthIdx, entryDelay: 1 });
    const metrics = computeMetrics(bt.allDates, sim.equity, sim.totalInvested, { fromIdx, toIdx });
    results[n] = { sim, metrics };
  }
  const benchSim = simulateBenchmarkDCAMax(benchTickers, { startMonthIdx, endMonthIdx, entryDelay: 1 });
  results.bench = { sim: benchSim, metrics: computeMetrics(bt.allDates, benchSim.equity, benchSim.totalInvested, { fromIdx, toIdx }) };
  results.window = { fromIdx, toIdx };
  return results;
}

function runTopBacktestPerPick(which, universe, benchTickers, periodId) {
  const bt = buildBT();
  if (!bt.allDates || !bt.allDates.length) return null;
  const horizonByTicker = TOP_HORIZONS[which] || {};
  const fallback = which === "topcrypto" ? 60 : 252; // topstocks defaults to 1Y
  const firstValid = firstValidMonthIdx(universe, 3);
  const { startMonthIdx, endMonthIdx } = periodWindow(periodId, firstValid);
  const fromIdx = bt.monthFirstIdx[startMonthIdx] || 0;
  const toIdx = endMonthIdx < bt.monthFirstIdx.length
    ? bt.monthFirstIdx[endMonthIdx]
    : bt.allDates.length;
  const results = {};
  for (const n of [1, 5, 10]) {
    const sim = simulateDCAPerPick(universe, n, horizonByTicker, { startMonthIdx, endMonthIdx, fallbackHold: fallback });
    const metrics = computeMetrics(bt.allDates, sim.equity, sim.totalInvested, { fromIdx, toIdx });
    results[n] = { sim, metrics };
  }
  // Benchmark mirrors Top-5's per-month horizon schedule so capital timing matches.
  const benchSim = simulateBenchmarkDCAPerPick(benchTickers, results[5].sim.pickHorizonsByMonth, { startMonthIdx });
  results.bench = { sim: benchSim, metrics: computeMetrics(bt.allDates, benchSim.equity, benchSim.totalInvested, { fromIdx, toIdx }) };
  results.window = { fromIdx, toIdx };
  return results;
}

/* ---------- "Best horizon for section" ---------- */
function renderBestHorizonNote(section) {
  const note = byId(`${section}-best-horizon-note`);
  if (!note) return;
  note.textContent = "Calculating best horizon for this section…";
  // Compute top-5 DCA Sharpe across all horizons — lightweight but not instant.
  const items = sectionItems(section);
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

/* ---------- This Month's Picks + Concentration (CAP5) ---------- */

/* Pull the cumulative CAP5 portfolio state by running the Max sim on the stocks
   universe (SPY excluded) from the first valid month to the most recent. Used
   by renderThisMonthsPicks (to know which tickers are already capped) and by
   renderConcentration (to show cost-basis distribution). */
function computeCap5State(topN = 5) {
  const bt = buildBT();
  if (!bt.allDates || !bt.allDates.length) return null;
  const universe = ITEMS_STOCKS.map(i => i.ticker).filter(t => t !== "SPY");
  const firstValid = firstValidMonthIdx(universe, 3);
  const sim = simulateDCAMax(universe, topN, {
    startMonthIdx: firstValid,
    entryDelay: 1,
    maxTickerFrac: MAX_TICKER_FRAC,
  });
  return { sim, universe, firstValid };
}

function tickerItemByTicker(ticker) {
  return ITEMS_STOCKS.find(i => i.ticker === ticker) || null;
}

function renderThisMonthsPicks() {
  const loading = byId("this-month-loading");
  const content = byId("this-month-content");
  if (!loading || !content) return;

  const state = computeCap5State(5);
  if (!state) {
    loading.textContent = "Could not compute picks (no backtest data).";
    return;
  }

  // Rank all stocks by today's CAP5+SMA12M score (trailing 12-month mean of
  // conviction) — the step39 production winner. The CAP5 *exclusions* come
  // from the long-running sim state, so we can tell the user "this ticker
  // is capped, we'd backfill with the next candidate". Tickers without 12M
  // of smoothed history fall back to raw conviction, then final_score.
  const stockRankScore = (it) => {
    const s = it.conviction_smooth12m;
    if (s != null && Number.isFinite(Number(s))) return Number(s);
    const c = it.conviction;
    if (c != null && Number.isFinite(Number(c))) return Number(c);
    const f = it.final_score;
    if (f != null && Number.isFinite(Number(f))) return Number(f);
    return null;
  };
  const stocks = ITEMS_STOCKS
    .filter(i => i.ticker !== "SPY" && stockRankScore(i) != null)
    .map(i => ({ it: i, rs: stockRankScore(i) }));
  stocks.sort((a, b) => b.rs - a.rs);

  // Accumulated cost basis per ticker from the running sim.
  const tkCost = new Map();
  for (const pos of state.sim.positions) {
    tkCost.set(pos.tk, (tkCost.get(pos.tk) || 0) + pos.cost);
  }
  const capDollars = (state.sim.totalInvested + DCA_MONTHLY) * MAX_TICKER_FRAC;

  // Walk ranked list; mark each name as "eligible" or "capped". First 5
  // eligible become the picks; any capped ones we show as context.
  const picks = [];
  const skipped = [];
  for (const { it, rs } of stocks) {
    const basis = tkCost.get(it.ticker) || 0;
    if (basis >= capDollars) {
      skipped.push({ item: it, basis, rs });
      continue;
    }
    picks.push({ item: it, basis, rs });
    if (picks.length === 5) break;
  }

  if (!picks.length) {
    loading.textContent = "No eligible stocks this month.";
    return;
  }

  // Rank-weight normalized allocations over exactly picks.length (5).
  const rawW = picks.map((_, i) => 1 / (i + 1));
  const sumW = rawW.reduce((a, b) => a + b, 0);
  const weights = rawW.map(r => r / sumW);

  // Top-1 = first ranked eligible. Skipped names get shown if any were bumped.
  const top = picks[0];
  const topItem = top.item;
  const asOfStr = FULL.as_of ? String(FULL.as_of).slice(0, 10) : "";

  byId("single-pick-asof").textContent = asOfStr ? `as of ${asOfStr}` : "";
  byId("single-pick-ticker").textContent = topItem.ticker;

  const probBits = [];
  for (const h of ["1y", "3y", "5y"]) {
    const v = topItem[`prob_${h}`];
    if (v != null && Number.isFinite(Number(v))) probBits.push(`${h.toUpperCase()} prob ${Number(v).toFixed(0)}%`);
  }
  const topRankScore = top.rs;
  byId("single-pick-meta").innerHTML = `
    <span>Price <strong>${fmtPriceVal(topItem.last_price)}</strong></span>
    <span>CAP5 Score <strong>${Number(topRankScore).toFixed(1)}</strong></span>
    <span>Pullback <strong>${Number(topItem.washout_today ?? 0).toFixed(0)}</strong></span>
    <span>Quality <strong>${Number(topItem.quality ?? 0).toFixed(0)}</strong></span>
    ${probBits.map(b => `<span>${b}</span>`).join("")}
  `;
  byId("single-pick-foot").textContent =
    `The highest-ranked stock today by the CAP5+SMA12M opportunity score (trailing ` +
    `12-month mean of conviction), after the 5% concentration filter. This is the ` +
    `step39 production winner — +0.90pp CAGR over raw conviction ranking, robust ` +
    `across 5/5 rolling 10Y windows. Still do your own work. ` +
    `Drawdowns of 30% or more on single names are routine.`;

  // Render the top-5 table.
  const rowsEl = byId("month-picks-rows");
  rowsEl.innerHTML = "";
  for (let i = 0; i < picks.length; i++) {
    const p = picks[i].item;
    const rs = picks[i].rs;
    const alloc = weights[i] * DCA_MONTHLY;
    const wPct = (weights[i] * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "max-ticker-row month-picks-row";
    row.innerHTML = `
      <span class="pk-rank">#${i + 1}</span>
      <span class="pk-ticker">${p.ticker}</span>
      <span class="pk-cell">${wPct}%</span>
      <span class="pk-cell">$${alloc.toFixed(0)}</span>
      <span class="pk-cell">${fmtPriceVal(p.last_price)}</span>
      <span class="pk-cell">${Number(rs ?? 0).toFixed(1)}</span>
      <span class="pk-cell pk-cell-hide-mobile">${Number(p.washout_today ?? 0).toFixed(0)}</span>
      <span class="pk-cell pk-cell-hide-mobile">${Number(p.quality ?? 0).toFixed(0)}</span>
    `;
    rowsEl.appendChild(row);
  }

  // Note about any ticker the cap bumped past.
  const noteEl = byId("month-picks-note");
  if (skipped.length) {
    const names = skipped.slice(0, 3).map(s => `<strong>${s.item.ticker}</strong>`).join(", ");
    noteEl.innerHTML = `The concentration cap filtered out ${names}${skipped.length > 3 ? ` and ${skipped.length - 3} more` : ""} — they&rsquo;re currently above 5% of your cumulative basis, so this month&rsquo;s capital rotates to the next-ranked candidates above.`;
  } else {
    noteEl.innerHTML = `No ticker is currently above the 5% cap, so the top-5 raw ranking and the post-cap picks are the same this month.`;
  }

  loading.style.display = "none";
  content.style.display = "block";
}

function renderConcentration() {
  const loading = byId("concentration-loading");
  const content = byId("concentration-content");
  if (!loading || !content) return;

  const state = computeCap5State(5);
  if (!state) { loading.textContent = "No data."; return; }
  const { sim } = state;

  // Roll up cost basis, position count, and current value by ticker.
  const lastDayIdx = buildBT().allDates.length - 1;
  const byTk = new Map();
  for (const pos of sim.positions) {
    const e = byTk.get(pos.tk) || { n: 0, cost: 0, value: 0 };
    e.n += 1;
    e.cost += pos.cost;
    const px = buildBT().priceLookup[pos.tk]?.get(buildBT().allDates[lastDayIdx]);
    if (px != null && Number.isFinite(px)) e.value += pos.shares * px;
    byTk.set(pos.tk, e);
  }
  const total = sim.totalInvested;
  if (!total) { loading.textContent = "No positions yet."; return; }

  const rows = [...byTk.entries()].map(([tk, e]) => ({
    tk,
    n: e.n,
    cost: e.cost,
    value: e.value,
    pct: e.cost / total,
    pnl: e.value - e.cost,
    pnlPct: e.cost > 0 ? (e.value - e.cost) / e.cost : 0,
  }));
  rows.sort((a, b) => b.pct - a.pct);

  const rowsEl = byId("concentration-rows");
  rowsEl.innerHTML = "";
  let cappedCount = 0;
  let nearCount = 0;
  for (const r of rows.slice(0, 20)) {
    const capped = r.pct >= MAX_TICKER_FRAC;
    const near = r.pct >= MAX_TICKER_FRAC * 0.8;
    const status = capped ? "CAPPED" : near ? "near cap" : "headroom";
    const statusClass = capped ? "status-cap" : near ? "status-near" : "status-ok";
    const pnlClass = r.pnl >= 0 ? "cn-pnl-pos" : "cn-pnl-neg";
    if (capped) cappedCount++;
    else if (near) nearCount++;
    const row = document.createElement("div");
    row.className = "concentration-row";
    row.innerHTML = `
      <span class="cn-ticker">${r.tk}</span>
      <span class="cn-cell">${r.n}</span>
      <span class="cn-cell">$${r.cost.toFixed(0)}</span>
      <span class="cn-cell">${(r.pct * 100).toFixed(2)}%</span>
      <span class="cn-cell cn-cell-hide-mobile">$${r.value.toFixed(0)}</span>
      <span class="cn-cell cn-cell-hide-mobile ${pnlClass}">${r.pnlPct >= 0 ? "+" : ""}${(r.pnlPct * 100).toFixed(0)}%</span>
      <span class="cn-status ${statusClass}">${status}</span>
    `;
    rowsEl.appendChild(row);
  }

  const totalVal = rows.reduce((a, r) => a + r.value, 0);
  const summary = byId("concentration-summary");
  const sinceMonth = state.firstValid != null && buildBT().allDates[buildBT().monthFirstIdx[state.firstValid]]
    ? String(buildBT().allDates[buildBT().monthFirstIdx[state.firstValid]]).slice(0, 7)
    : "start";
  summary.innerHTML = `
    ${sim.positions.length} positions opened since ${sinceMonth}.
    Cumulative invested <strong>$${total.toLocaleString()}</strong>,
    current portfolio value <strong>$${totalVal.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong>
    (${totalVal > total ? "+" : ""}${((totalVal - total) / total * 100).toFixed(1)}%).
    ${cappedCount} ticker${cappedCount === 1 ? " is" : "s are"} currently at or above the 5% cap;
    ${nearCount} within 20% of the cap.
    Showing top 20 of ${rows.length} held tickers by basis share.
  `;

  loading.style.display = "none";
  content.style.display = "block";
}

/* ---------- go ---------- */
load();
