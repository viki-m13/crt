/**
 * CRT Recovery Predictor v3.0 - Frontend JavaScript
 * Displays undervalued stocks with recovery potential
 */

const DATA_URL = "./data/full.json";
const CACHE_BUST = String(Date.now());

function withBust(url) {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}v=${encodeURIComponent(CACHE_BUST)}`;
}

function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${Math.round(v)}%`;
}

function fmtSignedPct(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${Math.round(v)}%`;
}

function fmtNum0(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${v.toFixed(0)}`;
}

function fmtNum1(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${v.toFixed(1)}`;
}

function fmtPrice(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `$${v.toFixed(2)}`;
}

function byId(id) { return document.getElementById(id); }

function signalClass(signal) {
  switch (signal) {
    case "STRONG_BUY": return "signal-strong";
    case "BUY": return "signal-buy";
    case "WATCH": return "signal-watch";
    default: return "";
  }
}

function signalText(signal) {
  switch (signal) {
    case "STRONG_BUY": return "STRONG BUY";
    case "BUY": return "BUY";
    case "WATCH": return "WATCH";
    default: return signal || "—";
  }
}

// Draw price chart with gradient based on trend
function drawPriceChart(canvas, dates, prices) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
  const h = canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
  ctx.clearRect(0, 0, w, h);

  const n = prices.length;
  if (n < 3) return;

  const pad = 12 * devicePixelRatio;
  let minP = Infinity, maxP = -Infinity;
  for (let i = 0; i < n; i++) {
    const p = Number(prices[i]);
    if (!Number.isFinite(p)) continue;
    if (p < minP) minP = p;
    if (p > maxP) maxP = p;
  }
  if (!(maxP > minP)) return;

  const x0 = pad, x1 = w - pad, y0 = pad, y1 = h - pad;
  function xAt(i) { return x0 + (x1 - x0) * (i / (n - 1)); }
  function yAt(p) { return y1 - (y1 - y0) * ((p - minP) / (maxP - minP)); }

  // Calculate overall trend for color
  const firstPrice = prices[0];
  const lastPrice = prices[n - 1];
  const trend = (lastPrice - firstPrice) / firstPrice;
  const isUp = trend > 0;

  // Draw the line
  ctx.lineWidth = 2.5 * devicePixelRatio;
  ctx.strokeStyle = isUp ? "rgba(15, 120, 70, 0.85)" : "rgba(180, 50, 50, 0.75)";
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const p = Number(prices[i]);
    if (!Number.isFinite(p)) continue;
    const x = xAt(i), y = yAt(p);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Today marker
  ctx.fillStyle = isUp ? "rgba(15, 120, 70, 0.9)" : "rgba(180, 50, 50, 0.85)";
  ctx.strokeStyle = "rgba(0,0,0,0.7)";
  ctx.lineWidth = 1.2 * devicePixelRatio;
  ctx.beginPath();
  ctx.arc(xAt(n - 1), yAt(prices[n - 1]), 5 * devicePixelRatio, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

// Outcome box for 1Y/3Y/5Y
function outcomeBox(label, item, horizon) {
  const prob = item[`prob_positive_${horizon}`];
  const beatSpy = item[`prob_beat_spy_${horizon}`];
  const median = item[`median_return_${horizon}`];
  const downside = item[`downside_${horizon}`];
  const upside = item[`upside_${horizon}`];
  const samples = item[`sample_size_${horizon}`];

  const b = document.createElement("div");
  b.className = "outbox";

  if (!Number.isFinite(prob) || !samples || samples === 0) {
    b.innerHTML = `<div class="h">${label}</div><div class="r"><span>Not enough data</span><strong>—</strong></div>`;
    return b;
  }

  b.innerHTML = `
    <div class="h">${label}</div>
    <div class="r"><span>Prob positive</span><strong class="prob-value">${fmtPct(prob)}</strong></div>
    ${beatSpy !== null && beatSpy !== undefined ? `<div class="r"><span>Beat SPY</span><strong>${fmtPct(beatSpy)}</strong></div>` : ""}
    <div class="r"><span>Median return</span><strong>${fmtSignedPct(median)}</strong></div>
    ${downside !== null && upside !== null ? `<div class="r"><span>Range (10-90th)</span><strong>${fmtSignedPct(downside)} to ${fmtSignedPct(upside)}</strong></div>` : ""}
    <div class="r"><span>Based on</span><strong>${samples} similar cases</strong></div>
  `;
  return b;
}

// Render a stock card
function renderCard(container, item) {
  const card = document.createElement("div");
  card.className = "card";

  // Header
  const h = document.createElement("div");
  h.className = "card-head";
  h.innerHTML = `
    <div>
      <div class="ticker">${item.ticker}</div>
      <div class="signal-badge ${signalClass(item.signal)}">${signalText(item.signal)}</div>
      <div class="price">${fmtPrice(item.price)}</div>
    </div>
    <div class="metrics">
      <div class="metric">
        <div class="mline"><span>Score</span> <strong>${fmtNum0(item.opportunity_score)}/100</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>Discount</span> <strong class="prob-highlight">${fmtPct(item.discount_from_high)}</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>Beat SPY (1Y)</span> <strong>${fmtPct(item.prob_beat_spy_1y)}</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>5Y Return</span> <strong>${fmtSignedPct(item.five_year_return)}</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>Recovery</span> <strong>${item.early_recovery ? 'Yes' : 'No'}</strong></div>
      </div>
    </div>
  `;
  card.appendChild(h);

  // Thesis
  if (item.thesis) {
    const thesis = document.createElement("div");
    thesis.className = "thesis";
    thesis.innerHTML = `<strong>Why buy now?</strong> ${item.thesis}`;
    card.appendChild(thesis);
  }

  const grid = document.createElement("div");
  grid.className = "grid2";

  // Left: Key metrics + Outcomes
  const left = document.createElement("div");

  // Key metrics list
  const ul = document.createElement("ul");
  ul.className = "bullets";

  const trendStatus = item.above_sma20 ? "above" : "below";
  const trendColor = item.above_sma20 ? "positive" : "negative";
  const trend50Status = item.above_sma50 ? "above" : "below";
  const trend50Color = item.above_sma50 ? "positive" : "negative";

  ul.innerHTML = `
    <li>Trading <strong class="prob-highlight">${fmtPct(item.discount_from_high)}</strong> below 52-week high of <strong>${fmtPrice(item.high_52w)}</strong></li>
    <li>Position in 52-week range: <strong>${fmtNum0(item.position_in_52w_range)}%</strong> (0% = at low, 100% = at high)</li>
    <li>Price is <strong class="${trendColor}">${trendStatus}</strong> the 20-day moving average</li>
    <li>Price is <strong class="${trend50Color}">${trend50Status}</strong> the 50-day moving average</li>
    <li>1-month momentum: <strong>${fmtSignedPct(item.momentum_1m)}</strong></li>
    <li>Monthly win rate: <strong>${fmtPct(item.monthly_win_rate)}</strong></li>
  `;
  left.appendChild(ul);

  // Outcomes grid
  const outcomes = document.createElement("div");
  outcomes.className = "outcomes";
  outcomes.appendChild(outcomeBox("1 Year", item, "1y"));
  outcomes.appendChild(outcomeBox("3 Years", item, "3y"));
  outcomes.appendChild(outcomeBox("5 Years", item, "5y"));
  left.appendChild(outcomes);

  // Right: Chart
  const right = document.createElement("div");
  right.className = "chart";
  const canvas = document.createElement("canvas");
  canvas.className = "canvas";
  right.appendChild(canvas);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = `<span class="legend-text">5-Year Price History</span>`;
  right.appendChild(legend);

  grid.appendChild(left);
  grid.appendChild(right);
  card.appendChild(grid);

  // Draw chart if we have data
  const series = item.series;
  if (series && series.prices && series.prices.length) {
    requestAnimationFrame(() => drawPriceChart(canvas, series.dates, series.prices));
  }

  container.appendChild(card);
}

// Table row HTML
function rowHtml(item) {
  const t = item.ticker;
  const thesisShort = item.thesis ? (item.thesis.length > 80 ? item.thesis.substring(0, 80) + '...' : item.thesis) : '';
  return `
    <tr data-ticker="${t}">
      <td class="tcell">${t}</td>
      <td><span class="signal-badge-sm ${signalClass(item.signal)}">${signalText(item.signal)}</span></td>
      <td class="num">${fmtNum0(item.opportunity_score)}</td>
      <td class="num">${fmtPct(item.discount_from_high)}</td>
      <td class="num">${fmtPct(item.prob_beat_spy_1y)}</td>
      <td class="num">${fmtSignedPct(item.median_return_1y)}</td>
      <td class="num">${fmtSignedPct(item.median_return_3y)}</td>
      <td class="num">${fmtNum0(item.sample_size_1y)}</td>
      <td><span class="${item.early_recovery ? 'recovery-yes' : 'recovery-no'}">${item.early_recovery ? 'YES' : 'No'}</span></td>
      <td class="thesis-cell">${thesisShort}</td>
    </tr>
  `;
}

async function loadJSON(url) {
  const u = withBust(url);
  const r = await fetch(u, {
    cache: "no-store",
    headers: {
      "Pragma": "no-cache",
      "Cache-Control": "no-cache",
    },
  });
  if (!r.ok) throw new Error(`Fetch failed ${r.status}: ${url}`);
  return await r.json();
}

function setSortButtons(active) {
  document.querySelectorAll(".btn-lite").forEach(b => {
    b.classList.toggle("active", b.dataset.sort === active);
  });
}

function formatAsOf(asOf) {
  if (!asOf) return "—";
  let s = String(asOf).trim();
  if (s.includes(" ") && !s.includes("T")) s = s.replace(" ", "T");
  let d = new Date(s);
  if (Number.isNaN(d.getTime())) {
    s = s.replace(/\.(\d+)(Z|[+-]\d\d:\d\d)?$/, "$2");
    d = new Date(s);
  }
  if (Number.isNaN(d.getTime())) return String(asOf);

  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).formatToParts(d);

  const get = (type) => (parts.find(p => p.type === type)?.value || "");
  return `${get("year")}-${get("month")}-${get("day")} ${get("hour")}:${get("minute")} ${get("dayPeriod")} ET`;
}

(async function main() {
  let full;
  try {
    full = await loadJSON(DATA_URL);
  } catch (e) {
    byId("top10").innerHTML = `<div class="footnote">No data yet. Run the daily scan to generate data.</div>`;
    return;
  }

  byId("asOf").textContent = formatAsOf(full.as_of);

  // Update signal counts
  if (full.summary) {
    byId("countStrong").textContent = full.summary.strong_buy || 0;
    byId("countBuy").textContent = full.summary.buy || 0;
    byId("countWatch").textContent = full.summary.watch || 0;
  }

  let items = full.items || [];
  let sortMode = "score";
  let filterRecoveryOnly = false;

  function getFilteredItems() {
    let list = [...items];
    if (filterRecoveryOnly) {
      list = list.filter(x => x.early_recovery);
    }
    return list;
  }

  function renderTable(list) {
    byId("rows").innerHTML = list.map(it => rowHtml(it)).join("");
  }

  async function loadDetail(ticker) {
    // Check if embedded
    if (full.details && full.details[ticker]) {
      return full.details[ticker];
    }
    // Load from individual file
    try {
      return await loadJSON(`./data/tickers/${ticker}.json`);
    } catch (e) {
      // Return the basic item if detail load fails
      return items.find(it => it.ticker === ticker) || null;
    }
  }

  async function renderTop(list, count = 10) {
    const c = byId("top10");
    c.innerHTML = "";
    const top = list.slice(0, count);
    for (const it of top) {
      let detail;
      try {
        detail = await loadDetail(it.ticker);
      } catch (err) {
        detail = it;
      }
      renderCard(c, detail || it);
    }
  }

  function applySort(list) {
    const sorted = [...list];
    if (sortMode === "score") {
      sorted.sort((a, b) => (b.opportunity_score - a.opportunity_score));
    } else if (sortMode === "discount") {
      sorted.sort((a, b) => (b.discount_from_high - a.discount_from_high));
    } else if (sortMode === "beatspy") {
      sorted.sort((a, b) => ((b.prob_beat_spy_1y || 0) - (a.prob_beat_spy_1y || 0)));
    } else if (sortMode === "median") {
      sorted.sort((a, b) => ((b.median_return_1y || 0) - (a.median_return_1y || 0)));
    }
    return sorted;
  }

  async function rerender() {
    const filtered = getFilteredItems();
    const list = applySort(filtered);
    renderTable(list);
    await renderTop(list);
  }

  // Sort button handlers
  document.querySelectorAll(".btn-lite").forEach(btn => {
    btn.addEventListener("click", async () => {
      sortMode = btn.dataset.sort;
      setSortButtons(sortMode);
      await rerender();
    });
  });
  setSortButtons(sortMode);

  // Filter checkbox handler
  byId("filterRecovery").addEventListener("change", async (e) => {
    filterRecoveryOnly = e.target.checked;
    await rerender();
  });

  await rerender();

  // Table row click handler
  byId("rows").addEventListener("click", async (e) => {
    const tr = e.target.closest("tr");
    if (!tr) return;
    const t = tr.dataset.ticker;
    if (!t) return;

    document.querySelectorAll("#rows tr").forEach(r => r.classList.remove("highlight"));
    tr.classList.add("highlight");

    const filtered = getFilteredItems();
    const current = applySort(filtered);
    const idx = current.findIndex(x => x.ticker === t);
    if (idx < 0) return;
    const rotated = [current[idx], ...current.filter((_, i) => i !== idx)];
    await renderTop(rotated, 1);
    document.querySelector(".masthead").scrollIntoView({ behavior: "smooth" });
  });

  // Search handler
  function applySearch() {
    const q = (byId("q").value || "").trim().toUpperCase();
    const filtered = getFilteredItems();
    if (!q) {
      rerender();
      return;
    }
    const searched = applySort(filtered).filter(x => x.ticker.includes(q));
    renderTable(searched);
    (async () => { await renderTop(searched); })();
  }

  byId("go").addEventListener("click", applySearch);
  byId("q").addEventListener("input", applySearch);
})();
