/**
 * CRT Recovery Predictor v3.0 - Financial Times Style UI
 * Clean, minimal, information-dense design
 */

const DATA_URL = "./data/full.json";
const CACHE_BUST = String(Date.now());

// State
let allItems = [];
let expandedTickers = new Set(); // Support multiple expanded rows
let chartRange = 'max';
let signalFilters = { STRONG_BUY: true, BUY: true, WATCH: true };
let filterRecoveryOnly = false;
let sortMode = 'score';

// Helpers
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

function valueClass(value, threshold = 0) {
  if (value === null || value === undefined) return '';
  return value > threshold ? 'positive' : (value < threshold ? 'negative' : '');
}

// Calculate rolling 52-week high for historical buy zones
function calculateBuyZones(dates, prices) {
  const n = prices.length;
  if (n < 252) return null;

  const zones = [];
  const lookback = 252;

  for (let i = lookback; i < n; i++) {
    let high52w = -Infinity;
    for (let j = i - lookback; j <= i; j++) {
      if (prices[j] > high52w) high52w = prices[j];
    }

    const price = prices[i];
    const discount = ((high52w - price) / high52w) * 100;

    let zone = null;
    if (discount >= 30) zone = 'strong';
    else if (discount >= 20) zone = 'buy';
    else if (discount >= 10) zone = 'watch';

    zones.push({ date: dates[i], price, discount, zone, high52w });
  }

  return zones;
}

// Filter data based on date range
function filterByRange(dates, prices, range) {
  const n = dates.length;
  if (n === 0) return { dates: [], prices: [] };

  const lastDate = new Date(dates[n - 1]);
  let cutoffDate;

  if (range === '1y') {
    cutoffDate = new Date(lastDate);
    cutoffDate.setFullYear(cutoffDate.getFullYear() - 1);
  } else if (range === '5y') {
    cutoffDate = new Date(lastDate);
    cutoffDate.setFullYear(cutoffDate.getFullYear() - 5);
  } else if (range === '10y') {
    cutoffDate = new Date(lastDate);
    cutoffDate.setFullYear(cutoffDate.getFullYear() - 10);
  } else {
    return { dates, prices };
  }

  const filteredDates = [];
  const filteredPrices = [];

  for (let i = 0; i < n; i++) {
    const d = new Date(dates[i]);
    if (d >= cutoffDate) {
      filteredDates.push(dates[i]);
      filteredPrices.push(prices[i]);
    }
  }

  return { dates: filteredDates, prices: filteredPrices };
}

// Draw chart with buy zones
function drawChart(canvas, dates, prices, range = 'max') {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const filtered = filterByRange(dates, prices, range);
  const filteredDates = filtered.dates;
  const filteredPrices = filtered.prices;

  const n = filteredPrices.length;
  if (n < 3) {
    ctx.font = "12px system-ui";
    ctx.fillStyle = "#999";
    ctx.textAlign = "center";
    ctx.fillText("Insufficient data", w / 2, h / 2);
    return;
  }

  const pad = { top: 15, right: 15, bottom: 25, left: 50 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  let minP = Infinity, maxP = -Infinity;
  for (let i = 0; i < n; i++) {
    const p = Number(filteredPrices[i]);
    if (Number.isFinite(p)) {
      if (p < minP) minP = p;
      if (p > maxP) maxP = p;
    }
  }

  const range_p = maxP - minP;
  minP -= range_p * 0.05;
  maxP += range_p * 0.05;

  function xAt(i) { return pad.left + chartW * (i / (n - 1)); }
  function yAt(p) { return pad.top + chartH * (1 - (p - minP) / (maxP - minP)); }

  const zones = calculateBuyZones(filteredDates, filteredPrices);

  // Draw buy zones
  if (zones && zones.length > 0) {
    const zoneStartIdx = filteredDates.length - zones.length;
    let currentZone = null;
    let zoneStart = null;

    for (let i = 0; i < zones.length; i++) {
      const z = zones[i];
      const idx = zoneStartIdx + i;

      if (z.zone !== currentZone) {
        if (currentZone && zoneStart !== null) {
          const x1 = xAt(zoneStart);
          const x2 = xAt(idx);
          ctx.fillStyle = currentZone === 'strong' ? 'rgba(6, 78, 59, 0.3)' :
                          currentZone === 'buy' ? 'rgba(34, 197, 94, 0.2)' :
                          'rgba(251, 191, 36, 0.2)';
          ctx.fillRect(x1, pad.top, x2 - x1, chartH);
        }
        currentZone = z.zone;
        zoneStart = z.zone ? idx : null;
      }
    }

    if (currentZone && zoneStart !== null) {
      const x1 = xAt(zoneStart);
      const x2 = xAt(n - 1);
      ctx.fillStyle = currentZone === 'strong' ? 'rgba(6, 78, 59, 0.3)' :
                      currentZone === 'buy' ? 'rgba(34, 197, 94, 0.2)' :
                      'rgba(251, 191, 36, 0.2)';
      ctx.fillRect(x1, pad.top, x2 - x1, chartH);
    }
  }

  // Grid
  ctx.strokeStyle = 'rgba(0,0,0,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    const price = maxP - ((maxP - minP) / 4) * i;
    ctx.fillStyle = '#888';
    ctx.font = "10px system-ui";
    ctx.textAlign = 'right';
    ctx.fillText(`$${price.toFixed(0)}`, pad.left - 6, y + 3);
  }

  // Date labels
  ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const idx = Math.floor((n - 1) * i / 4);
    const x = xAt(idx);
    const date = new Date(filteredDates[idx]);
    ctx.fillText(date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' }), x, h - 6);
  }

  // Price line
  const firstPrice = filteredPrices[0];
  const lastPrice = filteredPrices[n - 1];
  const isUp = lastPrice > firstPrice;

  ctx.strokeStyle = isUp ? '#0f7846' : '#c0392b';
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  for (let i = 0; i < n; i++) {
    const p = Number(filteredPrices[i]);
    if (!Number.isFinite(p)) continue;
    const x = xAt(i), y = yAt(p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Current price dot
  const lastX = xAt(n - 1);
  const lastY = yAt(lastPrice);
  ctx.fillStyle = isUp ? '#0f7846' : '#c0392b';
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
  ctx.fill();
}

// Create compact expanded content
function createExpandedContent(item, currentRange) {
  return `
    <td colspan="12" class="expand-cell">
      <div class="expand-panel">
        <div class="expand-top">
          <div class="expand-thesis">
            <strong>Thesis:</strong> ${item.thesis || 'Analysis based on historical discount patterns and recovery probability.'}
          </div>
          <div class="expand-range">
            <button class="range-btn ${currentRange === '1y' ? 'active' : ''}" data-range="1y">1Y</button>
            <button class="range-btn ${currentRange === '5y' ? 'active' : ''}" data-range="5y">5Y</button>
            <button class="range-btn ${currentRange === '10y' ? 'active' : ''}" data-range="10y">10Y</button>
            <button class="range-btn ${currentRange === 'max' ? 'active' : ''}" data-range="max">Max</button>
          </div>
        </div>
        <div class="expand-body">
          <div class="expand-chart">
            <canvas class="chart-canvas" id="chart-${item.ticker}"></canvas>
            <div class="chart-key">
              <span><i class="key-strong"></i>Strong Buy 30%+</span>
              <span><i class="key-buy"></i>Buy 20-30%</span>
              <span><i class="key-watch"></i>Watch 10-20%</span>
            </div>
          </div>
          <div class="expand-stats">
            <div class="stat-section">
              <div class="stat-title">Current</div>
              <div class="stat-row"><span>Price</span><b>${fmtPrice(item.price)}</b></div>
              <div class="stat-row"><span>52W High</span><b>${fmtPrice(item.high_52w)}</b></div>
              <div class="stat-row"><span>Discount</span><b class="positive">${fmtPct(item.discount_from_high)}</b></div>
              <div class="stat-row"><span>5Y Return</span><b class="${valueClass(item.five_year_return)}">${fmtSignedPct(item.five_year_return)}</b></div>
            </div>
            <div class="stat-section">
              <div class="stat-title">1 Year</div>
              <div class="stat-row"><span>Prob +</span><b class="${valueClass(item.prob_positive_1y, 50)}">${fmtPct(item.prob_positive_1y)}</b></div>
              <div class="stat-row"><span>Beat SPY</span><b class="${valueClass(item.prob_beat_spy_1y, 50)}">${fmtPct(item.prob_beat_spy_1y)}</b></div>
              <div class="stat-row"><span>Median</span><b class="${valueClass(item.median_return_1y)}">${fmtSignedPct(item.median_return_1y)}</b></div>
              <div class="stat-row"><span>Samples</span><b>${item.sample_size_1y || '—'}</b></div>
            </div>
            <div class="stat-section">
              <div class="stat-title">3 Year</div>
              <div class="stat-row"><span>Prob +</span><b class="${valueClass(item.prob_positive_3y, 50)}">${fmtPct(item.prob_positive_3y)}</b></div>
              <div class="stat-row"><span>Beat SPY</span><b class="${valueClass(item.prob_beat_spy_3y, 50)}">${fmtPct(item.prob_beat_spy_3y)}</b></div>
              <div class="stat-row"><span>Median</span><b class="${valueClass(item.median_return_3y)}">${fmtSignedPct(item.median_return_3y)}</b></div>
              <div class="stat-row"><span>Samples</span><b>${item.sample_size_3y || '—'}</b></div>
            </div>
            <div class="stat-section">
              <div class="stat-title">5 Year</div>
              <div class="stat-row"><span>Prob +</span><b class="${valueClass(item.prob_positive_5y, 50)}">${fmtPct(item.prob_positive_5y)}</b></div>
              <div class="stat-row"><span>Median</span><b class="${valueClass(item.median_return_5y)}">${fmtSignedPct(item.median_return_5y)}</b></div>
              <div class="stat-row"><span>Samples</span><b>${item.sample_size_5y || '—'}</b></div>
            </div>
          </div>
        </div>
      </div>
    </td>
  `;
}

// Table row HTML with +/- toggle
function rowHtml(item, index) {
  const isExpanded = expandedTickers.has(item.ticker);
  const toggleIcon = isExpanded ? '−' : '+';

  return `
    <tr data-ticker="${item.ticker}" data-index="${index}" class="${isExpanded ? 'row-expanded' : ''}">
      <td class="col-toggle"><button class="toggle-btn" data-ticker="${item.ticker}">${toggleIcon}</button></td>
      <td class="col-ticker"><strong>${item.ticker}</strong></td>
      <td class="col-signal"><span class="signal ${signalClass(item.signal)}">${signalText(item.signal)}</span></td>
      <td class="col-num">${fmtNum0(item.opportunity_score)}</td>
      <td class="col-num">${fmtPct(item.discount_from_high)}</td>
      <td class="col-num ${valueClass(item.prob_beat_spy_1y, 50)}">${fmtPct(item.prob_beat_spy_1y)}</td>
      <td class="col-num ${valueClass(item.median_return_1y)}">${fmtSignedPct(item.median_return_1y)}</td>
      <td class="col-num ${valueClass(item.median_return_3y)}">${fmtSignedPct(item.median_return_3y)}</td>
      <td class="col-num ${valueClass(item.median_return_5y)}">${fmtSignedPct(item.median_return_5y)}</td>
      <td class="col-num hide-mobile">${fmtNum0(item.sample_size_1y)}</td>
      <td class="col-recovery hide-mobile">${item.early_recovery ? '✓' : ''}</td>
    </tr>
  `;
}

// Get filtered items
function getFilteredItems() {
  let list = [...allItems];
  list = list.filter(x => signalFilters[x.signal]);
  if (filterRecoveryOnly) {
    list = list.filter(x => x.early_recovery);
  }
  return list;
}

// Apply sort
function applySort(list) {
  const sorted = [...list];

  switch (sortMode) {
    case 'score':
      sorted.sort((a, b) => (b.opportunity_score || 0) - (a.opportunity_score || 0));
      break;
    case 'discount':
      sorted.sort((a, b) => (b.discount_from_high || 0) - (a.discount_from_high || 0));
      break;
    case 'beatspy1y':
      sorted.sort((a, b) => (b.prob_beat_spy_1y || 0) - (a.prob_beat_spy_1y || 0));
      break;
    case 'beatspy3y':
      sorted.sort((a, b) => (b.prob_beat_spy_3y || 0) - (a.prob_beat_spy_3y || 0));
      break;
    case 'median1y':
      sorted.sort((a, b) => (b.median_return_1y || 0) - (a.median_return_1y || 0));
      break;
    case 'median3y':
      sorted.sort((a, b) => (b.median_return_3y || 0) - (a.median_return_3y || 0));
      break;
    case 'median5y':
      sorted.sort((a, b) => (b.median_return_5y || 0) - (a.median_return_5y || 0));
      break;
    case 'prob1y':
      sorted.sort((a, b) => (b.prob_positive_1y || 0) - (a.prob_positive_1y || 0));
      break;
    case 'prob3y':
      sorted.sort((a, b) => (b.prob_positive_3y || 0) - (a.prob_positive_3y || 0));
      break;
    case 'prob5y':
      sorted.sort((a, b) => (b.prob_positive_5y || 0) - (a.prob_positive_5y || 0));
      break;
    case 'samples':
      sorted.sort((a, b) => (b.sample_size_1y || 0) - (a.sample_size_1y || 0));
      break;
    case '5yreturn':
      sorted.sort((a, b) => (b.five_year_return || 0) - (a.five_year_return || 0));
      break;
    case 'ticker':
      sorted.sort((a, b) => a.ticker.localeCompare(b.ticker));
      break;
  }

  return sorted;
}

// Render table with expanded rows inline
function renderTable(list) {
  let html = '';
  list.forEach((item, index) => {
    html += rowHtml(item, index);
    if (expandedTickers.has(item.ticker)) {
      html += `<tr class="expand-row" data-ticker="${item.ticker}">${createExpandedContent(item, chartRange)}</tr>`;
    }
  });
  byId('rows').innerHTML = html;
  byId('filteredCount').textContent = `(${list.length})`;

  // Draw charts for expanded rows
  expandedTickers.forEach(ticker => {
    const item = list.find(x => x.ticker === ticker);
    if (item && item.series && item.series.prices) {
      const canvas = document.getElementById(`chart-${ticker}`);
      if (canvas) {
        setTimeout(() => drawChart(canvas, item.series.dates, item.series.prices, chartRange), 30);
      }
    }
  });
}

// Full rerender
function rerender() {
  const filtered = getFilteredItems();
  const sorted = applySort(filtered);

  const q = (byId('q').value || '').trim().toUpperCase();
  const finalList = q ? sorted.filter(x => x.ticker.includes(q)) : sorted;

  renderTable(finalList);
}

// Update signal box states
function updateSignalBoxes() {
  document.querySelectorAll('.signal-box.clickable').forEach(box => {
    const signal = box.dataset.signal;
    if (signal === 'ALL') {
      const allActive = signalFilters.STRONG_BUY && signalFilters.BUY && signalFilters.WATCH;
      box.classList.toggle('active', allActive);
    } else {
      box.classList.toggle('active', signalFilters[signal]);
    }
  });
}

// Toggle row expansion
async function toggleRow(ticker) {
  if (expandedTickers.has(ticker)) {
    expandedTickers.delete(ticker);
    rerender();
  } else {
    // Load detailed data first
    const item = allItems.find(x => x.ticker === ticker);
    if (item && !item.series) {
      try {
        const detail = await loadJSON(`./data/tickers/${ticker}.json`);
        Object.assign(item, detail);
      } catch {}
    }
    expandedTickers.add(ticker);
    rerender();
  }
}

// Format date
function formatAsOf(asOf) {
  if (!asOf) return "—";
  let s = String(asOf).trim();
  if (s.includes(" ") && !s.includes("T")) s = s.replace(" ", "T");
  let d = new Date(s);
  if (Number.isNaN(d.getTime())) return String(asOf);
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

// Load JSON
async function loadJSON(url) {
  const u = withBust(url);
  const r = await fetch(u, { cache: "no-store" });
  if (!r.ok) throw new Error(`Fetch failed ${r.status}`);
  return await r.json();
}

// Main
(async function main() {
  let full;
  try {
    full = await loadJSON(DATA_URL);
  } catch (e) {
    byId('rows').innerHTML = '<tr><td colspan="12" style="text-align:center;padding:40px;color:#666;">No data available.</td></tr>';
    return;
  }

  byId('asOf').textContent = formatAsOf(full.as_of);

  if (full.summary) {
    byId('countStrong').textContent = full.summary.strong_buy || 0;
    byId('countBuy').textContent = full.summary.buy || 0;
    byId('countWatch').textContent = full.summary.watch || 0;
    byId('countAll').textContent = full.summary.total_analyzed ||
      ((full.summary.strong_buy || 0) + (full.summary.buy || 0) + (full.summary.watch || 0));
  }

  allItems = full.items || [];

  // Auto-expand top 5 and load their data
  const sorted = applySort(getFilteredItems());
  const top5 = sorted.slice(0, 5);

  for (const item of top5) {
    try {
      const detail = await loadJSON(`./data/tickers/${item.ticker}.json`);
      Object.assign(item, detail);
      expandedTickers.add(item.ticker);
    } catch {}
  }

  rerender();

  // Signal box handlers
  document.querySelectorAll('.signal-box.clickable').forEach(box => {
    box.addEventListener('click', () => {
      const signal = box.dataset.signal;
      if (signal === 'ALL') {
        signalFilters.STRONG_BUY = true;
        signalFilters.BUY = true;
        signalFilters.WATCH = true;
      } else {
        signalFilters[signal] = !signalFilters[signal];
        if (!signalFilters.STRONG_BUY && !signalFilters.BUY && !signalFilters.WATCH) {
          signalFilters[signal] = true;
        }
      }
      updateSignalBoxes();
      rerender();
    });
  });

  // Recovery filter
  byId('filterRecovery').addEventListener('change', (e) => {
    filterRecoveryOnly = e.target.checked;
    rerender();
  });

  // Sort select
  byId('sortSelect').addEventListener('change', (e) => {
    sortMode = e.target.value;
    rerender();
  });

  // Table header sorting
  document.querySelectorAll('.tbl th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const sort = th.dataset.sort;
      if (sort) {
        sortMode = sort;
        byId('sortSelect').value = sort;
        rerender();
      }
    });
  });

  // Row toggle click
  byId('rows').addEventListener('click', async (e) => {
    const btn = e.target.closest('.toggle-btn');
    if (btn) {
      e.preventDefault();
      const ticker = btn.dataset.ticker;
      await toggleRow(ticker);
      return;
    }

    // Range button click
    const rangeBtn = e.target.closest('.range-btn');
    if (rangeBtn) {
      const range = rangeBtn.dataset.range;
      const expandRow = rangeBtn.closest('.expand-row');
      if (expandRow) {
        const ticker = expandRow.dataset.ticker;
        chartRange = range;

        // Update active state
        expandRow.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
        rangeBtn.classList.add('active');

        // Redraw chart
        const item = allItems.find(x => x.ticker === ticker);
        if (item && item.series) {
          const canvas = document.getElementById(`chart-${ticker}`);
          if (canvas) drawChart(canvas, item.series.dates, item.series.prices, range);
        }
      }
    }
  });

  // Search
  byId('go').addEventListener('click', rerender);
  byId('q').addEventListener('input', rerender);

  // Resize handler
  window.addEventListener('resize', () => {
    expandedTickers.forEach(ticker => {
      const item = allItems.find(x => x.ticker === ticker);
      if (item && item.series) {
        const canvas = document.getElementById(`chart-${ticker}`);
        if (canvas) drawChart(canvas, item.series.dates, item.series.prices, chartRange);
      }
    });
  });
})();
