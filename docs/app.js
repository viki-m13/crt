/**
 * CRT Recovery Predictor v3.0 - Frontend JavaScript
 * Professional stock analysis with historical buy zone visualization
 */

const DATA_URL = "./data/full.json";
const CACHE_BUST = String(Date.now());

// State
let allItems = [];
let currentItem = null;
let chartRange = 'max'; // Default range - show full history
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
  if (n < 252) return null; // Need at least 1 year

  const zones = [];
  const lookback = 252; // ~1 year of trading days

  for (let i = lookback; i < n; i++) {
    // Find 52-week high up to this point
    let high52w = -Infinity;
    for (let j = i - lookback; j <= i; j++) {
      if (prices[j] > high52w) high52w = prices[j];
    }

    const price = prices[i];
    const discount = ((high52w - price) / high52w) * 100;

    // Determine zone
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

  if (range === '5y') {
    cutoffDate = new Date(lastDate);
    cutoffDate.setFullYear(cutoffDate.getFullYear() - 5);
  } else if (range === '10y') {
    cutoffDate = new Date(lastDate);
    cutoffDate.setFullYear(cutoffDate.getFullYear() - 10);
  } else {
    // Max - return all
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

// Draw enhanced chart with buy zones
function drawChart(canvas, dates, prices, range = '10y') {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  // Filter by range
  const filtered = filterByRange(dates, prices, range);
  const filteredDates = filtered.dates;
  const filteredPrices = filtered.prices;

  const n = filteredPrices.length;
  if (n < 3) {
    ctx.font = "14px IBM Plex Mono";
    ctx.fillStyle = "#666";
    ctx.textAlign = "center";
    ctx.fillText("Not enough data for this range", w / 2, h / 2);
    return;
  }

  const pad = { top: 20, right: 20, bottom: 30, left: 60 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  // Find min/max
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

  // Calculate buy zones for the filtered data
  const zones = calculateBuyZones(filteredDates, filteredPrices);

  // Draw buy zone backgrounds
  if (zones && zones.length > 0) {
    const zoneStartIdx = filteredDates.length - zones.length;

    // Group consecutive zones
    let currentZone = null;
    let zoneStart = null;

    for (let i = 0; i < zones.length; i++) {
      const z = zones[i];
      const idx = zoneStartIdx + i;

      if (z.zone !== currentZone) {
        // Draw previous zone
        if (currentZone && zoneStart !== null) {
          const x1 = xAt(zoneStart);
          const x2 = xAt(idx);

          ctx.fillStyle = currentZone === 'strong' ? 'rgba(6, 78, 59, 0.25)' :
                          currentZone === 'buy' ? 'rgba(15, 120, 70, 0.15)' :
                          'rgba(251, 191, 36, 0.15)';
          ctx.fillRect(x1, pad.top, x2 - x1, chartH);
        }

        currentZone = z.zone;
        zoneStart = z.zone ? idx : null;
      }
    }

    // Draw final zone
    if (currentZone && zoneStart !== null) {
      const x1 = xAt(zoneStart);
      const x2 = xAt(n - 1);

      ctx.fillStyle = currentZone === 'strong' ? 'rgba(6, 78, 59, 0.25)' :
                      currentZone === 'buy' ? 'rgba(15, 120, 70, 0.15)' :
                      'rgba(251, 191, 36, 0.15)';
      ctx.fillRect(x1, pad.top, x2 - x1, chartH);
    }
  }

  // Draw grid lines
  ctx.strokeStyle = 'rgba(0,0,0,0.08)';
  ctx.lineWidth = 1;
  const gridLines = 5;
  for (let i = 0; i <= gridLines; i++) {
    const y = pad.top + (chartH / gridLines) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    // Price labels
    const price = maxP - ((maxP - minP) / gridLines) * i;
    ctx.fillStyle = '#666';
    ctx.font = "10px IBM Plex Mono";
    ctx.textAlign = 'right';
    ctx.fillText(`$${price.toFixed(0)}`, pad.left - 8, y + 3);
  }

  // Draw date labels
  const dateLabels = 5;
  ctx.textAlign = 'center';
  for (let i = 0; i <= dateLabels; i++) {
    const idx = Math.floor((n - 1) * i / dateLabels);
    const x = xAt(idx);
    const date = new Date(filteredDates[idx]);
    const label = date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
    ctx.fillText(label, x, h - 8);
  }

  // Draw the price line
  const firstPrice = filteredPrices[0];
  const lastPrice = filteredPrices[n - 1];
  const isUp = lastPrice > firstPrice;

  ctx.strokeStyle = isUp ? 'rgba(15, 120, 70, 0.9)' : 'rgba(180, 50, 50, 0.9)';
  ctx.lineWidth = 2;
  ctx.beginPath();

  for (let i = 0; i < n; i++) {
    const p = Number(filteredPrices[i]);
    if (!Number.isFinite(p)) continue;
    const x = xAt(i), y = yAt(p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw current price marker
  const lastX = xAt(n - 1);
  const lastY = yAt(lastPrice);

  ctx.fillStyle = isUp ? 'rgba(15, 120, 70, 1)' : 'rgba(180, 50, 50, 1)';
  ctx.beginPath();
  ctx.arc(lastX, lastY, 6, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Current price label
  ctx.fillStyle = '#111';
  ctx.font = "bold 11px IBM Plex Mono";
  ctx.textAlign = 'left';
  ctx.fillText(`$${lastPrice.toFixed(2)}`, lastX + 10, lastY + 4);
}

// Render detail panel
function renderDetail(item) {
  currentItem = item;

  const panel = byId('detailPanel');
  panel.classList.remove('hidden');

  // Header
  byId('detailTicker').textContent = item.ticker;
  const signalEl = byId('detailSignal');
  signalEl.textContent = signalText(item.signal);
  signalEl.className = 'detail-signal ' + signalClass(item.signal);
  byId('detailPrice').textContent = fmtPrice(item.price);

  // Thesis
  byId('detailThesis').innerHTML = `
    <div class="detail-thesis-title">Why Buy ${item.ticker} Now?</div>
    <div class="detail-thesis-text">${item.thesis || 'No thesis available.'}</div>
  `;

  // Metrics
  const metricsHtml = `
    <div class="detail-metric">
      <div class="detail-metric-label">Opportunity Score</div>
      <div class="detail-metric-value">${fmtNum0(item.opportunity_score)}/100</div>
    </div>
    <div class="detail-metric">
      <div class="detail-metric-label">Discount from High</div>
      <div class="detail-metric-value positive">${fmtPct(item.discount_from_high)}</div>
    </div>
    <div class="detail-metric">
      <div class="detail-metric-label">52-Week High</div>
      <div class="detail-metric-value">${fmtPrice(item.high_52w)}</div>
    </div>
    <div class="detail-metric">
      <div class="detail-metric-label">52-Week Low</div>
      <div class="detail-metric-value">${fmtPrice(item.low_52w)}</div>
    </div>
    <div class="detail-metric">
      <div class="detail-metric-label">5-Year Return</div>
      <div class="detail-metric-value ${valueClass(item.five_year_return)}">${fmtSignedPct(item.five_year_return)}</div>
    </div>
    <div class="detail-metric">
      <div class="detail-metric-label">Monthly Win Rate</div>
      <div class="detail-metric-value">${fmtPct(item.monthly_win_rate)}</div>
    </div>
    <div class="detail-metric">
      <div class="detail-metric-label">Above SMA-20</div>
      <div class="detail-metric-value ${item.above_sma20 ? 'positive' : ''}">${item.above_sma20 ? 'Yes' : 'No'}</div>
    </div>
    <div class="detail-metric">
      <div class="detail-metric-label">Early Recovery</div>
      <div class="detail-metric-value ${item.early_recovery ? 'positive' : ''}">${item.early_recovery ? 'Yes' : 'No'}</div>
    </div>
  `;
  byId('detailMetrics').innerHTML = metricsHtml;

  // Outcomes
  const outcomesHtml = `
    <div class="detail-outcome">
      <div class="detail-outcome-title">1 Year Outlook</div>
      <div class="detail-outcome-row"><span>Prob Positive</span><strong class="${valueClass(item.prob_positive_1y, 50)}">${fmtPct(item.prob_positive_1y)}</strong></div>
      <div class="detail-outcome-row"><span>Beat SPY</span><strong class="${valueClass(item.prob_beat_spy_1y, 50)}">${fmtPct(item.prob_beat_spy_1y)}</strong></div>
      <div class="detail-outcome-row"><span>Median Return</span><strong class="${valueClass(item.median_return_1y)}">${fmtSignedPct(item.median_return_1y)}</strong></div>
      <div class="detail-outcome-row"><span>Downside (10th)</span><strong class="negative">${fmtSignedPct(item.downside_1y)}</strong></div>
      <div class="detail-outcome-row"><span>Upside (90th)</span><strong class="positive">${fmtSignedPct(item.upside_1y)}</strong></div>
      <div class="detail-outcome-row"><span>Sample Size</span><strong>${item.sample_size_1y || '—'}</strong></div>
    </div>
    <div class="detail-outcome">
      <div class="detail-outcome-title">3 Year Outlook</div>
      <div class="detail-outcome-row"><span>Prob Positive</span><strong class="${valueClass(item.prob_positive_3y, 50)}">${fmtPct(item.prob_positive_3y)}</strong></div>
      <div class="detail-outcome-row"><span>Beat SPY</span><strong class="${valueClass(item.prob_beat_spy_3y, 50)}">${fmtPct(item.prob_beat_spy_3y)}</strong></div>
      <div class="detail-outcome-row"><span>Median Return</span><strong class="${valueClass(item.median_return_3y)}">${fmtSignedPct(item.median_return_3y)}</strong></div>
      <div class="detail-outcome-row"><span>Sample Size</span><strong>${item.sample_size_3y || '—'}</strong></div>
    </div>
    <div class="detail-outcome">
      <div class="detail-outcome-title">5 Year Outlook</div>
      <div class="detail-outcome-row"><span>Prob Positive</span><strong class="${valueClass(item.prob_positive_5y, 50)}">${fmtPct(item.prob_positive_5y)}</strong></div>
      <div class="detail-outcome-row"><span>Median Return</span><strong class="${valueClass(item.median_return_5y)}">${fmtSignedPct(item.median_return_5y)}</strong></div>
      <div class="detail-outcome-row"><span>Sample Size</span><strong>${item.sample_size_5y || '—'}</strong></div>
    </div>
  `;
  byId('detailOutcomes').innerHTML = outcomesHtml;

  // Draw chart
  const series = item.series;
  if (series && series.prices && series.prices.length > 0) {
    const canvas = byId('detailCanvas');
    drawChart(canvas, series.dates, series.prices, chartRange);
  }

  // Scroll to detail
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Close detail panel
function closeDetail() {
  byId('detailPanel').classList.add('hidden');
  currentItem = null;
  document.querySelectorAll('#rows tr').forEach(r => r.classList.remove('selected'));
}

// Table row HTML
function rowHtml(item) {
  const thesisShort = item.thesis ?
    (item.thesis.length > 60 ? item.thesis.substring(0, 60) + '...' : item.thesis) : '';

  return `
    <tr data-ticker="${item.ticker}">
      <td class="tcell">${item.ticker}</td>
      <td><span class="signal-badge-sm ${signalClass(item.signal)}">${signalText(item.signal)}</span></td>
      <td class="num">${fmtNum0(item.opportunity_score)}</td>
      <td class="num">${fmtPct(item.discount_from_high)}</td>
      <td class="num ${valueClass(item.prob_beat_spy_1y, 50)}">${fmtPct(item.prob_beat_spy_1y)}</td>
      <td class="num ${valueClass(item.median_return_1y)}">${fmtSignedPct(item.median_return_1y)}</td>
      <td class="num ${valueClass(item.median_return_3y)}">${fmtSignedPct(item.median_return_3y)}</td>
      <td class="num ${valueClass(item.median_return_5y)}">${fmtSignedPct(item.median_return_5y)}</td>
      <td class="num">${fmtNum0(item.sample_size_1y)}</td>
      <td><span class="${item.early_recovery ? 'recovery-yes' : 'recovery-no'}">${item.early_recovery ? 'YES' : '—'}</span></td>
      <td class="thesis-cell">${thesisShort}</td>
    </tr>
  `;
}

// Get filtered items
function getFilteredItems() {
  let list = [...allItems];

  // Signal filters
  list = list.filter(x => signalFilters[x.signal]);

  // Recovery filter
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

// Render table
function renderTable(list) {
  byId('rows').innerHTML = list.map(it => rowHtml(it)).join('');
  byId('filteredCount').textContent = `(${list.length})`;
}

// Full rerender
function rerender() {
  const filtered = getFilteredItems();
  const sorted = applySort(filtered);
  renderTable(sorted);

  // Apply search if any
  const q = (byId('q').value || '').trim().toUpperCase();
  if (q) {
    const searched = sorted.filter(x => x.ticker.includes(q));
    renderTable(searched);
  }
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

// Format date
function formatAsOf(asOf) {
  if (!asOf) return "—";
  let s = String(asOf).trim();
  if (s.includes(" ") && !s.includes("T")) s = s.replace(" ", "T");
  let d = new Date(s);
  if (Number.isNaN(d.getTime())) return String(asOf);

  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short'
  });
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
    byId('rows').innerHTML = '<tr><td colspan="11" style="text-align:center;padding:40px;color:#666;">No data available. Run the daily scan to generate data.</td></tr>';
    return;
  }

  byId('asOf').textContent = formatAsOf(full.as_of);

  // Update counts
  if (full.summary) {
    byId('countStrong').textContent = full.summary.strong_buy || 0;
    byId('countBuy').textContent = full.summary.buy || 0;
    byId('countWatch').textContent = full.summary.watch || 0;
    byId('countAll').textContent = full.summary.total_analyzed ||
      ((full.summary.strong_buy || 0) + (full.summary.buy || 0) + (full.summary.watch || 0));
  }

  allItems = full.items || [];

  // Initial render
  rerender();

  // Signal box click handlers
  document.querySelectorAll('.signal-box.clickable').forEach(box => {
    box.addEventListener('click', () => {
      const signal = box.dataset.signal;

      if (signal === 'ALL') {
        // Toggle all on
        signalFilters.STRONG_BUY = true;
        signalFilters.BUY = true;
        signalFilters.WATCH = true;
      } else {
        // Toggle individual
        signalFilters[signal] = !signalFilters[signal];

        // Ensure at least one is selected
        if (!signalFilters.STRONG_BUY && !signalFilters.BUY && !signalFilters.WATCH) {
          signalFilters[signal] = true;
        }
      }

      updateSignalBoxes();
      closeDetail();
      rerender();
    });
  });

  // Recovery filter
  byId('filterRecovery').addEventListener('change', (e) => {
    filterRecoveryOnly = e.target.checked;
    closeDetail();
    rerender();
  });

  // Sort select
  byId('sortSelect').addEventListener('change', (e) => {
    sortMode = e.target.value;
    rerender();
  });

  // Table header click sorting
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

  // Table row click
  byId('rows').addEventListener('click', async (e) => {
    const tr = e.target.closest('tr');
    if (!tr) return;

    const ticker = tr.dataset.ticker;
    if (!ticker) return;

    // Update selection
    document.querySelectorAll('#rows tr').forEach(r => r.classList.remove('selected'));
    tr.classList.add('selected');

    // Find item and render detail
    const item = allItems.find(x => x.ticker === ticker);
    if (item) {
      // Try to load detailed data
      try {
        const detail = await loadJSON(`./data/tickers/${ticker}.json`);
        renderDetail({ ...item, ...detail });
      } catch {
        renderDetail(item);
      }
    }
  });

  // Close detail button
  byId('detailClose').addEventListener('click', closeDetail);

  // Chart range buttons
  document.querySelectorAll('.chart-range-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.chart-range-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      chartRange = btn.dataset.range;

      if (currentItem && currentItem.series) {
        const canvas = byId('detailCanvas');
        drawChart(canvas, currentItem.series.dates, currentItem.series.prices, chartRange);
      }
    });
  });

  // Search
  function applySearch() {
    const q = (byId('q').value || '').trim().toUpperCase();
    const filtered = getFilteredItems();
    const sorted = applySort(filtered);

    if (!q) {
      renderTable(sorted);
      return;
    }

    const searched = sorted.filter(x => x.ticker.includes(q));
    renderTable(searched);
  }

  byId('go').addEventListener('click', applySearch);
  byId('q').addEventListener('input', applySearch);

  // Escape key closes detail
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDetail();
  });

  // Window resize - redraw chart
  window.addEventListener('resize', () => {
    if (currentItem && currentItem.series) {
      const canvas = byId('detailCanvas');
      drawChart(canvas, currentItem.series.dates, currentItem.series.prices, chartRange);
    }
  });
})();
