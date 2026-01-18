/**
 * CRT Recovery Predictor v4.0
 * Clean table-based UI with in-row expansion
 */

const DATA_URL = "./data/full.json";

let allItems = [];
let expandedTicker = null;
let signalFilters = { STRONG_BUY: true, BUY: true, WATCH: true };
let sortMode = 'score';

// Helpers
const fmtPct = x => x == null || !isFinite(x) ? '—' : `${Math.round(x)}%`;
const fmtSign = x => x == null || !isFinite(x) ? '—' : `${x > 0 ? '+' : ''}${Math.round(x)}%`;
const fmtPrice = x => x == null || !isFinite(x) ? '—' : `$${x.toFixed(2)}`;
const byId = id => document.getElementById(id);

function signalClass(s) {
  return s === 'STRONG_BUY' ? 'signal-strong' : s === 'BUY' ? 'signal-buy' : 'signal-watch';
}

function signalText(s) {
  return s === 'STRONG_BUY' ? 'STRONG BUY' : s;
}

function valClass(v, t = 0) {
  return v > t ? 'positive' : v < t ? 'negative' : '';
}

// Calculate discount zones (using neutral colors, not signal colors)
function calcZones(dates, prices) {
  const n = prices.length;
  if (n < 252) return [];

  const zones = [];
  for (let i = 252; i < n; i++) {
    let high = -Infinity;
    for (let j = i - 252; j <= i; j++) {
      if (prices[j] > high) high = prices[j];
    }
    const disc = ((high - prices[i]) / high) * 100;
    zones.push({
      date: dates[i],
      zone: disc >= 30 ? 'deep' : disc >= 20 ? 'moderate' : disc >= 10 ? 'light' : null
    });
  }
  return zones;
}

// Filter by range
function filterRange(dates, prices, range) {
  if (range === 'max') return { dates, prices };
  const n = dates.length;
  const last = new Date(dates[n - 1]);
  const years = range === '1y' ? 1 : range === '5y' ? 5 : 10;
  const cut = new Date(last);
  cut.setFullYear(cut.getFullYear() - years);
  const fd = [], fp = [];
  for (let i = 0; i < n; i++) {
    if (new Date(dates[i]) >= cut) { fd.push(dates[i]); fp.push(prices[i]); }
  }
  return { dates: fd, prices: fp };
}

// Draw chart with neutral discount zones
function drawChart(canvas, dates, prices, range) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  // Calc zones from full data, filter for display
  const fullZones = calcZones(dates, prices);
  const { dates: fd, prices: fp } = filterRange(dates, prices, range);
  const n = fp.length;

  if (n < 3) {
    ctx.fillStyle = '#999';
    ctx.textAlign = 'center';
    ctx.fillText('Not enough data', w/2, h/2);
    return;
  }

  const pad = { t: 15, r: 15, b: 25, l: 50 };
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

  let min = Infinity, max = -Infinity;
  for (const p of fp) { if (p < min) min = p; if (p > max) max = p; }
  const rng = max - min; min -= rng * 0.05; max += rng * 0.05;

  const x = i => pad.l + cw * i / (n - 1);
  const y = p => pad.t + ch * (1 - (p - min) / (max - min));

  // Draw discount zones (neutral colors - grays/blues, not green/yellow)
  if (fullZones.length > 0) {
    const zoneMap = new Map();
    fullZones.forEach(z => zoneMap.set(z.date, z.zone));
    const visibleZones = fd.map(d => zoneMap.get(d) || null);

    let cur = null, start = null;
    for (let i = 0; i < visibleZones.length; i++) {
      const z = visibleZones[i];
      if (z !== cur) {
        if (cur && start != null) {
          // Neutral colors: slate for deep, light blue for moderate, cream for light
          ctx.fillStyle = cur === 'deep' ? 'rgba(71, 85, 105, 0.2)' :
                          cur === 'moderate' ? 'rgba(148, 163, 184, 0.2)' :
                          'rgba(226, 232, 240, 0.4)';
          ctx.fillRect(x(start), pad.t, x(i) - x(start), ch);
        }
        cur = z; start = z ? i : null;
      }
    }
    if (cur && start != null) {
      ctx.fillStyle = cur === 'deep' ? 'rgba(71, 85, 105, 0.2)' :
                      cur === 'moderate' ? 'rgba(148, 163, 184, 0.2)' :
                      'rgba(226, 232, 240, 0.4)';
      ctx.fillRect(x(start), pad.t, x(n-1) - x(start), ch);
    }
  }

  // Grid
  ctx.strokeStyle = 'rgba(0,0,0,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yy = pad.t + ch * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    ctx.fillStyle = '#888'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
    ctx.fillText('$' + (max - (max - min) * i / 4).toFixed(0), pad.l - 6, yy + 3);
  }

  // Date labels
  ctx.textAlign = 'center';
  ctx.fillStyle = '#888';
  for (let i = 0; i <= 4; i++) {
    const idx = Math.floor((n - 1) * i / 4);
    const d = new Date(fd[idx]);
    ctx.fillText(d.toLocaleDateString('en', { month: 'short', year: '2-digit' }), x(idx), h - 6);
  }

  // Price line
  const up = fp[n-1] >= fp[0];
  ctx.strokeStyle = up ? '#0f7846' : '#b43232';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    i === 0 ? ctx.moveTo(x(i), y(fp[i])) : ctx.lineTo(x(i), y(fp[i]));
  }
  ctx.stroke();

  // End dot
  ctx.fillStyle = up ? '#0f7846' : '#b43232';
  ctx.beginPath();
  ctx.arc(x(n-1), y(fp[n-1]), 4, 0, Math.PI * 2);
  ctx.fill();
}

// Get filtered/sorted list
function getList() {
  let list = allItems.filter(x => signalFilters[x.signal]);
  const q = (byId('q')?.value || '').toUpperCase().trim();
  if (q) list = list.filter(x => x.ticker.includes(q));

  list.sort((a, b) => {
    switch (sortMode) {
      case 'score': return (b.opportunity_score || 0) - (a.opportunity_score || 0);
      case 'discount': return (b.discount_from_high || 0) - (a.discount_from_high || 0);
      case 'beatspy1y': return (b.prob_beat_spy_1y || 0) - (a.prob_beat_spy_1y || 0);
      case 'beatspy3y': return (b.prob_beat_spy_3y || 0) - (a.prob_beat_spy_3y || 0);
      case 'median1y': return (b.median_return_1y || 0) - (a.median_return_1y || 0);
      case 'median3y': return (b.median_return_3y || 0) - (a.median_return_3y || 0);
      case 'median5y': return (b.median_return_5y || 0) - (a.median_return_5y || 0);
      case 'prob1y': return (b.prob_positive_1y || 0) - (a.prob_positive_1y || 0);
      case 'prob3y': return (b.prob_positive_3y || 0) - (a.prob_positive_3y || 0);
      case 'prob5y': return (b.prob_positive_5y || 0) - (a.prob_positive_5y || 0);
      case 'samples': return (b.sample_size || 0) - (a.sample_size || 0);
      case 'ticker': return a.ticker.localeCompare(b.ticker);
      default: return 0;
    }
  });
  return list;
}

// Render table row
function renderRow(item) {
  const isExpanded = expandedTicker === item.ticker;
  return `
    <tr class="${isExpanded ? 'expanded' : ''}" data-ticker="${item.ticker}">
      <td class="tcell">${item.ticker}</td>
      <td><span class="signal-badge-sm ${signalClass(item.signal)}">${signalText(item.signal)}</span></td>
      <td class="num">${Math.round(item.opportunity_score || 0)}</td>
      <td class="num">${fmtPct(item.discount_from_high)}</td>
      <td class="num ${valClass(item.prob_beat_spy_1y, 50)}">${fmtPct(item.prob_beat_spy_1y)}</td>
      <td class="num ${valClass(item.median_return_1y)}">${fmtSign(item.median_return_1y)}</td>
      <td class="num ${valClass(item.prob_positive_5y, 70)}">${fmtPct(item.prob_positive_5y)}</td>
      <td class="num ${valClass(item.median_return_5y)}">${fmtSign(item.median_return_5y)}</td>
      <td class="num">${item.sample_size || '—'}</td>
    </tr>
    ${isExpanded ? renderExpanded(item) : ''}
  `;
}

// Render expanded detail row
function renderExpanded(item) {
  const range = item._chartRange || 'max';
  return `
    <tr class="expand-row">
      <td colspan="9">
        <div class="expand-content">
          <div class="expand-left">
            <div class="thesis-box">
              <div class="thesis-title">Why ${item.ticker}?</div>
              <div class="thesis-text">${item.thesis || 'Undervalued stock with recovery potential.'}</div>
            </div>
            <div class="metrics-grid">
              <div class="metric"><span class="metric-label">Price</span><span class="metric-value">${fmtPrice(item.price)}</span></div>
              <div class="metric"><span class="metric-label">52W High</span><span class="metric-value">${fmtPrice(item.high_52w)}</span></div>
              <div class="metric"><span class="metric-label">Discount</span><span class="metric-value ${valClass(item.discount_from_high, 20)}">${fmtPct(item.discount_from_high)}</span></div>
              <div class="metric"><span class="metric-label">5Y Return</span><span class="metric-value ${valClass(item.return_5y)}">${fmtSign(item.return_5y)}</span></div>
            </div>
            <div class="outlook-grid">
              <div class="outlook">
                <div class="outlook-title">1 Year</div>
                <div class="outlook-row"><span>Prob +</span><strong class="${valClass(item.prob_positive_1y, 50)}">${fmtPct(item.prob_positive_1y)}</strong></div>
                <div class="outlook-row"><span>Beat SPY</span><strong class="${valClass(item.prob_beat_spy_1y, 50)}">${fmtPct(item.prob_beat_spy_1y)}</strong></div>
                <div class="outlook-row"><span>Median</span><strong class="${valClass(item.median_return_1y)}">${fmtSign(item.median_return_1y)}</strong></div>
              </div>
              <div class="outlook">
                <div class="outlook-title">3 Year</div>
                <div class="outlook-row"><span>Prob +</span><strong class="${valClass(item.prob_positive_3y, 50)}">${fmtPct(item.prob_positive_3y)}</strong></div>
                <div class="outlook-row"><span>Beat SPY</span><strong class="${valClass(item.prob_beat_spy_3y, 50)}">${fmtPct(item.prob_beat_spy_3y)}</strong></div>
                <div class="outlook-row"><span>Median</span><strong class="${valClass(item.median_return_3y)}">${fmtSign(item.median_return_3y)}</strong></div>
              </div>
              <div class="outlook">
                <div class="outlook-title">5 Year</div>
                <div class="outlook-row"><span>Prob +</span><strong class="${valClass(item.prob_positive_5y, 70)}">${fmtPct(item.prob_positive_5y)}</strong></div>
                <div class="outlook-row"><span>Median</span><strong class="${valClass(item.median_return_5y)}">${fmtSign(item.median_return_5y)}</strong></div>
              </div>
            </div>
          </div>
          <div class="expand-right">
            <div class="chart-header">
              <span class="chart-title">Price History</span>
              <div class="range-btns">
                <button class="range-btn ${range === '1y' ? 'active' : ''}" data-range="1y">1Y</button>
                <button class="range-btn ${range === '5y' ? 'active' : ''}" data-range="5y">5Y</button>
                <button class="range-btn ${range === '10y' ? 'active' : ''}" data-range="10y">10Y</button>
                <button class="range-btn ${range === 'max' ? 'active' : ''}" data-range="max">Max</button>
              </div>
            </div>
            <canvas class="expand-chart" id="chart-${item.ticker}"></canvas>
            <div class="chart-legend">
              <span><i class="leg-deep"></i>30%+ off high</span>
              <span><i class="leg-mod"></i>20-30% off</span>
              <span><i class="leg-light"></i>10-20% off</span>
            </div>
          </div>
        </div>
      </td>
    </tr>
  `;
}

// Render all
function render() {
  const list = getList();
  byId('rows').innerHTML = list.map(renderRow).join('');
  byId('filteredCount').textContent = `(${list.length})`;

  // Draw chart if expanded
  if (expandedTicker) {
    const item = allItems.find(x => x.ticker === expandedTicker);
    if (item?.series?.prices) {
      const canvas = byId(`chart-${item.ticker}`);
      if (canvas) {
        setTimeout(() => drawChart(canvas, item.series.dates, item.series.prices, item._chartRange || 'max'), 10);
      }
    }
  }
}

// Update counts
function updateCounts() {
  const counts = { STRONG_BUY: 0, BUY: 0, WATCH: 0 };
  allItems.forEach(x => counts[x.signal]++);
  byId('countStrong').textContent = counts.STRONG_BUY;
  byId('countBuy').textContent = counts.BUY;
  byId('countWatch').textContent = counts.WATCH;
  byId('countAll').textContent = allItems.length;
}

// Load ticker data
async function loadTickerData(ticker) {
  const item = allItems.find(x => x.ticker === ticker);
  if (!item || item.series) return item;

  try {
    const resp = await fetch(`./data/tickers/${ticker}.json`);
    if (resp.ok) {
      const data = await resp.json();
      Object.assign(item, data);
    }
  } catch (e) {
    console.warn(`Failed to load ${ticker}:`, e);
  }
  return item;
}

// Initialize
async function init() {
  try {
    const resp = await fetch(DATA_URL);
    const data = await resp.json();
    allItems = data.items || [];

    if (data.as_of) {
      byId('asOf').textContent = new Date(data.as_of).toLocaleDateString('en-US', {
        weekday: 'short', month: 'short', day: 'numeric', year: 'numeric'
      });
    }

    updateCounts();
    render();
  } catch (e) {
    console.error('Failed to load data:', e);
  }
}

// Event handlers
document.addEventListener('DOMContentLoaded', () => {
  init();

  // Signal filter clicks
  byId('signalSummary').addEventListener('click', e => {
    const box = e.target.closest('.signal-box');
    if (!box) return;

    const signal = box.dataset.signal;
    if (signal === 'ALL') {
      signalFilters = { STRONG_BUY: true, BUY: true, WATCH: true };
      document.querySelectorAll('.signal-box').forEach(b => b.classList.add('active'));
    } else {
      box.classList.toggle('active');
      signalFilters[signal] = box.classList.contains('active');
    }
    render();
  });

  // Sort change
  byId('sortSelect').addEventListener('change', e => {
    sortMode = e.target.value;
    render();
  });

  // Search
  byId('q').addEventListener('input', () => render());
  byId('go').addEventListener('click', () => render());

  // Table row click - expand in place
  byId('rows').addEventListener('click', async e => {
    // Range button click
    const rangeBtn = e.target.closest('.range-btn');
    if (rangeBtn) {
      const item = allItems.find(x => x.ticker === expandedTicker);
      if (item) {
        item._chartRange = rangeBtn.dataset.range;
        render();
      }
      return;
    }

    // Row click - toggle expand
    const row = e.target.closest('tr[data-ticker]');
    if (!row) return;

    const ticker = row.dataset.ticker;
    if (expandedTicker === ticker) {
      expandedTicker = null;
    } else {
      expandedTicker = ticker;
      await loadTickerData(ticker);
    }
    render();
  });

  // Resize
  window.addEventListener('resize', () => {
    if (expandedTicker) render();
  });
});
