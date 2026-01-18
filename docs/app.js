/**
 * Stock Recovery Predictor - WSJ Style
 */

const DATA_URL = "./data/full.json";

let allItems = [];
let openItems = new Set();
let filters = { STRONG_BUY: true, BUY: true, WATCH: true };
let sortMode = 'score';
let chartRanges = {}; // Track range per ticker

// Helpers
const $ = id => document.getElementById(id);
const fmtPct = x => x == null || !isFinite(x) ? '—' : `${Math.round(x)}%`;
const fmtSign = x => x == null || !isFinite(x) ? '—' : `${x > 0 ? '+' : ''}${Math.round(x)}%`;
const fmtPrice = x => x == null || !isFinite(x) ? '—' : `$${x.toFixed(2)}`;
const valClass = (v, t = 0) => v > t ? 'pos' : v < t ? 'neg' : '';

// Calculate buy zones from FULL price history
// Returns zones array with corresponding dates for proper range filtering
function calcZones(dates, prices) {
  const n = prices.length;
  if (n < 252) return null;
  const zones = [];
  for (let i = 252; i < n; i++) {
    let high = -Infinity;
    for (let j = i - 252; j <= i; j++) if (prices[j] > high) high = prices[j];
    const disc = ((high - prices[i]) / high) * 100;
    zones.push({
      date: dates[i],
      zone: disc >= 30 ? 'strong' : disc >= 20 ? 'buy' : disc >= 10 ? 'watch' : null
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

// Draw chart
function drawChart(canvas, dates, prices, range) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  // Calculate zones from FULL data first (before filtering)
  const fullZones = calcZones(dates, prices);

  // Now filter data for display
  const { dates: fd, prices: fp } = filterRange(dates, prices, range);
  const n = fp.length;
  if (n < 3) {
    ctx.fillStyle = '#999';
    ctx.textAlign = 'center';
    ctx.fillText('No data', w/2, h/2);
    return;
  }

  const pad = { t: 10, r: 10, b: 20, l: 40 };
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
  let min = Infinity, max = -Infinity;
  for (const p of fp) { if (p < min) min = p; if (p > max) max = p; }
  const rng = max - min; min -= rng * 0.05; max += rng * 0.05;

  const x = i => pad.l + cw * i / (n - 1);
  const y = p => pad.t + ch * (1 - (p - min) / (max - min));

  // Filter zones to match visible date range and draw
  if (fullZones) {
    // Create a map of date -> zone for quick lookup
    const zoneMap = new Map();
    fullZones.forEach(z => zoneMap.set(z.date, z.zone));

    // Build zones array for visible dates
    const visibleZones = fd.map(d => zoneMap.get(d) || null);

    let cur = null, start = null;
    for (let i = 0; i < visibleZones.length; i++) {
      const z = visibleZones[i];
      if (z !== cur) {
        if (cur && start != null) {
          ctx.fillStyle = cur === 'strong' ? 'rgba(10,102,64,0.3)' : cur === 'buy' ? 'rgba(10,102,64,0.15)' : 'rgba(200,150,0,0.2)';
          ctx.fillRect(x(start), pad.t, x(i) - x(start), ch);
        }
        cur = z; start = z ? i : null;
      }
    }
    if (cur && start != null) {
      ctx.fillStyle = cur === 'strong' ? 'rgba(10,102,64,0.3)' : cur === 'buy' ? 'rgba(10,102,64,0.15)' : 'rgba(200,150,0,0.2)';
      ctx.fillRect(x(start), pad.t, x(n-1) - x(start), ch);
    }
  }

  // Grid
  ctx.strokeStyle = '#e0e0e0';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const yy = pad.t + ch * i / 3;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    ctx.fillStyle = '#999'; ctx.font = '9px system-ui'; ctx.textAlign = 'right';
    ctx.fillText('$' + (max - (max - min) * i / 3).toFixed(0), pad.l - 4, yy + 3);
  }

  // Dates
  ctx.textAlign = 'center';
  for (let i = 0; i <= 3; i++) {
    const idx = Math.floor((n - 1) * i / 3);
    const d = new Date(fd[idx]);
    ctx.fillText(d.toLocaleDateString('en', { month: 'short', year: '2-digit' }), x(idx), h - 4);
  }

  // Line
  const up = fp[n-1] > fp[0];
  ctx.strokeStyle = up ? '#0a6640' : '#b00020';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const px = x(i), py = y(fp[i]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Dot
  ctx.fillStyle = up ? '#0a6640' : '#b00020';
  ctx.beginPath();
  ctx.arc(x(n-1), y(fp[n-1]), 3, 0, Math.PI * 2);
  ctx.fill();
}

// Render item
function renderItem(item, idx) {
  const isOpen = openItems.has(item.ticker);
  const range = chartRanges[item.ticker] || 'max';
  const sigCls = item.signal === 'STRONG_BUY' ? 'signal-strong' : item.signal === 'BUY' ? 'signal-buy' : 'signal-watch';
  const sigTxt = item.signal === 'STRONG_BUY' ? 'Strong Buy' : item.signal === 'BUY' ? 'Buy' : 'Watch';
  const brief = `${fmtPct(item.discount_from_high)} off high · ${fmtPct(item.prob_beat_spy_1y)} beat SPY`;

  return `
    <div class="stock-item${isOpen ? ' open' : ''}" data-ticker="${item.ticker}">
      <div class="stock-row">
        <div class="toggle">${isOpen ? '−' : '+'}</div>
        <div class="stock-info">
          <div class="stock-header">
            <span class="ticker">${item.ticker}</span>
            <span class="signal ${sigCls}">${sigTxt}</span>
          </div>
          <div class="stock-brief">${brief}</div>
        </div>
        <div class="stock-score">
          <div class="score-num">${Math.round(item.opportunity_score || 0)}</div>
          <div class="score-label">Score</div>
        </div>
      </div>
      <div class="stock-expand">
        <div class="thesis">${item.thesis || 'Undervalued stock showing recovery potential based on historical patterns.'}</div>
        <div class="chart-section">
          <div class="chart-header">
            <span class="chart-title">Price History</span>
            <div class="range-btns">
              <button class="range-btn${range === '1y' ? ' active' : ''}" data-range="1y">1Y</button>
              <button class="range-btn${range === '5y' ? ' active' : ''}" data-range="5y">5Y</button>
              <button class="range-btn${range === '10y' ? ' active' : ''}" data-range="10y">10Y</button>
              <button class="range-btn${range === 'max' ? ' active' : ''}" data-range="max">Max</button>
            </div>
          </div>
          <div class="chart-wrap">
            <canvas class="chart-canvas" id="chart-${item.ticker}"></canvas>
          </div>
          <div class="chart-legend">
            <span><i class="leg-strong"></i>Strong Buy</span>
            <span><i class="leg-buy"></i>Buy</span>
            <span><i class="leg-watch"></i>Watch</span>
          </div>
        </div>
        <div class="stats">
          <div class="stat-group">
            <div class="stat-group-title">Current</div>
            <div class="stat-line"><span>Price</span><b>${fmtPrice(item.price)}</b></div>
            <div class="stat-line"><span>52W High</span><b>${fmtPrice(item.high_52w)}</b></div>
            <div class="stat-line"><span>Discount</span><b class="pos">${fmtPct(item.discount_from_high)}</b></div>
          </div>
          <div class="stat-group">
            <div class="stat-group-title">1 Year</div>
            <div class="stat-line"><span>Prob +</span><b class="${valClass(item.prob_positive_1y, 50)}">${fmtPct(item.prob_positive_1y)}</b></div>
            <div class="stat-line"><span>Beat SPY</span><b class="${valClass(item.prob_beat_spy_1y, 50)}">${fmtPct(item.prob_beat_spy_1y)}</b></div>
            <div class="stat-line"><span>Median</span><b class="${valClass(item.median_return_1y)}">${fmtSign(item.median_return_1y)}</b></div>
          </div>
          <div class="stat-group">
            <div class="stat-group-title">3 Year</div>
            <div class="stat-line"><span>Prob +</span><b class="${valClass(item.prob_positive_3y, 50)}">${fmtPct(item.prob_positive_3y)}</b></div>
            <div class="stat-line"><span>Beat SPY</span><b class="${valClass(item.prob_beat_spy_3y, 50)}">${fmtPct(item.prob_beat_spy_3y)}</b></div>
            <div class="stat-line"><span>Median</span><b class="${valClass(item.median_return_3y)}">${fmtSign(item.median_return_3y)}</b></div>
          </div>
          <div class="stat-group">
            <div class="stat-group-title">5 Year</div>
            <div class="stat-line"><span>Prob +</span><b class="${valClass(item.prob_positive_5y, 50)}">${fmtPct(item.prob_positive_5y)}</b></div>
            <div class="stat-line"><span>Median</span><b class="${valClass(item.median_return_5y)}">${fmtSign(item.median_return_5y)}</b></div>
          </div>
        </div>
      </div>
    </div>
  `;
}

// Get sorted/filtered list
function getList() {
  let list = allItems.filter(x => filters[x.signal]);
  const q = ($('search').value || '').toUpperCase().trim();
  if (q) list = list.filter(x => x.ticker.includes(q));

  list.sort((a, b) => {
    switch (sortMode) {
      case 'score': return (b.opportunity_score || 0) - (a.opportunity_score || 0);
      case 'discount': return (b.discount_from_high || 0) - (a.discount_from_high || 0);
      case 'beatspy1y': return (b.prob_beat_spy_1y || 0) - (a.prob_beat_spy_1y || 0);
      case 'median1y': return (b.median_return_1y || 0) - (a.median_return_1y || 0);
      case 'prob5y': return (b.prob_positive_5y || 0) - (a.prob_positive_5y || 0);
      default: return 0;
    }
  });
  return list;
}

// Render all
function render() {
  const list = getList();
  $('stockList').innerHTML = list.map((it, i) => renderItem(it, i)).join('');

  // Draw charts for open items
  openItems.forEach(ticker => {
    const item = allItems.find(x => x.ticker === ticker);
    if (item?.series?.prices) {
      const canvas = $(`chart-${ticker}`);
      if (canvas) setTimeout(() => drawChart(canvas, item.series.dates, item.series.prices, chartRanges[ticker] || 'max'), 20);
    }
  });
}

// Update filter pills
function updatePills() {
  document.querySelectorAll('.pill').forEach(p => {
    const f = p.dataset.filter;
    if (f === 'all') {
      p.classList.toggle('active', filters.STRONG_BUY && filters.BUY && filters.WATCH);
    } else {
      p.classList.toggle('active', filters[f]);
    }
  });
}

// Toggle item
async function toggle(ticker) {
  if (openItems.has(ticker)) {
    openItems.delete(ticker);
  } else {
    const item = allItems.find(x => x.ticker === ticker);
    if (item && !item.series) {
      try {
        const r = await fetch(`./data/tickers/${ticker}.json?v=${Date.now()}`);
        if (r.ok) Object.assign(item, await r.json());
      } catch {}
    }
    openItems.add(ticker);
  }
  render();
}

// Init
(async function init() {
  console.log('Stock Recovery Predictor: Starting...');
  try {
    console.log('Fetching data from:', DATA_URL);
    const r = await fetch(`${DATA_URL}?v=${Date.now()}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
    const data = await r.json();
    console.log('Data loaded:', data.items?.length || 0, 'items');

    $('asOf').textContent = new Date(data.as_of).toLocaleDateString('en', { month: 'short', day: 'numeric', year: 'numeric' });
    $('countStrong').textContent = data.summary?.strong_buy || 0;
    $('countBuy').textContent = data.summary?.buy || 0;
    $('countWatch').textContent = data.summary?.watch || 0;
    $('countAll').textContent = (data.summary?.strong_buy || 0) + (data.summary?.buy || 0) + (data.summary?.watch || 0);

    allItems = data.items || [];

    // Open top 5
    const list = getList().slice(0, 5);
    for (const item of list) {
      try {
        const r = await fetch(`./data/tickers/${item.ticker}.json?v=${Date.now()}`);
        if (r.ok) Object.assign(item, await r.json());
        openItems.add(item.ticker);
      } catch {}
    }

    render();
    console.log('Stock Recovery Predictor: Ready');

    // Event handlers
    $('filterPills').addEventListener('click', e => {
      const pill = e.target.closest('.pill');
      if (!pill) return;
      const f = pill.dataset.filter;
      if (f === 'all') {
        filters.STRONG_BUY = filters.BUY = filters.WATCH = true;
      } else {
        filters[f] = !filters[f];
        if (!filters.STRONG_BUY && !filters.BUY && !filters.WATCH) filters[f] = true;
      }
      updatePills();
      render();
    });

    $('search').addEventListener('input', render);
    $('sortSelect').addEventListener('change', e => { sortMode = e.target.value; render(); });

    $('stockList').addEventListener('click', e => {
      const row = e.target.closest('.stock-row');
      if (row) {
        const ticker = row.closest('.stock-item').dataset.ticker;
        toggle(ticker);
        return;
      }

      const rb = e.target.closest('.range-btn');
      if (rb) {
        const ticker = rb.closest('.stock-item').dataset.ticker;
        chartRanges[ticker] = rb.dataset.range;
        rb.closest('.range-btns').querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
        rb.classList.add('active');
        const item = allItems.find(x => x.ticker === ticker);
        if (item?.series) drawChart($(`chart-${ticker}`), item.series.dates, item.series.prices, chartRanges[ticker]);
      }
    });

    window.addEventListener('resize', () => {
      openItems.forEach(ticker => {
        const item = allItems.find(x => x.ticker === ticker);
        if (item?.series) {
          const canvas = $(`chart-${ticker}`);
          if (canvas) drawChart(canvas, item.series.dates, item.series.prices, chartRanges[ticker] || 'max');
        }
      });
    });

  } catch (e) {
    console.error('Init error:', e);
    $('stockList').innerHTML = `<p style="text-align:center;padding:40px;color:#666">Unable to load data: ${e.message}</p>`;
  }
})();
