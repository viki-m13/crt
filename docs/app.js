/**
 * CRT Recovery Predictor v3.0 - Simple Frontend
 */

const DATA_URL = "./data/full.json";

let allStocks = [];

// Helpers
function $(id) { return document.getElementById(id); }
function fmtPct(x) { return x != null ? `${Math.round(x)}%` : '—'; }
function fmtReturn(x) {
  if (x == null) return '—';
  const sign = x > 0 ? '+' : '';
  return `${sign}${Math.round(x)}%`;
}

// Load data
async function loadData() {
  try {
    const res = await fetch(DATA_URL + '?v=' + Date.now());
    const data = await res.json();

    // Update header
    if (data.as_of) {
      const date = new Date(data.as_of);
      $('asOf').textContent = date.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric'
      });
    }

    // Update counts
    $('countStrong').textContent = data.summary?.strong_buy || 0;
    $('countBuy').textContent = data.summary?.buy || 0;
    $('countWatch').textContent = data.summary?.watch || 0;

    // Store and render stocks
    allStocks = data.items || [];
    renderStocks(allStocks);

  } catch (err) {
    console.error('Failed to load data:', err);
    $('stockList').innerHTML = '<div class="loading">Failed to load data. Please try again later.</div>';
  }
}

// Render stock cards
function renderStocks(stocks) {
  const container = $('stockList');

  if (stocks.length === 0) {
    container.innerHTML = '<div class="loading">No stocks match your search.</div>';
    return;
  }

  container.innerHTML = stocks.map(stock => {
    const signalClass = stock.signal === 'STRONG_BUY' ? 'strong' :
                        stock.signal === 'BUY' ? 'buy' : 'watch';
    const signalText = stock.signal === 'STRONG_BUY' ? 'Strong Buy' :
                       stock.signal === 'BUY' ? 'Buy' : 'Watch';

    // Truncate thesis for card
    const shortThesis = stock.thesis ?
      (stock.thesis.length > 120 ? stock.thesis.substring(0, 120) + '...' : stock.thesis) : '';

    return `
      <div class="stock-card ${signalClass}" data-ticker="${stock.ticker}">
        <div class="stock-header">
          <span class="stock-ticker">${stock.ticker}</span>
          <span class="stock-signal ${signalClass}">${signalText}</span>
        </div>
        <div class="stock-thesis">${shortThesis}</div>
        <div class="stock-stats">
          <div class="stock-stat">
            <span class="stat-value">${fmtPct(stock.discount_from_high)}</span>
            <span class="stat-label">Below High</span>
          </div>
          <div class="stock-stat">
            <span class="stat-value ${stock.prob_beat_spy_1y > 50 ? 'positive' : ''}">${fmtPct(stock.prob_beat_spy_1y)}</span>
            <span class="stat-label">Beat SPY</span>
          </div>
          <div class="stock-stat">
            <span class="stat-value ${stock.median_return_1y > 0 ? 'positive' : 'negative'}">${fmtReturn(stock.median_return_1y)}</span>
            <span class="stat-label">Median 1Y</span>
          </div>
          <div class="stock-stat">
            <span class="stat-value">${stock.sample_size_1y || '—'}</span>
            <span class="stat-label">Samples</span>
          </div>
        </div>
      </div>
    `;
  }).join('');

  // Add click handlers
  container.querySelectorAll('.stock-card').forEach(card => {
    card.addEventListener('click', () => {
      const ticker = card.dataset.ticker;
      const stock = allStocks.find(s => s.ticker === ticker);
      if (stock) showModal(stock);
    });
  });
}

// Show detail modal
function showModal(stock) {
  const modal = $('modal');
  const body = $('modalBody');

  const signalClass = stock.signal === 'STRONG_BUY' ? 'strong' :
                      stock.signal === 'BUY' ? 'buy' : 'watch';

  body.innerHTML = `
    <div class="modal-header">
      <div>
        <div class="modal-ticker">${stock.ticker}</div>
        <span class="stock-signal ${signalClass}">${stock.signal.replace('_', ' ')}</span>
      </div>
      <div class="modal-price">$${stock.price?.toFixed(2) || '—'}</div>
    </div>

    <div class="modal-thesis">
      <strong>Why buy now?</strong><br>
      ${stock.thesis || 'No thesis available.'}
    </div>

    <div class="modal-section">
      <div class="modal-section-title">Valuation</div>
      <div class="modal-grid">
        <div class="modal-stat">
          <div class="modal-stat-value">${fmtPct(stock.discount_from_high)}</div>
          <div class="modal-stat-label">Below 52w High</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value">$${stock.high_52w?.toFixed(2) || '—'}</div>
          <div class="modal-stat-label">52w High</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value">$${stock.low_52w?.toFixed(2) || '—'}</div>
          <div class="modal-stat-label">52w Low</div>
        </div>
      </div>
    </div>

    <div class="modal-section">
      <div class="modal-section-title">Probability of Positive Return</div>
      <div class="modal-grid">
        <div class="modal-stat">
          <div class="modal-stat-value ${stock.prob_positive_1y > 60 ? 'positive' : ''}">${fmtPct(stock.prob_positive_1y)}</div>
          <div class="modal-stat-label">1 Year</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value ${stock.prob_positive_3y > 60 ? 'positive' : ''}">${fmtPct(stock.prob_positive_3y)}</div>
          <div class="modal-stat-label">3 Years</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value ${stock.prob_positive_5y > 60 ? 'positive' : ''}">${fmtPct(stock.prob_positive_5y)}</div>
          <div class="modal-stat-label">5 Years</div>
        </div>
      </div>
    </div>

    <div class="modal-section">
      <div class="modal-section-title">Probability of Beating SPY</div>
      <div class="modal-grid">
        <div class="modal-stat">
          <div class="modal-stat-value ${stock.prob_beat_spy_1y > 50 ? 'positive' : ''}">${fmtPct(stock.prob_beat_spy_1y)}</div>
          <div class="modal-stat-label">1 Year</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value ${stock.prob_beat_spy_3y > 50 ? 'positive' : ''}">${fmtPct(stock.prob_beat_spy_3y)}</div>
          <div class="modal-stat-label">3 Years</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value">—</div>
          <div class="modal-stat-label">5 Years</div>
        </div>
      </div>
    </div>

    <div class="modal-section">
      <div class="modal-section-title">Expected Returns (Median)</div>
      <div class="modal-grid">
        <div class="modal-stat">
          <div class="modal-stat-value ${stock.median_return_1y > 0 ? 'positive' : 'negative'}">${fmtReturn(stock.median_return_1y)}</div>
          <div class="modal-stat-label">1 Year</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value ${stock.median_return_3y > 0 ? 'positive' : 'negative'}">${fmtReturn(stock.median_return_3y)}</div>
          <div class="modal-stat-label">3 Years</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value ${stock.median_return_5y > 0 ? 'positive' : 'negative'}">${fmtReturn(stock.median_return_5y)}</div>
          <div class="modal-stat-label">5 Years</div>
        </div>
      </div>
    </div>

    <div class="modal-section">
      <div class="modal-section-title">Risk/Reward (1 Year)</div>
      <div class="modal-grid">
        <div class="modal-stat">
          <div class="modal-stat-value negative">${fmtReturn(stock.downside_1y)}</div>
          <div class="modal-stat-label">10th Pctl (Bad)</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value">${fmtReturn(stock.median_return_1y)}</div>
          <div class="modal-stat-label">Median</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value positive">${fmtReturn(stock.upside_1y)}</div>
          <div class="modal-stat-label">90th Pctl (Good)</div>
        </div>
      </div>
    </div>

    <div class="modal-section">
      <div class="modal-section-title">Quality Metrics</div>
      <div class="modal-grid">
        <div class="modal-stat">
          <div class="modal-stat-value ${stock.five_year_return > 0 ? 'positive' : 'negative'}">${fmtReturn(stock.five_year_return)}</div>
          <div class="modal-stat-label">5 Year Return</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value">${fmtPct(stock.monthly_win_rate)}</div>
          <div class="modal-stat-label">Monthly Win Rate</div>
        </div>
        <div class="modal-stat">
          <div class="modal-stat-value">${stock.early_recovery ? 'Yes' : 'No'}</div>
          <div class="modal-stat-label">Early Recovery</div>
        </div>
      </div>
    </div>

    <div class="sample-size">
      Based on ${stock.sample_size_1y || 0} similar historical situations (1Y) •
      ${stock.sample_size_3y || 0} (3Y) • ${stock.sample_size_5y || 0} (5Y)
    </div>
  `;

  modal.classList.add('open');
}

// Close modal
function closeModal() {
  $('modal').classList.remove('open');
}

// Search
function handleSearch(e) {
  const query = e.target.value.toLowerCase().trim();

  if (!query) {
    renderStocks(allStocks);
    return;
  }

  const filtered = allStocks.filter(s =>
    s.ticker.toLowerCase().includes(query)
  );
  renderStocks(filtered);
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  loadData();

  // Search
  $('search').addEventListener('input', handleSearch);

  // Modal close
  $('modalClose').addEventListener('click', closeModal);
  $('modal').addEventListener('click', (e) => {
    if (e.target === $('modal')) closeModal();
  });

  // Escape key closes modal
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
});
