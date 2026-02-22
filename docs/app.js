
const DATA_URL = "./data/full.json";
const API_URL = window.REBOUND_API_URL || "";

const CACHE_BUST = String(Date.now());

function withBust(url){
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}v=${encodeURIComponent(CACHE_BUST)}`;
}

/* ---------- formatting helpers ---------- */

function fmtPct(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "\u2014";
  const v = Number(x);
  if (!Number.isFinite(v)) return "\u2014";
  return `${Math.round(v * 100)}%`;
}

function fmtPctWhole(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "\u2014";
  const v = Number(x);
  if (!Number.isFinite(v)) return "\u2014";
  return `${Math.round(v)}%`;
}

function fmtNum0(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "\u2014";
  const v = Number(x);
  if (!Number.isFinite(v)) return "\u2014";
  return v.toFixed(0);
}

function clamp01(x){ return Math.max(0, Math.min(1, x)); }

function byId(id){ return document.getElementById(id); }

/* ---------- score badges ---------- */

function oppBadge(v){
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return `<span class="badge badge-na">\u2014</span>`;
  const n = Number(v);
  if (n >= 65) return `<span class="badge badge-opp-high">${Math.round(n)}</span>`;
  if (n >= 50) return `<span class="badge badge-opp-med">${Math.round(n)}</span>`;
  return `<span class="badge badge-opp-low">${Math.round(n)}</span>`;
}

function probColor(p){
  if (p === null || p === undefined || !Number.isFinite(Number(p))) return "var(--muted)";
  const v = Number(p);
  if (v >= 80) return "#064e2b";
  if (v >= 65) return "#0a6636";
  if (v >= 50) return "var(--ink)";
  return "#b35900";
}

/* ---------- chart ---------- */

function drawGradientLine(canvas, dates, prices, score){
  const ctx = canvas.getContext("2d");
  const w = canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
  const h = canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
  ctx.clearRect(0, 0, w, h);

  const n = prices.length;
  if (n < 3) return;

  const pad = 12 * devicePixelRatio;
  let minP = Infinity, maxP = -Infinity;
  for (let i = 0; i < n; i++){
    const p = Number(prices[i]);
    if (!Number.isFinite(p)) continue;
    if (p < minP) minP = p;
    if (p > maxP) maxP = p;
  }
  if (!(maxP > minP)) return;

  const x0 = pad, x1 = w - pad, y0 = pad, y1 = h - pad;
  function xAt(i){ return x0 + (x1 - x0) * (i / (n - 1)); }
  function yAt(p){ return y1 - (y1 - y0) * ((p - minP) / (maxP - minP)); }

  ctx.lineWidth = 1.8 * devicePixelRatio;
  ctx.strokeStyle = "rgba(0,0,0,.25)";
  ctx.beginPath();
  for (let i = 0; i < n; i++){
    const p = Number(prices[i]);
    if (!Number.isFinite(p)) continue;
    const x = xAt(i), y = yAt(p);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  for (let i = 0; i < n - 1; i++){
    const s = Number(score?.[i]);
    if (!Number.isFinite(s) || s < 25) continue;
    const a = clamp01((s - 25) / 75);  // map 25–100 → 0–1
    ctx.lineWidth = 3 * devicePixelRatio;
    ctx.strokeStyle = `rgba(6,78,43,${0.12 + 0.65 * a})`;
    ctx.beginPath();
    ctx.moveTo(xAt(i), yAt(prices[i]));
    ctx.lineTo(xAt(i + 1), yAt(prices[i + 1]));
    ctx.stroke();
  }

  const lastScore = Number(score?.[n - 1]);
  const rawA = clamp01((Number.isFinite(lastScore) ? lastScore : 0) / 100);
  ctx.fillStyle = `rgba(6,78,43,${0.25 + 0.70 * rawA})`;
  ctx.strokeStyle = "rgba(0,0,0,.85)";
  ctx.lineWidth = 1.2 * devicePixelRatio;
  ctx.beginPath();
  ctx.arc(xAt(n - 1), yAt(prices[n - 1]), 4 * devicePixelRatio, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
}

/* ---------- outcome box (legacy, used by evidence section) ---------- */

function outcomeBox(label, s){
  const b = document.createElement("div");
  b.className = "outbox";
  if (!s || !Number.isFinite(s.n) || s.n <= 0){
    b.innerHTML = `<div class="h">${label}</div><div class="r"><span>Not enough data</span><strong>\u2014</strong></div>`;
    return b;
  }
  b.innerHTML = `
    <div class="h">${label}</div>
    <div class="r"><span>Chance of gain</span><strong style="color:${probColor(s.win * 100)}">${Math.round(s.win * 100)}%</strong></div>
    <div class="r"><span>Typical return</span><strong>${fmtPct(s.median)}</strong></div>
    <div class="r"><span>Bad case (1 in 10)</span><strong>${fmtPct(s.p10)}</strong></div>
    <div class="r"><span>Based on</span><strong>${s.n} past cases</strong></div>
  `;
  return b;
}

/* ---------- unified outcomes table ---------- */

function outcomesTable(outcomes){
  const horizons = [["1Y", "1 Year"], ["3Y", "3 Years"], ["5Y", "5 Years"]];
  const has = horizons.filter(([k]) => {
    const s = outcomes?.[k];
    return s && Number.isFinite(s.n) && s.n > 0;
  });
  if (has.length === 0) return null;

  const tbl = document.createElement("table");
  tbl.className = "outcomes-table";

  // Header row
  let hdr = `<thead><tr><th></th>`;
  for (const [k, label] of has) hdr += `<th>${label}</th>`;
  hdr += `</tr></thead>`;

  // Data rows
  const metrics = [
    ["Chance of gain", (s) => { const v = Math.round(s.win * 100); return `<span style="color:${probColor(v)}">${v}%</span>`; }],
    ["Typical return", (s) => fmtPct(s.median)],
    ["Bad case (1 in 10)", (s) => fmtPct(s.p10)],
    ["Past cases", (s) => String(s.n)],
  ];

  let body = `<tbody>`;
  for (const [label, fn] of metrics){
    body += `<tr><td>${label}</td>`;
    for (const [k] of has){
      body += `<td>${fn(outcomes[k])}</td>`;
    }
    body += `</tr>`;
  }
  body += `</tbody>`;

  tbl.innerHTML = hdr + body;
  return tbl;
}

/* ---------- evidence section ---------- */

function evidenceSection(detail){
  const base = detail?.evidence_baseline || detail?.evidence_finalscore || detail?.evidence_washout || null;
  const outs = detail?.outcomes || null;
  if (!base || !outs) return null;

  const box = document.createElement("details");
  box.className = "details evidence-details";
  box.innerHTML = `
    <summary class="details-summary">
      <div class="evidence-summary-left">
        <span class="section-title">EVIDENCE</span>
        <span class="ev-sub">Does this pullback actually help?</span>
      </div>
      <span class="plus" aria-hidden="true">+</span>
    </summary>
    <div class="details-body">
      <div class="ev-explain">
        Each column compares two things: what happened when this stock was in a similar pullback before
        (<strong>after pullback</strong>) vs. what normally happens on any random day (<strong>any day</strong>).
        If the "after pullback" numbers are better, it means dips like this have historically been
        good buying opportunities.
      </div>
      <div class="outcomes ev-grid"></div>
    </div>
  `;

  const gridEl = box.querySelector(".ev-grid");
  const horizons = [["1Y", "1 Year"], ["3Y", "3 Years"], ["5Y", "5 Years"]];

  for (const [k, label] of horizons){
    const a = outs?.[k];
    const b = base?.[k];
    if (!a || !b) continue;

    const winA = a.win, winB = b.win_norm;
    const medA = a.median, medB = b.med_norm;
    const p10A = a.p10, p10B = b.p10_norm;
    const nA = a.n, nB = b.n_norm;

    if (!(Number.isFinite(winA) && Number.isFinite(winB))) continue;

    const bx = document.createElement("div");
    bx.className = "outbox";
    bx.innerHTML = `
      <div class="h">${label}</div>
      <div class="ev-cols-header"><span></span><span>After pullback</span><span>Any day</span></div>
      <div class="r"><span>Chance of gain</span><strong>${Math.round(winA * 100)}%</strong><strong class="ev-norm">${Math.round(winB * 100)}%</strong></div>
      <div class="r"><span>Typical return</span><strong>${fmtPct(medA)}</strong><strong class="ev-norm">${fmtPct(medB)}</strong></div>
      <div class="r"><span>Bad case</span><strong>${fmtPct(p10A)}</strong><strong class="ev-norm">${fmtPct(p10B)}</strong></div>
      <div class="r"><span>Cases</span><strong>${nA}</strong><strong class="ev-norm">${nB}</strong></div>
    `;
    gridEl.appendChild(bx);
  }

  if (!gridEl.children.length) return null;
  return box;
}

/* ---------- quality proof section ---------- */

function proofSection(detail){
  const qp = detail?.quality_parts;
  const rh = detail?.recovery_history;
  if (!qp && !rh) return null;

  const box = document.createElement("details");
  box.className = "details proof-details";

  const hasTrend = qp?.trend != null && Number.isFinite(Number(qp.trend));
  const hasRecov = qp?.recovery != null && Number.isFinite(Number(qp.recovery));
  const hasMom   = qp?.momentum != null && Number.isFinite(Number(qp.momentum));

  let barsHtml = "";
  if (hasTrend || hasRecov || hasMom){
    barsHtml += `<div class="proof-grid">`;
    if (hasTrend){
      const v = Math.round(Number(qp.trend));
      barsHtml += `
        <div class="proof-row">
          <span class="proof-label">Does it usually go up? (45%)</span>
          <div class="proof-bar-track"><div class="proof-bar-fill" style="width:${v}%;background:${v >= 60 ? '#064e2b' : v >= 40 ? '#555' : '#b35900'}"></div></div>
          <strong class="proof-val">${v}</strong>
        </div>`;
    }
    if (hasRecov){
      const v = Math.round(Number(qp.recovery));
      barsHtml += `
        <div class="proof-row">
          <span class="proof-label">Does it bounce back from drops? (35%)</span>
          <div class="proof-bar-track"><div class="proof-bar-fill" style="width:${v}%;background:${v >= 60 ? '#064e2b' : v >= 40 ? '#555' : '#b35900'}"></div></div>
          <strong class="proof-val">${v}</strong>
        </div>`;
    }
    if (hasMom){
      const v = Math.round(Number(qp.momentum));
      barsHtml += `
        <div class="proof-row">
          <span class="proof-label">Is the selling slowing down? (20%)</span>
          <div class="proof-bar-track"><div class="proof-bar-fill" style="width:${v}%;background:${v >= 60 ? '#064e2b' : v >= 40 ? '#555' : '#b35900'}"></div></div>
          <strong class="proof-val">${v}</strong>
        </div>`;
    }
    barsHtml += `</div>`;
  }

  let recovHtml = "";
  if (rh && Number.isFinite(Number(rh.recovery_rate))){
    const rate = Math.round(Number(rh.recovery_rate) * 100);
    const nDd = rh.n_drawdowns != null ? Number(rh.n_drawdowns) : null;
    const nRec = rh.n_recovered != null ? Number(rh.n_recovered) : null;
    recovHtml = `
      <div class="proof-recovery">
        <div class="proof-recov-title">How often has it recovered from big drops?</div>
        <div class="proof-recov-stats">
          <div class="proof-recov-stat"><span>Bounced back</span><strong style="color:${rate >= 75 ? '#064e2b' : rate >= 50 ? 'var(--ink)' : '#b35900'}">${rate}% of the time</strong></div>
          ${nDd != null ? `<div class="proof-recov-stat"><span>Times it dropped 20%+</span><strong>${nDd}</strong></div>` : ""}
          ${nRec != null ? `<div class="proof-recov-stat"><span>Times it recovered within 3 years</span><strong>${nRec}</strong></div>` : ""}
        </div>
      </div>`;
  }

  if (!barsHtml && !recovHtml) return null;

  box.innerHTML = `
    <summary class="details-summary">
      <div class="proof-summary-left">
        <span class="section-title">QUALITY BREAKDOWN</span>
        <span class="ev-sub">Why this stock scored ${detail?.quality != null ? Math.round(Number(detail.quality)) + '/100' : 'its'} quality rating</span>
      </div>
      <span class="plus" aria-hidden="true">+</span>
    </summary>
    <div class="details-body">
      <div class="proof-explain">Each bar shows how this stock scores on the three factors that make up its quality rating. Higher = better. The percentages show how much each factor counts.</div>
      ${barsHtml}
      ${recovHtml}
    </div>
  `;

  return box;
}

/* ---------- rationale (top 10 only) ---------- */

function buildRationale(item){
  const pullback = Number(item.washout_today);
  const prob1y = Number(item.prob_1y);
  const cases = Number(item.n_analogs);
  const typical = item.median_1y;
  const downside = item.downside_1y;
  const quality = Number(item.quality);

  const parts = [];
  if (Number.isFinite(pullback) && pullback > 0)
    parts.push(`Dropped <strong>${fmtNum0(pullback)}/100</strong> from its normal range.`);
  if (Number.isFinite(cases) && cases > 0 && Number.isFinite(prob1y))
    parts.push(`In <strong>${fmtNum0(cases)}</strong> similar past pullbacks, it was higher 1 year later <strong>${fmtPctWhole(prob1y)}</strong> of the time${typical != null && Number.isFinite(Number(typical)) ? `, gaining <strong>${fmtPct(typical)}</strong> typically` : ""}.`);
  if (downside != null && Number.isFinite(Number(downside)))
    parts.push(`Worst 1-in-10 scenario: <strong>${fmtPct(downside)}</strong>.`);
  if (Number.isFinite(quality))
    parts.push(`Quality score: <strong>${fmtNum0(quality)}</strong> — ${quality >= 70 ? "strong stock with a history of bouncing back" : quality >= 45 ? "decent stock, moderate recovery track record" : "weaker stock, less consistent recoveries"}.`);

  return parts.join(" ");
}

/* ---------- card body (chart + quality + evidence) ---------- */

function renderCardBody(body, item, detail, isTop10){
  const series = detail?.series || {};

  // Rationale for top 10
  if (isTop10){
    const rat = document.createElement("div");
    rat.className = "rationale";
    rat.innerHTML = buildRationale(item);
    body.appendChild(rat);
  }

  // Chart
  const chartWrap = document.createElement("div");
  chartWrap.className = "chart";
  const canvas = document.createElement("canvas");
  canvas.className = "canvas";
  chartWrap.appendChild(canvas);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = `<span class="legend-bar" aria-hidden="true"></span><span class="legend-text"><span class="legend-label">Opportunity score</span><span class="legend-note">darker = higher score</span></span>`;
  chartWrap.appendChild(legend);
  body.appendChild(chartWrap);

  // Price + date below chart
  if (series.prices?.length){
    const lastPrice = Number(series.prices[series.prices.length - 1]);
    const lastDate = series.dates?.[series.dates.length - 1] || "";
    if (Number.isFinite(lastPrice)){
      const priceLine = document.createElement("div");
      priceLine.className = "chart-price-line";
      priceLine.innerHTML = `<span class="chart-price-val">$${lastPrice.toFixed(2)}</span><span class="chart-price-date">${lastDate}</span>`;
      body.appendChild(priceLine);
    }
  }

  // Quality proof
  const proof = proofSection(detail);
  if (proof) body.appendChild(proof);

  // Evidence
  const ev = evidenceSection(detail);
  if (ev) body.appendChild(ev);

  // Draw chart
  if (series.prices && series.prices.length){
    requestAnimationFrame(() => drawGradientLine(canvas, series.dates, series.prices, series.final));
  }
}

/* ---------- data loading ---------- */

async function loadJSON(url){
  const u = withBust(url);
  const r = await fetch(u, {
    cache: "no-store",
    headers: { "Pragma": "no-cache", "Cache-Control": "no-cache" },
  });
  if (!r.ok) throw new Error(`Fetch failed ${r.status}: ${url}`);
  return await r.json();
}

function setSortButtons(active){
  document.querySelectorAll(".sort .btn-lite").forEach(b => {
    b.classList.toggle("active", b.dataset.sort === active);
  });
}

function formatAsOf(asOf){
  if (!asOf) return "\u2014";
  let s = String(asOf).trim();
  if (s.includes(" ") && !s.includes("T")) s = s.replace(" ", "T");
  let d = new Date(s);
  if (Number.isNaN(d.getTime())){
    s = s.replace(/\.(\d+)(Z|[+-]\d\d:\d\d)?$/, "$2");
    d = new Date(s);
  }
  if (Number.isNaN(d.getTime())) return String(asOf);

  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "numeric", minute: "2-digit", hour12: true,
  }).formatToParts(d);

  const get = (type) => (parts.find(p => p.type === type)?.value || "");
  return `${get("year")}-${get("month")}-${get("day")} ${get("hour")}:${get("minute")} ${get("dayPeriod")} EST`;
}

/* ---------- sort helpers ---------- */

function safeNum(x){
  if (x === null || x === undefined) return -Infinity;
  const v = Number(x);
  return Number.isFinite(v) ? v : -Infinity;
}

function sortItems(items, mode){
  const list = [...items];
  switch (mode){
    case "prob_1y":
      list.sort((a, b) => safeNum(b.prob_1y) - safeNum(a.prob_1y) || safeNum(b.conviction) - safeNum(a.conviction));
      break;
    case "prob_3y":
      list.sort((a, b) => safeNum(b.prob_3y) - safeNum(a.prob_3y) || safeNum(b.prob_1y) - safeNum(a.prob_1y));
      break;
    case "prob_5y":
      list.sort((a, b) => safeNum(b.prob_5y) - safeNum(a.prob_5y) || safeNum(b.prob_1y) - safeNum(a.prob_1y));
      break;
    case "conviction":
      list.sort((a, b) => safeNum(b.conviction) - safeNum(a.conviction) || safeNum(b.prob_1y) - safeNum(a.prob_1y));
      break;
    default:
      list.sort((a, b) => safeNum(b.conviction) - safeNum(a.conviction));
  }
  return list;
}

/* ---------- marquee builder ---------- */

function buildMarquee(items){
  const track = byId("marquee-track");
  if (!track || !items.length) return;

  // Build content from top tickers
  const top = items.slice(0, 15);
  const parts = top.map(item => {
    const score = Math.round(Number(item.conviction));
    const prob = Math.round(Number(item.prob_1y));
    return `<span class="marquee-ticker">${item.ticker}</span>` +
           `<span class="marquee-score">${score}</span>` +
           `<span class="marquee-prob">${prob}% 1Y</span>`;
  });
  const html = parts.join('<span class="marquee-dot">\u00b7</span>');

  // Duplicate for seamless loop
  track.innerHTML =
    `<div class="marquee-content">${html}<span class="marquee-dot">\u00b7</span></div>` +
    `<div class="marquee-content" aria-hidden="true">${html}<span class="marquee-dot">\u00b7</span></div>`;
}

/* ---------- main ---------- */

(async function main(){
  let full;
  try {
    full = await loadJSON(DATA_URL);
  } catch (e){
    console.error("loadJSON error:", e);
    byId("listing").innerHTML = `<div class="footnote">No data available yet. The daily scan has not run.</div>`;
    return;
  }
  try {

  byId("asOf").textContent = formatAsOf(full.as_of);

  let items = full.items || [];

  // Populate stats — real numbers
  const uniEl = byId("statUniverse");
  if (uniEl) uniEl.textContent = "1,000+";

  const depthEl = byId("statDepth");
  if (depthEl) depthEl.textContent = "20+";

  // Build dynamic marquee with top tickers
  buildMarquee(items);

  let sortMode = "conviction";

  async function loadDetail(ticker){
    const embedded = (full.details && full.details[ticker]) ? full.details[ticker] : null;
    if (embedded) return embedded;
    return await loadJSON(`./data/tickers/${ticker}.json`);
  }

  async function renderListing(sorted){
    const c = byId("listing");
    c.innerHTML = "";

    for (let i = 0; i < sorted.length; i++){
      const item = sorted[i];
      const isTop10 = i < 10;

      const card = document.createElement("details");
      card.className = "ticker-card";
      card.dataset.ticker = item.ticker;
      if (isTop10) card.open = true;

      // Summary row (always visible)
      const summary = document.createElement("summary");
      summary.className = "ticker-row";
      summary.innerHTML = `
        <span class="row-ticker">${item.ticker}</span>
        <span class="row-cell" data-label="Opp Score">${oppBadge(item.conviction)}</span>
        <span class="row-cell" data-label="Pullback">${fmtNum0(item.washout_today)}</span>
        <span class="row-cell" data-label="1Y Prob" style="color:${probColor(item.prob_1y)}">${fmtPctWhole(item.prob_1y)}</span>
        <span class="row-cell" data-label="3Y Prob" style="color:${probColor(item.prob_3y)}">${fmtPctWhole(item.prob_3y)}</span>
        <span class="row-cell" data-label="5Y Prob" style="color:${probColor(item.prob_5y)}">${fmtPctWhole(item.prob_5y)}</span>
        <span class="row-cell" data-label="Typical">${fmtPct(item.median_1y)}</span>
        <span class="row-cell" data-label="Bad case">${fmtPct(item.downside_1y)}</span>
        <span class="row-cell" data-label="Cases">${fmtNum0(item.n_analogs)}</span>
        <span class="row-cell" data-label="Quality">${fmtNum0(item.quality)}</span>
      `;
      card.appendChild(summary);

      // Body (expanded content)
      const body = document.createElement("div");
      body.className = "ticker-body";
      card.appendChild(body);

      // Top 10: load detail immediately
      if (isTop10){
        try {
          const detail = await loadDetail(item.ticker);
          renderCardBody(body, item, detail, true);
        } catch (err){
          body.innerHTML = `<div class="footnote">Details unavailable. Try refreshing.</div>`;
        }
      } else {
        // Lazy load on expand
        card.addEventListener("toggle", async function(){
          if (card.open && !card.dataset.loaded){
            card.dataset.loaded = "true";
            body.innerHTML = `<div class="footnote" style="padding:10px 0">Loading...</div>`;
            try {
              const detail = await loadDetail(item.ticker);
              body.innerHTML = "";
              renderCardBody(body, item, detail, false);
            } catch (err){
              body.innerHTML = `<div class="footnote">Details unavailable. Try refreshing.</div>`;
            }
          }
        });
      }

      c.appendChild(card);
    }
  }

  async function rerender(){
    const sorted = sortItems(items, sortMode);
    await renderListing(sorted);
  }

  // Sort button clicks
  document.querySelectorAll(".sort .btn-lite").forEach(btn => {
    btn.addEventListener("click", async () => {
      sortMode = btn.dataset.sort;
      setSortButtons(sortMode);
      await rerender();
    });
  });

  setSortButtons(sortMode);
  await rerender();

  // Search — filter existing + on-demand analysis for unknown tickers
  const onDemandCache = {};   // ticker -> { row, detail }
  let searchDebounce = null;

  function setSearchStatus(msg, isError){
    let el = document.querySelector(".search-status");
    if (!el){
      el = document.createElement("span");
      el.className = "search-status";
      byId("go").parentNode.appendChild(el);
    }
    el.textContent = msg;
    el.classList.toggle("error", !!isError);
  }

  function clearSearchStatus(){
    const el = document.querySelector(".search-status");
    if (el) el.textContent = "";
  }

  function applySearch(){
    const q = (byId("q").value || "").trim().toUpperCase();
    clearSearchStatus();
    if (!q){
      rerender();
      return;
    }
    const sorted = sortItems(items, sortMode);
    const filtered = sorted.filter(x => x.ticker.includes(q));

    // Also include any on-demand results that match
    for (const [tk, cached] of Object.entries(onDemandCache)){
      if (tk.includes(q) && !filtered.some(x => x.ticker === tk)){
        filtered.unshift({...cached.row, _onDemand: true});
      }
    }

    renderListing(filtered);
  }

  async function analyzeOnDemand(ticker){
    ticker = ticker.trim().toUpperCase();
    if (!ticker) return;

    // Already in our dataset?
    if (items.some(x => x.ticker === ticker)){
      applySearch();
      return;
    }

    // Already cached from a previous on-demand lookup?
    if (onDemandCache[ticker]){
      applySearch();
      return;
    }

    setSearchStatus("Analyzing " + ticker + " — this may take up to a minute...");
    byId("go").disabled = true;

    try {
      const resp = await fetch(`${API_URL}/api/analyze?ticker=${encodeURIComponent(ticker)}`);
      const data = await resp.json();

      if (!resp.ok){
        setSearchStatus(data.error || "Analysis failed", true);
        byId("go").disabled = false;
        return;
      }

      onDemandCache[ticker] = { row: data.row, detail: data.detail };
      clearSearchStatus();

      // Render the on-demand result
      const c = byId("listing");
      c.innerHTML = "";

      const item = {...data.row, _onDemand: true};
      const card = document.createElement("details");
      card.className = "ticker-card ondemand-card";
      card.dataset.ticker = item.ticker;
      card.open = true;

      const summary = document.createElement("summary");
      summary.className = "ticker-row";
      summary.innerHTML = `
        <span class="row-ticker">${item.ticker}</span>
        <span class="row-cell" data-label="Opp Score">${oppBadge(item.conviction)}</span>
        <span class="row-cell" data-label="Pullback">${fmtNum0(item.washout_today)}</span>
        <span class="row-cell" data-label="1Y Prob" style="color:${probColor(item.prob_1y)}">${fmtPctWhole(item.prob_1y)}</span>
        <span class="row-cell" data-label="3Y Prob" style="color:${probColor(item.prob_3y)}">${fmtPctWhole(item.prob_3y)}</span>
        <span class="row-cell" data-label="5Y Prob" style="color:${probColor(item.prob_5y)}">${fmtPctWhole(item.prob_5y)}</span>
        <span class="row-cell" data-label="Typical">${fmtPct(item.median_1y)}</span>
        <span class="row-cell" data-label="Bad case">${fmtPct(item.downside_1y)}</span>
        <span class="row-cell" data-label="Cases">${fmtNum0(item.n_analogs)}</span>
        <span class="row-cell" data-label="Quality">${fmtNum0(item.quality)}</span>
      `;
      card.appendChild(summary);

      const body = document.createElement("div");
      body.className = "ticker-body";
      renderCardBody(body, item, data.detail, true);
      card.appendChild(body);
      c.appendChild(card);

    } catch (err){
      console.error("On-demand analysis error:", err);
      setSearchStatus("Could not connect to analysis server", true);
    }
    byId("go").disabled = false;
  }

  // Wire up search: typing filters existing, "Find" button also triggers on-demand
  byId("q").addEventListener("input", function(){
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(applySearch, 150);
  });

  byId("go").addEventListener("click", function(){
    const q = (byId("q").value || "").trim().toUpperCase();
    if (!q) { rerender(); return; }
    // If exact match exists in items, just filter
    if (items.some(x => x.ticker === q)){
      applySearch();
      return;
    }
    // Otherwise trigger on-demand analysis
    analyzeOnDemand(q);
  });

  byId("q").addEventListener("keydown", function(e){
    if (e.key === "Enter"){
      e.preventDefault();
      byId("go").click();
    }
  });

  /* ============================================================
     BACKTEST — DCA OPTIMIZER
     ============================================================ */

  const btSection = document.getElementById("backtest-section");
  let btLoaded = false;
  let btResults = null;   // cached results across tab switches
  let btHoldMonths = 12;  // default

  btSection.addEventListener("toggle", function(){
    if (btSection.open && !btLoaded){
      btLoaded = true;
      runBacktest();
    }
  });

  // Hold-period tab buttons
  document.querySelectorAll(".bt-hold-tabs .btn-lite").forEach(btn => {
    btn.addEventListener("click", function(){
      document.querySelectorAll(".bt-hold-tabs .btn-lite").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      btHoldMonths = Number(btn.dataset.hold);
      if (btResults) renderBacktestResults(btResults, btHoldMonths);
    });
  });

  async function runBacktest(){
    const body = byId("backtest-body");
    body.innerHTML = `<div class="footnote">Loading all ticker data — this may take a moment&hellip;</div>`;

    try {
      // 1. Load all ticker data
      const tickers = items.map(i => i.ticker);
      const allData = {};
      const promises = tickers.map(async tk => {
        try {
          allData[tk] = await loadDetail(tk);
        } catch(e){ /* skip */ }
      });
      await Promise.all(promises);

      // Ensure SPY is loaded
      if (!allData["SPY"]){
        body.innerHTML = `<div class="footnote" style="color:#b00">SPY data not available for benchmarking.</div>`;
        return;
      }

      // 2. Build date→index maps and aligned date grid using SPY dates
      const spySeries = allData["SPY"].series;
      const allDates = spySeries.dates;

      // Build price & score lookups per ticker (date string → value)
      const priceLookup = {};  // tk → Map(date → price)
      const scoreLookup = {};  // tk → Map(date → final score)
      for (const [tk, data] of Object.entries(allData)){
        const s = data.series;
        if (!s || !s.dates) continue;
        const pm = new Map();
        const fm = new Map();
        for (let i = 0; i < s.dates.length; i++){
          pm.set(s.dates[i], Number(s.prices[i]));
          fm.set(s.dates[i], Number(s.final[i]));
        }
        priceLookup[tk] = pm;
        scoreLookup[tk] = fm;
      }

      // 3. Identify first trading day of each month
      const monthFirstIdx = [];
      let prevYM = "";
      for (let i = 0; i < allDates.length; i++){
        const ym = allDates[i].substring(0, 7);
        if (ym !== prevYM){
          monthFirstIdx.push(i);
          prevYM = ym;
        }
      }

      // 4. Run simulations
      const CRYPTO = new Set(["BTC-USD", "ETH-USD"]);
      const availTickers = Object.keys(priceLookup).filter(t => t !== "SPY" && !CRYPTO.has(t));
      const strategies = [1, 5, 10];
      const holdPeriods = [12, 36, 60];

      // For each month, rank tickers by score and record top N
      const monthlyRanks = [];  // [{ date, dateIdx, ranked: [ticker, ...] }]
      for (const mIdx of monthFirstIdx){
        const date = allDates[mIdx];
        const scored = [];
        for (const tk of availTickers){
          const s = scoreLookup[tk]?.get(date);
          const p = priceLookup[tk]?.get(date);
          if (s != null && Number.isFinite(s) && p != null && Number.isFinite(p) && p > 0){
            scored.push({ ticker: tk, score: s });
          }
        }
        scored.sort((a, b) => b.score - a.score);
        monthlyRanks.push({ date, dateIdx: mIdx, ranked: scored.map(s => s.ticker) });
      }

      // Precompute day→month ordinal for fast lookup
      const dayToMonth = new Int32Array(allDates.length);
      for (let m = 0; m < monthFirstIdx.length; m++){
        const start = monthFirstIdx[m];
        const end = m + 1 < monthFirstIdx.length ? monthFirstIdx[m + 1] : allDates.length;
        for (let d = start; d < end; d++) dayToMonth[d] = m;
      }

      // Generic DCA simulation (works for both strategy picks and SPY)
      function simulateDCA(holdMonths, picksFn){
        const DCA = 1000;
        const positions = [];
        let cash = 0;
        const equity = new Float64Array(allDates.length);

        // Build positions
        for (let m = 0; m < monthlyRanks.length; m++){
          const date = monthlyRanks[m].date;
          const picks = picksFn(m, date); // [{ticker, price}, ...]
          if (picks.length === 0) continue;
          const perStock = DCA / picks.length;
          for (const { ticker, price } of picks){
            if (!price || price <= 0) continue;
            positions.push({ ticker, shares: perStock / price, cost: perStock, buyPrice: price, buyMonth: m, sellMonth: m + holdMonths, sold: false, sellPrice: 0 });
          }
        }

        // Walk daily
        let totalInvested = 0;
        let currentMonth = -1;
        for (let d = 0; d < allDates.length; d++){
          const date = allDates[d];
          const mOrd = dayToMonth[d];

          if (mOrd !== currentMonth){
            currentMonth = mOrd;
            // Count investment
            const picks = picksFn(mOrd, monthlyRanks[mOrd]?.date);
            if (picks.length > 0) totalInvested += DCA;

            // Sell expired positions
            for (const pos of positions){
              if (!pos.sold && pos.sellMonth <= mOrd){
                const sp = priceLookup[pos.ticker]?.get(date);
                if (sp && sp > 0){ cash += pos.shares * sp; pos.sold = true; pos.sellPrice = sp; }
              }
            }
          }

          // Value open positions
          let openVal = 0;
          for (const pos of positions){
            if (pos.sold || pos.buyMonth > mOrd) continue;
            const p = priceLookup[pos.ticker]?.get(date);
            if (p && Number.isFinite(p)) openVal += pos.shares * p;
          }
          equity[d] = cash + openVal;
        }

        return { equity, totalInvested, positions };
      }

      function simulate(topN, holdMonths){
        return simulateDCA(holdMonths, (m, date) => {
          const ranked = monthlyRanks[m]?.ranked || [];
          const picks = ranked.slice(0, Math.min(topN, ranked.length));
          return picks.map(tk => ({ ticker: tk, price: priceLookup[tk]?.get(date) }));
        });
      }

      function simulateSPY(holdMonths){
        return simulateDCA(holdMonths, (m, date) => {
          const price = priceLookup["SPY"]?.get(date);
          return price && price > 0 ? [{ ticker: "SPY", price }] : [];
        });
      }

      // Pre-compute all results
      btResults = { allDates, priceLookup, holdPeriods: {} };
      for (const hp of holdPeriods){
        const spy = simulateSPY(hp);
        const strats = {};
        for (const n of strategies){
          strats[n] = simulate(n, hp);
        }
        btResults.holdPeriods[hp] = { spy, strats };
      }

      renderBacktestResults(btResults, btHoldMonths);

    } catch(err){
      console.error("Backtest error:", err);
      body.innerHTML = `<div class="footnote" style="color:#b00">Backtest error: ${err.message}</div>`;
    }
  }

  function aggregatePositions(positions, priceLookup, lastDate){
    const map = {};  // ticker → { totalCost, totalShares, soldValue, soldShares, openShares }
    for (const pos of positions){
      if (!map[pos.ticker]) map[pos.ticker] = { totalCost: 0, totalShares: 0, soldValue: 0, soldShares: 0, openShares: 0, buys: 0 };
      const agg = map[pos.ticker];
      agg.totalCost += pos.cost;
      agg.totalShares += pos.shares;
      agg.buys++;
      if (pos.sold){
        agg.soldValue += pos.shares * pos.sellPrice;
        agg.soldShares += pos.shares;
      } else {
        agg.openShares += pos.shares;
      }
    }

    const result = [];
    for (const [ticker, agg] of Object.entries(map)){
      const currentPrice = priceLookup[ticker]?.get(lastDate) || 0;
      const openValue = agg.openShares * currentPrice;
      const totalValue = agg.soldValue + openValue;
      const avgCost = agg.totalShares > 0 ? agg.totalCost / agg.totalShares : 0;
      const blendedExit = agg.totalShares > 0 ? totalValue / agg.totalShares : 0;
      const returnPct = agg.totalCost > 0 ? (totalValue - agg.totalCost) / agg.totalCost : 0;
      const status = agg.openShares === 0 ? "closed" : agg.soldShares === 0 ? "open" : "partial";
      result.push({ ticker, buys: agg.buys, totalInvested: agg.totalCost, totalShares: agg.totalShares, avgCost, blendedExit, totalValue, returnPct, status });
    }
    result.sort((a, b) => b.returnPct - a.returnPct);
    return result;
  }

  function renderBacktestResults(results, holdMonths){
    const body = byId("backtest-body");
    body.innerHTML = "";

    const { allDates, priceLookup } = results;
    const data = results.holdPeriods[holdMonths];
    if (!data){ body.innerHTML = `<div class="footnote">No data for this hold period.</div>`; return; }

    const { spy, strats } = data;
    const lastDate = allDates[allDates.length - 1];

    // --- Equity Curve Chart ---
    const chartWrap = document.createElement("div");
    chartWrap.className = "bt-chart-wrap";

    const canvas = document.createElement("canvas");
    canvas.className = "bt-canvas";
    chartWrap.appendChild(canvas);

    // Legend
    const legend = document.createElement("div");
    legend.className = "bt-legend";
    const colors = { 1: "#064e2b", 5: "#1a6dd1", 10: "#d4820e", spy: "#999" };
    const labels = { 1: "Top 1", 5: "Top 5", 10: "Top 10", spy: "SPY (benchmark)" };
    for (const key of [1, 5, 10, "spy"]){
      const swatch = document.createElement("span");
      swatch.className = "bt-legend-item";
      swatch.innerHTML = `<span class="bt-swatch" style="background:${colors[key]}"></span>${labels[key]}`;
      legend.appendChild(swatch);
    }
    chartWrap.appendChild(legend);
    body.appendChild(chartWrap);

    // Draw chart
    requestAnimationFrame(() => drawBacktestChart(canvas, allDates, strats, spy, colors));

    // --- Performance Metrics Table ---
    const metrics = computeMetrics(allDates, strats, spy, holdMonths);
    const table = document.createElement("table");
    table.className = "outcomes-table bt-table";

    const metricRows = [
      ["Total Invested", (m) => "$" + m.totalInvested.toLocaleString()],
      ["Final Value", (m) => "$" + m.finalValue.toLocaleString()],
      ["Total Return", (m) => fmtPctSigned(m.totalReturn)],
      ["CAGR", (m) => fmtPctSigned(m.cagr)],
      ["Max Drawdown", (m) => fmtPctSigned(-m.maxDrawdown)],
      ["Best Month", (m) => fmtPctSigned(m.bestMonth)],
      ["Worst Month", (m) => fmtPctSigned(m.worstMonth)],
      ["Win Rate (months)", (m) => Math.round(m.winRate * 100) + "%"],
      ["Sharpe Ratio", (m) => m.sharpe.toFixed(2)],
    ];

    let hdr = `<thead><tr><th></th><th>Top 1</th><th>Top 5</th><th>Top 10</th><th>SPY</th></tr></thead>`;
    let tbody = "<tbody>";
    for (const [label, fn] of metricRows){
      tbody += `<tr><td>${label}</td>`;
      tbody += `<td>${fn(metrics[1])}</td>`;
      tbody += `<td>${fn(metrics[5])}</td>`;
      tbody += `<td>${fn(metrics[10])}</td>`;
      tbody += `<td>${fn(metrics.spy)}</td>`;
      tbody += `</tr>`;
    }
    tbody += "</tbody>";
    table.innerHTML = hdr + tbody;
    body.appendChild(table);

    // --- DCA Positions by Strategy (collapsible) ---
    for (const n of [1, 5, 10]){
      const agg = aggregatePositions(strats[n].positions, priceLookup, lastDate);
      if (agg.length === 0) continue;

      const det = document.createElement("details");
      det.className = "details bt-positions-details";
      const stratLabel = n === 1 ? "Top 1" : n === 5 ? "Top 5" : "Top 10";

      let posRows = "";
      for (const p of agg){
        const retColor = p.returnPct >= 0 ? "#064e2b" : "#b35900";
        const statusTag = p.status === "open" ? `<span class="bt-status-open">holding</span>` : p.status === "partial" ? `<span class="bt-status-partial">partial</span>` : "";
        posRows += `<tr>
          <td style="text-align:left;font-weight:600">${p.ticker} ${statusTag}</td>
          <td>${p.buys}</td>
          <td>$${Math.round(p.totalInvested).toLocaleString()}</td>
          <td>$${p.avgCost < 1 ? p.avgCost.toFixed(4) : p.avgCost < 100 ? p.avgCost.toFixed(2) : Math.round(p.avgCost).toLocaleString()}</td>
          <td>$${p.blendedExit < 1 ? p.blendedExit.toFixed(4) : p.blendedExit < 100 ? p.blendedExit.toFixed(2) : Math.round(p.blendedExit).toLocaleString()}</td>
          <td>$${Math.round(p.totalValue).toLocaleString()}</td>
          <td style="color:${retColor};font-weight:600">${fmtPctSigned(p.returnPct)}</td>
        </tr>`;
      }

      det.innerHTML = `
        <summary class="details-summary">
          <div class="evidence-summary-left">
            <span class="section-title">${stratLabel} — DCA POSITIONS</span>
            <span class="ev-sub">${agg.length} stocks, averaged across ${agg.reduce((s,p) => s + p.buys, 0)} monthly purchases</span>
          </div>
          <span class="plus" aria-hidden="true">+</span>
        </summary>
        <div class="details-body">
          <table class="outcomes-table bt-pos-table">
            <thead><tr>
              <th style="text-align:left">Ticker</th>
              <th>Buys</th>
              <th>Invested</th>
              <th>Avg Cost</th>
              <th>Avg Exit</th>
              <th>Value</th>
              <th>Return</th>
            </tr></thead>
            <tbody>${posRows}</tbody>
          </table>
        </div>
      `;
      body.appendChild(det);
    }

    // Disclaimer
    const disc = document.createElement("div");
    disc.className = "footnote";
    disc.textContent = "Stocks only — crypto (BTC, ETH) excluded from this backtest. Including them would improve results further. Hypothetical simulation using historical opportunity scores. Past performance does not predict future results. Does not account for transaction costs, taxes, slippage, or survivorship bias.";
    body.appendChild(disc);
  }

  function fmtPctSigned(v){
    if (!Number.isFinite(v)) return "\u2014";
    const sign = v >= 0 ? "+" : "";
    return sign + (v * 100).toFixed(1) + "%";
  }

  function drawBacktestChart(canvas, dates, strats, spy, colors){
    const ctx = canvas.getContext("2d");
    const dpr = devicePixelRatio || 1;
    const w = canvas.width = Math.floor(canvas.clientWidth * dpr);
    const h = canvas.height = Math.floor(canvas.clientHeight * dpr);
    ctx.clearRect(0, 0, w, h);

    const pad = { top: 16 * dpr, right: 14 * dpr, bottom: 28 * dpr, left: 60 * dpr };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Collect all equity series
    const series = [];
    for (const n of [1, 5, 10]){
      series.push({ key: n, eq: strats[n].equity, color: colors[n] });
    }
    series.push({ key: "spy", eq: spy.equity, color: colors.spy });

    // Find global min/max (skip leading zeros)
    let gMin = Infinity, gMax = -Infinity;
    let startIdx = 0;
    for (const s of series){
      for (let i = 0; i < s.eq.length; i++){
        if (s.eq[i] > 0 && startIdx === 0) startIdx = i;
        if (s.eq[i] > 0){
          if (s.eq[i] < gMin) gMin = s.eq[i];
          if (s.eq[i] > gMax) gMax = s.eq[i];
        }
      }
    }
    // Find actual start (first non-zero across all)
    startIdx = 0;
    for (let i = 0; i < dates.length; i++){
      if (series.some(s => s.eq[i] > 0)){ startIdx = i; break; }
    }

    if (gMax <= gMin){ gMin = 0; gMax = 1; }
    const range = gMax - gMin;
    gMin -= range * 0.05;
    gMax += range * 0.05;

    const n = dates.length - startIdx;
    function xAt(i){ return pad.left + plotW * ((i - startIdx) / Math.max(1, n - 1)); }
    function yAt(v){ return pad.top + plotH * (1 - (v - gMin) / (gMax - gMin)); }

    // Grid lines & y-axis labels
    ctx.strokeStyle = "rgba(0,0,0,0.06)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.font = `${10 * dpr}px "IBM Plex Mono", monospace`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++){
      const v = gMin + (gMax - gMin) * (i / yTicks);
      const y = yAt(v);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      ctx.fillText("$" + Math.round(v).toLocaleString(), pad.left - 6 * dpr, y);
    }

    // X-axis: year labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    let prevYear = "";
    for (let i = startIdx; i < dates.length; i++){
      const yr = dates[i].substring(0, 4);
      if (yr !== prevYear){
        prevYear = yr;
        const x = xAt(i);
        ctx.fillText(yr, x, pad.top + plotH + 8 * dpr);
      }
    }

    // Draw lines (SPY first so it's behind, then strategies)
    const drawOrder = [{ key: "spy", eq: spy.equity, color: colors.spy }, ...([10, 5, 1].map(k => ({ key: k, eq: strats[k].equity, color: colors[k] })))];
    for (const s of drawOrder){
      ctx.strokeStyle = s.color;
      ctx.lineWidth = (s.key === "spy" ? 1.8 : 2.4) * dpr;
      ctx.setLineDash(s.key === "spy" ? [6 * dpr, 4 * dpr] : []);
      ctx.beginPath();
      let started = false;
      for (let i = startIdx; i < dates.length; i++){
        if (s.eq[i] <= 0) continue;
        const x = xAt(i), y = yAt(s.eq[i]);
        if (!started){ ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  function computeMetrics(dates, strats, spy, holdMonths){
    const result = {};

    function calcForSeries(eq, totalInvested){
      // Find first non-zero
      let start = 0;
      for (let i = 0; i < eq.length; i++){ if (eq[i] > 0){ start = i; break; } }
      const finalVal = eq[eq.length - 1];

      // Monthly returns
      const monthlyReturns = [];
      let prevYM = "";
      let monthStart = 0;
      for (let i = start; i < eq.length; i++){
        const ym = dates[i].substring(0, 7);
        if (ym !== prevYM){
          if (prevYM && eq[monthStart] > 0){
            monthlyReturns.push((eq[i] - eq[monthStart]) / eq[monthStart]);
          }
          prevYM = ym;
          monthStart = i;
        }
      }

      // Max drawdown (from peak equity)
      let peak = 0, maxDD = 0;
      for (let i = start; i < eq.length; i++){
        if (eq[i] > peak) peak = eq[i];
        const dd = (peak - eq[i]) / peak;
        if (dd > maxDD) maxDD = dd;
      }

      const totalReturn = totalInvested > 0 ? (finalVal - totalInvested) / totalInvested : 0;

      // CAGR: approximate years from data
      const startDate = new Date(dates[start]);
      const endDate = new Date(dates[dates.length - 1]);
      const years = (endDate - startDate) / (365.25 * 86400000);
      const cagr = years > 0 ? Math.pow(finalVal / totalInvested, 1 / years) - 1 : 0;

      // Sharpe (annualized from monthly)
      const avgR = monthlyReturns.reduce((a, b) => a + b, 0) / (monthlyReturns.length || 1);
      const stdR = Math.sqrt(monthlyReturns.reduce((a, b) => a + (b - avgR) ** 2, 0) / (monthlyReturns.length || 1));
      const sharpe = stdR > 0 ? (avgR / stdR) * Math.sqrt(12) : 0;

      const winRate = monthlyReturns.length > 0 ? monthlyReturns.filter(r => r > 0).length / monthlyReturns.length : 0;

      return {
        totalInvested: Math.round(totalInvested),
        finalValue: Math.round(finalVal),
        totalReturn,
        cagr,
        maxDrawdown: maxDD,
        bestMonth: monthlyReturns.length ? Math.max(...monthlyReturns) : 0,
        worstMonth: monthlyReturns.length ? Math.min(...monthlyReturns) : 0,
        winRate,
        sharpe,
      };
    }

    for (const n of [1, 5, 10]){
      result[n] = calcForSeries(strats[n].equity, strats[n].totalInvested);
    }
    result.spy = calcForSeries(spy.equity, spy.totalInvested);
    return result;
  }

  } catch (err){
    console.error("Rebound Ledger render error:", err);
    byId("listing").innerHTML = `<div class="footnote" style="color:#b00">Render error: ${err.message}</div>`;
  }
})();

/* ============================================================
   NAV — shrink on scroll + active section highlight
   ============================================================ */

(function(){
  const nav = document.querySelector(".site-nav");
  if (!nav) return;

  let ticking = false;
  window.addEventListener("scroll", function(){
    if (!ticking){
      requestAnimationFrame(function(){
        nav.classList.toggle("nav-scrolled", window.scrollY > 10);
        ticking = false;
      });
      ticking = true;
    }
  });
})();

/* ============================================================
   INTERSECTION OBSERVER — fade-in sections on scroll
   ============================================================ */

(function(){
  const targets = document.querySelectorAll(".stat-block, .details-card");
  if (!targets.length || !("IntersectionObserver" in window)) return;

  const observer = new IntersectionObserver(function(entries){
    for (const entry of entries){
      if (entry.isIntersecting){
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    }
  }, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });

  targets.forEach(function(el){
    el.classList.add("fade-target");
    observer.observe(el);
  });
})();
