
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
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return `<span class="badge badge-na">N/A</span>`;
  const n = Number(v);
  if (n >= 65) return `<span class="badge badge-opp-high">${Math.round(n)}</span>`;
  if (n >= 50) return `<span class="badge badge-opp-med">${Math.round(n)}</span>`;
  return `<span class="badge badge-opp-low">${Math.round(n)}</span>`;
}

function probColor(p){
  if (p === null || p === undefined || !Number.isFinite(Number(p))) return "var(--muted)";
  const v = Number(p);
  if (v >= 75) return "var(--green)";
  if (v >= 60) return "#3d6b3d";
  if (v >= 50) return "var(--ink)";
  return "#8b4513";
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

  ctx.lineWidth = 2.2 * devicePixelRatio;
  ctx.strokeStyle = "rgba(0,0,0,.35)";
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
    if (!Number.isFinite(s)) continue;
    const a = clamp01(s / 100);
    if (a <= 0.02) continue;
    ctx.lineWidth = 3.4 * devicePixelRatio;
    ctx.strokeStyle = `rgba(15,61,46,${0.18 + 0.70 * a})`;
    ctx.beginPath();
    ctx.moveTo(xAt(i), yAt(prices[i]));
    ctx.lineTo(xAt(i + 1), yAt(prices[i + 1]));
    ctx.stroke();
  }

  const lastScore = Number(score?.[n - 1]);
  const a = clamp01((Number.isFinite(lastScore) ? lastScore : 0) / 100);
  ctx.fillStyle = `rgba(15,61,46,${0.25 + 0.70 * a})`;
  ctx.strokeStyle = "rgba(0,0,0,.85)";
  ctx.lineWidth = 1.2 * devicePixelRatio;
  ctx.beginPath();
  ctx.arc(xAt(n - 1), yAt(prices[n - 1]), 4.3 * devicePixelRatio, 0, Math.PI * 2);
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
          <div class="proof-bar-track"><div class="proof-bar-fill" style="width:${v}%;background:${v >= 60 ? 'var(--green)' : v >= 40 ? 'var(--ink)' : '#8b4513'}"></div></div>
          <strong class="proof-val">${v}</strong>
        </div>`;
    }
    if (hasRecov){
      const v = Math.round(Number(qp.recovery));
      barsHtml += `
        <div class="proof-row">
          <span class="proof-label">Does it bounce back from drops? (35%)</span>
          <div class="proof-bar-track"><div class="proof-bar-fill" style="width:${v}%;background:${v >= 60 ? 'var(--green)' : v >= 40 ? 'var(--ink)' : '#8b4513'}"></div></div>
          <strong class="proof-val">${v}</strong>
        </div>`;
    }
    if (hasMom){
      const v = Math.round(Number(qp.momentum));
      barsHtml += `
        <div class="proof-row">
          <span class="proof-label">Is the selling slowing down? (20%)</span>
          <div class="proof-bar-track"><div class="proof-bar-fill" style="width:${v}%;background:${v >= 60 ? 'var(--green)' : v >= 40 ? 'var(--ink)' : '#8b4513'}"></div></div>
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
          <div class="proof-recov-stat"><span>Bounced back</span><strong style="color:${rate >= 75 ? 'var(--green)' : rate >= 50 ? 'var(--ink)' : '#8b4513'}">${rate}% of the time</strong></div>
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

/* ---------- card body (chart + outcomes + evidence) ---------- */

function renderCardBody(body, item, detail, isTop10){
  const series = detail?.series || {};

  // Rationale for top 10
  if (isTop10){
    const rat = document.createElement("div");
    rat.className = "rationale";
    rat.innerHTML = buildRationale(item);
    body.appendChild(rat);
  }

  // Grid: outcomes table left, chart right
  const grid = document.createElement("div");
  grid.className = "grid2";

  const left = document.createElement("div");
  const tbl = outcomesTable(detail?.outcomes);
  if (tbl) left.appendChild(tbl);

  const right = document.createElement("div");
  right.className = "chart";
  const canvas = document.createElement("canvas");
  canvas.className = "canvas";
  right.appendChild(canvas);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = `<span class="legend-bar" aria-hidden="true"></span><span class="legend-text"><span class="legend-label">Pullback intensity</span><span class="legend-note">darker = deeper pullback</span></span>`;
  right.appendChild(legend);

  grid.appendChild(left);
  grid.appendChild(right);
  body.appendChild(grid);

  // Quality proof
  const proof = proofSection(detail);
  if (proof) body.appendChild(proof);

  // Evidence
  const ev = evidenceSection(detail);
  if (ev) body.appendChild(ev);

  // Draw chart
  if (series.prices && series.prices.length){
    requestAnimationFrame(() => drawGradientLine(canvas, series.dates, series.prices, series.wash));
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
        <span class="row-ticker">${item.ticker}<span class="ondemand-label">Live analysis</span></span>
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

  } catch (err){
    console.error("Rebound Ledger render error:", err);
    byId("listing").innerHTML = `<div class="footnote" style="color:#b00">Render error: ${err.message}</div>`;
  }
})();
