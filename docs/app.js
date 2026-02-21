
const DATA_URL = "./data/full.json";

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
  // For values already in 0-100 scale (like prob_1y, quality)
  if (x === null || x === undefined || Number.isNaN(x)) return "\u2014";
  const v = Number(x);
  if (!Number.isFinite(v)) return "\u2014";
  return `${Math.round(v)}%`;
}

function fmtSignedPct(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "\u2014";
  const v = Number(x);
  if (!Number.isFinite(v)) return "\u2014";
  const s = Math.round(v * 100);
  return `${s > 0 ? "+" : ""}${s}%`;
}

function fmtNum0(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "\u2014";
  const v = Number(x);
  if (!Number.isFinite(v)) return "\u2014";
  return v.toFixed(0);
}

function fmtNum1(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "\u2014";
  const v = Number(x);
  if (!Number.isFinite(v)) return "\u2014";
  return v.toFixed(1);
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

function qualityBadge(q){
  if (q === null || q === undefined || !Number.isFinite(Number(q))) return `<span class="badge badge-na">N/A</span>`;
  const v = Number(q);
  if (v >= 70) return `<span class="badge badge-high">High ${Math.round(v)}</span>`;
  if (v >= 45) return `<span class="badge badge-med">Med ${Math.round(v)}</span>`;
  return `<span class="badge badge-low">Low ${Math.round(v)}</span>`;
}

function probColor(p){
  // Returns a CSS color based on probability (0-100)
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

/* ---------- outcome box ---------- */

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
      <div class="ev-cols-header"><span>After pullback</span><span>Any day</span></div>
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
          <span class="proof-label">Trend (45%)</span>
          <div class="proof-bar-track"><div class="proof-bar-fill" style="width:${v}%;background:${v >= 60 ? 'var(--green)' : v >= 40 ? 'var(--ink)' : '#8b4513'}"></div></div>
          <strong class="proof-val">${v}</strong>
        </div>`;
    }
    if (hasRecov){
      const v = Math.round(Number(qp.recovery));
      barsHtml += `
        <div class="proof-row">
          <span class="proof-label">Recovery (35%)</span>
          <div class="proof-bar-track"><div class="proof-bar-fill" style="width:${v}%;background:${v >= 60 ? 'var(--green)' : v >= 40 ? 'var(--ink)' : '#8b4513'}"></div></div>
          <strong class="proof-val">${v}</strong>
        </div>`;
    }
    if (hasMom){
      const v = Math.round(Number(qp.momentum));
      barsHtml += `
        <div class="proof-row">
          <span class="proof-label">Momentum (20%)</span>
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
        <div class="proof-recov-title">Drawdown Recovery Track Record</div>
        <div class="proof-recov-stats">
          <div class="proof-recov-stat"><span>Recovery rate</span><strong style="color:${rate >= 75 ? 'var(--green)' : rate >= 50 ? 'var(--ink)' : '#8b4513'}">${rate}%</strong></div>
          ${nDd != null ? `<div class="proof-recov-stat"><span>20%+ drawdowns found</span><strong>${nDd}</strong></div>` : ""}
          ${nRec != null ? `<div class="proof-recov-stat"><span>Recovered within 3Y</span><strong>${nRec}</strong></div>` : ""}
        </div>
      </div>`;
  }

  if (!barsHtml && !recovHtml) return null;

  box.innerHTML = `
    <summary class="details-summary">
      <div class="proof-summary-left">
        <span class="section-title">QUALITY PROOF</span>
        <span class="ev-sub">Trend health, recovery history &amp; selling momentum</span>
      </div>
      <span class="plus" aria-hidden="true">+</span>
    </summary>
    <div class="details-body">
      ${barsHtml}
      ${recovHtml}
    </div>
  `;

  return box;
}

/* ---------- card rendering ---------- */

function renderCard(container, item, detail){
  const series = detail?.series || {};

  const card = document.createElement("div");
  card.className = "card";

  const prob1y = item.prob_1y;
  const prob3y = item.prob_3y;
  const prob5y = item.prob_5y;
  const quality = item.quality;
  const conviction = item.conviction;

  const h = document.createElement("div");
  h.className = "card-head";
  h.innerHTML = `
    <div>
      <div class="ticker">${item.ticker}</div>
      <div class="card-badges">
        ${oppBadge(conviction)}
        <span class="quality-text">Quality ${quality != null ? Math.round(Number(quality)) : "\u2014"}</span>
      </div>
    </div>
    <div class="metrics">
      <div class="metric">
        <div class="mline"><span>1Y Prob</span> <strong style="color:${probColor(prob1y)};font-size:16px">${fmtPctWhole(prob1y)}</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>3Y Prob</span> <strong style="color:${probColor(prob3y)}">${fmtPctWhole(prob3y)}</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>5Y Prob</span> <strong style="color:${probColor(prob5y)}">${fmtPctWhole(prob5y)}</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>Typical 1Y</span> <strong>${fmtPct(item.median_1y)}</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>Downside</span> <strong>${fmtPct(item.downside_1y)}</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>Past Cases</span> <strong>${fmtNum0(item.n_analogs)}</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>Pullback</span> <strong>${fmtNum0(item.washout_today)}/100</strong></div>
      </div>
    </div>
  `;
  card.appendChild(h);

  const grid = document.createElement("div");
  grid.className = "grid2";

  const left = document.createElement("div");

  // Explain bullets
  const ul = document.createElement("ul");
  ul.className = "bullets";
  for (const line of (detail?.explain || [])){
    const li = document.createElement("li");
    li.innerHTML = line;
    ul.appendChild(li);
  }
  if (!ul.children.length){
    const li = document.createElement("li");
    li.innerHTML = "No pullback details available for this stock today.";
    ul.appendChild(li);
  }
  left.appendChild(ul);

  // Outcome boxes
  const outcomes = document.createElement("div");
  outcomes.className = "outcomes";
  outcomes.appendChild(outcomeBox("1 Year", detail?.outcomes?.["1Y"]));
  outcomes.appendChild(outcomeBox("3 Years", detail?.outcomes?.["3Y"]));
  outcomes.appendChild(outcomeBox("5 Years", detail?.outcomes?.["5Y"]));
  left.appendChild(outcomes);

  // Chart
  const right = document.createElement("div");
  right.className = "chart";
  const canvas = document.createElement("canvas");
  canvas.className = "canvas";
  right.appendChild(canvas);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = `<span class="legend-bar" aria-hidden="true"></span><span class="legend-text"><span class="legend-label">Opportunity Score</span><span class="legend-note">darker = higher score at that point in time</span></span>`;
  right.appendChild(legend);

  grid.appendChild(left);
  grid.appendChild(right);
  card.appendChild(grid);

  // Quality proof
  const proof = proofSection(detail);
  if (proof) card.appendChild(proof);

  // Evidence
  const ev = evidenceSection(detail);
  if (ev) card.appendChild(ev);

  if (series.prices && series.prices.length){
    requestAnimationFrame(() => drawGradientLine(canvas, series.dates, series.prices, series.final));
  }

  container.appendChild(card);
}

/* ---------- table row ---------- */

function rowHtml(item){
  const t = item.ticker;
  return `
    <tr data-ticker="${t}">
      <td class="tcell">${t}</td>
      <td class="num">${oppBadge(item.conviction)}</td>
      <td class="num">${fmtNum0(item.washout_today)}</td>
      <td class="num" style="color:${probColor(item.prob_1y)};font-weight:600">${fmtPctWhole(item.prob_1y)}</td>
      <td class="num" style="color:${probColor(item.prob_3y)}">${fmtPctWhole(item.prob_3y)}</td>
      <td class="num" style="color:${probColor(item.prob_5y)}">${fmtPctWhole(item.prob_5y)}</td>
      <td class="num">${fmtPct(item.median_1y)}</td>
      <td class="num">${fmtPct(item.downside_1y)}</td>
      <td class="num">${fmtNum0(item.n_analogs)}</td>
      <td class="num">${fmtNum0(item.quality)}</td>
    </tr>
  `;
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
    case "quality":
      list.sort((a, b) => safeNum(b.quality) - safeNum(a.quality) || safeNum(b.prob_1y) - safeNum(a.prob_1y));
      break;
    case "conviction":
      list.sort((a, b) => safeNum(b.conviction) - safeNum(a.conviction) || safeNum(b.prob_1y) - safeNum(a.prob_1y));
      break;
    case "washout_today":
      list.sort((a, b) => safeNum(b.washout_today) - safeNum(a.washout_today));
      break;
    case "median_1y":
      list.sort((a, b) => safeNum(b.median_1y) - safeNum(a.median_1y));
      break;
    case "downside_1y":
      list.sort((a, b) => safeNum(b.downside_1y) - safeNum(a.downside_1y));
      break;
    case "n_analogs":
      list.sort((a, b) => safeNum(b.n_analogs) - safeNum(a.n_analogs));
      break;
    case "ticker":
      list.sort((a, b) => (a.ticker || "").localeCompare(b.ticker || ""));
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
    byId("top10").innerHTML = `<div class="footnote">No data available yet. The daily scan has not run.</div>`;
    return;
  }
  try {

  byId("asOf").textContent = formatAsOf(full.as_of);

  let items = full.items || [];
  let sortMode = "conviction";

  function renderTable(list){
    byId("rows").innerHTML = list.map(it => rowHtml(it)).join("");
  }

  async function loadDetail(ticker){
    const embedded = (full.details && full.details[ticker]) ? full.details[ticker] : null;
    if (embedded) return embedded;
    return await loadJSON(`./data/tickers/${ticker}.json`);
  }

  async function renderTop10(list){
    const c = byId("top10");
    c.innerHTML = "";
    const top = list.slice(0, 10);
    if (top.length === 0){
      c.innerHTML = `<div class="footnote">No stocks meet the conviction threshold today.</div>`;
      return;
    }
    for (const it of top){
      let detail;
      try {
        detail = await loadDetail(it.ticker);
      } catch (err){
        detail = {
          explain: [`Details unavailable for <strong>${it.ticker}</strong>. Try refreshing the page.`],
          outcomes: {},
          series: {},
        };
      }
      renderCard(c, it, detail);
    }
  }

  // Top convictions: quality >= 60, prob_1y >= 65, AND a meaningful pullback.
  // Without a pullback there's no rebound opportunity — just a good stock at normal prices.
  const MIN_PULLBACK_FOR_CONVICTION = 20;

  function getConvictions(list){
    return list.filter(it => {
      const q = safeNum(it.quality);
      const p = safeNum(it.prob_1y);
      const w = safeNum(it.washout_today);
      return q >= 60 && p >= 65 && w >= MIN_PULLBACK_FOR_CONVICTION;
    }).sort((a, b) => safeNum(b.conviction) - safeNum(a.conviction));
  }

  async function rerender(){
    const sorted = sortItems(items, sortMode);
    renderTable(sorted);

    // Top section shows convictions (quality >= 60 AND prob_1y >= 65)
    const convictions = getConvictions(sorted);
    if (convictions.length > 0){
      await renderTop10(convictions);
    } else {
      // Fallback: show top 10 by current sort
      await renderTop10(sorted);
      byId("convictionNote").textContent = "No stocks meet the opportunity threshold today \u2014 showing top 10 by current sort";
    }
  }

  // Sort button clicks
  document.querySelectorAll(".sort .btn-lite").forEach(btn => {
    btn.addEventListener("click", async () => {
      sortMode = btn.dataset.sort;
      setSortButtons(sortMode);
      await rerender();
    });
  });

  // Column header clicks for sorting
  document.querySelectorAll("th.sortable").forEach(th => {
    th.style.cursor = "pointer";
    th.addEventListener("click", async () => {
      const col = th.dataset.col;
      if (col){
        sortMode = col;
        setSortButtons(sortMode);
        await rerender();
      }
    });
  });

  setSortButtons(sortMode);
  await rerender();

  // Table row click -> show that ticker's card at top
  byId("rows").addEventListener("click", async (e) => {
    const tr = e.target.closest("tr");
    if (!tr) return;
    const t = tr.dataset.ticker;
    if (!t) return;

    document.querySelectorAll("#rows tr").forEach(r => r.classList.remove("highlight"));
    tr.classList.add("highlight");

    const sorted = sortItems(items, sortMode);
    const idx = sorted.findIndex(x => x.ticker === t);
    if (idx < 0) return;
    const rotated = [sorted[idx], ...sorted.filter((_, i) => i !== idx)];
    await renderTop10(rotated);
    document.querySelector(".masthead").scrollIntoView({ behavior: "smooth" });
  });

  // Search
  function applySearch(){
    const q = (byId("q").value || "").trim().toUpperCase();
    if (!q){
      rerender();
      return;
    }
    const sorted = sortItems(items, sortMode);
    const filtered = sorted.filter(x => x.ticker.includes(q));
    renderTable(filtered);
    (async () => { await renderTop10(filtered); })();
  }

  byId("go").addEventListener("click", applySearch);
  byId("q").addEventListener("input", applySearch);

  } catch (err){
    console.error("Rebound Ledger render error:", err);
    byId("top10").innerHTML = `<div class="footnote" style="color:#b00">Render error: ${err.message}</div>`;
  }
})();
