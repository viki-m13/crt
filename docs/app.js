
const DATA_URL = "./data/full.json";

// One cache-buster per page load so *every* JSON request bypasses GitHub Pages/CDN caches.
// (Stable per-load keeps URLs consistent across all requests during a single page render.)
const CACHE_BUST = String(Date.now());

function withBust(url){
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}v=${encodeURIComponent(CACHE_BUST)}`;
}

function fmtPct(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${Math.round(v*100)}%`;
}

function fmtSignedPct(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  const s = Math.round(v*100);
  const sign = (s>0?"+":"");
  return `${sign}${s}%`;
}

function fmtPP(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  const s = Math.round(v*100);
  const sign = (s>0?"+":"");
  return `${sign}${s}pp`;
}

function fmtNum0(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${v.toFixed(0)}`;
}

function fmtNum1(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${v.toFixed(1)}`;
}

function clamp01(x){ return Math.max(0, Math.min(1, x)); }

function washoutTopPctFromSeries(wash){
  const arr = (wash || []).map(Number).filter(v => Number.isFinite(v));
  if (arr.length < 60) return null;
  const v = arr[arr.length - 1];
  let le = 0;
  for (const x of arr){ if (x <= v) le++; }
  const pct = le / arr.length;              // percentile (higher = more washed-out)
  const topPct = (1 - pct) * 100;           // "top X%" most washed-out days
  return topPct;
}

function finalScoreTopPctFromSeries(final){
  const arr = (final || []).map(Number).filter(v => Number.isFinite(v));
  if (arr.length < 60) return null;
  const v = arr[arr.length - 1];
  let le = 0;
  for (const x of arr){ if (x <= v) le++; }
  const pct = le / arr.length;              // percentile (higher = stronger signal)
  const topPct = (1 - pct) * 100;           // "top X%" strongest Final Score days
  return topPct;
}


function washoutTopRankText(topPct){
  if (topPct === null || topPct === undefined || Number.isNaN(topPct)) return null;
  const v = Number(topPct);
  if (!Number.isFinite(v)) return null;
  if (v < 1) return "Top <1%";
  return `Top ${Math.round(v)}%`;
}

function byId(id){ return document.getElementById(id); }

// Gradient line: dark green = higher score (0..100)
function drawGradientLine(canvas, dates, prices, score){
  const ctx = canvas.getContext("2d");
  const w = canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
  const h = canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
  ctx.clearRect(0,0,w,h);

  const n = prices.length;
  if (n < 3) return;

  const pad = 12 * devicePixelRatio;
  let minP = Infinity, maxP = -Infinity;
  for (let i=0;i<n;i++){
    const p = Number(prices[i]);
    if (!Number.isFinite(p)) continue;
    if (p<minP) minP=p;
    if (p>maxP) maxP=p;
  }
  if (!(maxP>minP)) return;

  const x0=pad, x1=w-pad, y0=pad, y1=h-pad;
  function xAt(i){ return x0 + (x1-x0) * (i/(n-1)); }
  function yAt(p){ return y1 - (y1-y0) * ((p-minP)/(maxP-minP)); }

  // Base price line (thin)
  ctx.lineWidth = 2.2*devicePixelRatio;
  ctx.strokeStyle = "rgba(0,0,0,.35)";
  ctx.beginPath();
  for (let i=0;i<n;i++){
    const p = Number(prices[i]);
    if (!Number.isFinite(p)) continue;
    const x=xAt(i), y=yAt(p);
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();

  // Score overlay (thicker; darker with higher score)
  for (let i=0;i<n-1;i++){
    const s = Number(score?.[i]);
    if (!Number.isFinite(s)) continue;
    const a = clamp01(s/100);
    if (a <= 0.02) continue;
    ctx.lineWidth = 3.4*devicePixelRatio;
    ctx.strokeStyle = `rgba(15,61,46,${0.18 + 0.70*a})`;
    ctx.beginPath();
    ctx.moveTo(xAt(i), yAt(prices[i]));
    ctx.lineTo(xAt(i+1), yAt(prices[i+1]));
    ctx.stroke();
  }

  // Today marker (colored by score)
  const lastScore = Number(score?.[n-1]);
  const a = clamp01((Number.isFinite(lastScore)?lastScore:0)/100);
  ctx.fillStyle = `rgba(15,61,46,${0.25 + 0.70*a})`;
  ctx.strokeStyle = "rgba(0,0,0,.85)";
  ctx.lineWidth = 1.2*devicePixelRatio;
  ctx.beginPath();
  ctx.arc(xAt(n-1), yAt(prices[n-1]), 4.3*devicePixelRatio, 0, Math.PI*2);
  ctx.fill(); ctx.stroke();
}

function outcomeBox(label, s){
  const b = document.createElement("div");
  b.className = "outbox";
  if (!s || !Number.isFinite(s.n) || s.n<=0){
    b.innerHTML = `<div class="h">${label}</div><div class="r"><span>Not enough</span><strong>—</strong></div>`;
    return b;
  }
  b.innerHTML = `
    <div class="h">${label}</div>
    <div class="r"><span>Chance of gain</span><strong>${Math.round(s.win*100)}%</strong></div>
    <div class="r"><span>Typical</span><strong>${fmtPct(s.median)}</strong></div>
    <div class="r"><span>Downside (1 in 10)</span><strong>${fmtPct(s.p10)}</strong></div>
    <div class="r"><span>Based on N</span><strong>${s.n}</strong></div>
  `;
  return b;
}

function evidenceSection(kind, ev){
  // kind: "A" (washout) or "B" (final)
  const title = (kind === "A") ? "EVIDENCE A" : "EVIDENCE B";
  const subtitle = (kind === "A")
    ? "Top 10% Washout Meter days vs normal"
    : "Top 10% Final Score days vs normal";

  const explain = (kind === "A")
    ? `This is a broad A/B check across the stock’s entire history. <strong>A</strong> = the <strong>10% most washed‑out</strong> days for this stock. <strong>B</strong> = a <strong>normal</strong> historical day (baseline). Each line is written as <strong>A vs B</strong>. <span class="mono">pp</span> = percentage points.`
    : `Same idea, but the “signal” days are defined by the <strong>same Final Score used for ranking</strong>. <strong>A</strong> = the top‑decile Final Score days for this stock. <strong>B</strong> = a normal historical day. Each line is written as <strong>A vs B</strong>. <span class="mono">pp</span> = percentage points.`;

  const box = document.createElement("details");
  box.className = "details evidence-details";
  box.innerHTML = `
    <summary class="details-summary">
      <div class="evidence-summary-left">
        <span class="section-title">${title}</span>
        <span class="ev-sub">${subtitle}</span>
      </div>
      <span class="plus" aria-hidden="true">+</span>
    </summary>
    <div class="details-body">
      <div class="ev-explain">${explain}</div>
      <div class="outcomes ev-grid"></div>
    </div>
  `;

  const gridEl = box.querySelector(".ev-grid");
  const horizons = [["1Y","1 year"],["3Y","3 years"],["5Y","5 years"]];

  for (const [k,label] of horizons){
    const e = ev?.[k];
    if (!e) continue;

    const winA = (kind === "A") ? e.win_wash : e.win_top;
    const winB = e.win_norm;
    const medA = (kind === "A") ? e.med_wash : e.med_top;
    const medB = e.med_norm;
    const p10A = (kind === "A") ? e.p10_wash : e.p10_top;
    const p10B = e.p10_norm;
    const nA   = (kind === "A") ? e.n_wash : e.n_top;
    const nB   = e.n_norm;

    const dWin = (winA - winB);
    const dMed = (medA - medB);
    const dP10 = (p10A - p10B);

    const b = document.createElement("div");
    b.className = "outbox";
    b.innerHTML = `
      <div class="h">${label}</div>
      <div class="r"><span>Chance of gain</span><strong>${Math.round(winA*100)}% vs ${Math.round(winB*100)}% (${fmtPP(dWin)})</strong></div>
      <div class="r"><span>Typical</span><strong>${fmtPct(medA)} vs ${fmtPct(medB)} (${fmtSignedPct(dMed)})</strong></div>
      <div class="r"><span>Downside (1 in 10)</span><strong>${fmtPct(p10A)} vs ${fmtPct(p10B)} (${fmtSignedPct(dP10)})</strong></div>
      <div class="r"><span>N</span><strong>${nA} vs ${nB}</strong></div>
    `;
    gridEl.appendChild(b);
  }

  if (!gridEl.children.length) return null;
  return box;
}

function renderCard(container, item, detail){
  const series = detail.series || {};

  // Washed-out rank (higher washout = more washed-out)
  const topPctFromItem = (item.washout_top_pct != null && Number.isFinite(item.washout_top_pct))
    ? Number(item.washout_top_pct)
    : null;
  const topPctFromSeries = washoutTopPctFromSeries(series.wash);
  const topPct = (topPctFromItem != null) ? topPctFromItem : topPctFromSeries;
  const washRank = washoutTopRankText(topPct) || "—";

  // Final-score rank (higher final = stronger signal)
  const finalTopPctFromItem = (item.final_score_top_pct != null && Number.isFinite(item.final_score_top_pct))
    ? Number(item.final_score_top_pct)
    : null;
  const finalTopPctSeries = finalScoreTopPctFromSeries(series.final);
  const finalTopPct = (finalTopPctFromItem != null) ? finalTopPctFromItem : finalTopPctSeries;
  const finalRank = washoutTopRankText(finalTopPct) || "—";

  const card = document.createElement("div");
  card.className = "card";

  const h = document.createElement("div");
  h.className = "card-head";
  h.innerHTML = `
    <div>
      <div class="ticker">${item.ticker}</div>
      <div class="verdict">${item.verdict}</div>
    </div>
    <div class="metrics">
      <div class="metric"><span>Final score</span> <strong>${fmtNum1(item.final_score)}</strong></div>
      <div class="metric"><span>Final‑score rank</span> <strong>${finalRank}</strong></div>
      <div class="metric"><span>Wash</span> <strong>${fmtNum0(item.washout_today)}</strong></div>
      <div class="metric"><span>Edge</span> <strong>${fmtNum1(item.edge_score)}</strong></div>
      <div class="metric"><span>Conf</span> <strong>${fmtNum0(item.confidence)}</strong></div>
      <div class="metric"><span>Stab</span> <strong>${fmtNum0(item.stability)}</strong></div>
      <div class="metric"><span>Washed‑out rank</span> <strong>${washRank}</strong></div>
      <div class="metric"><span>Risk</span> <strong>${item.risk || "—"}</strong></div>
    </div>
  `;
  card.appendChild(h);

  const grid = document.createElement("div");
  grid.className = "grid2";

  const left = document.createElement("div");

  const ul = document.createElement("ul");
  ul.className = "bullets";
  for (const line of (detail.explain || [])){
    const li = document.createElement("li");
    li.innerHTML = line;
    ul.appendChild(li);
  }
  if (!ul.children.length){
    const li = document.createElement("li");
    li.innerHTML = "No plain‑English explanation available for this ticker today.";
    ul.appendChild(li);
  }
  left.appendChild(ul);

  const outcomes = document.createElement("div");
  outcomes.className = "outcomes";
  outcomes.appendChild(outcomeBox("1 year", detail.outcomes?.["1Y"]));
  outcomes.appendChild(outcomeBox("3 years", detail.outcomes?.["3Y"]));
  outcomes.appendChild(outcomeBox("5 years", detail.outcomes?.["5Y"]));
  left.appendChild(outcomes);

  const right = document.createElement("div");
  right.className = "chart";
  const canvas = document.createElement("canvas");
  canvas.className = "canvas";
  right.appendChild(canvas);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = `<span class="legend-bar" aria-hidden="true"></span><span class="legend-label">Final Score</span><span class="legend-note">higher → darker</span>`;
  right.appendChild(legend);

  grid.appendChild(left);
  grid.appendChild(right);
  card.appendChild(grid);

  // Evidence A / B
  const evA = detail.evidence_washout || null;
  const evB = detail.evidence_finalscore || null;
  const secA = evidenceSection("A", evA);
  const secB = evidenceSection("B", evB);
  if (secA) card.appendChild(secA);
  if (secB) card.appendChild(secB);

  if (series && series.prices && series.prices.length){
    requestAnimationFrame(()=>drawGradientLine(canvas, series.dates, series.prices, series.final));
  }

  container.appendChild(card);
}

function rowHtml(item){
  const t = item.ticker;
  const t1 = item.typical?.["1Y"];
  const t3 = item.typical?.["3Y"];
  const t5 = item.typical?.["5Y"];
  return `
    <tr data-ticker="${t}">
      <td class="tcell">${t}</td>
      <td>${item.verdict}</td>
      <td class="num">${fmtNum1(item.final_score)}</td>
      <td class="num">${fmtNum0(item.washout_today)}</td>
      <td class="num">${fmtNum1(item.edge_score)}</td>
      <td class="num">${fmtNum0(item.confidence)}</td>
      <td class="num">${fmtNum0(item.stability)}</td>
      <td>${item.risk || "—"}</td>
      <td class="num">${fmtPct(t1)}</td>
      <td class="num">${fmtPct(t3)}</td>
      <td class="num">${fmtPct(t5)}</td>
    </tr>
  `;
}

async function loadJSON(url){
  // Force a unique URL + ask the browser not to cache the response.
  // This avoids stale JSON on GitHub Pages/CDNs after Actions pushes new data.
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

function setSortButtons(active){
  document.querySelectorAll(".btn-lite").forEach(b=>{
    b.classList.toggle("active", b.dataset.sort === active);
  });
}

function formatAsOf(asOf){
  if (!asOf) return "—";
  let s = String(asOf).trim();
  // Accept "YYYY-MM-DD HH:MM:SS...-05:00" and ISO variants.
  if (s.includes(" ") && !s.includes("T")) s = s.replace(" ", "T");
  let d = new Date(s);
  if (Number.isNaN(d.getTime())){
    // Last resort: try stripping fractional seconds
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

  const get = (type)=> (parts.find(p=>p.type===type)?.value || "");
  const yyyy = get("year");
  const mm = get("month");
  const dd = get("day");
  const hh = get("hour");
  const min = get("minute");
  const ap = get("dayPeriod");
  return `${yyyy}-${mm}-${dd} ${hh}:${min} ${ap} EST`;
}

(async function main(){
  let full;
  try{
    full = await loadJSON(DATA_URL);
  }catch(e){
    byId("top10").innerHTML = `<div class="footnote">No data yet. Run the GitHub Action to generate <span class="mono">docs/data/full.json</span>.</div>`;
    return;
  }

  byId("asOf").textContent = formatAsOf(full.as_of);

  let items = full.items || [];
  let sortMode = "final";

  function renderTable(list){
    byId("rows").innerHTML = list.map(rowHtml).join("");
  }

  async function loadDetail(ticker){
    const embedded = (full.details && full.details[ticker]) ? full.details[ticker] : null;
    if (embedded) return embedded;
    return await loadJSON(`./data/tickers/${ticker}.json`);
  }

  async function renderTop10(list){
    const c = byId("top10");
    c.innerHTML = "";
    const top = list.slice(0,10);
    for (const it of top){
      let detail;
      try{
        detail = await loadDetail(it.ticker);
      }catch(err){
        // Don't let one missing/stale ticker JSON blank the entire Top 10 section.
        detail = {
          explain: [
            `⚠️ Detail JSON failed to load for <strong>${it.ticker}</strong>.`,
            `This is almost always a caching or deploy timing issue. Try a hard refresh, or wait a minute and reload.`,
          ],
          outcomes: {},
          series: {},
        };
      }
      renderCard(c, it, detail);
    }
  }

  function applySort(){
    const list = [...items];
    if (sortMode === "final"){
      list.sort((a,b)=> (b.final_score - a.final_score) || (b.washout_today - a.washout_today) || (b.edge_score - a.edge_score));
    }else if (sortMode === "washout"){
      // higher washout_today = more washed-out
      list.sort((a,b)=> (b.washout_today - a.washout_today) || (b.final_score - a.final_score) || (b.edge_score - a.edge_score));
    }else if (sortMode === "edge"){
      list.sort((a,b)=> (b.edge_score - a.edge_score) || (b.final_score - a.final_score) || (b.washout_today - a.washout_today));
    }
    return list;
  }

async function rerender(){
    const list = applySort();
    renderTable(list);
    await renderTop10(list);
  }

  document.querySelectorAll(".btn-lite").forEach(btn=>{
    btn.addEventListener("click", async ()=>{
      sortMode = btn.dataset.sort;
      setSortButtons(sortMode);
      await rerender();
    });
  });
  setSortButtons(sortMode);

  await rerender();

  byId("rows").addEventListener("click", async (e)=>{
    const tr = e.target.closest("tr");
    if (!tr) return;
    const t = tr.dataset.ticker;
    if (!t) return;

    document.querySelectorAll("#rows tr").forEach(r=>r.classList.remove("highlight"));
    tr.classList.add("highlight");

    // scroll to top10 and highlight there by re-rendering top10 with that ticker first
    const current = applySort();
    const idx = current.findIndex(x=>x.ticker===t);
    if (idx < 0) return;
    const rotated = [current[idx], ...current.filter((_,i)=>i!==idx)];
    await renderTop10(rotated);
    document.querySelector(".masthead").scrollIntoView({behavior:"smooth"});
  });

  function applySearch(){
    const q = (byId("q").value || "").trim().toUpperCase();
    if (!q){
      rerender();
      return;
    }
    const filtered = applySort().filter(x=>x.ticker.includes(q));
    renderTable(filtered);
    // show top10 as the best 10 from the filtered list
    (async ()=>{ await renderTop10(filtered); })();
  }

  byId("go").addEventListener("click", applySearch);
  byId("q").addEventListener("input", ()=>{
    // lightweight live filter
    applySearch();
  });
})();
