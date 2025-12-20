
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

function finalTopPctFromSeries(final){
  const arr = (final || []).map(Number).filter(v => Number.isFinite(v));
  if (arr.length < 60) return null;
  const v = arr[arr.length - 1];
  let le = 0;
  for (const x of arr){ if (x <= v) le++; }
  const pct = le / arr.length;              // percentile (higher = stronger signal)
  const topPct = (1 - pct) * 100;           // "top X%" strongest Final Score days
  return topPct;
}

function upperBound(sortedArr, v){
  // index of first element > v
  let lo = 0, hi = sortedArr.length;
  while (lo < hi){
    const mid = (lo + hi) >> 1;
    if (sortedArr[mid] <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function quantileSorted(sortedArr, q){
  if (!sortedArr || !sortedArr.length) return null;
  const n = sortedArr.length;
  const qq = Math.max(0, Math.min(1, Number(q)));
  const pos = (n - 1) * qq;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sortedArr[lo];
  const w = pos - lo;
  return sortedArr[lo] * (1 - w) + sortedArr[hi] * w;
}

function topPctFromValue(sortedArr, v){
  if (!sortedArr || sortedArr.length < 60) return null;
  const vv = Number(v);
  if (!Number.isFinite(vv)) return null;
  const le = upperBound(sortedArr, vv);
  const pct = le / sortedArr.length;
  return (1 - pct) * 100;
}

function fmtRange(a, b, decimals=1){
  const x = Number(a), y = Number(b);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return "—";
  const fx = decimals === 0 ? x.toFixed(0) : x.toFixed(1);
  const fy = decimals === 0 ? y.toFixed(0) : y.toFixed(1);
  return `${fx}–${fy}`;
}

function verdictFromRanks(finalTopPct, washTopPct){
  const f = (finalTopPct != null && Number.isFinite(finalTopPct)) ? Number(finalTopPct) : null;
  const w = (washTopPct != null && Number.isFinite(washTopPct)) ? Number(washTopPct) : null;
  const inF = (f != null && f <= 10);
  const inW = (w != null && w <= 10);
  if (inF && inW) return "Signal today";
  if (inF) return "Strong setup (final-score signal)";
  if (inW) return "Washed-out (needs edge confirmation)";
  return "Not compelling today";
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
    <div class="r"><span>Based on N similar days</span><strong>${s.n}</strong></div>
  `;
  return b;
}

function evidenceAnalogVsNormalSection(detail){
  // One consistent evidence block:
  //   A = the SAME closest-analog days used to compute the main 1/3/5Y outcomes above
  //   B = a "normal" historical day baseline (unconditional) for this ticker
  const base = detail?.evidence_finalscore || detail?.evidence_washout || null;
  const outs = detail?.outcomes || null;
  if (!base || !outs) return null;

  const box = document.createElement("details");
  box.className = "details evidence-details";
  box.innerHTML = `
    <summary class="details-summary">
      <div class="evidence-summary-left">
        <span class="section-title">EVIDENCE</span>
        <span class="ev-sub">Similar past setups vs normal</span>
      </div>
      <span class="plus" aria-hidden="true">+</span>
    </summary>
    <div class="details-body">
      <div class="ev-explain">
        <strong>A</strong> = the same closest historical “analog” days used for the main recommendation above.
        <strong>B</strong> = a normal historical day baseline for this stock.
        Each line is written as <strong>A vs B</strong>. <span class="mono">pp</span> = percentage points.
      </div>
      <div class="outcomes ev-grid"></div>
    </div>
  `;

  const gridEl = box.querySelector(".ev-grid");
  const horizons = [["1Y","1 year"],["3Y","3 years"],["5Y","5 years"]];

  for (const [k,label] of horizons){
    const a = outs?.[k];
    const b = base?.[k];
    if (!a || !b) continue;

    const winA = a.win,    winB = b.win_norm;
    const medA = a.median, medB = b.med_norm;
    const p10A = a.p10,    p10B = b.p10_norm;
    const nA   = a.n,      nB   = b.n_norm;

    if (!(Number.isFinite(winA) && Number.isFinite(winB) && Number.isFinite(medA) && Number.isFinite(medB) && Number.isFinite(p10A) && Number.isFinite(p10B))) continue;

    const dWin = (winA - winB);
    const dMed = (medA - medB);
    const dP10 = (p10A - p10B);

    const bx = document.createElement("div");
    bx.className = "outbox";
    bx.innerHTML = `
      <div class="h">${label}</div>
      <div class="r"><span>Chance of gain</span><strong>${Math.round(winA*100)}% vs ${Math.round(winB*100)}% (${fmtPP(dWin)})</strong></div>
      <div class="r"><span>Typical</span><strong>${fmtPct(medA)} vs ${fmtPct(medB)} (${fmtSignedPct(dMed)})</strong></div>
      <div class="r"><span>Downside (1 in 10)</span><strong>${fmtPct(p10A)} vs ${fmtPct(p10B)} (${fmtSignedPct(dP10)})</strong></div>
      <div class="r"><span>N</span><strong>${nA} vs ${nB}</strong></div>
    `;
    gridEl.appendChild(bx);
  }

  if (!gridEl.children.length) return null;
  return box;
}

function renderCard(container, item, detail, derived){
  const series = detail.series || {};

  // Derived stats (computed once in main() for consistency across the table + cards)
  const d = derived || {};
  const washRank = d.washRank || "—";
  const finalRank = d.finalRank || "—";
  const washTop10Range = d.washTop10Range || "—";
  const finalTop10Range = d.finalTop10Range || "—";
  const verdict = d.verdict || item.verdict || "—";

  const card = document.createElement("div");
  card.className = "card";

  const h = document.createElement("div");
  h.className = "card-head";
  h.innerHTML = `
    <div>
      <div class="ticker">${item.ticker}</div>
      <div class="verdict">${verdict}</div>
    </div>
    <div class="metrics">
      <div class="metric">
        <div class="mline"><span>Final score</span> <strong>${fmtNum1(item.final_score)}/100</strong></div>
        <div class="msub">Top‑10% range ${finalTop10Range}</div>
      </div>
      <div class="metric">
        <div class="mline"><span>Wash</span> <strong>${fmtNum0(item.washout_today)}/100</strong></div>
        <div class="msub">Top‑10% range ${washTop10Range}</div>
      </div>
      <div class="metric">
        <div class="mline"><span>Final‑score rank</span> <strong>${finalRank}</strong></div>
      </div>
      <div class="metric">
        <div class="mline"><span>Washed‑out rank</span> <strong>${washRank}</strong></div>
      </div>
      <div class="metric"><div class="mline"><span>Edge</span> <strong>${fmtNum1(item.edge_score)}/100</strong></div></div>
      <div class="metric"><div class="mline"><span>Conf</span> <strong>${fmtNum0(item.confidence)}/100</strong></div></div>
      <div class="metric"><div class="mline"><span>Stab</span> <strong>${fmtNum0(item.stability)}/100</strong></div></div>
      <div class="metric"><div class="mline"><span>Risk</span> <strong>${item.risk || "—"}</strong></div></div>
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
  legend.innerHTML = `<span class="legend-bar" aria-hidden="true"></span><span class="legend-text"><span class="legend-label">Final Score</span><span class="legend-note">higher → darker</span></span>`;
  right.appendChild(legend);

  grid.appendChild(left);
  grid.appendChild(right);
  card.appendChild(grid);

  // Evidence (consistent with the main recommendation): analogs vs baseline
  const ev = evidenceAnalogVsNormalSection(detail);
  if (ev) card.appendChild(ev);

  if (series && series.prices && series.prices.length){
    requestAnimationFrame(()=>drawGradientLine(canvas, series.dates, series.prices, series.final));
  }

  container.appendChild(card);
}

function rowHtml(item, derived){
  const t = item.ticker;
  const verdict = derived?.verdict || item.verdict || "—";
  const t1 = item.typical?.["1Y"];
  const t3 = item.typical?.["3Y"];
  const t5 = item.typical?.["5Y"];
  return `
    <tr data-ticker="${t}">
      <td class="tcell">${t}</td>
      <td>${verdict}</td>
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

function renderHistoricalSignals(full, derivedByTicker){
  const host = byId("histRows");
  if (!host) return;

  const signals = [];
  const H1 = 252, H3 = 252*3, H5 = 252*5;

  for (const [ticker, det] of Object.entries(full.details || {})){
    const s = det?.series;
    if (!s || !Array.isArray(s.dates) || !Array.isArray(s.prices) || !Array.isArray(s.final) || !Array.isArray(s.wash)) continue;
    const n = s.dates.length;
    if (n < 260) continue;

    const finalArr = s.final.map(Number).filter(Number.isFinite);
    const washArr  = s.wash.map(Number).filter(Number.isFinite);
    if (finalArr.length < 60 || washArr.length < 60) continue;
    const sortedFinal = [...finalArr].sort((a,b)=>a-b);
    const sortedWash  = [...washArr].sort((a,b)=>a-b);

    // Use full-history cutoffs so the “Top X%” text matches how we describe ranks elsewhere.
    for (let i=0;i<n;i++){
      const fv = s.final[i];
      const wv = s.wash[i];
      const fTop = topPctFromValue(sortedFinal, fv);
      const wTop = topPctFromValue(sortedWash, wv);
      if (!(Number.isFinite(fTop) && Number.isFinite(wTop))) continue;
      if (fTop > 10 || wTop > 10) continue;

      const p0 = Number(s.prices[i]);
      if (!Number.isFinite(p0) || p0 <= 0) continue;

      const r5 = (i+H5 < n) ? (Number(s.prices[i+H5]) / p0 - 1) : null;
      if (!(r5 !== null && Number.isFinite(r5))) continue; // require ≥5Y forward performance

      const r1 = (i+H1 < n) ? (Number(s.prices[i+H1]) / p0 - 1) : null;
      const r3 = (i+H3 < n) ? (Number(s.prices[i+H3]) / p0 - 1) : null;

      signals.push({
        date: String(s.dates[i] || ""),
        ticker,
        final_score: Number(fv),
        wash: Number(wv),
        final_rank: washoutTopRankText(fTop) || "—",
        wash_rank: washoutTopRankText(wTop) || "—",
        r1, r3, r5,
      });
    }
  }

  signals.sort((a,b)=> (b.date.localeCompare(a.date)) || (a.ticker.localeCompare(b.ticker)));
  const last10 = signals.slice(0, 10);

  host.innerHTML = last10.map(s=>{
    return `
      <tr data-ticker="${s.ticker}">
        <td class="mono">${s.date || "—"}</td>
        <td class="tcell">${s.ticker}</td>
        <td class="num">${fmtNum1(s.final_score)}</td>
        <td class="num">${fmtNum0(s.wash)}</td>
        <td>${s.final_rank}</td>
        <td>${s.wash_rank}</td>
        <td class="num">${fmtPct(s.r1)}</td>
        <td class="num">${fmtPct(s.r3)}</td>
        <td class="num">${fmtPct(s.r5)}</td>
      </tr>
    `;
  }).join("");
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
  const derivedByTicker = {};

  // Precompute per-ticker rank text + top-decile score ranges so:
  //   - the Top 10 cards and the All Tickers table stay consistent
  //   - users can see whether a “10.3/100” is actually high for that ticker
  for (const it of items){
    const t = it.ticker;
    const det = full.details?.[t];
    const s = det?.series || {};

    const finalArrRaw = (s.final || []).map(Number).filter(Number.isFinite);
    const washArrRaw  = (s.wash  || []).map(Number).filter(Number.isFinite);

    // For UI stats we want the distributions to include TODAY's displayed values.
    // Some detail series are sparsely computed/ffilled and may not include the exact latest value.
    const finalToday = Number(it.final_score);
    const washToday  = Number(it.washout_today);
    const finalArr = (finalArrRaw.length ? [...finalArrRaw] : []);
    const washArr  = (washArrRaw.length  ? [...washArrRaw]  : []);
    if (Number.isFinite(finalToday)){
      const has = finalArr.some(v => Math.abs(v - finalToday) < 1e-9);
      if (!has) finalArr.push(finalToday);
    }
    if (Number.isFinite(washToday)){
      const has = washArr.some(v => Math.abs(v - washToday) < 1e-9);
      if (!has) washArr.push(washToday);
    }

    const sortedFinal = (finalArr.length ? [...finalArr].sort((a,b)=>a-b) : null);
    const sortedWash  = (washArr.length  ? [...washArr].sort((a,b)=>a-b) : null);

    // Today's ranks (Top X% strongest)
    const finalTopPct = (it.finalscore_top_pct != null && Number.isFinite(it.finalscore_top_pct))
      ? Number(it.finalscore_top_pct)
      : topPctFromValue(sortedFinal, finalToday);
    const washTopPct  = (it.washout_top_pct != null && Number.isFinite(it.washout_top_pct))
      ? Number(it.washout_top_pct)
      : topPctFromValue(sortedWash, washToday);

    const finalRank = washoutTopRankText(finalTopPct) || "—";
    const washRank  = washoutTopRankText(washTopPct)  || "—";

    // Top-decile score ranges (90th percentile .. max)
    const f90 = sortedFinal ? quantileSorted(sortedFinal, 0.90) : null;
    const fmx = sortedFinal ? sortedFinal[sortedFinal.length-1] : null;
    const w90 = sortedWash  ? quantileSorted(sortedWash, 0.90)  : null;
    const wmx = sortedWash  ? sortedWash[sortedWash.length-1]   : null;

    const verdict = verdictFromRanks(finalTopPct, washTopPct);

    derivedByTicker[t] = {
      finalTopPct, washTopPct,
      finalRank, washRank,
      finalTop10Range: fmtRange(f90, fmx, 1),
      washTop10Range: fmtRange(w90, wmx, 0),
      verdict,
    };
  }
  let sortMode = "final";

  function renderTable(list){
    byId("rows").innerHTML = list.map(it=>rowHtml(it, derivedByTicker[it.ticker])).join("");
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
      renderCard(c, it, detail, derivedByTicker[it.ticker]);
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

  // One-time: show the latest 10 historical signals across all tickers.
  renderHistoricalSignals(full, derivedByTicker);

  // Clicking a historical signal row jumps you to that ticker's card.
  const histBody = byId("histRows");
  if (histBody){
    histBody.addEventListener("click", async (e)=>{
      const tr = e.target.closest("tr");
      if (!tr) return;
      const t = tr.dataset.ticker;
      if (!t) return;
      const current = applySort();
      const idx = current.findIndex(x=>x.ticker===t);
      if (idx < 0) return;
      const rotated = [current[idx], ...current.filter((_,i)=>i!==idx)];
      await renderTop10(rotated);
      document.querySelector(".masthead").scrollIntoView({behavior:"smooth"});
    });
  }

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
