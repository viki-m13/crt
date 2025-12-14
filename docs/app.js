const DATA_URL = "data/full.json";

let FULL = null;
let rows = [];
let sortKey = "ReboundScore";
let sortDir = "desc";

const fmt0 = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : String(Math.round(+x));
const fmt1 = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : String((+x).toFixed(1));
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function setStatus(msg){ const el=document.getElementById("status"); if(el) el.textContent = msg || ""; }

function compare(a,b){
  const va = a?.[sortKey];
  const vb = b?.[sortKey];
  const num = (x)=> (x===null||x===undefined||Number.isNaN(x)) ? null : +x;
  const na = num(va), nb=num(vb);

  if(na===null && nb===null){
    const sa = String(va ?? "");
    const sb = String(vb ?? "");
    return sortDir === "asc" ? sa.localeCompare(sb) : sb.localeCompare(sa);
  }
  if(na===null) return 1;
  if(nb===null) return -1;
  return sortDir === "asc" ? (na-nb) : (nb-na);
}

function sizeCanvas(canvas){
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
}

function seriesToArrays(series){
  // Accept {d:[...], px:[...], wm:[...]} or [{d,px,wm}] or [[d,px,wm]]
  if(!series) return {d:[], px:[], wm:[]};
  if(series.d && series.px && series.wm) return series;
  if(Array.isArray(series)){
    const d=[], px=[], wm=[];
    for(const it of series){
      if(Array.isArray(it)){
        d.push(it[0]); px.push(it[1]); wm.push(it[2]);
      }else{
        d.push(it.d); px.push(it.px); wm.push(it.wm);
      }
    }
    return {d, px, wm};
  }
  return {d:[], px:[], wm:[]};
}

function washRGB(w){
  const t = Math.min(1, Math.max(0, (+w||0)/100));
  // near-white to deep green, but keep readable on white
  const c0 = [235, 247, 238];
  const c1 = [11, 61, 36];
  const r = Math.round(c0[0] + (c1[0]-c0[0])*t);
  const g = Math.round(c0[1] + (c1[1]-c0[1])*t);
  const b = Math.round(c0[2] + (c1[2]-c0[2])*t);
  return `rgb(${r},${g},${b})`;
}

function drawChart(canvas, series){
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width, H = canvas.height;

  ctx.clearRect(0,0,W,H);
  ctx.save();
  ctx.scale(dpr,dpr);

  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const pad = 10;

  const S = seriesToArrays(series);
  const n = Math.min(S.d.length, S.px.length, S.wm.length);
  if(n < 2){
    ctx.restore();
    return;
  }

  // prices
  const px = [];
  const wm = [];
  for(let i=0;i<n;i++){
    const p = S.px[i];
    if(p===null || p===undefined || Number.isNaN(p)) continue;
    px.push(+p);
    wm.push(+S.wm[i] || 0);
  }
  if(px.length < 2){
    ctx.restore();
    return;
  }

  const minY = Math.min(...px);
  const maxY = Math.max(...px);
  const dy = (maxY - minY) || 1;
  const toX = (i)=> pad + (i/(n-1))*(w - 2*pad);
  const toY = (p)=> pad + (maxY - p)/dy*(h - 2*pad);

  // draw segments with washout color
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  for(let i=1;i<n;i++){
    const p0 = S.px[i-1], p1 = S.px[i];
    if(p0===null||p1===null||p0===undefined||p1===undefined) continue;
    const x0 = toX(i-1), y0 = toY(+p0);
    const x1 = toX(i),   y1 = toY(+p1);
    ctx.strokeStyle = washRGB(S.wm[i-1]);
    ctx.beginPath();
    ctx.moveTo(x0,y0);
    ctx.lineTo(x1,y1);
    ctx.stroke();
  }

  // last-point marker
  const xL = toX(n-1);
  const yL = toY(+S.px[n-1]);
  ctx.fillStyle = washRGB(S.wm[n-1]);
  ctx.strokeStyle = "#0a0a0a";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(xL,yL,4.5,0,Math.PI*2);
  ctx.fill();
  ctx.stroke();

  ctx.restore();
}

function simplifyExplain(line){
  if(!line) return "";
  const s = String(line).replace(/\*\*/g,"").trim();
  return s
    .replace(/more extreme than /g, "rarer than ")
    .replace(/of past days for this stock\.?/g, "of days.")
    .replace(/only about /g, "only ")
    .replace(/were this low-in-range or lower\.?/g, "were lower.")
    .replace(/Trading volume is unusually high \(higher than /g, "Volume: higher than ")
    .replace(/\)\.?$/g, "")
    .trim();
}

function outcomeLine(h, s){
  if(!s || !s.n) return "";
  const win = Math.round(s.win*100);
  const typ = Math.round(s.median*100);
  const bad = Math.round(s.p10*100);
  const n = s.n;
  return `
    <div class="hline">
      <b>After ${h}</b>
      <span class="pill">Chance of profit ${win}%</span>
      <span class="pill">Typical ${typ}%</span>
      <span class="pill">Bad case (1 in 10) ${bad}%</span>
      <span class="pill">Based on ${n} similar past days</span>
    </div>`;
}

function buildCard(r, {withChart=true}={}){
  const explain = (r.Explain || []).slice(0,3).map(simplifyExplain).filter(Boolean);
  const outcomes = r.Outcomes || {};
  const outHtml = ["1Y","3Y","5Y"].map(h => outcomeLine(h, outcomes[h])).filter(Boolean).join("");

  const verdict = r.Verdict || "–";
  const risk = r.Risk || "–";

  const chart = withChart ? `<canvas class="chart" id="c_${esc(r.Ticker)}"></canvas>` : "";

  return `
    <article class="card" id="card_${esc(r.Ticker)}">
      <div class="card-head">
        <div class="lhs">
          <div class="ticker">${esc(r.Ticker)}</div>
          <div class="verdict">${esc(verdict)}</div>
        </div>
        <div class="rhs">
          <div class="kpi"><span>Score</span><b>${fmt1(r.ReboundScore)}</b></div>
          <div class="kpi"><span>Confidence</span><b>${fmt0(r.Confidence)}</b></div>
          <div class="kpi"><span>Stability</span><b>${fmt0(r.Stability)}</b></div>
          <div class="kpi"><span>Risk</span><b>${esc(risk)}</b></div>
        </div>
      </div>

      ${chart}

      <div class="block">
        <div class="label">Why it’s flagged today</div>
        ${explain.length ? `<ul class="explain">${explain.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : `<div class="muted">–</div>`}
      </div>

      <div class="block">
        <div class="label">What happened next in similar past days</div>
        ${outHtml || `<div class="muted">Not enough similar cases to summarize.</div>`}
      </div>
    </article>
  `;
}

async function fetchTickerDetails(ticker){
  const t = String(ticker||"").toUpperCase().trim();
  if(!t) return null;

  // If already has embedded Series in full.json
  const found = (FULL?.rows || []).find(x => x.Ticker === t);
  if(found?.Series) return found;

  // Otherwise: per-ticker static json
  try{
    const resp = await fetch(`data/tickers/${t}.json`, {cache:"no-store"});
    if(!resp.ok) throw new Error("notfound");
    return await resp.json();
  }catch(e){
    return found || null;
  }
}

async function showSpotlight(ticker){
  const wrap = document.getElementById("spotlight_wrap");
  const hdr  = document.getElementById("spotlight_hdr");
  const spot = document.getElementById("spotlight");
  if(!wrap || !hdr || !spot) return;

  const t = String(ticker||"").toUpperCase().trim();
  hdr.textContent = t ? `Selected: ${t}` : "";
  spot.innerHTML = t ? `<div class="statusline">Loading…</div>` : "";
  if(!t) return;

  const r = await fetchTickerDetails(t);
  if(!r){
    spot.innerHTML = `<div class="statusline">Ticker not in today’s universe: ${esc(t)}</div>`;
    return;
  }

  // Ensure ticker is present
  r.Ticker = r.Ticker || t;

  spot.innerHTML = buildCard(r, {withChart:true});
  const c = spot.querySelector("canvas");
  if(c && r.Series){
    sizeCanvas(c);
    drawChart(c, r.Series);
  }
}

function render(){
  const subtitle = document.getElementById("subtitle");
  const foot = document.getElementById("footnote");
  const asof = FULL?.meta?.asof;

  subtitle.textContent = asof ? `Precomputed daily at ~5pm ET • As of ${asof}` : "Precomputed daily at ~5pm ET";
  foot.textContent = FULL?.meta?.note || "Universe = iShares Russell 1000 (IWB) holdings + a small always-list.";

  rows = FULL?.rows || [];
  rows.sort(compare);

  const top10 = rows.slice(0,10);
  document.getElementById("top10").innerHTML = top10.map(r => buildCard(r, {withChart:true})).join("");

  // Table rows
  const tb = document.getElementById("tbody");
  tb.innerHTML = rows.map(r => `
    <tr data-t="${esc(r.Ticker)}">
      <td class="mono">${esc(r.Ticker)}</td>
      <td>${esc(r.Verdict || "–")}</td>
      <td class="num">${fmt1(r.ReboundScore)}</td>
      <td class="num">${fmt0(r.Confidence)}</td>
      <td class="num">${fmt0(r.Stability)}</td>
      <td>${esc(r.Risk || "–")}</td>
      <td class="num">${fmt1(r.WashoutToday)}</td>
    </tr>
  `).join("");

  // Click handler: show spotlight if not in top10
  tb.querySelectorAll("tr").forEach(tr=>{
    tr.addEventListener("click", ()=>{
      const t = tr.getAttribute("data-t");
      const card = document.getElementById(`card_${t}`);
      if(card){
        card.scrollIntoView({behavior:"smooth", block:"start"});
      }else{
        showSpotlight(t);
        document.getElementById("spotlight_wrap").scrollIntoView({behavior:"smooth", block:"start"});
      }
    });
  });

  // Draw charts for top10 (embedded series)
  for(const r of top10){
    const canvas = document.getElementById(`c_${r.Ticker}`);
    if(!canvas || !r.Series) continue;
    sizeCanvas(canvas);
    drawChart(canvas, r.Series);
  }
}

function setupSorting(){
  document.querySelectorAll("th[data-k]").forEach(th=>{
    th.addEventListener("click", ()=>{
      const k = th.getAttribute("data-k");
      if(sortKey === k) sortDir = (sortDir === "asc") ? "desc" : "asc";
      else { sortKey = k; sortDir = "desc"; }
      render();
    });
  });
}

function setupSearch(){
  const input = document.getElementById("search");
  input.addEventListener("input", ()=>{
    const q = input.value.trim().toUpperCase();
    const tb = document.getElementById("tbody");
    tb.querySelectorAll("tr").forEach(tr=>{
      const t = tr.getAttribute("data-t");
      const show = !q || (t && t.includes(q));
      tr.style.display = show ? "" : "none";
    });
  });

  input.addEventListener("keydown", (e)=>{
    if(e.key !== "Enter") return;
    const q = input.value.trim().toUpperCase();
    if(!q) return;

    // if in table, click row (scrolls if top10; else shows spotlight)
    const tr = document.querySelector(`#tbody tr[data-t="${q}"]`);
    if(tr){ tr.click(); return; }

    // if not in table, still attempt spotlight (will show "not in universe")
    showSpotlight(q);
  });
}

async function boot(){
  setStatus("Loading…");
  try{
    const resp = await fetch(DATA_URL, {cache:"no-store"});
    if(!resp.ok) throw new Error(`HTTP ${resp.status}`);
    FULL = await resp.json();
    setStatus("");
    setupSorting();
    setupSearch();
    render();
  }catch(e){
    setStatus(`Failed to load: ${e}`);
  }
}

window.addEventListener("resize", ()=>{
  // Redraw visible charts on resize
  document.querySelectorAll("canvas.chart").forEach(c=>{
    const id = c.id || "";
    const t = id.startsWith("c_") ? id.slice(2) : null;
    if(!t) return;
    const row = (FULL?.rows || []).find(x=>x.Ticker===t);
    if(row?.Series){
      sizeCanvas(c);
      drawChart(c, row.Series);
    }
  });
});

boot();
