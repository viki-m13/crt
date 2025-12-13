
const DATA_URL = "data/full.json";

let FULL = null;
let rows = [];
let sortKey = "ReboundScore";
let sortDir = "desc";

const fmt0 = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : String(Math.round(x));
const fmt1 = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : String((+x).toFixed(1));
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function setStatus(msg){ document.getElementById("status").textContent = msg; }

function compare(a,b){
  const va = a[sortKey];
  const vb = b[sortKey];
  if(typeof va === "string" || typeof vb === "string"){
    const sa = (va ?? "").toString();
    const sb = (vb ?? "").toString();
    return sortDir === "asc" ? sa.localeCompare(sb) : sb.localeCompare(sa);
  }
  const na = (va ?? -1e18);
  const nb = (vb ?? -1e18);
  return sortDir === "asc" ? (na - nb) : (nb - na);
}

function washColor(w){
  const t = Math.min(1, Math.max(0, (w||0)/100));
  const c0 = [232, 245, 234];
  const c1 = [11, 61, 36];
  const r = Math.round(c0[0] + (c1[0]-c0[0])*t);
  const g = Math.round(c0[1] + (c1[1]-c0[1])*t);
  const b = Math.round(c0[2] + (c1[2]-c0[2])*t);
  return `rgb(${r},${g},${b})`;
}

function sizeCanvas(canvas){
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
}

function drawChart(canvas, series){
  const ctx = canvas.getContext("2d");
  const W = canvas.getBoundingClientRect().width;
  const H = canvas.getBoundingClientRect().height;
  ctx.clearRect(0,0,W,H);

  const padL = 42, padR = 14, padT = 14, padB = 26;
  const iw = W - padL - padR;
  const ih = H - padT - padB;

  ctx.fillStyle = "#fff";
  ctx.fillRect(0,0,W,H);

  const px = series.px;
  const wm = series.wm;
  const n = px.length;
  if(n < 5) return;

  let ymin = Infinity, ymax = -Infinity;
  for(let i=0;i<n;i++){
    const y = px[i];
    if(!Number.isFinite(y)) continue;
    if(y<ymin) ymin=y;
    if(y>ymax) ymax=y;
  }
  if(!Number.isFinite(ymin) || !Number.isFinite(ymax) || ymax<=ymin) return;
  const ypad = 0.07*(ymax-ymin);
  ymin -= ypad; ymax += ypad;

  const xTo = (i) => padL + (i/(n-1))*iw;
  const yTo = (y) => padT + (1 - (y - ymin)/(ymax-ymin))*ih;

  ctx.strokeStyle = "#e8e8e8";
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, iw, ih);

  ctx.lineWidth = 2.4;
  ctx.lineCap = "round";
  for(let i=0;i<n-1;i++){
    const y0 = px[i], y1 = px[i+1];
    if(!Number.isFinite(y0) || !Number.isFinite(y1)) continue;
    const w = Number.isFinite(wm[i]) ? wm[i] : 0;
    ctx.strokeStyle = washColor(w);
    ctx.beginPath();
    ctx.moveTo(xTo(i), yTo(y0));
    ctx.lineTo(xTo(i+1), yTo(y1));
    ctx.stroke();
  }

  ctx.fillStyle = "#5a5a5a";
  ctx.font = "11px ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.textAlign = "left";
  ctx.fillText(ymax.toFixed(2), 6, padT+10);
  ctx.fillText(ymin.toFixed(2), 6, padT+ih);

  const last = n-1;
  const yLast = px[last];
  if(Number.isFinite(yLast)){
    const wLast = Number.isFinite(wm[last]) ? wm[last] : 0;
    ctx.fillStyle = washColor(wLast);
    ctx.strokeStyle = "#0a0a0a";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(xTo(last), yTo(yLast), 5.6, 0, Math.PI*2);
    ctx.fill();
    ctx.stroke();
  }
}

function buildTopCard(r){
  const id = `c_${r.Ticker}`;
  const explain = (r.Explain || []).slice(0,3);
  const outcomes = r.Outcomes || {};
  const parts = ["1Y","3Y","5Y"].filter(k => outcomes[k] && outcomes[k].n).map(k => {
    const s = outcomes[k];
    return `<span><b>${k}</b> win ${Math.round(s.win*100)}% • typ ${Math.round(s.median*100)}% • bad ${Math.round(s.p10*100)}% • n ${s.n}</span>`;
  }).join(" ");

  return `
    <div class="card" id="card_${esc(r.Ticker)}">
      <div class="card-head">
        <div>
          <div class="ticker">${esc(r.Ticker)}</div>
          <div class="verdict">${esc(r.Verdict || "–")}</div>
        </div>
        <div class="meta">
          <span>Score <b>${fmt1(r.ReboundScore)}</b></span>
          <span>Conf <b>${fmt0(r.Confidence)}</b></span>
          <span>Stab <b>${fmt0(r.Stability)}</b></span>
          <span>Washout <b>${fmt1(r.WashoutToday)}</b></span>
          <span>Risk <b>${esc(r.Risk || "–")}</b></span>
        </div>
      </div>

      <div class="canvas-wrap">
        <canvas class="chart" id="${id}"></canvas>
      </div>

      <ul class="explain">
        ${explain.map(x => `<li>${esc(x)}</li>`).join("")}
      </ul>

      <div class="outcomes">
        <div class="row">${parts || "<span>Outcomes: –</span>"}</div>
      </div>

      <div class="hr"></div>
    </div>
  `;
}

function render(){
  const subtitle = document.getElementById("subtitle");
  const foot = document.getElementById("footnote");

  subtitle.textContent = FULL?.meta?.asof ? `Precomputed daily • as of ${FULL.meta.asof} (New York time)` : "Precomputed daily • static";
  foot.textContent = FULL?.meta?.note || "Universe = iShares Russell 1000 (IWB) holdings + a small always-list.";

  rows = FULL.rows || [];
  rows.sort(compare);

  const top10 = rows.slice(0,10);
  const topDiv = document.getElementById("top10");
  topDiv.innerHTML = top10.map(buildTopCard).join("");

  const tb = document.getElementById("tbody");
  tb.innerHTML = rows.map(r => `
    <tr data-t="${esc(r.Ticker)}">
      <td class="td-ticker">${esc(r.Ticker)}</td>
      <td>${fmt1(r.ReboundScore)}</td>
      <td>${fmt0(r.Confidence)}</td>
      <td>${fmt0(r.Stability)}</td>
      <td>${esc(r.Verdict || "–")}</td>
      <td>${esc(r.Risk || "–")}</td>
      <td>${fmt1(r.WashoutToday)}</td>
    </tr>
  `).join("");

  tb.querySelectorAll("tr").forEach(tr=>{
    tr.addEventListener("click", ()=>{
      const t = tr.getAttribute("data-t");
      tb.querySelectorAll("tr").forEach(x=>x.classList.remove("highlight"));
      tr.classList.add("highlight");
      const card = document.getElementById(`card_${t}`);
      if(card) card.scrollIntoView({behavior:"smooth", block:"start"});
    });
  });

  // draw charts
  for(const r of top10){
    const canvas = document.getElementById(`c_${r.Ticker}`);
    if(!canvas || !r.Series) continue;
    sizeCanvas(canvas);
    drawChart(canvas, r.Series);
  }
}

function setupSorting(){
  document.querySelectorAll("th[data-sort]").forEach(th=>{
    th.addEventListener("click", ()=>{
      const key = th.getAttribute("data-sort");
      if(sortKey === key){
        sortDir = (sortDir === "asc") ? "desc" : "asc";
      } else {
        sortKey = key;
        sortDir = (key === "Ticker" || key === "Verdict" || key === "Risk") ? "asc" : "desc";
      }
      render();
    });
  });
}

function setupSearch(){
  const input = document.getElementById("search");
  input.addEventListener("input", ()=>{
    const q = input.value.trim().toUpperCase();
    const tb = document.getElementById("tbody");
    let first = null;
    tb.querySelectorAll("tr").forEach(tr=>{
      const t = tr.getAttribute("data-t");
      const show = !q || (t && t.includes(q));
      tr.style.display = show ? "" : "none";
      if(show && !first) first = tr;
    });
  });

  input.addEventListener("keydown", (e)=>{
    if(e.key !== "Enter") return;
    const q = input.value.trim().toUpperCase();
    if(!q) return;
    const tr = document.querySelector(`#tbody tr[data-t="${q}"]`);
    if(tr){ tr.click(); }
    else { setStatus(`Ticker not in today’s universe: ${q}`); }
  });
}

async function boot(){
  setStatus("Loading…");
  try{
    const res = await fetch(DATA_URL, {cache:"no-store"});
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    FULL = await res.json();
    setStatus("");
    setupSorting();
    setupSearch();
    render();
  } catch(err){
    console.error(err);
    setStatus("No data found yet. Run the GitHub Action to generate docs/data/full.json.");
  }
}

window.addEventListener("resize", ()=>{
  if(!FULL) return;
  const top10 = (FULL.rows || []).slice(0,10);
  for(const r of top10){
    const canvas = document.getElementById(`c_${r.Ticker}`);
    if(!canvas || !r.Series) continue;
    sizeCanvas(canvas);
    drawChart(canvas, r.Series);
  }
});

boot();
