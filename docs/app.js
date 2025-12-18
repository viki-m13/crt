// Rebound Ledger — website renderer (v7 schema)

const DATA_URL = "./data/full.json";

// -----------------------
// Small utilities
// -----------------------
const byId = (id)=>document.getElementById(id);

function withBust(url){
  const bust = "v=" + Date.now();
  const sep = url.includes("?") ? "&" : "?";
  return url + sep + bust;
}

async function loadJSON(url){
  const u = withBust(url);
  const r = await fetch(u, {
    cache: "no-store",
    headers: {"Pragma":"no-cache","Cache-Control":"no-cache"},
  });
  if (!r.ok) throw new Error(`Fetch failed ${r.status}: ${url}`);
  return await r.json();
}

function fmtNum1(x){
  const v = Number(x);
  return Number.isFinite(v) ? v.toFixed(1) : "—";
}
function fmtNum0(x){
  const v = Number(x);
  return Number.isFinite(v) ? v.toFixed(0) : "—";
}
function fmtPct(x){
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return (v*100).toFixed(0) + "%";
}

function washoutTopRankText(topPct){
  const p = Number(topPct);
  if (!Number.isFinite(p)) return "—";
  if (p < 1) return `Top ${p.toFixed(1)}%`;
  return `Top ${p.toFixed(0)}%`;
}

function fmtRange(lo, hi, decimals=0){
  const a = Number(lo), b = Number(hi);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return "—";
  const fa = decimals===0 ? a.toFixed(0) : a.toFixed(1);
  const fb = decimals===0 ? b.toFixed(0) : b.toFixed(1);
  return `${fa}–${fb}`;
}

function clamp01(x){
  const v = Number(x);
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

// -----------------------
// Mini chart (canvas)
// -----------------------

function scoreToGreenRGBA(score01, alphaMin=0.18, alphaMax=0.95){
  const a = alphaMin + (alphaMax - alphaMin) * clamp01(score01);
  // deep green (matches site theme)
  return `rgba(15, 61, 46, ${a})`;
}

function drawWashoutGradientLine(canvas, dates, prices, wash, markerScore){
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);

  const p = (Array.isArray(prices) ? prices : []).map(Number);
  const xN = p.length;
  if (xN < 5) {
    // placeholder
    ctx.fillStyle = "rgba(0,0,0,0.08)";
    ctx.fillRect(0, h-1, w, 1);
    return;
  }

  const wsh = (Array.isArray(wash) ? wash : []).map(Number);
  const n = Math.min(p.length, wsh.length || p.length);

  const ys = p.slice(0,n).filter(Number.isFinite);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const pad = (ymax > ymin) ? 0.07*(ymax-ymin) : 1;
  const y0 = ymin - pad;
  const y1 = ymax + pad;
  const yMap = (v)=>{
    const t = (v - y0) / (y1 - y0);
    return h - 6 - t*(h-12);
  };

  const xMap = (i)=> 6 + (i/(n-1))*(w-12);

  // Draw colored segments (no grey base)
  ctx.lineWidth = 3.2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (let i=0;i<n-1;i++){
    const a = p[i], b = p[i+1];
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const wx = Number.isFinite(wsh[i]) ? wsh[i] : 0;
    ctx.strokeStyle = scoreToGreenRGBA(wx/100);
    ctx.beginPath();
    ctx.moveTo(xMap(i), yMap(a));
    ctx.lineTo(xMap(i+1), yMap(b));
    ctx.stroke();
  }

  // Today marker (colored by Rebound Score, like baseline v7)
  const ms = Number(markerScore);
  const last = p[n-1];
  if (Number.isFinite(ms) && Number.isFinite(last)){
    const cx = xMap(n-1);
    const cy = yMap(last);
    ctx.fillStyle = scoreToGreenRGBA(ms/100, 0.35, 0.98);
    ctx.strokeStyle = "#0a0a0a";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(cx, cy, 5.7, 0, Math.PI*2);
    ctx.fill();
    ctx.stroke();
  }
}

// -----------------------
// Rendering
// -----------------------

function rowHtml(it){
  return `
    <tr data-ticker="${it.ticker}">
      <td class="tcell mono">${it.ticker}</td>
      <td>${it.verdict || "—"}</td>
      <td class="num">${fmtNum1(it.rebound_score)}</td>
      <td class="num">${fmtNum0(it.washout_today)}</td>
      <td class="num">${fmtNum0(it.confidence)}</td>
      <td class="num">${fmtNum0(it.stability)}</td>
      <td>${it.risk || "—"}</td>
      <td class="num">${fmtPct(it.y1_typical)}</td>
      <td class="num">${fmtPct(it.y3_typical)}</td>
      <td class="num">${fmtPct(it.y5_typical)}</td>
    </tr>
  `;
}

function outcomesBlock(outcomes){
  const order = ["1Y","3Y","5Y"];
  const rows = [];
  for (const h of order){
    const o = outcomes?.[h];
    if (!o || !Number.isFinite(Number(o.n)) || Number(o.n) <= 0) continue;
    rows.push(`
      <div class="ob">
        <div class="ob-h">${h}</div>
        <div class="ob-k">Win chance</div>
        <div class="ob-v">${fmtPct(o.win)}</div>
        <div class="ob-k">Typical</div>
        <div class="ob-v">${fmtPct(o.median)}</div>
        <div class="ob-k">Bad-case (P10)</div>
        <div class="ob-v">${fmtPct(o.p10)}</div>
        <div class="ob-k">Good-case (P90)</div>
        <div class="ob-v">${fmtPct(o.p90)}</div>
        <div class="ob-k">Similar cases</div>
        <div class="ob-v">${fmtNum0(o.n)}</div>
      </div>
    `);
  }
  if (!rows.length) return `<div class="muted">Not enough similar past cases to summarize.</div>`;
  return `<div class="outcome-grid">${rows.join("")}</div>`;
}

function evidenceBlock(evidence){
  const order = ["1Y","3Y","5Y"];
  const lines = [];
  for (const h of order){
    const e = evidence?.[h];
    if (!e) continue;
    const nw = Number(e.n_wash);
    const nn = Number(e.n_norm);
    if (!(Number.isFinite(nw) && Number.isFinite(nn)) || nw < 30 || nn < 120) continue;
    lines.push(`
      <div class="ev-row">
        <div class="ev-h">${h}</div>
        <div class="ev-m">
          <div><span class="pill">Washout days (top 10%)</span> win ${fmtPct(e.win_wash)} • typical ${fmtPct(e.med_wash)} • bad-case ${fmtPct(e.p10_wash)} <span class="muted">(n=${fmtNum0(nw)})</span></div>
          <div><span class="pill pill-lite">Normal days (all)</span> win ${fmtPct(e.win_norm)} • typical ${fmtPct(e.med_norm)} • bad-case ${fmtPct(e.p10_norm)} <span class="muted">(n=${fmtNum0(nn)})</span></div>
        </div>
      </div>
    `);
  }
  if (!lines.length) return `<div class="muted">Not enough history to compute evidence.</div>`;
  return `
    <div class="evidence">
      ${lines.join("")}
      <div class="footnote">
        Evidence uses the same definition as the model: <b>washout days</b> are the <b>top 10% Washout Meter</b> days for this stock; <b>normal days</b> are <b>all days</b> for this stock.
      </div>
    </div>
  `;
}

function renderCard(host, it, detail){
  const washRank = washoutTopRankText(it.washout_top_pct);
  const washTop10Range = fmtRange(it.washout_top10_lo, it.washout_top10_hi, 0);
  const card = document.createElement("article");
  card.className = "card";
  card.innerHTML = `
    <div class="card-top">
      <div>
        <div class="ticker">${it.ticker}</div>
        <div class="sub">${it.verdict || "—"}</div>
      </div>
      <div class="mini-metrics">
        <div class="mm"><div class="mm-k">Rebound score</div><div class="mm-v">${fmtNum1(it.rebound_score)}/100</div></div>
        <div class="mm"><div class="mm-k">Washout today</div><div class="mm-v">${fmtNum0(it.washout_today)}/100</div></div>
      </div>
    </div>

    <div class="mini">
      <canvas class="spark" width="560" height="130"></canvas>
      <div class="legend">
        <div><span class="dot dot-line"></span> Line color = Washout Meter (0 → 100)</div>
        <div><span class="dot dot-today"></span> Dot = today (colored by Rebound Score)</div>
      </div>
    </div>

    <div class="kpi-grid">
      <div class="kpi"><div class="kpi-k">Confidence</div><div class="kpi-v">${fmtNum0(it.confidence)}/100</div></div>
      <div class="kpi"><div class="kpi-k">Stability</div><div class="kpi-v">${fmtNum0(it.stability)}/100</div></div>
      <div class="kpi"><div class="kpi-k">Washed-out rank</div><div class="kpi-v">${washRank}</div><div class="kpi-s">Top‑10% range: ${washTop10Range}</div></div>
      <div class="kpi"><div class="kpi-k">Similar cases</div><div class="kpi-v">${fmtNum0(it.similar_cases)}</div></div>
      <div class="kpi"><div class="kpi-k">Risk</div><div class="kpi-v">${it.risk || "—"}</div></div>
      <div class="kpi"><div class="kpi-k">As of</div><div class="kpi-v">${String(it.as_of||"—").slice(0,10)}</div></div>
    </div>

    <div class="block">
      <div class="block-h">Why is it flagged today?</div>
      <ol class="why">
        ${(detail?.explain || []).map(x=>`<li>${x}</li>`).join("") || `<li class="muted">No explanation available.</li>`}
      </ol>
    </div>

    <div class="block">
      <div class="block-h">Current projection (based on similar past setups for this stock)</div>
      ${outcomesBlock(detail?.outcomes)}
    </div>

    <div class="block">
      <div class="block-h">Evidence (washout days vs normal days)</div>
      ${evidenceBlock(detail?.evidence)}
    </div>
  `;

  host.appendChild(card);

  // Draw chart
  const c = card.querySelector("canvas");
  const s = detail?.series || {};
  drawWashoutGradientLine(c, s.dates, s.prices, s.wash, it.rebound_score);
}

function setSortButtons(active){
  document.querySelectorAll(".btn-lite").forEach(b=>{
    b.classList.toggle("active", b.dataset.sort === active);
  });
}

function formatAsOf(asOf){
  if (!asOf) return "—";
  let s = String(asOf).trim();
  if (s.includes(" ") && !s.includes("T")) s = s.replace(" ", "T");
  let d = new Date(s);
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
  return `${yyyy}-${mm}-${dd} ${hh}:${min} ${ap} ET`;
}

function renderHistoricalSignals(full){
  const host = byId("histRows");
  if (!host) return;
  const sigs = Array.isArray(full.historical_signals) ? full.historical_signals : [];
  if (!sigs.length){
    host.innerHTML = `<tr><td colspan="6" class="muted">No historical signals available.</td></tr>`;
    return;
  }

  host.innerHTML = sigs.map(s=>{
    return `
      <tr data-ticker="${s.ticker}">
        <td class="mono">${s.date || "—"}</td>
        <td class="tcell">${s.ticker}</td>
        <td class="num">${fmtNum0(s.washout)}</td>
        <td>${washoutTopRankText(s.washout_top_pct)}</td>
        <td class="num">${fmtPct(s.r1)}</td>
        <td class="num">${fmtPct(s.r3)}</td>
        <td class="num">${fmtPct(s.r5)}</td>
      </tr>
    `;
  }).join("");
}

// -----------------------
// Main
// -----------------------

(async function main(){
  let full;
  try{
    full = await loadJSON(DATA_URL);
  }catch(e){
    byId("top10").innerHTML = `<div class="footnote">No data yet. Run the GitHub Action to generate <span class="mono">docs/data/full.json</span>.</div>`;
    return;
  }

  byId("asOf").textContent = formatAsOf(full.as_of);
  const items = Array.isArray(full.items) ? full.items : [];

  async function loadDetail(ticker){
    const embedded = full.details?.[ticker];
    if (embedded) return embedded;
    return await loadJSON(`./data/tickers/${ticker}.json`);
  }

  let sortMode = "rebound";
  function applySort(){
    const list = [...items];
    if (sortMode === "rebound"){
      list.sort((a,b)=> (b.rebound_score - a.rebound_score) || (b.confidence - a.confidence) || (b.stability - a.stability));
    }else if (sortMode === "washout"){
      list.sort((a,b)=> (b.washout_today - a.washout_today) || (b.rebound_score - a.rebound_score));
    }else if (sortMode === "confidence"){
      list.sort((a,b)=> (b.confidence - a.confidence) || (b.rebound_score - a.rebound_score));
    }
    return list;
  }

  function renderTable(list){
    byId("rows").innerHTML = list.map(rowHtml).join("");
  }

  async function renderTop10(list){
    const host = byId("top10");
    host.innerHTML = "";
    const top = list.slice(0,10);
    for (const it of top){
      let detail;
      try{
        detail = await loadDetail(it.ticker);
      }catch(err){
        detail = { explain: [`⚠️ Detail JSON failed to load for <strong>${it.ticker}</strong>.`], outcomes:{}, evidence:{}, series:{} };
      }
      renderCard(host, it, detail);
    }
  }

  async function rerender(){
    const list = applySort();
    renderTable(list);
    await renderTop10(list);
  }

  // Controls
  document.querySelectorAll(".btn-lite").forEach(btn=>{
    btn.addEventListener("click", async ()=>{
      sortMode = btn.dataset.sort;
      setSortButtons(sortMode);
      await rerender();
    });
  });
  setSortButtons(sortMode);

  byId("go").addEventListener("click", ()=>applySearch());
  byId("q").addEventListener("input", ()=>applySearch());

  async function applySearch(){
    const q = (byId("q").value || "").trim().toUpperCase();
    if (!q){
      await rerender();
      return;
    }
    const filtered = applySort().filter(x=>String(x.ticker||"").includes(q));
    renderTable(filtered);
    await renderTop10(filtered);
  }

  // Table click -> focus ticker
  byId("rows").addEventListener("click", async (e)=>{
    const tr = e.target.closest("tr");
    if (!tr) return;
    const t = tr.dataset.ticker;
    if (!t) return;

    document.querySelectorAll("#rows tr").forEach(r=>r.classList.remove("highlight"));
    tr.classList.add("highlight");

    const current = applySort();
    const idx = current.findIndex(x=>x.ticker===t);
    if (idx < 0) return;
    const rotated = [current[idx], ...current.filter((_,i)=>i!==idx)];
    await renderTop10(rotated);
    document.querySelector(".masthead").scrollIntoView({behavior:"smooth"});
  });

  // Historical signals click -> focus ticker
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

  await rerender();
  renderHistoricalSignals(full);
})();
