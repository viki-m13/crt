const DATA_URL = "./data/full.json";

function fmtPct(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${Math.round(v*100)}%`;
}
function fmtNum(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${v.toFixed(0)}`;
}
function byId(id){ return document.getElementById(id); }

function drawGradientLine(canvas, dates, prices, wash){
  const ctx = canvas.getContext("2d");
  const w = canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
  const h = canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
  ctx.clearRect(0,0,w,h);

  const n = prices.length;
  if (n < 3) return;

  const pad = 12 * devicePixelRatio;
  let minP = Infinity, maxP = -Infinity;
  for (let i=0;i<n;i++){ const p=prices[i]; if (p<minP) minP=p; if (p>maxP) maxP=p; }
  if (!(maxP>minP)) return;

  const x0=pad, x1=w-pad, y0=pad, y1=h-pad;

  function xAt(i){ return x0 + (x1-x0) * (i/(n-1)); }
  function yAt(p){ return y1 - (y1-y0) * ((p-minP)/(maxP-minP)); }

  ctx.lineWidth = 2.2*devicePixelRatio;
  ctx.strokeStyle = "rgba(0,0,0,.35)";
  ctx.beginPath();
  for (let i=0;i<n;i++){
    const x=xAt(i), y=yAt(prices[i]);
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();

  for (let i=0;i<n-1;i++){
    const a = Math.max(0, Math.min(1, (wash[i]||0)/100));
    if (a <= 0.02) continue;
    ctx.lineWidth = 3.4*devicePixelRatio;
    ctx.strokeStyle = `rgba(15,61,46,${0.18 + 0.70*a})`;
    ctx.beginPath();
    ctx.moveTo(xAt(i), yAt(prices[i]));
    ctx.lineTo(xAt(i+1), yAt(prices[i+1]));
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(15,61,46,.95)";
  ctx.strokeStyle = "rgba(0,0,0,.85)";
  ctx.lineWidth = 1.2*devicePixelRatio;
  ctx.beginPath();
  ctx.arc(xAt(n-1), yAt(prices[n-1]), 4.3*devicePixelRatio, 0, Math.PI*2);
  ctx.fill(); ctx.stroke();
}

function renderCard(container, item, detail){
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
      <div class="metric"><span>Score</span> <strong>${item.score.toFixed(1)}</strong></div>
      <div class="metric"><span>Conf</span> <strong>${fmtNum(item.confidence)}</strong></div>
      <div class="metric"><span>Stab</span> <strong>${fmtNum(item.stability)}</strong></div>
      <div class="metric"><span>Risk</span> <strong>${item.risk}</strong></div>
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
  left.appendChild(ul);

  const outcomes = document.createElement("div");
  outcomes.className = "outcomes";

  const makeBox = (label, s) => {
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
  };

  outcomes.appendChild(makeBox("1 year", detail.outcomes?.["1Y"]));
  outcomes.appendChild(makeBox("3 years", detail.outcomes?.["3Y"]));
  outcomes.appendChild(makeBox("5 years", detail.outcomes?.["5Y"]));
  left.appendChild(outcomes);

  const right = document.createElement("div");
  right.className = "chart";
  const canvas = document.createElement("canvas");
  canvas.className = "canvas";
  right.appendChild(canvas);

  grid.appendChild(left);
  grid.appendChild(right);
  card.appendChild(grid);

  const series = detail.series || null;
  if (series && series.prices && series.prices.length){
    requestAnimationFrame(()=>drawGradientLine(canvas, series.dates, series.prices, series.wash));
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
      <td class="num">${item.score.toFixed(1)}</td>
      <td class="num">${fmtNum(item.confidence)}</td>
      <td class="num">${fmtNum(item.stability)}</td>
      <td>${item.risk}</td>
      <td class="num">${fmtPct(t1)}</td>
      <td class="num">${fmtPct(t3)}</td>
      <td class="num">${fmtPct(t5)}</td>
    </tr>
  `;
}

async function loadJSON(url){
  const r = await fetch(url, {cache: "no-cache"});
  if (!r.ok) throw new Error(`Fetch failed ${r.status}: ${url}`);
  return await r.json();
}

function setSortButtons(active){
  document.querySelectorAll(".btn-lite").forEach(b=>{
    b.classList.toggle("active", b.dataset.sort === active);
  });
}

(async function main(){
  let full;
  try{
    full = await loadJSON(DATA_URL);
  }catch(e){
    byId("top10").innerHTML = `<div class="footnote">No data yet. Run the GitHub Action once to generate <span class="mono">docs/data/full.json</span>.</div>`;
    return;
  }

  byId("asOf").textContent = full.as_of || "—";

  let items = full.items || [];
  let sortMode = "default";

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
      const detail = await loadDetail(it.ticker);
      renderCard(c, it, detail);
    }
  }

  function applySort(){
    const list = [...items];
    if (sortMode === "score"){
      list.sort((a,b)=> (b.score - a.score) || (b.confidence - a.confidence) || (b.stability - a.stability));
    }else if (sortMode === "confidence"){
      list.sort((a,b)=> (b.confidence - a.confidence) || (b.score - a.score) || (b.stability - a.stability));
    }else if (sortMode === "stability"){
      list.sort((a,b)=> (b.stability - a.stability) || (b.score - a.score) || (b.confidence - a.confidence));
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
  setSortButtons("default");

  await rerender();

  byId("rows").addEventListener("click", async (e)=>{
    const tr = e.target.closest("tr");
    if (!tr) return;
    const t = tr.dataset.ticker;
    if (!t) return;

    document.querySelectorAll("#rows tr").forEach(r=>r.classList.remove("highlight"));
    tr.classList.add("highlight");

    const current = applySort();
    const inTop = current.slice(0,10).some(x=>x.ticker===t);
    if (inTop){
      document.querySelector(".masthead").scrollIntoView({behavior:"smooth"});
      return;
    }

    const top = byId("top10");
    const existing = top.querySelector("[data-selected='1']");
    if (existing) existing.remove();

    const it = items.find(x=>x.ticker===t);
    if (!it) return;

    const detail = await loadDetail(t);
    const holder = document.createElement("div");
    holder.dataset.selected = "1";
    holder.className = "card";
    holder.innerHTML = `<div class="section-title">Selected</div>`;
    top.prepend(holder);

    renderCard(holder, it, detail);

    holder.scrollIntoView({behavior:"smooth"});
  });

  const q = byId("q");
  const go = byId("go");
  function normalizeTicker(s){ return (s||"").trim().toUpperCase().replace(".", "-"); }

  async function doSearch(){
    const t = normalizeTicker(q.value);
    if (!t) return;
    const row = document.querySelector(`#rows tr[data-ticker="${t}"]`);
    if (row){
      row.scrollIntoView({behavior:"smooth", block:"center"});
      row.click();
      return;
    }
    alert("Ticker not in today’s universe (Russell 1000 / IWB holdings).");
  }
  go.addEventListener("click", doSearch);
  q.addEventListener("keydown", (e)=>{ if (e.key==="Enter\") doSearch(); });

})();

