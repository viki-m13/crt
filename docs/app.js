let FULL = null;
let ROWS = [];
let chart = null;

const $ = (id) => document.getElementById(id);

function fmtPct(x){
  if (x === null || x === undefined || !isFinite(x)) return "—";
  const p = x * 100;
  return (Math.abs(p) < 1 ? p.toFixed(1) : p.toFixed(0)) + "%";
}
function fmtNum(x){
  if (x === null || x === undefined || !isFinite(x)) return "—";
  return Number(x).toFixed(1);
}
function clsVerdict(v){
  if (!v) return "";
  const s = v.toLowerCase();
  if (s.includes("strong") || s.includes("top pick")) return "good";
  if (s.includes("mixed") || s.includes("promising") || s.includes("unstable")) return "warn";
  return "bad";
}

async function loadFull(){
  const res = await fetch("data/full.json", {cache:"no-store"});
  if (!res.ok) throw new Error("Missing docs/data/full.json. Run the workflow once.");
  return await res.json();
}

function buildTable(rows){
  const tb = document.querySelector("#tbl tbody");
  tb.innerHTML = "";
  for (const r of rows){
    const tr = document.createElement("tr");
    tr.dataset.ticker = r.ticker;

    const t = r.ticker;
    const v = r.verdict || "";
    const typ = r.typical || {};
    const wash = (r.washout_top_pct === null || r.washout_top_pct === undefined) ? "—" : (Number(r.washout_top_pct).toFixed(0) + "%");

    tr.innerHTML = `
      <td>${r.rank ?? ""}</td>
      <td class="ticker">${t}</td>
      <td class="verdict ${clsVerdict(v)}">${v}</td>
      <td>${fmtNum(r.score)}</td>
      <td>${fmtNum(r.confidence)}</td>
      <td>${fmtNum(r.stability)}</td>
      <td>${wash}</td>
      <td>${fmtPct(typ["1Y"])}</td>
      <td>${fmtPct(typ["3Y"])}</td>
      <td>${fmtPct(typ["5Y"])}</td>
    `;
    tr.addEventListener("click", () => showDetails(t));
    tb.appendChild(tr);
  }
}

function applyFilters(){
  const q = $("search").value.trim().toUpperCase();
  const vf = $("verdictFilter").value;
  const sb = $("sortBy").value;

  let out = ROWS.slice();

  if (q){
    out = out.filter(r => r.ticker.includes(q));
  }
  if (vf){
    out = out.filter(r => (r.verdict || "") === vf);
  }

  out.sort((a,b)=>{
    if (sb === "rank") return (a.rank ?? 1e9) - (b.rank ?? 1e9);
    const av = a[sb]; const bv = b[sb];
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    if (typeof av === "number" && typeof bv === "number") return bv - av;
    return String(bv).localeCompare(String(av));
  });

  buildTable(out);
}

async function fetchTickerDetail(ticker){
  if (FULL?.details?.[ticker]) return FULL.details[ticker];
  const res = await fetch(`data/tickers/${ticker}.json`, {cache:"no-store"});
  if (!res.ok) throw new Error(`Missing data/tickers/${ticker}.json`);
  return await res.json();
}

function renderEvidence(evidence){
  if (!evidence) return `<div class="notice">No evidence payload found (your generator must write detail.evidence).</div>`;

  const horizons = ["1Y","3Y","5Y"].filter(h=>evidence[h]);
  if (!horizons.length) return `<div class="notice">No evidence for horizons yet.</div>`;

  const statRow = (label, s) => {
    if (!s || !s.n) return `<tr><td>${label}</td><td colspan="5" class="small">—</td></tr>`;
    const win = (s.win*100).toFixed(0)+"%";
    return `
      <tr>
        <td>${label}</td>
        <td>${s.n}</td>
        <td>${win}</td>
        <td>${fmtPct(s.median)}</td>
        <td>${fmtPct(s.p10)}</td>
        <td>${fmtPct(s.p90)}</td>
      </tr>`;
  };

  let html = "";
  for (const h of horizons){
    const block = evidence[h];
    html += `
      <div style="border:1px solid var(--border);border-radius:14px;padding:10px;margin:10px 0;">
        <div class="small" style="font-weight:700;color:var(--muted);">${h} forward returns</div>
        <div style="overflow:auto;margin-top:8px;">
          <table>
            <thead>
              <tr>
                <th>Group</th><th>n</th><th>win%</th><th>median</th><th>p10</th><th>p90</th>
              </tr>
            </thead>
            <tbody>
              ${statRow("Similar days", block.similar_days)}
              ${statRow("Normal days", block.normal_days)}
              ${statRow("Normal (same regime)", block.normal_same_regime)}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }
  return html;
}

function renderExplain(lines){
  if (!lines || !lines.length) return `<div class="notice">—</div>`;
  return `<ul>${lines.map(x=>`<li>${x}</li>`).join("")}</ul>`;
}

function renderChart(series){
  const ctx = $("chart").getContext("2d");
  const labels = series.dates;
  const prices = series.prices;
  const wash = series.wash;

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          type: "line",
          label: "Price",
          data: prices,
          yAxisID: "yPrice",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.15
        },
        {
          type: "bar",
          label: "Washout meter",
          data: wash,
          yAxisID: "yWash",
          borderWidth: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "right" },   // <- legend moved to the right
        tooltip: { enabled: true }
      },
      scales: {
        x: { ticks: { maxTicksLimit: 7 } },
        yPrice: { position: "left", grid: { drawOnChartArea: true } },
        yWash: { position: "right", min: 0, max: 100, grid: { drawOnChartArea: false } }
      }
    }
  });
}

async function showDetails(ticker){
  $("empty").style.display = "none";
  $("detail").style.display = "block";

  $("dTicker").textContent = ticker;
  $("dAsOf").textContent = "Loading…";
  $("dVerdict").textContent = "";
  $("dExplain").innerHTML = "";
  $("dEvidence").innerHTML = "";
  if (chart) chart.destroy();

  try{
    const d = await fetchTickerDetail(ticker);

    $("dAsOf").textContent = `As of ${d.as_of}`;
    $("dVerdict").textContent = d.verdict || "";
    $("dVerdict").className = `verdict ${clsVerdict(d.verdict || "")}`;
    $("dScore").textContent = fmtNum(d.score);
    $("dConf").textContent = fmtNum(d.confidence);
    $("dStab").textContent = fmtNum(d.stability);
    $("dRisk").textContent = d.risk || "—";

    $("dExplain").innerHTML = renderExplain(d.explain || []);
    $("dEvidence").innerHTML = renderEvidence(d.evidence);

    if (d.series) renderChart(d.series);
  }catch(e){
    $("dAsOf").textContent = "Error";
    $("dExplain").innerHTML = `<div class="notice">${e.message}</div>`;
  }
}

async function init(){
  try{
    FULL = await loadFull();
    $("asOf").textContent = "• " + (FULL.as_of || "");
    $("status").textContent = "ready";

    const items = FULL.items || [];
    ROWS = items.map((r, idx)=>({
      ...r,
      rank: r.rank ?? (idx+1),
      typ1: r.typical?.["1Y"] ?? null,
      typ3: r.typical?.["3Y"] ?? null,
      typ5: r.typical?.["5Y"] ?? null,
    }));

    applyFilters();

    $("search").addEventListener("input", applyFilters);
    $("verdictFilter").addEventListener("change", applyFilters);
    $("sortBy").addEventListener("change", applyFilters);

    if (ROWS.length) showDetails(ROWS[0].ticker);
  }catch(e){
    $("status").textContent = "error";
    $("status").className = "badge bad";
    document.querySelector("#tbl tbody").innerHTML =
      `<tr><td colspan="10" class="small">${e.message}</td></tr>`;
  }
}

init();
