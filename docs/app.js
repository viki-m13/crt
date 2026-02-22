
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
  legend.innerHTML = `<span class="legend-bar" aria-hidden="true"></span><span class="legend-text"><span class="legend-label">Pullback intensity</span><span class="legend-note">darker = deeper pullback</span></span>`;
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

/* ---------- backtest engine ---------- */

async function runBacktest(full, loadDetailFn){
  const items = full.items || [];
  const tickers = items.map(x => x.ticker);

  // Load all ticker details
  const allDetails = {};
  await Promise.all(tickers.map(async tk => {
    try { allDetails[tk] = await loadDetailFn(tk); } catch(e){}
  }));

  // Build a unified date index from SPY (longest, most reliable)
  const spyDetail = allDetails["SPY"];
  if (!spyDetail || !spyDetail.series) return null;
  const refDates = spyDetail.series.dates;
  const refPrices = spyDetail.series.prices;

  // For each ticker, build aligned arrays keyed by date
  // Normalize oppScore per ticker: series.final = quality × win_1y, where quality
  // is a CURRENT constant.  Dividing by max(final) removes today's quality and
  // leaves only the historical win probability, preventing look-ahead bias.
  const tickerData = {};  // ticker -> { prices, oppScore (normalized to ~win_1y) }
  for (const tk of tickers){
    const d = allDetails[tk];
    if (!d || !d.series) continue;
    const s = d.series;
    const pm = new Map(), fm = new Map();
    let maxFinal = 0;
    for (let i = 0; i < s.dates.length; i++){
      if (s.prices[i] != null) pm.set(s.dates[i], s.prices[i]);
      if (s.final && s.final[i] != null && s.final[i] > 0){
        if (s.final[i] > maxFinal) maxFinal = s.final[i];
      }
    }
    for (let i = 0; i < s.dates.length; i++){
      if (s.final && s.final[i] != null && s.final[i] > 0){
        fm.set(s.dates[i], maxFinal > 0 ? s.final[i] / maxFinal : 0);
      }
    }
    tickerData[tk] = { prices: pm, oppScore: fm };
  }

  // Get monthly boundaries (first trading day of each month)
  const monthIdx = [];
  let prevYM = "";
  for (let i = 0; i < refDates.length; i++){
    const ym = refDates[i].slice(0, 7);
    if (ym !== prevYM){ monthIdx.push(i); prevYM = ym; }
  }

  // Find the index N trading days (~1Y=252, 3Y=756, 5Y=1260) forward
  function fwdIndex(fromIdx, years){
    const target = fromIdx + Math.round(years * 252);
    return target < refDates.length ? target : -1;
  }

  // Run backtest for a given hold period and top-N selection
  function simulate(holdYears, topN){
    const monthlyInvest = 1000;
    let totalInvested = 0;
    const equityCurve = [];   // { date, strategyValue, spyValue }
    const trades = [];        // individual trades for stats

    // Track open positions: { ticker, buyDate, buyPrice, shares, sellIdx, lastPrice }
    const openPositions = [];
    let realizedPnl = 0;
    let spyShares = 0;
    let spyRealized = 0;
    const spyOpenPositions = [];

    for (const mIdx of monthIdx){
      const date = refDates[mIdx];

      // Close positions that have reached their hold period
      for (let p = openPositions.length - 1; p >= 0; p--){
        const pos = openPositions[p];
        if (mIdx >= pos.sellIdx){
          const sellDate = refDates[pos.sellIdx];
          const sellPrice = tickerData[pos.ticker]?.prices.get(sellDate);
          const sp = sellPrice != null ? sellPrice : pos.lastPrice;
          const ret = (sp / pos.buyPrice) - 1;
          realizedPnl += pos.shares * sp;
          trades.push({ ticker: pos.ticker, buyDate: pos.buyDate, sellDate, buyPrice: pos.buyPrice, sellPrice: sp, ret });
          openPositions.splice(p, 1);
        }
      }
      // Close SPY positions
      for (let p = spyOpenPositions.length - 1; p >= 0; p--){
        const pos = spyOpenPositions[p];
        if (mIdx >= pos.sellIdx){
          const sellPrice = refPrices[pos.sellIdx];
          if (sellPrice != null) spyRealized += pos.shares * sellPrice;
          else spyRealized += pos.shares * pos.buyPrice;
          spyOpenPositions.splice(p, 1);
        }
      }

      // Can we buy and have a sell date?
      const sellIdx = fwdIndex(mIdx, holdYears);
      if (sellIdx < 0) continue; // not enough future data

      // Rank tickers by normalized win probability (quality bias removed)
      const ranked = [];
      for (const tk of tickers){
        if (tk === "SPY" || tk === "DIA" || tk === "QQQ" || tk === "IWM" || tk === "BTC-USD" || tk === "ETH-USD") continue;
        const td = tickerData[tk];
        if (!td) continue;
        const score = td.oppScore.get(date);
        const price = td.prices.get(date);
        if (score == null || price == null || score <= 0) continue;
        ranked.push({ ticker: tk, score, price });
      }
      ranked.sort((a, b) => b.score - a.score);
      const picks = ranked.slice(0, topN);

      if (picks.length === 0) continue;

      totalInvested += monthlyInvest;
      const perStock = monthlyInvest / picks.length;

      // Buy strategy stocks
      for (const pick of picks){
        const shares = perStock / pick.price;
        openPositions.push({
          ticker: pick.ticker,
          buyDate: date,
          buyPrice: pick.price,
          shares,
          sellIdx,
          lastPrice: pick.price,
        });
      }

      // Buy SPY
      const spyPrice = refPrices[mIdx];
      if (spyPrice != null){
        const shares = monthlyInvest / spyPrice;
        spyOpenPositions.push({ buyPrice: spyPrice, shares, sellIdx });
      }

      // Calculate current portfolio value for equity curve
      let stratVal = realizedPnl;
      for (const pos of openPositions){
        const curPrice = tickerData[pos.ticker]?.prices.get(date);
        if (curPrice != null) pos.lastPrice = curPrice;
        stratVal += pos.shares * pos.lastPrice;
      }
      let spyVal = spyRealized;
      for (const pos of spyOpenPositions){
        spyVal += pos.shares * spyPrice;
      }

      equityCurve.push({ date, strategyValue: stratVal, spyValue: spyVal, invested: totalInvested });
    }

    // Mark to market at last available date for remaining open positions
    const lastDate = refDates[refDates.length - 1];
    const lastSpyPrice = refPrices[refPrices.length - 1];
    let finalStrat = realizedPnl;
    const openTrades = [];
    for (const pos of openPositions){
      const curPrice = tickerData[pos.ticker]?.prices.get(lastDate);
      const price = curPrice != null ? curPrice : pos.lastPrice;
      finalStrat += pos.shares * price;
      openTrades.push({ ticker: pos.ticker, buyDate: pos.buyDate, buyPrice: pos.buyPrice, curPrice: price, ret: (price / pos.buyPrice) - 1, open: true });
    }
    let finalSpy = spyRealized;
    for (const pos of spyOpenPositions){
      finalSpy += pos.shares * (lastSpyPrice != null ? lastSpyPrice : pos.buyPrice);
    }

    if (equityCurve.length > 0){
      const last = equityCurve[equityCurve.length - 1];
      last.strategyValue = finalStrat;
      last.spyValue = finalSpy;
    }

    // Compute metrics
    const winningTrades = trades.filter(t => t.ret > 0).length;
    const totalReturn = totalInvested > 0 ? (finalStrat / totalInvested - 1) : 0;
    const spyReturn = totalInvested > 0 ? (finalSpy / totalInvested - 1) : 0;
    const avgTradeReturn = trades.length > 0 ? trades.reduce((s, t) => s + t.ret, 0) / trades.length : 0;
    const medianReturn = trades.length > 0 ? (() => {
      const sorted = trades.map(t => t.ret).sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    })() : 0;

    // Max drawdown on return ratio (value / invested), not absolute value
    // Absolute value almost never drops in DCA since new capital keeps flowing in
    let peakRatio = 0, maxDd = 0;
    for (const pt of equityCurve){
      const ratio = pt.invested > 0 ? pt.strategyValue / pt.invested : 1;
      if (ratio > peakRatio) peakRatio = ratio;
      const dd = peakRatio > 0 ? (peakRatio - ratio) / peakRatio : 0;
      if (dd > maxDd) maxDd = dd;
    }

    return {
      equityCurve,
      totalInvested,
      finalValue: finalStrat,
      spyFinalValue: finalSpy,
      totalReturn,
      spyReturn,
      trades: trades.length,
      tradeLog: trades,
      openTrades,
      winRate: trades.length > 0 ? winningTrades / trades.length : 0,
      avgTradeReturn,
      medianReturn,
      maxDrawdown: maxDd,
    };
  }

  // Run all combinations
  const results = {};
  for (const hold of [1, 3, 5]){
    results[hold] = {};
    for (const topN of [1, 5, 10]){
      results[hold][topN] = simulate(hold, topN);
    }
  }
  return results;
}

/* ---------- backtest chart ---------- */

function drawBacktestChart(canvas, curves, labels, colors){
  const ctx = canvas.getContext("2d");
  const dpr = devicePixelRatio || 1;
  const w = canvas.width = Math.floor(canvas.clientWidth * dpr);
  const h = canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx.clearRect(0, 0, w, h);

  if (!curves.length || !curves[0].length) return;

  const pad = { top: 16 * dpr, right: 14 * dpr, bottom: 30 * dpr, left: 60 * dpr };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const n = curves[0].length;
  let gMin = Infinity, gMax = -Infinity;
  for (const c of curves){
    for (const pt of c){
      if (pt.val < gMin) gMin = pt.val;
      if (pt.val > gMax) gMax = pt.val;
    }
  }
  if (gMax <= gMin){ gMax = gMin + 1; }

  // Also include invested line
  const invested = curves[0].map(pt => pt.invested);
  for (const v of invested){
    if (v < gMin) gMin = v;
    if (v > gMax) gMax = v;
  }

  // Add 5% padding
  const range = gMax - gMin;
  gMin -= range * 0.05;
  gMax += range * 0.05;

  function xAt(i){ return pad.left + plotW * (i / (n - 1)); }
  function yAt(v){ return pad.top + plotH * (1 - (v - gMin) / (gMax - gMin)); }

  // Grid lines
  ctx.strokeStyle = "rgba(0,0,0,.06)";
  ctx.lineWidth = 1;
  const nGrid = 5;
  ctx.font = `${10 * dpr}px "IBM Plex Mono", monospace`;
  ctx.fillStyle = "rgba(0,0,0,.35)";
  ctx.textAlign = "right";
  for (let i = 0; i <= nGrid; i++){
    const v = gMin + (gMax - gMin) * (i / nGrid);
    const y = yAt(v);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    const label = v >= 1000 ? `$${(v / 1000).toFixed(0)}k` : `$${v.toFixed(0)}`;
    ctx.fillText(label, pad.left - 6 * dpr, y + 3 * dpr);
  }

  // X-axis date labels — show only year boundaries, short format
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(0,0,0,.35)";
  const MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  // Pick ~4-6 evenly spaced labels that land on year or half-year boundaries
  const allDates = curves[0].map(pt => pt.date);
  const shownYears = new Set();
  const labelIdxs = [];
  for (let i = 0; i < n; i++){
    const yr = allDates[i].slice(0, 4);
    const mo = allDates[i].slice(5, 7);
    if (mo === "01" && !shownYears.has(yr)){
      shownYears.add(yr);
      labelIdxs.push(i);
    }
  }
  // Fallback: if too few labels, space evenly
  if (labelIdxs.length < 2){
    const step = Math.max(1, Math.floor(n / 5));
    for (let i = 0; i < n; i += step) labelIdxs.push(i);
  }
  // Thin out if too many labels (keep max 7)
  while (labelIdxs.length > 7){
    const thinned = [];
    for (let i = 0; i < labelIdxs.length; i++){
      if (i % 2 === 0 || i === labelIdxs.length - 1) thinned.push(labelIdxs[i]);
    }
    labelIdxs.length = 0;
    labelIdxs.push(...thinned);
  }
  for (const i of labelIdxs){
    const x = xAt(i);
    const d = allDates[i];
    const mo = parseInt(d.slice(5, 7), 10) - 1;
    const yr = d.slice(2, 4); // '21
    ctx.fillText(`${MONTH_NAMES[mo]} '${yr}`, x, h - 6 * dpr);
  }

  // Invested line (dashed)
  ctx.setLineDash([6 * dpr, 4 * dpr]);
  ctx.strokeStyle = "rgba(0,0,0,.18)";
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath();
  for (let i = 0; i < n; i++){
    const x = xAt(i), y = yAt(invested[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw each curve (Top 1 thickest, SPY thinnest)
  const lineWidths = [2.8, 2.2, 1.8, 1.5];
  for (let c = 0; c < curves.length; c++){
    ctx.strokeStyle = colors[c];
    ctx.lineWidth = (lineWidths[c] || 2) * dpr;
    ctx.beginPath();
    for (let i = 0; i < curves[c].length; i++){
      const x = xAt(i), y = yAt(curves[c][i].val);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

/* ---------- backtest UI ---------- */

function renderBacktestUI(results){
  const container = byId("btContent");
  if (!container) return;

  let activeHold = 1;

  function render(){
    const data = results[activeHold];
    if (!data){ container.innerHTML = `<div class="footnote">No data for this period.</div>`; return; }

    container.innerHTML = "";

    // Metrics boxes
    const metricsWrap = document.createElement("div");
    metricsWrap.className = "bt-metrics";

    const strats = [
      { label: "Top 1", n: 1, color: "var(--green)" },
      { label: "Top 5", n: 5, color: "#2478b3" },
      { label: "Top 10", n: 10, color: "#8e6bbf" },
      { label: "SPY (benchmark)", n: "spy", color: "#999" },
    ];

    for (const s of strats){
      const d = s.n === "spy" ? null : data[s.n];
      const box = document.createElement("div");
      box.className = "bt-metric-box";
      if (s.n === "spy"){
        const spyRet = data[1]?.spyReturn ?? 0;
        const cls = spyRet >= 0 ? "bt-pos" : "bt-neg";
        box.innerHTML = `<div class="h">${s.label}</div><div class="val ${cls}">${(spyRet * 100).toFixed(1)}%</div><div class="sub">Total return on $${((data[1]?.totalInvested || 0) / 1000).toFixed(0)}k invested</div>`;
      } else if (d){
        const cls = d.totalReturn >= 0 ? "bt-pos" : "bt-neg";
        box.innerHTML = `<div class="h">${s.label}</div><div class="val ${cls}">${(d.totalReturn * 100).toFixed(1)}%</div><div class="sub">vs SPY ${((d.totalReturn - d.spyReturn) * 100).toFixed(1)}pp &bull; ${d.trades} trades</div>`;
      }
      metricsWrap.appendChild(box);
    }
    container.appendChild(metricsWrap);

    // Chart
    const chartWrap = document.createElement("div");
    chartWrap.className = "bt-chart-wrap";
    const canvas = document.createElement("canvas");
    canvas.className = "bt-canvas";
    chartWrap.appendChild(canvas);
    container.appendChild(chartWrap);

    // Legend
    const legend = document.createElement("div");
    legend.className = "bt-legend";
    const legendItems = [
      { label: "Top 1", color: "var(--green)" },
      { label: "Top 5", color: "#2478b3" },
      { label: "Top 10", color: "#8e6bbf" },
      { label: "SPY", color: "#999" },
      { label: "Invested", color: "rgba(0,0,0,.18)", dashed: true },
    ];
    for (const li of legendItems){
      const el = document.createElement("div");
      el.className = "bt-legend-item";
      el.innerHTML = `<span class="bt-legend-swatch" style="background:${li.color}${li.dashed ? ";border-top:2px dashed rgba(0,0,0,.3);background:transparent;height:0" : ""}"></span><span>${li.label}</span>`;
      legend.appendChild(el);
    }
    container.appendChild(legend);

    // Performance table
    const tbl = document.createElement("table");
    tbl.className = "bt-perf-table";
    tbl.style.marginTop = "14px";
    tbl.innerHTML = `
      <thead><tr>
        <th>Strategy</th>
        <th>Total Return</th>
        <th>vs SPY</th>
        <th>Win Rate</th>
        <th>Avg Trade</th>
        <th>Median Trade</th>
        <th>Max Drawdown</th>
        <th>Trades</th>
      </tr></thead>
      <tbody>
      ${[1, 5, 10].map(n => {
        const d = data[n];
        if (!d) return "";
        const excess = d.totalReturn - d.spyReturn;
        return `<tr>
          <td>Top ${n}</td>
          <td class="${d.totalReturn >= 0 ? "bt-pos" : "bt-neg"}">${(d.totalReturn * 100).toFixed(1)}%</td>
          <td class="${excess >= 0 ? "bt-pos" : "bt-neg"}">${excess >= 0 ? "+" : ""}${(excess * 100).toFixed(1)}pp</td>
          <td>${(d.winRate * 100).toFixed(0)}%</td>
          <td class="${d.avgTradeReturn >= 0 ? "bt-pos" : "bt-neg"}">${(d.avgTradeReturn * 100).toFixed(1)}%</td>
          <td class="${d.medianReturn >= 0 ? "bt-pos" : "bt-neg"}">${(d.medianReturn * 100).toFixed(1)}%</td>
          <td>${(d.maxDrawdown * 100).toFixed(1)}%</td>
          <td>${d.trades}</td>
        </tr>`;
      }).join("")}
      <tr>
        <td>SPY (DCA)</td>
        <td class="${(data[1]?.spyReturn ?? 0) >= 0 ? "bt-pos" : "bt-neg"}">${((data[1]?.spyReturn ?? 0) * 100).toFixed(1)}%</td>
        <td>\u2014</td>
        <td>\u2014</td>
        <td>\u2014</td>
        <td>\u2014</td>
        <td>\u2014</td>
        <td>\u2014</td>
      </tr>
      </tbody>
    `;
    const tableWrap = document.createElement("div");
    tableWrap.className = "bt-table-wrap";
    tableWrap.appendChild(tbl);
    container.appendChild(tableWrap);

    // Trade log section (using Top 5 strategy)
    const top5 = data[5];
    if (top5){
      const closed = (top5.tradeLog || []).slice();
      const open = (top5.openTrades || []).slice();
      const allTrades = [...closed.map(t => ({...t, open: false})), ...open.map(t => ({...t, open: true}))];
      const hasAny = allTrades.length > 0;

      if (hasAny){
        const tradesSection = document.createElement("div");
        tradesSection.className = "bt-trades-section";

        // Compute aggregate stats over ALL trades
        const allRets = allTrades.map(t => t.ret);
        const totalCount = allTrades.length;
        const closedCount = closed.length;
        const winCount = allTrades.filter(t => t.ret > 0).length;
        const winPct = totalCount > 0 ? ((winCount / totalCount) * 100).toFixed(0) : "\u2014";
        const avgRet = totalCount > 0 ? allRets.reduce((s, r) => s + r, 0) / totalCount : 0;
        const sortedRets = allRets.slice().sort((a, b) => a - b);
        const medRet = sortedRets.length > 0 ? (sortedRets.length % 2
          ? sortedRets[Math.floor(sortedRets.length / 2)]
          : (sortedRets[sortedRets.length / 2 - 1] + sortedRets[sortedRets.length / 2]) / 2) : 0;
        const bestRet = sortedRets.length > 0 ? sortedRets[sortedRets.length - 1] : 0;
        const worstRet = sortedRets.length > 0 ? sortedRets[0] : 0;

        const hdr = document.createElement("div");
        hdr.className = "bt-trades-header";
        hdr.innerHTML = `<span class="bt-trades-title">TRADE LOG</span><span class="bt-trades-sub">Top 5 strategy &bull; ${activeHold}Y hold &bull; ${totalCount} trades (${closedCount} closed, ${open.length} open)</span>`;
        tradesSection.appendChild(hdr);

        // Aggregate stats row
        const statsRow = document.createElement("div");
        statsRow.className = "bt-trades-stats";
        const fmtPct = (v) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
        statsRow.innerHTML = `
          <span class="bt-ts"><b>Win rate:</b> ${winPct}%</span>
          <span class="bt-ts"><b>Avg return:</b> <span class="${avgRet >= 0 ? "bt-pos" : "bt-neg"}">${fmtPct(avgRet)}</span></span>
          <span class="bt-ts"><b>Median:</b> <span class="${medRet >= 0 ? "bt-pos" : "bt-neg"}">${fmtPct(medRet)}</span></span>
          <span class="bt-ts"><b>Best:</b> <span class="bt-pos">${fmtPct(bestRet)}</span></span>
          <span class="bt-ts"><b>Worst:</b> <span class="bt-neg">${fmtPct(worstRet)}</span></span>
        `;
        tradesSection.appendChild(statsRow);

        // Build trade rows: open positions, then recent closed
        const tradesTbl = document.createElement("table");
        tradesTbl.className = "bt-perf-table bt-trades-table";

        const recentClosed = closed.slice().sort((a, b) => b.sellDate.localeCompare(a.sellDate));
        const openSorted = open.slice().sort((a, b) => b.buyDate.localeCompare(a.buyDate));

        const MAX_ROWS = 20;
        let rows = "";
        let shown = 0;

        if (openSorted.length > 0){
          rows += `<tr class="bt-trades-group"><td colspan="5">Open positions (unrealized)</td></tr>`;
          for (const t of openSorted.slice(0, 8)){
            if (shown >= MAX_ROWS) break;
            const cls = t.ret >= 0 ? "bt-pos" : "bt-neg";
            rows += `<tr class="bt-trade-open"><td>${t.ticker}</td><td>${t.buyDate.slice(0,7)}</td><td>\u2014</td><td>$${t.buyPrice.toFixed(0)} \u2192 $${t.curPrice.toFixed(0)}</td><td class="${cls}">${t.ret >= 0 ? "+" : ""}${(t.ret * 100).toFixed(1)}%</td></tr>`;
            shown++;
          }
        }

        if (recentClosed.length > 0){
          rows += `<tr class="bt-trades-group"><td colspan="5">Latest closed trades</td></tr>`;
          for (const t of recentClosed){
            if (shown >= MAX_ROWS) break;
            const cls = t.ret >= 0 ? "bt-pos" : "bt-neg";
            rows += `<tr><td>${t.ticker}</td><td>${t.buyDate.slice(0,7)}</td><td>${t.sellDate.slice(0,7)}</td><td>$${t.buyPrice.toFixed(0)} \u2192 $${t.sellPrice.toFixed(0)}</td><td class="${cls}">${t.ret >= 0 ? "+" : ""}${(t.ret * 100).toFixed(1)}%</td></tr>`;
            shown++;
          }
        }

        tradesTbl.innerHTML = `<thead><tr><th>Ticker</th><th>Bought</th><th>Sold</th><th>Price</th><th>Return</th></tr></thead><tbody>${rows}</tbody>`;

        const tradesWrap = document.createElement("div");
        tradesWrap.className = "bt-table-wrap";
        tradesWrap.appendChild(tradesTbl);
        tradesSection.appendChild(tradesWrap);

        container.appendChild(tradesSection);
      }
    }

    // Note
    const note = document.createElement("div");
    note.className = "bt-note";
    note.textContent = `Simulated $1,000/month DCA from ${data[1]?.equityCurve?.[0]?.date?.slice(0,7) || "start"} to ${data[1]?.equityCurve?.[data[1].equityCurve.length-1]?.date?.slice(0,7) || "end"}. Each purchase held ${activeHold}Y then sold. Stock universe is today\u2019s screener picks applied retroactively. Past results do not predict future performance.`;
    container.appendChild(note);

    // Draw chart
    const curves = [];
    const colors = ["#0f3d2e", "#2478b3", "#8e6bbf", "#999"];
    for (const n of [1, 5, 10]){
      const d = data[n];
      if (d && d.equityCurve.length){
        curves.push(d.equityCurve.map(pt => ({ date: pt.date, val: pt.strategyValue, invested: pt.invested })));
      }
    }
    // SPY curve (from top-1 data, same invested amount)
    const spyCurve = data[1]?.equityCurve;
    if (spyCurve && spyCurve.length){
      curves.push(spyCurve.map(pt => ({ date: pt.date, val: pt.spyValue, invested: pt.invested })));
    }

    if (curves.length){
      requestAnimationFrame(() => drawBacktestChart(canvas, curves, ["Top 1","Top 5","Top 10","SPY"], colors));
    }
  }

  // Tab clicks
  document.querySelectorAll("#btHoldTabs .bt-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll("#btHoldTabs .bt-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      activeHold = Number(tab.dataset.hold);
      render();
    });
  });

  render();
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

  // Backtest — lazy load when section is opened
  let backtestLoaded = false;
  const backtestEl = byId("backtestSection");
  if (backtestEl){
    backtestEl.addEventListener("toggle", async function(){
      if (backtestEl.open && !backtestLoaded){
        backtestLoaded = true;
        byId("btContent").innerHTML = `<div class="footnote">Running backtest — loading all ticker data...</div>`;
        try {
          const results = await runBacktest(full, loadDetail);
          if (results){
            renderBacktestUI(results);
          } else {
            byId("btContent").innerHTML = `<div class="footnote">Could not run backtest — SPY data not available.</div>`;
          }
        } catch (err){
          console.error("Backtest error:", err);
          byId("btContent").innerHTML = `<div class="footnote" style="color:#8b4513">Backtest error: ${err.message}</div>`;
        }
      }
    });
  }

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
