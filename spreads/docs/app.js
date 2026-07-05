/* CreditFloor — two-tier credit-spread page.
 *   Tier 1 "Sigma-Clear"  (max accuracy, put_signals/call_signals)
 *   Tier 2 "Vol-Alpha"    (max profit at high accuracy, tier2_signals)
 *
 * Data: data/signals.json (today's published trades, hold-to-expiry),
 *       data/live_log.json (append-only record of every signal ever
 *       published, resolved at expiry close), data/last_run.txt.
 */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const fmt$ = (x, d = 2) =>
    "$" + Number(x).toLocaleString(undefined, {
      minimumFractionDigits: d,
      maximumFractionDigits: d,
    });
  const fmtPct = (x, d = 2) => `${Number(x).toFixed(d)}%`;
  const fmtInt = (x) => Number(x).toLocaleString();

  const state = {
    trades: [],            // flattened rungs, one per tradeable signal
    sort: "credit",
    tierFilter: "all",     // "all" | 1 | 2
    log: null,             // live_log signals array
    logFiltered: [],
    logShown: 0,
    logStatus: "all",
    logEngine: "all",
    logTicker: "",
  };

  async function fetchJSON(url) {
    const r = await fetch(url + "?v=" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error("failed to load " + url);
    return r.json();
  }

  /* ---------------- stats band ---------------- */

  function renderStats(d) {
    const asof = (d.put_signals || []).concat(d.call_signals || [])[0]?.end_date
      || (d.generated_at || "").slice(0, 10);
    const el = $("#stat-asof");
    if (el) el.textContent = asof || "—";
  }

  /* ---------------- today's trades ---------------- */

  function flattenTrades(d) {
    const out = [];
    // Tier 2 (Vol-Alpha GBM puts): reality-verified single-rung trades.
    for (const s of d.tier2_signals || []) {
      const re = s.real;
      if (!re) continue;
      out.push({
        tier: 2,
        ticker: s.ticker,
        side: "put",
        spot: s.today_close,
        asof: s.end_date,
        horizon: s.horizon,
        expiry: re.expiry,
        expiryType: "listed",
        calDays: re.cal_days_to_expiry,
        real: re,
        shortK: re.short_strike,
        longK: re.long_strike,
        width: re.width,
        buffer: re.real_buffer_pct,
        certBuffer: null,
        histWorst: null,
        credit: re.net_natural_credit * 100,
        maxLoss: re.max_loss * 100,
        ror: re.ror_natural * 100,
        annRor: (re.ror_natural * 100) * (365 / Math.max(re.cal_days_to_expiry, 1)),
        modelCredit: null,
        rv: s.sigma60_pct,
        stress: null,
        wing: null,
        nTest: null,
        nFolds: null,
        variant: "gbm",
        gbm: s.gbm_confidence,
        folds: [],
      });
    }
    for (const s of (d.put_signals || []).concat(d.call_signals || [])) {
      for (const r of s.ladder || []) {
        const p = r.profit || {};
        const re = r.real || null;
        out.push({
          ticker: s.ticker,
          side: s.side,
          spot: s.today_close,
          asof: s.end_date,
          horizon: r.horizon,
          expiry: r.expiry_date,
          expiryType: r.expiry_type,
          calDays: r.calendar_days_to_expiry,
          real: re,
          shortK: re ? re.short_strike : p.short_strike,
          longK: re ? re.long_strike : p.long_strike,
          width: re ? re.width : p.spread_width,
          buffer: r.buffer_pct,
          certBuffer: r.certified_buffer_pct,
          histWorst: r.history_worst_buffer_pct,
          credit: re ? re.net_natural_credit * 100 : (p.net_credit_per_share || 0) * 100,
          maxLoss: re ? re.max_loss * 100 : (p.est_max_loss_per_share || 0) * 100,
          ror: re ? re.ror_natural * 100 : (p.return_on_risk_pct || 0),
          annRor: re
            ? (re.ror_natural * 100) * (365 / Math.max(re.cal_days_to_expiry, 1))
            : (p.annualized_ror_pct || 0),
          modelCredit: (p.net_credit_per_share || 0) * 100,
          rv: p.realized_vol_pct,
          stress: r.stress_profit ? (r.stress_profit.net_credit_per_share || 0) * 100 : null,
          wing: r.crash_wing || null,
          nTest: r.n_test,
          nFolds: r.n_folds,
          variant: r.variant,
          folds: r.folds || [],
        });
      }
    }
    return out;
  }

  function sortTrades() {
    let t = state.trades.slice();
    if (state.tierFilter !== "all") {
      const want = Number(state.tierFilter);
      t = t.filter((x) => (x.tier === 2 ? 2 : 1) === want);
    }
    if (state.sort === "credit") t.sort((a, b) => b.credit - a.credit);
    else if (state.sort === "ror") t.sort((a, b) => b.ror - a.ror);
    else t.sort((a, b) => (a.expiry < b.expiry ? -1 : a.expiry > b.expiry ? 1 : b.credit - a.credit));
    return t;
  }

  function optWord(side) { return side === "put" ? "put" : "call"; }

  function tradeCard(t, idx) {
    const dir = t.side === "put" ? "below" : "above";
    const winWord = t.side === "put" ? "at or above" : "at or below";
    const wing = t.tier === 2
      ? ""
      : t.wing
      ? `<div class="cf-wing-line">
           <strong>Optional crash wing:</strong> also buy ${t.wing.ratio} &times;
           the ${fmt$(t.wing.strike)} ${optWord(t.side)} (same expiry)
           for &asymp; ${fmt$(t.wing.est_cost_per_share * 100, 0)}/contract
           &rarr; net credit ${fmt$(t.wing.net_credit_after_wing * 100, 0)}.
           Costs part of the credit; turns a crash-sized move into a profit
           instead of a max loss.
         </div>`
      : `<div class="cf-wing-line" style="background:#f5f5f5;border-left-color:var(--muted)">
           Crash wing not offered on this trade — the credit is too thin to
           pay for it.
         </div>`;
    const foldRows = t.folds.map((f) =>
      `<tr><td>${f.year}</td><td>${fmtInt(f.n_train)}</td><td>${fmtInt(f.n_test)}</td>
       <td>${fmtPct(f.b_hat_pct, 1)}</td><td>${fmtPct(f.worst_test_buf_pct, 1)}</td>
       <td class="${f.losses ? "loss" : "win"}">${f.losses ? f.losses + " LOSS" : "clean"}</td></tr>`
    ).join("");
    return `
      <div class="cf-card" data-idx="${idx}">
        <div class="cf-card-head">
          <span>
            <span class="cf-card-ticker">${t.ticker}</span>
            <span class="cf-side-badge ${t.side}">${t.side} credit spread</span>
            ${t.tier === 2
              ? `<span class="cf-side-badge" style="color:#6b21a8;border-color:#6b21a8">tier 2 · vol-alpha · validated 98% / 24% ROR</span>`
              : `<span class="cf-side-badge">tier 1 · max accuracy · validated 99.4%</span>`}
            ${t.real
              ? `<span class="cf-side-badge" style="color:var(--green);border-color:var(--green)">verified live chain</span>`
              : `<span class="cf-side-badge">model estimate</span>`}
          </span>
          <span class="cf-card-price">last close <strong>${fmt$(t.spot)}</strong> · ${t.asof}</span>
        </div>
        <div class="cf-trade-line">
          <strong>SELL</strong> the ${fmt$(t.shortK)} ${optWord(t.side)} ·
          <strong>BUY</strong> the ${fmt$(t.longK)} ${optWord(t.side)} ·
          expires <strong>${t.expiry}</strong> (${t.calDays} days, ${t.expiryType}) ·
          hold to expiry.
          Wins if ${t.ticker} closes ${winWord} ${fmt$(t.shortK)} on ${t.expiry}.
        </div>
        <div class="cf-trade-nums">
          <div class="cf-num"><span class="v good">${fmt$(t.credit, 0)}</span><span class="k">Est. credit / contract</span></div>
          <div class="cf-num"><span class="v">${fmt$(t.maxLoss, 0)}</span><span class="k">Max loss / contract</span></div>
          <div class="cf-num"><span class="v good">${fmtPct(t.ror, 2)}</span><span class="k">Return on risk</span></div>
          <div class="cf-num"><span class="v">${fmtPct(t.annRor, 1)}</span><span class="k">Annualized</span></div>
          <div class="cf-num"><span class="v">${fmtPct(t.buffer, 1)}</span><span class="k">Cushion (${dir} spot)</span></div>
          <div class="cf-num"><span class="v">${fmtInt(t.nTest)}</span><span class="k">Certification tests</span></div>
        </div>
        ${wing}
        <div class="cf-card-expand" data-toggle="${idx}">Show full details &amp; certification record</div>
        <div class="cf-detail">
          ${t.real ? `<div class="cf-detail-kv" style="margin-bottom:8px">
            <strong>Live-chain quotes</strong> (delayed; as of ${t.real.quote_time}):
            short ${fmt$(t.real.short_strike)} bid ${fmt$(t.real.short_bid)} / ask ${fmt$(t.real.short_ask)} (OI ${fmtInt(t.real.short_oi)}) ·
            long ${fmt$(t.real.long_strike)} bid ${fmt$(t.real.long_bid)} / ask ${fmt$(t.real.long_ask)} (OI ${fmtInt(t.real.long_oi)})<br/>
            Natural credit (sell bid / buy ask) <strong>${fmt$(t.real.natural_credit)}</strong>/share ·
            mid ${fmt$(t.real.mid_credit)}/share${t.modelCredit != null ? ` · model estimate ${fmt$(t.modelCredit / 100)}/share` : ""} —
            the published number is the natural credit, the fill you can get without negotiating.<br/>
            <strong>Liquidity:</strong>
            underlying ${t.real.adv_usd ? "$" + (t.real.adv_usd / 1e6).toFixed(0) + "M/day volume" : "volume n/a"} ·
            short-leg bid/ask spread ${t.real.short_spread_frac != null ? fmtPct(t.real.short_spread_frac * 100, 0) + " of width" : "n/a"} ·
            open interest ${fmtInt(t.real.short_oi)}/${fmtInt(t.real.long_oi)}. Only contracts passing
            an underlying dollar-volume floor, an open-interest floor, and a tight-short-leg-spread
            test are published.
          </div>` : ""}
          ${t.tier === 2 ? `<div class="cf-detail-kv">
            Published <strong>${t.asof}</strong> · expiry <strong>${t.expiry}</strong> (${t.calDays} calendar days) ·
            spread width <strong>${fmt$(t.width)}</strong><br/>
            Strike sits 0.6&sigma; below spot (60-day realized vol ${t.rv != null ? fmtPct(t.rv, 0) : "—"});
            selection: gradient-boosted model confidence <strong>${(t.gbm * 100).toFixed(1)}%</strong>
            above the frozen deep-confidence threshold. Validated on untouched 2019&ndash;2026:
            98.2% accuracy, 24.3% ROR/trade (19.7% under zero-vol-premium stress),
            ~7 trades/week, worst trade &minus;$516/contract. <strong>~1.8 of every 100
            trades lose</strong> — size accordingly.
          </div>` : `<div class="cf-detail-kv">
            Published <strong>${t.asof}</strong> · expiry <strong>${t.expiry}</strong> (${t.calDays} calendar days, ${t.horizon}-session certified window) ·
            spread width <strong>${fmt$(t.width)}</strong><br/>
            Cushion math: short strike sits <strong>${fmtPct(t.buffer, 2)}</strong> ${dir} spot
            = 2.5&sigma; of current volatility (60-day realized vol ${t.rv != null ? fmtPct(t.rv, 0) : "—"})
            &middot; worst ${t.horizon}-session move in ${t.ticker}&rsquo;s entire listed history:
            <strong>${fmtPct(t.histWorst, 1)}</strong> (this strike clears &ge;80% of it) ·
            variant: ${t.variant}<br/>
            Stress credit (options priced at bare realized vol, no premium at all):
            <strong>${t.stress != null ? fmt$(t.stress, 0) : "—"}</strong>/contract —
            real fills should land between this and the estimate above.
          </div>`}
          ${t.folds.length ? `<table class="cf-fold-tbl">
            <thead><tr><th>Test year</th><th>Train</th><th>Test</th><th>Certified buffer</th><th>Worst test move</th><th>Result</th></tr></thead>
            <tbody>${foldRows}</tbody>
          </table>` : ""}
        </div>
      </div>`;
  }

  function renderTrades() {
    const list = $("#trade-list");
    const t = sortTrades();
    if (!t.length) {
      const re = state.reality;
      let why = "";
      if (re && re.drops && Object.keys(re.drops).length) {
        const parts = Object.entries(re.drops)
          .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`);
        why = `<br/><br/>Candidates passed the model gates today but failed
          live-chain verification — ${parts.join(" · ")}. A signal that
          doesn't exist as a real, liquid contract is not published.`;
      }
      list.innerHTML = `<div class="cf-empty">No tradeable trades today. The gates are strict
        on purpose — most stocks, most days, the market doesn't pay enough for a
        strike this safe, or the real chain can't fill it.${why}<br/><br/>
        Check back after the next market close.</div>`;
      return;
    }
    list.innerHTML = t.map((x, i) => tradeCard(x, i)).join("");
    $$("#trade-list .cf-card-expand").forEach((el) => {
      el.addEventListener("click", () => el.closest(".cf-card").classList.toggle("open"));
    });
  }

  /* ---------------- history: live log ---------------- */

  function renderLiveStats(log) {
    const s = log.summary || {};
    const be = s.by_engine || {};
    const v3 = be["v3-sigmaclear"] || {};
    const t2 = be["t2-volalpha-gbm"] || {};
    const v1 = be["v1"] || {};
    // The big tiles reflect the CURRENT strategy (Tier 1 + Tier 2) — the
    // live record of what the page publishes today. The legacy engine's
    // large resolved count would otherwise drown it out; it stays visible
    // below, clearly marked retired.
    const cur = {
      resolved: (v3.resolved || 0) + (t2.resolved || 0),
      wins: (v3.wins || 0) + (t2.wins || 0),
      losses: (v3.losses || 0) + (t2.losses || 0),
      pending: (v3.pending || 0) + (t2.pending || 0),
    };
    const curWR = cur.resolved > 0 ? cur.wins / cur.resolved : null;
    $("#live-resolved").textContent = fmtInt(cur.resolved);
    $("#live-wins").textContent = fmtInt(cur.wins);
    $("#live-losses").textContent = fmtInt(cur.losses);
    $("#live-wr").textContent = curWR != null ? fmtPct(100 * curWR, 2)
      : "building";
    $("#live-sub").innerHTML =
      `<span>First current-strategy signal: <strong>2026-06-12</strong></span>` +
      `<span>Pending: <strong>${fmtInt(cur.pending)}</strong></span>` +
      `<span>Tier 1 &ldquo;Sigma-Clear&rdquo;: <strong>${fmtInt(v3.total || 0)}</strong> published · ` +
      `<strong>${fmtInt(v3.wins || 0)}</strong> wins · <strong>${fmtInt(v3.losses || 0)}</strong> losses` +
      `${(v3.pending || 0) ? ` · ${fmtInt(v3.pending)} pending` : ""}</span>` +
      `<span>Tier 2 &ldquo;Vol-Alpha&rdquo;: <strong>${fmtInt(t2.total || 0)}</strong> published · ` +
      `<strong>${fmtInt(t2.wins || 0)}</strong> wins · <strong>${fmtInt(t2.losses || 0)}</strong> losses` +
      `${(t2.pending || 0) ? ` · ${fmtInt(t2.pending)} pending` : ""}</span>` +
      `<span style="opacity:.75">Retired legacy engine (v1, before 2026-06-12): <strong>${fmtInt(v1.resolved || 0)}</strong> resolved · ` +
      `<strong>${fmtInt(v1.losses || 0)}</strong> losses — kept on the record, not part of the current strategy</span>`;
  }

  function applyLogFilters() {
    const sig = state.log || [];
    const tk = state.logTicker.trim().toUpperCase();
    state.logFiltered = sig.filter((s) => {
      if (state.logStatus !== "all" && s.status !== state.logStatus) return false;
      const eng = s.engine === "v3-sigmaclear" ? "v3"
        : (s.engine || "").startsWith("t2") ? "t2" : "v1";
      if (state.logEngine !== "all" && eng !== state.logEngine) return false;
      if (tk && s.ticker !== tk) return false;
      return true;
    });
    // newest first
    state.logFiltered.sort((a, b) =>
      a.publish_date < b.publish_date ? 1 : a.publish_date > b.publish_date ? -1 :
      a.ticker < b.ticker ? -1 : 1);
    state.logShown = 0;
    $("#log-body").innerHTML = "";
    renderMoreLog();
  }

  function renderMoreLog() {
    const CHUNK = 200;
    const rows = state.logFiltered.slice(state.logShown, state.logShown + CHUNK);
    const html = rows.map((s) => {
      const cls = s.status === "win" ? "win" : s.status === "loss" ? "loss" : "pending";
      const res = s.status === "win" ? "WIN" : s.status === "loss" ? "LOSS" : "pending";
      const eng = s.engine === "v3-sigmaclear" ? " · T1"
        : (s.engine || "").startsWith("t2") ? " · T2" : "";
      return `<tr>
        <td>${s.publish_date}</td>
        <td class="tkr">${s.ticker}</td>
        <td>${s.side.toUpperCase()} ${s.horizon}d${eng}</td>
        <td>${fmt$(s.strike)}</td>
        <td>${fmt$(s.spot_at_publish)}</td>
        <td>${s.expiry_date}</td>
        <td>${s.close_at_expiry != null ? fmt$(s.close_at_expiry) : "—"}</td>
        <td class="${cls}">${res}</td>
      </tr>`;
    }).join("");
    $("#log-body").insertAdjacentHTML("beforeend", html);
    state.logShown += rows.length;
    const more = $("#log-more");
    more.hidden = state.logShown >= state.logFiltered.length;
    more.textContent = `Show 200 more (${fmtInt(state.logFiltered.length - state.logShown)} remaining)`;
  }

  function wireLogFilters() {
    $$("#log-filters button[data-status]").forEach((b) => {
      b.addEventListener("click", () => {
        $$("#log-filters button[data-status]").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        state.logStatus = b.dataset.status;
        applyLogFilters();
      });
    });
    $$("#log-filters button[data-engine]").forEach((b) => {
      b.addEventListener("click", () => {
        $$("#log-filters button[data-engine]").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        state.logEngine = b.dataset.engine;
        applyLogFilters();
      });
    });
    $("#log-ticker").addEventListener("input", (e) => {
      state.logTicker = e.target.value;
      applyLogFilters();
    });
    $("#log-more").addEventListener("click", renderMoreLog);
  }

  /* ---------------- boot ---------------- */

  function wireSort() {
    $$("#trade-sort button[data-sort]").forEach((b) => {
      b.addEventListener("click", () => {
        $$("#trade-sort button[data-sort]").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        state.sort = b.dataset.sort;
        renderTrades();
      });
    });
    $$("#trade-tier button[data-tier]").forEach((b) => {
      b.addEventListener("click", () => {
        $$("#trade-tier button[data-tier]").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        state.tierFilter = b.dataset.tier;
        renderTrades();
      });
    });
  }

  async function boot() {
    try {
      const d = await fetchJSON("data/signals.json");
      state.trades = flattenTrades(d);
      state.reality = (d.summary || {}).reality || null;
      renderStats(d);
      renderConvictionPick(d);
      renderTrades();
      wireSort();
    } catch (e) {
      $("#trade-list").innerHTML =
        `<div class="cf-empty">Could not load today's signals. Try a hard refresh.</div>`;
    }
    try {
      const log = await fetchJSON("data/live_log.json");
      state.log = log.signals || [];
      renderLiveStats(log);
      wireLogFilters();
      applyLogFilters();
    } catch (e) {
      $("#log-body").innerHTML =
        `<tr><td colspan="8" style="color:var(--muted)">Live log unavailable.</td></tr>`;
    }
    try {
      const h = await fetchJSON("data/tier2_history.json");
      renderTier2History(h);
    } catch (e) {
      const b = $("#t2-trade-body");
      if (b) b.innerHTML =
        `<tr><td colspan="10" style="color:var(--muted)">Tier 2 backtest history unavailable.</td></tr>`;
    }
    try {
      const c = await fetchJSON("data/conviction_history.json");
      renderConvictionHistory(c);
    } catch (e) {
      const b = $("#cp-trade-body");
      if (b) b.innerHTML =
        `<tr><td colspan="10" style="color:var(--muted)">Conviction Pick history unavailable.</td></tr>`;
    }
  }

  /* ---------------- Conviction Pick (the headline trade) ---------------- */

  function renderConvictionPick(d) {
    const el = $("#pick-card");
    const cp = d.conviction_pick;
    const asof = ((d.put_signals || []).concat(d.call_signals || [])[0]
      || {}).end_date || (d.generated_at || "").slice(0, 10);
    if (!cp || !cp.real) {
      el.innerHTML = `<div class="cf-empty">
        No Conviction Pick right now — no setup cleared the 96.9%-validated
        bar at the last scan${asof ? " (" + asof + ")" : ""}. That&rsquo;s
        expected: the pick fires only about once every 2&ndash;3 weeks. Waiting
        for conviction beats forcing a weaker trade. Check back after the next
        market close.
      </div>`;
      return;
    }
    const r = cp.real;
    const annRor = (r.ror_natural * 100) * (365 / Math.max(r.cal_days_to_expiry, 1));
    el.innerHTML = `
      <div class="cf-card" style="border-width:2px;border-color:#6b21a8">
        <div class="cf-card-head">
          <span>
            <span class="cf-card-ticker">${cp.ticker}</span>
            <span class="cf-side-badge" style="color:#6b21a8;border-color:#6b21a8">conviction pick</span>
            <span class="cf-side-badge" style="color:var(--green);border-color:var(--green)">verified live chain</span>
          </span>
          <span class="cf-card-price">last close <strong>${fmt$(cp.today_close)}</strong> · ${cp.end_date}</span>
        </div>
        <div class="cf-trade-line">
          <strong>SELL</strong> the ${fmt$(r.short_strike)} put ·
          <strong>BUY</strong> the ${fmt$(r.long_strike)} put ·
          expires <strong>${r.expiry}</strong> (${r.cal_days_to_expiry} days) · hold to expiry.
          Wins if ${cp.ticker} closes at or above ${fmt$(r.short_strike)} on ${r.expiry}.
        </div>
        <div class="cf-trade-nums">
          <div class="cf-num"><span class="v good">${fmt$(r.net_natural_credit * 100, 0)}</span><span class="k">Credit / contract</span></div>
          <div class="cf-num"><span class="v">${fmt$(r.max_loss * 100, 0)}</span><span class="k">Max loss / contract</span></div>
          <div class="cf-num"><span class="v good">${fmtPct(r.ror_natural * 100, 1)}</span><span class="k">Return on risk</span></div>
          <div class="cf-num"><span class="v">${fmtPct(annRor, 0)}</span><span class="k">Annualized</span></div>
          <div class="cf-num"><span class="v">${fmtPct(cp.gbm_confidence * 100, 1)}</span><span class="k">Model confidence</span></div>
          <div class="cf-num"><span class="v">$${fmtInt(Math.round((r.adv_usd || 0) / 1e6))}M</span><span class="k">Underlying volume/day</span></div>
        </div>
        <div class="cf-detail-kv" style="margin-top:10px">
          Live-chain quotes (delayed, as of ${r.quote_time}): short ${fmt$(r.short_strike)}
          bid ${fmt$(r.short_bid)} / ask ${fmt$(r.short_ask)} (OI ${fmtInt(r.short_oi)}) ·
          long ${fmt$(r.long_strike)} bid ${fmt$(r.long_bid)} / ask ${fmt$(r.long_ask)} (OI ${fmtInt(r.long_oi)}).
          Published credit is the natural credit (sell bid / buy ask), the fill available without
          negotiating. Validated 96.9% win rate, ~24.7% ROR per trade on untouched 2019&ndash;2026 —
          <strong>not a guarantee</strong>; ~3 of 100 lose.
        </div>
      </div>`;
  }

  function renderConvictionHistory(h) {
    const s = h.summary || {};
    if ($("#cp-trades")) $("#cp-trades").textContent = fmtInt(s.trades || 0);
    if ($("#cp-wr")) $("#cp-wr").textContent = s.win_rate != null ? fmtPct(s.win_rate, 2) : "—";
    if ($("#cp-ror")) $("#cp-ror").textContent = s.avg_ror != null ? fmtPct(s.avg_ror, 1) : "—";
    if ($("#cp-pnl")) $("#cp-pnl").textContent = s.pnl != null
      ? (s.pnl >= 0 ? "+" : "") + "$" + fmtInt(Math.round(s.pnl)) : "—";
    const yb = $("#cp-year-body");
    if (yb) yb.innerHTML = (h.by_year || []).map((y) =>
      `<tr><td>${y.year}</td><td>${fmtInt(y.trades)}</td>
       <td class="${y.losses ? "bad" : "good"}">${y.losses}</td>
       <td>${fmtPct(y.win_rate, 1)}</td>
       <td class="${y.pnl >= 0 ? "good" : "bad"}">${y.pnl >= 0 ? "+" : "−"}$${fmtInt(Math.abs(Math.round(y.pnl)))}</td></tr>`
    ).join("");
    state.cptrades = h.trades || [];
    state.cpshown = 0;
    if ($("#cp-trade-body")) $("#cp-trade-body").innerHTML = "";
    renderMoreCP();
    if ($("#cp-more")) $("#cp-more").addEventListener("click", renderMoreCP);
  }

  function renderMoreCP() {
    const CHUNK = 200;
    const rows = (state.cptrades || []).slice(state.cpshown, state.cpshown + CHUNK);
    const html = rows.map((t) => {
      const cls = t.outcome === "win" ? "win" : "loss";
      return `<tr>
        <td>${t.date}</td><td class="tkr">${t.ticker}</td>
        <td>${fmt$(t.short_strike)}</td><td>${fmt$(t.long_strike)}</td>
        <td>${t.expiry}</td><td>$${fmtInt(Math.round(t.credit))}</td>
        <td>${fmtPct(t.ror, 0)}</td><td>${fmtPct(t.confidence, 1)}</td>
        <td>${t.close_at_expiry != null ? fmt$(t.close_at_expiry) : "—"}</td>
        <td class="${cls}">${t.outcome.toUpperCase()}</td>
      </tr>`;
    }).join("");
    $("#cp-trade-body").insertAdjacentHTML("beforeend", html);
    state.cpshown += rows.length;
    const more = $("#cp-more");
    if (more) {
      more.hidden = state.cpshown >= state.cptrades.length;
      more.textContent = `Show 200 more (${fmtInt(state.cptrades.length - state.cpshown)} remaining)`;
    }
  }

  /* ---------------- Tier 2 backtest history ---------------- */

  function renderTier2History(h) {
    const s = h.summary || {};
    $("#t2-trades").textContent = fmtInt(s.trades || 0);
    $("#t2-wr").textContent = s.win_rate != null ? fmtPct(s.win_rate, 2) : "—";
    $("#t2-ror").textContent = s.avg_ror != null ? fmtPct(s.avg_ror, 1) : "—";
    $("#t2-pnl").textContent = s.pnl != null
      ? (s.pnl >= 0 ? "+" : "") + "$" + fmtInt(Math.round(s.pnl)) : "—";
    $("#t2-year-body").innerHTML = (h.by_year || []).map((y) =>
      `<tr><td>${y.year}</td><td>${fmtInt(y.trades)}</td>
       <td class="${y.losses ? "bad" : "good"}">${y.losses}</td>
       <td>${fmtPct(y.win_rate, 1)}</td>
       <td class="${y.pnl >= 0 ? "good" : "bad"}">${y.pnl >= 0 ? "+" : "−"}$${fmtInt(Math.abs(Math.round(y.pnl)))}</td></tr>`
    ).join("");
    state.t2trades = h.trades || [];
    state.t2shown = 0;
    $("#t2-trade-body").innerHTML = "";
    renderMoreT2();
    $("#t2-more").addEventListener("click", renderMoreT2);
  }

  function renderMoreT2() {
    const CHUNK = 200;
    const rows = state.t2trades.slice(state.t2shown, state.t2shown + CHUNK);
    const html = rows.map((t) => {
      const cls = t.outcome === "win" ? "win" : "loss";
      return `<tr>
        <td>${t.date}</td><td class="tkr">${t.ticker}</td>
        <td>PUT ${t.expiry ? "" : ""}spread</td>
        <td>${fmt$(t.short_strike)}</td><td>${fmt$(t.long_strike)}</td>
        <td>${t.expiry}</td><td>$${fmtInt(Math.round(t.credit))}</td>
        <td>${fmtPct(t.ror, 0)}</td>
        <td>${t.close_at_expiry != null ? fmt$(t.close_at_expiry) : "—"}</td>
        <td class="${cls}">${t.outcome.toUpperCase()}</td>
      </tr>`;
    }).join("");
    $("#t2-trade-body").insertAdjacentHTML("beforeend", html);
    state.t2shown += rows.length;
    const more = $("#t2-more");
    if (more) {
      more.hidden = state.t2shown >= state.t2trades.length;
      more.textContent = `Show 200 more (${fmtInt(state.t2trades.length - state.t2shown)} remaining)`;
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
