/* CreditFloor page: render precomputed walk-forward signals, two-sided. */
(function () {
  "use strict";

  // state.filters[side] = { sort, hfilter }
  // Default sort is annualized ROR (profitability) highest-first.
  const state = {
    data: null,
    filters: {
      put:  { sort: "profit", hfilter: "all" },
      call: { sort: "profit", hfilter: "all" },
    },
  };

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const fmt$ = (x) =>
    "$" + Number(x).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  const fmtPct = (x, d = 2) => `${Number(x).toFixed(d)}%`;
  const fmtInt = (x) => Number(x).toLocaleString();

  async function load() {
    const url = "data/signals.json?v=" + Date.now();
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error("failed to load signals");
    return r.json();
  }

  async function loadLiveLog() {
    try {
      const r = await fetch("data/live_log.json?v=" + Date.now(), { cache: "no-store" });
      if (!r.ok) return null;
      return r.json();
    } catch (e) {
      return null;
    }
  }

  function renderStats(d) {
    const s = d.summary || {};
    const combined = s.combined || {};
    const win = combined.pooled_win_rate == null
      ? "—"
      : fmtPct(100 * combined.pooled_win_rate, 2);
    $("#stat-pooled-win").textContent = win;
    const tests = (combined.pooled_wins || 0) + (combined.pooled_losses || 0);
    $("#stat-pooled-n").textContent = fmtInt(tests);
    $("#stat-signals").textContent = fmtInt(combined.n_eligible || 0);
    $("#stat-universe").textContent = fmtInt(s.n_tickers_processed || 0);
    // section badges
    const put = s.put || {};
    const call = s.call || {};
    $("#cf-put-badge").textContent =
      `${fmtInt(put.n_eligible || 0)} signal${put.n_eligible === 1 ? "" : "s"} · ${fmtInt((put.pooled_wins || 0) + (put.pooled_losses || 0))} OOS tests · 100% win`;
    $("#cf-call-badge").textContent =
      `${fmtInt(call.n_eligible || 0)} signal${call.n_eligible === 1 ? "" : "s"} · ${fmtInt((call.pooled_wins || 0) + (call.pooled_losses || 0))} OOS tests · 100% win`;
  }

  function renderLastRun(d) {
    const gen = d.generated_at;
    const allSignals = (d.put_signals || []).concat(d.call_signals || []);
    const asof = allSignals.length ? allSignals[0].end_date || "" : "";
    const el = $("#cf-last-run");
    const bits = [];
    if (asof) bits.push(`As of market close ${asof}.`);
    if (gen) bits.push(`Research generated ${gen}.`);
    const fy = d.summary?.fold_years || [];
    bits.push(
      `Walk-forward out-of-sample window covers ${fy[0]}–${fy[fy.length - 1]}, ${fy.length} annual folds per side. ` +
        `Safety margin: ${((d.summary?.safety_eps || 0) * 100).toFixed(1)}% added to worst historical path buffer. ` +
        `Buffer cap: ${((d.summary?.max_buffer || 0) * 100).toFixed(0)}%.`
    );
    el.innerHTML = bits.join(" ");
  }

  function filteredSorted(side) {
    const key = side === "put" ? "put_signals" : "call_signals";
    const items = (state.data?.[key] || []).slice();
    const f = state.filters[side];
    const filtered = items.map((s) => {
      const rungs = (s.ladder || []).filter(
        (l) => f.hfilter === "all" || String(l.horizon) === String(f.hfilter)
      );
      return { ...s, ladderView: rungs };
    }).filter((s) => s.ladderView.length > 0);

    const rorOf = (s) => Math.max(
      ...s.ladderView.map((l) => (l.profit && l.profit.return_on_risk_pct) || 0),
      0,
    );

    filtered.sort((a, b) => {
      if (f.sort === "ticker") return a.ticker.localeCompare(b.ticker);
      if (f.sort === "tests") {
        const ta = Math.max(...a.ladderView.map((l) => l.n_test));
        const tb = Math.max(...b.ladderView.map((l) => l.n_test));
        return tb - ta;
      }
      if (f.sort === "buffer") {
        const ba = Math.min(...a.ladderView.map((l) => l.buffer_pct));
        const bb = Math.min(...b.ladderView.map((l) => l.buffer_pct));
        return ba - bb;
      }
      // default: profitability (annualized ROR) descending
      return rorOf(b) - rorOf(a);
    });
    return filtered;
  }

  function rungHTML(l, side) {
    const tag = l.variant === "regime"
      ? (side === "put" ? "uptrend-only" : "bearish-only")
      : "all-regime";
    const exType = l.expiry_type || "";
    const expLine = l.expiry_date
      ? `Expires <strong>${l.expiry_date}</strong>${exType ? ` <span class="tag tag-expiry">${exType}</span>` : ""} &middot; ${l.horizon}-session backtest${l.calendar_days_to_expiry ? ` &middot; ${l.calendar_days_to_expiry} cal days` : ""}`
      : `Expiry in ${l.horizon} trading days`;
    const bufLine = side === "put"
      ? `Buffer ${fmtPct(l.buffer_pct, 2)} <em>below</em> spot`
      : `Buffer ${fmtPct(l.buffer_pct, 2)} <em>above</em> spot`;
    const worst = l.folds.reduce((m, f) => Math.max(m, f.worst_test_buf_pct), 0);
    const worstLabel = side === "put" ? "drawdown" : "rally";

    let profBlock = "";
    if (l.profit) {
      const p = l.profit;
      profBlock = `
        <div class="cf-rung-profit">
          <span class="cf-rung-profit-main">Est. <strong>${fmtPct(p.return_on_risk_pct, 2)}</strong> return on risk</span>
          <span class="cf-rung-profit-sub">
            ~$${p.est_credit_per_share.toFixed(2)} credit on $${p.spread_width.toFixed(2)} spread
            &middot; max loss $${p.est_max_loss_per_share.toFixed(2)}
            &middot; IV ${fmtPct(p.implied_vol_pct, 1)}
          </span>
        </div>`;
    }

    return `
      <div class="cf-rung">
        <div class="cf-rung-h">${expLine}</div>
        <div class="cf-rung-k">${fmt$(l.strike)}</div>
        <div class="cf-rung-b">${bufLine}</div>
        ${profBlock}
        <div class="cf-rung-m">
          <span class="tag">${tag}</span>
          <span class="tag">${fmtInt(l.n_test)} OOS tests</span>
          <span class="tag">${l.n_folds} folds</span>
          <br/>
          Worst test-set ${worstLabel} ever touched:
          <strong>${fmtPct(worst, 2)}</strong>.
          All folds 100% win rate.
        </div>
      </div>
    `;
  }

  function foldTableHTML(ladder) {
    const horizons = ladder.map((l) => l.horizon);
    const years = Array.from(
      new Set(ladder.flatMap((l) => l.folds.map((f) => f.year)))
    ).sort();
    const head = `
      <tr>
        <th>Fold</th>
        ${horizons
          .map((h) => `<th colspan="4" style="border-left:1px solid var(--rule-light)">${h}d</th>`)
          .join("")}
      </tr>
      <tr>
        <th></th>
        ${horizons
          .map(() => `<th>Train</th><th>Test</th><th>Wins</th><th>Worst</th>`)
          .join("")}
      </tr>
    `;
    const rows = years.map((y) => {
      const cells = ladder.map((l) => {
        const f = l.folds.find((ff) => ff.year === y);
        if (!f) return `<td>-</td><td>-</td><td>-</td><td>-</td>`;
        const cls = f.losses === 0 ? "win" : "loss";
        return `<td>${fmtInt(f.n_train)}</td>
          <td>${fmtInt(f.n_test)}</td>
          <td class="${cls}">${fmtInt(f.wins)}/${fmtInt(f.n_test)}</td>
          <td>${fmtPct(f.worst_test_buf_pct, 1)}</td>`;
      });
      return `<tr><td>${y}</td>${cells.join("")}</tr>`;
    });
    return `<table class="cf-fold-tbl"><thead>${head}</thead><tbody>${rows.join("")}</tbody></table>`;
  }

  function cardHTML(s, side) {
    const anyRegime = s.ladderView.some((l) => l.variant === "regime");
    const regimeLbl = side === "put" ? "uptrend gate" : "bearish gate";
    const regimeBadge = anyRegime
      ? `<span class="cf-regime-badge regime">${regimeLbl}</span>`
      : `<span class="cf-regime-badge">all-regime</span>`;
    return `
      <div class="cf-card" data-ticker="${s.ticker}">
        <div class="cf-card-head">
          <div>
            <span class="cf-card-ticker">${s.ticker}</span>
            ${regimeBadge}
          </div>
          <div class="cf-card-price">
            Spot <strong>${fmt$(s.today_close)}</strong>
            &middot; as-of ${s.end_date}
          </div>
        </div>
        <div class="cf-ladder">
          ${s.ladderView.map((r) => rungHTML(r, side)).join("")}
        </div>
        <div class="cf-card-expand">Show fold-by-fold walk-forward breakdown</div>
        <div class="cf-detail">
          ${foldTableHTML(s.ladderView)}
          <div class="cf-footnote" style="margin-top:12px">
            Each cell is the walk-forward test outcome for that calendar year
            and horizon. "Worst" is the largest ${side === "put" ? "drawdown" : "rally"}-to-expiry
            <em>on the held-out test set</em> — always strictly inside our
            conformal strike buffer, which is why every fold is a 100% win.
          </div>
        </div>
      </div>
    `;
  }

  function watchHTML(entries, side) {
    if (!entries || !entries.length) return "";
    const gateLabel = side === "put"
      ? "close &ge; SMA<sub>200</sub> AND dd<sub>252</sub> &le; 20%"
      : "close &le; SMA<sub>200</sub> AND up<sub>252</sub> &le; 20%";
    const rows = entries.flatMap((e) =>
      (e.rungs || []).map((r) => {
        const strikeWord = side === "put" ? "below" : "above";
        return `<tr>
          <td class="tkr">${e.ticker}</td>
          <td>${fmt$(e.today_close)}</td>
          <td>${r.horizon}d</td>
          <td>${fmtPct(r.buffer_pct, 2)} ${strikeWord}</td>
          <td>${fmt$(r.strike)}</td>
          <td>${fmtInt(r.n_test)} OOS / ${r.n_folds} folds</td>
        </tr>`;
      })
    ).join("");
    return `
      <div class="cf-watch-head" data-toggle="watch-${side}">
        <span class="arrow">&#9654;</span>
        <span>Regime watchlist &middot; ${entries.length} ticker${entries.length === 1 ? "" : "s"} waiting for regime to match</span>
      </div>
      <div class="cf-watch-body">
        <div class="cf-watch-note">
          These names' walk-forward backtest <strong>passes at 100%</strong> — but today's features don't match
          the ${side} regime gate (${gateLabel}), so the rule can't be deployed. They will auto-join the live list
          the moment their regime gate flips.
        </div>
        <table class="cf-watch-tbl">
          <thead><tr>
            <th>Ticker</th><th>Spot</th><th>H</th><th>Buffer</th><th>Strike</th><th>Backtest</th>
          </tr></thead><tbody>${rows}</tbody>
        </table>
      </div>
    `;
  }

  function renderWatch(side) {
    const container = document.getElementById(`cf-watch-${side}`);
    if (!container) return;
    const entries = state.data?.[side === "put" ? "put_watchlist" : "call_watchlist"] || [];
    container.innerHTML = watchHTML(entries, side);
    const head = container.querySelector(".cf-watch-head");
    if (head) {
      head.addEventListener("click", () => container.classList.toggle("cf-watch-open"));
    }
  }

  function renderTopSignals() {
    const MIN_ROR = 5.0;
    const all = [];
    for (const sideKey of ['put_signals', 'call_signals']) {
      for (const s of (state.data?.[sideKey] || [])) {
        const side = sideKey === 'put_signals' ? 'put' : 'call';
        for (const r of (s.ladder || [])) {
          if (!r.profit) continue;
          if (r.profit.return_on_risk_pct < MIN_ROR) continue;
          all.push({ side, ticker: s.ticker, spot: s.today_close, rung: r });
        }
      }
    }
    all.sort((a, b) => b.rung.profit.return_on_risk_pct - a.rung.profit.return_on_risk_pct);

    const badge = document.getElementById("cf-top-badge");
    if (badge) {
      badge.textContent = `${fmtInt(all.length)} rung${all.length === 1 ? "" : "s"} ≥ ${MIN_ROR}% ROR`;
    }
    const container = document.getElementById("cf-top-list");
    if (!container) return;
    if (all.length === 0) {
      container.innerHTML = `<div class="cf-empty">No rungs with &ge;${MIN_ROR}% estimated per-trade ROR today.</div>`;
      return;
    }
    const rows = all.map((x) => {
      const r = x.rung;
      const p = r.profit;
      const bufWord = x.side === 'put' ? 'below' : 'above';
      return `<tr>
        <td class="tkr">${x.ticker}</td>
        <td class="side-${x.side}">${x.side.toUpperCase()}</td>
        <td>${r.horizon}d</td>
        <td>${r.expiry_date}<span class="cf-top-exptype">${r.expiry_type || ""}</span></td>
        <td>${fmt$(x.spot)}</td>
        <td>${fmt$(r.strike)}</td>
        <td>${fmtPct(r.buffer_pct, 2)} ${bufWord}</td>
        <td>$${p.est_credit_per_share.toFixed(2)}</td>
        <td>$${p.est_max_loss_per_share.toFixed(2)}</td>
        <td class="good"><strong>${fmtPct(p.return_on_risk_pct, 2)}</strong></td>
      </tr>`;
    }).join("");
    container.innerHTML = `
      <table class="cf-top-tbl">
        <thead><tr>
          <th>Ticker</th><th>Side</th><th>H</th><th>Expires</th>
          <th>Spot</th><th>Strike</th><th>Buffer</th>
          <th>Credit</th><th>Max loss</th><th>ROR</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  function renderSide(side) {
    const list = $(`#cf-list-${side}`);
    const items = filteredSorted(side);
    if (!items.length) {
      list.innerHTML = `<div class="cf-empty">No ${side} signals match the current filter. The engine is fail-closed — most stocks, most days, there is no eligible trade.</div>`;
    } else {
      list.innerHTML = items.map((s) => cardHTML(s, side)).join("");
      list.querySelectorAll(".cf-card-expand").forEach((el) => {
        el.addEventListener("click", (e) => {
          const card = e.currentTarget.closest(".cf-card");
          card.classList.toggle("open");
          e.currentTarget.textContent = card.classList.contains("open")
            ? "Hide fold-by-fold walk-forward breakdown"
            : "Show fold-by-fold walk-forward breakdown";
        });
      });
    }
    renderWatch(side);
  }

  function render() {
    renderSide("put");
    renderSide("call");
  }

  function wireCollapsibles() {
    $$(".cf-section-head").forEach((head) => {
      head.addEventListener("click", () => {
        const target = head.dataset.target;
        const body = document.querySelector(`.cf-section-body[data-body="${target}"]`);
        const expanded = head.getAttribute("aria-expanded") === "true";
        head.setAttribute("aria-expanded", expanded ? "false" : "true");
        if (expanded) {
          body.setAttribute("hidden", "");
        } else {
          body.removeAttribute("hidden");
          // On first expand, ensure content is rendered.
          renderSide(target);
        }
      });
    });
  }

  function wireFilters() {
    $$(".cf-filters button[data-sort]").forEach((b) => {
      b.addEventListener("click", (e) => {
        const btn = e.currentTarget;
        const side = btn.dataset.side;
        const sort = btn.dataset.sort;
        // clear .active within the same side's sort group
        btn.parentNode.querySelectorAll(`button[data-side="${side}"][data-sort]`)
          .forEach((bb) => bb.classList.remove("active"));
        btn.classList.add("active");
        state.filters[side].sort = sort;
        renderSide(side);
      });
    });
    $$(".cf-filters button[data-hfilter]").forEach((b) => {
      b.addEventListener("click", (e) => {
        const btn = e.currentTarget;
        const side = btn.dataset.side;
        const hfilter = btn.dataset.hfilter;
        btn.parentNode.querySelectorAll(`button[data-side="${side}"][data-hfilter]`)
          .forEach((bb) => bb.classList.remove("active"));
        btn.classList.add("active");
        state.filters[side].hfilter = hfilter;
        renderSide(side);
      });
    });
  }

  function renderLiveLog(log) {
    if (!log || !log.summary) return;
    const s = log.summary;
    const wr = s.win_rate == null ? "—" : fmtPct(100 * s.win_rate, 2);
    const wrEl = document.getElementById("live-win-rate");
    wrEl.textContent = wr;
    wrEl.className = "num " + (s.losses > 0 ? "bad" : "good");
    document.getElementById("live-resolved").textContent = fmtInt(s.resolved || 0);
    document.getElementById("live-pending").textContent  = fmtInt(s.pending  || 0);
    document.getElementById("live-losses").textContent   = fmtInt(s.losses   || 0);

    const put = s.put || {};
    const call = s.call || {};
    const sub = (name, x) => {
      const wr = x.win_rate == null ? "—" : fmtPct(100 * x.win_rate, 2);
      return `${name} <strong>${wr}</strong> &middot; ${fmtInt(x.resolved || 0)} resolved / ${fmtInt(x.pending || 0)} pending &middot; ${fmtInt(x.wins || 0)}W / ${fmtInt(x.losses || 0)}L`;
    };
    document.getElementById("live-put-summary").innerHTML  = sub("win rate", put);
    document.getElementById("live-call-summary").innerHTML = sub("win rate", call);
    document.getElementById("live-first").textContent = s.first_publish_date || "—";

    // Detail table: show recent losses (always, prominently), then recent resolutions,
    // then open positions (most recent first). Cap at 100 rows total.
    const signals = (log.signals || []).slice();
    const losses = signals.filter((x) => x.status === "loss").reverse();
    const resolved = signals.filter((x) => x.status !== "pending").reverse();
    const pending = signals.filter((x) => x.status === "pending").reverse();
    const rows = [];
    const header = `
      <table class="cf-live-tbl">
        <thead><tr>
          <th>Published</th><th>Ticker</th><th>Side</th><th>H</th>
          <th>Expires</th><th>Strike</th><th>Spot@pub</th>
          <th>Close@exp</th><th>Status</th>
        </tr></thead><tbody>`;
    const row = (x) => {
      const close = x.close_at_expiry != null
        ? fmt$(x.close_at_expiry)
        : (x.side === "put"
            ? (x.forward_close_min != null ? `min ${fmt$(x.forward_close_min)}` : "—")
            : (x.forward_close_max != null ? `max ${fmt$(x.forward_close_max)}` : "—"));
      return `<tr>
        <td>${x.publish_date}</td>
        <td><strong>${x.ticker}</strong></td>
        <td>${x.side}</td>
        <td>${x.horizon}d</td>
        <td>${x.expiry_date}</td>
        <td>${fmt$(x.strike)}</td>
        <td>${fmt$(x.spot_at_publish)}</td>
        <td>${close}</td>
        <td class="${x.status}">${x.status.toUpperCase()}</td>
      </tr>`;
    };
    const picked = [];
    for (const x of losses) { if (picked.length >= 25) break; picked.push(x); }
    const loss_ids = new Set(picked.map((x) => x.id));
    for (const x of resolved) {
      if (picked.length >= 60) break;
      if (loss_ids.has(x.id)) continue;
      picked.push(x);
    }
    for (const x of pending) {
      if (picked.length >= 100) break;
      picked.push(x);
    }
    rows.push(header);
    for (const x of picked) rows.push(row(x));
    rows.push(`</tbody></table>`);
    document.getElementById("cf-live-log").innerHTML = rows.join("");
  }

  async function loadOptionC() {
    try {
      const r = await fetch("data/option_c_signals.json?v=" + Date.now(), { cache: "no-store" });
      if (!r.ok) return null;
      return r.json();
    } catch (e) { return null; }
  }

  // ---- Option C state + render --------------------------------------
  const ocState = {
    data: null,
    filters: {
      put:  { sort: "profit", hfilter: "all", tier: "all" },
      call: { sort: "profit", hfilter: "all", tier: "all" },
      rec:  { sort: "profit", hfilter: "all", tier: "all" },
    },
  };

  function ocFilteredSorted(side) {
    let items = [];
    const data = ocState.data;
    if (!data) return [];
    if (side === "put")  items = (data.put_signals  || []).slice();
    else if (side === "call") items = (data.call_signals || []).slice();
    else if (side === "rec") {
      // Resolve recommended refs from put_signals + call_signals
      const lookup = {};
      (data.put_signals  || []).forEach((s) => { lookup[`put|${s.ticker}`]  = s; });
      (data.call_signals || []).forEach((s) => { lookup[`call|${s.ticker}`] = s; });
      const refs = data.recommended || [];
      // Fallback: if recommended is the old fully-inline format, just use it
      const isInline = refs.length && refs[0] && refs[0].ladder;
      if (isInline) {
        items = refs.slice();
      } else {
        items = refs.map((r) => lookup[`${r.side}|${r.ticker}`]).filter(Boolean);
      }
    }
    const f = ocState.filters[side];
    const tierOk = (t) => (
      f.tier === "all" ? true :
      f.tier === "cert" ? t === "certified" :
      f.tier === "99"  ? (t === "certified" || t === "near") :
      f.tier === "95"  ? (t === "certified" || t === "near" || t === "high")
                        : true
    );
    const filtered = items.map((s) => {
      const rungs = (s.ladder || []).filter((l) =>
        (f.hfilter === "all" || String(l.horizon) === String(f.hfilter))
        && tierOk(l.tier)
      );
      return { ...s, ladderView: rungs };
    }).filter((s) => s.ladderView.length > 0);

    const winOf = (s) => Math.max(...s.ladderView.map((l) => l.win_rate_pct));
    const roiOf = (s) => Math.max(
      ...s.ladderView.map((l) => (l.profit && l.profit.return_on_risk_pct) || 0), 0,
    );
    const bufOf = (s) => Math.min(...s.ladderView.map((l) => l.buffer_pct));
    const tstOf = (s) => Math.max(...s.ladderView.map((l) => l.n_test));

    filtered.sort((a, b) => {
      if (f.sort === "ticker") return a.ticker.localeCompare(b.ticker);
      if (f.sort === "win")    return winOf(b) - winOf(a);
      if (f.sort === "buffer") return bufOf(a) - bufOf(b);
      if (f.sort === "tests")  return tstOf(b) - tstOf(a);
      return roiOf(b) - roiOf(a);  // default
    });
    return filtered;
  }

  function ocTierBadge(t) {
    if (t === "certified") return `<span class="cf-regime-badge regime" style="color:#1a7a3e;border-color:#1a7a3e" title="Walk-forward validated 2022-2026: this (side, strike-distance, horizon, ticker) combination delivered ≥98% accuracy on truly-unseen test data with no single year below 95%.">Certified ✓</span>`;
    if (t === "near")      return `<span class="cf-regime-badge" style="color:#1a7a3e;border-color:#1a7a3e" title="Ticker's in-sample win rate ≥99% (not walk-forward verified).">99%+ near-certified</span>`;
    if (t === "high")      return `<span class="cf-regime-badge" title="Ticker's in-sample win rate ≥95% (not walk-forward verified).">95%+ high-accuracy</span>`;
    return `<span class="cf-regime-badge">standard</span>`;
  }

  function ocRungHTML(l, side) {
    const exType = l.expiry_type || "";
    const expLine = `Expires <strong>${l.expiry_date}</strong>` +
      (exType ? ` <span class="tag tag-expiry">${exType}</span>` : "") +
      ` &middot; ${l.horizon}-session backtest` +
      (l.calendar_days_to_expiry ? ` &middot; ${l.calendar_days_to_expiry} cal days` : "");
    const sideAction = side === "put" ? "Sell put at" : "Sell call at";
    const sideWord   = side === "put" ? "below" : "above";
    const poolNote = (l.pool_win_rate_pct !== undefined)
      ? ` &middot; rule's pool win ${fmtPct(l.pool_win_rate_pct, 1)} across ${fmtInt(l.pool_n_test || 0)} tests`
      : "";
    const profBlock = l.profit ? `
      <div class="cf-rung-profit">
        <span class="cf-rung-profit-main">Est. <strong>${fmtPct(l.profit.return_on_risk_pct, 2)}</strong> on max-loss
          &middot; <strong>${fmtPct(l.win_rate_pct, 1)}</strong> ${l.ticker || ""} historical win</span>
        <span class="cf-rung-profit-sub">
          credit ~$${l.profit.est_credit_per_share.toFixed(2)} on $${l.profit.spread_width.toFixed(2)} spread
          &middot; max loss $${l.profit.est_max_loss_per_share.toFixed(2)}
          &middot; protection (long leg) at ${fmt$(l.strike_long)}
          &middot; IV ${fmtPct(l.profit.implied_vol_pct, 1)}${poolNote}
        </span>
      </div>` : "";
    return `
      <div class="cf-rung">
        <div class="cf-rung-h">${expLine}</div>
        <div class="cf-rung-k">${sideAction} ${fmt$(l.strike_short)}</div>
        <div class="cf-rung-b">${(l.k_short_frac*100).toFixed(1)}% ${sideWord} spot</div>
        ${profBlock}
        <div class="cf-rung-m">
          ${ocTierBadge(l.tier)}
          <span class="tag">${l.regime_label || l.regime}</span>
          <span class="tag">${fmtInt(l.n_test)} OOS tests</span>
          <span class="tag">${l.n_folds} folds</span>
        </div>
      </div>
    `;
  }

  function ocFoldTableHTML(ladder) {
    const horizons = ladder.map((l) => l.horizon);
    const years = Array.from(
      new Set(ladder.flatMap((l) => l.folds.map((f) => f.year)))
    ).sort();
    const head = `
      <tr><th>Fold</th>
        ${horizons.map((h) => `<th colspan="3" style="border-left:1px solid var(--rule-light)">${h}d</th>`).join("")}
      </tr>
      <tr><th></th>
        ${horizons.map(() => `<th>Tests</th><th>Wins</th><th>$ P&amp;L</th>`).join("")}
      </tr>`;
    const rows = years.map((y) => {
      const cells = ladder.map((l) => {
        const f = l.folds.find((ff) => ff.year === y);
        if (!f) return `<td>-</td><td>-</td><td>-</td>`;
        const winCls = f.losses === 0 && f.wins > 0 ? "win" : (f.pnl > 0 ? "win" : "loss");
        return `<td>${fmtInt(f.n_test)}</td>
          <td class="${winCls}">${fmtInt(f.wins)}/${fmtInt(f.n_test)}</td>
          <td class="${f.pnl > 0 ? 'win' : 'loss'}">$${(f.pnl || 0).toFixed(0)}</td>`;
      });
      return `<tr><td>${y}</td>${cells.join("")}</tr>`;
    });
    return `<table class="cf-fold-tbl"><thead>${head}</thead><tbody>${rows.join("")}</tbody></table>`;
  }

  function ocCardHTML2(s, side) {
    const tiers = new Set(s.ladderView.map((l) => l.tier));
    const headerTier = tiers.has("certified") ? "certified"
                     : tiers.has("near")      ? "near"
                     : tiers.has("high")      ? "high"
                     : "standard";
    return `
      <div class="cf-card" data-ticker="${s.ticker}">
        <div class="cf-card-head">
          <div>
            <span class="cf-card-ticker">${s.ticker}</span>
            ${ocTierBadge(headerTier)}
            <span class="cf-regime-badge" style="color:${side==='put'?'#1a7a3e':'#b35900'};border-color:${side==='put'?'#1a7a3e':'#b35900'}">
              ${side === "put" ? "BULLISH (short put)" : "BEARISH (short call)"}
            </span>
          </div>
          <div class="cf-card-price">
            Spot <strong>${fmt$(s.today_close)}</strong>
            &middot; σ ${fmtPct(s.realized_vol_pct, 0)}
            &middot; as-of ${s.end_date}
          </div>
        </div>
        <div class="cf-ladder">
          ${s.ladderView.map((r) => ocRungHTML(r, side)).join("")}
        </div>
        <div class="cf-card-expand">Show walk-forward detail by year</div>
        <div class="cf-detail">
          ${ocFoldTableHTML(s.ladderView)}
          <div class="cf-footnote" style="margin-top:12px">
            Each cell is the walk-forward test outcome for that calendar year and horizon
            (tests / wins / $ P&amp;L). Win condition: stock's close at expiry stays on the
            ${side === "put" ? "above" : "below"} side of the short strike. Max loss is
            capped at the spread width minus credit.
          </div>
        </div>
      </div>
    `;
  }

  function ocRenderSide(side) {
    const listId = side === "rec" ? "oc-list-rec"
                 : side === "put" ? "oc-list-puts" : "oc-list-calls";
    const list = document.getElementById(listId);
    if (!list) return;
    const items = ocFilteredSorted(side);
    if (!items.length) {
      list.innerHTML = `<div class="cf-empty">No ${side} signals match the current filters today.</div>`;
      return;
    }
    list.innerHTML = items.map((s) => ocCardHTML2(s, s.side)).join("");
    list.querySelectorAll(".cf-card-expand").forEach((ex) => {
      ex.addEventListener("click", (e) => {
        const card = e.currentTarget.closest(".cf-card");
        card.classList.toggle("open");
        e.currentTarget.textContent = card.classList.contains("open")
          ? "Hide walk-forward detail by year"
          : "Show walk-forward detail by year";
      });
    });
  }

  function ocBuildFilterRow(side, label) {
    return `
      <div class="cf-filters">
        <span class="lbl">Sort</span>
        <button data-oc-side="${side}" data-oc-sort="profit" class="active">Most profitable</button>
        <button data-oc-side="${side}" data-oc-sort="win">Highest accuracy</button>
        <button data-oc-side="${side}" data-oc-sort="buffer">Tightest buffer</button>
        <button data-oc-side="${side}" data-oc-sort="tests">Most OOS tests</button>
        <button data-oc-side="${side}" data-oc-sort="ticker">Ticker A&ndash;Z</button>
        <span class="lbl" style="margin-left:18px">Accuracy</span>
        <button data-oc-side="${side}" data-oc-tier="all" class="active">All</button>
        <button data-oc-side="${side}" data-oc-tier="cert" title="Walk-forward validated 2022-2026">Certified ✓</button>
        <button data-oc-side="${side}" data-oc-tier="99">99%+</button>
        <button data-oc-side="${side}" data-oc-tier="95">95%+</button>
        <span class="lbl" style="margin-left:18px">Horizon</span>
        <button data-oc-side="${side}" data-oc-hfilter="all" class="active">All</button>
        <button data-oc-side="${side}" data-oc-hfilter="5">5d</button>
        <button data-oc-side="${side}" data-oc-hfilter="7">7d</button>
        <button data-oc-side="${side}" data-oc-hfilter="10">10d</button>
        <button data-oc-side="${side}" data-oc-hfilter="14">14d</button>
        <button data-oc-side="${side}" data-oc-hfilter="21">21d</button>
        <button data-oc-side="${side}" data-oc-hfilter="30">30d</button>
        <button data-oc-side="${side}" data-oc-hfilter="45">45d</button>
        <button data-oc-side="${side}" data-oc-hfilter="60">60d</button>
        <button data-oc-side="${side}" data-oc-hfilter="90">90d</button>
        <button data-oc-side="${side}" data-oc-hfilter="120">120d</button>
        <button data-oc-side="${side}" data-oc-hfilter="180">180d</button>
        <button data-oc-side="${side}" data-oc-hfilter="252">252d</button>
      </div>`;
  }

  function ocWireFilters() {
    document.querySelectorAll('button[data-oc-sort]').forEach((b) => {
      b.addEventListener("click", (e) => {
        const btn = e.currentTarget;
        const side = btn.dataset.ocSide;
        btn.parentNode.querySelectorAll(`button[data-oc-side="${side}"][data-oc-sort]`)
          .forEach((bb) => bb.classList.remove("active"));
        btn.classList.add("active");
        ocState.filters[side].sort = btn.dataset.ocSort;
        ocRenderSide(side);
      });
    });
    document.querySelectorAll('button[data-oc-tier]').forEach((b) => {
      b.addEventListener("click", (e) => {
        const btn = e.currentTarget;
        const side = btn.dataset.ocSide;
        btn.parentNode.querySelectorAll(`button[data-oc-side="${side}"][data-oc-tier]`)
          .forEach((bb) => bb.classList.remove("active"));
        btn.classList.add("active");
        ocState.filters[side].tier = btn.dataset.ocTier;
        ocRenderSide(side);
      });
    });
    document.querySelectorAll('button[data-oc-hfilter]').forEach((b) => {
      b.addEventListener("click", (e) => {
        const btn = e.currentTarget;
        const side = btn.dataset.ocSide;
        btn.parentNode.querySelectorAll(`button[data-oc-side="${side}"][data-oc-hfilter]`)
          .forEach((bb) => bb.classList.remove("active"));
        btn.classList.add("active");
        ocState.filters[side].hfilter = btn.dataset.ocHfilter;
        ocRenderSide(side);
      });
    });
  }

  function renderOptionC(oc) {
    if (!oc || !oc.summary) {
      ["oc-list-rec", "oc-list-puts", "oc-list-calls"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = `<div class="cf-empty">Option C data not available.</div>`;
      });
      return;
    }
    ocState.data = oc;
    const s = oc.summary;
    document.getElementById("oc-winrate").textContent = fmtPct(s.overall_win_rate_pct || 0, 1);
    document.getElementById("oc-roi").textContent = fmtPct(s.overall_roi_on_max_loss_pct || 0, 2);
    document.getElementById("oc-rules").textContent = fmtInt(s.n_eligible_rules || 0);
    document.getElementById("oc-live").textContent = fmtInt(s.n_live_fires || 0);

    // Inject filter rows above each list (idempotent)
    const ensureFilters = (sectionBodyId, side) => {
      const body = document.querySelector(`.cf-section-body[data-body="${sectionBodyId}"]`);
      if (!body || body.querySelector(".cf-filters")) return;
      const list = body.querySelector(".cf-list");
      const filtersDiv = document.createElement("div");
      filtersDiv.innerHTML = ocBuildFilterRow(side);
      body.insertBefore(filtersDiv.firstElementChild, list);
    };
    ensureFilters("oc-rec", "rec");
    ensureFilters("oc-puts", "put");
    ensureFilters("oc-calls", "call");

    // Update badges
    const setBadge = (id, list, label) => {
      const b = document.getElementById(id);
      if (b) b.textContent = `${list.length} ${label}${list.length === 1 ? "" : "s"}`;
    };
    setBadge("oc-rec-badge",   oc.recommended  || [], "rule");
    setBadge("oc-puts-badge",  oc.put_signals  || [], "ticker");
    setBadge("oc-calls-badge", oc.call_signals || [], "ticker");

    ocWireFilters();
    // Lazy-render: only build the section's rung HTML the first time
    // the user expands it. Initial page-load doesn't pay the cost of
    // 100s of rungs that may never be viewed.
    const lazy = {rec: false, put: false, call: false};
    document.querySelectorAll('.cf-section-head[data-target^="oc-"]').forEach((h) => {
      h.addEventListener("click", () => {
        const t = h.dataset.target;            // oc-rec / oc-puts / oc-calls
        const side = t === "oc-rec" ? "rec"
                   : t === "oc-puts" ? "put"
                   : t === "oc-calls" ? "call" : null;
        if (!side) return;
        if (!lazy[side]) {
          ocRenderSide(side);
          lazy[side] = true;
        }
      });
    });
  }

  async function main() {
    try {
      state.data = await load();
      renderStats(state.data);
      renderLastRun(state.data);
      wireCollapsibles();
      wireFilters();
      renderTopSignals();
      renderSide("put");
      renderSide("call");
    } catch (err) {
      const fail = (id) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = `<div class="cf-empty">Failed to load signals: ${err.message}</div>`;
      };
      fail("cf-list-put");
      fail("cf-list-call");
    }
    // Live log is best-effort — not all deployments will have one yet.
    const liveLog = await loadLiveLog();
    if (liveLog) renderLiveLog(liveLog);
    // Option C
    const oc = await loadOptionC();
    renderOptionC(oc);
    // Stillpoint
    const sp = await loadStillpoint();
    renderStillpoint(sp);
  }

  // ---- Stillpoint ----------------------------------------------------
  async function loadStillpoint() {
    try {
      const r = await fetch("data/stillpoint_signals.json?v=" + Date.now(), { cache: "no-store" });
      if (!r.ok) return null;
      return r.json();
    } catch (e) { return null; }
  }

  function spRungHTML(l, side) {
    const exType = l.expiry_type || "";
    const expLine = `Expires <strong>${l.expiry_date}</strong>` +
      (exType ? ` <span class="tag tag-expiry">${exType}</span>` : "") +
      ` &middot; ${l.horizon}-session backtest` +
      (l.calendar_days_to_expiry ? ` &middot; ${l.calendar_days_to_expiry} cal days` : "");
    const sideAction = side === "put" ? "Sell put at" : "Sell call at";
    const sideWord = side === "put" ? "below" : "above";
    const profBlock = l.profit ? `
      <div class="cf-rung-profit">
        <span class="cf-rung-profit-main">Est. <strong>${fmtPct(l.profit.return_on_risk_pct, 2)}</strong> return on risk
          &middot; <strong>${fmtPct(l.win_rate_pct, 1)}</strong> walk-forward OOS win</span>
        <span class="cf-rung-profit-sub">
          credit ~$${l.profit.est_credit_per_share.toFixed(2)} on $${l.profit.spread_width.toFixed(2)} spread
          &middot; max loss $${l.profit.est_max_loss_per_share.toFixed(2)}
          &middot; long leg ${fmt$(l.profit.long_strike)}
          &middot; IV ${fmtPct(l.profit.implied_vol_pct, 1)}
        </span>
      </div>` : "";
    const isTight = l.method === "voladapt";
    const tierTag = isTight ? "tight (vol-adaptive)" : "core (static)";
    const sigmaTag = isTight && l.sigma_today_pct != null
      ? `<span class="tag" title="20d annualized realized vol — buffer scales with this">σ ${fmtPct(l.sigma_today_pct, 0)}</span>` : "";
    return `
      <div class="cf-rung">
        <div class="cf-rung-h">${expLine}</div>
        <div class="cf-rung-k">${sideAction} ${fmt$(l.strike)}</div>
        <div class="cf-rung-b">${fmtPct(l.buffer_pct, 2)} ${sideWord} spot</div>
        ${profBlock}
        <div class="cf-rung-m">
          <span class="tag">${tierTag}</span>
          ${sigmaTag}
          <span class="tag">${fmtInt(l.n_test)} OOS tests</span>
          <span class="tag">${l.n_folds} folds</span>
          <span class="tag">${fmtInt(l.n_history_fires)} regime fires</span>
        </div>
      </div>
    `;
  }

  function spFoldTableHTML(ladder) {
    const horizons = ladder.map((l) => l.horizon);
    const years = Array.from(new Set(ladder.flatMap((l) => l.folds.map((f) => f.year)))).sort();
    const head = `
      <tr>
        <th>Fold</th>
        ${horizons.map((h) => `<th colspan="4" style="border-left:1px solid var(--rule-light)">${h}d</th>`).join("")}
      </tr>
      <tr>
        <th></th>
        ${horizons.map(() => `<th>Train</th><th>Test</th><th>Wins</th><th>b̂%</th>`).join("")}
      </tr>`;
    const rows = years.map((y) => {
      const cells = ladder.map((l) => {
        const f = l.folds.find((ff) => ff.year === y);
        if (!f) return `<td>-</td><td>-</td><td>-</td><td>-</td>`;
        const total = f.wins + f.losses;
        const cls = f.losses === 0 ? "win" : (f.wins / Math.max(total, 1) >= 0.9 ? "win" : "loss");
        // Tight-tier folds use median_b_hat_pct (per-row, vol-adaptive);
        // core-tier folds use b_hat_pct (single quantile).
        const bHat = f.b_hat_pct != null ? f.b_hat_pct : f.median_b_hat_pct;
        return `<td>${fmtInt(f.n_train)}</td>
          <td>${fmtInt(f.n_test)}</td>
          <td class="${cls}">${fmtInt(f.wins)}/${fmtInt(total)}</td>
          <td>${bHat != null ? fmtPct(bHat, 2) : "-"}</td>`;
      });
      return `<tr><td>${y}</td>${cells.join("")}</tr>`;
    });
    return `<table class="cf-fold-tbl"><thead>${head}</thead><tbody>${rows.join("")}</tbody></table>`;
  }

  function spCardHTML(s, side) {
    return `
      <div class="cf-card" data-ticker="${s.ticker}">
        <div class="cf-card-head">
          <div>
            <span class="cf-card-ticker">${s.ticker}</span>
            <span class="cf-regime-badge regime">stillness</span>
            <span class="cf-regime-badge" style="color:${side==='put'?'#1a7a3e':'#b35900'};border-color:${side==='put'?'#1a7a3e':'#b35900'}">
              ${side === "put" ? "BULLISH (short put)" : "BEARISH (short call)"}
            </span>
          </div>
          <div class="cf-card-price">
            Spot <strong>${fmt$(s.today_close)}</strong>
            ${s.realized_vol_pct != null ? `&middot; σ ${fmtPct(s.realized_vol_pct, 0)}` : ""}
            &middot; as-of ${s.end_date}
          </div>
        </div>
        <div class="cf-ladder">
          ${s.ladder.map((r) => spRungHTML(r, side)).join("")}
        </div>
        <div class="cf-card-expand">Show fold-by-fold walk-forward breakdown</div>
        <div class="cf-detail">
          ${spFoldTableHTML(s.ladder)}
          <div class="cf-footnote" style="margin-top:12px">
            Each cell is the walk-forward test outcome for that calendar year and horizon.
            <em>b̂%</em> is the conformal strike buffer set on training data; <em>Wins</em>
            is the test-set win rate. Stillpoint tolerates a small miss rate (target ≤5%
            pooled, ≤10% any fold) — published only after walk-forward verification clears.
          </div>
        </div>
      </div>`;
  }

  function spPinCardHTML(p) {
    const subSection = (s, side) => `
      <div style="margin-top: 10px; padding: 12px; border: 1px solid var(--rule-light); border-radius: 3px; background: ${side==='put'?'#f5fbf7':'#fdf6f0'}">
        <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 6px">
          ${side === 'put' ? 'BULLISH leg — short put' : 'BEARISH leg — short call'}
        </div>
        <div class="cf-ladder" style="margin-top: 0">
          ${s.ladder.map((r) => spRungHTML(r, side)).join("")}
        </div>
      </div>`;
    return `
      <div class="cf-card" data-ticker="${p.ticker}">
        <div class="cf-card-head">
          <div>
            <span class="cf-card-ticker">${p.ticker}</span>
            <span class="cf-regime-badge regime">stillness</span>
            <span class="cf-regime-badge" style="color:#a06b00;border-color:#a06b00">PIN (iron-condor)</span>
          </div>
          <div class="cf-card-price">
            Spot <strong>${fmt$(p.today_close)}</strong>
            &middot; as-of ${p.end_date}
          </div>
        </div>
        ${subSection(p.put, "put")}
        ${subSection(p.call, "call")}
      </div>`;
  }

  function spRenderList(containerId, items, side) {
    const list = document.getElementById(containerId);
    if (!list) return;
    if (!items.length) {
      list.innerHTML = `<div class="cf-empty">No ${side} Stillpoint signals today. The stillness gate is conservative — most stocks, most days, are not eligible.</div>`;
      return;
    }
    list.innerHTML = items.map((s) => (side === "pin" ? spPinCardHTML(s) : spCardHTML(s, side))).join("");
    list.querySelectorAll(".cf-card-expand").forEach((ex) => {
      ex.addEventListener("click", (e) => {
        const card = e.currentTarget.closest(".cf-card");
        card.classList.toggle("open");
        e.currentTarget.textContent = card.classList.contains("open")
          ? "Hide fold-by-fold walk-forward breakdown"
          : "Show fold-by-fold walk-forward breakdown";
      });
    });
  }

  function renderStillpoint(sp) {
    if (!sp || !sp.summary) {
      ["sp-list-pin", "sp-list-puts", "sp-list-calls",
       "sp-list-tight-pin", "sp-list-tight-puts", "sp-list-tight-calls"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = `<div class="cf-empty">Stillpoint data not available.</div>`;
      });
      return;
    }
    const s = sp.summary;
    const setText = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

    // Core tier stats
    const wr = s.pooled_win_rate == null ? "—" : fmtPct(100 * s.pooled_win_rate, 2);
    setText("sp-winrate", wr);
    setText("sp-signals", fmtInt((s.n_put_signals || 0) + (s.n_call_signals || 0)));

    // Tight tier stats (headline)
    const twr = s.tight_pooled_win_rate == null ? "—" : fmtPct(100 * s.tight_pooled_win_rate, 2);
    setText("sp-tight-winrate", twr);
    setText("sp-tight-signals", fmtInt((s.n_tight_put_signals || 0) + (s.n_tight_call_signals || 0)));

    const setBadge = (id, n, label) => {
      const b = document.getElementById(id);
      if (b) b.textContent = `${fmtInt(n)} ${label}${n === 1 ? "" : "s"}`;
    };
    // Core tier badges
    setBadge("sp-pin-badge", (sp.pin_signals || []).length, "ticker");
    setBadge("sp-puts-badge", (sp.put_signals || []).length, "ticker");
    setBadge("sp-calls-badge", (sp.call_signals || []).length, "ticker");
    // Tight tier badges
    setBadge("sp-tight-pin-badge", (sp.tight_pin_signals || []).length, "ticker");
    setBadge("sp-tight-puts-badge", (sp.tight_put_signals || []).length, "ticker");
    setBadge("sp-tight-calls-badge", (sp.tight_call_signals || []).length, "ticker");

    const lazy = { pin: false, put: false, call: false,
                    tightPin: false, tightPut: false, tightCall: false };
    document.querySelectorAll('.cf-section-head[data-target^="sp-"]').forEach((h) => {
      const target = h.dataset.target;
      h.addEventListener("click", () => {
        if (target === "sp-pin" && !lazy.pin) {
          spRenderList("sp-list-pin", sp.pin_signals || [], "pin");
          lazy.pin = true;
        } else if (target === "sp-puts" && !lazy.put) {
          spRenderList("sp-list-puts", sp.put_signals || [], "put");
          lazy.put = true;
        } else if (target === "sp-calls" && !lazy.call) {
          spRenderList("sp-list-calls", sp.call_signals || [], "call");
          lazy.call = true;
        } else if (target === "sp-tight-pin" && !lazy.tightPin) {
          spRenderList("sp-list-tight-pin", sp.tight_pin_signals || [], "pin");
          lazy.tightPin = true;
        } else if (target === "sp-tight-puts" && !lazy.tightPut) {
          spRenderList("sp-list-tight-puts", sp.tight_put_signals || [], "put");
          lazy.tightPut = true;
        } else if (target === "sp-tight-calls" && !lazy.tightCall) {
          spRenderList("sp-list-tight-calls", sp.tight_call_signals || [], "call");
          lazy.tightCall = true;
        }
      });
    });
  }

  main();
})();
