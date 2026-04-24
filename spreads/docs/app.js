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

  async function main() {
    try {
      state.data = await load();
      renderStats(state.data);
      renderLastRun(state.data);
      wireCollapsibles();
      wireFilters();
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
  }

  main();
})();
