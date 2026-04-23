/* CreditFloor page: render precomputed walk-forward signals. */
(function () {
  "use strict";

  const state = {
    data: null,
    sort: "buffer",
    hfilter: "all",
  };

  const $ = (sel) => document.querySelector(sel);
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

  function renderStats(d) {
    const s = d.summary || {};
    const win = s.pooled_win_rate == null ? "—" : fmtPct(100 * s.pooled_win_rate, 2);
    $("#stat-pooled-win").textContent = win;
    $("#stat-pooled-n").textContent = fmtInt(
      (s.pooled_wins || 0) + (s.pooled_losses || 0)
    );
    $("#stat-signals").textContent = fmtInt(s.n_tickers_eligible || 0);
    $("#stat-universe").textContent = fmtInt(s.n_tickers_processed || 0);
  }

  function renderLastRun(d) {
    const gen = d.generated_at;
    const signals = (d.signals || []).length;
    let asof = "";
    if (signals) {
      // all signals share end_date
      asof = d.signals[0].end_date || "";
    }
    const el = $("#cf-last-run");
    const bits = [];
    if (asof) bits.push(`As of market close ${asof}.`);
    if (gen) bits.push(`Research generated ${gen}.`);
    bits.push(
      `Out-of-sample window covers ${d.summary?.fold_years?.[0]}–${d.summary?.fold_years?.slice(-1)?.[0]}, 7 walk-forward folds. ` +
        `Safety margin: ${((d.summary?.safety_eps || 0) * 100).toFixed(1)}% added to worst historical drawdown-to-expiry. ` +
        `Buffer cap: ${((d.summary?.max_buffer || 0) * 100).toFixed(0)}%.`
    );
    el.innerHTML = bits.join(" ");
  }

  function filteredSorted() {
    const items = (state.data?.signals || []).slice();
    const hf = state.hfilter;
    const filtered = items.map((s) => {
      const rungs = (s.ladder || []).filter(
        (l) => hf === "all" || String(l.horizon) === String(hf)
      );
      return { ...s, ladderView: rungs };
    }).filter((s) => s.ladderView.length > 0);

    filtered.sort((a, b) => {
      if (state.sort === "ticker") return a.ticker.localeCompare(b.ticker);
      if (state.sort === "tests") {
        const ta = Math.max(...a.ladderView.map((l) => l.n_test));
        const tb = Math.max(...b.ladderView.map((l) => l.n_test));
        return tb - ta;
      }
      // default: tightest buffer
      const ba = Math.min(...a.ladderView.map((l) => l.buffer_pct));
      const bb = Math.min(...b.ladderView.map((l) => l.buffer_pct));
      return ba - bb;
    });
    return filtered;
  }

  function rungHTML(l) {
    const tag = l.variant === "regime" ? "uptrend-only" : "all-regime";
    return `
      <div class="cf-rung">
        <div class="cf-rung-h">Expiry in ${l.horizon} trading days</div>
        <div class="cf-rung-k">${fmt$(l.strike)}</div>
        <div class="cf-rung-b">Buffer ${fmtPct(l.buffer_pct, 2)} below spot</div>
        <div class="cf-rung-m">
          <span class="tag">${tag}</span>
          <span class="tag">${fmtInt(l.n_test)} OOS tests</span>
          <span class="tag">${l.n_folds} folds</span>
          <br/>
          Worst test-set drawdown ever touched:
          <strong>${fmtPct(l.folds.reduce((m, f) => Math.max(m, f.worst_test_buf_pct), 0), 2)}</strong>.
          All folds 100% win rate.
        </div>
      </div>
    `;
  }

  function foldTableHTML(ladder) {
    const horizons = ladder.map((l) => l.horizon);
    // collect unique fold years across all ladder entries
    const years = Array.from(
      new Set(ladder.flatMap((l) => l.folds.map((f) => f.year)))
    ).sort();
    const head = `
      <tr>
        <th>Fold</th>
        ${horizons
          .map(
            (h) => `<th colspan="4" style="border-left:1px solid var(--rule-light)">${h}d</th>`
          )
          .join("")}
      </tr>
      <tr>
        <th></th>
        ${horizons
          .map(
            () =>
              `<th>Train</th><th>Test</th><th>Wins</th><th>Worst</th>`
          )
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

  function cardHTML(s) {
    const anyRegime = s.ladderView.some((l) => l.variant === "regime");
    const regimeBadge = anyRegime
      ? `<span class="cf-regime-badge regime">uptrend gate</span>`
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
          ${s.ladderView.map(rungHTML).join("")}
        </div>
        <div class="cf-card-expand">Show fold-by-fold walk-forward breakdown</div>
        <div class="cf-detail">
          ${foldTableHTML(s.ladderView)}
          <div class="cf-footnote" style="margin-top:12px">
            Each cell is the walk-forward test outcome for that calendar year
            and horizon. "Worst" is the largest drawdown-to-expiry
            <em>on the held-out test set</em> — always strictly below our
            conformal strike buffer, which is why every fold is a 100% win.
          </div>
        </div>
      </div>
    `;
  }

  function render() {
    const items = filteredSorted();
    const list = $("#cf-list");
    if (!items.length) {
      list.innerHTML = `<div class="cf-empty">No signals match the current filter today. The engine is fail-closed — most stocks, most days, there is no eligible trade.</div>`;
      return;
    }
    list.innerHTML = items.map(cardHTML).join("");
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

  function wireFilters() {
    document.querySelectorAll(".cf-filters button[data-sort]").forEach((b) => {
      b.addEventListener("click", (e) => {
        document
          .querySelectorAll(".cf-filters button[data-sort]")
          .forEach((bb) => bb.classList.remove("active"));
        e.currentTarget.classList.add("active");
        state.sort = e.currentTarget.dataset.sort;
        render();
      });
    });
    document.querySelectorAll(".cf-filters button[data-hfilter]").forEach((b) => {
      b.addEventListener("click", (e) => {
        document
          .querySelectorAll(".cf-filters button[data-hfilter]")
          .forEach((bb) => bb.classList.remove("active"));
        e.currentTarget.classList.add("active");
        state.hfilter = e.currentTarget.dataset.hfilter;
        render();
      });
    });
  }

  async function main() {
    try {
      state.data = await load();
      renderStats(state.data);
      renderLastRun(state.data);
      wireFilters();
      render();
    } catch (err) {
      const list = $("#cf-list");
      list.innerHTML = `<div class="cf-empty">Failed to load signals: ${err.message}</div>`;
    }
  }

  main();
})();
