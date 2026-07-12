/* SPX Direction — render the daily signal from data/signal.json */
(function () {
  "use strict";
  var pct = function (x, d) { return (x >= 0 ? "+" : "") + (x * 100).toFixed(d == null ? 0 : d) + "%"; };
  var usd = function (x) { return "$" + Number(x).toFixed(2); };
  var el = function (t, c, h) { var e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  var signClass = function (x) { return x >= 0 ? "pos" : "neg"; };

  Promise.all([
    fetch("/spx/data/signal.json?_=" + Date.now())
      .then(function (r) { if (!r.ok) throw new Error("no data"); return r.json(); }),
    fetch("/spx/data/research.json?_=" + Date.now())
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; })
  ]).then(function (both) { render(both[0], both[1]); })
    .catch(function () {
      document.getElementById("asof").textContent = "Signal data not yet published.";
      document.getElementById("cards").innerHTML = '<p class="muted">The nightly cron has not published a signal yet.</p>';
    });

  function render(d, research) {
    var reg = d.regime || {};
    var regTxt = reg.uptrend
      ? '<span class="pos">uptrend — active</span>'
      : '<span class="neg">below 200-day avg — standing aside</span>';
    document.getElementById("asof").innerHTML =
      "SPY <strong>" + usd(d.spot) + "</strong> · as of " + d.as_of +
      " · pricing model v" + ((d.pricing_model || {}).version || 2) +
      " · regime: " + regTxt;

    renderCards(d);
    renderAnatomy(d);
    renderRecord(d);
    renderEquity(d);
    renderDrawdown(d);
    renderHist(d);
    renderExamples(d);
    renderTrades(d);
    if (research) {
      renderStressCharts(research);
      renderEdgeChart(research);
      renderAuditChart(research);
      renderIvChart(research);
      renderSensChart(research);
      renderEraTable(research);
    }
    var lastW = window.innerWidth;
    window.addEventListener("resize", debounce(function () {
      // Mobile browsers fire resize on scroll (URL bar hide/show) with an
      // unchanged width — ignore those; only re-render when width changes.
      if (window.innerWidth === lastW) return;
      lastW = window.innerWidth;
      renderEquity(d); renderDrawdown(d);
      if (research) { renderStressCharts(research); renderIvChart(research); }
    }, 200));
  }

  function debounce(fn, ms) {
    var t; return function () { clearTimeout(t); t = setTimeout(fn, ms); };
  }

  function bookMeta(key) {
    return key === "put"
      ? { badge: "acc", tag: "The strategy", verb: "Sell" }
      : { badge: "ror", tag: "Max ROR / trade", verb: "Buy" };
  }

  function renderCards(d) {
    var wrap = document.getElementById("cards");
    wrap.innerHTML = "";
    ["put", "call"].forEach(function (key) {
      var bk = d.books[key]; if (!bk) return;
      var meta = bookMeta(key), et = bk.enter_today, tr = bk.track_record, op = bk.open_position;
      var card = el("div", "card");

      var head = el("div", "card-head");
      head.appendChild(el("h3", null, bk.label.replace(/\s*\(.*\)/, "")));
      head.appendChild(el("span", "badge " + meta.badge, meta.tag));
      card.appendChild(head);

      var body = el("div", "card-body");

      // action
      var act = bk.today_action || "HOLD";
      var acls = /^ENTER/.test(act) ? "enter" : /^EXIT/.test(act) ? "exit"
        : /^STAND/.test(act) ? "exit" : "";
      body.appendChild(el("div", "action " + acls, act));

      // the trade to enter today
      var tl = el("div", "trade-line");
      if (key === "call") {
        tl.innerHTML = '<span class="big">' + meta.verb + " the " + et.expiry_date +
          " " + et.long_strike + " / " + et.short_strike + " call spread</span><br>" +
          "debit " + usd(et.est_debit) + " &nbsp;·&nbsp; max ROR " + pct(et.max_ror) +
          " &nbsp;·&nbsp; breakeven " + usd(et.breakeven);
      } else {
        tl.innerHTML = '<span class="big">' + meta.verb + " the " + et.expiry_date +
          " " + et.sell_strike + " / " + et.buy_strike + " put spread</span><br>" +
          "credit " + usd(et.est_credit) + " &nbsp;·&nbsp; max ROR " + pct(et.max_ror) +
          " &nbsp;·&nbsp; " + et.profit_target;
      }
      body.appendChild(tl);

      // key stats
      var kv = el("dl", "kv");
      var rows = [
        ["Win rate", pct(tr.win_rate, 1) + " (" + pct(tr.win_rate_val, 0).replace("+", "") + " out-of-sample)"],
        ["Mean ROR / trade", pct(tr.mean_ror) + " (median " + pct(tr.median_ror) + ")"],
        ["Avg hold", Math.round(tr.avg_hold_days) + " days"],
        ["Annualized ROR", pct(tr.annualized_ror)]
      ];
      rows.forEach(function (r) {
        kv.appendChild(el("dt", null, r[0]));
        kv.appendChild(el("dd", null, r[1]));
      });
      body.appendChild(kv);

      // open rungs (ladder)
      var ops = bk.open_positions || [];
      if (ops.length) {
        var o = el("div", "openpos");
        var cad = (bk.spec && bk.spec.every <= 5) ? "week" : "month";
        o.appendChild(el("div", "muted", "Open rungs (" + ops.length + ") — one opened each " + cad));
        var tbl = el("table", "rungs");
        tbl.innerHTML = "<thead><tr><th>Entered</th><th>Strikes</th><th>ROR</th><th>Expiry</th></tr></thead>";
        var tb = el("tbody");
        ops.slice().reverse().forEach(function (op) {
          var row = el("tr");
          [op.entry_date,
           op.k1 + " / " + op.k2,
           '<span class="' + signClass(op.current_ror) + '">' + pct(op.current_ror) + "</span>",
           op.days_to_expiry + "d"].forEach(function (v, i) {
            row.appendChild(el("td", null, v));
          });
          tb.appendChild(row);
        });
        tbl.appendChild(tb);
        o.appendChild(tbl);
        body.appendChild(o);
      }

      card.appendChild(body);
      wrap.appendChild(card);
    });
  }

  function renderRecord(d) {
    var tb = document.querySelector("#record-tbl tbody");
    tb.innerHTML = "";
    ["put", "call"].forEach(function (key) {
      var bk = d.books[key]; if (!bk) return;
      var tr = bk.track_record;
      var row = el("tr", key === "put" ? "hl" : "");
      [
        bk.label, pct(tr.win_rate, 1), pct(tr.win_rate_val, 0).replace("+", ""),
        pct(tr.mean_ror), pct(tr.median_ror), Math.round(tr.avg_hold_days) + "d",
        pct(tr.annualized_ror), pct(tr.worst_ror)
      ].forEach(function (v, i) { row.appendChild(el(i === 0 ? "th" : "td", null, v)); });
      tb.appendChild(row);
    });

    // sizing → CAGR table
    var host = document.getElementById("sizing");
    host.innerHTML = "";
    var call = d.books.put && d.books.put.track_record.sizing;
    if (!call) return;
    var wrap = el("div", "tbl-wrap");
    wrap.style.marginTop = "18px";
    var t = el("table");
    t.innerHTML = "<caption>Rung size → portfolio CAGR is a sizing choice (put-spread ladder; " +
      "total at-risk cap scales with rung size). Aggressive sizing lifts return and drawdown together.</caption>" +
      "<thead><tr><th>Risk per rung</th><th>CAGR</th><th>Max drawdown</th></tr></thead>";
    var tb2 = el("tbody");
    call.forEach(function (s) {
      var row = el("tr");
      row.appendChild(el("th", null, Math.round(s.frac * 100) + "% of equity"));
      row.appendChild(el("td", null, s.cagr == null ? "—" : pct(s.cagr, 1)));
      row.appendChild(el("td", null, s.maxdd == null ? "—" : pct(s.maxdd, 0)));
      tb2.appendChild(row);
    });
    t.appendChild(tb2); wrap.appendChild(t); host.appendChild(wrap);
  }

  // ---- charts (self-contained Canvas, no libraries) ----
  var COL = { call: "#7f8fb8", put: "#7aa98c", spy: "#d0a36a", strat: "#0b1f5e" };

  function setupCanvas(cv, hLogical) {
    // hLogical is passed explicitly — NEVER read back from the height
    // attribute, because setting cv.height rewrites it (dpr-multiplied),
    // which would compound on every resize and squash the chart.
    var dpr = window.devicePixelRatio || 1;
    var w = cv.clientWidth || cv.parentNode.clientWidth || 700;
    cv.style.height = hLogical + "px";       // pin CSS display height
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(hLogical * dpr);  // drawing buffer
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: hLogical };
  }
  var ts = function (s) { return Date.parse(s); };
  var fmtYear = function (t) { return new Date(t).getUTCFullYear(); };

  function renderEquity(d) {
    var cv = document.getElementById("eq-chart"); if (!cv) return;
    var s = setupCanvas(cv, 380), ctx = s.ctx;
    var series = [
      { name: "SPY buy & hold", color: COL.spy, points: d.spy_benchmark.curve, w: 1.2 },
      { name: "Call ladder (max ROR/trade)", color: COL.call, points: d.books.call.equity, w: 1.2 },
      { name: "Strategy — weekly put-spread ladder", color: COL.strat,
        points: (d.strategy_equity || {}).curve, w: 3 }
    ].filter(function (x) { return x.points && x.points.length; });

    var xs = [], vals = [];
    series.forEach(function (se) { se.points.forEach(function (p) { xs.push(ts(p[0])); vals.push(p[1]); }); });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var vmin = Math.min.apply(null, vals), vmax = Math.max.apply(null, vals);
    var ly0 = Math.log(Math.max(vmin, 0.3)), ly1 = Math.log(vmax * 1.05);
    var padL = 46, padR = 10, padT = 10, padB = 24;
    var PX = function (t) { return padL + (t - x0) / (x1 - x0) * (s.w - padL - padR); };
    var PY = function (v) { return padT + (ly1 - Math.log(v)) / (ly1 - ly0) * (s.h - padT - padB); };

    // y gridlines (log: 1,2,5,10,20,50,100…)
    ctx.font = "11px 'IBM Plex Mono',monospace"; ctx.textBaseline = "middle";
    [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].forEach(function (g) {
      if (g < Math.exp(ly0) || g > Math.exp(ly1)) return;
      var y = PY(g);
      ctx.strokeStyle = "#eee"; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(s.w - padR, y); ctx.stroke();
      ctx.fillStyle = "#999"; ctx.textAlign = "right"; ctx.fillText(g + "×", padL - 6, y);
    });
    // x year ticks
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (var yr = fmtYear(x0) + 1; yr <= fmtYear(x1); yr += 5) {
      var t = Date.UTC(yr, 0, 1); if (t < x0 || t > x1) continue;
      var x = PX(t);
      ctx.strokeStyle = "#f3f3f3"; ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, s.h - padB); ctx.stroke();
      ctx.fillStyle = "#999"; ctx.fillText(yr, x, s.h - padB + 5);
    }
    // series lines — draw thin references first, bold strategy last (on top)
    ctx.lineJoin = "round";
    series.forEach(function (se) {
      ctx.strokeStyle = se.color; ctx.lineWidth = se.w || 1.5;
      ctx.beginPath();
      se.points.forEach(function (p, i) {
        var x = PX(ts(p[0])), y = PY(Math.max(p[1], 0.3));
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    });

    // legend: strategy first (bold swatch), then references
    var lg = document.getElementById("eq-legend"); lg.innerHTML = "";
    series.slice().reverse().forEach(function (se) {
      var thick = (se.w || 1.5) >= 3 ? "height:4px;" : "";
      var sp = el("span", null, '<span class="swatch" style="' + thick + 'background:' + se.color + '"></span>' + se.name);
      lg.appendChild(sp);
    });

    // equity stat strip — lead with the strategy (the call ladder)
    var host = document.getElementById("eq-stats"); host.innerHTML = "";
    var refFrac = Math.round((d.equity_sizing || 0.05) * 100);
    var capPct = Math.round((d.ladder_cap || 0.30) * 100);
    var strat = d.strategy_equity;
    if (strat) {
      host.appendChild(statBox("Strategy CAGR", pct(strat.cagr, 1),
        refFrac + "%/rung · " + capPct + "% cap"));
      host.appendChild(statBox("Strategy max DD", pct(strat.maxdd, 0), "worst peak-to-trough"));
    }
    host.appendChild(statBox("SPY CAGR", pct(d.spy_benchmark.cagr, 1), "buy & hold"));
    host.appendChild(statBox("SPY max DD", pct(d.spy_benchmark.maxdd, 0), "buy & hold"));
    var cm = d.books.call.equity_metrics;
    if (cm && cm.cagr != null) {
      host.appendChild(statBox("Call ladder CAGR", pct(cm.cagr, 1), "max ROR/trade book"));
      host.appendChild(statBox("Call ladder max DD", pct(cm.maxdd, 0), "max ROR/trade book"));
    }
  }

  function statBox(num, val, lab) {
    var b = el("div", "stat");
    b.appendChild(el("div", "num", val));
    b.appendChild(el("div", "lab", "<strong>" + num + "</strong> — " + lab));
    return b;
  }

  function ddSeries(points) {
    var peak = -Infinity, out = [];
    points.forEach(function (p) { peak = Math.max(peak, p[1]); out.push([p[0], p[1] / peak - 1]); });
    return out;
  }

  function renderDrawdown(d) {
    var cv = document.getElementById("dd-chart"); if (!cv) return;
    var s = setupCanvas(cv, 240), ctx = s.ctx;
    var series = [
      { name: "call", color: COL.call, points: ddSeries(d.books.call.equity) },
      { name: "put", color: COL.put, points: ddSeries(d.books.put.equity) }
    ];
    var xs = [];
    series.forEach(function (se) { se.points.forEach(function (p) { xs.push(ts(p[0])); }); });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var dmin = -0.45;
    var padL = 46, padR = 10, padT = 8, padB = 22;
    var PX = function (t) { return padL + (t - x0) / (x1 - x0) * (s.w - padL - padR); };
    var PY = function (v) { return padT + (0 - v) / (0 - dmin) * (s.h - padT - padB); };
    ctx.font = "11px 'IBM Plex Mono',monospace"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    [0, -0.1, -0.2, -0.3, -0.4].forEach(function (g) {
      var y = PY(g);
      ctx.strokeStyle = "#eee"; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(s.w - padR, y); ctx.stroke();
      ctx.fillStyle = "#999"; ctx.fillText(Math.round(g * 100) + "%", padL - 6, y);
    });
    series.forEach(function (se) {
      ctx.fillStyle = se.color + "22"; ctx.strokeStyle = se.color; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(PX(ts(se.points[0][0])), PY(0));
      se.points.forEach(function (p) { ctx.lineTo(PX(ts(p[0])), PY(Math.max(p[1], dmin))); });
      ctx.lineTo(PX(ts(se.points[se.points.length - 1][0])), PY(0)); ctx.closePath(); ctx.fill();
      ctx.beginPath();
      se.points.forEach(function (p, i) {
        var x = PX(ts(p[0])), y = PY(Math.max(p[1], dmin)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    });
  }

  function renderHist(d) {
    var host = document.getElementById("hist"); if (!host) return;
    host.innerHTML = "";
    ["put", "call"].forEach(function (key) {
      var bk = d.books[key]; if (!bk || !bk.ror_histogram) return;
      host.appendChild(el("div", "muted", "<strong>" + (key === "call" ? "Call spread" : "Put spread") + "</strong>"));
      var max = Math.max.apply(null, bk.ror_histogram.map(function (h) { return h.count; })) || 1;
      bk.ror_histogram.forEach(function (h) {
        var row = el("div", "hist-row");
        row.appendChild(el("div", "lab", h.label));
        var bwrap = el("div");
        var bar = el("div", "hist-bar" + (h.label.indexOf("−") === 0 && h.label !== "−50–0%" ? " neg" : ""));
        if (/≤−50/.test(h.label) || /−50–0/.test(h.label)) bar.className = "hist-bar neg";
        bar.style.width = Math.round(h.count / max * 100) + "%";
        bwrap.appendChild(bar);
        row.appendChild(bwrap);
        row.appendChild(el("div", "ct", h.count));
        host.appendChild(row);
      });
    });
  }

  function renderExamples(d) {
    var host = document.getElementById("examples"); if (!host) return;
    host.innerHTML = "";
    var ex = d.books.put.examples || {};
    [["winner", "win", "Representative winner"], ["loser", "loss", "Worst loser"]].forEach(function (e) {
      var t = ex[e[0]]; if (!t) return;
      var c = el("div", "ex-card");
      c.appendChild(el("div", null, '<span class="tag ' + e[1] + '">' + e[2] + "</span>"));
      [["Entered", t.entry_date + " · SPY " + usd(t.spot_at_entry)],
       ["Strikes", t.k1 + " / " + t.k2 + " spread"],
       ["Exited", t.exit_date + " · " + t.reason.replace("gtc-", "").replace("-", " ")],
       ["Held", t.hold_days + " days"],
       ["Result", '<span class="' + signClass(t.ror) + '">' + pct(t.ror) + " ROR</span>"]].forEach(function (r) {
        var row = el("div", "row");
        row.appendChild(el("span", "k", r[0])); row.appendChild(el("span", null, r[1]));
        c.appendChild(row);
      });
      host.appendChild(c);
    });
  }

  // ---- research documentation charts ----

  function renderAnatomy(d) {
    var host = document.getElementById("anatomy"); if (!host) return;
    var et = d.books.put && d.books.put.enter_today; if (!et) return;
    host.innerHTML = "";
    var perRisk = (et.width - et.est_credit) * 100;   // $ at risk per contract
    var perCredit = et.est_credit * 100;
    var sizing = d.equity_sizing || 0.03;
    // one contract IS the rung; the account size that makes it the
    // reference 3% rung follows from the per-contract risk
    var acct = Math.round(perRisk / sizing / 1000) * 1000;
    host.appendChild(el("div", null,
      "<strong>Sell 1× the " + et.expiry_date + " " + et.sell_strike +
      " / " + et.buy_strike + " put spread</strong> (SPY at " + usd(et.spot) + ")"));
    var flow = el("div", "flow");
    [["$" + acct.toLocaleString(), "account for which one contract = the " +
      Math.round(sizing * 100) + "% reference rung"],
     [usd(perCredit), "credit collected up front"],
     [usd(perRisk), "maximum possible loss (defined)"],
     [pct(et.max_ror), "return on risk if SPY holds above " + et.sell_strike],
     ["~13 wks", "held to the " + et.expiry_date + " expiry"]]
      .forEach(function (x) {
        var st = el("div", "step");
        st.appendChild(el("div", "v", x[0]));
        st.appendChild(el("div", "k", x[1]));
        flow.appendChild(st);
      });
    host.appendChild(flow);
    host.appendChild(el("div", "outcomes",
      "<strong>At expiry:</strong> SPY above " + et.sell_strike + " (−3%) → keep the full " +
      usd(perCredit) + " <span class='pos'>(" + pct(et.max_ror) + " on risk)</span>. " +
      "Between the strikes → partial loss. Below " + et.buy_strike + " (−6%) → lose " +
      usd(perRisk) + ", never more <span class='neg'>(−100% of risk, ~" +
      Math.round(sizing * 100) + "% of the account)</span>. " +
      "Historically the full credit is kept ~88% of the time. Smaller accounts: trade the " +
      "same structure less often, or accept a larger fraction at risk per rung."));
  }

  function miniLine(canvas, seriesArr, fmtY) {
    var s = setupCanvas(canvas, 180), ctx = s.ctx;
    var xs = [], ys = [];
    seriesArr.forEach(function (se) {
      se.points.forEach(function (p) { xs.push(ts(p[0])); ys.push(p[1]); });
    });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var pad = (y1 - y0) * 0.06 || 0.1; y0 -= pad; y1 += pad;
    var padL = 40, padR = 6, padT = 6, padB = 18;
    var PX = function (t) { return padL + (t - x0) / (x1 - x0) * (s.w - padL - padR); };
    var PY = function (v) { return padT + (y1 - v) / (y1 - y0) * (s.h - padT - padB); };
    ctx.font = "10px 'IBM Plex Mono',monospace"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    for (var g = 0; g <= 3; g++) {
      var v = y0 + (y1 - y0) * g / 3, y = PY(v);
      ctx.strokeStyle = "#eee"; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(s.w - padR, y); ctx.stroke();
      ctx.fillStyle = "#999"; ctx.fillText(fmtY(v), padL - 5, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    [x0, (x0 + x1) / 2, x1].forEach(function (t) {
      ctx.fillStyle = "#999"; ctx.fillText(fmtYear(t), PX(t), s.h - padB + 4);
    });
    ctx.lineJoin = "round";
    seriesArr.forEach(function (se) {
      ctx.strokeStyle = se.color; ctx.lineWidth = se.w || 1.5;
      ctx.beginPath();
      se.points.forEach(function (p, i) {
        var x = PX(ts(p[0])), y = PY(p[1]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    });
  }

  function renderStressCharts(r) {
    var host = document.getElementById("stress-charts"); if (!host || !r.stress) return;
    host.innerHTML = "";
    r.stress.forEach(function (sc) {
      var card = el("div", "mini-chart");
      card.appendChild(el("div", "mc-title", sc.name));
      var cv = document.createElement("canvas");
      cv.setAttribute("height", "180");
      card.appendChild(cv);
      var sub = el("div", "mc-sub",
        '<span class="' + signClass(sc.strategy_final) + '">strategy ' + pct(sc.strategy_final, 1) + "</span>" +
        ' · <span class="' + signClass(sc.spy_final) + '">SPY ' + pct(sc.spy_final, 1) + "</span>");
      card.appendChild(sub);
      host.appendChild(card);
      miniLine(cv, [
        { color: COL.spy, points: sc.spy, w: 1.2 },
        { color: COL.strat, points: sc.strategy, w: 2.2 }
      ], function (v) { return v.toFixed(1) + "×"; });
    });
  }

  function renderEdgeChart(r) {
    var host = document.getElementById("edge-chart"); if (!host || !r.edge_by_horizon) return;
    host.innerHTML = "";
    r.edge_by_horizon.forEach(function (e) {
      var row = el("div", "pair-row");
      row.appendChild(el("div", "lab", e.horizon + "d"));
      var bars = el("div", "pair-bars");
      var a = el("div", "pb actual"); a.style.width = (e.actual * 100) + "%";
      a.innerHTML = "<span>actual " + Math.round(e.actual * 100) + "%</span>";
      var b = el("div", "pb implied"); b.style.width = (e.implied * 100) + "%";
      b.innerHTML = "<span>market " + Math.round(e.implied * 100) + "%</span>";
      bars.appendChild(a); bars.appendChild(b);
      row.appendChild(bars);
      host.appendChild(row);
    });
  }

  function hbars(hostId, rows, cls) {
    var host = document.getElementById(hostId); if (!host || !rows) return;
    host.innerHTML = "";
    var max = Math.max.apply(null, rows.map(function (x) { return x.cagr; }));
    rows.forEach(function (x, i) {
      var row = el("div", "hbar-row");
      row.appendChild(el("div", "lab", x.label));
      var wrap = el("div");
      var bar = el("div", "hbar " + (cls ? cls(x, i, rows) : ""));
      bar.style.width = Math.round(x.cagr / max * 100) + "%";
      wrap.appendChild(bar);
      row.appendChild(wrap);
      row.appendChild(el("div", "val", pct(x.cagr, 1)));
      host.appendChild(row);
    });
  }

  function renderAuditChart(r) {
    hbars("audit-chart", r.audit, function (x, i, rows) {
      return i === 0 ? "warn" : (i === rows.length - 1 ? "" : "dim");
    });
  }

  function renderSensChart(r) {
    hbars("sens-chart", r.sensitivity, function (x, i) { return i === 0 ? "" : "dim"; });
  }

  function renderIvChart(r) {
    var cv = document.getElementById("iv-chart"); if (!cv || !r.iv_series) return;
    var s = setupCanvas(cv, 230), ctx = s.ctx;
    var v2 = r.iv_series.map(function (p) { return [p[0], p[1] * 100]; });
    var v1 = r.iv_series.map(function (p) { return [p[0], p[2] * 100]; });
    miniLineOn(s, ctx, [
      { color: "#c9a0a0", points: v1, w: 1.1 },
      { color: COL.strat, points: v2, w: 2 }
    ], function (v) { return Math.round(v) + "%"; });
    var lg = document.getElementById("iv-legend");
    if (lg) lg.innerHTML =
      '<span><span class="swatch" style="background:#0b1f5e"></span>v2 audited 1y ATM IV</span>' +
      '<span><span class="swatch" style="background:#c9a0a0"></span>v1 naive (1.12 × realized)</span>';
  }

  function miniLineOn(s, ctx, seriesArr, fmtY) {
    var xs = [], ys = [];
    seriesArr.forEach(function (se) {
      se.points.forEach(function (p) { xs.push(ts(p[0])); ys.push(p[1]); });
    });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = 0, y1 = Math.max.apply(null, ys) * 1.06;
    var padL = 40, padR = 6, padT = 6, padB = 18;
    var PX = function (t) { return padL + (t - x0) / (x1 - x0) * (s.w - padL - padR); };
    var PY = function (v) { return padT + (y1 - v) / (y1 - y0) * (s.h - padT - padB); };
    ctx.font = "10px 'IBM Plex Mono',monospace"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    for (var g = 0; g <= 4; g++) {
      var v = y1 * g / 4, y = PY(v);
      ctx.strokeStyle = "#eee"; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(s.w - padR, y); ctx.stroke();
      ctx.fillStyle = "#999"; ctx.fillText(fmtY(v), padL - 5, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (var yr = fmtYear(x0) + 2; yr <= fmtYear(x1); yr += 5) {
      var t = Date.UTC(yr, 0, 1);
      ctx.fillStyle = "#999"; ctx.fillText(yr, PX(t), s.h - padB + 4);
    }
    ctx.lineJoin = "round";
    seriesArr.forEach(function (se) {
      ctx.strokeStyle = se.color; ctx.lineWidth = se.w || 1.5;
      ctx.beginPath();
      se.points.forEach(function (p, i) {
        var x = PX(ts(p[0])), y = PY(p[1]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    });
  }

  function renderEraTable(r) {
    var tb = document.querySelector("#era-tbl tbody"); if (!tb || !r.era_examples) return;
    tb.innerHTML = "";
    r.era_examples.forEach(function (t) {
      var row = el("tr");
      var credit = t.credit != null ? "$" + Math.round(t.credit * 100) : "—";
      var risk = t.risk != null ? "$" + Math.round(t.risk * 100) : "—";
      [t.era + (t.kind === "loss" ? ' <span class="neg">✕</span>' : ""),
       t.entry_date, t.k1 + " / " + t.k2, credit, risk,
       t.exit_date + " · " + t.reason.replace("gtc-", ""),
       t.hold_days + "d",
       '<span class="' + signClass(t.ror) + '">' + pct(t.ror) + "</span>"]
        .forEach(function (v, i) { row.appendChild(el(i === 0 ? "th" : "td", null, v)); });
      tb.appendChild(row);
    });
  }

  function renderTrades(d) {
    var tb = document.querySelector("#trades-tbl tbody");
    tb.innerHTML = "";
    var all = [];
    ["put", "call"].forEach(function (key) {
      var bk = d.books[key]; if (!bk) return;
      (bk.recent_trades || []).forEach(function (t) { all.push([key, t]); });
    });
    all.sort(function (a, b) { return a[1].exit_date < b[1].exit_date ? 1 : -1; });
    all.slice(0, 100).forEach(function (pair) {
      var key = pair[0], t = pair[1];
      var row = el("tr");
      [
        key === "call" ? "Call spread" : "Put spread",
        t.entry_date, t.exit_date, t.k1 + " / " + t.k2,
        t.hold_days + "d", t.reason.replace("gtc-", "").replace("-", " "),
        '<span class="' + signClass(t.ror) + '">' + pct(t.ror) + "</span>"
      ].forEach(function (v, i) { row.appendChild(el(i === 0 ? "th" : "td", null, v)); });
      tb.appendChild(row);
    });
  }
})();
