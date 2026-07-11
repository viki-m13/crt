/* SPX Direction — render the daily signal from data/signal.json */
(function () {
  "use strict";
  var pct = function (x, d) { return (x >= 0 ? "+" : "") + (x * 100).toFixed(d == null ? 0 : d) + "%"; };
  var usd = function (x) { return "$" + Number(x).toFixed(2); };
  var el = function (t, c, h) { var e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  var signClass = function (x) { return x >= 0 ? "pos" : "neg"; };

  fetch("/spx/data/signal.json?_=" + Date.now())
    .then(function (r) { if (!r.ok) throw new Error("no data"); return r.json(); })
    .then(render)
    .catch(function () {
      document.getElementById("asof").textContent = "Signal data not yet published.";
      document.getElementById("cards").innerHTML = '<p class="muted">The nightly cron has not published a signal yet.</p>';
    });

  function render(d) {
    var reg = d.regime || {};
    var regTxt = reg.uptrend
      ? '<span class="pos">uptrend — active</span>'
      : '<span class="neg">below 200-day avg — standing aside</span>';
    document.getElementById("asof").innerHTML =
      "SPY <strong>" + usd(d.spot) + "</strong> · as of " + d.as_of +
      " · " + d.horizon_sessions + "-session horizon · regime: " + regTxt;

    renderCards(d);
    renderRecord(d);
    renderEquity(d);
    renderDrawdown(d);
    renderHist(d);
    renderExamples(d);
    renderTrades(d);
    window.addEventListener("resize", debounce(function () {
      renderEquity(d); renderDrawdown(d);
    }, 200));
  }

  function debounce(fn, ms) {
    var t; return function () { clearTimeout(t); t = setTimeout(fn, ms); };
  }

  function bookMeta(key) {
    return key === "call"
      ? { badge: "ror", tag: "Max ROR", verb: "Buy" }
      : { badge: "acc", tag: "Max accuracy", verb: "Sell" };
  }

  function renderCards(d) {
    var wrap = document.getElementById("cards");
    wrap.innerHTML = "";
    ["call", "put"].forEach(function (key) {
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

      // open position
      if (op) {
        var o = el("div", "openpos");
        o.appendChild(el("div", "muted", "Open position"));
        var okv = el("dl", "kv");
        var entryPx = (key === "call")
          ? "debit " + usd(op.entry_debit) : "credit " + usd(op.entry_credit);
        [
          ["Entered", op.entry_date + " @ " + usd(op.spot_at_entry)],
          ["Strikes", op.k1 + " / " + op.k2 + " (" + entryPx + ")"],
          ["Current ROR", '<span class="' + signClass(op.current_ror) + '">' + pct(op.current_ror) + "</span>"],
          ["Expiry", op.expiry_date + " (" + op.days_to_expiry + "d left)"]
        ].forEach(function (r) {
          okv.appendChild(el("dt", null, r[0]));
          okv.appendChild(el("dd", null, r[1]));
        });
        o.appendChild(okv);
        body.appendChild(o);
      }

      card.appendChild(body);
      wrap.appendChild(card);
    });
  }

  function renderRecord(d) {
    var tb = document.querySelector("#record-tbl tbody");
    tb.innerHTML = "";
    ["call", "put"].forEach(function (key) {
      var bk = d.books[key]; if (!bk) return;
      var tr = bk.track_record;
      var row = el("tr", key === "call" ? "hl" : "");
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
    var call = d.books.call && d.books.call.track_record.sizing;
    if (!call) return;
    var wrap = el("div", "tbl-wrap");
    wrap.style.marginTop = "18px";
    var t = el("table");
    t.innerHTML = "<caption>Per-trade ROR → portfolio CAGR is a sizing choice (call-spread book). " +
      "Aggressive sizing lifts return and drawdown together.</caption>" +
      "<thead><tr><th>Risk per trade</th><th>CAGR</th><th>Max drawdown</th></tr></thead>";
    var tb2 = el("tbody");
    call.forEach(function (s) {
      var row = el("tr");
      row.appendChild(el("th", null, Math.round(s.frac * 100) + "% of capital"));
      row.appendChild(el("td", null, s.cagr == null ? "—" : pct(s.cagr, 1)));
      row.appendChild(el("td", null, s.maxdd == null ? "—" : pct(s.maxdd, 0)));
      tb2.appendChild(row);
    });
    t.appendChild(tb2); wrap.appendChild(t); host.appendChild(wrap);
  }

  // ---- charts (self-contained Canvas, no libraries) ----
  var COL = { call: "#1a2ea1", put: "#064e2b", spy: "#b35900" };

  function setupCanvas(cv) {
    var dpr = window.devicePixelRatio || 1;
    var w = cv.clientWidth || cv.parentNode.clientWidth || 700;
    var h = cv.getAttribute("height") * 1;
    cv.width = w * dpr; cv.height = h * dpr;
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }
  var ts = function (s) { return Date.parse(s); };
  var fmtYear = function (t) { return new Date(t).getUTCFullYear(); };

  function renderEquity(d) {
    var cv = document.getElementById("eq-chart"); if (!cv) return;
    var s = setupCanvas(cv), ctx = s.ctx;
    var series = [
      { name: "SPY buy & hold", color: COL.spy, points: d.spy_benchmark.curve },
      { name: "Call spread book", color: COL.call, points: d.books.call.equity },
      { name: "Put spread book", color: COL.put, points: d.books.put.equity }
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
    // series lines
    series.forEach(function (se) {
      ctx.strokeStyle = se.color; ctx.lineWidth = se.name.indexOf("SPY") === 0 ? 1.3 : 2;
      ctx.beginPath();
      se.points.forEach(function (p, i) {
        var x = PX(ts(p[0])), y = PY(Math.max(p[1], 0.3));
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    });

    var lg = document.getElementById("eq-legend"); lg.innerHTML = "";
    series.forEach(function (se) {
      var sp = el("span", null, '<span class="swatch" style="background:' + se.color + '"></span>' + se.name);
      lg.appendChild(sp);
    });

    // equity stat strip
    var host = document.getElementById("eq-stats"); host.innerHTML = "";
    var refFrac = Math.round((d.equity_sizing || 0.15) * 100);
    [["Call book", d.books.call.track_record.sizing[1]],
     ["Put book", d.books.put.track_record.sizing[1]]].forEach(function (r) {
      host.appendChild(statBox(r[0] + " CAGR", pct(r[1].cagr, 1), r[0] + " · " + refFrac + "% sizing"));
      host.appendChild(statBox(r[0] + " max DD", pct(r[1].maxdd, 0), "worst peak-to-trough"));
    });
    host.appendChild(statBox("SPY CAGR", pct(d.spy_benchmark.cagr, 1), "buy & hold"));
    host.appendChild(statBox("SPY max DD", pct(d.spy_benchmark.maxdd, 0), "buy & hold"));
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
    var s = setupCanvas(cv), ctx = s.ctx;
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
    ["call", "put"].forEach(function (key) {
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
    var ex = d.books.call.examples || {};
    [["winner", "win", "Representative winner"], ["loser", "loss", "Worst loser"]].forEach(function (e) {
      var t = ex[e[0]]; if (!t) return;
      var c = el("div", "ex-card");
      c.appendChild(el("div", null, '<span class="tag ' + e[1] + '">' + e[2] + "</span>"));
      [["Entered", t.entry_date + " · SPY " + usd(t.spot_at_entry)],
       ["Strikes", t.k1 + " / " + t.k2 + " call spread"],
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

  function renderTrades(d) {
    var tb = document.querySelector("#trades-tbl tbody");
    tb.innerHTML = "";
    var all = [];
    ["call", "put"].forEach(function (key) {
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
