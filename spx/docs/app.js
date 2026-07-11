/* SPX Direction — render the daily signal from data/signal.json */
(function () {
  "use strict";
  var pct = function (x, d) { return (x >= 0 ? "+" : "") + (x * 100).toFixed(d == null ? 0 : d) + "%"; };
  var usd = function (x) { return "$" + Number(x).toFixed(2); };
  var el = function (t, c, h) { var e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  var signClass = function (x) { return x >= 0 ? "pos" : "neg"; };

  fetch("data/signal.json?_=" + Date.now())
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
    renderTrades(d);
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

  function renderTrades(d) {
    var tb = document.querySelector("#trades-tbl tbody");
    tb.innerHTML = "";
    var all = [];
    ["call", "put"].forEach(function (key) {
      var bk = d.books[key]; if (!bk) return;
      (bk.recent_trades || []).forEach(function (t) { all.push([key, t]); });
    });
    all.sort(function (a, b) { return a[1].exit_date < b[1].exit_date ? 1 : -1; });
    all.slice(0, 24).forEach(function (pair) {
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
