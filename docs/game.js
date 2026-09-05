/* Signal — client for the stock matching game.
 *
 * The server is stateless, so this file owns the whole session: it carries
 * the answer history, replays it on every request, and the server rebuilds
 * (and re-validates) the profile from it. Nothing is persisted anywhere.
 *
 * Latency is the product here. A model call between questions is ~0.3-1.5s,
 * which is long enough to feel broken if the screen just sits there, so the
 * selection animation runs in parallel with the request and the next
 * question waits for BOTH — the pause reads as feedback rather than lag.
 */
(function () {
  "use strict";

  var API = "/api/quiz";
  var MIN_BEAT = 420;          // selection animation floor, ms
  var state = { answers: [], busy: false, lastQuestion: null, result: null };

  var $ = function (id) { return document.getElementById(id); };

  var screens = ["Intro", "Q", "Wait", "Result", "Error"];
  function show(name) {
    screens.forEach(function (s) {
      var el = $("screen" + s);
      if (el) el.classList.toggle("show", s === name);
    });
    // the footer competes for space on a phone while a question is up
    document.body.classList.toggle("playing", name === "Q" || name === "Wait");
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  }

  function buzz(ms) {
    try { if (navigator.vibrate) navigator.vibrate(ms || 8); } catch (e) {}
  }

  function post(body) {
    return fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  // --------------------------------------------------------------- progress
  function setLock(p) {
    var pct = Math.max(0, Math.min(100, Math.round((p || 0) * 100)));
    $("lockWrap").hidden = false;
    var fill = $("lockFill");
    fill.style.width = pct + "%";
    fill.classList.toggle("hot", pct >= 70);
    $("lockLabel").textContent = pct + "% locked";
  }

  // --------------------------------------------------------------- question
  function renderQuestion(data) {
    var q = data.question;
    state.lastQuestion = q;
    $("qCount").textContent = "Question " + (state.answers.length + 1);
    $("qText").textContent = q.text;
    var sub = $("qSub");
    if (q.subtext) { sub.textContent = q.subtext; sub.hidden = false; }
    else { sub.hidden = true; }

    var wrap = $("options");
    wrap.innerHTML = "";
    (q.options || []).forEach(function (opt, i) {
      var b = document.createElement("button");
      b.className = "opt";
      b.type = "button";
      b.style.animationDelay = (i * 45) + "ms";
      b.setAttribute("data-i", String(i));

      if (opt.emoji) {
        var e = document.createElement("span");
        e.className = "opt-emoji";
        e.setAttribute("aria-hidden", "true");
        e.textContent = opt.emoji;
        b.appendChild(e);
      }
      var label = document.createElement("span");
      label.className = "opt-label";
      label.textContent = opt.label;
      b.appendChild(label);

      var key = document.createElement("span");
      key.className = "opt-key";
      key.textContent = String(i + 1);
      b.appendChild(key);

      b.addEventListener("click", function () { choose(i); });
      wrap.appendChild(b);
    });

    setLock(data.progress);
    show("Q");
    // move focus for keyboard and screen-reader users without stealing it
    // from a mouse user mid-tap
    var first = wrap.querySelector(".opt");
    if (first && document.activeElement === document.body) first.focus();
  }

  function choose(i) {
    if (state.busy) return;
    var q = state.lastQuestion;
    if (!q || !q.options || !q.options[i]) return;
    state.busy = true;
    buzz(9);

    var opt = q.options[i];
    var buttons = $("options").querySelectorAll(".opt");
    Array.prototype.forEach.call(buttons, function (b, j) {
      b.disabled = true;
      b.classList.add(j === i ? "chosen" : "dimmed");
    });

    state.answers.push({
      id: q.id, q: q.text, a: opt.label, effects: opt.effects
    });

    var beat = new Promise(function (res) { setTimeout(res, MIN_BEAT); });
    var call = post({ action: "next", answers: state.answers });

    Promise.all([call, beat]).then(function (r) {
      state.busy = false;
      var data = r[0];
      if (data.done) { reveal(); return; }
      renderQuestion(data);
    }).catch(function (err) {
      state.busy = false;
      // A failed turn must not lose the player's progress: drop the answer
      // we optimistically recorded so a retry re-asks the same question.
      state.answers.pop();
      fail("We lost the connection mid-question.", err);
    });
  }

  // ----------------------------------------------------------------- reveal
  var WAIT_LINES = [
    ["Reading your answers…", "Working out what you actually meant"],
    ["Scanning listings…", "New York, London, Tokyo, Mumbai, Paris"],
    ["Checking live prices…", "Every candidate is verified against real data"],
    ["Measuring the ride…", "Volatility, drawdown, momentum"],
    ["Locking your match…", "Almost there"]
  ];

  function reveal() {
    show("Wait");
    setLock(1);
    var i = 0;
    $("waitText").textContent = WAIT_LINES[0][0];
    $("waitSub").textContent = WAIT_LINES[0][1];
    var timer = setInterval(function () {
      i = (i + 1) % WAIT_LINES.length;
      $("waitText").textContent = WAIT_LINES[i][0];
      $("waitSub").textContent = WAIT_LINES[i][1];
    }, 1900);

    post({ action: "pick", answers: state.answers }).then(function (data) {
      clearInterval(timer);
      if (!data.ok) { failPick(data); return; }
      state.result = data;
      renderResult(data);
    }).catch(function (err) {
      clearInterval(timer);
      fail("We couldn't reach the market data service.", err);
    });
  }

  function fmtPrice(v, cur) {
    if (v === null || v === undefined) return "—";
    var digits = v >= 1000 ? 0 : (v >= 10 ? 2 : 3);
    try {
      return new Intl.NumberFormat(undefined, {
        minimumFractionDigits: digits, maximumFractionDigits: digits
      }).format(v);
    } catch (e) { return String(v); }
  }
  function pct(v, signed) {
    if (v === null || v === undefined) return "—";
    var s = Math.round(v * 100);
    return (signed && s > 0 ? "+" : "") + s + "%";
  }

  function metric(k, v, cls, note) {
    var d = document.createElement("div");
    d.className = "metric";
    var kk = document.createElement("span");
    kk.className = "metric-k"; kk.textContent = k;
    var vv = document.createElement("span");
    vv.className = "metric-v" + (cls ? " " + cls : ""); vv.textContent = v;
    d.appendChild(kk); d.appendChild(vv);
    if (note) {
      var n = document.createElement("span");
      n.className = "metric-note"; n.textContent = note;
      d.appendChild(n);
    }
    return d;
  }

  function renderResult(data) {
    var p = data.pick, c = data.copy || {};

    $("archName").textContent = data.archetype ? data.archetype.name : "Your match";
    $("archBlurb").textContent = data.archetype ? data.archetype.blurb : "";

    $("pickName").textContent = (p.name || p.symbol);
    var bits = [p.symbol];
    if (p.exchange) bits.push(p.exchange);
    if (p.sector) bits.push(p.sector);
    $("pickMeta").textContent = bits.join(" · ");
    $("pickPrice").textContent = fmtPrice(p.price, p.currency);
    $("pickCur").textContent = p.currency || "";

    $("pickHeadline").textContent = c.headline || ("You matched with " + (p.name || p.symbol));

    var why = $("whyList");
    why.innerHTML = "";
    (c.why || []).forEach(function (w) {
      var li = document.createElement("li");
      li.textContent = w;
      why.appendChild(li);
    });

    var m = $("metrics");
    m.innerHTML = "";
    var tv = data.profile && data.profile.target_vol;
    if (p.ann_vol !== undefined && p.ann_vol !== null) {
      m.appendChild(metric("Volatility", pct(p.ann_vol), "",
        tv ? "you asked for ~" + pct(tv) : ""));
    }
    if (p.max_dd !== undefined && p.max_dd !== null) {
      m.appendChild(metric("Worst fall", pct(p.max_dd), "neg", "peak to trough, 2y"));
    }
    if (p.mom_12m !== undefined && p.mom_12m !== null) {
      m.appendChild(metric("12 months", pct(p.mom_12m, true),
        p.mom_12m >= 0 ? "pos" : "neg", "price change"));
    }
    if (p.dividend_yield) {
      m.appendChild(metric("Dividend", pct(p.dividend_yield), "", "trailing yield"));
    } else if (p.pos_52w !== undefined && p.pos_52w !== null) {
      m.appendChild(metric("52-week range", Math.round(p.pos_52w * 100) + "%",
        "", "0 = low, 100 = high"));
    }

    $("watchOut").textContent = c.watch_out || "";
    $("watchOut").hidden = !c.watch_out;

    var rw = $("runnersWrap"), rl = $("runnerList");
    rl.innerHTML = "";
    if (data.runners_up && data.runners_up.length) {
      data.runners_up.forEach(function (r) {
        var row = document.createElement("div");
        row.className = "runner";
        var left = document.createElement("b");
        left.textContent = (r.name || r.symbol) + " · " + r.symbol;
        var right = document.createElement("span");
        right.textContent = r.ann_vol !== null && r.ann_vol !== undefined
          ? pct(r.ann_vol) + " volatility" : "";
        row.appendChild(left); row.appendChild(right);
        rl.appendChild(row);
      });
      rw.hidden = false;
    } else { rw.hidden = true; }

    renderReason(data);
    show("Result");
    buzz([12, 40, 18]);
  }

  function renderReason(data) {
    var body = $("reasonBody");
    body.className = "detail-body";
    body.innerHTML = "";
    var d = data.diagnostics || {};
    var prof = (data.profile && data.profile.dims) || {};

    var p = document.createElement("p");
    p.textContent =
      "Your answers were turned into a preference profile. Candidates were " +
      "proposed, then each one was independently verified against real price " +
      "history — anything that could not be verified was discarded. A scoring " +
      "function ranked the survivors; the write-up above only describes that " +
      "result, it does not choose it.";
    body.appendChild(p);

    var t = document.createElement("table");
    function row(k, v) {
      var tr = document.createElement("tr");
      var a = document.createElement("td"); a.textContent = k;
      var b = document.createElement("td"); b.textContent = v;
      tr.appendChild(a); tr.appendChild(b); t.appendChild(tr);
    }
    row("Questions answered", String(state.answers.length));
    row("Profile confidence", Math.round((data.confidence || 0) * 100) + "%");
    row("Candidates proposed", String(d.proposed || 0));
    row("Verified as real", String(d.verified || 0));
    row("Match score", (data.pick.fit !== undefined)
      ? data.pick.fit.toFixed(2) + " of 1.00" : "—");
    var names = { risk: "Risk appetite", horizon: "Time horizon",
      volatility: "Tolerance for swings", value_growth: "Value vs growth",
      momentum: "Momentum vs contrarian", size: "Company size",
      income: "Wants income", conviction: "Concentration" };
    Object.keys(names).forEach(function (k) {
      if (prof[k] !== undefined) row(names[k], Math.round(prof[k] * 100) + " / 100");
    });
    body.appendChild(t);

    if (data.profile && data.profile.themes &&
        Object.keys(data.profile.themes).length) {
      var th = document.createElement("p");
      th.textContent = "Themes you leaned toward: " +
        Object.keys(data.profile.themes).join(", ") + ".";
      body.appendChild(th);
    }
  }

  // ------------------------------------------------------------------ share
  function shareText() {
    var d = state.result;
    if (!d) return "";
    return "I'm " + d.archetype.name + " — Signal matched me with " +
      (d.pick.name || d.pick.symbol) + " (" + d.pick.symbol + "). " +
      "Find your match:";
  }

  function doShare() {
    var text = shareText();
    var url = location.origin + location.pathname;
    if (navigator.share) {
      navigator.share({ title: "Signal", text: text, url: url })
        .catch(function () {});
      return;
    }
    var payload = text + " " + url;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(payload).then(function () {
        var b = $("shareBtn"), old = b.textContent;
        b.textContent = "Copied";
        setTimeout(function () { b.textContent = old; }, 1600);
      }).catch(function () { window.prompt("Copy your result:", payload); });
    } else {
      window.prompt("Copy your result:", payload);
    }
  }

  // ------------------------------------------------------------------ errors
  function fail(msg, err) {
    $("errTitle").textContent = "That didn't work";
    $("errBody").textContent = msg + " Your answers are still here — try again.";
    if (err && window.console) console.error(err);
    show("Error");
  }

  function failPick(data) {
    var d = data.diagnostics || {};
    $("errTitle").textContent = "No verified match right now";
    var why = d.throttled
      ? "Our market data source is rate-limiting us at the moment, so we " +
        "could not verify any candidate against real prices."
      : "We could not verify a suitable candidate against live market data.";
    $("errBody").textContent = why +
      " We would rather show you nothing than invent a recommendation. " +
      "Please try again in a minute.";
    show("Error");
  }

  // ------------------------------------------------------------------- start
  function begin() {
    state.answers = [];
    state.result = null;
    state.busy = false;
    show("Wait");
    $("waitText").textContent = "Starting…";
    $("waitSub").textContent = "Building your first question";
    post({ action: "next", answers: [] }).then(function (data) {
      if (data.done) { reveal(); return; }
      renderQuestion(data);
    }).catch(function (err) {
      fail("We couldn't start the quiz.", err);
    });
  }

  function restart() {
    $("lockWrap").hidden = true;
    setLock(0);
    state.answers = [];
    state.result = null;
    show("Intro");
  }

  // --------------------------------------------------------------- listeners
  $("startBtn").addEventListener("click", begin);
  $("againBtn").addEventListener("click", begin);
  $("shareBtn").addEventListener("click", doShare);
  $("retryBtn").addEventListener("click", function () {
    if (state.answers.length) {
      // resume mid-game rather than throwing the session away
      show("Wait");
      post({ action: "next", answers: state.answers }).then(function (data) {
        if (data.done) { reveal(); return; }
        renderQuestion(data);
      }).catch(function (err) { fail("Still no connection.", err); });
    } else {
      begin();
    }
  });
  $("brandHome").addEventListener("click", restart);
  $("brandHome").addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); restart(); }
  });

  document.addEventListener("keydown", function (e) {
    if (!$("screenQ").classList.contains("show")) {
      if ((e.key === "Enter") && $("screenIntro").classList.contains("show")) begin();
      return;
    }
    var n = parseInt(e.key, 10);
    if (n >= 1 && n <= 9) {
      var b = $("options").querySelector('[data-i="' + (n - 1) + '"]');
      if (b) { e.preventDefault(); choose(n - 1); }
    }
  });
})();
