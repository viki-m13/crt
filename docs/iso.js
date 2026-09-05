/* ISO / AMT calculator client.
 *
 * The numbers come from the server (api/_tax.py) — nothing is computed here,
 * so the page cannot disagree with the engine that was tested. The prose
 * arrives in the same response and may be absent; the result is complete and
 * useful without it, which is why the layout does not depend on it.
 */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  function digits(s) {
    return String(s == null ? "" : s).replace(/[^0-9.]/g, "");
  }
  function num(id) {
    var v = digits($(id).value);
    if (v === "" || isNaN(parseFloat(v))) return null;
    return parseFloat(v);
  }
  function money(v, dp) {
    if (v == null || isNaN(v)) return "—";
    return "$" + Number(v).toLocaleString(undefined, {
      minimumFractionDigits: dp || 0, maximumFractionDigits: dp || 0 });
  }
  function count(v) {
    if (v == null || isNaN(v)) return "—";
    return Number(v).toLocaleString();
  }

  // live thousands separators on the money fields, caret kept at the end
  ["income", "shares", "ltcg", "deduction"].forEach(function (id) {
    var el = $(id);
    if (!el) return;
    el.addEventListener("input", function () {
      var raw = digits(el.value).split(".")[0];
      if (raw === "") { el.value = ""; return; }
      el.value = Number(raw).toLocaleString();
    });
  });

  function setError(msg, title) {
    $("errTitle").textContent = title || "That didn't work";
    $("errBody").textContent = msg;
    $("errBox").hidden = false;
    $("results").hidden = true;
  }

  function renderConstants(c) {
    $("constYear").textContent = "(" + c.tax_year + ")";
    $("constSrc").textContent = c.source;
    var body = $("constTable").querySelector("tbody");
    body.innerHTML = "";
    var st = $("status").value === "mfj" ? "mfj" : "single";
    var label = st === "mfj" ? "married filing jointly" : "single";
    [["AMT exemption (" + label + ")", money(c.amt_exemption[st])],
     ["Exemption phaseout starts", money(c.amt_phaseout_start[st])],
     ["Phaseout rate", (c.amt_phaseout_rate * 100).toFixed(0) + "¢ per dollar"],
     ["AMT rates", (c.amt_rate_low * 100) + "% then " + (c.amt_rate_high * 100) + "%"],
     ["28% rate starts above", money(c.amt_28_threshold) + " of AMT base"],
     ["Standard deduction", money(c.standard_deduction[st])]
    ].forEach(function (r) {
      var tr = document.createElement("tr");
      var a = document.createElement("td"); a.textContent = r[0];
      var b = document.createElement("td"); b.textContent = r[1];
      tr.appendChild(a); tr.appendChild(b); body.appendChild(tr);
    });
  }

  function renderLadder(rows, freeShares) {
    var body = $("ladderTable").querySelector("tbody");
    body.innerHTML = "";
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      tr.className = r.amt_owed > 0.5 ? "hit" : "free";
      [count(r.shares), money(r.exercise_cost), money(r.bargain_element),
       r.amt_owed > 0.5 ? money(r.amt_owed) : "none"].forEach(function (v, i) {
        var td = document.createElement("td");
        td.textContent = v;
        if (i === 3) td.className = "amt";
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }

  function renderExplain(e) {
    var card = $("explainCard");
    if (!e || !e.text) { card.hidden = true; return; }
    var box = $("explainText");
    box.innerHTML = "";
    // Belt and braces: the prompt forbids markdown, but a model that emits
    // **bold** anyway would otherwise render as literal asterisks, since
    // this is inserted as text (never as HTML) on purpose.
    var clean = e.text.replace(/\*\*(.+?)\*\*/g, "$1")
                      .replace(/(^|\s)[*_]([^*_\n]+)[*_](?=\s|[.,;:!?)]|$)/g, "$1$2")
                      .replace(/^#{1,6}\s*/gm, "");
    clean.split(/\n\s*\n/).forEach(function (para) {
      var t = para.trim();
      if (!t) return;
      var p = document.createElement("p");
      p.textContent = t;
      box.appendChild(p);
    });
    $("explainSrc").textContent =
      "Written from the figures above — every number in this paragraph is "
      + "computed, not estimated.";
    card.hidden = false;
  }

  function render(d) {
    var p = d.plan, ex = p.exercise_all;
    $("errBox").hidden = true;

    $("freeShares").textContent = count(p.amt_free_shares);
    $("freeSub").textContent = p.amt_free_shares === 1
      ? "share this year with zero AMT" : "shares this year with zero AMT";
    $("freeCost").textContent = money(p.amt_free_exercise_cost);
    $("freeBe").textContent = money(p.amt_free_bargain_element);
    $("freeTax").textContent = "$0";

    $("allCost").textContent = money(ex.exercise_cost);
    $("allAmt").textContent = money(ex.amt_owed);
    $("allMarginal").textContent = Math.round(ex.marginal_amt_rate * 100) + "¢";

    var note = "";
    if (ex.amt_owed > 0.5) {
      note = "Exercising everything at once would cost <b>" +
        money(ex.amt_owed) + "</b> in AMT on top of " +
        money(p.baseline_tax) + " regular tax";
      if (p.years_to_exercise_all_amt_free) {
        note += ", against roughly <b>" + p.years_to_exercise_all_amt_free +
          " years</b> of AMT-free exercising to get through the whole grant " +
          "at this income.";
      } else { note += "."; }
      if (ex.marginal_amt_rate > 0.30) {
        note += " At that size each extra dollar of bargain element costs " +
          Math.round(ex.marginal_amt_rate * 100) + "¢ — you would be inside " +
          "the exemption phaseout, where the rate is well above the headline " +
          "26–28%.";
      }
    } else {
      note = "Your whole grant fits under the crossover — exercising all of " +
        "it this year would not trigger AMT at this income.";
    }
    $("ladderNote").innerHTML = note;

    renderLadder(d.ladder || [], p.amt_free_shares);
    renderExplain(d.explanation);
    renderConstants(d.constants);

    var ul = $("assumptions");
    ul.innerHTML = "";
    (d.assumptions || []).forEach(function (a) {
      var li = document.createElement("li");
      li.textContent = a;
      ul.appendChild(li);
    });

    $("results").hidden = false;
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  $("isoForm").addEventListener("submit", function (e) {
    e.preventDefault();
    var shares = num("shares"), strike = num("strike"), fmv = num("fmv");
    var income = num("income");

    [["shares", shares], ["fmv", fmv]].forEach(function (f) {
      $(f[0]).classList.toggle("bad", !f[1] || f[1] <= 0);
    });
    if (!shares || shares <= 0) { setError("Enter how many ISO options you hold."); return; }
    if (!fmv || fmv <= 0) { setError("Enter the current share price."); return; }
    if (strike == null) { setError("Enter your strike price (use 0 if it really is zero)."); return; }
    if (fmv <= strike) {
      setError("Your current price is at or below your strike, so there's no "
        + "bargain element and no AMT to plan around.", "Nothing to calculate");
      return;
    }

    var btn = $("calcBtn");
    btn.disabled = true;
    btn.textContent = "Calculating…";

    fetch("/api/iso", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        shares: shares, strike: strike, fmv: fmv,
        income: income || 0,
        status: $("status").value,
        ltcg: num("ltcg") || 0,
        deduction: num("deduction"),
        itemized: $("itemized").checked
      })
    }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (d) {
      btn.disabled = false;
      btn.textContent = "Recalculate";
      if (!d.ok) {
        setError(d.message || d.reason || "Check your inputs and try again.",
                 d.reason === "underwater" ? "Nothing to calculate" : undefined);
        return;
      }
      render(d);
    }).catch(function (err) {
      btn.disabled = false;
      btn.textContent = "Find my AMT crossover";
      setError("We couldn't reach the calculator. Please try again.");
      if (window.console) console.error(err);
    });
  });

  $("brandHome").addEventListener("click", function () { location.href = "/"; });
  $("brandHome").addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); location.href = "/"; }
  });

  // show the constants in force before anyone calculates, so the page is
  // useful (and checkable) on arrival
  fetch("/api/iso?action=constants").then(function (r) { return r.json(); })
    .then(function (c) {
      if (c && c.tax_year) $("yearTag").textContent = c.tax_year + " rules";
    }).catch(function () {});
})();
