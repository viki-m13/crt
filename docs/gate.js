/* CreditFloor / Daily Stock Guide — site-wide password gate.
 *
 * Static-site client-side gate. Loaded synchronously in <head> via
 *   <script src="/gate.js"></script>
 * and paired with
 *   <style id="cf-gate-init">html{visibility:hidden!important}</style>
 *
 * Once the user enters the correct password (stored in PASSWORD below)
 * we set `cf_unlocked=1` in localStorage so subsequent page loads on
 * the same device skip the prompt. Clearing site data restores the gate.
 *
 * NOTE: This is gating, not security. The password is in plain JS and
 * any motivated visitor can read it from source. Use this for casual
 * "no public crawl" purposes only.
 */
(function () {
  "use strict";
  var KEY = "cf_unlocked";
  var PASSWORD = "marti";

  function reveal() {
    var s = document.getElementById("cf-gate-init");
    if (s) s.remove();
  }

  if (localStorage.getItem(KEY) === "1") {
    reveal();
    return;
  }

  function buildGate() {
    reveal();
    var g = document.createElement("div");
    g.id = "cf-site-gate";
    g.style.cssText =
      "position:fixed;inset:0;z-index:2147483647;background:#fff;" +
      "display:flex;align-items:center;justify-content:center;" +
      "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif";
    g.innerHTML =
      "<div style=\"max-width:360px;width:90%;padding:32px 28px;" +
      "border:1px solid #000;border-radius:4px;background:#fff\">" +
      "<h1 style=\"font-size:18px;font-weight:600;margin:0 0 12px;letter-spacing:-0.02em\">" +
      "Restricted access</h1>" +
      "<p style=\"font-size:14px;color:#555;margin:0 0 16px;line-height:1.5\">" +
      "Enter the password to view this site.</p>" +
      "<input type=\"password\" id=\"cf-gate-pw\" placeholder=\"Password\" " +
      "autocomplete=\"current-password\" autofocus " +
      "style=\"width:100%;padding:10px 12px;font-size:15px;" +
      "border:1px solid #000;border-radius:3px;" +
      "font-family:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;" +
      "box-sizing:border-box\">" +
      "<button id=\"cf-gate-go\" style=\"margin-top:12px;width:100%;padding:10px;" +
      "background:#000;color:#fff;border:none;border-radius:3px;" +
      "font-size:14px;font-weight:600;cursor:pointer\">Unlock</button>" +
      "<div id=\"cf-gate-err\" style=\"color:#a8321a;font-size:13px;" +
      "margin-top:8px;min-height:18px\"></div></div>";
    document.body.appendChild(g);
    var pw = g.querySelector("#cf-gate-pw");
    var btn = g.querySelector("#cf-gate-go");
    var err = g.querySelector("#cf-gate-err");
    function attempt() {
      if (pw.value === PASSWORD) {
        try { localStorage.setItem(KEY, "1"); } catch (e) {}
        g.remove();
      } else {
        err.textContent = "Incorrect password.";
        pw.select();
      }
    }
    btn.addEventListener("click", attempt);
    pw.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); attempt(); }
    });
    setTimeout(function () { try { pw.focus(); } catch (e) {} }, 30);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", buildGate);
  } else {
    buildGate();
  }
})();
