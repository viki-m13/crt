/* Pattern finder.
 *
 * Draws the analog chart the way the reference charts do: every path indexed
 * to 100 at the start of the matched window, the current stretch overlaid in
 * white, and each analog continuing past "now" into what actually followed
 * it. The vertical rule at the join is the only honest place to look —
 * everything left of it is matched, everything right of it is history that
 * has not been matched to anything.
 *
 * The accuracy panel renders before the chart and cannot be dismissed. It is
 * populated from the API response rather than hardcoded here, so it cannot
 * drift away from the measurement.
 */
(function () {
  'use strict';

  var API = '/api/analog';
  var $ = function (id) { return document.getElementById(id); };
  var SERIES = ['--s1', '--s2', '--s3', '--s4', '--s5'];

  var state = { data: null, view: 'chart' };

  function pct(x, dp) {
    if (x === null || x === undefined || isNaN(x)) return '—';
    return (x * 100).toFixed(dp === undefined ? 1 : dp) + '%';
  }
  function signed(x) {
    if (x === null || x === undefined || isNaN(x)) return '—';
    return (x >= 0 ? '+' : '') + (x * 100).toFixed(1) + '%';
  }
  function cssVar(name) {
    return getComputedStyle(document.querySelector('.pat-main'))
      .getPropertyValue(name).trim();
  }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  /* ------------------------------------------------------- accuracy ---- */
  function renderTruth(a) {
    if (!a) return;
    $('tv-acc').textContent = pct(a.directional_accuracy);
    $('tv-base').textContent = pct(a.always_up_baseline);
    $('tv-rand').textContent = pct(a.random_control_accuracy);
    $('truth-note').textContent = a.verdict || '';
    $('truth').hidden = false;
    $('method').hidden = false;
  }

  /* ---------------------------------------------------------- chart ---- */
  function draw(d) {
    var svg = $('chart');
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var L = d.lookback, H = d.horizon;
    var total = L + H;                     // x runs 0..total, join at L
    // ~3.4px per trading day keeps 180 points readable; the container
    // scrolls rather than the chart shrinking to a phone's width. On a wide
    // screen there is no reason to leave the card half empty, so it grows to
    // fill instead.
    var avail = (svg.parentNode.clientWidth || 600);
    var W = Math.max(600, Math.round(total * 3.4), avail);
    var Ht = Math.min(420, Math.max(300, Math.round(window.innerHeight * 0.44)));
    var m = { t: 16, r: 16, b: 22, l: 46 };
    var iw = W - m.l - m.r, ih = Ht - m.t - m.b;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + Ht);
    svg.setAttribute('width', W);
    svg.setAttribute('height', Ht);
    svg.style.width = W + 'px';
    svg.style.height = Ht + 'px';

    /* every analog's full path: matched window then what followed */
    var lines = d.analogs.map(function (a, i) {
      return {
        label: a.start.slice(0, 7) + ' → ' + a.end.slice(0, 7),
        colour: cssVar(SERIES[i % SERIES.length]),
        // forward[0] repeats the last matched bar, so drop it when joining
        pts: a.path.concat(a.forward.slice(1)),
        ret: a.forward_return,
        analog: a
      };
    });
    var now = { label: 'Now', colour: cssVar('--now'), pts: d.query_path,
                now: true };

    var all = [];
    lines.forEach(function (l) { all = all.concat(l.pts); });
    all = all.concat(now.pts);
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    var pad = (hi - lo) * 0.08 || 1;
    lo -= pad; hi += pad;

    var X = function (i) { return m.l + (i / total) * iw; };
    var Y = function (v) { return m.t + ih - ((v - lo) / (hi - lo)) * ih; };

    var ns = 'http://www.w3.org/2000/svg';
    function el(n, at) {
      var e = document.createElementNS(ns, n);
      for (var k in at) if (at.hasOwnProperty(k)) e.setAttribute(k, at[k]);
      svg.appendChild(e);
      return e;
    }

    /* recessive gridlines; their labels live on the pinned axis below */
    var grid = cssVar('--line') || '#262b3d';
    var muted = cssVar('--muted') || '#8f97ae';
    var ticks = [];
    for (var g = 0; g <= 4; g++) {
      var v = lo + (hi - lo) * (g / 4), y = Y(v);
      el('line', { x1: m.l, x2: W - m.r, y1: y, y2: y,
                   stroke: grid, 'stroke-width': 1 });
      ticks.push({ v: v, y: y });
    }

    var ax = $('yaxis');
    while (ax.firstChild) ax.removeChild(ax.firstChild);
    ax.setAttribute('viewBox', '0 0 ' + m.l + ' ' + Ht);
    ax.setAttribute('width', m.l);
    ax.setAttribute('height', Ht);
    ax.style.width = m.l + 'px';
    ax.style.height = Ht + 'px';
    ticks.forEach(function (tk) {
      var e = document.createElementNS(ns, 'text');
      e.setAttribute('x', m.l - 8);
      e.setAttribute('y', tk.y + 4);
      e.setAttribute('fill', muted);
      e.setAttribute('font-size', 11);
      e.setAttribute('text-anchor', 'end');
      e.textContent = tk.v.toFixed(0);
      ax.appendChild(e);
    });

    /* the join: left of it is matched, right of it is the sequel */
    var jx = X(L);
    el('line', { x1: jx, x2: jx, y1: m.t, y2: m.t + ih,
                 stroke: cssVar('--line-bright') || '#3a4159',
                 'stroke-width': 1.5, 'stroke-dasharray': '4 4' });
    var jl = el('text', { x: jx + 6, y: m.t + 13, fill: muted, 'font-size': 11 });
    jl.textContent = 'today';

    function path(pts, colour, width, dash, opacity) {
      var dstr = pts.map(function (p, i) {
        return (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(p).toFixed(1);
      }).join(' ');
      el('path', { d: dstr, fill: 'none', stroke: colour,
                   'stroke-width': width, 'stroke-linejoin': 'round',
                   'stroke-linecap': 'round',
                   'stroke-dasharray': dash || 'none',
                   opacity: opacity === undefined ? 1 : opacity });
    }

    lines.forEach(function (l) { path(l.pts, l.colour, 2); });
    /* the current window last so it sits on top, and thicker: it is the
       one series the reader is orienting from */
    path(now.pts, now.colour, 2.6);

    /* hover: a crosshair plus every series' value at that step */
    var hover = el('line', { x1: 0, x2: 0, y1: m.t, y2: m.t + ih,
                             stroke: cssVar('--line-bright'),
                             'stroke-width': 1, opacity: 0 });
    var dots = lines.concat([now]).map(function (l) {
      return el('circle', { r: 4, fill: l.colour, stroke: cssVar('--surface'),
                            'stroke-width': 2, opacity: 0 });
    });
    var tip = $('tip');
    var hit = el('rect', { x: m.l, y: m.t, width: iw, height: ih,
                           fill: 'transparent', style: 'cursor:crosshair' });

    function move(ev) {
      var r = svg.getBoundingClientRect();
      var cx = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
      var i = Math.round(((cx / r.width * W) - m.l) / iw * total);
      var wrapR = $('chart-wrap').getBoundingClientRect();
      i = Math.max(0, Math.min(total, i));
      hover.setAttribute('x1', X(i)); hover.setAttribute('x2', X(i));
      hover.setAttribute('opacity', 1);

      var rows = '', series = lines.concat([now]);
      series.forEach(function (l, k) {
        var v = l.pts[i];
        if (v === undefined) { dots[k].setAttribute('opacity', 0); return; }
        dots[k].setAttribute('cx', X(i));
        dots[k].setAttribute('cy', Y(v));
        dots[k].setAttribute('opacity', 1);
        rows += '<div class="tt-r"><span class="tt-k">' +
          '<span class="sw" style="background:' + l.colour + '"></span>' +
          esc(l.label) + '</span><span>' + v.toFixed(1) + '</span></div>';
      });
      var day = i - L;
      tip.innerHTML = '<div class="tt-h">' +
        (day === 0 ? 'today' : day < 0 ? Math.abs(day) + ' days before'
          : day + ' days after') + '</div>' + rows;
      tip.hidden = false;
      // position against the WRAPPER, not the svg: the svg is wider than the
      // visible area and scrolls, so svg coordinates would push the tooltip
      // off screen exactly when it is needed.
      var tw = tip.offsetWidth || 170;
      var px = r.left + (X(i) / W) * r.width - wrapR.left;
      tip.style.left =
        Math.max(4, Math.min(wrapR.width - tw - 4, px + 14)) + 'px';
      tip.style.top = '10px';
    }
    function leave() {
      hover.setAttribute('opacity', 0);
      dots.forEach(function (c) { c.setAttribute('opacity', 0); });
      tip.hidden = true;
    }
    hit.addEventListener('mousemove', move);
    hit.addEventListener('mouseleave', leave);
    hit.addEventListener('touchstart', move, { passive: true });
    hit.addEventListener('touchmove', move, { passive: true });
    hit.addEventListener('touchend', leave);

    /* legend — identity is never colour alone */
    var leg = $('legend');
    leg.innerHTML = '';
    lines.concat([now]).forEach(function (l) {
      var s = document.createElement('span');
      s.className = 'li';
      s.innerHTML = '<span class="sw" style="background:' + l.colour +
        '"></span>' + esc(l.label);
      leg.appendChild(s);
    });

    /* The forward half is the point of the chart, so open with the join in
       view rather than at the far-left edge of a window that is wider than
       the phone. */
    var sc = svg.parentNode;
    requestAnimationFrame(function () {
      var want = jx - sc.clientWidth * 0.38;
      sc.scrollLeft = Math.max(0, Math.min(W - sc.clientWidth, want));
    });

    $('ch-desc').textContent =
      'Line chart indexed to 100. The current ' + L + '-day window of ' +
      d.symbol + ' with the ' + d.analogs.length + ' closest historical ' +
      'matches, each continuing ' + H + ' trading days past the match.';
  }

  /* ---------------------------------------------------------- table ---- */
  function table(d) {
    var tb = $('tbody');
    tb.innerHTML = '';
    d.analogs.forEach(function (a, i) {
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td><span class="mk"><span class="sw" style="background:' +
          cssVar(SERIES[i % SERIES.length]) + '"></span>Match ' + (i + 1) +
          '</span></td>' +
        '<td>' + esc(a.start) + ' → ' + esc(a.end) + '</td>' +
        '<td class="n">' + a.correlation.toFixed(2) + '</td>' +
        '<td class="n ' + (a.forward_return >= 0 ? 'up' : 'dn') + '">' +
          signed(a.forward_return) + '</td>';
      tb.appendChild(tr);
    });
  }

  function setView(v) {
    state.view = v;
    $('chart-wrap').hidden = v !== 'chart';
    $('legend').hidden = v !== 'chart';
    $('table-wrap').hidden = v !== 'table';
    $('btn-chart').classList.toggle('on', v === 'chart');
    $('btn-table').classList.toggle('on', v === 'table');
    if (v === 'chart' && state.data) draw(state.data);
  }

  /* ----------------------------------------------------------- load ---- */
  var REASONS = {
    not_found: 'No such ticker. Try the exchange suffix — 7203.T, SAP.DE, BP.L.',
    insufficient_history: 'Too little price history to match a pattern against.',
    rate_limited: 'The market data source is throttling us. Try again shortly.',
    unreachable: 'The market data source is not responding right now.',
    no_symbol: 'Enter a ticker to search.',
    server_error: 'Something broke on our side.'
  };

  function status(msg, isErr) {
    var s = $('status');
    s.textContent = msg || '';
    s.className = 'pat-status' + (isErr ? ' err' : '');
    s.hidden = !msg;
  }

  function load(sym) {
    status('Searching ' + sym + "'s history…", false);
    $('go').disabled = true;
    fetch(API + '?symbol=' + encodeURIComponent(sym))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        $('go').disabled = false;
        renderTruth(d.accuracy);          // shown even on failure
        if (!d.ok) {
          $('chartcard').hidden = true;
          status(REASONS[d.reason] || ('Could not load ' + sym + '.'), true);
          return;
        }
        if (!d.analogs || !d.analogs.length) {
          $('chartcard').hidden = true;
          status(d.reason || 'No comparable stretch found in its history.', true);
          return;
        }
        state.data = d;
        status('');
        $('ch-title').textContent = (d.name || d.symbol) + ' — ' +
          d.analogs.length + ' closest matches';
        var ups = d.analogs.filter(function (a) { return a.forward_return > 0; }).length;
        $('ch-sub').textContent =
          'Matching the last ' + d.lookback + ' trading days against ' +
          d.searched.toLocaleString() + ' windows since ' +
          (d.analogs.map(function (a) { return a.start; }).sort()[0] || '') +
          ' · ' + ups + ' of ' + d.analogs.length +
          ' rose over the following ' + d.horizon + ' days (median ' +
          signed(d.median_return) + ')';
        $('chartcard').hidden = false;
        table(d);
        setView(state.view);
        history.replaceState(null, '', '?symbol=' + encodeURIComponent(sym));
      })
      .catch(function () {
        $('go').disabled = false;
        status('Network error. Try again.', true);
      });
  }

  $('form').addEventListener('submit', function (e) {
    e.preventDefault();
    var s = $('sym').value.trim().toUpperCase();
    if (s) load(s);
  });
  $('btn-chart').addEventListener('click', function () { setView('chart'); });
  $('btn-table').addEventListener('click', function () { setView('table'); });

  var resizeT;
  window.addEventListener('resize', function () {
    clearTimeout(resizeT);
    resizeT = setTimeout(function () {
      if (state.data && state.view === 'chart') draw(state.data);
    }, 150);
  });

  /* accuracy is fetched up front so the honesty panel is on screen before
     anyone has typed a ticker */
  fetch(API + '?action=accuracy')
    .then(function (r) { return r.json(); })
    .then(function (d) { renderTruth(d.accuracy); })
    .catch(function () {});

  var qp = new URLSearchParams(location.search).get('symbol');
  if (qp) { $('sym').value = qp.toUpperCase(); load(qp.toUpperCase()); }
})();
