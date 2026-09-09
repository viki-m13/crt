#!/usr/bin/env python3
"""Drive the pattern finder in a real browser and audit its layout.

  python scripts/shoot_patterns.py [base_url] [outdir]

Screenshots the page at several viewports and reports the layout defects that
can be measured rather than eyeballed — horizontal overflow, tap targets under
the 44px minimum, elements clipped by the viewport, and SVG text spilling out
of the chart. A chart that looks fine at 1280px and breaks at 390px is the
normal failure, so the small sizes are the point.
"""
from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099").rstrip("/")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/patterns_shots"

VIEWPORTS = [
    ("iphone-se", 375, 667, 2),
    ("iphone-15", 393, 852, 3),
    ("pixel-8", 412, 915, 2.6),
    ("ipad", 820, 1180, 2),
    ("desktop", 1280, 900, 1),
]

AUDIT = """() => {
  const out = {overflow: [], small: [], clipped: []};
  const vw = document.documentElement.clientWidth;
  out.docScrollW = document.documentElement.scrollWidth;
  out.vw = vw;
  document.querySelectorAll('*').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none') return;
    // horizontal overflow that is not inside a deliberate scroller
    // an element parked far off-canvas is the skip-link pattern, not a
    // layout bug — only flag things that are nearly on screen
    if ((r.right > vw + 1 || (r.left < -1 && r.right > -200))) {
      let p = el.parentElement, scroller = false;
      while (p) {
        const pc = getComputedStyle(p);
        if (pc.overflowX === 'auto' || pc.overflowX === 'scroll') { scroller = true; break; }
        p = p.parentElement;
      }
      if (!scroller) out.overflow.push({
        tag: el.tagName.toLowerCase(), cls: el.className.toString().slice(0,40),
        left: Math.round(r.left), right: Math.round(r.right)});
    }
    // interactive targets below the 44px minimum
    const tappable = ['A','BUTTON','INPUT','SELECT'].includes(el.tagName);
    if (tappable && (r.height < 44 || r.width < 24)) out.small.push({
      tag: el.tagName.toLowerCase(), txt: (el.textContent||'').trim().slice(0,26),
      w: Math.round(r.width), h: Math.round(r.height)});
    // text taller than its own box
    if (el.children.length === 0 && el.scrollHeight > el.clientHeight + 2
        && cs.overflow === 'visible' && el.clientHeight > 0)
      out.clipped.push({tag: el.tagName.toLowerCase(),
        txt: (el.textContent||'').trim().slice(0,26)});
  });
  return out;
}"""


def main():
    os.makedirs(OUT, exist_ok=True)
    problems = 0
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        for name, w, h, dpr in VIEWPORTS:
            ctx = b.new_context(viewport={"width": w, "height": h},
                                device_scale_factor=dpr,
                                is_mobile=w < 800, has_touch=w < 800)
            pg = ctx.new_page()
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: errs.append(m.text)
                  if m.type == "error" else None)

            pg.goto(f"{BASE}/patterns", wait_until="networkidle")
            pg.wait_for_timeout(500)
            pg.screenshot(path=f"{OUT}/{name}-1-landing.png", full_page=True)

            pg.fill("#sym", "NVDA")
            pg.click("#go")
            pg.wait_for_selector("#chartcard:not([hidden])", timeout=25000)
            pg.wait_for_timeout(800)
            pg.screenshot(path=f"{OUT}/{name}-2-chart.png", full_page=True)

            a = pg.evaluate(AUDIT)
            print(f"\n=== {name} {w}x{h} @{dpr}x")
            print(f"  document scrollWidth {a['docScrollW']} vs viewport {a['vw']}"
                  + ("  << HORIZONTAL SCROLL" if a["docScrollW"] > a["vw"] + 1 else "  ok"))
            if a["docScrollW"] > a["vw"] + 1:
                problems += 1
            for k, label in (("overflow", "overflowing"), ("small", "tap target <44px"),
                             ("clipped", "clipped text")):
                if a[k]:
                    problems += len(a[k])
                    print(f"  {label}: {len(a[k])}")
                    for x in a[k][:6]:
                        print("    ", x)
                else:
                    print(f"  {label}: none")

            # the table view is a required accessibility path — check it too
            pg.click("#btn-table")
            pg.wait_for_timeout(300)
            pg.screenshot(path=f"{OUT}/{name}-3-table.png", full_page=True)
            rows = pg.eval_on_selector_all("#tbody tr", "els => els.length")
            print(f"  table rows: {rows}" + ("" if rows else "  << EMPTY"))
            if not rows:
                problems += 1

            if errs:
                problems += len(errs)
                print(f"  JS errors: {errs[:3]}")
            else:
                print("  JS errors: none")
            ctx.close()
        b.close()

    print(f"\n{'=' * 60}\n{problems} problem(s) found; shots in {OUT}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
