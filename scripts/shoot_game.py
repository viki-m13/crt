#!/usr/bin/env python3
"""Drive the quiz in a real browser and capture screenshots for review.

  python scripts/shoot_game.py [base_url] [outdir]

Plays a full game at several viewport sizes, screenshotting intro, questions
and the reveal, and reports any layout defect that can be measured rather
than eyeballed: horizontal overflow, targets below the 44px touch minimum,
text clipped by its container, and elements colliding with the viewport edge.
"""
from __future__ import annotations

import json
import os
import sys

from playwright.sync_api import sync_playwright

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/shots"

DEVICES = [
    ("iphone-se", 375, 667, 2),      # the smallest screen still in wide use
    ("iphone-14", 390, 844, 3),
    ("pixel-narrow", 360, 800, 3),   # narrowest common Android
    ("ipad", 768, 1024, 2),
    ("desktop", 1440, 900, 1),
]

AUDIT_JS = r"""
() => {
  const problems = [];
  const de = document.documentElement;
  if (de.scrollWidth > de.clientWidth + 1) {
    problems.push(`horizontal overflow: scrollWidth ${de.scrollWidth} > ${de.clientWidth}`);
  }
  const vw = de.clientWidth;
  document.querySelectorAll('button, a, summary, [role="button"]').forEach(el => {
    // the skip-to-content link is parked off-screen by design and only
    // appears on keyboard focus; flagging it is a false positive
    if (el.classList.contains('skip')) return;
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return;
    if (r.height < 44) {
      problems.push(`tap target ${r.height.toFixed(0)}px tall: ` +
        `${el.tagName}.${el.className || '-'} "${(el.textContent||'').trim().slice(0,28)}"`);
    }
    if (r.right > vw + 1 || r.left < -1) {
      problems.push(`off-screen: ${el.tagName}.${el.className||'-'} ` +
        `left=${r.left.toFixed(0)} right=${r.right.toFixed(0)} vw=${vw}`);
    }
  });
  document.querySelectorAll('h1,h2,h3,h4,p,li,span,div,button').forEach(el => {
    if (el.children.length) return;
    if (el.scrollWidth > el.clientWidth + 2 && el.clientWidth > 0) {
      const cs = getComputedStyle(el);
      if (cs.overflowX === 'visible' || cs.overflow === 'visible') {
        problems.push(`text clipped: "${(el.textContent||'').trim().slice(0,32)}" ` +
          `${el.scrollWidth}>${el.clientWidth}`);
      }
    }
  });
  return problems;
}
"""


def audit(page, label, results):
    probs = page.evaluate(AUDIT_JS)
    if probs:
        results.append((label, probs))
    return probs


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        for name, w, h, dpr in DEVICES:
            ctx = browser.new_context(viewport={"width": w, "height": h},
                                      device_scale_factor=dpr,
                                      is_mobile=w < 700, has_touch=w < 700)
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                    if m.type == "error" else None)

            page.goto(BASE, wait_until="networkidle")
            page.wait_for_timeout(400)
            page.screenshot(path=f"{OUT}/{name}-1-intro.png", full_page=False)
            audit(page, f"{name} intro", results)

            page.click("#startBtn")
            page.wait_for_selector("#screenQ.show .opt", timeout=20000)
            page.wait_for_timeout(500)
            page.screenshot(path=f"{OUT}/{name}-2-question.png")
            audit(page, f"{name} question", results)

            # play the whole game, always taking the last option
            for i in range(20):
                if page.locator("#screenResult.show").count():
                    break
                opts = page.locator("#screenQ.show .opt")
                if not opts.count():
                    page.wait_for_timeout(700)
                    continue
                if i == 2:
                    page.screenshot(path=f"{OUT}/{name}-3-midgame.png")
                    audit(page, f"{name} midgame", results)
                opts.nth(opts.count() - 1).click()
                page.wait_for_timeout(900)

            try:
                page.wait_for_selector("#screenResult.show", timeout=30000)
            except Exception:
                page.screenshot(path=f"{OUT}/{name}-X-stuck.png", full_page=True)
                results.append((f"{name} result", ["never reached result screen"]))
                ctx.close()
                continue

            page.wait_for_timeout(700)
            page.screenshot(path=f"{OUT}/{name}-4-result.png")
            audit(page, f"{name} result", results)
            page.screenshot(path=f"{OUT}/{name}-5-result-full.png", full_page=True)

            page.click("details.detail summary")
            page.wait_for_timeout(300)
            page.screenshot(path=f"{OUT}/{name}-6-reason.png", full_page=True)
            audit(page, f"{name} reason open", results)

            if errors:
                results.append((f"{name} JS", errors[:6]))
            ctx.close()
        browser.close()

    print("=" * 74)
    if not results:
        print("NO LAYOUT PROBLEMS DETECTED")
    else:
        for label, probs in results:
            print(f"\n{label}:")
            seen = set()
            for p in probs:
                if p in seen:
                    continue
                seen.add(p)
                print(f"  - {p}")
    print("=" * 74)
    print(f"screenshots in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
