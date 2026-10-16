"""Render a self-contained research snapshot; no external scripts or live claims."""
import argparse
import html
import json
from pathlib import Path


def pct(value):
    return 'Not defined' if value is None else f'{value:.1%}'


def render(scan,report):
    e=html.escape
    rows=[]
    for name,r in report['policies']['post2019'].items():
        rows.append('<tr>'+''.join('<td>'+e(str(x))+'</td>' for x in [name,
          r['issued'],r['matured'],r.get('successes',0),pct(r['precision']),
          r['pending'],pct(r['coverage'])])+'</tr>')
    reasons=''.join('<li>'+e(v.replace('_',' '))+'</li>' for v in scan['reasons'])
    research=''
    for c in scan.get('research_candidates_not_recommendations',[]):
        fan=[{'horizon':0,'low_ratio':1,'median_ratio':1,'high_ratio':1}]+c['checkpoint_fan']
        hi=max(p['high_ratio'] for p in fan)*1.05
        def coords(key):
            return ' '.join(f'{45+700*p["horizon"]/fan[-1]["horizon"]:.2f},{230-195*p[key]/hi:.2f}' for p in fan)
        points=coords('high_ratio')+' '+' '.join(reversed(coords('low_ratio').split()))
        graph=f'<svg viewBox="0 0 800 270" role="img" aria-label="Unvalidated checkpoint forecast fan"><polygon points="{points}" fill="#dde5eb"/><polyline points="{coords("median_ratio")}" fill="none" stroke="#244b63" stroke-width="3"/><text x="45" y="257">0</text><text x="625" y="257">{fan[-1]["horizon"]} trading sessions</text></svg>'
        research+=f'<details><summary>Historical research candidate: {e(c["ticker"])} — NOT a buy recommendation</summary><p>Reference date: {e(str(c["as_of"]))}. Horizon: {c["horizon"]} sessions. Raw model score: {pct(c["p_model"])}. This score is not verified accuracy.</p>{graph}<p>The fan joins model checkpoints. It is not a validated daily path or guaranteed floor. Reference index = 100; dividends/corporate-action basis is unverified.</p></details>'
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Return-floor research scanner</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#19303f;background:#f6f8fa;max-width:980px;margin:auto;padding:28px 18px}}h1{{font-size:30px;margin-bottom:5px}}h2{{font-size:21px}}.status{{border-left:5px solid #ae582e;background:#fff2e8;padding:18px;margin:24px 0}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:11px;border-bottom:1px solid #dce3e9;text-align:left}}.scroll{{overflow-x:auto}}.muted{{color:#516777}}details{{background:white;padding:18px;margin:20px 0}}summary{{cursor:pointer;font-weight:600}}svg{{width:100%;height:auto}}footer{{margin-top:25px;font-size:13px}}</style>
<h1>Return-floor research scanner</h1><div class="muted">{e(report['meta']['universe'].upper())} • Generated for {e(scan['as_of'])} • Snapshot, not a live feed</div>
<section class="status"><h2>No validated buy recommendation</h2><p>The &gt;95% goal has not been achieved. Prices in this archive end {e(scan['data_as_of'])}.</p><ul>{reasons}</ul></section>
<h2>Completed historical experiment</h2><p>{report['forecasts']:,} stock/horizon forecasts across {report['dates']} monthly decision dates. Policies below are evaluated on decisions from 2019 onward. Pending outcomes are not wins; missing matured outcomes count as failures.</p>
<div class="scroll"><table><thead><tr><th>Policy</th><th>Issued</th><th>Matured</th><th>Higher</th><th>Hit rate</th><th>Pending</th><th>Date coverage</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">Scores p80/p90/p95 refer to raw model thresholds, not validated success rates. Selection is at most one stock per monthly decision. Forecast horizons span 30–756 trading sessions (up to about three years). All results reuse previously researched historical data.</p>{research}
<footer>Research only. No orders or brokerage connection. Incomplete historical coverage and unverified terminal corporate actions remain material limitations. Empty recommendations are not evidence of achieved forecasting accuracy.</footer></html>'''

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.write_text(render(json.loads(a.scan.read_text()),json.loads(a.report.read_text())))
