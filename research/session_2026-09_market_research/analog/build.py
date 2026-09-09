import json
data=open('/tmp/analog/data.json').read()
html=r'''<title>Analog Finder</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap">
<style>
:root{
  --ground:#f6f7f9; --panel:#ffffff; --ink:#131820; --dim:#5b6675; --faint:#8b95a3;
  --rule:#dde2e9; --rule2:#eef1f5;
  --blue:#2563a8; --blue-soft:rgba(37,99,168,.13); --rust:#a4552b; --rust-bg:#fbf1ea;
  --up:#2f7d5f; --down:#a33a3a;
  --mono:'IBM Plex Mono',ui-monospace,monospace;
  --sans:'IBM Plex Sans',system-ui,sans-serif;
  --serif:'IBM Plex Serif',Georgia,serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0e1319; --panel:#151b23; --ink:#e6eaf0; --dim:#96a1b0; --faint:#6b7684;
  --rule:#252d38; --rule2:#1c232c;
  --blue:#6ba3dd; --blue-soft:rgba(107,163,221,.16); --rust:#d9905f; --rust-bg:#2a1e16;
  --up:#5aae87; --down:#d4746f;
}}
:root[data-theme="dark"]{
  --ground:#0e1319; --panel:#151b23; --ink:#e6eaf0; --dim:#96a1b0; --faint:#6b7684;
  --rule:#252d38; --rule2:#1c232c;
  --blue:#6ba3dd; --blue-soft:rgba(107,163,221,.16); --rust:#d9905f; --rust-bg:#2a1e16;
  --up:#5aae87; --down:#d4746f;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);margin:0;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 64px}
header{border-bottom:1px solid var(--rule);padding-bottom:20px;margin-bottom:22px}
h1{font-family:var(--serif);font-size:34px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--dim);font-size:15px;max-width:64ch;margin:0}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--faint);margin:0 0 10px}

.verdict{background:var(--rust-bg);border:1px solid var(--rust);border-left-width:4px;
  padding:16px 18px;margin:0 0 24px;display:grid;gap:10px}
.verdict h2{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--rust);margin:0;font-weight:600}
.verdict p{margin:0;font-size:14px;color:var(--ink);max-width:72ch}
.vgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:14px;margin-top:4px}
.vg{display:grid;gap:2px}
.vg b{font-family:var(--mono);font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.vg span{font-size:11.5px;color:var(--dim);line-height:1.35}

.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-end;
  background:var(--panel);border:1px solid var(--rule);padding:16px 18px;margin-bottom:20px}
.ctl{display:grid;gap:5px}
.ctl label{font-family:var(--mono);font-size:10.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--faint)}
select,input[type=range]{font-family:var(--mono);font-size:13px;background:var(--ground);color:var(--ink);
  border:1px solid var(--rule);padding:7px 9px;min-width:112px}
input[type=range]{padding:0;min-width:132px;accent-color:var(--blue)}
select:focus-visible,input:focus-visible{outline:2px solid var(--blue);outline-offset:1px}
.val{font-family:var(--mono);font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}

.panel{background:var(--panel);border:1px solid var(--rule);padding:18px;margin-bottom:20px}
.panel h3{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--faint);margin:0 0 4px;font-weight:600}
.panel .cap{font-size:13px;color:var(--dim);margin:0 0 14px;max-width:70ch}
canvas{display:block;width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:12px;font-size:12px;color:var(--dim)}
.legend i{display:inline-block;width:18px;height:2px;vertical-align:middle;margin-right:6px}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:1px;background:var(--rule);
  border:1px solid var(--rule)}
.stat{background:var(--panel);padding:13px 15px;display:grid;gap:3px}
.stat b{font-family:var(--mono);font-size:19px;font-weight:600;font-variant-numeric:tabular-nums}
.stat span{font-size:11.5px;color:var(--dim)}
.pos{color:var(--up)}.neg{color:var(--down)}

table{width:100%;border-collapse:collapse;font-size:13px}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--faint);
  text-align:right;padding:7px 9px;border-bottom:1px solid var(--rule);font-weight:600}
th:first-child,td:first-child{text-align:left}
td{padding:7px 9px;border-bottom:1px solid var(--rule2);font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:12.5px}
.scroll{overflow-x:auto}
footer{color:var(--faint);font-size:12px;border-top:1px solid var(--rule);padding-top:16px;margin-top:8px;max-width:76ch}
footer code{font-family:var(--mono);font-size:11.5px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Historical analog projection &middot; 34 tickers &middot; daily closes since 2005</p>
  <h1>Analog Finder</h1>
  <p class="sub">Finds the historical windows whose price path most closely matches the recent one, and shows what
  followed each. The panel below reports how accurate that projection actually proved out of sample &mdash; measured,
  not asserted.</p>
</header>

<section class="verdict">
  <h2>Measured accuracy &mdash; read before using</h2>
  <p><strong>This tool does not predict the future.</strong> Across <strong>121,287</strong> out-of-sample projections
  on 20 tickers (each using only data available at the time), the analog method got the direction right
  <strong>55.5%</strong> of the time &mdash; while simply assuming &ldquo;up&rdquo; was right <strong>65.3%</strong>.
  Correlation between projected and actual 3-month return was <strong>&minus;0.006</strong>: no information.</p>
  <div class="vgrid">
    <div class="vg"><b>55.5%</b><span>Directional hit rate</span></div>
    <div class="vg"><b style="color:var(--rust)">65.3%</b><span>Hit rate of always saying &ldquo;up&rdquo;</span></div>
    <div class="vg"><b>&minus;0.006</b><span>corr(projected, actual)</span></div>
    <div class="vg"><b>16.0%</b><span>Projections within &plusmn;2% of actual</span></div>
  </div>
  <p>When 14+ of 20 analogs agreed on &ldquo;up&rdquo;, the hit rate was 64.5% &mdash; but the unconditional up-rate in
  those same windows was 65.4%. The agreement carried nothing; the apparent accuracy was the market&rsquo;s drift.
  Use this to see <em>the range of outcomes</em> that followed similar setups, not as a forecast.</p>
</section>

<div class="controls">
  <div class="ctl"><label for="tk">Ticker</label><select id="tk"></select></div>
  <div class="ctl"><label for="asof">As of</label><select id="asof"></select></div>
  <div class="ctl"><label for="w">Match window <span class="val" id="wv"></span></label><input type="range" id="w" min="40" max="250" step="10" value="120"></div>
  <div class="ctl"><label for="h">Horizon <span class="val" id="hv"></span></label><input type="range" id="h" min="21" max="252" step="21" value="63"></div>
  <div class="ctl"><label for="k">Analogs <span class="val" id="kv"></span></label><input type="range" id="k" min="5" max="40" step="5" value="20"></div>
</div>

<div class="panel">
  <h3>Path and analog outcomes</h3>
  <p class="cap" id="chartcap"></p>
  <canvas id="cv" width="1040" height="440"></canvas>
  <div class="legend">
    <span><i style="background:var(--ink)"></i>Actual path to date</span>
    <span><i style="background:var(--blue)"></i>Analog forward paths</span>
    <span><i style="background:var(--blue);height:8px;opacity:.35"></i>10th&ndash;90th percentile band</span>
  </div>
</div>

<div class="stats" id="stats"></div>

<div class="panel" style="margin-top:20px">
  <h3>The matched windows</h3>
  <p class="cap">Each row is a historical window whose normalized path was closest to the current one, and what actually
  followed it. Spread across these rows is the honest output of this tool.</p>
  <div class="scroll"><table id="tbl">
    <thead><tr><th>Window ended</th><th>Match distance</th><th>Forward return</th><th>Max gain</th><th>Max loss</th></tr></thead>
    <tbody></tbody>
  </table></div>
</div>

<footer>
  Prices are dividend-adjusted daily closes. Matching is Euclidean distance on the log path normalized to its first
  day, so only shape is compared, not level or volatility. Candidate windows are constrained to end before the
  current window begins, so no analog overlaps the period being projected. Validation ran the identical procedure at
  every historical date using only prior data (<code>W=120, H=63, K=20</code>, 20 tickers, 2008&ndash;2026).
  Not investment advice.
</footer>
</div>

<script>const DATA=__DATA__;</script>
<script>
const $=id=>document.getElementById(id);
const tk=$('tk'),asof=$('asof'),W=$('w'),H=$('h'),K=$('k');
const syms=Object.keys(DATA).sort();
syms.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;tk.appendChild(o)});
tk.value=syms.includes('SPY')?'SPY':syms[0];

function fillAsOf(){
  const p=DATA[tk.value], n=p.p.length; asof.innerHTML='';
  const opts=[]; for(let i=n-1;i>=750;i-=21) opts.push(i);
  opts.slice(0,60).forEach(i=>{const o=document.createElement('option');o.value=i;o.textContent=p.d[i];asof.appendChild(o)});
}
function analogs(){
  const p=DATA[tk.value], lp=p.p.map(Math.log), t=+asof.value, w=+W.value, h=+H.value, k=+K.value;
  const cur=lp.slice(t-w+1,t+1).map(v=>v-lp[t-w+1]);
  const out=[];
  for(let e=w; e<=t-w-h; e++){
    let d=0; const base=lp[e-w+1];
    for(let j=0;j<w;j++){const x=(lp[e-w+1+j]-base)-cur[j]; d+=x*x;}
    d=Math.sqrt(d/w);
    if(e+h<lp.length) out.push({e,d});
  }
  out.sort((a,b)=>a.d-b.d);
  return out.slice(0,k).map(o=>{
    const path=[]; let mx=-9,mn=9;
    for(let j=0;j<=h;j++){const r=lp[o.e+j]-lp[o.e]; path.push(r); if(r>mx)mx=r; if(r<mn)mn=r;}
    return {...o,path,fwd:path[h],mx,mn,date:p.d[o.e]};
  });
}
function q(a,x){const s=[...a].sort((u,v)=>u-v);const i=(s.length-1)*x;const lo=Math.floor(i),hi=Math.ceil(i);
  return s[lo]+(s[hi]-s[lo])*(i-lo);}

function draw(A){
  const cv=$('cv'),ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1;
  const cssW=cv.clientWidth||1040, cssH=440;
  cv.width=cssW*dpr; cv.height=cssH*dpr; cv.style.height=cssH+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,cssW,cssH);
  const cs=getComputedStyle(document.documentElement);
  const ink=cs.getPropertyValue('--ink').trim(), blue=cs.getPropertyValue('--blue').trim(),
        rule=cs.getPropertyValue('--rule').trim(), faint=cs.getPropertyValue('--faint').trim(),
        soft=cs.getPropertyValue('--blue-soft').trim();
  const p=DATA[tk.value], lp=p.p.map(Math.log), t=+asof.value, w=+W.value, h=+H.value;
  const hist=[]; for(let j=t-w+1;j<=t;j++) hist.push(lp[j]-lp[t]);
  const L=60,R=64,T=18,B=34, gw=cssW-L-R, gh=cssH-T-B;
  let lo=Math.min(...hist), hi=Math.max(...hist);
  A.forEach(a=>a.path.forEach(v=>{if(v<lo)lo=v;if(v>hi)hi=v}));
  const pad=(hi-lo)*0.12||0.05; lo-=pad; hi+=pad;
  const X=i=>L+gw*(i/(w+h)), Y=v=>T+gh*(1-(v-lo)/(hi-lo));
  ctx.strokeStyle=rule; ctx.lineWidth=1; ctx.font='11px "IBM Plex Mono",monospace'; ctx.fillStyle=faint;
  for(let g=0;g<=4;g++){const v=lo+(hi-lo)*g/4, y=Y(v);
    ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(L+gw,y);ctx.stroke();
    ctx.textAlign='right';ctx.fillText(((Math.exp(v)-1)*100).toFixed(0)+'%',L-8,y+4);}
  ctx.setLineDash([3,3]);ctx.strokeStyle=faint;ctx.beginPath();ctx.moveTo(X(w-1),T);ctx.lineTo(X(w-1),T+gh);ctx.stroke();ctx.setLineDash([]);
  ctx.textAlign='center';ctx.fillText('today',X(w-1),T+gh+20);
  ctx.fillText('← matched window',X(w/2),T+gh+20);
  ctx.fillText('projection →',X(w+h/2),T+gh+20);
  const band=[];
  for(let j=0;j<=h;j++){const col=A.map(a=>a.path[j]); band.push([q(col,.1),q(col,.9)]);}
  ctx.fillStyle=soft; ctx.beginPath();
  band.forEach((b,j)=>{const x=X(w-1+j); j?ctx.lineTo(x,Y(b[1])):ctx.moveTo(x,Y(b[1]))});
  for(let j=band.length-1;j>=0;j--){ctx.lineTo(X(w-1+j),Y(band[j][0]))}
  ctx.closePath();ctx.fill();
  ctx.strokeStyle=blue;ctx.globalAlpha=.42;ctx.lineWidth=1;
  A.forEach(a=>{ctx.beginPath();a.path.forEach((v,j)=>{const x=X(w-1+j),y=Y(v);j?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()});
  ctx.globalAlpha=1;
  ctx.strokeStyle=ink;ctx.lineWidth=2;ctx.beginPath();
  hist.forEach((v,j)=>{const x=X(j),y=Y(v);j?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();
  const med=[];for(let j=0;j<=h;j++)med.push(q(A.map(a=>a.path[j]),.5));
  ctx.strokeStyle=blue;ctx.lineWidth=2;ctx.setLineDash([5,4]);ctx.beginPath();
  med.forEach((v,j)=>{const x=X(w-1+j),y=Y(v);j?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle=blue;ctx.textAlign='left';ctx.font='600 11px "IBM Plex Mono",monospace';
  ctx.fillText('median '+((Math.exp(med[h])-1)*100).toFixed(1)+'%',X(w+h)+6,Y(med[h])+4);
}
function render(){
  $('wv').textContent=W.value+'d'; $('hv').textContent=H.value+'d'; $('kv').textContent=K.value;
  const A=analogs();
  if(!A.length){$('chartcap').textContent='Not enough history for these settings.';return}
  draw(A);
  const f=A.map(a=>a.fwd), up=f.filter(v=>v>0).length;
  const p=DATA[tk.value];
  $('chartcap').textContent=`${tk.value} as of ${p.d[+asof.value]} — the ${A.length} closest historical matches to the last ${W.value} trading days, and the ${H.value} days that followed each.`;
  const pct=v=>((Math.exp(v)-1)*100).toFixed(1)+'%';
  $('stats').innerHTML=[
    ['Median outcome',pct(q(f,.5)),q(f,.5)>0?'pos':'neg'],
    ['10th percentile',pct(q(f,.1)),'neg'],
    ['90th percentile',pct(q(f,.9)),'pos'],
    ['Analogs positive',up+' of '+A.length,''],
    ['Spread (90th−10th)',((Math.exp(q(f,.9))-Math.exp(q(f,.1)))*100).toFixed(1)+' pts','']
  ].map(([s,v,c])=>`<div class="stat"><b class="${c}">${v}</b><span>${s}</span></div>`).join('');
  $('tbl').querySelector('tbody').innerHTML=A.slice(0,12).map(a=>
    `<tr><td>${a.date}</td><td>${a.d.toFixed(4)}</td><td class="${a.fwd>0?'pos':'neg'}">${pct(a.fwd)}</td><td class="pos">${pct(a.mx)}</td><td class="neg">${pct(a.mn)}</td></tr>`).join('');
}
tk.addEventListener('change',()=>{fillAsOf();render()});
[asof,W,H,K].forEach(e=>e.addEventListener('input',render));
window.addEventListener('resize',()=>{const A=analogs();if(A.length)draw(A)});
fillAsOf(); render();
</script>'''
open('/tmp/analog/tool.html','w').write(html.replace('__DATA__',data))
import os; print(f"{os.path.getsize('/tmp/analog/tool.html')/1e6:.2f} MB")
