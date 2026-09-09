"""Build the v4 per-stock-day feature panel in ONE pass. PIT: everything from bars <=16:00.
TEST (2024+) excluded at load and never read."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import lfilter
D = Path('/tmp/databranch/data/equity_1m_alpaca')
SYMS = sorted(p.stem for p in D.glob('*.parquet'))

def ema(x,n):
    a=2.0/(n+1.0); return lfilter([a],[1.0,-(1.0-a)],x,zi=[(1.0-a)*x[0]])[0]

def cs_spread(h,l):
    h=np.asarray(h,float); l=np.asarray(l,float)
    if len(h)<2: return np.nan
    b=(np.log(h[1:]/l[1:]))**2+(np.log(h[:-1]/l[:-1]))**2
    hh=np.maximum(h[1:],h[:-1]); ll=np.minimum(l[1:],l[:-1]); g=(np.log(hh/ll))**2
    k=3-2*np.sqrt(2)
    a=(np.sqrt(2*b)-np.sqrt(b))/k-np.sqrt(g/k)
    s=2*(np.exp(a)-1)/(1+np.exp(a)); s=s[np.isfinite(s)]; s=s[s>0]
    return float(np.median(s)) if len(s) else np.nan

rows=[]
for sym in SYMS:
    d=pd.read_parquet(D/f'{sym}.parquet',columns=['t','o','h','l','c','v','n'] if 'n' in pd.read_parquet(D/f'{sym}.parquet').columns[:20] else ['t','o','h','l','c','v'])
    if 'n' not in d.columns: d['n']=np.nan
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.normalize().dt.tz_localize(None)
    d=d[d.day<pd.Timestamp('2024-01-01')]
    d['mn']=d.t.dt.hour*60+d.t.dt.minute
    for day,g in d.groupby('day',sort=True):
        g=g.sort_values('mn')
        c=g.c.to_numpy(float); o=g.o.to_numpy(float); hi=g.h.to_numpy(float); lo=g.l.to_numpy(float)
        v=g.v.to_numpy(float); nn=g.n.to_numpy(float); mn=g.mn.to_numpy()
        if len(c)<300 or not np.all(np.isfinite(c)): continue
        r=np.diff(np.log(c)); mm=mn[1:]
        if not np.all(np.isfinite(r)): continue
        rv=float(np.sqrt((r**2).sum()))
        up=r[r>0]; dn=r[r<0]
        sv_u=float((up**2).sum()); sv_d=float((dn**2).sum())
        vwap=float((c*v).sum()/max(v.sum(),1e-9))
        dollar=float((c*v).sum())
        dayret=float(np.log(c[-1]/o[0]))
        rows.append(dict(sym=sym, day=day, close=float(c[-1]), open=float(o[0]),
            dayret=dayret, rv=rv, nbar=len(c),
            ret_last30=float(np.log(c[-1]/c[-31])) if len(c)>31 else np.nan,
            ret_last60=float(np.log(c[-1]/c[-61])) if len(c)>61 else np.nan,
            ret_first30=float(np.log(c[30]/o[0])) if len(c)>31 else np.nan,
            semi_asym=(sv_u-sv_d)/max(sv_u+sv_d,1e-12),
            skew1m=float(pd.Series(r).skew()),
            kurt1m=float(pd.Series(r).kurt()),
            vwap_dev=float((c[-1]-vwap)/c[-1]),
            rng=float(np.log(hi.max()/lo.min())),
            eff=abs(dayret)/max(rv,1e-9),
            vol=float(v.sum()), dollar=dollar, ntr=float(np.nansum(nn)),
            amihud=abs(dayret)/max(dollar,1.0)*1e9,
            spread=cs_spread(hi,lo)))
    print(f'{sym} ok', flush=True)

P=pd.DataFrame(rows).sort_values(['sym','day']).reset_index(drop=True)
# forward targets, per symbol
P['next_open']=P.groupby('sym')['open'].shift(-1)
P['next_close']=P.groupby('sym')['close'].shift(-1)
P['T_on']=np.log(P.next_open/P.close)
P['T_oc']=np.log(P.next_close/P.next_open)
P['T_cc']=np.log(P.next_close/P.close)
# rolling 20d normalizers (causal: shift 1 so day D uses D-1..D-20)
for col in ('ntr','rng','vol','rv'):
    ma=P.groupby('sym')[col].transform(lambda s: s.shift(1).rolling(20,min_periods=10).mean())
    P[f'{col}_z']=P[col]/ma
P.to_parquet('/tmp/timbre/panel4.parquet',index=False)
print(f'\n{len(P):,} stock-days, {P.day.nunique():,} days, {P.sym.nunique()} syms, {P.day.min().date()}..{P.day.max().date()}')
print('median CS spread (bp):', round(P.spread.median()*1e4,2))
