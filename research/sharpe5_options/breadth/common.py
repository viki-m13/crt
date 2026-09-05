import json, numpy as np, pandas as pd, datetime as dt, os
SP = "/tmp/claude-0/-home-user/4219574e-98a2-5854-9fab-6b12bf27ea5d/scratchpad"
BH = SP + "/bh"

def load(path, adjust=True):
    d = json.load(open(path))['chart']['result'][0]
    ts = d['timestamp']; q = d['indicators']['quote'][0]
    idx = pd.to_datetime(pd.Series(ts), unit='s', utc=True).dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None)
    df = pd.DataFrame({k: q[k] for k in ('open','high','low','close')}, index=idx)
    adj = d['indicators'].get('adjclose')
    if adjust and adj:
        ac = pd.Series(adj[0]['adjclose'], index=idx)
        f = ac / df['close']
        for c in ('open','high','low','close'):
            df[c] = df[c] * f
    df = df.dropna(subset=['close'])
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df

def close(path, adjust=True):
    return load(path, adjust)['close']

def eff_n_pr(C):
    """Eigenvalue participation ratio effective number of independent series."""
    C = np.asarray(C, float)
    w = np.linalg.eigvalsh(C); w = np.clip(w, 0, None)
    return (w.sum()**2) / (w**2).sum()

def eff_n_equi(N, rbar):
    rbar = max(rbar, -1.0/(N-1)+1e-9)
    return N / (1.0 + (N-1)*rbar)

def mean_offdiag(C):
    C = np.asarray(C, float); N = C.shape[0]
    m = ~np.eye(N, dtype=bool)
    return np.nanmean(C[m])

def required_ic(target_sharpe, BR):
    return target_sharpe / np.sqrt(BR)
