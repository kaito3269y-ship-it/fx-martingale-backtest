import argparse, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd

PAIRS=['eurjpy','usdjpy','gbpusd','audusd','eurgbp']
TOTAL=2_000_000; DEPLOYED=1_500_000; RESERVE=500_000
MULT=[1.20,1.28,1.35,1.42,1.50]; LEVELS=[5,6,7,8,10]
SPREAD={'eurjpy':1.0,'usdjpy':0.8,'gbpusd':1.0,'audusd':1.0,'eurgbp':1.2}
SLIP=0.2

def pip(pair): return .01 if pair.endswith('jpy') else .0001

def download(pair,tf,start,end):
    d=Path('data')/pair; d.mkdir(parents=True,exist_ok=True)
    # dukascopy-node CLI. Keep raw data out of git; workflow downloads it on demand.
    cmd=['npx','dukascopy-node','-i',pair,'-from',start,'-to',end,'-t',tf,'-f','csv','-v','true','-dir',str(d)]
    subprocess.run(cmd,check=True)
    fs=sorted(d.rglob('*.csv'),key=lambda x:x.stat().st_mtime,reverse=True)
    if not fs: raise RuntimeError(f'No CSV for {pair}')
    return fs[0]

def load(f):
    x=pd.read_csv(f); x.columns=[str(c).lower().strip() for c in x.columns]
    tc=next((c for c in x if c in ('timestamp','time','date','datetime')),x.columns[0])
    x['time']=pd.to_datetime(x[tc],utc=True,errors='coerce')
    for q in ['open','high','low','close']:
        if q not in x:
            c=next((c for c in x if q in c),None)
            if c: x[q]=x[c]
        x[q]=pd.to_numeric(x[q],errors='coerce')
    return x.dropna(subset=['time','open','high','low','close']).sort_values('time').reset_index(drop=True)

def signals(x,k):
    c=x.close
    if k==0:
        m=c.rolling(96).mean(); s=c.rolling(96).std(); z=(c-m)/s
        return np.where(z<-2,1,np.where(z>2,-1,0))
    if k==1:
        m=c.rolling(80).mean(); s=c.rolling(80).std()
        return np.where(c<m-2*s,1,np.where(c>m+2*s,-1,0))
    if k==2:
        d=c.diff(); u=d.clip(lower=0).rolling(14).mean(); dn=(-d.clip(upper=0)).rolling(14).mean(); r=100-100/(1+u/dn.replace(0,np.nan))
        return np.where(r<28,1,np.where(r>72,-1,0))
    if k==3:
        m=c.ewm(span=120,adjust=False).mean(); z=(c-m)/c
        return np.where(z<-.008,1,np.where(z>.008,-1,0))
    hi=c.rolling(96).max().shift(1); lo=c.rolling(96).min().shift(1)
    return np.where(c<lo,1,np.where(c>hi,-1,0))

def jpy_pip_value(pair,lot,px,usdjpy=150.0):
    # 1 lot=100k base units. Quote-currency pip value converted to JPY.
    quote=pair[3:]
    v=100000*lot*pip(pair)
    if quote=='jpy': return v
    if quote=='usd': return v*usdjpy
    if quote=='gbp': return v*px*usdjpy if pair=='eurgbp' else v*usdjpy
    return v*usdjpy

def run(x,pair,k,capital,mult,levels,lot,grid,tp):
    sg=signals(x,k); cash=float(capital); wd=0.; pos=[]; side=0; fills=baskets=busts=0; peak=capital; maxdd=0.; month=None
    for i,r in x.iterrows():
        ym=(r.time.year,r.time.month)
        if month is not None and ym!=month and not pos and cash>capital: wd+=cash-capital; cash=float(capital)
        month=ym; p=pip(pair); px=float(r.close)
        if not pos and sg[i]:
            side=int(sg[i]); e=px+side*(SPREAD[pair]/2+SLIP)*p; pos=[(e,lot)]; fills+=1; baskets+=1
        elif pos:
            tq=sum(q for _,q in pos); avg=sum(e*q for e,q in pos)/tq; target=avg+side*tp*p; adverse=pos[-1][0]-side*grid*p
            grid_hit=(side==1 and r.low<=adverse) or (side==-1 and r.high>=adverse)
            tp_hit=(side==1 and r.high>=target) or (side==-1 and r.low<=target)
            # pessimistic intrabar ordering
            if grid_hit and len(pos)<levels:
                q=lot*(mult**len(pos)); e=adverse+side*(SPREAD[pair]/2+SLIP)*p; pos.append((e,q)); fills+=1; tp_hit=False
            if tp_hit:
                ex=target-side*(SPREAD[pair]/2+SLIP)*p
                cash+=sum(side*(ex-e)/p*jpy_pip_value(pair,q,ex) for e,q in pos); pos=[]; side=0
        unreal=sum(side*(px-e)/p*jpy_pip_value(pair,q,px) for e,q in pos) if pos else 0
        eq=max(0.,cash+unreal); peak=max(peak,eq); maxdd=max(maxdd,(peak-eq)/peak if peak else 1.)
        # Approximate margin-level stopout; conversion is conservative/approximate for non-JPY bases.
        margin=sum(100000*q*px/1000 for _,q in pos) if pos else 0
        if margin and eq/margin*100<=20:
            busts+=1; cash=0.; pos=[]; break
    return {'final_equity':round(cash,2),'withdrawals':round(wd,2),'wealth':round(cash+wd,2),'busts':busts,'baskets':baskets,'fills':fills,'max_dd':round(maxdd,6)}

def main():
    a=argparse.ArgumentParser(); a.add_argument('--tf',default='m15'); a.add_argument('--start',default='2016-09-01'); a.add_argument('--end',default='2026-09-01'); z=a.parse_args()
    # Candidate capital allocations in 50k increments, seeded toward higher ladder risk.
    alloc=np.array([200,225,275,350,450])*1000; alloc[-1]+=DEPLOYED-int(alloc.sum())
    rows=[]
    for k,pair in enumerate(PAIRS):
        x=load(download(pair,z.tf,z.start,z.end)); cut=int(len(x)*.7); tr=x.iloc[:cut].reset_index(drop=True); oo=x.iloc[cut:].reset_index(drop=True)
        candidates=[]
        for lot in [.01,.015,.02,.025,.03]:
          for grid in [30,40,60,80,100,130]:
            for tp in [10,15,20,25,30]:
              r=run(tr,pair,k,int(alloc[k]),MULT[k],LEVELS[k],lot,grid,tp)
              score=r['withdrawals']+(r['final_equity']-alloc[k])-alloc[k]*4*r['max_dd']-alloc[k]*8*r['busts']
              candidates.append((score,lot,grid,tp,r))
        _,lot,grid,tp,train=max(candidates,key=lambda q:q[0]); oos=run(oo,pair,k,int(alloc[k]),MULT[k],LEVELS[k],lot,grid,tp)
        rows.append({'pair':pair,'allocation':int(alloc[k]),'mult':MULT[k],'levels':LEVELS[k],'lot':lot,'grid_pips':grid,'tp_pips':tp,'rows':len(x),'data_start':str(x.time.iloc[0]),'data_end':str(x.time.iloc[-1]),**{f'train_{q}':v for q,v in train.items()},**{f'oos_{q}':v for q,v in oos.items()}})
    Path('results').mkdir(exist_ok=True); df=pd.DataFrame(rows); df.to_csv('results/summary.csv',index=False)
    summary={'total_start_jpy':TOTAL,'deployed_jpy':DEPLOYED,'reserve_jpy':RESERVE,'timeframe':z.tf,'notes':['70/30 train/OOS','pessimistic same-bar ordering','spread/slippage modeled','historical HFM swap not claimed; add sensitivity before production use'],'accounts':rows}
    Path('results/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(df.to_string(index=False))
if __name__=='__main__': main()
