import pandas as pd, numpy as np, glob, os, re, warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
import torch, joblib, torch.nn as nn
class EmbeddingNet(nn.Module):
    def __init__(s,i=11,e=16):
        super().__init__(); s.net=nn.Sequential(nn.Linear(i,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,e))
    def forward(s,x): return nn.functional.normalize(s.net(x),p=2,dim=1)
PARAMS=['innodb_buffer_pool_size','innodb_read_io_threads','innodb_write_io_threads','innodb_flush_log_at_trx_commit','innodb_adaptive_hash_index','sync_binlog','innodb_lru_scan_depth','innodb_buffer_pool_instances','innodb_change_buffer_max_size','innodb_io_capacity','innodb_log_file_size','table_open_cache']
MN=['Average Memory Usage Percentage','InnoDB Buffer Pool Cache Hit Rate','InnoDB Dirty Buffer Pages','Current QPS (Queries Per Second)','Max CPU Usage (100 - Idle)','InnoDB Rows Deleted (60s Rate)','InnoDB Rows Inserted (60s Rate)','InnoDB Rows Read (60s Rate)','InnoDB Rows Updated (60s Rate)','Average Disk IOPS (Read)','Average Disk IOPS (Write)']
def nf(v):                       # normalize a knob value to canonical numeric
    s=str(v).strip()
    if s in ('ON','Yes','True'): return 1.0
    if s in ('OFF','No','False'): return 0.0
    if s.endswith('GB'): return float(s[:-2])
    if s.endswith('MB'): return float(s[:-2])/1000.0
    if s.endswith('KB'): return float(s[:-2])/1e6
    try: return float(s)
    except: return np.nan

tg0=pd.read_csv('DBMSTransferLearning/dataset/full_data/112c125g-result.csv')
METRICS114=[c for c in tg0.columns[19:]]                 # 114 metric names (target schema)
def load(f):
    df=pd.read_csv(f); df=df.loc[:,~df.columns.duplicated()]
    hw=os.path.basename(f).replace('-result.csv','')
    K=np.array([[nf(x) for x in df[p]] for p in PARAMS]).T   # (n,12) canonical knobs
    ok=np.isfinite(K).all(1)&pd.to_numeric(df['tps'],errors='coerce').notna().values
    df=df[ok].reset_index(drop=True); K=K[ok]
    ck=['|'.join(str(round(v,6)) for v in row) for row in K]
    if 'workload_label' in df.columns:
        wl=df['workload_label'].astype(str)
    else:
        sk=df['skew'].apply(lambda x:'nan' if pd.isna(x) else (str(int(x)) if float(x)==int(x) else str(x)))
        wl=(df['num_table'].astype(int).astype(str)+'-'+df['table_size'].astype(int).astype(str)+'-'+
            df['num_client'].astype(int).astype(str)+'-'+df['workload'].astype(str)+'-'+sk)
    out=pd.DataFrame({'ctx':hw+'_'+wl,'ck':ck,'tps':pd.to_numeric(df['tps'],errors='coerce').values,'id':df['id'].values})
    for m in METRICS114: out[m]=pd.to_numeric(df[m],errors='coerce').values if m in df.columns else np.nan
    return out

tgt=load('DBMSTransferLearning/dataset/full_data/112c125g-result.csv')
srcs=pd.concat([load(f) for f in glob.glob('DBMSTransferLearning/dataset/full_data/*-result.csv') if '112c125g' not in f],ignore_index=True)
alld=pd.concat([tgt,srcs],ignore_index=True)
print('target ctxs:',sorted(tgt['ctx'].unique()))
print('source ctxs:',srcs['ctx'].nunique(),'| source hw:',sorted(set(c.split('_')[0] for c in srcs['ctx'])))

# default config_key = ck of id==0 (verify consistent)
dk=tgt[tgt['id']==0]['ck'].mode().iloc[0]
print('default_key:',dk,'| present in all target ctxs:',all((tgt[tgt.ctx==c]['ck']==dk).any() for c in tgt['ctx'].unique()))

# tps pivot on shared grid (per ctx max tps per ck)
tps=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx')
piv=tps.dropna()                          # configs present in ALL ctxs
ctxs=list(piv.columns); shared=list(piv.index)
print('shared grid:',piv.shape[0],'configs x',piv.shape[1],'ctxs')

# metric vectors at default (mean over default-key rows) per ctx
defrow=alld[alld['ck']==dk].groupby('ctx')[METRICS114].mean()
defrow=defrow.reindex(ctxs)
# usable metric cols: numeric & not all-NaN across ctxs
usable=[m for m in METRICS114 if defrow[m].notna().all()]
M114=defrow[usable].to_numpy(float)
M11=defrow[MN].to_numpy(float)
print('usable 114->',len(usable),'metric cols for OT')

# embed + raw on 11 metrics (dml_models_all scaler+model)
base='autotune/optimizer/dml_models_all/'
sc=joblib.load(base+'scaler.pkl'); model=EmbeddingNet(); model.load_state_dict(torch.load(base+'context_model.pth',map_location='cpu')); model.eval()
Xs=sc.transform(M11); emb=model(torch.tensor(Xs,dtype=torch.float32)).detach().numpy()
# OT decile bins on usable-114 (edges from all shared-grid observations)
pop=alld[alld['ck'].isin(set(shared))][usable].apply(pd.to_numeric,errors='coerce').to_numpy(float)
edges=np.nanpercentile(pop,np.linspace(0,100,11),axis=0)
def binv(V):
    B=np.empty_like(V,dtype=int)
    for j in range(V.shape[1]): B[:,j]=np.clip(np.digitize(V[:,j],edges[:,j],right=True),1,10)
    return B
defB=binv(M114)
idx={c:i for i,c in enumerate(ctxs)}
pv=piv; bestcfg={s:pv[s].idxmax() for s in ctxs}; tr=lambda c,s: pv.at[bestcfg[s],c]
def pick(c,cand,m):
    ci=idx[c]; cj=[idx[s] for s in cand]
    if m=='raw': d=cdist(Xs[ci:ci+1],Xs[cj]).flatten()
    elif m=='emb': d=cdist(emb[ci:ci+1],emb[cj]).flatten()
    elif m=='ot': d=cdist(defB[ci:ci+1].astype(float),defB[cj].astype(float)).flatten()
    return cand[int(d.argmin())]

tgt_ctxs=[c for c in ctxs if c.startswith('112c125g_')]
src_ctxs=[c for c in ctxs if not c.startswith('112c125g_')]
def short(c):
    c=c.replace('_64-1000000-4-oltp','').replace('_64-100000-4-oltp','*').replace('-4-4-tpcc-nan','wh-tpcc')
    return c
rows=[]
for c in tgt_ctxs:
    b=pv[c].max(); cand=src_ctxs
    sr,se,so=pick(c,cand,'raw'),pick(c,cand,'emb'),pick(c,cand,'ot')
    rows.append(dict(target=c.replace('112c125g_',''),
        best_tps=round(b,1), default_ratio=round(pv.at[dk,c]/b,3),
        raw=round(tr(c,sr)/b,3), embed=round(tr(c,se)/b,3), ot114=round(tr(c,so)/b,3),
        oracle=round(max(tr(c,s) for s in cand)/b,3),
        embed_src=short(se), ot_src=short(so)))
df=pd.DataFrame(rows)
out='scripts/lab/transfer_quality_miyabi.csv'; df.to_csv(out,index=False)
print('\n================ Miyabi (112c125g) targets — transferred/best ratio ================')
print('   pool = all',len(src_ctxs),'source contexts (7 HW, all ratios); OT uses all',len(usable),'metrics\n')
import shutil
with pd.option_context('display.width',200,'display.max_columns',20):
    print(df.to_string(index=False))
print(f'\nwrote {out}')
