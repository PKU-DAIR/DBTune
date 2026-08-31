import pandas as pd, numpy as np, glob, os, math, time, warnings, re
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import torch, joblib, torch.nn as nn
import torch.nn.functional as F
t0=time.time()

PARAMS=['innodb_buffer_pool_size','innodb_read_io_threads','innodb_write_io_threads','innodb_flush_log_at_trx_commit','innodb_adaptive_hash_index','sync_binlog','innodb_lru_scan_depth','innodb_buffer_pool_instances','innodb_change_buffer_max_size','innodb_io_capacity','innodb_log_file_size','table_open_cache']
MN11=['Average Memory Usage Percentage','InnoDB Buffer Pool Cache Hit Rate','InnoDB Dirty Buffer Pages','Current QPS (Queries Per Second)','Max CPU Usage (100 - Idle)','InnoDB Rows Deleted (60s Rate)','InnoDB Rows Inserted (60s Rate)','InnoDB Rows Read (60s Rate)','InnoDB Rows Updated (60s Rate)','Average Disk IOPS (Read)','Average Disk IOPS (Write)']
def nf(v):
    s=str(v).strip()
    if s in('ON','Yes','True'):return 1.0
    if s in('OFF','No','False'):return 0.0
    if s.endswith('GB'):return float(s[:-2])
    if s.endswith('MB'):return float(s[:-2])/1000.0
    if s.endswith('KB'):return float(s[:-2])/1e6
    try:return float(s)
    except:return np.nan
def load(f):
    df=pd.read_csv(f);df=df.loc[:,~df.columns.duplicated()];hw=os.path.basename(f).replace('-result.csv','')
    K=np.array([[nf(x) for x in df[p]] for p in PARAMS]).T
    ok=np.isfinite(K).all(1)&pd.to_numeric(df['tps'],errors='coerce').notna().values
    df=df[ok].reset_index(drop=True);K=K[ok];ck=['|'.join(str(round(v,6)) for v in row) for row in K]
    if 'workload_label' in df.columns: wl=df['workload_label'].astype(str)
    else:
        sk=df['skew'].apply(lambda x:'nan' if pd.isna(x) else (str(int(x)) if float(x)==int(x) else str(x)))
        wl=(df['num_table'].astype(int).astype(str)+'-'+df['table_size'].astype(int).astype(str)+'-'+df['num_client'].astype(int).astype(str)+'-'+df['workload'].astype(str)+'-'+sk)
    out=pd.DataFrame({'ctx':hw+'_'+wl,'ck':ck,'tps':pd.to_numeric(df['tps'],errors='coerce').values})
    cols114=list(pd.read_csv(f,nrows=0).columns)[19:]
    for m in cols114: out[m]=pd.to_numeric(df[m],errors='coerce').values if m in df.columns else np.nan
    return out
files=sorted(glob.glob('DBMSTransferLearning/dataset/full_data/*-result.csv'))
alld=pd.concat([load(f) for f in files],ignore_index=True)
ctx_tps={c:g.groupby('ck')['tps'].max().to_dict() for c,g in alld.groupby('ctx')}
src=sorted(c for c in ctx_tps if not c.startswith('112c125g_'))
N=len(src); ctx2i={c:i for i,c in enumerate(src)}
piv=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx').dropna()
bestcfg={s:piv[s].idxmax() for s in piv.columns}
DK='1.0|1.0|2.0|1.0|1.0|1.0|1024.0|1.0|25.0|100.0|0.048|4000.0'
print(f'loaded ({time.time()-t0:.0f}s) | N={N} source contexts',flush=True)

# default metric vectors
COLS114=[c for c in alld.columns if c not in {'ctx','ck','tps'}]
def_all=alld[alld.ck==DK].groupby('ctx')[COLS114].mean()
src_df=def_all.loc[src]
usable114=[c for c in COLS114 if src_df[c].notna().all()]
src_M114=src_df[usable114].to_numpy(float)
cm=pd.read_csv('DBMSTransferLearning/dataset/context_default_metrics_all.csv').set_index('context_id').loc[src]
src_M11=cm[MN11].to_numpy(float)

# top-K% Jaccard matrices (precomputed once — same population for LOOCV; per-context sets are independent of others)
def top_set(c,frac):
    t=ctx_tps[c]; k=max(1,math.ceil(len(t)*frac))
    return set(x for x,_ in sorted(t.items(),key=lambda x:x[1],reverse=True)[:k])
T1={c:top_set(c,0.01) for c in src}
T5={c:top_set(c,0.05) for c in src}
S_top1=np.zeros((N,N)); S_top5=np.zeros((N,N))
for i,a in enumerate(src):
    for j,b in enumerate(src):
        if i==j: continue
        S_top1[i,j]=len(T1[a]&T1[b])/max(1,len(T1[a]|T1[b]))
        S_top5[i,j]=len(T5[a]&T5[b])/max(1,len(T5[a]|T5[b]))
print(f'top-K% matrices ready ({time.time()-t0:.0f}s)',flush=True)

# Concordance (Kendall-tau on pairwise shared configs)
S_conc=np.full((N,N),-2.0)
for i in range(N):
    a_tps=ctx_tps[src[i]]; a_set=set(a_tps)
    for j in range(N):
        if i==j: continue
        b_tps=ctx_tps[src[j]]; shared=a_set&set(b_tps)
        if len(shared)<10: continue
        x=np.fromiter((a_tps[k] for k in shared),float,len(shared))
        y=np.fromiter((b_tps[k] for k in shared),float,len(shared))
        tau,_=kendalltau(x,y)
        if not np.isnan(tau): S_conc[i,j]=tau
print(f'concordance ready ({time.time()-t0:.0f}s)',flush=True)

# OT decile bins (global edges from all observations on shared grid)
shared_keys=set(piv.index)
all_rows=alld[alld['ck'].isin(shared_keys)][usable114].dropna().to_numpy(float)
edges=np.percentile(all_rows,np.linspace(0,100,11),axis=0)
def binv(V):
    B=np.empty_like(V,dtype=int)
    for j in range(V.shape[1]): B[:,j]=np.clip(np.digitize(V[:,j],edges[:,j],right=True),1,10)
    return B
src_B114=binv(src_M114).astype(float)
print(f'OT bins ready ({time.time()-t0:.0f}s)',flush=True)

# Embedding (deployed dml_models_all)  --  NOTE: model trained on all 293; leaky.
class EmbeddingNet(nn.Module):
    def __init__(s,i=11,e=16):
        super().__init__(); s.net=nn.Sequential(nn.Linear(i,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,e))
    def forward(s,x): return F.normalize(s.net(x),p=2,dim=1)
base='autotune/optimizer/dml_models_all/'
emb_sc=joblib.load(base+'scaler.pkl')
emb_model=EmbeddingNet(); emb_model.load_state_dict(torch.load(base+'context_model.pth',map_location='cpu')); emb_model.eval()
with torch.no_grad():
    E_src=emb_model(torch.tensor(emb_sc.transform(src_M11),dtype=torch.float32)).numpy()

# raw NN: MinMax-scale 11 metrics
sc_raw=MinMaxScaler().fit(src_M11)
X_raw=sc_raw.transform(src_M11)

# RF pair feature matrix (proper LOOCV)
def feats(a,b): return np.concatenate([np.abs(a-b),(a+b)/2])
pair_i=[]; pair_j=[]; X_full=[]; y_full=[]
for i in range(N):
    for j in range(N):
        if i==j: continue
        pair_i.append(i); pair_j.append(j)
        X_full.append(feats(src_M114[i],src_M114[j]))
        y_full.append(S_top1[i,j])
pair_i=np.array(pair_i); pair_j=np.array(pair_j)
X_full=np.array(X_full,dtype=np.float32); y_full=np.array(y_full,dtype=np.float32)
print(f'RF pair-features ready: {X_full.shape} ({time.time()-t0:.0f}s)',flush=True)

def fam(c):
    m=re.search(r'oltp_(read_only|read_write_\d+|write_only)|tpcc',c); return m.group(0) if m else '?'

# ---- LOOCV ----
print(f'\nstarting LOOCV over {N} contexts ({time.time()-t0:.0f}s)',flush=True)
rows=[]
for ci in range(N):
    held=src[ci]; b=piv[held].max()
    cand=[j for j in range(N) if j!=ci]; cand_a=np.array(cand)
    def ratio_of(pick_idx): return piv.at[bestcfg[src[pick_idx]],held]/b
    def_r=piv.at[DK,held]/b
    pick_raw=cand[int(cdist(X_raw[ci:ci+1],X_raw[cand_a]).argmin())]
    pick_emb=cand[int(cdist(E_src[ci:ci+1],E_src[cand_a]).argmin())]
    pick_ot=cand[int(cdist(src_B114[ci:ci+1],src_B114[cand_a]).argmin())]
    pick_conc=cand[int(np.argmax(S_conc[ci,cand_a]))]
    pick_t5=cand[int(np.argmax(S_top5[ci,cand_a]))]
    pick_t1=cand[int(np.argmax(S_top1[ci,cand_a]))]
    # RF: exclude all pairs touching ci, retrain
    mask=(pair_i!=ci)&(pair_j!=ci)
    rf=RandomForestRegressor(n_estimators=100,random_state=0,n_jobs=-1).fit(X_full[mask],y_full[mask])
    X_test=np.array([feats(src_M114[ci],src_M114[s]) for s in cand],dtype=np.float32)
    pick_rf=cand[int(rf.predict(X_test).argmax())]
    # oracle: argmax actual transferred ratio over candidates
    or_ratios=[piv.at[bestcfg[src[s]],held]/b for s in cand]
    pick_or=cand[int(np.argmax(or_ratios))]
    rows.append(dict(held_out=held,family=fam(held),
        default=def_r,raw=ratio_of(pick_raw),embed=ratio_of(pick_emb),
        concordance=ratio_of(pick_conc),ot114=ratio_of(pick_ot),
        top5_direct=ratio_of(pick_t5),top1_direct=ratio_of(pick_t1),
        rf=ratio_of(pick_rf),oracle=or_ratios[cand.index(pick_or)]))
    if (ci+1)%20==0:
        print(f'  LOOCV {ci+1}/{N} ({time.time()-t0:.0f}s)',flush=True)

df=pd.DataFrame(rows)
df.to_csv('scripts/lab/loocv_all_methods.csv',index=False)
print(f'\nsaved scripts/lab/loocv_all_methods.csv ({time.time()-t0:.0f}s)\n')

cols=['default','raw','embed','concordance','ot114','top5_direct','top1_direct','rf','oracle']
print(f'=== LOOCV mean transferred/best over {N} held-out source contexts ===')
print(f'  {"method":18s}{"mean":>8s}{"std":>8s}')
for c in cols: print(f'  {c:18s}{df[c].mean():8.3f}{df[c].std():8.3f}')

print(f'\n=== mean by held-out workload family ===')
fam_means=df.groupby('family')[cols].mean().round(3)
fam_counts=df.groupby('family').size()
fam_means.insert(0,'n',fam_counts)
print(fam_means.to_string())
print('\n  caveat: embed uses the deployed dml_models_all model which was trained on all 293 sources')
print('  -> embed row contains soft leakage (held-out context\'s training-time location is informed)')
