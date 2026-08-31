import pandas as pd, numpy as np, glob, os, math, time, warnings, re
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch, joblib, torch.nn as nn, torch.optim as optim
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
alld=pd.concat([load(f) for f in sorted(glob.glob('DBMSTransferLearning/dataset/full_data/*-result.csv'))],ignore_index=True)
ctx_tps={c:g.groupby('ck')['tps'].max().to_dict() for c,g in alld.groupby('ctx')}
src=sorted(c for c in ctx_tps if not c.startswith('112c125g_'))
N=len(src); ctx2i={c:i for i,c in enumerate(src)}
piv=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx').dropna()
bestcfg={s:piv[s].idxmax() for s in piv.columns}
DK='1.0|1.0|2.0|1.0|1.0|1.0|1024.0|1.0|25.0|100.0|0.048|4000.0'
COLS114=[c for c in alld.columns if c not in {'ctx','ck','tps'}]
def_all=alld[alld.ck==DK].groupby('ctx')[COLS114].mean()
src_df=def_all.loc[src]
usable114=[c for c in COLS114 if src_df[c].notna().all()]
src_M114=src_df[usable114].to_numpy(float)
cm=pd.read_csv('DBMSTransferLearning/dataset/context_default_metrics_all.csv').set_index('context_id').loc[src]
src_M11=cm[MN11].to_numpy(float)

# workload signature = context_id minus hardware prefix
def wsig(c): p=c.split('_',1); return p[1] if len(p)==2 else c
def fam(c):
    m=re.search(r'oltp_(read_only|read_write_\d+|write_only)|tpcc',c); return m.group(0) if m else '?'
workloads=sorted(set(wsig(c) for c in src))
wl_to_ctxs={w:[i for i in range(N) if wsig(src[i])==w] for w in workloads}
print(f'loaded ({time.time()-t0:.0f}s) | N={N}, {len(workloads)} unique workloads',flush=True)

# Top-K% Jaccard
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

# Concordance
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
print(f'sim matrices ready ({time.time()-t0:.0f}s)',flush=True)

# RF pair features
def feats(a,b): return np.concatenate([np.abs(a-b),(a+b)/2])
pair_i=[]; pair_j=[]; X_full=[]; y_full=[]
for i in range(N):
    for j in range(N):
        if i==j: continue
        pair_i.append(i); pair_j.append(j)
        X_full.append(feats(src_M114[i],src_M114[j])); y_full.append(S_top1[i,j])
pair_i=np.array(pair_i); pair_j=np.array(pair_j)
X_full=np.array(X_full,dtype=np.float32); y_full=np.array(y_full,dtype=np.float32)

conc_rank=pd.read_csv('DBMSTransferLearning/dataset/concordant_pair_ranking.csv')

class EmbeddingNet(nn.Module):
    def __init__(s,i=11,e=16):
        super().__init__(); s.net=nn.Sequential(nn.Linear(i,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,e))
    def forward(s,x): return F.normalize(s.net(x),p=2,dim=1)

def train_embed_for_lowo(non_excluded_set, seed=42, K=5, epochs=50):
    sub=conc_rank[(conc_rank.context_id.isin(non_excluded_set))&(conc_rank.similar_context_id.isin(non_excluded_set))]
    trips=[]
    for anchor,g in sub.groupby('context_id'):
        g=g.sort_values('similarity_score',ascending=False)
        pos=g.head(K)['similar_context_id'].tolist()
        neg=g.tail(K)['similar_context_id'].tolist()
        for p in pos:
            for n in neg: trips.append((anchor,p,n))
    sub_cm=cm.loc[list(non_excluded_set)]
    sc=MinMaxScaler().fit(sub_cm[MN11])
    cm_scaled=sc.transform(cm[MN11])
    A=np.array([cm_scaled[ctx2i[a]] for a,_,_ in trips],np.float32)
    P=np.array([cm_scaled[ctx2i[p]] for _,p,_ in trips],np.float32)
    Ng=np.array([cm_scaled[ctx2i[n]] for _,_,n in trips],np.float32)
    class DS(Dataset):
        def __len__(s_): return len(A)
        def __getitem__(s_,i): return torch.tensor(A[i]),torch.tensor(P[i]),torch.tensor(Ng[i])
    torch.manual_seed(seed); np.random.seed(seed)
    dl=DataLoader(DS(),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m=EmbeddingNet(); opt=optim.Adam(m.parameters(),lr=1e-3); crit=nn.TripletMarginLoss(margin=1.0,p=2)
    for _ in range(epochs):
        m.train()
        for a,p,n in dl: opt.zero_grad(); crit(m(a),m(p),m(n)).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        E_all=m(torch.tensor(cm_scaled,dtype=torch.float32)).numpy()
    return E_all, len(trips)

def make_raw_scaled(non_excluded_set):
    sub=cm.loc[list(non_excluded_set)]
    sc=MinMaxScaler().fit(sub[MN11])
    return sc.transform(cm[MN11])

shared_keys=set(piv.index)
def compute_ot_bins(non_excluded_set):
    rows=alld[alld.ctx.isin(non_excluded_set)&alld.ck.isin(shared_keys)][usable114].dropna().to_numpy(float)
    edges=np.percentile(rows,np.linspace(0,100,11),axis=0)
    def binv(V):
        B=np.empty_like(V,dtype=int)
        for j in range(V.shape[1]): B[:,j]=np.clip(np.digitize(V[:,j],edges[:,j],right=True),1,10)
        return B
    return binv(src_M114).astype(float)

# ---- LOWO ----
print(f'\nstarting LOWO over {len(workloads)} workloads ({time.time()-t0:.0f}s)',flush=True)
rows=[]
for wi,w in enumerate(workloads):
    held_idx=set(wl_to_ctxs[w])
    cand_idx=[i for i in range(N) if i not in held_idx]
    cand_a=np.array(cand_idx)
    cand_ctx=set(src[i] for i in cand_idx)
    # retrain
    mask=np.isin(pair_i,cand_idx)&np.isin(pair_j,cand_idx)
    rf=RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=-1).fit(X_full[mask],y_full[mask])
    E_lowo,n_trips=train_embed_for_lowo(cand_ctx)
    X_raw_lowo=make_raw_scaled(cand_ctx)
    src_B114_lowo=compute_ot_bins(cand_ctx)
    print(f'  [{wi+1:2d}/{len(workloads)}] workload={w[:45]:45s} held={len(held_idx):2d} cand={len(cand_idx)}  ({time.time()-t0:.0f}s)',flush=True)
    for ci in wl_to_ctxs[w]:
        held=src[ci]; b=piv[held].max()
        def ratio_of(pick_idx): return piv.at[bestcfg[src[pick_idx]],held]/b
        def_r=piv.at[DK,held]/b
        pick_raw=cand_idx[int(cdist(X_raw_lowo[ci:ci+1],X_raw_lowo[cand_a]).argmin())]
        pick_emb=cand_idx[int(cdist(E_lowo[ci:ci+1],E_lowo[cand_a]).argmin())]
        pick_ot=cand_idx[int(cdist(src_B114_lowo[ci:ci+1],src_B114_lowo[cand_a]).argmin())]
        pick_conc=cand_idx[int(np.argmax(S_conc[ci,cand_a]))]
        pick_t5=cand_idx[int(np.argmax(S_top5[ci,cand_a]))]
        pick_t1=cand_idx[int(np.argmax(S_top1[ci,cand_a]))]
        X_test=np.array([feats(src_M114[ci],src_M114[s]) for s in cand_idx],dtype=np.float32)
        pick_rf=cand_idx[int(rf.predict(X_test).argmax())]
        or_ratios=[piv.at[bestcfg[src[s]],held]/b for s in cand_idx]
        pick_or=cand_idx[int(np.argmax(or_ratios))]
        rows.append(dict(held_out=held,workload=w,family=fam(held),
            default=def_r,raw=ratio_of(pick_raw),embed=ratio_of(pick_emb),
            concordance=ratio_of(pick_conc),ot114=ratio_of(pick_ot),
            top5_direct=ratio_of(pick_t5),top1_direct=ratio_of(pick_t1),
            rf=ratio_of(pick_rf),oracle=or_ratios[cand_idx.index(pick_or)]))

df=pd.DataFrame(rows)
df.to_csv('scripts/lab/lowo_all_methods.csv',index=False)
print(f'\nsaved scripts/lab/lowo_all_methods.csv ({time.time()-t0:.0f}s)\n')
cols=['default','raw','embed','concordance','ot114','top5_direct','top1_direct','rf','oracle']
print(f'=== LOWO mean transferred/best over {N} held-out contexts ({len(workloads)} workloads) ===')
print(f'  {"method":18s}{"mean":>8s}{"std":>8s}')
for c in cols: print(f'  {c:18s}{df[c].mean():8.3f}{df[c].std():8.3f}')
print(f'\n=== mean by family (each row aggregates LOWO folds within that family) ===')
fam_means=df.groupby('family')[cols].mean().round(3); fam_means.insert(0,'n',df.groupby('family').size())
print(fam_means.to_string())
print(f'\ndone ({time.time()-t0:.0f}s)')
