import pandas as pd, numpy as np, glob, os, math, time, warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
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
tgt=sorted(c for c in ctx_tps if c.startswith('112c125g_'))
N=len(src); ctx2i={c:i for i,c in enumerate(src)}
piv=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx').dropna()
bestcfg={s:piv[s].idxmax() for s in piv.columns}; tr=lambda t,s:piv.at[bestcfg[s],t]
DK='1.0|1.0|2.0|1.0|1.0|1.0|1024.0|1.0|25.0|100.0|0.048|4000.0'
print(f'loaded ({time.time()-t0:.0f}s): N={N} sources',flush=True)

# ---- per-context default metric vectors ----
COLS114=[c for c in alld.columns if c not in {'ctx','ck','tps'}]
def_all=alld[alld.ck==DK].groupby('ctx')[COLS114].mean()
src_114_df=def_all.loc[src]; tgt_114_df=def_all.loc[tgt]
usable114=[c for c in COLS114 if src_114_df[c].notna().all() and tgt_114_df[c].notna().all()]
src_M114=src_114_df[usable114].to_numpy(float); tgt_M114=tgt_114_df[usable114].to_numpy(float)
cm=pd.read_csv('DBMSTransferLearning/dataset/context_default_metrics_all.csv').set_index('context_id').loc[src]
src_M11=cm[MN11].to_numpy(float)
tg_def=alld[(alld.ctx.str.startswith('112c125g_'))&(alld.ck==DK)].groupby('ctx')[MN11].mean().reindex(tgt)
tgt_M11=tg_def[MN11].to_numpy(float)

# ---- similarity matrices among sources (N x N) ----
# 1) top-1% overlap
def top_set(c,frac=0.01):
    t=ctx_tps[c]; k=max(1,math.ceil(len(t)*frac))
    return set(x for x,_ in sorted(t.items(),key=lambda x:x[1],reverse=True)[:k])
T1={c:top_set(c) for c in src}
S_top1=np.zeros((N,N))
for i,a in enumerate(src):
    for j,b in enumerate(src):
        if i!=j: S_top1[i,j]=len(T1[a]&T1[b])/max(1,len(T1[a]|T1[b]))
# 2) transfer-rank: T[i,j] = tps_i(bestcfg_j)/max_tps_i — i.e. how well j's argmax transfers to i
piv_arr=piv[src].to_numpy()
ck_index={k:i for i,k in enumerate(piv.index)}
bidx=np.array([ck_index[bestcfg[c]] for c in src])
maxs=piv_arr.max(axis=0)
S_trans=piv_arr[bidx][:,:].T / maxs[:,None]    # shape (N,N): row=i (target), col=j (source)
# symmetrize for triplet selection: avg both directions
S_trans_sym=(S_trans+S_trans.T)/2
np.fill_diagonal(S_trans_sym,0)
# 3) RF-distill: train RF on top-1% overlap with 114-metric pair features, predict on src-src pairs
def feats(a,b): return np.concatenate([np.abs(a-b),(a+b)/2])
Xtr=[]; ytr=[]
for i in range(N):
    for j in range(N):
        if i!=j: Xtr.append(feats(src_M114[i],src_M114[j])); ytr.append(S_top1[i,j])
Xtr=np.array(Xtr); ytr=np.array(ytr)
rf=RandomForestRegressor(n_estimators=300,random_state=0,n_jobs=-1).fit(Xtr,ytr)
preds=rf.predict(Xtr).reshape(N,N-1)
S_rf=np.zeros((N,N))
for i in range(N):
    js=[j for j in range(N) if j!=i]
    for k,j in enumerate(js): S_rf[i,j]=preds[i,k]
np.fill_diagonal(S_rf,0)
# 4) concordance: use existing concordant_pair_ranking.csv as similarity
conc=pd.read_csv('DBMSTransferLearning/dataset/concordant_pair_ranking.csv')
ctx_set=set(src)
S_conc=np.zeros((N,N))
have_conc=set(conc['context_id'])&ctx_set
for _,r in conc.iterrows():
    a,b,s=r['context_id'],r['similar_context_id'],r['similarity_score']
    if a in ctx2i and b in ctx2i: S_conc[ctx2i[a],ctx2i[b]]=float(s)
print(f'similarity matrices built ({time.time()-t0:.0f}s)',flush=True)

# ---- triplet construction from similarity matrix ----
def build_trips(S,K=5):
    trips=[]
    for i in range(N):
        sc=S[i].copy(); sc[i]=-np.inf
        order=np.argsort(sc)[::-1]
        pos=order[:K]; neg=order[-K:]
        for p in pos:
            for n in neg: trips.append((i,p,n))
    return trips

# ---- embedding net + train/eval ----
class Net(nn.Module):
    def __init__(s,d,emb=16,h1=64,h2=32):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(d,h1),nn.ReLU(),nn.Linear(h1,h2),nn.ReLU(),nn.Linear(h2,emb))
    def forward(s,x): return nn.functional.normalize(s.net(x),p=2,dim=1)

def train_eval(Mat_src,Mat_tgt,trips,seed,epochs=50,h1=64,h2=32):
    torch.manual_seed(seed); np.random.seed(seed)
    sc=MinMaxScaler().fit(Mat_src)
    Xs=sc.transform(Mat_src); Xt=sc.transform(Mat_tgt)
    d=Xs.shape[1]
    A=np.array([Xs[a] for a,_,_ in trips],np.float32)
    P=np.array([Xs[p] for _,p,_ in trips],np.float32)
    Ng=np.array([Xs[n] for _,_,n in trips],np.float32)
    class DS(Dataset):
        def __len__(s): return len(A)
        def __getitem__(s,i): return torch.tensor(A[i]),torch.tensor(P[i]),torch.tensor(Ng[i])
    dl=DataLoader(DS(),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m=Net(d,h1=h1,h2=h2); opt=optim.Adam(m.parameters(),lr=1e-3); crit=nn.TripletMarginLoss(margin=1.0,p=2)
    for _ in range(epochs):
        m.train()
        for a,p,n in dl: opt.zero_grad(); crit(m(a),m(p),m(n)).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Es=m(torch.tensor(Xs,dtype=torch.float32)).numpy()
        Et=m(torch.tensor(Xt,dtype=torch.float32)).numpy()
    ratios=[]
    for ti,t in enumerate(tgt):
        j=int(cdist(Et[ti:ti+1],Es).argmin()); ratios.append(tr(t,src[j])/piv[t].max())
    return float(np.mean(ratios)),ratios

# ---- run grid of experiments (5 seeds each) ----
exps=[
    ('embed 11D, concordance signal',           src_M11,tgt_M11,S_conc,64,32),
    ('embed 114D, concordance signal',          src_M114,tgt_M114,S_conc,64,32),
    ('embed 114D, top-1% overlap signal',       src_M114,tgt_M114,S_top1,64,32),
    ('embed 114D, transfer-rank signal',        src_M114,tgt_M114,S_trans_sym,64,32),
    ('embed 114D, RF-distilled signal',         src_M114,tgt_M114,S_rf,64,32),
    ('embed 114D, RF-distill + WIDE 128-64-16', src_M114,tgt_M114,S_rf,128,64),
]
SEEDS=range(5)
results={}
for name,Ms,Mt,S,h1,h2 in exps:
    trips=build_trips(S,K=5)
    means=[]
    for s in SEEDS:
        mn,_=train_eval(Ms,Mt,trips,s,h1=h1,h2=h2); means.append(mn)
    results[name]=np.array(means)
    print(f'  {name:48s}: {results[name].mean():.3f} ± {results[name].std():.3f}   '
          f'range [{results[name].min():.3f},{results[name].max():.3f}]   ({time.time()-t0:.0f}s)',flush=True)

print('\n===== summary: pushing the embedding toward the RF/oracle ceiling =====')
print('  baseline (existing)')
print(f'    embed 11D concordance reseeded:           0.801 ± 0.041')
print(f'    embed 9D  concordance reseeded no-IOPS:   0.837 ± 0.061')
print(f'    embed 11D top-1% signal:                  0.759 ± 0.064')
print(f'    embed 9D  top-1% signal no-IOPS:          0.810 ± 0.072')
print('  new')
for name,arr in results.items():
    print(f'    {name:42s}: {arr.mean():.3f} ± {arr.std():.3f}')
print('  fixed reference')
print(f'    RF 114 metrics                            0.910')
print(f'    top-1% overlap DIRECT (full sweeps)       0.923')
print(f'    ORACLE                                    0.994')
print(f'\ndone ({time.time()-t0:.0f}s)')
