import pandas as pd, numpy as np, glob, os, math, time, warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
t0=time.time()
PARAMS=['innodb_buffer_pool_size','innodb_read_io_threads','innodb_write_io_threads','innodb_flush_log_at_trx_commit','innodb_adaptive_hash_index','sync_binlog','innodb_lru_scan_depth','innodb_buffer_pool_instances','innodb_change_buffer_max_size','innodb_io_capacity','innodb_log_file_size','table_open_cache']
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
COLS114=[c for c in alld.columns if c not in {'ctx','ck','tps'}]
def_all=alld[alld.ck==DK].groupby('ctx')[COLS114].mean()
src_df=def_all.loc[src]; tgt_df=def_all.loc[tgt]
usable114=[c for c in COLS114 if src_df[c].notna().all() and tgt_df[c].notna().all()]
src_M114=src_df[usable114].to_numpy(float); tgt_M114=tgt_df[usable114].to_numpy(float)
print(f'loaded ({time.time()-t0:.0f}s) | N={N}, dim_usable={len(usable114)}',flush=True)

# Top-1% Jaccard
def top_set(c,frac=0.01):
    t=ctx_tps[c]; k=max(1,math.ceil(len(t)*frac))
    return set(x for x,_ in sorted(t.items(),key=lambda x:x[1],reverse=True)[:k])
T1={c:top_set(c) for c in src}
S_top1=np.zeros((N,N))
for i,a in enumerate(src):
    for j,b in enumerate(src):
        if i!=j: S_top1[i,j]=len(T1[a]&T1[b])/max(1,len(T1[a]|T1[b]))
print(f'S_top1 ready ({time.time()-t0:.0f}s)',flush=True)

# Train RF on 114-metric pair features for feature importance
def feats(a,b): return np.concatenate([np.abs(a-b),(a+b)/2])
Xtr=[]; ytr=[]
for i in range(N):
    for j in range(N):
        if i!=j: Xtr.append(feats(src_M114[i],src_M114[j])); ytr.append(S_top1[i,j])
Xtr=np.array(Xtr); ytr=np.array(ytr)
rf=RandomForestRegressor(n_estimators=300,random_state=0,n_jobs=-1).fit(Xtr,ytr)
d=src_M114.shape[1]
per_metric_imp=np.array([rf.feature_importances_[k]+rf.feature_importances_[d+k] for k in range(d)])
ranked=sorted(zip(usable114,per_metric_imp),key=lambda x:x[1],reverse=True)
print(f'\nTOP 25 metrics by RF importance:',flush=True)
for m,imp in ranked[:25]: print(f'  {imp:.4f}  {m[:80]}')
del Xtr,ytr

# Select top-N feature subsets
def select(N_feats):
    sel=[m for m,_ in ranked[:N_feats]]
    return src_df[sel].to_numpy(float), tgt_df[sel].to_numpy(float), sel
src_M8,tgt_M8,names8=select(8)
src_M16,tgt_M16,names16=select(16)
print(f'\nTop-8 selected:  {names8}')
print(f'Top-16 selected: {names16[:5]} ... (+{len(names16)-5} more)')

def build_trips(S,K):
    trips=[]
    for i in range(N):
        sc=S[i].copy(); sc[i]=-np.inf; order=np.argsort(sc)[::-1]
        for p in order[:K]:
            for n in order[-K:]: trips.append((i,p,n))
    return trips
def build_hard_trips(S,E,K=10,low_sim_frac=0.5):
    """Hard negatives: low-similarity sources that are CLOSE in the current embedding."""
    trips=[]
    for i in range(N):
        sc=S[i].copy(); sc[i]=-np.inf
        order=np.argsort(sc)[::-1]
        positives=order[:K]
        # candidates: bottom half by similarity (excluding self)
        thresh=np.quantile(S[i],low_sim_frac)
        low_sim=np.where(S[i]<=thresh)[0]
        low_sim=low_sim[low_sim!=i]
        if len(low_sim)>=K:
            dists=np.linalg.norm(E[low_sim]-E[i],axis=1)
            hard_neg=low_sim[np.argsort(dists)[:K]]
        else:
            hard_neg=low_sim
        for p in positives:
            for n in hard_neg: trips.append((i,p,n))
    return trips

class Net(nn.Module):
    def __init__(s,d,h_sizes,dropout=0.3):
        super().__init__(); layers=[]; prev=d
        for h in h_sizes[:-1]:
            layers.append(nn.Linear(prev,h)); layers.append(nn.ReLU())
            if dropout>0: layers.append(nn.Dropout(dropout))
            prev=h
        layers.append(nn.Linear(prev,h_sizes[-1])); s.net=nn.Sequential(*layers)
    def forward(s,x): return F.normalize(s.net(x),p=2,dim=1)

def _train_one_pass(m,opt,crit,Xs,trips,batch=64,seed=0):
    A=np.array([Xs[a] for a,_,_ in trips],np.float32)
    P=np.array([Xs[p] for _,p,_ in trips],np.float32)
    Ng=np.array([Xs[n] for _,_,n in trips],np.float32)
    class DS(Dataset):
        def __len__(s_): return len(A)
        def __getitem__(s_,i): return torch.tensor(A[i]),torch.tensor(P[i]),torch.tensor(Ng[i])
    dl=DataLoader(DS(),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m.train()
    for a,p,n in dl: opt.zero_grad(); crit(m(a),m(p),m(n)).backward(); opt.step()

def train_eval(Ms,Mt,seed,K=10,h_sizes=[128,64,16],hard_mine=False,
               warmup_epochs=10,mining_epochs=15,total_epochs=25,
               dropout=0.3,wd=1e-4,lr=1e-3,margin=1.0,clip=True):
    torch.manual_seed(seed); np.random.seed(seed)
    sc=MinMaxScaler().fit(Ms); Xs=sc.transform(Ms); Xt=sc.transform(Mt)
    if clip: Xs=np.clip(Xs,0,1); Xt=np.clip(Xt,0,1)
    d=Xs.shape[1]
    m=Net(d,h_sizes,dropout=dropout); opt=optim.Adam(m.parameters(),lr=lr,weight_decay=wd)
    crit=nn.TripletMarginLoss(margin=margin,p=2)
    if not hard_mine:
        trips=build_trips(S_top1,K)
        for ep in range(total_epochs):
            _train_one_pass(m,opt,crit,Xs,trips,seed=seed+ep)
    else:
        # Phase 1: warmup with static triplets
        trips=build_trips(S_top1,K)
        for ep in range(warmup_epochs):
            _train_one_pass(m,opt,crit,Xs,trips,seed=seed+ep)
        # Mine hard negatives every 5 epochs in phase 2
        for ep in range(mining_epochs):
            if ep%5==0:
                m.eval()
                with torch.no_grad():
                    E=m(torch.tensor(Xs,dtype=torch.float32)).numpy()
                trips=build_hard_trips(S_top1,E,K=K,low_sim_frac=0.5)
            _train_one_pass(m,opt,crit,Xs,trips,seed=seed+warmup_epochs+ep)
    m.eval()
    with torch.no_grad():
        Es=m(torch.tensor(Xs,dtype=torch.float32)).numpy()
        Et=m(torch.tensor(Xt,dtype=torch.float32)).numpy()
    return float(np.mean([tr(t,src[int(cdist(Et[i:i+1],Es).argmin())])/piv[t].max() for i,t in enumerate(tgt)]))

SEEDS=[0,1,2,3]
exps=[
    ('A: 114D baseline + recipe (K=10, no mining)',          src_M114,tgt_M114,[256,128,16],False),
    ('B: top-8 features + recipe (K=10, no mining)',         src_M8, tgt_M8, [128,64,16], False),
    ('C: top-16 features + recipe (K=10, no mining)',        src_M16,tgt_M16,[128,64,16], False),
    ('D: top-8 features + recipe + HARD MINING (K=10)',      src_M8, tgt_M8, [128,64,16], True),
    ('E: top-16 features + recipe + HARD MINING (K=10)',     src_M16,tgt_M16,[128,64,16], True),
    ('F: 114D + recipe + HARD MINING (K=10)',                src_M114,tgt_M114,[256,128,16],True),
]
results={}
for name,Ms,Mt,h,mine in exps:
    arr=[]
    for s in SEEDS:
        v=train_eval(Ms,Mt,s,h_sizes=h,hard_mine=mine); arr.append(v)
        print(f'    {name[:60]:60s} seed {s}: {v:.3f}   ({time.time()-t0:.0f}s)',flush=True)
    results[name]=np.array(arr)

print('\n===== Feature selection + Hard negative mining for TOP-1% embedding =====')
print('  prior reference (best top-1% embedding):')
print(f'    F: 114D top-1% recipe K=10+clip (5 seeds): 0.829 ± 0.049')
print(f'    V0 (recent 4 seeds): 0.808 ± 0.029')
print(f'    OT 114: 0.845 | RF 114: 0.910 | top-1% DIRECT: 0.923 | ORACLE: 0.994\n')
print('  this run (4 seeds each):')
for name,arr in results.items():
    print(f'    {name:60s}: {arr.mean():.3f} ± {arr.std():.3f}  range [{arr.min():.3f},{arr.max():.3f}]')
print(f'\ndone ({time.time()-t0:.0f}s)')
