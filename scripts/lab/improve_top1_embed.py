import pandas as pd, numpy as np, glob, os, math, time, warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
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
src_M=src_df[usable114].to_numpy(float); tgt_M=tgt_df[usable114].to_numpy(float)
print(f'loaded ({time.time()-t0:.0f}s) | N={N}, dim={src_M.shape[1]}',flush=True)

# Top-K% sets (with rank weights) per context
def topk_with_weights(c,frac):
    t=ctx_tps[c]; n=len(t); k=max(1,math.ceil(n*frac))
    sortedl=sorted(t.items(),key=lambda x:x[1],reverse=True)[:k]
    # weight by rank: best=1, worst -> 1/k roughly
    return {cfg:1.0-(i/k) for i,(cfg,_) in enumerate(sortedl)}    # in [1/k, 1]
def topk_set(c,frac): return set(topk_with_weights(c,frac).keys())

T1_w={c:topk_with_weights(c,0.01) for c in src}
T3_w={c:topk_with_weights(c,0.03) for c in src}
T5_w={c:topk_with_weights(c,0.05) for c in src}

def jaccard_sets(A,B):
    if not A and not B: return 0.0
    return len(A&B)/max(1,len(A|B))
def weighted_jaccard(Aw,Bw):
    keys=set(Aw)|set(Bw)
    if not keys: return 0.0
    num=0.0; den=0.0
    for k in keys:
        a=Aw.get(k,0.0); b=Bw.get(k,0.0)
        num+=min(a,b); den+=max(a,b)
    return num/max(1e-12,den)

S_top1=np.zeros((N,N))    # pure top-1% Jaccard (binary)
S_top1_w=np.zeros((N,N))  # weighted top-1%
S_multiK=np.zeros((N,N))  # avg of top-1%, top-3%, top-5% (binary Jaccards)
for i,a in enumerate(src):
    for j,b in enumerate(src):
        if i==j: continue
        S_top1[i,j]=jaccard_sets(set(T1_w[a]),set(T1_w[b]))
        S_top1_w[i,j]=weighted_jaccard(T1_w[a],T1_w[b])
        S_multiK[i,j]=(jaccard_sets(set(T1_w[a]),set(T1_w[b]))+
                       jaccard_sets(set(T3_w[a]),set(T3_w[b]))+
                       jaccard_sets(set(T5_w[a]),set(T5_w[b])))/3.0
print(f'sim matrices ready ({time.time()-t0:.0f}s)',flush=True)
print(f'  S_top1   : mean={S_top1.mean():.3f}, nonzero pairs={int((S_top1>0).sum())}/{N*(N-1)}')
print(f'  S_top1_w : mean={S_top1_w.mean():.3f}, nonzero={int((S_top1_w>0).sum())}/{N*(N-1)}')
print(f'  S_multiK : mean={S_multiK.mean():.3f}, nonzero={int((S_multiK>0).sum())}/{N*(N-1)}')

def build_trips(S,K):
    trips=[]
    for i in range(N):
        sc=S[i].copy(); sc[i]=-np.inf
        order=np.argsort(sc)[::-1]
        for p in order[:K]:
            for n in order[-K:]: trips.append((i,p,n))
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

def train_triplet(Ms,Mt,S_target,seed,K=10,h=[256,128,16],dropout=0.3,wd=1e-4,
                  clip=True,epochs=25,batch=64,lr=1e-3,margin=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    sc=MinMaxScaler().fit(Ms); Xs=sc.transform(Ms); Xt=sc.transform(Mt)
    if clip: Xs=np.clip(Xs,0,1); Xt=np.clip(Xt,0,1)
    trips=build_trips(S_target,K); d=Xs.shape[1]
    A=np.array([Xs[a] for a,_,_ in trips],np.float32)
    P=np.array([Xs[p] for _,p,_ in trips],np.float32)
    Ng=np.array([Xs[n] for _,_,n in trips],np.float32)
    class DS(Dataset):
        def __len__(s_): return len(A)
        def __getitem__(s_,i): return torch.tensor(A[i]),torch.tensor(P[i]),torch.tensor(Ng[i])
    dl=DataLoader(DS(),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m=Net(d,h,dropout=dropout); opt=optim.Adam(m.parameters(),lr=lr,weight_decay=wd); crit=nn.TripletMarginLoss(margin=margin,p=2)
    for _ in range(epochs):
        m.train()
        for a,p,n in dl: opt.zero_grad(); crit(m(a),m(p),m(n)).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Es=m(torch.tensor(Xs,dtype=torch.float32)).numpy()
        Et=m(torch.tensor(Xt,dtype=torch.float32)).numpy()
    return float(np.mean([tr(t,src[int(cdist(Et[i:i+1],Es).argmin())])/piv[t].max() for i,t in enumerate(tgt)]))

def train_regression(Ms,Mt,S_target,seed,h=[256,128,16],dropout=0.3,wd=1e-4,clip=True,
                     epochs=25,batch=256,lr=1e-3):
    """Predict pairwise top-1% overlap directly: target = cos_sim of unit-sphere embedding ≈ S_target.
       Uses every (i,j), i≠j as supervised pair (≈85K), not triplets."""
    torch.manual_seed(seed); np.random.seed(seed)
    sc=MinMaxScaler().fit(Ms); Xs=sc.transform(Ms); Xt=sc.transform(Mt)
    if clip: Xs=np.clip(Xs,0,1); Xt=np.clip(Xt,0,1)
    d=Xs.shape[1]
    pairs=[(i,j,S_target[i,j]) for i in range(N) for j in range(N) if i!=j]
    I=np.array([p[0] for p in pairs],np.int64); J=np.array([p[1] for p in pairs],np.int64)
    Y=np.array([p[2] for p in pairs],np.float32)
    class DS(Dataset):
        def __len__(s_): return len(I)
        def __getitem__(s_,k): return torch.tensor(Xs[I[k]],dtype=torch.float32),torch.tensor(Xs[J[k]],dtype=torch.float32),torch.tensor(Y[k])
    dl=DataLoader(DS(),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m=Net(d,h,dropout=dropout); opt=optim.Adam(m.parameters(),lr=lr,weight_decay=wd); mse=nn.MSELoss()
    for _ in range(epochs):
        m.train()
        for a,b,y in dl:
            opt.zero_grad()
            ea=m(a); eb=m(b)
            cos=(ea*eb).sum(dim=1)                     # both unit-norm → cosine ∈ [-1,1]
            pred=(cos+1)/2                              # → [0,1] to match Jaccard target range
            loss=mse(pred,y); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Es=m(torch.tensor(Xs,dtype=torch.float32)).numpy()
        Et=m(torch.tensor(Xt,dtype=torch.float32)).numpy()
    return float(np.mean([tr(t,src[int(cdist(Et[i:i+1],Es).argmin())])/piv[t].max() for i,t in enumerate(tgt)]))

SEEDS=[0,1,2,3]
RECIPE='114D + 256-128-16 + drop0.3 + wd1e-4 + clip + 25 epochs'
exps=[
    ('V0: top-1% Jaccard triplet (K=10) — recipe baseline',  lambda s: train_triplet(src_M,tgt_M,S_top1,s,K=10)),
    ('V1: TPS-weighted top-1% Jaccard triplet (K=10)',        lambda s: train_triplet(src_M,tgt_M,S_top1_w,s,K=10)),
    ('V2: Multi-K avg(top-1%,3%,5%) Jaccard triplet (K=10)', lambda s: train_triplet(src_M,tgt_M,S_multiK,s,K=10)),
    ('V3: top-1% Jaccard REGRESSION (cos≈overlap, ~85K pairs)', lambda s: train_regression(src_M,tgt_M,S_top1,s)),
]
results={}
for name,fn in exps:
    arr=[]
    for s in SEEDS:
        v=fn(s); arr.append(v)
        print(f'    {name[:45]:45s} seed {s}: {v:.3f}   ({time.time()-t0:.0f}s)',flush=True)
    results[name]=np.array(arr)

print('\n===== improving the TOP-1%-target embedding (recipe: '+RECIPE+') =====')
print('  prior references (from earlier runs)')
print(f'    F: 114D top-1% Jaccard recipe (5 seeds):     0.829 ± 0.049')
print(f'    OT 114 metrics:                              0.845')
print(f'    RF 114 metrics:                              0.910')
print(f'    top-1% overlap DIRECT (full sweep):          0.923')
print(f'    ORACLE:                                      0.994')
print('\n  this run (4 seeds each):')
for name,arr in results.items():
    print(f'    {name:60s}: {arr.mean():.3f} ± {arr.std():.3f}   range [{arr.min():.3f},{arr.max():.3f}]')
print(f'\ndone ({time.time()-t0:.0f}s)')
