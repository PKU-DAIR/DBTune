import pandas as pd, numpy as np, glob, os, math, time, warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
t0=time.time()
PARAMS=['innodb_buffer_pool_size','innodb_read_io_threads','innodb_write_io_threads','innodb_flush_log_at_trx_commit','innodb_adaptive_hash_index','sync_binlog','innodb_lru_scan_depth','innodb_buffer_pool_instances','innodb_change_buffer_max_size','innodb_io_capacity','innodb_log_file_size','table_open_cache']
MN11=['Average Memory Usage Percentage','InnoDB Buffer Pool Cache Hit Rate','InnoDB Dirty Buffer Pages','Current QPS (Queries Per Second)','Max CPU Usage (100 - Idle)','InnoDB Rows Deleted (60s Rate)','InnoDB Rows Inserted (60s Rate)','InnoDB Rows Read (60s Rate)','InnoDB Rows Updated (60s Rate)','Average Disk IOPS (Read)','Average Disk IOPS (Write)']
IOPS=['Average Disk IOPS (Read)','Average Disk IOPS (Write)']; MN9=[m for m in MN11 if m not in IOPS]
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
print(f'loaded ({time.time()-t0:.0f}s)',flush=True)

# default metric vectors
COLS114=[c for c in alld.columns if c not in {'ctx','ck','tps'}]
def_all=alld[alld.ck==DK].groupby('ctx')[COLS114].mean()
src_114_df=def_all.loc[src]; tgt_114_df=def_all.loc[tgt]
usable114=[c for c in COLS114 if src_114_df[c].notna().all() and tgt_114_df[c].notna().all()]
src_M114=src_114_df[usable114].to_numpy(float); tgt_M114=tgt_114_df[usable114].to_numpy(float)
cm=pd.read_csv('DBMSTransferLearning/dataset/context_default_metrics_all.csv').set_index('context_id').loc[src]
src_M11=cm[MN11].to_numpy(float); src_M9=cm[MN9].to_numpy(float)
tg_def=alld[(alld.ctx.str.startswith('112c125g_'))&(alld.ck==DK)].groupby('ctx')[MN11].mean().reindex(tgt)
tgt_M11=tg_def[MN11].to_numpy(float); tgt_M9=tg_def[MN9].to_numpy(float)

# similarity matrices
def top_set(c,frac=0.01):
    t=ctx_tps[c]; k=max(1,math.ceil(len(t)*frac))
    return set(x for x,_ in sorted(t.items(),key=lambda x:x[1],reverse=True)[:k])
T1={c:top_set(c) for c in src}
S_top1=np.zeros((N,N))
for i,a in enumerate(src):
    for j,b in enumerate(src):
        if i!=j: S_top1[i,j]=len(T1[a]&T1[b])/max(1,len(T1[a]|T1[b]))
conc=pd.read_csv('DBMSTransferLearning/dataset/concordant_pair_ranking.csv')
S_conc=np.zeros((N,N))
for _,r in conc.iterrows():
    a,b,s=r['context_id'],r['similar_context_id'],r['similarity_score']
    if a in ctx2i and b in ctx2i: S_conc[ctx2i[a],ctx2i[b]]=float(s)
print(f'sim matrices ready ({time.time()-t0:.0f}s)',flush=True)

def build_trips(S,K):
    trips=[]
    for i in range(N):
        sc=S[i].copy(); sc[i]=-np.inf
        order=np.argsort(sc)[::-1]
        pos=order[:K]; neg=order[-K:]
        for p in pos:
            for n in neg: trips.append((i,p,n))
    return trips

class Net(nn.Module):
    def __init__(s,d,h_sizes,dropout=0.0):
        super().__init__(); layers=[]; prev=d
        for h in h_sizes[:-1]:
            layers.append(nn.Linear(prev,h)); layers.append(nn.ReLU())
            if dropout>0: layers.append(nn.Dropout(dropout))
            prev=h
        layers.append(nn.Linear(prev,h_sizes[-1])); s.net=nn.Sequential(*layers)
    def forward(s,x): return F.normalize(s.net(x),p=2,dim=1)

def train_eval(Ms,Mt,trips,seed,h_sizes,dropout=0.0,wd=0.0,clip_inputs=False,
               epochs=50,batch_size=64,lr=1e-3,margin=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    sc=MinMaxScaler().fit(Ms)
    Xs=sc.transform(Ms); Xt=sc.transform(Mt)
    if clip_inputs: Xs=np.clip(Xs,0,1); Xt=np.clip(Xt,0,1)
    d=Xs.shape[1]
    A=np.array([Xs[a] for a,_,_ in trips],np.float32)
    P=np.array([Xs[p] for _,p,_ in trips],np.float32)
    Ng=np.array([Xs[n] for _,_,n in trips],np.float32)
    class DS(Dataset):
        def __len__(s_): return len(A)
        def __getitem__(s_,i): return torch.tensor(A[i]),torch.tensor(P[i]),torch.tensor(Ng[i])
    dl=DataLoader(DS(),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m=Net(d,h_sizes,dropout=dropout)
    opt=optim.Adam(m.parameters(),lr=lr,weight_decay=wd); crit=nn.TripletMarginLoss(margin=margin,p=2)
    for _ in range(epochs):
        m.train()
        for a,p,n in dl: opt.zero_grad(); crit(m(a),m(p),m(n)).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Es=m(torch.tensor(Xs,dtype=torch.float32)).numpy()
        Et=m(torch.tensor(Xt,dtype=torch.float32)).numpy()
    ratios=[tr(t,src[int(cdist(Et[i:i+1],Es).argmin())])/piv[t].max() for i,t in enumerate(tgt)]
    return float(np.mean(ratios)),ratios

# precompute trips
trips_top1_K5=build_trips(S_top1,5); trips_top1_K10=build_trips(S_top1,10)
trips_conc_K5=build_trips(S_conc,5); trips_conc_K10=build_trips(S_conc,10)
print(f'trips: top1_K5={len(trips_top1_K5)}, top1_K10={len(trips_top1_K10)}, conc_K5={len(trips_conc_K5)}, conc_K10={len(trips_conc_K10)}',flush=True)

# experiment grid
exps=[
    # 114D + top-1% signal (the variant in question)
    ('A: 114D top1 control 64-32-16, K=5, no reg',          src_M114,tgt_M114,trips_top1_K5,[64,32,16],0.0,0.0,False,50),
    ('B: 114D top1 WIDER 256-128-16, K=5, no reg',           src_M114,tgt_M114,trips_top1_K5,[256,128,16],0.0,0.0,False,50),
    ('C: 114D top1 WIDER + dropout 0.3 + wd 1e-4, K=5',      src_M114,tgt_M114,trips_top1_K5,[256,128,16],0.3,1e-4,False,50),
    ('D: 114D top1 WIDER + drop 0.3 + wd + K=10',            src_M114,tgt_M114,trips_top1_K10,[256,128,16],0.3,1e-4,False,25),
    ('E: 114D top1 WIDER + drop 0.3 + wd + clip-inputs',     src_M114,tgt_M114,trips_top1_K5,[256,128,16],0.3,1e-4,True,50),
    ('F: 114D top1 WIDER + drop 0.3 + wd + K=10 + clip',     src_M114,tgt_M114,trips_top1_K10,[256,128,16],0.3,1e-4,True,25),
    # apply the recipe to the 11D / 9D concordance baselines
    ('G: 11D conc WIDER 128-64-16 + drop 0.3 + wd + K=10',   src_M11,tgt_M11,trips_conc_K10,[128,64,16],0.3,1e-4,False,25),
    ('H: 9D no-IOPS conc WIDER 128-64-16 + drop + wd + K=10',src_M9, tgt_M9, trips_conc_K10,[128,64,16],0.3,1e-4,False,25),
]
SEEDS=[0,1,2,3,4]
results={}
for name,Ms,Mt,trips,hs,dp,wd,clip,ep in exps:
    means=[]
    for s in SEEDS:
        mn,_=train_eval(Ms,Mt,trips,s,h_sizes=hs,dropout=dp,wd=wd,clip_inputs=clip,epochs=ep); means.append(mn)
    results[name]=np.array(means)
    a=results[name]
    print(f'  {name:60s}: {a.mean():.3f} ± {a.std():.3f}  [{a.min():.3f},{a.max():.3f}]  ({time.time()-t0:.0f}s)',flush=True)

print('\n===== summary: architectural & regularization fixes =====')
print('  baseline references (from prior runs)')
print(f'    embed 11D concordance K=5 (10 seeds): 0.801 ± 0.041')
print(f'    embed 9D  concordance K=5 (10 seeds): 0.837 ± 0.061')
print(f'    embed 11D top-1% signal K=5:          0.759 ± 0.064')
print(f'    embed 114D top-1% K=5 narrow:         0.753 ± 0.059  (from prior run)')
print(f'    RF on 114 metrics:                    0.910')
print(f'    top-1% overlap DIRECT:                0.923')
print(f'    ORACLE:                               0.994')
print('\n  new variants (5 seeds each):')
for name,arr in results.items():
    print(f'    {name:60s}: {arr.mean():.3f} ± {arr.std():.3f}')
print(f'\ndone ({time.time()-t0:.0f}s)')
