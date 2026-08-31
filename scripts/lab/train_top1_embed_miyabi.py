import pandas as pd, numpy as np, glob, os, math, time, warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
t0=time.time()
PARAMS=['innodb_buffer_pool_size','innodb_read_io_threads','innodb_write_io_threads','innodb_flush_log_at_trx_commit','innodb_adaptive_hash_index','sync_binlog','innodb_lru_scan_depth','innodb_buffer_pool_instances','innodb_change_buffer_max_size','innodb_io_capacity','innodb_log_file_size','table_open_cache']
MN11=['Average Memory Usage Percentage','InnoDB Buffer Pool Cache Hit Rate','InnoDB Dirty Buffer Pages','Current QPS (Queries Per Second)','Max CPU Usage (100 - Idle)','InnoDB Rows Deleted (60s Rate)','InnoDB Rows Inserted (60s Rate)','InnoDB Rows Read (60s Rate)','InnoDB Rows Updated (60s Rate)','Average Disk IOPS (Read)','Average Disk IOPS (Write)']
IOPS=['Average Disk IOPS (Read)','Average Disk IOPS (Write)']
MN9=[m for m in MN11 if m not in IOPS]
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
    df=pd.read_csv(f); df=df.loc[:,~df.columns.duplicated()]; hw=os.path.basename(f).replace('-result.csv','')
    K=np.array([[nf(x) for x in df[p]] for p in PARAMS]).T
    ok=np.isfinite(K).all(1)&pd.to_numeric(df['tps'],errors='coerce').notna().values
    df=df[ok].reset_index(drop=True); K=K[ok]
    ck=['|'.join(str(round(v,6)) for v in row) for row in K]
    if 'workload_label' in df.columns: wl=df['workload_label'].astype(str)
    else:
        sk=df['skew'].apply(lambda x:'nan' if pd.isna(x) else (str(int(x)) if float(x)==int(x) else str(x)))
        wl=(df['num_table'].astype(int).astype(str)+'-'+df['table_size'].astype(int).astype(str)+'-'+df['num_client'].astype(int).astype(str)+'-'+df['workload'].astype(str)+'-'+sk)
    return pd.DataFrame({'ctx':hw+'_'+wl,'ck':ck,'tps':pd.to_numeric(df['tps'],errors='coerce').values})

alld=pd.concat([load(f) for f in glob.glob('DBMSTransferLearning/dataset/full_data/*-result.csv')],ignore_index=True)
ctx_tps={c:g.groupby('ck')['tps'].max().to_dict() for c,g in alld.groupby('ctx')}
src_ctxs=sorted(c for c in ctx_tps if not c.startswith('112c125g_'))
tgt_ctxs=sorted(c for c in ctx_tps if c.startswith('112c125g_'))
piv=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx').dropna()
bestcfg={s:piv[s].idxmax() for s in piv.columns}; tr=lambda t,s:piv.at[bestcfg[s],t]
print(f'loaded ({time.time()-t0:.0f}s): {len(src_ctxs)} src + {len(tgt_ctxs)} miyabi targets',flush=True)

# ---- Top-1% set per source context, pairwise Jaccard overlap ----
def top_set(ctx,frac=0.01):
    tps=ctx_tps[ctx]; n=len(tps); k=max(1,math.ceil(n*frac))
    return set(c for c,_ in sorted(tps.items(),key=lambda x:x[1],reverse=True)[:k])
T1={c:top_set(c) for c in src_ctxs}
N=len(src_ctxs); ovM=np.zeros((N,N))
for i,a in enumerate(src_ctxs):
    Ta=T1[a]
    for j,b in enumerate(src_ctxs):
        if i==j: continue
        Tb=T1[b]
        ovM[i,j]=len(Ta&Tb)/max(1,len(Ta|Tb))
print(f'pairwise top-1% overlap matrix built ({time.time()-t0:.0f}s)',flush=True)

# ---- Triplets: per anchor, 5 highest-overlap positives × 5 lowest-overlap negatives ----
K=5
ctx2idx={c:i for i,c in enumerate(src_ctxs)}
trips=[]
for i,anchor in enumerate(src_ctxs):
    scores=ovM[i].copy(); scores[i]=-np.inf
    order=np.argsort(scores)[::-1]
    pos=[src_ctxs[j] for j in order[:K]]; neg=[src_ctxs[j] for j in order[-K:]]
    for p in pos:
        for n in neg: trips.append((anchor,p,n))
print(f'triplets built: {len(trips)} from {N} anchors (K={K})',flush=True)

# ---- Feature vectors per source context (from context_default_metrics_all.csv) ----
cm=pd.read_csv('DBMSTransferLearning/dataset/context_default_metrics_all.csv').set_index('context_id').loc[src_ctxs]
# default metric vectors for target (from full_data 112c125g)
DK='1.0|1.0|2.0|1.0|1.0|1.0|1024.0|1.0|25.0|100.0|0.048|4000.0'
tg_defrow=alld[(alld['ctx'].str.startswith('112c125g_'))&(alld['ck']==DK)].groupby('ctx').mean(numeric_only=True)
tg_defrow=tg_defrow.reindex(tgt_ctxs)
# Need 11 metric cols on target — pull from full_data 112c125g raw
tg_raw=pd.read_csv('DBMSTransferLearning/dataset/full_data/112c125g-result.csv')
tg_raw['ctx']='112c125g_'+(tg_raw['num_table'].astype(int).astype(str)+'-'+tg_raw['table_size'].astype(int).astype(str)+'-'+tg_raw['num_client'].astype(int).astype(str)+'-'+tg_raw['workload'].astype(str)+'-'+tg_raw['skew'].apply(lambda x:'nan' if pd.isna(x) else (str(int(x)) if float(x)==int(x) else str(x))))
K12=np.array([[nf(x) for x in tg_raw[p]] for p in PARAMS]).T
tg_raw['ck']=['|'.join(str(round(v,6)) for v in row) for row in K12]
tg_def=tg_raw[tg_raw['ck']==DK].groupby('ctx')[MN11].mean().reindex(tgt_ctxs)
print(f'target default 11-metric vectors:\n{tg_def[MN11[:3]].round(2)}',flush=True)

class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,16))
    def forward(s,x): return nn.functional.normalize(s.net(x),p=2,dim=1)

def train_eval(metrics,seed,epochs=50):
    torch.manual_seed(seed); np.random.seed(seed)
    sc=MinMaxScaler().fit(cm[metrics])
    Xs=sc.transform(cm[metrics].to_numpy(float))               # (293, d) source scaled
    XtFull=sc.transform(tg_def[metrics].to_numpy(float))       # (5, d) target scaled
    # build triplet feature tensor
    d=len(metrics); A=np.zeros((len(trips),d),np.float32); P=np.zeros_like(A); Ng=np.zeros_like(A)
    for i,(a,p,n) in enumerate(trips):
        A[i]=Xs[ctx2idx[a]]; P[i]=Xs[ctx2idx[p]]; Ng[i]=Xs[ctx2idx[n]]
    class DS(Dataset):
        def __len__(s): return len(A)
        def __getitem__(s,i): return torch.tensor(A[i]),torch.tensor(P[i]),torch.tensor(Ng[i])
    dl=DataLoader(DS(),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m=Net(d); opt=optim.Adam(m.parameters(),lr=1e-3); crit=nn.TripletMarginLoss(margin=1.0,p=2)
    for _ in range(epochs):
        m.train()
        for a,p,n in dl:
            opt.zero_grad(); crit(m(a),m(p),m(n)).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        E_src=m(torch.tensor(Xs,dtype=torch.float32)).numpy()
        E_tg=m(torch.tensor(XtFull,dtype=torch.float32)).numpy()
    # nearest source per target -> transferred ratio
    ratios=[]; picks=[]
    for i,t in enumerate(tgt_ctxs):
        d_=cdist(E_tg[i:i+1],E_src).flatten(); j=int(d_.argmin())
        picks.append(src_ctxs[j])
        ratios.append(tr(t,src_ctxs[j])/piv[t].max())
    return np.mean(ratios),ratios,picks

# ---- Multi-seed evaluation ----
SEEDS=range(10); results={'e11':[],'e9':[]}
detail_seed=None
for s in SEEDS:
    m11,r11,p11=train_eval(MN11,s)
    m9,r9,p9=train_eval(MN9,s)
    results['e11'].append(m11); results['e9'].append(m9)
    if s==0: detail_seed=(r11,p11,r9,p9)
    print(f'seed {s}: e11(top1)={m11:.3f}  e9(top1)={m9:.3f}  ({time.time()-t0:.0f}s)',flush=True)

print('\n===== Mean Miyabi transferred/best over 10 seeds (TOP-1% OVERLAP triplets) =====')
for k,lbl in [('e11','embed 11 (top-1% signal)'),('e9','embed 9 no-IOPS (top-1% signal)')]:
    a=np.array(results[k]); print(f'  {lbl:38s} {a.mean():.3f} ± {a.std():.3f}   (min {a.min():.3f}, max {a.max():.3f})')

print('\n=== fixed references (from prior runs) ===')
print('  raw (11):                      0.821')
print('  OT (114 metrics):              0.845')
print('  embed 11 (CONCORDANCE signal): 0.801 ± 0.041 (reseeded, 10 seeds)')
print('  embed 9  (CONCORDANCE signal): 0.837 ± 0.061 (reseeded, 10 seeds, no IOPS)')
print('  top-1% overlap DIRECT:         0.923 (no embedding, oracle pick)')
print('  ORACLE:                        0.994')

# Per-target detail for seed 0
r11,p11,r9,p9=detail_seed
print(f'\n=== per-target detail (seed 0) ===')
print(f'  {"target":36s}{"e11_top1":>10s}{"e11_pick":36s}{"e9_top1":>10s}{"e9_pick":36s}')
def short(c): return c.replace('_64-1000000-4-oltp_','').replace('_64-100000-4-oltp_','*').replace('-4-4-tpcc-nan','wh-tpcc')
for i,t in enumerate(tgt_ctxs):
    print(f'  {t.replace("112c125g_",""):36s}{r11[i]:10.3f}{short(p11[i]):36s}{r9[i]:10.3f}{short(p9[i]):36s}')
print(f'\ndone ({time.time()-t0:.0f}s)')
