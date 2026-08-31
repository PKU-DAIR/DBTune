import pandas as pd, numpy as np, glob, os, warnings, time
warnings.filterwarnings('ignore'); t0=time.time()
from scipy.spatial.distance import cdist
import torch, joblib, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
class EmbeddingNet(nn.Module):
    def __init__(s,i,e=16):
        super().__init__(); s.net=nn.Sequential(nn.Linear(i,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,e))
    def forward(s,x): return nn.functional.normalize(s.net(x),p=2,dim=1)
PARAMS=['innodb_buffer_pool_size','innodb_read_io_threads','innodb_write_io_threads','innodb_flush_log_at_trx_commit','innodb_adaptive_hash_index','sync_binlog','innodb_lru_scan_depth','innodb_buffer_pool_instances','innodb_change_buffer_max_size','innodb_io_capacity','innodb_log_file_size','table_open_cache']
MN11=['Average Memory Usage Percentage','InnoDB Buffer Pool Cache Hit Rate','InnoDB Dirty Buffer Pages','Current QPS (Queries Per Second)','Max CPU Usage (100 - Idle)','InnoDB Rows Deleted (60s Rate)','InnoDB Rows Inserted (60s Rate)','InnoDB Rows Read (60s Rate)','InnoDB Rows Updated (60s Rate)','Average Disk IOPS (Read)','Average Disk IOPS (Write)']
IOPS=['Average Disk IOPS (Read)','Average Disk IOPS (Write)']; CPU=['Max CPU Usage (100 - Idle)']
MN9=[m for m in MN11 if m not in IOPS]; MN8=[m for m in MN9 if m not in CPU]
tdf=pd.read_csv('DBMSTransferLearning/dataset/full_triplet_data_concordance.csv')
ctxdf=pd.read_csv('DBMSTransferLearning/dataset/context_default_metrics_all.csv')
def train_embed(metrics,seed):
    cols=[f'{p}_{m}' for p in ['anchor','pos','neg'] for m in metrics]
    X=tdf[cols].to_numpy('float32'); d=len(metrics)
    torch.manual_seed(seed); np.random.seed(seed)
    class DS(Dataset):
        def __len__(s):return len(X)
        def __getitem__(s,i):r=X[i];return torch.tensor(r[:d]),torch.tensor(r[d:2*d]),torch.tensor(r[2*d:3*d])
    dl=DataLoader(DS(),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m=EmbeddingNet(d);opt=optim.Adam(m.parameters(),lr=1e-3);crit=nn.TripletMarginLoss(margin=1.0,p=2)
    for _ in range(50):
        for a,p,n in dl: opt.zero_grad();crit(m(a),m(p),m(n)).backward();opt.step()
    m.eval();return m,MinMaxScaler().fit(ctxdf[metrics])
# ---- eval pipeline (once) ----
def nf(v):
    s=str(v).strip()
    if s in('ON','Yes','True'):return 1.0
    if s in('OFF','No','False'):return 0.0
    if s.endswith('GB'):return float(s[:-2])
    if s.endswith('MB'):return float(s[:-2])/1000.0
    if s.endswith('KB'):return float(s[:-2])/1e6
    try:return float(s)
    except:return np.nan
tg0=pd.read_csv('DBMSTransferLearning/dataset/full_data/112c125g-result.csv');METRICS114=list(tg0.columns[19:])
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
    for m in METRICS114: out[m]=pd.to_numeric(df[m],errors='coerce').values if m in df.columns else np.nan
    return out
alld=pd.concat([load(f) for f in glob.glob('DBMSTransferLearning/dataset/full_data/*-result.csv')],ignore_index=True)
dk='1.0|1.0|2.0|1.0|1.0|1.0|1024.0|1.0|25.0|100.0|0.048|4000.0'
piv=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx').dropna()
ctxs=list(piv.columns);idx={c:i for i,c in enumerate(ctxs)};bestcfg={s:piv[s].idxmax() for s in ctxs};tr=lambda c,s:piv.at[bestcfg[s],c]
defrow=alld[alld['ck']==dk].groupby('ctx')[METRICS114].mean().reindex(ctxs)
src=[c for c in ctxs if not c.startswith('112c125g_')];tgt=[c for c in ctxs if c.startswith('112c125g_')]
sj=[idx[s] for s in src]
def emb_mean(metrics,seed):
    m,sc=train_embed(metrics,seed);Xs=sc.transform(defrow[metrics].to_numpy(float))
    with torch.no_grad(): E=m(torch.tensor(Xs,dtype=torch.float32)).numpy()
    r=[tr(c,src[int(cdist(E[idx[c]:idx[c]+1],E[sj]).argmin())])/piv[c].max() for c in tgt]
    return np.mean(r)
SEEDS=range(10)
res={'e11':[],'e9':[],'e8':[]}
for s in SEEDS:
    res['e11'].append(emb_mean(MN11,s));res['e9'].append(emb_mean(MN9,s));res['e8'].append(emb_mean(MN8,s))
    print(f'seed {s}: e11={res["e11"][-1]:.3f} e9={res["e9"][-1]:.3f} e8={res["e8"][-1]:.3f}  ({time.time()-t0:.0f}s)',flush=True)
print('\n===== mean Miyabi transferred/best over',len(list(SEEDS)),'seeds (±std) =====')
for k,lbl in [('e11','embed 11 metrics'),('e9','embed 9 (no disk-IOPS)'),('e8','embed 8 (no IOPS+CPU)')]:
    a=np.array(res[k]); print(f'  {lbl:26s} {a.mean():.3f} ± {a.std():.3f}   (min {a.min():.3f}, max {a.max():.3f})')
print('  (fixed baselines from prior run: raw=0.821, ot114=0.845, oracle=0.994, default=0.672)')
