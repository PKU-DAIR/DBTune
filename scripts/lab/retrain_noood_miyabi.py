import pandas as pd, numpy as np, glob, os, warnings
warnings.filterwarnings('ignore')
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

# ---------- train an embedding on a metric subset (drop cols from existing scaled triplets) ----------
TRIP='DBMSTransferLearning/dataset/full_triplet_data_concordance.csv'
CTX='DBMSTransferLearning/dataset/context_default_metrics_all.csv'
tdf=pd.read_csv(TRIP); ctxdf=pd.read_csv(CTX)
def train_embed(metrics,seed=42,epochs=50):
    cols=[f'anchor_{m}' for m in metrics]+[f'pos_{m}' for m in metrics]+[f'neg_{m}' for m in metrics]
    X=tdf[cols].to_numpy('float32'); d=len(metrics)
    torch.manual_seed(seed); np.random.seed(seed)
    class DS(Dataset):
        def __len__(s): return len(X)
        def __getitem__(s,i): r=X[i]; return torch.tensor(r[:d]),torch.tensor(r[d:2*d]),torch.tensor(r[2*d:3*d])
    dl=DataLoader(DS(),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m=EmbeddingNet(d); opt=optim.Adam(m.parameters(),lr=1e-3); crit=nn.TripletMarginLoss(margin=1.0,p=2)
    for _ in range(epochs):
        m.train()
        for a,p,n in dl:
            opt.zero_grad(); l=crit(m(a),m(p),m(n)); l.backward(); opt.step()
    m.eval()
    sc=MinMaxScaler().fit(ctxdf[metrics])
    return m,sc
print('training embeddings (11r/9/8)...',flush=True)
models={k:train_embed(ms) for k,ms in [('e11',MN11),('e9',MN9),('e8',MN8)]}
print('done training',flush=True)

# ---------- Miyabi eval pipeline (reuse full_data) ----------
def nf(v):
    s=str(v).strip()
    if s in('ON','Yes','True'):return 1.0
    if s in('OFF','No','False'):return 0.0
    if s.endswith('GB'):return float(s[:-2])
    if s.endswith('MB'):return float(s[:-2])/1000.0
    if s.endswith('KB'):return float(s[:-2])/1e6
    try:return float(s)
    except:return np.nan
tg0=pd.read_csv('DBMSTransferLearning/dataset/full_data/112c125g-result.csv'); METRICS114=list(tg0.columns[19:])
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
    out=pd.DataFrame({'ctx':hw+'_'+wl,'ck':ck,'tps':pd.to_numeric(df['tps'],errors='coerce').values,'id':df['id'].values})
    for m in METRICS114: out[m]=pd.to_numeric(df[m],errors='coerce').values if m in df.columns else np.nan
    return out
alld=pd.concat([load(f) for f in glob.glob('DBMSTransferLearning/dataset/full_data/*-result.csv')],ignore_index=True)
dk='1.0|1.0|2.0|1.0|1.0|1.0|1024.0|1.0|25.0|100.0|0.048|4000.0'
tps=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx'); piv=tps.dropna()
ctxs=list(piv.columns); shared=set(piv.index); idx={c:i for i,c in enumerate(ctxs)}
bestcfg={s:piv[s].idxmax() for s in ctxs}; tr=lambda c,s:piv.at[bestcfg[s],c]
defrow=alld[alld['ck']==dk].groupby('ctx')[METRICS114].mean().reindex(ctxs)
# OT on usable 114
usable=[m for m in METRICS114 if defrow[m].notna().all()]
pop=alld[alld['ck'].isin(shared)][usable].apply(pd.to_numeric,errors='coerce').to_numpy(float)
edges=np.nanpercentile(pop,np.linspace(0,100,11),axis=0)
def binv(V):
    B=np.empty_like(V,dtype=int)
    for j in range(V.shape[1]): B[:,j]=np.clip(np.digitize(V[:,j],edges[:,j],right=True),1,10)
    return B
defB=binv(defrow[usable].to_numpy(float))
# raw (11) + new embeddings
base='autotune/optimizer/dml_models_all/'; sc11=joblib.load(base+'scaler.pkl')
Xs11=sc11.transform(defrow[MN11].to_numpy(float))
EMB={}; SCX={}
for k,ms in [('e11',MN11),('e9',MN9),('e8',MN8)]:
    m,sc=models[k]; Xs=sc.transform(defrow[ms].to_numpy(float))
    with torch.no_grad(): EMB[k]=m(torch.tensor(Xs,dtype=torch.float32)).numpy()
    SCX[k]=Xs
src=[c for c in ctxs if not c.startswith('112c125g_')]; tgt=[c for c in ctxs if c.startswith('112c125g_')]
def nn_pick(c,cand,vec): cj=[idx[s] for s in cand]; return cand[int(cdist(vec[idx[c]:idx[c]+1],vec[cj]).argmin())]
rows=[]
for c in tgt:
    b=piv[c].max()
    sr=src[int(cdist(Xs11[idx[c]:idx[c]+1],Xs11[[idx[s] for s in src]]).argmin())]
    so=src[int(cdist(defB[idx[c]:idx[c]+1].astype(float),defB[[idx[s] for s in src]].astype(float)).argmin())]
    rows.append(dict(target=c.replace('112c125g_','').replace('64-1000000-4-oltp_','').replace('-4-4-tpcc-nan','wh-tpcc'),
        default=round(piv.at[dk,c]/b,3), raw=round(tr(c,sr)/b,3),
        ot114=round(tr(c,so)/b,3),
        embed11=round(tr(c,nn_pick(c,src,EMB['e11']))/b,3),
        embed9_noIOPS=round(tr(c,nn_pick(c,src,EMB['e9']))/b,3),
        embed8_noIOPS_noCPU=round(tr(c,nn_pick(c,src,EMB['e8']))/b,3),
        oracle=round(max(tr(c,s) for s in src)/b,3)))
df=pd.DataFrame(rows); df.to_csv('scripts/lab/transfer_quality_miyabi_noood.csv',index=False)
mean=df.drop(columns=['target']).mean().round(3); mean['target']='MEAN'
df=pd.concat([df,pd.DataFrame([mean])],ignore_index=True)
print('\n===== Miyabi targets: effect of dropping OOD columns from the embedding =====')
print('  (raw=11-metric NN, ot114=OtterTune all metrics; embeds reseeded for fair compare)\n')
with pd.option_context('display.width',200,'display.max_columns',20):
    print(df.to_string(index=False))
