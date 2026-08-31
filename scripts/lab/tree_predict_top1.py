import pandas as pd, numpy as np, glob, os, math, time, warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
    out=pd.DataFrame({'ctx':hw+'_'+wl,'ck':ck,'tps':pd.to_numeric(df['tps'],errors='coerce').values})
    cols114=list(pd.read_csv(f,nrows=0).columns)[19:]
    for m in cols114: out[m]=pd.to_numeric(df[m],errors='coerce').values if m in df.columns else np.nan
    return out
files=sorted(glob.glob('DBMSTransferLearning/dataset/full_data/*-result.csv'))
alld=pd.concat([load(f) for f in files],ignore_index=True)
ctx_tps={c:g.groupby('ck')['tps'].max().to_dict() for c,g in alld.groupby('ctx')}
src_ctxs=sorted(c for c in ctx_tps if not c.startswith('112c125g_'))
tgt_ctxs=sorted(c for c in ctx_tps if c.startswith('112c125g_'))
piv=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx').dropna()
bestcfg={s:piv[s].idxmax() for s in piv.columns}; tr=lambda t,s:piv.at[bestcfg[s],t]
print(f'loaded ({time.time()-t0:.0f}s)',flush=True)

# Top-1% sets per context, pairwise overlap among sources, + (target, source) overlap
def top_set(c,frac=0.01):
    tps=ctx_tps[c]; n=len(tps); k=max(1,math.ceil(n*frac))
    return set(x for x,_ in sorted(tps.items(),key=lambda x:x[1],reverse=True)[:k])
T1={c:top_set(c) for c in src_ctxs+tgt_ctxs}
N=len(src_ctxs)
ovM=np.zeros((N,N))
for i,a in enumerate(src_ctxs):
    Ta=T1[a]
    for j,b in enumerate(src_ctxs):
        if i!=j: ovM[i,j]=len(Ta&T1[b])/max(1,len(Ta|T1[b]))
# (target, source) overlap (ground truth for Miyabi eval)
ov_TS={t:{s:len(T1[t]&T1[s])/max(1,len(T1[t]|T1[s])) for s in src_ctxs} for t in tgt_ctxs}

# Default metric vectors per source (from context_default_metrics_all.csv)
cm=pd.read_csv('DBMSTransferLearning/dataset/context_default_metrics_all.csv').set_index('context_id').loc[src_ctxs]
# Target default 11-metric vectors (from 112c125g full_data)
DK='1.0|1.0|2.0|1.0|1.0|1.0|1024.0|1.0|25.0|100.0|0.048|4000.0'
tg_def=alld[(alld.ctx.str.startswith('112c125g_'))&(alld.ck==DK)].groupby('ctx')[MN11].mean().reindex(tgt_ctxs)
# 114-metric defaults (sources + target) from full_data (col-name aligned)
COLS114=[c for c in alld.columns if c not in {'ctx','ck','tps'}]
def_all_full=alld[alld.ck==DK].groupby('ctx')[COLS114].mean()
src_114=def_all_full.loc[src_ctxs]; tg_114=def_all_full.loc[tgt_ctxs]
usable114=[c for c in COLS114 if src_114[c].notna().all() and tg_114[c].notna().all()]
print(f'  usable 114-metric cols (no NaN in either): {len(usable114)}',flush=True)

def build_pair_feats(M_i, M_j):           # element-wise |diff|, mean; row vectors
    a=np.asarray(M_i,float); b=np.asarray(M_j,float)
    return np.concatenate([np.abs(a-b),(a+b)/2.0])

# ---- (A) Spearman corr per metric: does |diff| predict top1 overlap? (source-source pairs, 11 metric) ----
print('\n=== (A) Per-metric Spearman ρ(|diff|, top1_overlap) on source-source pairs (11 metrics) ===')
src_mat11=cm[MN11].to_numpy(float)
pair_y=[]; pair_diff11=[]
for i in range(N):
    for j in range(i+1,N):
        pair_y.append(ovM[i,j])
        pair_diff11.append(np.abs(src_mat11[i]-src_mat11[j]))
pair_y=np.array(pair_y); pair_diff11=np.array(pair_diff11)
print(f'  pairs: {len(pair_y)}; mean overlap {pair_y.mean():.3f}\n  metric{"":33s}{"ρ(|diff|, overlap)":>22s}{"|ρ| rank":>10s}')
rhos=[]
for k,m in enumerate(MN11):
    rho,_=spearmanr(pair_diff11[:,k],pair_y); rhos.append((m,rho))
for m,rho in sorted(rhos,key=lambda x:abs(x[1]),reverse=True):
    print(f'  {m[:38]:38s}{rho:22.3f}')

# ---- (B) Tree-based regressor on pair features; evaluate on Miyabi ----
def train_and_score(M_src_arr, M_tgt_arr, dim_name, n_estimators=300, seed=0):
    # build full source-source training set (ordered both ways for symmetry)
    rows=[]; ys=[]
    for i in range(N):
        for j in range(N):
            if i==j: continue
            rows.append(build_pair_feats(M_src_arr[i],M_src_arr[j])); ys.append(ovM[i,j])
    X=np.array(rows); y=np.array(ys)
    rf=RandomForestRegressor(n_estimators=n_estimators,random_state=seed,n_jobs=-1,max_depth=None)
    rf.fit(X,y)
    # also linear baseline
    lr=LinearRegression().fit(X,y)
    # Miyabi eval
    results={}
    for picker,model in [('RF',rf),('Linear',lr)]:
        ratios=[]; picks=[]; truths=[]
        for ti,t in enumerate(tgt_ctxs):
            X_t=np.array([build_pair_feats(M_tgt_arr[ti],M_src_arr[si]) for si in range(N)])
            preds=model.predict(X_t)
            best_idx=int(preds.argmax())
            ratios.append(tr(t,src_ctxs[best_idx])/piv[t].max())
            picks.append(src_ctxs[best_idx])
            # rank of true-best source by predicted overlap
            true_best_src=max(src_ctxs,key=lambda s: tr(t,s)/piv[t].max())
            tb_idx=src_ctxs.index(true_best_src)
            rank_truebest=int((preds>preds[tb_idx]).sum())+1
            truths.append((true_best_src,rank_truebest))
        results[picker]=(np.mean(ratios),ratios,picks,truths)
    return rf,lr,results

# 11-metric
M_src_11=cm[MN11].to_numpy(float); M_tgt_11=tg_def[MN11].to_numpy(float)
M_src_9 =cm[MN9].to_numpy(float);  M_tgt_9 =tg_def[MN9].to_numpy(float)
M_src_114=src_114[usable114].to_numpy(float); M_tgt_114=tg_114[usable114].to_numpy(float)

print(f'\n=== (B) Random Forest predicting top-1% overlap (mean Miyabi ratio) ===')
for name,Ms,Mt,feat_labels in [('11 metrics',M_src_11,M_tgt_11,[f'|diff|_{m}' for m in MN11]+[f'mean_{m}' for m in MN11]),
                                ('9 metrics no-IOPS',M_src_9,M_tgt_9,[f'|diff|_{m}' for m in MN9]+[f'mean_{m}' for m in MN9]),
                                ('114 metrics',M_src_114,M_tgt_114,[f'|diff|_{m}' for m in usable114]+[f'mean_{m}' for m in usable114])]:
    rf,lr,results=train_and_score(Ms,Mt,name)
    print(f'\n  {name}:  RF={results["RF"][0]:.3f}   Linear={results["Linear"][0]:.3f}   ({time.time()-t0:.0f}s)')
    print(f'    per-target RF ratios:',[round(x,3) for x in results['RF'][1]])
    if '11 metrics' in name and '9' not in name:
        imp=sorted(zip(feat_labels,rf.feature_importances_),key=lambda x:x[1],reverse=True)
        print(f'    TOP-10 RF feature importances:')
        for f,i in imp[:10]: print(f'      {i:.3f}  {f}')

print('\n=== fixed references (from prior runs) ===')
print('  raw (11):                            0.821')
print('  OT (114 metrics, binned):            0.845')
print('  embed 11 (concordance, reseeded):    0.801 ± 0.041')
print('  embed 9  (concordance, no IOPS):     0.837 ± 0.061')
print('  embed 11 (top-1% overlap signal):    0.759 ± 0.064')
print('  embed 9  (top-1% overlap, no IOPS):  0.810 ± 0.072')
print('  top-1% overlap DIRECT (full sweep):  0.923')
print('  ORACLE:                              0.994')
print(f'\ndone ({time.time()-t0:.0f}s)')
