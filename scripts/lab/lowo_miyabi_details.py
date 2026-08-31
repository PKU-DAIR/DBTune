import pandas as pd, numpy as np, glob, os, math, time, warnings, re
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

PARAMS=['innodb_buffer_pool_size','innodb_read_io_threads','innodb_write_io_threads','innodb_flush_log_at_trx_commit','innodb_adaptive_hash_index','sync_binlog','innodb_lru_scan_depth','innodb_buffer_pool_instances','innodb_change_buffer_max_size','innodb_io_capacity','innodb_log_file_size','table_open_cache']
SHORT=['bp','rio','wio','fl','ah','sb','lru','bpi','cbm','ioc','lf','toc']
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
miyabi=sorted(c for c in ctx_tps if c.startswith('112c125g_'))
N=len(src); M=len(miyabi)
piv=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx').dropna()
bestcfg={s:piv[s].idxmax() for s in piv.columns}
DK='1.0|1.0|2.0|1.0|1.0|1.0|1024.0|1.0|25.0|100.0|0.048|4000.0'
COLS114=[c for c in alld.columns if c not in {'ctx','ck','tps'}]
def_all=alld[alld.ck==DK].groupby('ctx')[COLS114].mean()
src_df=def_all.loc[src]; tgt_df=def_all.loc[miyabi]
usable114=[c for c in COLS114 if src_df[c].notna().all() and tgt_df[c].notna().all()]
src_M114=src_df[usable114].to_numpy(float); tgt_M114=tgt_df[usable114].to_numpy(float)

def wsig(c): p=c.split('_',1); return p[1] if len(p)==2 else c
def top_set(c,frac=0.01):
    t=ctx_tps[c]; k=max(1,math.ceil(len(t)*frac))
    return set(x for x,_ in sorted(t.items(),key=lambda x:x[1],reverse=True)[:k])
T1={c:top_set(c) for c in src+miyabi}
S_top1_ss=np.zeros((N,N))
for i,a in enumerate(src):
    for j,b in enumerate(src):
        if i!=j: S_top1_ss[i,j]=len(T1[a]&T1[b])/max(1,len(T1[a]|T1[b]))
S_top1_ts=np.zeros((M,N))
for ti,t in enumerate(miyabi):
    for sj,s in enumerate(src):
        S_top1_ts[ti,sj]=len(T1[t]&T1[s])/max(1,len(T1[t]|T1[s]))

def feats(a,b): return np.concatenate([np.abs(a-b),(a+b)/2])
pair_i=[]; pair_j=[]; X_full=[]; y_full=[]
for i in range(N):
    for j in range(N):
        if i==j: continue
        pair_i.append(i); pair_j.append(j)
        X_full.append(feats(src_M114[i],src_M114[j])); y_full.append(S_top1_ss[i,j])
pair_i=np.array(pair_i); pair_j=np.array(pair_j)
X_full=np.array(X_full,dtype=np.float32); y_full=np.array(y_full,dtype=np.float32)

shared_keys=set(piv.index)
def compute_ot_bins(non_excluded):
    rows=alld[alld.ctx.isin(non_excluded)&alld.ck.isin(shared_keys)][usable114].dropna().to_numpy(float)
    edges=np.percentile(rows,np.linspace(0,100,11),axis=0)
    def binv(V):
        B=np.empty_like(V,dtype=int)
        for j in range(V.shape[1]): B[:,j]=np.clip(np.digitize(V[:,j],edges[:,j],right=True),1,10)
        return B
    return binv(src_M114).astype(float),binv(tgt_M114).astype(float)

def parse_cfg(ck):
    v=ck.split('|'); return dict(zip(SHORT,[float(x) for x in v]))
DEF_CFG=parse_cfg(DK)
def fmt_knob(s,val,defv):
    if abs(val-defv)<1e-9: return None    # at default
    if s=='bp': return f'bp={val:.0f}GB'
    if s=='ah': return f'ah={"ON" if val>0.5 else "OFF"}'
    if s=='lf':
        return f'lf={val*1000:.0f}MB' if val<1 else f'lf={val:.1f}GB'
    return f'{s}={int(val) if val==int(val) else val}'
def cfg_delta(ck):
    c=parse_cfg(ck)
    deltas=[fmt_knob(s,c[s],DEF_CFG[s]) for s in SHORT]
    deltas=[d for d in deltas if d]
    return ', '.join(deltas) if deltas else '(default)'
def short_src(c):
    return (c.replace('_64-1000000-4-oltp_','/1M/')
             .replace('_64-100000-4-oltp_','/100k/').replace('-4-4-tpcc-nan','/tpcc'))

# Per-target detailed analysis
print(f'Default config: {", ".join(f"{s}={DEF_CFG[s]}" for s in SHORT)}\n')
for ti,t in enumerate(miyabi):
    target_ws=wsig(t); short_t=t.replace('112c125g_','')
    excluded=[i for i,s in enumerate(src) if wsig(s)==target_ws]
    cand_idx=[i for i in range(N) if i not in set(excluded)]
    cand_a=np.array(cand_idx); cand_ctx=set(src[i] for i in cand_idx)
    # Train RF (no embed needed)
    mask=np.isin(pair_i,cand_idx)&np.isin(pair_j,cand_idx)
    rf=RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=-1).fit(X_full[mask],y_full[mask])
    B_src,B_tgt=compute_ot_bins(cand_ctx)
    pick_ot=cand_idx[int(cdist(B_tgt[ti:ti+1],B_src[cand_a]).argmin())]
    pick_t1=cand_idx[int(np.argmax(S_top1_ts[ti,cand_a]))]
    X_test=np.array([feats(tgt_M114[ti],src_M114[s]) for s in cand_idx],dtype=np.float32)
    pick_rf=cand_idx[int(rf.predict(X_test).argmax())]
    or_ratios=[piv.at[bestcfg[src[s]],t] for s in cand_idx]
    pick_or=cand_idx[int(np.argmax(or_ratios))]
    b=piv[t].max(); d=piv.at[DK,t]
    print(f'\n========================= {short_t} =========================')
    print(f'  default TPS = {d:.1f}     best TPS = {b:.1f}    headroom = {b/d:.2f}x   ({len(excluded)} same-workload sources excluded)\n')
    print(f'  {"method":8s} {"picked source":42s} {"transferred TPS":>15s} {"vs default":>11s} {"config (deltas from default)":40s}')
    for name,pk in [('OT',pick_ot),('RF',pick_rf),('top-1%',pick_t1),('oracle',pick_or)]:
        psrc=src[pk]; cfg=bestcfg[psrc]; tps=piv.at[cfg,t]
        print(f'  {name:8s} {short_src(psrc)[:42]:42s} {tps:15.1f} {tps/d:10.2f}x   {cfg_delta(cfg)[:60]}')

