import pandas as pd, numpy as np, glob, os, warnings, math, time
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
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
ctxs=list(ctx_tps); tgt=[c for c in ctxs if c.startswith('112c125g_')]; src=[c for c in ctxs if not c.startswith('112c125g_')]
piv=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx').dropna()
bestcfg={s:piv[s].idxmax() for s in piv.columns}; tr=lambda t,s: piv.at[bestcfg[s],t]
print(f'loaded ({time.time()-t0:.0f}s) | total unique configs in union: {len(set(alld["ck"]))}',flush=True)

# per-context top-K% set
def top_set(ctx, frac):
    tps=ctx_tps[ctx]; n=len(tps); k=max(1,math.ceil(n*frac))
    sortedcks=sorted(tps.items(),key=lambda x:x[1],reverse=True)
    return set(c for c,_ in sortedcks[:k]),k

def jaccard(a,b):
    if not a and not b: return 0.0
    return len(a&b)/max(1,len(a|b))

# ---- core: top-5% Jaccard for all (target, source) pairs ----
T5={c:top_set(c,0.05) for c in ctxs}                              # ctx -> (set, k)
rows=[]
for tc in tgt:
    Tt,kt=T5[tc]; ttps=ctx_tps[tc]
    for sc in src:
        Ts,ks=T5[sc]; inter=Tt&Ts; uni=Tt|Ts
        rows.append(dict(target=tc,source=sc,
            n_top_target=kt,n_top_source=ks,
            intersect=len(inter),union=len(uni),
            top5_overlap=round(len(inter)/max(1,len(uni)),6)))
ov=pd.DataFrame(rows)
ov.to_csv('scripts/lab/top5_overlap_miyabi.csv',index=False)
print(f'wrote scripts/lab/top5_overlap_miyabi.csv ({len(ov)} pairs)',flush=True)

# transferred_ratio per pair (for the comparison & correlation)
def trate(t,s): return tr(t,s)/piv[t].max()
ov['transferred_ratio']=ov.apply(lambda r:trate(r.target,r.source),axis=1)

# ---- (A) per-target pick by top-5% overlap, compare to other methods ----
prior=pd.read_csv('scripts/lab/transfer_quality_miyabi.csv').set_index('target')
conc=pd.read_csv('scripts/lab/concordance_miyabi.csv')
print('\n===== per-target: pick by TOP-5% OVERLAP vs other methods (transferred/best) =====')
print(f'  {"target":36s}{"default":>9s}{"raw":>7s}{"embed":>7s}{"ot114":>7s}{"conc":>7s}{"top5%":>7s}{"oracle":>8s}   top5%_picked  (overlap)')
mean={k:[] for k in ['default','raw','embed','ot114','conc','top5','oracle']}
for tc in tgt:
    sub=ov[ov.target==tc].sort_values('top5_overlap',ascending=False)
    pick=sub.iloc[0]['source']; ov_s=sub.iloc[0]['top5_overlap']
    top5_r=trate(tc,pick)
    short_t=tc.replace('112c125g_','')
    rowp=prior.loc[short_t]
    csub=conc[(conc.target==tc)&conc.similarity_score.notna()].sort_values('similarity_score',ascending=False)
    cpick=csub.iloc[0]['source']; conc_r=trate(tc,cpick)
    short=lambda c:(c.replace('_64-1000000-4-oltp_','').replace('_64-100000-4-oltp_','*').replace('-4-4-tpcc-nan','wh-tpcc'))
    print(f'  {short_t:36s}{rowp.default_ratio:9.3f}{rowp.raw:7.3f}{rowp.embed:7.3f}{rowp.ot114:7.3f}{conc_r:7.3f}{top5_r:7.3f}{rowp.oracle:8.3f}   {short(pick)}  ({ov_s:.3f})')
    for k,v in [('default',rowp.default_ratio),('raw',rowp.raw),('embed',rowp.embed),('ot114',rowp.ot114),('conc',conc_r),('top5',top5_r),('oracle',rowp.oracle)]:
        mean[k].append(v)
print(f'  {"MEAN":36s}{np.mean(mean["default"]):9.3f}{np.mean(mean["raw"]):7.3f}{np.mean(mean["embed"]):7.3f}{np.mean(mean["ot114"]):7.3f}{np.mean(mean["conc"]):7.3f}{np.mean(mean["top5"]):7.3f}{np.mean(mean["oracle"]):8.3f}')

# ---- (B) Spearman: does top-5% overlap predict transferred_ratio? compare to concordance ----
print('\n===== Spearman ρ vs transferred_ratio (top-5% overlap vs concordance) =====')
print(f'  {"target":36s}{"ρ_conc":>10s}{"ρ_top5%":>10s}{"oracle rank by top5%":>22s}')
for tc in tgt:
    o=ov[ov.target==tc].copy(); c=conc[(conc.target==tc)&conc.similarity_score.notna()].copy()
    merged=o.merge(c[['source','similarity_score']],on='source')
    rho_c,_=spearmanr(merged['similarity_score'],merged['transferred_ratio'])
    rho_o,_=spearmanr(merged['top5_overlap'],merged['transferred_ratio'])
    orc=merged.sort_values('transferred_ratio',ascending=False).iloc[0]
    rank_orc=int((merged['top5_overlap'] > orc['top5_overlap']).sum())+1
    print(f'  {tc.replace("112c125g_",""):36s}{rho_c:10.3f}{rho_o:10.3f}{rank_orc:22d}')

# ---- (C) Sensitivity: top-K% for K in {1, 2, 5, 10, 20} ----
print('\n===== Sensitivity: pick by top-K% overlap, mean transferred over 5 Miyabi targets =====')
print(f'  {"K%":>5s}{"mean_pick_ratio":>17s}{"mean_Spearman_ρ":>17s}')
for K in [1,2,5,10,20]:
    TK={c:top_set(c,K/100.0)[0] for c in ctxs}
    means=[]; rhos=[]
    for tc in tgt:
        pairs=[(sc,jaccard(TK[tc],TK[sc]),trate(tc,sc)) for sc in src]
        pdf=pd.DataFrame(pairs,columns=['src','ov','tr'])
        pick=pdf.sort_values('ov',ascending=False).iloc[0]
        means.append(pick['tr']); rhos.append(spearmanr(pdf['ov'],pdf['tr'])[0])
    print(f'  {K:5d}{np.mean(means):17.3f}{np.mean(rhos):17.3f}')
print(f'\ndone ({time.time()-t0:.0f}s)')
