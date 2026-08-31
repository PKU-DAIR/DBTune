import pandas as pd, numpy as np, glob, os, warnings, time
warnings.filterwarnings('ignore')
from scipy.stats import kendalltau
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
    df=pd.read_csv(f); df=df.loc[:,~df.columns.duplicated()]
    hw=os.path.basename(f).replace('-result.csv','')
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
# per-context dict ck->tps (use max if duplicate)
ctx_tps={}
for c,g in alld.groupby('ctx'):
    ctx_tps[c]=g.groupby('ck')['tps'].max().to_dict()
ctxs=list(ctx_tps); tgt=[c for c in ctxs if c.startswith('112c125g_')]; src=[c for c in ctxs if not c.startswith('112c125g_')]
print(f'loaded: {len(src)} sources + {len(tgt)} miyabi targets in {time.time()-t0:.0f}s',flush=True)

# pairwise Kendall tau between each target and each source on their PAIR-WISE shared configs
rows=[]
for ti,tc in enumerate(tgt):
    t_keys=ctx_tps[tc]; t_set=set(t_keys)
    for sc in src:
        s_keys=ctx_tps[sc]; shared=t_set & set(s_keys)
        n=len(shared)
        if n<10: 
            rows.append(dict(target=tc,source=sc,n_shared=n,kendalltau=np.nan,similarity_score=np.nan))
            continue
        x=np.fromiter((t_keys[k] for k in shared),float,n)
        y=np.fromiter((s_keys[k] for k in shared),float,n)
        tau,_=kendalltau(x,y)
        rows.append(dict(target=tc,source=sc,n_shared=n,kendalltau=round(float(tau),6),
                         similarity_score=round((1+float(tau))/2,6)))
    print(f'  target {ti+1}/{len(tgt)} {tc[:50]}  ({time.time()-t0:.0f}s)',flush=True)
df=pd.DataFrame(rows)
out='scripts/lab/concordance_miyabi.csv'; df.to_csv(out,index=False)
print(f'\nwrote {out} ({len(df)} pairs) in {time.time()-t0:.0f}s',flush=True)

# ----- transferred-config TPS for concordance pick, comparable to prior table -----
piv=alld.groupby(['ctx','ck'])['tps'].max().unstack('ctx').dropna()  # global shared grid
ctxs_g=list(piv.columns); bestcfg={s:piv[s].idxmax() for s in ctxs_g}
tr=lambda c,s: piv.at[bestcfg[s],c]
dk='1.0|1.0|2.0|1.0|1.0|1.0|1024.0|1.0|25.0|100.0|0.048|4000.0'
prior=pd.read_csv('scripts/lab/transfer_quality_miyabi.csv').set_index('target')
def short(c):
    return (c.replace('_64-1000000-4-oltp_','').replace('_64-100000-4-oltp_','*')
             .replace('-4-4-tpcc-nan','wh-tpcc'))
print('\n===== concordance pick vs prior methods on Miyabi (transferred/best) =====')
print(f'  {"target":36s}{"default":>9s}{"raw":>7s}{"embed":>7s}{"ot114":>7s}{"conc":>7s}{"oracle":>8s}   conc_picked  (sim_score, n_shared)')
mean_acc={'default':[],'raw':[],'embed':[],'ot114':[],'conc':[],'oracle':[]}
for tc in tgt:
    sub=df[(df.target==tc)&df.similarity_score.notna()].sort_values('similarity_score',ascending=False)
    pick=sub.iloc[0]['source']; ss=sub.iloc[0]['similarity_score']; ns=int(sub.iloc[0]['n_shared'])
    b=piv[tc].max(); conc_r=tr(tc,pick)/b
    short_t=tc.replace('112c125g_','')
    row=prior.loc[short_t]
    print(f'  {short_t:36s}{row.default_ratio:9.3f}{row.raw:7.3f}{row.embed:7.3f}{row.ot114:7.3f}{conc_r:7.3f}{row.oracle:8.3f}   {short(pick)}  ({ss:.3f}, n={ns})')
    for k,v in [('default',row.default_ratio),('raw',row.raw),('embed',row.embed),('ot114',row.ot114),('conc',conc_r),('oracle',row.oracle)]:
        mean_acc[k].append(v)
print(f'  {"MEAN":36s}{np.mean(mean_acc["default"]):9.3f}{np.mean(mean_acc["raw"]):7.3f}{np.mean(mean_acc["embed"]):7.3f}{np.mean(mean_acc["ot114"]):7.3f}{np.mean(mean_acc["conc"]):7.3f}{np.mean(mean_acc["oracle"]):8.3f}')
