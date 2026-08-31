import pandas as pd, numpy as np, glob, os, warnings, time
warnings.filterwarnings('ignore')
from scipy.stats import kendalltau, spearmanr
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
piv_idx=list(piv.index); bestcfg={s:piv[s].idxmax() for s in piv.columns}
tr=lambda t,s: piv.at[bestcfg[s],t]
print(f'loaded ({time.time()-t0:.0f}s)',flush=True)

# Load existing pairwise concordance
conc=pd.read_csv('scripts/lab/concordance_miyabi.csv')

# Compute transferred_ratio for ALL (target, source) pairs
rows=[]
for tc in tgt:
    b=piv[tc].max()
    for sc in src: rows.append(dict(target=tc,source=sc,transferred=tr(tc,sc),trans_ratio=tr(tc,sc)/b))
trans=pd.DataFrame(rows)
df=conc.merge(trans,on=['target','source'],how='left')

# ---- (1) Spearman correlation per target between concordance similarity and transferred_ratio ----
print('\n=== (A) Does similarity_score predict transferred_ratio? Spearman ρ per target ===')
print(f'  {"target":36s}{"n":>5s}{"Spearman ρ":>13s}{"p-value":>12s}')
def sht(c): return c.replace('112c125g_','')[:36]
for tc in tgt:
    sub=df[(df.target==tc)&df.similarity_score.notna()&df.trans_ratio.notna()]
    rho,p=spearmanr(sub['similarity_score'],sub['trans_ratio'])
    print(f'  {sht(tc):36s}{len(sub):5d}{rho:13.3f}{p:12.2e}')

# ---- (2) Where does the ORACLE source rank by concordance? And how does similarity compare across picks? ----
print('\n=== (B) For each target: rank of ORACLE source by similarity (1 = top concordance) ===')
print(f'  {"target":36s}{"oracle source":36s}{"oracle sim":>11s}{"oracle sim_rank":>16s}{"oracle tr":>10s}')
for tc in tgt:
    sub=df[(df.target==tc)&df.similarity_score.notna()&df.trans_ratio.notna()].copy()
    sub['sim_rank']=sub['similarity_score'].rank(ascending=False,method='min').astype(int)
    sub['tr_rank']=sub['trans_ratio'].rank(ascending=False,method='min').astype(int)
    orc=sub.sort_values('trans_ratio',ascending=False).iloc[0]
    print(f'  {sht(tc):36s}{orc["source"][:34]:36s}{orc["similarity_score"]:11.3f}{int(orc["sim_rank"]):16d}{orc["trans_ratio"]:10.3f}')

# ---- (3) Top-10 by similarity vs top-10 by transferred — overlap? ----
print('\n=== (C) Top-10 by SIMILARITY vs Top-10 by TRANSFERRED (overlap = how often best concordance is best transfer) ===')
print(f'  {"target":36s}{"top-10 overlap":>16s}{"top-30 overlap":>16s}{"max trans in top-10 by sim":>30s}')
for tc in tgt:
    sub=df[(df.target==tc)&df.similarity_score.notna()&df.trans_ratio.notna()].copy()
    top10_sim=set(sub.nlargest(10,'similarity_score')['source'])
    top10_tr =set(sub.nlargest(10,'trans_ratio')['source'])
    top30_sim=set(sub.nlargest(30,'similarity_score')['source'])
    top30_tr =set(sub.nlargest(30,'trans_ratio')['source'])
    best_in_top10sim=sub[sub['source'].isin(top10_sim)]['trans_ratio'].max()
    print(f'  {sht(tc):36s}{len(top10_sim&top10_tr):16d}{len(top30_sim&top30_tr):16d}{best_in_top10sim:30.3f}')

# ---- (4) Top-K concordance (rank agreement on the SOURCE'S TOP configs only) ----
# Hypothesis: only top configs matter for one-shot argmax transfer. Use top-20% of source by TPS.
def top_k_tau(t_tps, s_tps, shared, frac):
    # restrict shared to source's top frac
    s_sorted=sorted(shared, key=lambda k: s_tps[k], reverse=True)
    keep=set(s_sorted[:max(2,int(len(s_sorted)*frac))])
    if len(keep)<5: return np.nan
    xs=np.fromiter((t_tps[k] for k in keep),float,len(keep))
    ys=np.fromiter((s_tps[k] for k in keep),float,len(keep))
    return kendalltau(xs,ys)[0]
print('\n=== (D) Concordance restricted to top-X% of source configs ===')
print(f'  {"target":36s}{"full ρ":>9s}{"top10% ρ":>10s}{"top20% ρ":>10s}{"top50% ρ":>10s} | best by top20% picker:')
for tc in tgt:
    t_tps=ctx_tps[tc]; t_set=set(t_tps); b=piv[tc].max()
    pairs=[]
    for sc in src:
        s_tps=ctx_tps[sc]; shared=t_set & set(s_tps)
        if len(shared)<20: continue
        full=kendalltau(np.fromiter((t_tps[k] for k in shared),float,len(shared)),
                        np.fromiter((s_tps[k] for k in shared),float,len(shared)))[0]
        t10=top_k_tau(t_tps,s_tps,shared,0.10)
        t20=top_k_tau(t_tps,s_tps,shared,0.20)
        t50=top_k_tau(t_tps,s_tps,shared,0.50)
        pairs.append((sc,full,t10,t20,t50,tr(tc,sc)/b))
    pdf=pd.DataFrame(pairs,columns=['src','full','t10','t20','t50','trans'])
    # Spearman of each variant with transferred
    rho_full,_=spearmanr(pdf['full'],pdf['trans'])
    rho_t10,_=spearmanr(pdf['t10'].fillna(pdf['t10'].mean()),pdf['trans'])
    rho_t20,_=spearmanr(pdf['t20'].fillna(pdf['t20'].mean()),pdf['trans'])
    rho_t50,_=spearmanr(pdf['t50'].fillna(pdf['t50'].mean()),pdf['trans'])
    pick20=pdf.loc[pdf['t20'].idxmax()]
    print(f'  {sht(tc):36s}{rho_full:9.3f}{rho_t10:10.3f}{rho_t20:10.3f}{rho_t50:10.3f} | t20-pick={pick20["src"][:30]} → {pick20["trans"]:.3f}')

print(f'\ndone ({time.time()-t0:.0f}s)')
