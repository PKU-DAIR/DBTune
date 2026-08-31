import json, glob, os, re, time, warnings
import numpy as np, pandas as pd, torch, joblib
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
import torch.nn as nn
t0=time.time()

class EmbeddingNet(nn.Module):
    def __init__(s,i=11,e=16):
        super().__init__(); s.net=nn.Sequential(nn.Linear(i,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,e))
    def forward(s,x): return nn.functional.normalize(s.net(x),p=2,dim=1)
MN=['Average Memory Usage Percentage','InnoDB Buffer Pool Cache Hit Rate','InnoDB Dirty Buffer Pages','Current QPS (Queries Per Second)','Max CPU Usage (100 - Idle)','InnoDB Rows Deleted (60s Rate)','InnoDB Rows Inserted (60s Rate)','InnoDB Rows Read (60s Rate)','InnoDB Rows Updated (60s Rate)','Average Disk IOPS (Read)','Average Disk IOPS (Write)']
PARAMS=['innodb_buffer_pool_size','innodb_read_io_threads','innodb_write_io_threads','innodb_flush_log_at_trx_commit','innodb_adaptive_hash_index','sync_binlog','innodb_lru_scan_depth','innodb_buffer_pool_instances','innodb_change_buffer_max_size','innodb_io_capacity','innodb_log_file_size','table_open_cache']
def fam(c):
    m=re.search(r'oltp_(read_only|read_write_\d+|write_only)|tpcc',c); return m.group(0) if m else '?'
def hw(c): return c.split('_')[0]
def tsz(c):
    m=re.search(r'64-(\d+)-4-oltp',c); return m.group(1) if m else ('tpcc' if 'tpcc' in c else '?')
def ckey(cfg): return '|'.join(str(round(float(cfg[p]),6)) for p in PARAMS)

dml12=json.load(open('scripts/experiment/gen_knobs/DML_12.json'))
default_key='|'.join(str(round(float(dml12[p]['default']),6)) for p in PARAMS)

# ---- load all csv_source JSONs: per context -> {config_key: [tps, im(114)]} ----
files=sorted(glob.glob('scripts/DBTune_history/csv_source/history_*.json'))
ctx_tps={}; ctx_im={}              # ctx -> dict config_key->tps ; ctx -> dict config_key->im(np float32)
for f in files:
    c=os.path.basename(f)[len('history_'):-len('.json')]
    d=json.load(open(f)); tpsd={}; imd={}
    for o in d['data']:
        k=ckey(o['configuration']); t=o['external_metrics'].get('tps')
        if t is None: continue
        t=float(t)
        if k not in tpsd or t>tpsd[k]:        # keep max-tps row per config
            tpsd[k]=t; imd[k]=np.asarray(o['internal_metrics'],dtype=np.float32)
    ctx_tps[c]=tpsd; ctx_im[c]=imd
ctxs=sorted(ctx_tps)
print(f'loaded {len(ctxs)} contexts in {time.time()-t0:.0f}s',flush=True)

# ---- tps pivot on shared config grid ----
piv=pd.DataFrame(ctx_tps).dropna()       # index=config_key shared across ALL contexts
shared=list(piv.index)
print('shared grid:', piv.shape, '| default in grid:', default_key in piv.index,flush=True)
pv=piv
bestcfg={s:pv[s].idxmax() for s in ctxs}
tr=lambda c,s: pv.at[bestcfg[s],c]

# ---- 11-metric selection inputs (raw + embed), model = dml_models_all ----
base='autotune/optimizer/dml_models_all/'
cm=pd.read_csv(base+'context_default_metrics_all.csv').set_index('context_id').loc[ctxs]
sc=joblib.load(base+'scaler.pkl')
model=EmbeddingNet(); model.load_state_dict(torch.load(base+'context_model.pth',map_location='cpu')); model.eval()
Xs=sc.transform(cm[MN]); emb=model(torch.tensor(Xs,dtype=torch.float32)).detach().numpy()

# ---- OtterTune: all 114 metrics, default-config match ----
# default 114-vector per context (mean over default_key rows; here single)
defIM=np.vstack([ctx_im[c].get(default_key, np.full(114,np.nan,np.float32)) for c in ctxs])
# global decile edges from all observations on the shared grid (representative population)
pop=np.vstack([ctx_im[c][k] for c in ctxs for k in shared if k in ctx_im[c]])
edges=np.percentile(pop, np.linspace(0,100,11), axis=0)   # (11,114)
del pop
def binv(V):
    B=np.empty_like(V,dtype=np.int16)
    for j in range(V.shape[1]): B[:,j]=np.clip(np.digitize(V[:,j],edges[:,j],right=True),1,10)
    return B
defB=binv(defIM)                       # (n,114) binned default vectors
print(f'OT edges + binning done {time.time()-t0:.0f}s',flush=True)

idx={c:i for i,c in enumerate(ctxs)}
def select(c,cand,m):
    ci=idx[c]; cj=[idx[s] for s in cand]
    if m=='raw': d=cdist(Xs[ci:ci+1],Xs[cj]).flatten()
    elif m=='emb': d=cdist(emb[ci:ci+1],emb[cj]).flatten()
    elif m=='ot': d=cdist(defB[ci:ci+1].astype(float),defB[cj].astype(float)).flatten()
    return cand[int(d.argmin())]

TARGET_FAMS={'oltp_read_only','oltp_read_write_50','oltp_write_only','tpcc'}
rows=[]
for c in ctxs:
    if fam(c) not in TARGET_FAMS: continue
    cand=[s for s in ctxs if fam(s)!=fam(c)]        # leave-one-FAMILY-out; pool = all other families
    b=pv[c].max()
    sr=select(c,cand,'raw'); se=select(c,cand,'emb'); so=select(c,cand,'ot')
    rows.append(dict(
        target=c, family=fam(c), table_size=tsz(c), hw=hw(c),
        default_ratio=round(pv.at[default_key,c]/b,4),
        raw_ratio=round(tr(c,sr)/b,4), raw_src=sr,
        embed_ratio=round(tr(c,se)/b,4), embed_src=se,
        ot_ratio=round(tr(c,so)/b,4), ot_src=so,
        oracle_ratio=round(max(tr(c,s) for s in cand)/b,4),
    ))
df=pd.DataFrame(rows)
out='scripts/lab/transfer_quality_allsrc.csv'; df.to_csv(out,index=False)
print(f'\nwrote {out} ({len(df)} target rows) in {time.time()-t0:.0f}s\n',flush=True)
# summary
print('mean ratio by target family (leave-one-family-out, pool=all 293 contexts):')
print(f'  {"family":20s}{"n":>4s}{"raw":>8s}{"embed":>8s}{"ot(114)":>9s}{"oracle":>8s}{"default":>9s}')
for f in ['oltp_read_only','oltp_read_write_50','oltp_write_only','tpcc']:
    s=df[df.family==f]
    print(f'  {f:20s}{len(s):4d}{s.raw_ratio.mean():8.3f}{s.embed_ratio.mean():8.3f}{s.ot_ratio.mean():9.3f}{s.oracle_ratio.mean():8.3f}{s.default_ratio.mean():9.3f}')
s=df
print(f'  {"ALL":20s}{len(s):4d}{s.raw_ratio.mean():8.3f}{s.embed_ratio.mean():8.3f}{s.ot_ratio.mean():9.3f}{s.oracle_ratio.mean():8.3f}{s.default_ratio.mean():9.3f}')
