#!/usr/bin/env python3
"""
Precompute RF-based source picks for the Miyabi tuning experiment.

For each target workload (read_only / rw50 / write_only, all 1M tables, skew=0.2),
train a RandomForestRegressor that predicts pairwise top-1% Jaccard overlap from
[|metric_diff|, metric_mean] features over the 114 default-config metrics, using
all source-source pairs EXCEPT those matching the target's (workload, skew,
table_size) AND excluding any 112c125g (Miyabi) data. Then, at inference, build
features for (Miyabi target, each candidate source) and pick the source with the
highest predicted overlap. Write the picks to scripts/lab/rf_picks.json so the
parallel launcher can inject them via FORCE_SOURCE_CONTEXT.
"""
import pandas as pd, numpy as np, glob, os, math, json, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor

PARAMS = ['innodb_buffer_pool_size','innodb_read_io_threads','innodb_write_io_threads',
          'innodb_flush_log_at_trx_commit','innodb_adaptive_hash_index','sync_binlog',
          'innodb_lru_scan_depth','innodb_buffer_pool_instances','innodb_change_buffer_max_size',
          'innodb_io_capacity','innodb_log_file_size','table_open_cache']

# Targets: (short_name, workload_str, table_size_str, skew_str)
TARGETS = [
    ('read',  'oltp_read_only',     '1000000', '0.2'),
    ('rw50',  'oltp_read_write_50', '1000000', '0.2'),
    ('write', 'oltp_write_only',    '1000000', '0.2'),
]

def nf(v):
    s = str(v).strip()
    if s in ('ON','Yes','True'):   return 1.0
    if s in ('OFF','No','False'):  return 0.0
    if s.endswith('GB'): return float(s[:-2])
    if s.endswith('MB'): return float(s[:-2]) / 1000.0
    if s.endswith('KB'): return float(s[:-2]) / 1e6
    try: return float(s)
    except: return np.nan

def load(f):
    df = pd.read_csv(f); df = df.loc[:, ~df.columns.duplicated()]
    hw = os.path.basename(f).replace('-result.csv', '')
    K = np.array([[nf(x) for x in df[p]] for p in PARAMS]).T
    ok = np.isfinite(K).all(1) & pd.to_numeric(df['tps'], errors='coerce').notna().values
    df = df[ok].reset_index(drop=True); K = K[ok]
    ck = ['|'.join(str(round(v, 6)) for v in row) for row in K]
    if 'workload_label' in df.columns:
        wl = df['workload_label'].astype(str)
    else:
        sk = df['skew'].apply(lambda x: 'nan' if pd.isna(x) else
                              (str(int(x)) if float(x) == int(x) else str(x)))
        wl = (df['num_table'].astype(int).astype(str) + '-' +
              df['table_size'].astype(int).astype(str) + '-' +
              df['num_client'].astype(int).astype(str) + '-' +
              df['workload'].astype(str) + '-' + sk)
    out = pd.DataFrame({'ctx': hw + '_' + wl, 'ck': ck,
                        'tps': pd.to_numeric(df['tps'], errors='coerce').values})
    for m in pd.read_csv(f, nrows=0).columns[19:]:
        out[m] = pd.to_numeric(df[m], errors='coerce').values if m in df.columns else np.nan
    return out

def ctx_matches_exclusion(ctx, wl_str, ts_str, sk_str):
    """Source context_id format: <hw>_<num_table>-<table_size>-<num_client>-<workload>-<skew>."""
    # exclude if the context_id contains '-{ts_str}-...-{wl_str}-{sk_str}'
    needle = f'-{ts_str}-4-{wl_str}-{sk_str}'
    return needle in ctx

def build_pair_feats(m_i, m_j):
    a = np.asarray(m_i, float); b = np.asarray(m_j, float)
    return np.concatenate([np.abs(a - b), (a + b) / 2.0])

def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    files = sorted(glob.glob(os.path.join(root, 'DBMSTransferLearning/dataset/full_data/*-result.csv')))
    alld = pd.concat([load(f) for f in files], ignore_index=True)

    # split contexts: sources (everything not 112c125g), targets (112c125g)
    all_ctxs = sorted(alld['ctx'].unique())
    src_pool = [c for c in all_ctxs if not c.startswith('112c125g_')]
    target_ctxs = {short: f'112c125g_64-{ts}-4-{wl}-{sk}'
                   for (short, wl, ts, sk) in TARGETS}
    print(f'[rf-picks] {len(src_pool)} sources, target ctxs:')
    for k, v in target_ctxs.items():
        print(f'  {k:6s} -> {v}')

    # top-1% per context
    def top1(ctx):
        sub = alld[alld['ctx'] == ctx]
        tps = sub.groupby('ck')['tps'].max()
        k = max(1, math.ceil(len(tps) * 0.01))
        return set(tps.nlargest(k).index)
    T1 = {c: top1(c) for c in all_ctxs}

    # default-config 114-metric vectors (mean over default-config rows)
    DK = '1.0|1.0|2.0|1.0|1.0|1.0|1024.0|1.0|25.0|100.0|0.048|4000.0'
    METRICS = list(pd.read_csv(files[0], nrows=0).columns)[19:]  # 114 names
    defrow = alld[alld['ck'] == DK].groupby('ctx')[METRICS].mean()

    picks = {}
    for short, wl_str, ts_str, sk_str in TARGETS:
        tgt = target_ctxs[short]
        # Apply leave-one-(workload, skew, tables)-out
        usable_src = [c for c in src_pool
                      if not ctx_matches_exclusion(c, wl_str, ts_str, sk_str)]
        excluded = [c for c in src_pool if c not in usable_src]
        print(f'\n[rf-picks] target={short}  excluding {len(excluded)} ctxs: '
              f'{[c for c in excluded][:3]}...')

        # Usable 114 metric cols: not all-NaN in sources OR target
        sub = defrow.loc[usable_src + [tgt]]
        usable_cols = [m for m in METRICS if sub[m].notna().all()]
        print(f'           usable metric cols: {len(usable_cols)} / {len(METRICS)}')
        M_src = defrow.loc[usable_src, usable_cols].to_numpy(float)
        m_tgt = defrow.loc[tgt, usable_cols].to_numpy(float)

        # Build pair-wise training set: (i, j) both source contexts in usable_src.
        # Label = top-1% Jaccard overlap.
        N = len(usable_src)
        rows = []; ys = []
        for i in range(N):
            for j in range(N):
                if i == j: continue
                a, b = T1[usable_src[i]], T1[usable_src[j]]
                ov = len(a & b) / max(1, len(a | b))
                rows.append(build_pair_feats(M_src[i], M_src[j]))
                ys.append(ov)
        X = np.asarray(rows); y = np.asarray(ys)
        print(f'           training pairs: {len(y)}; mean overlap {y.mean():.3f}')

        rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
        rf.fit(X, y)

        # Predict (target, source) overlap for each usable source; pick argmax.
        X_t = np.asarray([build_pair_feats(m_tgt, M_src[s]) for s in range(N)])
        preds = rf.predict(X_t)
        best_idx = int(preds.argmax())
        picked = usable_src[best_idx]
        picks[short] = {
            'picked_source': picked,
            'predicted_overlap': float(round(preds[best_idx], 4)),
            'n_candidates': N,
            'n_excluded': len(excluded),
            'n_pairs_trained': len(y),
            'usable_metric_cols': len(usable_cols),
        }
        print(f'           PICK: {picked}  (predicted overlap {preds[best_idx]:.3f})')

    out = os.path.join(root, 'scripts/lab/rf_picks.json')
    with open(out, 'w') as f:
        json.dump(picks, f, indent=2)
    print(f'\n[rf-picks] wrote {out}')

if __name__ == '__main__':
    main()
