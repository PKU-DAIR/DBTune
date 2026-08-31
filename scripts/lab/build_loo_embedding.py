#!/usr/bin/env python3
"""Build a leave-one-out DML embedding that EXCLUDES every context whose id
contains <exclude_pattern>. Cached: if <output_dir>/context_model.pth exists,
does nothing. Filters the all-source triplet + context CSVs (drops any row that
references an excluded context), then runs scripts/train_dml_model.py.

usage: build_loo_embedding.py <exclude_pattern> <output_dir>
  e.g. build_loo_embedding.py 64-1000000-4-oltp_read_only-0.2 \
         autotune/optimizer/dml_models_loo_read
"""
import os, sys, subprocess
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DSET = os.path.join(ROOT, 'DBMSTransferLearning', 'dataset')
TRIPLET = os.path.join(DSET, 'full_triplet_data_concordance.csv')
CONTEXT = os.path.join(DSET, 'context_default_metrics_all.csv')
TRAINER = os.path.join(ROOT, 'scripts', 'train_dml_model.py')


def main():
    pat = sys.argv[1]
    outdir = sys.argv[2] if os.path.isabs(sys.argv[2]) else os.path.join(ROOT, sys.argv[2])
    if os.path.exists(os.path.join(outdir, 'context_model.pth')):
        print(f'[LOO] {outdir} already exists -> skip (cached)'); return
    os.makedirs(outdir, exist_ok=True)
    loo_dir = os.path.join(DSET, 'loo'); os.makedirs(loo_dir, exist_ok=True)
    tag = pat.replace('/', '_')

    cdf = pd.read_csv(CONTEXT)
    idcol = cdf.columns[0]
    cdf_loo = cdf[~cdf[idcol].astype(str).str.contains(pat, regex=False)]
    ctx_out = os.path.join(loo_dir, f'context_{tag}.csv'); cdf_loo.to_csv(ctx_out, index=False)

    tdf = pd.read_csv(TRIPLET)
    mask = ~(tdf['anchor_id'].astype(str).str.contains(pat, regex=False)
             | tdf['pos_id'].astype(str).str.contains(pat, regex=False)
             | tdf['neg_id'].astype(str).str.contains(pat, regex=False))
    tdf_loo = tdf[mask]
    trip_out = os.path.join(loo_dir, f'triplet_{tag}.csv'); tdf_loo.to_csv(trip_out, index=False)

    print(f'[LOO] exclude "{pat}": contexts {len(cdf)}->{len(cdf_loo)} '
          f'(dropped {len(cdf)-len(cdf_loo)}), triplets {len(tdf)}->{len(tdf_loo)}')
    subprocess.run([sys.executable, TRAINER, '--triplet_csv', trip_out,
                    '--context_csv', ctx_out, '--output_dir', outdir], check=True)
    print(f'[LOO] done -> {outdir}')


if __name__ == '__main__':
    main()
