#!/usr/bin/env python3
"""Train a RandomForest source-selection model on the OpAdviser repo histories.

Adapted from train_rf_model.py for the OpAdviser source pool, which differs from
csv_source in three ways:
  - internal_metrics are 65-D (information_schema INNODB_METRICS), not 114-D Prometheus.
  - knob space is the 197-knob master (mysql_all_197_32G.json); each source file
    tunes a different SUBSET (5-96 knobs), the rest implicitly at their defaults.
  - each file is one optimizer's 200-iteration trajectory (not a sweep), and only
    ~half contain an exact default-config observation.

Per-context feature vector (the 65-D metrics the live RF compares the target's
iter-0 vector against): the metrics at the observation CLOSEST to the global
default (fewest tuned knobs differing from their 197-defaults). When an exact
default obs exists this reduces to it (averaged over duplicates).

Label: pairwise top-K% TPS Jaccard overlap, configs canonicalized to the full
197-knob tuple (missing knobs imputed to default) so configs are comparable
across files. K defaults to 10% (top ~20 of 200) since 1% (~2 configs) is too
sparse for this heterogeneous, non-overlapping source set.

context_id == source task_id == filename stem (e.g. 'history_task1_mbo'), so it
matches WorkloadMapping.source_dict keys (tuner.load_history uses splitext(f)[0]).

Outputs to --output_dir:
  rf_model.joblib, source_default_metrics.csv (context_id, m0..m64), rf_meta.json
"""
import argparse, glob, json, math, os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor


def norm(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def get_defaults(knob_json):
    with open(knob_json) as f:
        knobs = json.load(f)
    return {k: norm(v['default']) for k, v in knobs.items()}


def n_nondefault(cfg, defaults):
    return sum(1 for k, v in cfg.items() if norm(v) != defaults.get(k))


def near_default_vector(data, defaults, ndim):
    """65-D metric vector at the obs closest to the global default."""
    best_n, best_ims = None, []
    for obs in data:
        cfg = obs.get('configuration', {})
        im = obs.get('internal_metrics')
        if not im or len(im) != ndim:
            continue
        nd = n_nondefault(cfg, defaults)
        if best_n is None or nd < best_n:
            best_n, best_ims = nd, [np.asarray(im, dtype=float)]
        elif nd == best_n:
            best_ims.append(np.asarray(im, dtype=float))
    if not best_ims:
        return None, None
    return np.mean(best_ims, axis=0), best_n


def canon_key(cfg, knob_names, defaults):
    """Full-197 canonical tuple (missing knobs imputed to default)."""
    return tuple(norm(cfg[k]) if k in cfg else defaults[k] for k in knob_names)


def topk_pct_set(data, knob_names, defaults, frac):
    cfg_tps = {}
    for obs in data:
        cfg = obs.get('configuration', {})
        tps = obs.get('external_metrics', {}).get('tps')
        if tps is None:
            continue
        key = canon_key(cfg, knob_names, defaults)
        if key not in cfg_tps or tps > cfg_tps[key]:
            cfg_tps[key] = tps
    if not cfg_tps:
        return set()
    k = max(1, math.ceil(len(cfg_tps) * frac))
    top = sorted(cfg_tps.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return set(t[0] for t in top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_dir', default='OpAdviser/repo')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--knob_json', default='scripts/experiment/gen_knobs/mysql_all_197_32G.json')
    ap.add_argument('--ndim', type=int, default=65)
    ap.add_argument('--frac', type=float, default=0.10, help='top-K%% fraction for the Jaccard label')
    ap.add_argument('--n_estimators', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    defaults = get_defaults(args.knob_json)
    knob_names = sorted(defaults.keys())
    files = sorted(glob.glob(os.path.join(args.source_dir, 'history_*.json')))
    print(f'[train_rf_op] {len(files)} source files; ndim={args.ndim}; top-{args.frac:.0%} Jaccard')

    ctxs, vecs, topk, near = [], [], {}, []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)['data']
        ctx = os.path.splitext(os.path.basename(f))[0]   # 'history_task1_mbo'
        v, nd = near_default_vector(data, defaults, args.ndim)
        if v is None:
            print(f'  skip {ctx}: no {args.ndim}-D metric obs'); continue
        ctxs.append(ctx); vecs.append(v); near.append(nd)
        topk[ctx] = topk_pct_set(data, knob_names, defaults, args.frac)
    vecs = np.vstack(vecs)
    print(f'[train_rf_op] kept {len(ctxs)} contexts; '
          f'near-default #nondefault knobs: min={min(near)} median={int(np.median(near))} max={max(near)}')

    finite = np.isfinite(vecs).all(axis=0)
    nonconst = vecs.std(axis=0) > 1e-12
    usable_idx = np.where(finite & nonconst)[0]
    print(f'[train_rf_op] usable metric dims: {len(usable_idx)} / {args.ndim}')

    src = vecs[:, usable_idx]
    N = len(ctxs)
    pair_X, pair_y = [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            a, b = topk.get(ctxs[i], set()), topk.get(ctxs[j], set())
            ov = (len(a & b) / max(1, len(a | b))) if (a or b) else 0.0
            ai, bj = src[i], src[j]
            pair_X.append(np.concatenate([np.abs(ai - bj), (ai + bj) / 2.0]))
            pair_y.append(ov)
    X = np.asarray(pair_X, dtype=np.float32)
    y = np.asarray(pair_y, dtype=np.float32)
    nz = (y > 0).mean()
    print(f'[train_rf_op] pairs={len(y)} featdim={X.shape[1]} mean_overlap={y.mean():.4f} nonzero_frac={nz:.3f}')

    rf = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.seed,
                               n_jobs=-1, max_depth=None)
    rf.fit(X, y)

    os.makedirs(args.output_dir, exist_ok=True)
    import joblib
    joblib.dump(rf, os.path.join(args.output_dir, 'rf_model.joblib'))
    cols = [f'm{i}' for i in range(args.ndim)]
    df = pd.DataFrame(vecs, columns=cols); df.insert(0, 'context_id', ctxs)
    df.to_csv(os.path.join(args.output_dir, 'source_default_metrics.csv'), index=False)
    meta = {'usable_idx': usable_idx.tolist(), 'n_sources': N, 'n_metrics_full': args.ndim,
            'source_dir': args.source_dir, 'frac': args.frac, 'n_estimators': args.n_estimators,
            'seed': args.seed, 'feature': 'near-default (min nondefault-knob obs)'}
    with open(os.path.join(args.output_dir, 'rf_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[train_rf_op] saved to {args.output_dir}')


if __name__ == '__main__':
    main()
