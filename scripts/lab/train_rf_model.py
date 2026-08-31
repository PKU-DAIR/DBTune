#!/usr/bin/env python3
"""
Train a per-target RandomForest source-selection model and save artifacts that
the WorkloadMapping(mapping_method='rf') consumer in workload_map.py loads at
inference time.

For each source context in scripts/DBTune_history/csv_source/, the default-config
observation's 114-D internal_metrics vector is extracted (averaged if multiple
default-config trials exist). The same array order is what DBTune emits at
runtime, so no name-based alignment is needed.

Pair-wise top-1% Jaccard overlap between source contexts is used as the regression
target. Pair features = [|m_i - m_j|, (m_i + m_j) / 2] over the usable subset of
the 114 metrics (non-NaN, non-constant across sources).

Outputs (under --output_dir):
  - rf_model.joblib         RandomForestRegressor
  - source_default_metrics.csv   context_id, m_0 ... m_113
  - rf_meta.json            {'usable_idx': [...], 'n_sources': N, 'exclude': '...'}

Default leave-one-out rule per `--target_context`: omit any source context that
either (a) shares the target's workload signature (everything after the hardware
prefix — i.e. workload type + table size + skew + num_table + num_client) OR
(b) shares the target's hardware (HW_prefix). It's a set UNION — a source is
dropped if it matches EITHER axis. `--no_workload_excl` / `--no_hw_excl` disable
each axis; `--exclude` adds arbitrary substring patterns.

Usage:
  python scripts/lab/train_rf_model.py \\
      --target_context 112c125g_64-1000000-4-oltp_read_only-0.2 \\
      --output_dir autotune/optimizer/rf_models_loo_read/
"""
import argparse, glob, json, math, os, sys
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor

# Canonical default config (DML_12.json defaults expressed in raw MySQL units —
# the same units that csv_source JSON and the live runtime store).
PARAMS = [
    'innodb_buffer_pool_size', 'innodb_read_io_threads', 'innodb_write_io_threads',
    'innodb_flush_log_at_trx_commit', 'innodb_adaptive_hash_index', 'sync_binlog',
    'innodb_lru_scan_depth', 'innodb_buffer_pool_instances', 'innodb_change_buffer_max_size',
    'innodb_io_capacity', 'innodb_log_file_size', 'table_open_cache',
]


def get_defaults(knob_json_path):
    """Return canonical default value per knob from DML_12.json."""
    with open(knob_json_path) as f:
        knobs = json.load(f)
    out = {}
    for k in PARAMS:
        d = knobs[k]['default']
        # Coerce categoricals to int (ON/OFF -> 1/0 if needed).
        if isinstance(d, str):
            try:
                d = int(d)
            except ValueError:
                d = 1 if d.lower() in ('on', 'yes', 'true') else 0
        out[k] = d
    return out


def is_default_config(cfg, defaults):
    """Whether an observation's configuration dict matches the defaults."""
    for k, v in defaults.items():
        cv = cfg.get(k)
        # Some knobs may be stored as str/int — normalize.
        if isinstance(cv, str):
            try: cv = int(cv)
            except ValueError: cv = 1 if cv.lower() in ('on','yes','true') else 0
        if cv != v:
            return False
    return True


def load_source_default_vector(json_path, defaults):
    """Return (context_id, 114-array) for the default-config observation(s)."""
    with open(json_path) as f:
        d = json.load(f)
    ctx = os.path.basename(json_path)[len('history_'):-len('.json')]
    matches = []
    for obs in d['data']:
        cfg = obs.get('configuration', {})
        if is_default_config(cfg, defaults):
            im = obs.get('internal_metrics')
            if im and len(im) == 114:
                matches.append(np.asarray(im, dtype=float))
    if not matches:
        return ctx, None
    return ctx, np.mean(matches, axis=0)


def top1_pct_set(json_path):
    """Return set of canonical-knob-tuple keys for the top 1% TPS configs of a context."""
    with open(json_path) as f:
        d = json.load(f)
    cfg_tps = {}
    for obs in d['data']:
        cfg = obs.get('configuration', {})
        try:
            key = tuple(cfg[p] for p in PARAMS)
        except KeyError:
            continue
        tps = obs.get('external_metrics', {}).get('tps')
        if tps is None:
            continue
        if key not in cfg_tps or tps > cfg_tps[key]:
            cfg_tps[key] = tps
    if not cfg_tps:
        return set()
    k = max(1, math.ceil(len(cfg_tps) * 0.01))
    top = sorted(cfg_tps.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return set(t[0] for t in top)


def split_context_id(ctx):
    """ '<hw>_<num_table>-<table_size>-<num_client>-<workload>-<skew>' → (hw, wl_sig). """
    if '_' not in ctx:
        return ctx, ''
    hw, wl_sig = ctx.split('_', 1)
    return hw, wl_sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_context',
                    help='Target context_id (e.g., 112c125g_64-1000000-4-oltp_read_only-0.2). '
                         'By default, any source matching the target hardware OR workload '
                         'signature is excluded (set UNION).')
    ap.add_argument('--no_workload_excl', action='store_true',
                    help='Disable exclusion of sources sharing the target workload signature.')
    ap.add_argument('--no_hw_excl', action='store_true',
                    help='Disable exclusion of sources sharing the target hardware.')
    ap.add_argument('--exclude', default='',
                    help='Extra comma-separated substring patterns to exclude.')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--csv_source_dir', default='scripts/DBTune_history/csv_source')
    ap.add_argument('--knob_json', default='scripts/experiment/gen_knobs/DML_12.json')
    ap.add_argument('--n_estimators', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    excludes = [s.strip() for s in args.exclude.split(',') if s.strip()]
    # Per-target auto-exclusions (set UNION across workload + hardware axes).
    tgt_hw, tgt_wl = (None, None)
    if args.target_context:
        tgt_hw, tgt_wl = split_context_id(args.target_context)
        if not args.no_workload_excl and tgt_wl:
            excludes.append(tgt_wl)         # any ctx containing the workload sig
        if not args.no_hw_excl and tgt_hw:
            excludes.append(tgt_hw + '_')   # any ctx whose hw prefix matches
        print(f'[train_rf] target context: {args.target_context}')
        print(f'           hw prefix: {tgt_hw!r}   workload sig: {tgt_wl!r}')
    defaults = get_defaults(args.knob_json)
    print(f'[train_rf] default knob values:\n  {defaults}')

    files = sorted(glob.glob(os.path.join(args.csv_source_dir, 'history_*.json')))
    print(f'[train_rf] {len(files)} source JSONs; excludes={excludes}')

    ctxs, vecs, top1 = [], [], {}
    skipped_no_default = 0
    for f in files:
        ctx, v = load_source_default_vector(f, defaults)
        if v is None:
            skipped_no_default += 1
            continue
        if any(p in ctx for p in excludes):
            continue
        ctxs.append(ctx); vecs.append(v)
        top1[ctx] = top1_pct_set(f)
    vecs = np.vstack(vecs)
    print(f'[train_rf] kept {len(ctxs)} sources after exclusion '
          f'(skipped {skipped_no_default} with no default-config obs)')

    # Usable metric indices: every source must have finite value AND non-constant variance.
    finite = np.isfinite(vecs).all(axis=0)
    nonconst = vecs.std(axis=0) > 1e-12
    usable_idx = np.where(finite & nonconst)[0]
    print(f'[train_rf] usable metric dims: {len(usable_idx)} / 114')

    src = vecs[:, usable_idx]
    N = len(ctxs)
    pair_X, pair_y = [], []
    for i in range(N):
        for j in range(N):
            if i == j: continue
            a, b = top1.get(ctxs[i], set()), top1.get(ctxs[j], set())
            ov = (len(a & b) / max(1, len(a | b))) if (a or b) else 0.0
            ai, bj = src[i], src[j]
            pair_X.append(np.concatenate([np.abs(ai - bj), (ai + bj) / 2.0]))
            pair_y.append(ov)
    X = np.asarray(pair_X, dtype=np.float32)
    y = np.asarray(pair_y, dtype=np.float32)
    print(f'[train_rf] training pairs: {len(y)}; feature dim: {X.shape[1]}; mean overlap: {y.mean():.3f}')

    rf = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.seed,
                               n_jobs=-1, max_depth=None)
    rf.fit(X, y)

    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(args.output_dir, 'rf_model.joblib'))

    cols = [f'm{i}' for i in range(114)]
    df = pd.DataFrame(vecs, columns=cols); df.insert(0, 'context_id', ctxs)
    df.to_csv(os.path.join(args.output_dir, 'source_default_metrics.csv'), index=False)

    meta = {
        'usable_idx': usable_idx.tolist(),
        'n_sources': N,
        'n_metrics_full': 114,
        'target_context': args.target_context,
        'exclude_workload': not args.no_workload_excl,
        'exclude_hardware': not args.no_hw_excl,
        'exclude_patterns': excludes,
        'extra_exclude_arg': args.exclude,
        'n_estimators': args.n_estimators,
        'seed': args.seed,
    }
    with open(os.path.join(args.output_dir, 'rf_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[train_rf] saved model + source vectors + meta to {args.output_dir}')

if __name__ == '__main__':
    main()
