#!/usr/bin/env python3
"""Assemble the multi-source repository `DBTune_history/pool_ALL` used by the
1 h-deadline transfer-method comparison (exp_notes/similar_workload.md, F7/F7a/F7b;
repro: claude_memory/REPRO_1H_DEADLINE.md).

pool_ALL = the 15 SUCCESS-only per-cell pools produced by build_eval_pools.py
(S0-S4 x {lhs, random, llama}; sweep excluded because its data is worthless as
warm-start payload, F2) copied into ONE data_repo, plus a pre-built loader cache
(`_history_cache.pkl`, pickled list of HistoryContainer in sorted-filename order).

The cache MUST be pre-built here, single-threaded: tuner.load_history builds it
lazily on first use, and the parallel pbsdsh arms racing to build it on Lustre
clobber each other (claude_memory/MIYABI.md section 5).

Run from scripts/ after build_eval_pools.py:
    ../venv/bin/python build_pool_all.py            # -> DBTune_history/pool_ALL
    ../venv/bin/python build_pool_all.py --out DBTune_history/pool_ALL_v2
Refuses to overwrite a non-empty --out unless --force is given.

Generic mode (any knob space, raw collection histories -> SUCCESS-only pool + cache):
    ../venv/bin/python build_pool_all.py --knob-file ./experiment/gen_knobs/mysql_perf_8.0_online.json \
        --from-history DBTune_history/history_miyabic_150-800000-128-oltp_read_write-0.7-llama_online.json \
        --out DBTune_history/pool_S0_llama_online
"""
import argparse
import json
import os
import pickle
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from test_llamatune import build_space  # same space construction as DBTuner  # noqa: E402
from autotune.utils.history_container import HistoryContainer  # noqa: E402

KNOB_FILE = './experiment/gen_knobs/mysql_perf_8.0.json'
HIST_DIR = 'DBTune_history'
CELLS = ['S0', 'S1', 'S2', 'S3', 'S4']
STRATEGIES = ['lhs', 'random', 'llama']   # sweep deliberately excluded (F2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HIST_DIR, 'pool_ALL'))
    ap.add_argument('--cells', nargs='+', default=CELLS)
    ap.add_argument('--strategies', nargs='+', default=STRATEGIES)
    ap.add_argument('--force', action='store_true', help='overwrite a non-empty --out')
    ap.add_argument('--knob-file', default=KNOB_FILE, help='knob JSON defining the config space of the pool')
    ap.add_argument('--from-history', nargs='+', metavar='JSON',
                    help='raw history files to pool (SUCCESS rows only) instead of the per-cell pools')
    a = ap.parse_args()

    if os.path.isdir(a.out) and os.listdir(a.out) and not a.force:
        sys.exit('%s exists and is non-empty; pass --force to rebuild it' % a.out)
    os.makedirs(a.out, exist_ok=True)

    # 1) collect the per-cell SUCCESS-only pool files (or filter the given raw histories)
    srcs = []
    if a.from_history:
        for src in a.from_history:
            raw = json.load(open(src))
            # tps<=0 = benchmark timeout recorded as SUCCESS (dbenv leaves trial_state alone); no valid metrics
            ok = [r for r in raw['data'] if r['trial_state'] == 0 and r['external_metrics']
                  and r['external_metrics'].get('tps', 0) > 0]
            dst = os.path.join(a.out, os.path.basename(src))
            with open(dst, 'w') as f:
                json.dump({'info': raw['info'], 'data': ok}, f, indent=2)
            print('%s: %d/%d SUCCESS rows kept' % (os.path.basename(src), len(ok), len(raw['data'])))
    for cell in ([] if a.from_history else a.cells):
        for strategy in a.strategies:
            d = os.path.join(HIST_DIR, 'pool_%s_%s' % (cell, strategy))
            files = sorted(f for f in os.listdir(d)) if os.path.isdir(d) else []
            files = [f for f in files if f.startswith('history_') and f.endswith('.json')]
            if len(files) != 1:
                sys.exit('%s: expected exactly one history_*.json (run build_eval_pools.py first), found %s'
                         % (d, files))
            srcs.append(os.path.join(d, files[0]))

    # 2) copy them in (same file names -> same task_id / context labels)
    for s in srcs:
        shutil.copyfile(s, os.path.join(a.out, os.path.basename(s)))

    # 3) pre-build the loader cache in sorted-filename order (must be newer than the JSONs)
    space, _ = build_space(a.knob_file)
    default = space.get_default_configuration()
    hcL, total = [], 0
    for f in sorted(os.listdir(a.out)):
        if not f.endswith('.json'):
            continue
        # context label = original collection task_id (same convention as build_eval_pools.py;
        # tuner.load_history would keep the 'history_' prefix if it built the cache itself)
        task_id = f[len('history_'):-len('.json')]
        fn = os.path.join(a.out, f)
        rows = json.load(open(fn))['data']
        assert all(r['trial_state'] == 0 and r['external_metrics'] for r in rows), \
            '%s: pool contains non-SUCCESS rows' % f
        hc = HistoryContainer(task_id, config_space=space)
        hc.load_history_from_json(fn)
        assert len(hc.configurations) == len(rows), \
            '%s: %d rows but %d loaded (config reconstruction failed?)' % (f, len(rows), len(hc.configurations))
        assert hc.configurations[0] == default, '%s: iteration 0 is not the default config' % f
        hcL.append(hc)
        total += len(rows)
        print('%4d rows  %s' % (len(rows), task_id))
    with open(os.path.join(a.out, '_history_cache.pkl'), 'wb') as fp:
        pickle.dump(hcL, fp)
    print('%s: %d contexts, %d rows, cache written.' % (a.out, len(hcL), total))


if __name__ == '__main__':
    main()
