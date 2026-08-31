#!/usr/bin/env python3
"""Build the warm-start evaluation inputs from the collected S0/S1 histories.

For each of the 8 collection cells this creates:
  1. DBTune_history/pool_<cell>_<strategy>/history_<orig task_id>.json
       - the cell's history filtered to SUCCESS rows only. FAILED rows carry
         penalty objectives and empty internal metrics; left in, they would
         poison the source surrogate and the OtterTune metric binning.
       - original filename kept so the context label (task_id) is unchanged.
  2. DBTune_history/pool_<cell>_<strategy>/_history_cache.pkl
       - pre-built loader cache (tuner.load_history format: pickled list of
         HistoryContainer). Avoids the concurrent-first-load race on Lustre.
  3. eval/replay_<cell>_<strategy>.json
       - the cell's top-3 SUCCESS configs by tps, for the transplant eval
         (Sampler strategy 'replay').

Run from scripts/:  ../venv/bin/python build_eval_pools.py
Idempotent; rebuilds everything each run.
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_llamatune import build_space  # same space construction as DBTuner
from autotune.utils.history_container import HistoryContainer

KNOB_FILE = './experiment/gen_knobs/mysql_perf_8.0.json'
HIST_DIR = 'DBTune_history'
# cell -> (source task_id label prefix, strategies collected for it)
CELLS = {
    'S0': ('miyabic_150-800000-128-oltp_read_write-0.7', ['sweep', 'lhs', 'random', 'llama']),
    'S1': ('miyabic_150-800000-32-oltp_read_write-0.7', ['sweep', 'lhs', 'random', 'llama']),
    'S2': ('miyabic_150-800000-128-oltp_read_write-uniform', ['random', 'lhs', 'llama']),
    'S3': ('miyabic_150-80000-128-oltp_read_write-0.7', ['random', 'lhs', 'llama']),
    'S4': ('miyabic_150-800000-128-oltp_read_write_ps30-0.7', ['random', 'lhs', 'llama']),
}
TOP_K = 3


def main():
    os.makedirs('eval', exist_ok=True)
    space, _ = build_space(KNOB_FILE)
    for cell, (label, strategies) in CELLS.items():
        for strategy in strategies:
            task_id = '%s-%s' % (label, strategy)
            src = os.path.join(HIST_DIR, 'history_%s.json' % task_id)
            if not os.path.exists(src):
                print('%s_%-6s : SKIP (no history yet)' % (cell, strategy)); continue
            raw = json.load(open(src))
            n_ok = sum(1 for r in raw['data'] if r['trial_state'] == 0 and r['external_metrics'])
            # S2-S4 stop at 100 successes / 300 attempts; don't build from in-flight cells
            if cell not in ('S0', 'S1') and n_ok < 100 and len(raw['data']) < 300:
                print('%s_%-6s : SKIP (in progress: %d ok / %d att)'
                      % (cell, strategy, n_ok, len(raw['data']))); continue
            ok = [r for r in raw['data'] if r['trial_state'] == 0 and r['external_metrics']]

            pool_dir = os.path.join(HIST_DIR, 'pool_%s_%s' % (cell, strategy))
            os.makedirs(pool_dir, exist_ok=True)
            pool_json = os.path.join(pool_dir, 'history_%s.json' % task_id)
            with open(pool_json, 'w') as f:
                json.dump({'info': raw['info'], 'data': ok}, f, indent=2)

            # pre-build the loader cache (must be newer than the json)
            hc = HistoryContainer(task_id, config_space=space)
            hc.load_history_from_json(pool_json)
            assert len(hc.configurations) == len(ok), \
                '%s: %d rows in json but %d loaded (config reconstruction failed?)' \
                % (task_id, len(ok), len(hc.configurations))
            default = space.get_default_configuration()
            assert hc.configurations[0] == default, \
                '%s: iteration 0 is not the default config' % task_id
            with open(os.path.join(pool_dir, '_history_cache.pkl'), 'wb') as f:
                pickle.dump([hc], f)

            # top-K configs by tps for the transplant eval
            ranked = sorted(ok, key=lambda r: r['external_metrics']['tps'], reverse=True)
            top = [{'tps_at_source': r['external_metrics']['tps'],
                    'configuration': r['configuration']} for r in ranked[:TOP_K]]
            replay_fn = os.path.join('eval', 'replay_%s_%s.json' % (cell, strategy))
            with open(replay_fn, 'w') as f:
                json.dump({'source_task_id': task_id,
                           'configs': [t['configuration'] for t in top],
                           'source_tps': [t['tps_at_source'] for t in top]}, f, indent=2)

            print('%s_%-6s : pool %3d/%3d rows  top%d tps=[%s]'
                  % (cell, strategy, len(ok), len(raw['data']), TOP_K,
                     ','.join('%.0f' % t['tps_at_source'] for t in top)))
    print('done.')


if __name__ == '__main__':
    main()
