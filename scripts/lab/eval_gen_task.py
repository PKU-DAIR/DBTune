#!/usr/bin/env python3
"""Generate the per-run ini + isolated cnf for one warm-start evaluation run.

usage: eval_gen_task.py <mode:warm|cold|replay> <cell:S0|S1|-> <strategy|-> <seed> <root>

All runs tune/measure the TARGET workload (= S0's: 150x800k, 128 threads, zipf 0.7).
  warm   : SMAC + workload_map(ottertune), data_repo = the cell's SUCCESS-only pool,
           max_runs=10  -> task_id eval_warm_<cell>_<strategy>_s<seed>
  cold   : SMAC, transfer_framework=none, max_runs=10 -> task_id eval_cold_s<seed>
  replay : Sampler[replay] over eval/replay_<cell>_<strategy>.json (default + top-3
           transplanted source configs), max_runs=4 -> task_id eval_replay_<cell>_<strategy>

Writes <root>/parallel/<task_id>/{config.ini,my.cnf.clean}; prints the task_id.
"""
import configparser, os, sys

mode, cell, strategy, seed, root = sys.argv[1:6]

if mode == 'warm':
    task_id = 'eval_warm_%s_%s_s%s' % (cell, strategy, seed)
elif mode == 'cold':
    task_id = 'eval_cold_s%s' % seed
elif mode == 'replay':
    task_id = 'eval_replay_%s_%s' % (cell, strategy)
else:
    raise SystemExit('bad mode %r' % mode)

paral = os.path.join(root, 'parallel', task_id)
os.makedirs(paral, exist_ok=True)
datadir = os.path.join(paral, 'data')
cnf = os.path.join(paral, 'my.cnf')
sock = '/tmp/dbtune.sock'

# base = the S0 collection ini: same target workload + knob file + prometheus keys
cfg = configparser.ConfigParser(); cfg.optionxform = str
cfg.read(os.path.join(root, 'scripts', 'config_collect_S0_sweep.ini'))
db, tune = cfg['database'], cfg['tune']
db['cnf'] = cnf
db['sock'] = sock
db['datadir'] = datadir
db['mysqld'] = os.path.join(root, 'mysql_build/bin/mysqld')

tune['task_id'] = task_id
tune['rand_seed'] = seed
for k in ('sampler_method', 'sweep_levels'):
    tune.pop(k, None)

if mode == 'warm':
    tune['optimize_method'] = 'SMAC'
    tune['transfer_framework'] = 'workload_map'
    tune['mapping_method'] = 'ottertune'
    tune['mapping_prune_metrics'] = 'false'
    tune['data_repo'] = './DBTune_history/pool_%s_%s' % (cell, strategy)
    tune['max_runs'] = '10'
    tune['initial_runs'] = '1'
elif mode == 'cold':
    tune['optimize_method'] = 'SMAC'
    tune['transfer_framework'] = 'none'
    tune['data_repo'] = './DBTune_history/empty_repo'
    tune['max_runs'] = '10'
    tune['initial_runs'] = '1'
else:  # replay
    tune['optimize_method'] = 'Sampler'
    tune['sampler_method'] = 'replay'
    tune['replay_file'] = './eval/replay_%s_%s.json' % (cell, strategy)
    tune['transfer_framework'] = 'none'
    tune['data_repo'] = './DBTune_history/empty_repo'
    tune['max_runs'] = '4'
    tune['initial_runs'] = '1'

with open(os.path.join(paral, 'config.ini'), 'w') as f:
    cfg.write(f)

# isolated clean cnf (same rewrite as collect_gen_task.py)
out = []
for line in open(os.path.join(root, 'mysql_build/cnf/my.cnf.clean')):
    k = line.split('=', 1)[0].strip()
    if k == 'datadir':     out.append('datadir = ' + datadir)
    elif k == 'socket':    out.append('socket = ' + sock)
    elif k == 'pid-file':  out.append('pid-file = /tmp/dbtune.pid')
    elif k == 'log-error': out.append('log-error = /tmp/dbtune.err')
    else:                  out.append(line.rstrip('\n'))
with open(cnf + '.clean', 'w') as f:
    f.write('\n'.join(out) + '\n')

print(task_id)
