#!/usr/bin/env python3
"""Generate the per-run ini + isolated cnf for one transfer-method comparison run.

usage: transfer_gen_task.py <mode:ottertune|rgpe|opadviser|opadviser_ns|cold> <seed> <root> [variant]

variant (default 'offline') = the original eval2 setup: 117-knob mysql_perf_8.0.json,
my.cnf + restart per iteration with the historical kill -9 (graceful_shutdown = False pinned),
source repo pool_ALL, task_id eval2_<mode>_s<seed>.
variant 'fast' = identical, but with the clean shutdown that is now the default
(innodb_fast_shutdown=1 + mysqladmin shutdown, REPRO §10), task_id eval4_<mode>_s<seed>.
variant 'online' = the dynamic-knob replication: 95-knob mysql_perf_8.0_online.json,
online_mode=True (SET GLOBAL, no restarts), source repo pool_S0_llama_online (the
online-collected S0 LlamaTune cell), task_id eval3_<mode>_s<seed>.

All arms tune the TARGET workload (150x800k, 128 threads, zipf 0.7) under a 1 h
hard deadline (enforced externally via `timeout`), max_runs=200 so time binds.
Source repository for the transfer arms: DBTune_history/pool_ALL (15 contexts,
S0-S4 x lhs/random/llama, SUCCESS-only) — each method selects on its own.

  ottertune : SMAC + workload_map(ottertune metric-binning source selection)
  rgpe      : SMAC + RGPE ensemble (ResTune-style meta-learner over all sources)
  opadviser : SMAC + space_transfer=True (OpAdviser-style: compact space from
              sources + initial design seeded with source-best configs)
  opadviser_ns : opadviser WITHOUT the replay init (space_transfer_replay=False):
              only the paper-documented compact-space mechanism
  cold      : SMAC, no transfer (baseline)
"""
import configparser, os, sys

mode, seed, root = sys.argv[1], sys.argv[2], sys.argv[3]
variant = sys.argv[4] if len(sys.argv) > 4 else 'offline'
assert variant in ('offline', 'online', 'fast'), variant
assert mode in ('ottertune', 'rgpe', 'opadviser', 'opadviser_ns', 'cold'), mode
task_id = '%s_%s_s%s' % ({'offline': 'eval2', 'online': 'eval3', 'fast': 'eval4'}[variant], mode, seed)

paral = os.path.join(root, 'parallel', task_id)
os.makedirs(paral, exist_ok=True)
datadir = os.path.join(paral, 'data')
cnf = os.path.join(paral, 'my.cnf')
sock = '/tmp/dbtune.sock'

cfg = configparser.ConfigParser(); cfg.optionxform = str
cfg.read(os.path.join(root, 'scripts', 'config_collect_S0_sweep.ini'))  # target workload base
db, tune = cfg['database'], cfg['tune']
db['cnf'] = cnf
db['sock'] = sock
db['datadir'] = datadir
db['mysqld'] = os.path.join(root, 'mysql_build/bin/mysqld')

for k in ('sampler_method', 'sweep_levels', 'min_success'):
    tune.pop(k, None)
tune['task_id'] = task_id
tune['rand_seed'] = seed
tune['optimize_method'] = 'SMAC'
tune['max_runs'] = '200'          # never reached: the 1 h timeout is the stop
tune['initial_runs'] = '1'
tune['space_transfer'] = 'None'
tune['data_repo'] = './DBTune_history/pool_ALL'
tune['transfer_framework'] = 'none'
if variant == 'offline':
    db['graceful_shutdown'] = 'False'   # the July/run-2 sessions used kill -9
if variant == 'online':
    # 93 dynamic knobs; innodb_buffer_pool_size / innodb_redo_log_capacity are pinned at their
    # defaults in the cnf (their online resize is a multi-minute blocking operation, REPRO §9a)
    db['knob_config_file'] = './experiment/gen_knobs/mysql_perf_8.0_online_noresize.json'
    db['online_mode'] = 'True'
    tune['data_repo'] = './DBTune_history/pool_S0_llama_online'

if mode == 'ottertune':
    tune['transfer_framework'] = 'workload_map'
    tune['mapping_method'] = 'ottertune'
    tune['mapping_prune_metrics'] = 'false'
elif mode == 'rgpe':
    tune['transfer_framework'] = 'rgpe'
elif mode == 'opadviser':
    tune['space_transfer'] = 'True'
    tune['initial_runs'] = '5'    # first 5 evals = source-best configs (space_transfer init)
elif mode == 'opadviser_ns':
    # ablation: paper's mechanism only — compact space from sources, NO replay init;
    # initial_runs=1 mirrors the cold arm exactly, so the only delta is the space
    tune['space_transfer'] = 'True'
    tune['space_transfer_replay'] = 'False'
    tune['initial_runs'] = '1'
else:  # cold
    tune['data_repo'] = './DBTune_history/empty_repo'

with open(os.path.join(paral, 'config.ini'), 'w') as f:
    cfg.write(f)

out = []
for line in open(os.path.join(root, 'mysql_build/cnf/my.cnf.clean')):
    k = line.split('=', 1)[0].strip()
    if k == 'datadir':     out.append('datadir = ' + datadir)
    elif k == 'socket':    out.append('socket = ' + sock)
    elif k == 'pid-file':  out.append('pid-file = /tmp/dbtune.pid')
    elif k == 'log-error': out.append('log-error = /tmp/dbtune.err')
    else:                  out.append(line.rstrip('\n'))
if variant == 'online':
    import json
    _K = json.load(open(os.path.join(root, 'scripts/experiment/gen_knobs/mysql_perf_8.0_online.json')))
    for k in ('innodb_buffer_pool_size', 'innodb_redo_log_capacity'):
        out.append('%s = %s' % (k, _K[k]['default']))
with open(cnf + '.clean', 'w') as f:
    f.write('\n'.join(out) + '\n')

print(task_id)
