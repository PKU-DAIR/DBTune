#!/usr/bin/env python3
"""Generate the ini + isolated cnf + replay list for ONE anchor-probe run.

Minimal test of Algorithm 1's probing step (Prepare, lines 11-12):
    qref <- Perf(c_w, w)     (probe workload's own best config, re-measured)
    p    <- Perf(c_s, w)     (the ANCHOR's best config run on the probe workload)
    pass iff p / qref > theta
with anchor s = S0 (150x800k rows, 128 clients, zipf 0.7, default mix).

usage: prep_gen_task.py <probe> <root>
probes (each = S0 with exactly ONE parameter changed):
    S1    32 clients            (edge probe, concurrency)
    S2    uniform access        (edge probe, skew)
    S3    80k rows/table        (edge probe, data scale)     snapshot data_150x80k
    S4    --point-selects=30    (edge probe, mix)
    C64   64 clients            (interior check, concurrency)
    R400k 400k rows/table       (interior check, data scale) snapshot data_150x400k

Replay list per probe (default config is always played first by the Sampler):
    anchors  : S0-random#1 (S0's nominal Tune best, 19,940 src) and S0-lhs#2
               (best validated on S0: 19,378 src / 19,283 re-measured) x3 each,
               spread over the session so datadir drift hits them evenly
    own top-2: the probe cell's own best-2 configs (fresh qref)   [S1..S4 only]
    library  : 6 strong configs from all cells (qref proxy where no own pool
               exists: C64, R400k), 1x each, deduplicated by config hash
Writes <root>/scripts/eval/prep_anchor_<probe>.json and
       <root>/parallel/prep_<probe>/{config.ini,my.cnf.clean}; prints task_id.
"""
import configparser, hashlib, json, os, sys

probe, root = sys.argv[1], sys.argv[2]
SCRIPTS = os.path.join(root, 'scripts')

PROBES = {
    'S1':    dict(base='S1', override={},                              snapshot='data_150x800k'),
    'S2':    dict(base='S2', override={},                              snapshot='data_150x800k'),
    'S3':    dict(base='S3', override={},                              snapshot='data_150x80k'),
    'S4':    dict(base='S4', override={},                              snapshot='data_150x800k'),
    'C64':   dict(base='S0', override={'thread_num': '64'},            snapshot='data_150x800k'),
    'R400k': dict(base='S0', override={'sysbench_table_size': '400000'}, snapshot='data_150x400k'),
}
if probe not in PROBES:
    raise SystemExit('unknown probe %r (choose from %s)' % (probe, ', '.join(PROBES)))
spec = PROBES[probe]

# (label, cell, strategy, index-in-replay-file) — indices are 0-based rank by source TPS
ANCHORS = [('S0-random#1', 'S0', 'random', 0),
           ('S0-lhs#2',    'S0', 'lhs',    1)]
OWN_TOP2 = {
    'S1': [('S1-random#1', 'S1', 'random', 0), ('S1-lhs#1',   'S1', 'lhs',   0)],
    'S2': [('S2-llama#1',  'S2', 'llama',  0), ('S2-random#1', 'S2', 'random', 0)],
    'S3': [('S3-llama#1',  'S3', 'llama',  0), ('S3-llama#2',  'S3', 'llama',  1)],
    'S4': [('S4-llama#1',  'S4', 'llama',  0), ('S4-lhs#1',    'S4', 'lhs',    0)],
}
LIBRARY = [('S4-lhs#1',   'S4', 'lhs',    0),   # 19,772 on S0 (5 measurements 18.9-19.8k)
           ('S1-random#1', 'S1', 'random', 0),   # 20,180 on S0
           ('S2-llama#1',  'S2', 'llama',  0),   # 19,081 on S0
           ('S0-lhs#1',    'S0', 'lhs',    0),   # 19,500 src on S0
           ('S3-llama#3',  'S3', 'llama',  2),   # 17,598 on S0, tuned on 80k
           ('S4-llama#2',  'S4', 'llama',  1)]   # 19,294 on S0


def chash(c):
    return hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:8]


_cache = {}
def load(cell, strategy, idx):
    key = (cell, strategy)
    if key not in _cache:
        _cache[key] = json.load(open(os.path.join(SCRIPTS, 'eval', 'replay_%s_%s.json' % (cell, strategy))))
    d = _cache[key]
    return d['configs'][idx], float(d['source_tps'][idx])


anchors = [(lab, ) + load(c, s, i) for lab, c, s, i in ANCHORS]
own = [(lab, ) + load(c, s, i) for lab, c, s, i in OWN_TOP2.get(probe, [])]
lib = [(lab, ) + load(c, s, i) for lab, c, s, i in LIBRARY]

seen = {chash(c) for _, c, _ in anchors} | {chash(c) for _, c, _ in own}
lib_unique = []
for lab, c, t in lib:
    if chash(c) in seen:
        continue
    seen.add(chash(c)); lib_unique.append((lab, c, t))

# session order: anchors early / middle / late; own + library in between
l1, l2 = lib_unique[:3], lib_unique[3:]
order = []
def add(items, role, rep=None):
    for lab, c, t in items:
        order.append(dict(label=lab + ('/r%d' % rep if rep else ''), role=role,
                          hash=chash(c), source_tps=t, config=c))
add(anchors, 'anchor', 1); add(own, 'own'); add(l1, 'library')
add(anchors, 'anchor', 2); add(l2, 'library'); add(anchors, 'anchor', 3)

task_id = 'prep_anchor_%s' % probe

# Retry rows: configs that FAILED (mysql did not start within the connect wait,
# usually a sticky-datadir streak, not the config) are re-queued once at the END
# of the list. The already-evaluated prefix is untouched, so the Sampler's resume
# check still passes; a re-run starts from a fresh datadir copy.
hist_fn = os.path.join(SCRIPTS, 'DBTune_history', 'history_%s.json' % task_id)
if os.path.exists(hist_fn):
    rows = json.load(open(hist_fn))['data']
    have_retry = {o['hash'] for o in order if o['label'].endswith('/retry')}
    for i, r in enumerate(rows[1:]):           # row 0 = default config
        if i >= len(order) or r['trial_state'] == 0:
            continue
        o = order[i]
        if o['hash'] in have_retry:
            continue
        have_retry.add(o['hash'])
        order.append(dict(o, label=o['label'].split('/')[0] + '/retry'))

payload = dict(source_task_id=task_id, probe=probe, snapshot=spec['snapshot'],
               anchor='S0 = 150x800k rows, 128 clients, zipf 0.7',
               configs=[o['config'] for o in order],
               source_tps=[o['source_tps'] for o in order],
               labels=[o['label'] for o in order],
               roles=[o['role'] for o in order],
               hashes=[o['hash'] for o in order])
os.makedirs(os.path.join(SCRIPTS, 'eval'), exist_ok=True)
replay_rel = './eval/%s.json' % task_id
with open(os.path.join(SCRIPTS, 'eval', task_id + '.json'), 'w') as f:
    json.dump(payload, f, indent=1)

# ---- per-run ini: the probe cell's own collection ini defines the workload exactly
paral = os.path.join(root, 'parallel', 'prep_%s' % probe)
os.makedirs(paral, exist_ok=True)
datadir = os.path.join(paral, 'data')
cnf = os.path.join(paral, 'my.cnf')
sock = '/tmp/dbtune.sock'

cfg = configparser.ConfigParser(); cfg.optionxform = str
cfg.read(os.path.join(SCRIPTS, 'config_collect_%s_llama.ini' % spec['base']))
db, tune = cfg['database'], cfg['tune']
for k, v in spec['override'].items():
    db[k] = v
db['cnf'] = cnf; db['sock'] = sock; db['datadir'] = datadir
db['mysqld'] = os.path.join(root, 'mysql_build/bin/mysqld')
db['host'] = '127.0.0.1'; db['port'] = '3306'

tune['task_id'] = task_id
tune['optimize_method'] = 'Sampler'
tune['sampler_method'] = 'replay'
tune['replay_file'] = replay_rel
tune['transfer_framework'] = 'none'
tune['data_repo'] = './DBTune_history/empty_repo'
tune['max_runs'] = str(1 + len(order))
tune['initial_runs'] = '1'
tune['rand_seed'] = '42'
for k in list(tune):
    if k.startswith('llamatune_') or k in ('min_success', 'sweep_levels'):
        tune.pop(k)
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
with open(cnf + '.clean', 'w') as f:
    f.write('\n'.join(out) + '\n')

print(task_id)
