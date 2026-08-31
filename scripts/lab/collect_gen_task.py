#!/usr/bin/env python3
"""Generate the isolated per-cell MySQL cnf + config for a sample-collection cell.

usage: collect_gen_task.py <cell:S0|S1> <strategy:sweep|lhs|random|llama> <root>

Clone of par_gen_task.py for the collection campaign. Creates under
<root>/parallel/collect_<cell>_<strategy>/:
  - my.cnf.clean : base clean cnf with datadir -> this cell's Lustre datadir,
                   socket/pid/log -> node-local /tmp (one cell per node)
  - config.ini   : scripts/config_collect_<cell>_<strategy>.ini with [database]
                   paths retargeted. task_id is NOT overridden: the ini's task_id
                   (miyabic_150-800000-...-<strategy>) names the history json that
                   multi-wave resume depends on.
The datadir itself is copied by collect_node_run.sh from the 150x800k snapshot.
"""
import configparser, os, sys

cell, strategy, root = sys.argv[1], sys.argv[2], sys.argv[3]
tag = f'collect_{cell}_{strategy}'
base_cfg = os.path.join(root, 'scripts', f'config_collect_{cell}_{strategy}.ini')
paral = os.path.join(root, 'parallel', tag)
os.makedirs(paral, exist_ok=True)

datadir = os.path.join(paral, 'data')
cnf = os.path.join(paral, 'my.cnf')
sock = '/tmp/dbtune.sock'          # node-local, uniform (one cell per node)

# 1) Per-cell my.cnf.clean = base clean with datadir/socket/pid/log overridden.
src_clean = os.path.join(root, 'mysql_build/cnf/my.cnf.clean')
out = []
for line in open(src_clean):
    k = line.split('=', 1)[0].strip()
    if k == 'datadir':     out.append(f'datadir = {datadir}')
    elif k == 'socket':    out.append(f'socket = {sock}')
    elif k == 'pid-file':  out.append('pid-file = /tmp/dbtune.pid')
    elif k == 'log-error': out.append('log-error = /tmp/dbtune.err')
    else:                  out.append(line.rstrip('\n'))
# pinned server settings for this cell ([database] cnf_extra = k=v;k=v), e.g. the two
# async-resize knobs held at their defaults in the online (no-resize) variant
_c0 = configparser.ConfigParser(); _c0.optionxform = str; _c0.read(base_cfg)
for kv in (_c0['database'].get('cnf_extra', '') or '').split(';'):
    if kv.strip():
        k, v = kv.split('=', 1)
        out.append(f'{k.strip()} = {v.strip()}')
with open(cnf + '.clean', 'w') as f:
    f.write('\n'.join(out) + '\n')

# 2) Per-cell config.ini = collection ini with [database] paths retargeted.
c = configparser.ConfigParser()
c.optionxform = str
c.read(base_cfg)
c['database']['cnf'] = cnf
c['database']['sock'] = sock
c['database']['datadir'] = datadir
c['database']['mysqld'] = os.path.join(root, 'mysql_build/bin/mysqld')
c['database']['host'] = '127.0.0.1'
c['database']['port'] = '3306'
with open(os.path.join(paral, 'config.ini'), 'w') as f:
    c.write(f)

print(tag)
