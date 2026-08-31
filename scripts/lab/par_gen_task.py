#!/usr/bin/env python3
"""Generate an isolated per-task MySQL config + cnf for a parallel tuning task.

usage: par_gen_task.py <method:ot|dml> <wl:read|rw50|write> <seed> <root>

Creates under <root>/parallel/<tag>/  (tag = <method>_<wl>_s<seed>):
  - my.cnf.clean : base cnf with datadir -> this task's Lustre datadir,
                   socket/pid/log -> node-local /tmp (uniform; 1 task per node)
  - config.ini   : the base method/workload config with [database] paths
                   pointed at this task's cnf/sock/datadir, task_id = tag.
The datadir itself is copied by the per-node runner, not here.
"""
import configparser, os, sys

method, wl, seed, root = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
# 'rf' uses config_sysbench_rf_<wl>.ini (mapping_method=rf at runtime).
# 'top1' reuses the OT base; the per-node runner pins FORCE_SOURCE_CONTEXT from
# scripts/lab/top1_picks.json (offline true-top-1%-overlap pick per target).
base_code = {'ot': 'ot', 'dml': 'dmlmap', 'rf': 'rf', 'top1': 'ot', 'otbo': 'otbo', 'rfbo': 'rfbo',
             'otbo1': 'otbo1', 'rfbo1': 'rfbo1', 'gpbo': 'gpbo',
             'osbo': 'osbo', 'osbo0': 'osbo0',
             'otop': 'otop', 'rfop': 'rfop', 'spop': 'spop', 'gpop': 'gpop'}[method]
tag = f'{method}_{wl}_s{seed}'
base_cfg = os.path.join(root, 'scripts', f'config_sysbench_{base_code}_{wl}.ini')
paral = os.path.join(root, 'parallel', tag)
os.makedirs(paral, exist_ok=True)

datadir = os.path.join(paral, 'data')
cnf = os.path.join(paral, 'my.cnf')
sock = '/tmp/dbtune.sock'          # node-local, uniform (one task per node)

# 1) Per-task my.cnf.clean = base clean with datadir/socket/pid/log overridden.
src_clean = os.path.join(root, 'mysql_build/cnf/my.cnf.clean')
out = []
for line in open(src_clean):
    k = line.split('=', 1)[0].strip()
    if k == 'datadir':     out.append(f'datadir = {datadir}')
    elif k == 'socket':    out.append(f'socket = {sock}')
    elif k == 'pid-file':  out.append('pid-file = /tmp/dbtune.pid')
    elif k == 'log-error': out.append('log-error = /tmp/dbtune.err')
    else:                  out.append(line.rstrip('\n'))
with open(cnf + '.clean', 'w') as f:
    f.write('\n'.join(out) + '\n')

# 2) Per-task config.ini = base config with [database] paths + task_id retargeted.
c = configparser.ConfigParser()
c.optionxform = str
c.read(base_cfg)
c['database']['cnf'] = cnf
c['database']['sock'] = sock
c['database']['datadir'] = datadir
c['database']['mysqld'] = os.path.join(root, 'mysql_build/bin/mysqld')
c['database']['host'] = '127.0.0.1'
c['database']['port'] = '3306'
c['tune']['task_id'] = tag
with open(os.path.join(paral, 'config.ini'), 'w') as f:
    c.write(f)

print(tag)
