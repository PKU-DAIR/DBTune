#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=1
#PBS -l walltime=01:45:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/probe_online_llama.log
#PBS -N dbtune-probe-online
# 1-hour probe: LlamaTune with ONLINE knob application (95 dynamic knobs, no restarts)
# on the S0 workload. Checks that online apply works and that no restart-failure
# spiral occurs. optimize.py is capped at 3600 s.
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
TAG="probe_online_llama"
PARAL="${ROOT}/parallel/${TAG}"
SNAP="${ROOT}/mysql_build/data_150x800k"
mkdir -p "${PARAL}" "${ROOT}/logs"
echo "[${TAG}] node=$(hostname) start=$(date -Iseconds)"

export USER="${USER:-$(id -un)}"
export HOME="${HOME:-/home/${USER}}"
export LANG="${LANG:-en_US.UTF-8}"
export PATH="${ROOT}/sysbench_install/bin:${ROOT}/mysql_build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${ROOT}/mysql_build/lib:${LD_LIBRARY_PATH:-}"
export SYSBENCH_BIN="${ROOT}/sysbench_install/bin/sysbench"
export MYSQL_SOCK="/tmp/dbtune.sock"
export STACK_LOG_DIR="/tmp/dbtune_stack"
export EXPORTER_MY_CNF="${ROOT}/scripts/lab/exporter_my_parallel.cnf"
export DBTUNE_TMP_CNF="/tmp/dbtune_mysqld.cnf"
mkdir -p "${STACK_LOG_DIR}"
source "${ROOT}/venv/bin/activate"

cleanup() {
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${PARAL}/data" 2>/dev/null || true
    echo "[${TAG}] end=$(date -Iseconds)"
}
trap cleanup EXIT INT TERM

# per-run ini + clean cnf with paths retargeted (same rewrite as collect_gen_task.py)
python3 - "${ROOT}" "${PARAL}" <<'PY'
import configparser, os, sys
root, paral = sys.argv[1:]
c = configparser.ConfigParser(); c.optionxform = str
c.read(os.path.join(root, 'scripts', 'config_probe_online_llama.ini'))
db = c['database']
db['cnf'] = os.path.join(paral, 'my.cnf'); db['sock'] = '/tmp/dbtune.sock'
db['datadir'] = os.path.join(paral, 'data'); db['mysqld'] = os.path.join(root, 'mysql_build/bin/mysqld')
db['host'] = '127.0.0.1'; db['port'] = '3306'
with open(os.path.join(paral, 'config.ini'), 'w') as f:
    c.write(f)
out = []
for line in open(os.path.join(root, 'mysql_build/cnf/my.cnf.clean')):
    k = line.split('=', 1)[0].strip()
    if k == 'datadir':     out.append('datadir = ' + db['datadir'])
    elif k == 'socket':    out.append('socket = /tmp/dbtune.sock')
    elif k == 'pid-file':  out.append('pid-file = /tmp/dbtune.pid')
    elif k == 'log-error': out.append('log-error = /tmp/dbtune.err')
    else:                  out.append(line.rstrip('\n'))
open(os.path.join(paral, 'my.cnf.clean'), 'w').write('\n'.join(out) + '\n')
PY

echo "[${TAG}] fresh datadir ..."; t0=$(date +%s)
rm -rf "${PARAL}/data"
cp -a "${SNAP}" "${PARAL}/data" || { echo "[${TAG}] copy failed"; exit 1; }
echo "[${TAG}] copied in $(( $(date +%s) - t0 ))s"
bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" || { echo "[${TAG}] reset failed"; exit 1; }
bash "${ROOT}/scripts/lab/start_stack.sh" || { echo "[${TAG}] stack failed"; exit 1; }

( cd "${ROOT}/scripts" && timeout --signal=TERM --kill-after=120 3600 \
    python optimize.py --config="${PARAL}/config.ini" )
echo "[${TAG}] optimize.py rc=$?"
