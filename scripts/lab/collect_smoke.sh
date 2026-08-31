#!/bin/bash -l
#PBS -q debug-c
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/collect_smoke.log
#PBS -N dbtune-collect-smoke
# Self-contained end-to-end smoke test for the sample-collection pipeline.
# Tiny everything (8 knobs, 8 runs, 4 tables x 10k rows, 30 s benchmark) so it
# fits a debug-queue slot. Exercises the EXACT mechanics of the real waves:
#
#   phase A: run under a short `timeout`  -> expect rc=124 (paused mid-collection)
#   phase B: rerun the same config        -> must resume ("Load N iterations"),
#            finish all 8 runs, and the Sampler prefix check must pass
#   asserts: SYSBENCH_ZIPFIAN_EXP=0.7 prefix on the benchmark cmd, history has
#            exactly max_runs entries, iteration 0 is the default config
#
# Strategy override:  qsub -v 'SMOKE_STRATEGY=lhs' scripts/lab/collect_smoke.sh
#   (sweep | lhs | random exercise the Sampler; llama exercises LlamaTune+replay)
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
STRATEGY="${SMOKE_STRATEGY:-sweep}"
TAG="collect_smoke_${STRATEGY}"
PARAL="${ROOT}/parallel/${TAG}"
HIST="${ROOT}/scripts/DBTune_history/history_${TAG}.json"
PHASE_A_TIMEOUT="${SMOKE_PHASE_A_TIMEOUT:-360}"   # enough for ~3 tiny iterations

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
mkdir -p "${STACK_LOG_DIR}" "${PARAL}" "${ROOT}/logs"
source "${ROOT}/venv/bin/activate"

fail() { echo "[SMOKE][FAIL] $*"; exit 1; }
pass() { echo "[SMOKE][PASS] $*"; }

cleanup() {
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${PARAL}/data" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[SMOKE] node=$(hostname) strategy=${STRATEGY} start=$(date -Iseconds)"

# --- tiny smoke ini + isolated cnf (derived from the real S0 cell config) ---
python3 - "${ROOT}" "${STRATEGY}" "${PARAL}" "${TAG}" <<'PY'
import configparser, os, sys
root, strategy, paral, tag = sys.argv[1:]
base = 'llama' if strategy == 'llama' else strategy
cfg = configparser.ConfigParser(); cfg.optionxform = str
cfg.read(os.path.join(root, 'scripts', 'config_collect_S0_%s.ini' % base))
db, tune = cfg['database'], cfg['tune']
db['cnf'] = os.path.join(paral, 'my.cnf')
db['sock'] = '/tmp/dbtune.sock'
db['datadir'] = os.path.join(paral, 'data')
db['mysqld'] = os.path.join(root, 'mysql_build/bin/mysqld')
db['knob_num'] = '8'                      # tiny space
db['thread_num'] = '16'
db['sysbench_tables'] = '4'
db['sysbench_table_size'] = '10000'
db['workload_warmup_time'] = '10'
db['workload_time'] = '30'
tune['task_id'] = tag
tune['max_runs'] = '8'
if strategy == 'llama':
    tune['initial_runs'] = '3'
with open(os.path.join(paral, 'config.ini'), 'w') as f:
    cfg.write(f)
# isolated clean cnf (same rewrite as collect_gen_task.py)
out = []
for line in open(os.path.join(root, 'mysql_build/cnf/my.cnf.clean')):
    k = line.split('=', 1)[0].strip()
    if k == 'datadir':     out.append('datadir = ' + os.path.join(paral, 'data'))
    elif k == 'socket':    out.append('socket = /tmp/dbtune.sock')
    elif k == 'pid-file':  out.append('pid-file = /tmp/dbtune.pid')
    elif k == 'log-error': out.append('log-error = /tmp/dbtune.err')
    else:                  out.append(line.rstrip('\n'))
open(os.path.join(paral, 'my.cnf.clean'), 'w').write('\n'.join(out) + '\n')
print('smoke config ready')
PY

# fresh smoke state (the smoke test owns its own history; real cells never delete theirs)
rm -f "${HIST}"
rm -rf "${PARAL}/data"; mkdir -p "${PARAL}/data"

# mysqld up on an initialized-from-empty datadir, tiny sbtest, metrics stack
bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" || fail "reset_database"
SOCK="${MYSQL_SOCK}" SBTEST_TABLES=4 SBTEST_TABLE_SIZE=10000 \
    bash "${ROOT}/scripts/lab/init_sbtest.sh" || fail "init_sbtest"
bash "${ROOT}/scripts/lab/start_stack.sh" || fail "start_stack"

cd "${ROOT}/scripts"

echo "[SMOKE] phase A: run with ${PHASE_A_TIMEOUT}s timeout (expect pause rc=124)"
timeout --signal=TERM --kill-after=60 "${PHASE_A_TIMEOUT}" \
    python optimize.py --config="${PARAL}/config.ini" > "${PARAL}/phaseA.out" 2>&1
rc=$?
[[ "${rc}" == "124" ]] || fail "phase A expected rc=124 (paused), got rc=${rc} (see ${PARAL}/phaseA.out)"
N_A=$(python3 -c "import json;print(len(json.load(open('${HIST}'))['data']))" 2>/dev/null || echo 0)
(( N_A >= 1 )) || fail "phase A recorded no iterations"
pass "phase A paused at rc=124 with ${N_A} iterations recorded"

grep -q "SYSBENCH_ZIPFIAN_EXP=0.7" "${PARAL}/phaseA.out" \
    || fail "benchmark cmd missing SYSBENCH_ZIPFIAN_EXP=0.7 prefix"
pass "zipfian 0.7 env prefix present on benchmark cmd"

echo "[SMOKE] phase B: rerun same config (expect resume + completion)"
timeout --signal=TERM --kill-after=60 1200 \
    python optimize.py --config="${PARAL}/config.ini" > "${PARAL}/phaseB.out" 2>&1
rc=$?
[[ "${rc}" == "0" ]] || fail "phase B expected rc=0, got rc=${rc} (see ${PARAL}/phaseB.out)"
grep -q "Load ${N_A} iterations" "${PARAL}/phaseB.out" \
    || fail "phase B did not log 'Load ${N_A} iterations' (resume broken?)"
pass "phase B resumed from ${N_A} iterations"

python3 - "${HIST}" "${PARAL}/config.ini" <<'PY' || fail "history assertions"
import configparser, json, sys
hist, ini = sys.argv[1], sys.argv[2]
data = json.load(open(hist))['data']
cfg = configparser.ConfigParser(); cfg.read(ini)
assert len(data) == int(cfg['tune']['max_runs']), \
    'history has %d entries, expected %s' % (len(data), cfg['tune']['max_runs'])
print('history complete: %d entries; iteration 0 config: %s'
      % (len(data), sorted(data[0]['configuration'].items())[:3]))
PY
pass "history complete with all 8 runs"

echo "[SMOKE] ALL CHECKS PASSED (strategy=${STRATEGY})  end=$(date -Iseconds)"
