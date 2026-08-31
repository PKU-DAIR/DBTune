#!/bin/bash -l
#PBS -q debug-c
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/eval_smoke.log
#PBS -N dbtune-eval-smoke
# Smoke test for the warm-start evaluation path, at FULL workload size but only
# 3 tuning iterations (~20 min total): SMAC + workload_map(ottertune) against the
# pool_S0_sweep source. Asserts:
#   - the pool loads (from the pre-built cache) and OtterTune matching selects it
#     at iteration 0 ("Matched context: ...-sweep")
#   - 3 iterations complete, iteration 0 = default config
# Uses its own task_id (eval_smoke_warm) so it cannot contaminate real eval runs.
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
SNAP="${ROOT}/mysql_build/data_150x800k"
TASK="eval_smoke_warm"
PARAL="${ROOT}/parallel/${TASK}"
HIST="${ROOT}/scripts/DBTune_history/history_${TASK}.json"

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
mkdir -p "${STACK_LOG_DIR}" "${ROOT}/logs"
source "${ROOT}/venv/bin/activate"
fail() { echo "[EVAL-SMOKE][FAIL] $*"; exit 1; }
pass() { echo "[EVAL-SMOKE][PASS] $*"; }

cleanup() {
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${PARAL}/data" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
echo "[EVAL-SMOKE] node=$(hostname) start=$(date -Iseconds)"

# warm task for S0:sweep, then shrink to 3 iterations under a smoke-only task_id
GEN="$(python3 "${ROOT}/scripts/lab/eval_gen_task.py" warm S0 sweep 42 "${ROOT}")" || fail "gen_task"
mkdir -p "${PARAL}"
sed -e "s/^task_id = .*/task_id = ${TASK}/" -e "s/^max_runs = .*/max_runs = 3/" \
    -e "s#${ROOT}/parallel/${GEN}#${PARAL}#g" \
    "${ROOT}/parallel/${GEN}/config.ini" > "${PARAL}/config.ini"
sed "s#${ROOT}/parallel/${GEN}#${PARAL}#g" \
    "${ROOT}/parallel/${GEN}/my.cnf.clean" > "${PARAL}/my.cnf.clean"
rm -f "${HIST}"

rm -rf "${PARAL}/data"
cp -a "${SNAP}" "${PARAL}/data" || fail "snapshot copy"
bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" || fail "reset_database"
bash "${ROOT}/scripts/lab/start_stack.sh" || fail "start_stack"

cd "${ROOT}/scripts"
timeout --signal=TERM --kill-after=60 1400 \
    python optimize.py --config="${PARAL}/config.ini" > "${PARAL}/run.out" 2>&1
rc=$?
[[ "${rc}" == "0" ]] || fail "optimize.py rc=${rc} (see ${PARAL}/run.out)"

grep -q "Matched context: .*-sweep" "${PARAL}/run.out" \
    || fail "OtterTune mapping did not select the pool source (no 'Matched context' line)"
pass "workload_map selected the pool_S0_sweep source"

python3 - "${HIST}" <<'PY' || fail "history assertions"
import json, sys
d = json.load(open(sys.argv[1]))['data']
assert len(d) == 3, 'expected 3 iterations, got %d' % len(d)
ok = [r for r in d if r['trial_state'] == 0]
print('iterations: %d (%d SUCCESS), tps: %s'
      % (len(d), len(ok), ['%.0f' % r['external_metrics']['tps'] for r in ok]))
PY
pass "3 iterations recorded"
echo "[EVAL-SMOKE] ALL CHECKS PASSED end=$(date -Iseconds)"
