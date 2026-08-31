#!/bin/bash -l
#PBS -q debug-c
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/transfer_smoke.log
#PBS -N dbtune-transfer-smoke
# Smoke one transfer arm at FULL workload size for 3 iterations against pool_ALL.
# Purpose: exercise the code paths never run in this repo by us (rgpe surrogate,
# space_transfer init) before committing the 12-run wave.
#   qsub -v MODE=rgpe      scripts/lab/transfer_smoke.sh
#   qsub -v MODE=opadviser scripts/lab/transfer_smoke.sh
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
MODE="${MODE:?qsub -v MODE=rgpe|opadviser|ottertune|cold}"
SNAP="${ROOT}/mysql_build/data_150x800k"
TASK="eval2smoke_${MODE}${GRACEFUL:+_graceful}"   # qsub -v MODE=cold,GRACEFUL=1 -> clean-shutdown variant
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
fail() { echo "[TSMOKE-${MODE}][FAIL] $*"; exit 1; }
pass() { echo "[TSMOKE-${MODE}][PASS] $*"; }
cleanup() {
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${PARAL}/data" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
echo "[TSMOKE-${MODE}] node=$(hostname) start=$(date -Iseconds)"

GEN="$(python3 "${ROOT}/scripts/lab/transfer_gen_task.py" "${MODE}" 42 "${ROOT}")" || fail gen_task
mkdir -p "${PARAL}"
sed -e "s/^task_id = .*/task_id = ${TASK}/" -e "s/^max_runs = .*/max_runs = 3/" \
    -e "s#${ROOT}/parallel/${GEN}#${PARAL}#g" \
    "${ROOT}/parallel/${GEN}/config.ini" > "${PARAL}/config.ini"
sed "s#${ROOT}/parallel/${GEN}#${PARAL}#g" \
    "${ROOT}/parallel/${GEN}/my.cnf.clean" > "${PARAL}/my.cnf.clean"
if [[ "${GRACEFUL:-}" == "1" ]]; then   # the offline variant pins graceful_shutdown = False; flip it
    sed -i "/^graceful_shutdown = /d" "${PARAL}/config.ini"
    sed -i "s/^\[database\]$/[database]\ngraceful_shutdown = True/" "${PARAL}/config.ini"
fi
rm -f "${HIST}"

rm -rf "${PARAL}/data"
cp -a "${SNAP}" "${PARAL}/data" || fail "snapshot copy"
bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" || fail reset_database
bash "${ROOT}/scripts/lab/start_stack.sh" || fail start_stack

cd "${ROOT}/scripts"
timeout --signal=TERM --kill-after=60 1400 \
    python optimize.py --config="${PARAL}/config.ini" > "${PARAL}/run.out" 2>&1
rc=$?
[[ "${rc}" == "0" ]] || fail "optimize.py rc=${rc} (tail: $(tail -3 ${PARAL}/run.out | tr '\n' ' '))"

python3 - "${HIST}" <<'PY' || fail "history assertions"
import json, sys
d = json.load(open(sys.argv[1]))['data']
assert len(d) == 3, len(d)
assert all(r.get('update_time') is not None for r in d), 'update_time missing'
print('3 iterations; update_time offsets: %s; tps: %s'
      % (['%.0fs' % r['update_time'] for r in d],
         ['%.0f' % r['external_metrics']['tps'] if r['trial_state']==0 else 'FAIL' for r in d]))
PY
pass "3 iterations with wall-clock stamps"
echo "[TSMOKE-${MODE}] ALL CHECKS PASSED end=$(date -Iseconds)"
