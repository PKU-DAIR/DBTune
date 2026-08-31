#!/bin/bash -l
# Run ONE transfer-method arm (3 seeds sequentially) on THIS node, each seed a
# fresh 1 h hard-deadline tuning session on the target workload.
#
#   usage: transfer_node_run.sh <root> <mode:ottertune|rgpe|opadviser|opadviser_ns|cold> [variant:offline|online]
#   variant online = dynamic knobs only, SET GLOBAL without restarts, pool_S0_llama_online source,
#   task/log prefix eval3_ (see transfer_gen_task.py)
#
# Per seed: fresh history (timed sessions never resume), fresh datadir from the
# snapshot, default-config reset, metrics stack, `timeout 3600 optimize.py`.
# rc=124 (deadline hit) is the EXPECTED outcome; rc=0 means max_runs finished early.
# Per-iteration wall-clock offsets are recorded in the history ('update_time').
set -uo pipefail
ROOT="$1"; MODE="$2"; VARIANT="${3:-offline}"
case "${VARIANT}" in online) PREFIX="eval3" ;; fast) PREFIX="eval4" ;; *) PREFIX="eval2" ;; esac
DEADLINE_S="${TRANSFER_DEADLINE_S:-3600}"
SNAP="${ROOT}/mysql_build/data_150x800k"
NTAG="${PREFIX}_${MODE}"
LOG="${ROOT}/logs/parallel/${NTAG}.log"
mkdir -p "${ROOT}/logs/parallel"
exec >>"${LOG}" 2>&1
echo "[${NTAG}] node=$(hostname) start=$(date -Iseconds) deadline=${DEADLINE_S}s variant=${VARIANT}"

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
[[ -d "${SNAP}" ]] || { echo "[${NTAG}] snapshot missing"; exit 1; }
source "${ROOT}/venv/bin/activate"

RCDIR="${ROOT}/parallel/${NTAG}"
mkdir -p "${RCDIR}"; rm -f "${RCDIR}/.rc"

cleanup() {
    local rc=$?
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    echo "${rc}" > "${RCDIR}/.rc"
}
trap cleanup EXIT INT TERM

overall=0
for seed in 42 43 44; do
    TASK="$(python3 "${ROOT}/scripts/lab/transfer_gen_task.py" "${MODE}" "${seed}" "${ROOT}" "${VARIANT}")" \
        || { echo "[${NTAG}] gen_task failed seed=${seed}"; overall=1; continue; }
    PARAL="${ROOT}/parallel/${TASK}"
    # timed session: always start fresh (resume would corrupt the timing record)
    rm -f "${ROOT}/scripts/DBTune_history/history_${TASK}.json"
    echo "[${NTAG}] ${TASK} start=$(date -Iseconds)"
    rm -rf "${PARAL}/data"
    cp -a "${SNAP}" "${PARAL}/data" || { echo "[${NTAG}] copy failed"; overall=1; continue; }
    bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" \
        || { echo "[${NTAG}] reset failed ${TASK}"; overall=1; continue; }
    bash "${ROOT}/scripts/lab/start_stack.sh" || { echo "[${NTAG}] stack failed"; overall=1; continue; }

    ( cd "${ROOT}/scripts" && timeout --signal=TERM --kill-after=120 "${DEADLINE_S}" \
        python optimize.py --config="${PARAL}/config.ini" )
    rc=$?
    n=$(python3 -c "import json;print(len(json.load(open('${ROOT}/scripts/DBTune_history/history_${TASK}.json'))['data']))" 2>/dev/null || echo 0)
    case "${rc}" in
        124) echo "[${NTAG}] ${TASK} DEADLINE (expected): ${n} iterations in ${DEADLINE_S}s" ;;
        0)   echo "[${NTAG}] ${TASK} finished max_runs early: ${n} iterations" ;;
        *)   echo "[${NTAG}] ${TASK} rc=${rc} after ${n} iterations (check log)"; overall=1 ;;
    esac
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${PARAL}/data"
done
echo "[${NTAG}] all seeds done overall=${overall} $(date -Iseconds)"
exit "${overall}"
