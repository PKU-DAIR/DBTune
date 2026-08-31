#!/bin/bash -l
# Run ONE anchor-probe replay (see prep_gen_task.py) on THIS compute node.
#
#   usage: prep_node_run.sh <root> <probe> [timeout_s]
#
# Steps: gen ini/replay list -> (build the 400k snapshot if the probe needs it
# and it is missing) -> fresh datadir copy -> reset to default cnf -> metrics
# stack -> optimize.py (Sampler[replay]) under a hard timeout -> cleanup.
# Resume-safe: a complete history exits immediately; a partial one continues.
set -uo pipefail
ROOT="$1"; PROBE="$2"; TIMEOUT_S="${3:-10800}"
TAG="prep_${PROBE}"
PARAL="${ROOT}/parallel/${TAG}"
LOG="${ROOT}/logs/parallel/${TAG}.log"
mkdir -p "${ROOT}/logs/parallel" "${PARAL}"
exec >>"${LOG}" 2>&1
echo "[${TAG}] node=$(hostname) start=$(date -Iseconds) timeout=${TIMEOUT_S}s"

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
rm -f "${PARAL}/.rc"

cleanup() {
    local rc=$?
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${PARAL}/data" 2>/dev/null || true
    echo "[${TAG}] end=$(date -Iseconds) rc=${rc}"
    echo "${rc}" > "${PARAL}/.rc"
}
trap cleanup EXIT INT TERM

TASK_ID="$(python3 "${ROOT}/scripts/lab/prep_gen_task.py" "${PROBE}" "${ROOT}")" \
    || { echo "[${TAG}] gen_task failed"; exit 1; }
SNAPNAME="$(python3 -c "import json;print(json.load(open('${ROOT}/scripts/eval/${TASK_ID}.json'))['snapshot'])")"
SNAP="${ROOT}/mysql_build/${SNAPNAME}"
HIST="${ROOT}/scripts/DBTune_history/history_${TASK_ID}.json"
WANT="$(python3 -c "import configparser;c=configparser.ConfigParser();c.optionxform=str;c.read('${PARAL}/config.ini');print(c['tune']['max_runs'])")"
have=$(python3 -c "import json;print(len(json.load(open('${HIST}'))['data']))" 2>/dev/null || echo 0)
echo "[${TAG}] task=${TASK_ID} snapshot=${SNAPNAME} have=${have}/${WANT}"
if (( have >= WANT )); then echo "[${TAG}] already complete - skip"; exit 0; fi

# Build the snapshot on this node if it does not exist yet (only R400k needs this).
if [[ ! -d "${SNAP}" ]]; then
    case "${SNAPNAME}" in
        data_150x400k) SIZE=400000 ;;
        *) echo "[${TAG}] snapshot ${SNAP} missing and not buildable here"; exit 1 ;;
    esac
    echo "[${TAG}] building snapshot ${SNAPNAME} (sysbench prepare 150 x ${SIZE}) ..."
    ( cd "${ROOT}" && SBTEST_TABLE_SIZE="${SIZE}" bash "${ROOT}/scripts/lab/collect_prep.sh" ) \
        || { echo "[${TAG}] snapshot build failed"; exit 1; }
    [[ -d "${SNAP}" ]] || { echo "[${TAG}] snapshot still missing after build"; exit 1; }
fi

echo "[${TAG}] fresh datadir from ${SNAPNAME} ..."; t0=$(date +%s)
bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
pkill -x mysqld 2>/dev/null || true; sleep 3
rm -rf "${PARAL}/data"
cp -a "${SNAP}" "${PARAL}/data" || { echo "[${TAG}] datadir copy failed"; exit 1; }
echo "[${TAG}] copied in $(( $(date +%s) - t0 ))s"
bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" \
    || { echo "[${TAG}] reset_database failed"; exit 1; }
bash "${ROOT}/scripts/lab/start_stack.sh" || { echo "[${TAG}] start_stack failed"; exit 1; }

( cd "${ROOT}/scripts" && timeout --signal=TERM --kill-after=120 "${TIMEOUT_S}" \
    python optimize.py --config="${PARAL}/config.ini" )
rc=$?
have=$(python3 -c "import json;print(len(json.load(open('${HIST}'))['data']))" 2>/dev/null || echo 0)
echo "[${TAG}] optimize.py rc=${rc} have=${have}/${WANT}"
(( have >= WANT )) && exit 0 || exit "${rc}"
