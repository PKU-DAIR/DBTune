#!/bin/bash -l
# Run ONE sample-collection cell (source workload x sampling strategy) on THIS
# compute node with a fully isolated MySQL instance. Launched one-per-node by
# collect_launch.sh inside a select=N PBS job.
#
#   usage: collect_node_run.sh <root> <cell:S0|S1> <strategy:sweep|lhs|random|llama> [timeout_s]
#
# Differences from par_node_run.sh (deliberate):
#   - NEVER deletes the history json: cells span multiple 48h waves and resume
#     from scripts/DBTune_history/history_<task_id>.json (Sampler/LlamaTune are
#     resume-safe; the Sampler verifies the already-evaluated prefix).
#   - optimize.py runs under `timeout` so the cell pauses gracefully (rc=124)
#     before PBS kills the job; the next wave continues where it stopped.
#   - datadir is copied from the 150x800k snapshot (mysql_build/data_150x800k,
#     built once by collect_prep.sh), not from the 64x1M template.
#   - no ablation env hacks (WM_SOURCE_ONLY etc.); zipfian exp 0.7 comes from
#     the ini key sysbench_zipfian_exp, not from an exported env var.
set -uo pipefail
ROOT="$1"; CELL="$2"; STRATEGY="$3"
TAG="collect_${CELL}_${STRATEGY}"
PARAL="${ROOT}/parallel/${TAG}"
LOG="${ROOT}/logs/parallel/${TAG}.log"
SNAP="${ROOT}/mysql_build/data_150x800k"
# stop optimize.py this long before the wave's walltime ends (history is saved
# every iteration, so at most one in-flight iteration is lost). Passed as arg 4
# by collect_launch.sh because pbsdsh strips exported env vars.
TIMEOUT_S="${4:-${COLLECT_TIMEOUT_S:-165600}}"   # 46h for a 48h wave

mkdir -p "${ROOT}/logs/parallel"
exec >>"${LOG}" 2>&1
echo "[${TAG}] node=$(hostname) start=$(date -Iseconds) timeout=${TIMEOUT_S}s"

# pbsdsh launches with a MINIMAL env (no USER/HOME/PATH extras).
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-/home/${USER}}"
export LANG="${LANG:-en_US.UTF-8}"
rm -f "${PARAL}/.rc"

export PATH="${ROOT}/sysbench_install/bin:${ROOT}/mysql_build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${ROOT}/mysql_build/lib:${LD_LIBRARY_PATH:-}"
export SYSBENCH_BIN="${ROOT}/sysbench_install/bin/sysbench"
export MYSQL_SOCK="/tmp/dbtune.sock"
export STACK_LOG_DIR="/tmp/dbtune_stack"
export EXPORTER_MY_CNF="${ROOT}/scripts/lab/exporter_my_parallel.cnf"
export DBTUNE_TMP_CNF="/tmp/dbtune_mysqld.cnf"
mkdir -p "${STACK_LOG_DIR}" "$(dirname "${DBTUNE_TMP_CNF}")"

[[ -d "${SNAP}" ]] || { echo "[${TAG}] snapshot ${SNAP} missing - run collect_prep.sh first"; echo 1 > "${PARAL}/.rc"; exit 1; }

source "${ROOT}/venv/bin/activate"

cleanup() {
    local rc=$?
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${PARAL}/data" 2>/dev/null || true   # reclaim ~30GB Lustre per wave
    echo "${rc}" > "${PARAL}/.rc"
}
trap cleanup EXIT INT TERM

# 1) Isolated cnf + config for this cell.
python3 "${ROOT}/scripts/lab/collect_gen_task.py" "${CELL}" "${STRATEGY}" "${ROOT}" \
    || { echo "[${TAG}] gen_task failed"; exit 1; }

# 2) Fresh datadir from the 150x800k snapshot (snapshot mysqld is stopped).
echo "[${TAG}] copying datadir from snapshot ..."; t0=$(date +%s)
rm -rf "${PARAL}/data"
cp -a "${SNAP}" "${PARAL}/data" || { echo "[${TAG}] datadir copy failed"; exit 1; }
echo "[${TAG}] datadir ready in $(( $(date +%s) - t0 ))s"

# 3) Default config on this datadir (starts mysqld on /tmp/dbtune.sock).
bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" \
    || { echo "[${TAG}] reset_database failed"; exit 1; }

# 4) Node-local Prometheus stack (internal metrics for later workload_map use).
bash "${ROOT}/scripts/lab/start_stack.sh" || { echo "[${TAG}] start_stack failed"; exit 1; }

# 5) Run / resume the collection. History json is intentionally preserved.
cd "${ROOT}/scripts"
echo "[${TAG}] optimize.py start=$(date -Iseconds)"
timeout --signal=TERM --kill-after=120 "${TIMEOUT_S}" \
    python optimize.py --config="${PARAL}/config.ini"
rc=$?
if [[ "${rc}" == "0" ]]; then
    echo "[${TAG}] COMPLETE end=$(date -Iseconds)"
elif [[ "${rc}" == "124" ]]; then
    echo "[${TAG}] PAUSED at walltime budget end=$(date -Iseconds) - resume in next wave"
else
    echo "[${TAG}] FAIL rc=${rc} end=$(date -Iseconds)"
fi
exit "${rc}"
