#!/bin/bash -l
# Run one node's share of the warm-start evaluation on THIS compute node.
#
#   usage: eval_node_run.sh <root> <cellspec>
#     cellspec = "S0:sweep" ... -> runs: replay (4 evals) + warm seeds 42,43,44 (10 evals each)
#     cellspec = "cold"         -> runs: cold-start seeds 42,43,44 (10 evals each)
#
# Each run gets a fresh datadir from the 150x800k snapshot + default-config reset
# (identical conditions across runs). Histories are never deleted -> reruns of an
# interrupted wave resume; completed runs load their history and exit immediately.
set -uo pipefail
ROOT="$1"; SPEC="$2"
SNAP="${ROOT}/mysql_build/data_150x800k"
NTAG="eval_$(echo "${SPEC}" | tr ':' '_')"
LOG="${ROOT}/logs/parallel/${NTAG}.log"
PER_RUN_TIMEOUT="${EVAL_RUN_TIMEOUT_S:-7200}"   # hard guard per optimize.py run

mkdir -p "${ROOT}/logs/parallel"
exec >>"${LOG}" 2>&1
echo "[${NTAG}] node=$(hostname) start=$(date -Iseconds)"

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
mkdir -p "${RCDIR}"
rm -f "${RCDIR}/.rc"

if [[ "${SPEC}" == "cold" ]]; then
    RUNS=("cold - - 42" "cold - - 43" "cold - - 44")
else
    IFS=: read -r cell strategy <<< "${SPEC}"
    RUNS=("replay ${cell} ${strategy} 42"
          "warm ${cell} ${strategy} 42"
          "warm ${cell} ${strategy} 43"
          "warm ${cell} ${strategy} 44")
fi

cleanup() {
    local rc=$?
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    echo "${rc}" > "${RCDIR}/.rc"
}
trap cleanup EXIT INT TERM

overall=0
for run in "${RUNS[@]}"; do
    read -r mode cell strategy seed <<< "${run}"
    TASK="$(python3 "${ROOT}/scripts/lab/eval_gen_task.py" "${mode}" "${cell}" "${strategy}" "${seed}" "${ROOT}")" \
        || { echo "[${NTAG}] gen_task failed for ${run}"; overall=1; continue; }
    PARAL="${ROOT}/parallel/${TASK}"

    # skip if already complete (resume-friendly reruns of the wave)
    want=10; [[ "${mode}" == "replay" ]] && want=4
    have=$(python3 -c "import json;print(len(json.load(open('${ROOT}/scripts/DBTune_history/history_${TASK}.json'))['data']))" 2>/dev/null || echo 0)
    if (( have >= want )); then
        echo "[${NTAG}] ${TASK} already complete (${have}/${want}) - skip"
        continue
    fi

    echo "[${NTAG}] ${TASK} start=$(date -Iseconds) (have ${have}/${want})"
    rm -rf "${PARAL}/data"
    cp -a "${SNAP}" "${PARAL}/data" || { echo "[${NTAG}] datadir copy failed"; overall=1; continue; }
    bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" \
        || { echo "[${NTAG}] reset_database failed for ${TASK}"; overall=1; continue; }
    bash "${ROOT}/scripts/lab/start_stack.sh" || { echo "[${NTAG}] start_stack failed"; overall=1; continue; }

    ( cd "${ROOT}/scripts" && timeout --signal=TERM --kill-after=120 "${PER_RUN_TIMEOUT}" \
        python optimize.py --config="${PARAL}/config.ini" )
    rc=$?
    if [[ "${rc}" == "0" ]]; then
        echo "[${NTAG}] ${TASK} COMPLETE end=$(date -Iseconds)"
    else
        echo "[${NTAG}] ${TASK} rc=${rc} end=$(date -Iseconds)"
        overall=1
    fi
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${PARAL}/data"
done

exit "${overall}"
