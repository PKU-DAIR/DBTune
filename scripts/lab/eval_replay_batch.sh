#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=1
#PBS -l walltime=08:00:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/eval_replay_batch.log
#PBS -N dbtune-eval-replay
# Transplant evaluation batch for the S2-S4 cells: for each cell, replay
# [default + top-3 source configs] on the TARGET workload (150x800k, 128 threads,
# zipf 0.7) -- 4 evaluations per cell, sequentially on one node.
# Skips cells whose replay history is already complete and cells whose replay
# file does not exist yet (collection still running) -> rerun the same script
# later to pick up the leftovers (e.g. the llama cells).
#
#   qsub scripts/lab/eval_replay_batch.sh
#   qsub -v 'CELLS=S2:llama;S4:llama' scripts/lab/eval_replay_batch.sh   # subset
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
SNAP="${ROOT}/mysql_build/data_150x800k"
if [[ -n "${CELLS:-}" ]]; then IFS=';' read -ra SPECS <<< "${CELLS}"
else SPECS=(S2:random S2:lhs S2:llama S3:random S3:lhs S3:llama S4:random S4:lhs S4:llama); fi
echo "[REPLAY] node=$(hostname) start=$(date -Iseconds) specs=${SPECS[*]}"

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

cleanup() {
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
}
trap cleanup EXIT INT TERM

overall=0
for spec in "${SPECS[@]}"; do
    IFS=: read -r cell strategy <<< "${spec}"
    if [[ ! -f "${ROOT}/scripts/eval/replay_${cell}_${strategy}.json" ]]; then
        echo "[REPLAY] ${spec}: no replay file yet (collection running) - skip"
        continue
    fi
    TASK="eval_replay_${cell}_${strategy}"
    have=$(python3 -c "import json;print(len(json.load(open('${ROOT}/scripts/DBTune_history/history_${TASK}.json'))['data']))" 2>/dev/null || echo 0)
    if (( have >= 4 )); then
        echo "[REPLAY] ${spec}: already complete (${have}/4) - skip"
        continue
    fi
    python3 "${ROOT}/scripts/lab/eval_gen_task.py" replay "${cell}" "${strategy}" 42 "${ROOT}" \
        || { echo "[REPLAY] ${spec}: gen_task failed"; overall=1; continue; }
    PARAL="${ROOT}/parallel/${TASK}"
    echo "[REPLAY] ${spec}: start (have ${have}/4) $(date -Iseconds)"
    rm -rf "${PARAL}/data"
    cp -a "${SNAP}" "${PARAL}/data" || { echo "[REPLAY] ${spec}: copy failed"; overall=1; continue; }
    bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" \
        || { echo "[REPLAY] ${spec}: reset failed"; overall=1; continue; }
    bash "${ROOT}/scripts/lab/start_stack.sh" || { echo "[REPLAY] ${spec}: stack failed"; overall=1; continue; }
    ( cd "${ROOT}/scripts" && timeout --signal=TERM --kill-after=120 4500 \
        python optimize.py --config="${PARAL}/config.ini" )
    rc=$?
    [[ "${rc}" == "0" ]] && echo "[REPLAY] ${spec}: COMPLETE" \
                         || { echo "[REPLAY] ${spec}: rc=${rc}"; overall=1; }
    cleanup 2>/dev/null || true
    rm -rf "${PARAL}/data"
done
echo "[REPLAY] batch done overall=${overall} $(date -Iseconds)"
exit "${overall}"
