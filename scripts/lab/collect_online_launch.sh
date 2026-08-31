#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/collect_online_launch.log
#PBS -N dbtune-collect-online
# ONE collection cell on ONE node with the chunked runner (collect2_node_run.sh):
# default = S0 x llama with ONLINE knob application (95 dynamic knobs, no restarts),
# ini scripts/config_collect_S0_llama_online.ini, stop at 100 SUCCESS / 300 attempts.
# Resume-safe: resubmit to continue a PAUSED cell.
#   qsub -o logs/collect_S0_llama_online.log scripts/lab/collect_online_launch.sh
#   qsub -v CELL=S1,STRATEGY=llama_online ... to run another cell/strategy ini.
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
CELL="${CELL:-S0}"; STRATEGY="${STRATEGY:-llama_online}"
BUDGET_S="${COLLECT_BUDGET_S:-39600}"   # 11 h of a 12 h walltime
echo "[COLLECT-ONLINE] ${CELL}_${STRATEGY} head=$(hostname) date=$(date -Iseconds) budget=${BUDGET_S}s"
bash "${ROOT}/scripts/lab/collect2_node_run.sh" "${ROOT}" "${CELL}" "${STRATEGY}" "${BUDGET_S}"
rc="$(cat "${ROOT}/parallel/collect_${CELL}_${STRATEGY}/.rc" 2>/dev/null || echo '?')"
case "${rc}" in
    0)   echo "[COLLECT-ONLINE][DONE]   ${CELL}_${STRATEGY}" ;;
    124) echo "[COLLECT-ONLINE][PAUSED] ${CELL}_${STRATEGY} (resubmit to continue)" ;;
    *)   echo "[COLLECT-ONLINE][FAIL rc=${rc}] ${CELL}_${STRATEGY} (logs/parallel/collect_${CELL}_${STRATEGY}.log)" ;;
esac
echo "[COLLECT-ONLINE] done rc=${rc} $(date -Iseconds)"
[[ "${rc}" == "0" ]]
