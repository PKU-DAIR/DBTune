#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=8
#PBS -l walltime=48:00:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/collect_launch.log
#PBS -N dbtune-collect
# Launch the 8 sample-collection cells (S0/S1 x sweep/lhs/random/llama) in
# parallel, one per node, inside ONE multi-node PBS job. Cells that do not
# finish within the wave PAUSE at the timeout (rc=124) and RESUME in the next
# wave from scripts/DBTune_history/history_<task_id>.json. Chain waves with:
#
#   w1=$(qsub scripts/lab/collect_launch.sh)
#   qsub -W depend=afterany:${w1} scripts/lab/collect_launch.sh
#
# Prerequisite (once): qsub scripts/lab/collect_prep.sh   # 150x800k snapshot
# Subset override:
#   qsub -l select=2 -v 'TASKS_OVERRIDE=S0:sweep;S0:lhs' scripts/lab/collect_launch.sh
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
echo "[COLLECT] ROOT=${ROOT} head=$(hostname) date=$(date -Iseconds)"

mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
echo "[COLLECT] ${#NODES[@]} nodes: ${NODES[*]}"

if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
    IFS=';' read -ra TASKS <<< "${TASKS_OVERRIDE}"
else
    TASKS=(
        S0:sweep S0:lhs S0:random S0:llama
        S1:sweep S1:lhs S1:random S1:llama
    )
fi
echo "[COLLECT] ${#TASKS[@]} cells: ${TASKS[*]}"

if (( ${#NODES[@]} < ${#TASKS[@]} )); then
    echo "[COLLECT][FATAL] need >= ${#TASKS[@]} nodes but only ${#NODES[@]} allocated" >&2
    exit 1
fi

for i in "${!TASKS[@]}"; do
    IFS=: read -r cell strategy <<< "${TASKS[$i]}"
    echo "[COLLECT] -> vnode ${i}: ${cell} ${strategy}"
    pbsdsh -n "${i}" -- /bin/bash "${ROOT}/scripts/lab/collect_node_run.sh" \
        "${ROOT}" "${cell}" "${strategy}" "${COLLECT_TIMEOUT_S:-165600}" &
done

echo "[COLLECT] waiting for ${#TASKS[@]} cells ..."
wait
overall=0
need_resume=0
for i in "${!TASKS[@]}"; do
    IFS=: read -r cell strategy <<< "${TASKS[$i]}"
    tag="collect_${cell}_${strategy}"
    rc="$(cat "${ROOT}/parallel/${tag}/.rc" 2>/dev/null || echo '?')"
    case "${rc}" in
        0)   echo "[COLLECT][COMPLETE] ${tag}" ;;
        124) echo "[COLLECT][PAUSED]   ${tag} (resume in next wave)"; need_resume=1 ;;
        *)   echo "[COLLECT][FAIL rc=${rc}] ${tag} (see logs/parallel/${tag}.log)"; overall=1 ;;
    esac
done
(( need_resume )) && echo "[COLLECT] some cells paused - submit another wave to finish them"
echo "[COLLECT] wave done. overall=${overall}  date=$(date -Iseconds)"
exit "${overall}"
