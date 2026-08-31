#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=9
#PBS -l walltime=08:00:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/eval_launch.log
#PBS -N dbtune-eval
# Warm-start evaluation wave: 9 nodes.
#   nodes 0-7 : one collection cell each -> transplant replay (4 evals) +
#               warm-started SMAC x 3 seeds (10 evals each)
#   node 8    : cold-start SMAC x 3 seeds (10 evals each)
# Rerunning the same script resumes: complete runs are skipped, partial runs
# continue from their saved history.
# Prereqs: scripts/build_eval_pools.py has been run (pools + replay files),
#          mysql_build/data_150x800k snapshot exists.
# Subset:  qsub -l select=1 -v 'TASKS_OVERRIDE=S0:sweep' scripts/lab/eval_launch.sh
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
echo "[EVAL] ROOT=${ROOT} head=$(hostname) date=$(date -Iseconds)"

mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
    IFS=';' read -ra TASKS <<< "${TASKS_OVERRIDE}"
else
    TASKS=(S0:sweep S0:lhs S0:random S0:llama S1:sweep S1:lhs S1:random S1:llama cold)
fi
echo "[EVAL] ${#NODES[@]} nodes for ${#TASKS[@]} specs: ${TASKS[*]}"
if (( ${#NODES[@]} < ${#TASKS[@]} )); then
    echo "[EVAL][FATAL] need ${#TASKS[@]} nodes, got ${#NODES[@]}" >&2; exit 1
fi

for i in "${!TASKS[@]}"; do
    echo "[EVAL] -> vnode ${i}: ${TASKS[$i]}"
    pbsdsh -n "${i}" -- /bin/bash "${ROOT}/scripts/lab/eval_node_run.sh" \
        "${ROOT}" "${TASKS[$i]}" &
done
wait

overall=0
for spec in "${TASKS[@]}"; do
    ntag="eval_$(echo "${spec}" | tr ':' '_')"
    rc="$(cat "${ROOT}/parallel/${ntag}/.rc" 2>/dev/null || echo '?')"
    if [[ "${rc}" == "0" ]]; then
        echo "[EVAL][OK]   ${spec}"
    else
        echo "[EVAL][FAIL rc=${rc}] ${spec} (see logs/parallel/${ntag}.log)"
        overall=1
    fi
done
echo "[EVAL] wave done overall=${overall} date=$(date -Iseconds)"
exit "${overall}"
