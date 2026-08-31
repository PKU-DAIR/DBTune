#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=3
#PBS -l walltime=48:00:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -N dbtune-collect2
# S2-S4 collection: ONE strategy per job, its three sources in PARALLEL (3 nodes).
# The job runs until all three cells are done (>=100 SUCCESS or 300 attempts) or
# walltime; unfinished cells resume in a resubmitted job (done cells exit fast).
#
#   qsub -v STRATEGY=random -o logs/collect2_random.log scripts/lab/collect2_launch.sh
#   qsub -v STRATEGY=lhs    -o logs/collect2_lhs.log    scripts/lab/collect2_launch.sh
#   qsub -v STRATEGY=llama  -o logs/collect2_llama.log  scripts/lab/collect2_launch.sh
#
# Prereqs: both snapshots exist (collect_prep.sh, default 800k and
#          qsub -v SBTEST_TABLE_SIZE=80000 for the S3 one).
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
STRATEGY="${STRATEGY:?qsub -v STRATEGY=random|lhs|llama}"
BUDGET_S="${COLLECT_BUDGET_S:-165600}"   # 46h of a 48h wave
SOURCES=(S2 S3 S4)
echo "[COLLECT2] strategy=${STRATEGY} head=$(hostname) date=$(date -Iseconds)"

mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
if (( ${#NODES[@]} < ${#SOURCES[@]} )); then
    echo "[COLLECT2][FATAL] need ${#SOURCES[@]} nodes, got ${#NODES[@]}" >&2; exit 1
fi

for i in "${!SOURCES[@]}"; do
    echo "[COLLECT2] -> vnode ${i}: ${SOURCES[$i]} ${STRATEGY}"
    pbsdsh -n "${i}" -- /bin/bash "${ROOT}/scripts/lab/collect2_node_run.sh" \
        "${ROOT}" "${SOURCES[$i]}" "${STRATEGY}" "${BUDGET_S}" &
done
wait

overall=0
for src in "${SOURCES[@]}"; do
    rc="$(cat "${ROOT}/parallel/collect_${src}_${STRATEGY}/.rc" 2>/dev/null || echo '?')"
    case "${rc}" in
        0)   echo "[COLLECT2][DONE]   ${src}_${STRATEGY}" ;;
        124) echo "[COLLECT2][PAUSED] ${src}_${STRATEGY} (resubmit to continue)"; overall=1 ;;
        *)   echo "[COLLECT2][FAIL rc=${rc}] ${src}_${STRATEGY} (logs/parallel/collect_${src}_${STRATEGY}.log)"; overall=1 ;;
    esac
done
echo "[COLLECT2] job done strategy=${STRATEGY} overall=${overall} date=$(date -Iseconds)"
exit "${overall}"
