#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=6
#PBS -l walltime=03:00:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/par_launch.log
#PBS -N dbtune-par
# Launch many independent single-node tuning tasks IN PARALLEL within ONE
# multi-node PBS job (works around the RUN=2 per-project cap: 1 job, N nodes).
# Each node runs par_node_run.sh with its own isolated mysqld.
#
# Override at submit:
#   qsub -l select=2 -q debug-c -l walltime=00:30:00 \
#        -v 'TASKS_OVERRIDE=ot:read:7 dml:read:7' scripts/lab/par_launch.sh
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
echo "[LAUNCH] ROOT=${ROOT} head=$(hostname) date=$(date -Iseconds)"

# Unique compute nodes in this allocation.
mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
echo "[LAUNCH] ${#NODES[@]} nodes: ${NODES[*]}"

# Task list: "method:wl:seed" entries, ';'-separated (PBS -v dislikes spaces/commas).
# Default = full 2x3x2 = 12.
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
    IFS=';' read -ra TASKS <<< "${TASKS_OVERRIDE}"
else
    TASKS=(
        ot:read:42 ot:rw50:42 ot:write:42 top1:read:42 top1:rw50:42 top1:write:42
    )
fi
echo "[LAUNCH] ${#TASKS[@]} tasks: ${TASKS[*]}"

if (( ${#NODES[@]} < ${#TASKS[@]} )); then
    echo "[LAUNCH][FATAL] need >= ${#TASKS[@]} nodes but only ${#NODES[@]} allocated" >&2
    exit 1
fi

# Dispatch one task per node via pbsdsh (PBS-native; no ssh keys needed).
# `pbsdsh -n <i>` runs on the i-th allocated node; par_node_run.sh is
# self-contained (absolute paths, sources venv, logs to its own file).
for i in "${!TASKS[@]}"; do
    IFS=: read -r m w s <<< "${TASKS[$i]}"
    echo "[LAUNCH] -> vnode ${i}: ${m} ${w} seed=${s}"
    pbsdsh -n "${i}" -- /bin/bash "${ROOT}/scripts/lab/par_node_run.sh" "${ROOT}" "${m}" "${w}" "${s}" &
done

echo "[LAUNCH] waiting for ${#TASKS[@]} tasks ..."
wait
# Report from each task's .rc marker (pbsdsh's own exit code is unreliable here).
overall=0
for i in "${!TASKS[@]}"; do
    IFS=: read -r m w s <<< "${TASKS[$i]}"
    tag="${m}_${w}_s${s}"
    rc="$(cat "${ROOT}/parallel/${tag}/.rc" 2>/dev/null || echo '?')"
    if [[ "${rc}" == "0" ]]; then
        echo "[LAUNCH][OK]   ${tag}"
    else
        echo "[LAUNCH][FAIL rc=${rc}] ${tag} (see logs/parallel/${tag}.log)"
        overall=1
    fi
done
echo "[LAUNCH] all tasks done. overall=${overall}  date=$(date -Iseconds)"
exit "${overall}"
