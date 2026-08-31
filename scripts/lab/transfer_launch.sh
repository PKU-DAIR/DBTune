#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=4
#PBS -l walltime=04:30:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/transfer_launch.log
#PBS -N dbtune-transfer
# Transfer-method comparison under a 1 h hard deadline: 4 arms x 3 seeds.
#   node 0: ottertune (SMAC + workload_map source selection over pool_ALL)
#   node 1: rgpe      (SMAC + RGPE ensemble, ResTune-style)
#   node 2: opadviser (SMAC + space_transfer: compact space + source-best init)
#   node 3: cold      (SMAC, no transfer)
# Prereqs: pool_ALL built (build_eval_pools.py + build_pool_all.py), snapshot exists.
# VARIANT=online (qsub -v) = dynamic-knob replication (eval3_ prefix, pool_S0_llama_online);
# VARIANT=fast = all knobs + clean shutdown (eval4_ prefix); default offline = July recipe (kill -9);
# ARMS_OVERRIDE='a;b;c' picks arms (default ottertune rgpe opadviser cold).
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
if [[ -n "${ARMS_OVERRIDE:-}" ]]; then IFS=';' read -ra ARMS <<< "${ARMS_OVERRIDE}"
else ARMS=(ottertune rgpe opadviser cold); fi
VARIANT="${VARIANT:-offline}"
case "${VARIANT}" in online) PREFIX="eval3" ;; fast) PREFIX="eval4" ;; *) PREFIX="eval2" ;; esac
echo "[TRANSFER] head=$(hostname) date=$(date -Iseconds) variant=${VARIANT} arms=${ARMS[*]}"
mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
(( ${#NODES[@]} >= ${#ARMS[@]} )) || { echo "[TRANSFER][FATAL] need ${#ARMS[@]} nodes"; exit 1; }

for i in "${!ARMS[@]}"; do
    echo "[TRANSFER] -> vnode ${i}: ${ARMS[$i]}"
    pbsdsh -n "${i}" -- /bin/bash "${ROOT}/scripts/lab/transfer_node_run.sh" \
        "${ROOT}" "${ARMS[$i]}" "${VARIANT}" &
done
wait

overall=0
for arm in "${ARMS[@]}"; do
    rc="$(cat "${ROOT}/parallel/${PREFIX}_${arm}/.rc" 2>/dev/null || echo '?')"
    [[ "${rc}" == "0" ]] && echo "[TRANSFER][OK] ${arm}" \
                          || { echo "[TRANSFER][FAIL rc=${rc}] ${arm}"; overall=1; }
done
echo "[TRANSFER] done overall=${overall} $(date -Iseconds)"
exit "${overall}"
