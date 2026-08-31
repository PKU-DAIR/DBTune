#!/bin/bash -l
#PBS -q regular-c
#PBS -l select=5
#PBS -l walltime=03:30:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/prep_launch.log
#PBS -N dbtune-prep-probe
# Anchor-probe wave: one probe workload per node, all in parallel (pbsdsh).
# Minimal test of Algorithm 1's probing step: the S0 anchor's best configs are
# run on each probe workload; see scripts/lab/prep_gen_task.py.
#
#   qsub scripts/lab/prep_launch.sh                                   # S1;S2;S3;S4;C64 on 5 nodes
#   qsub -l select=1 -l walltime=04:00:00 -v 'PROBES=R400k' scripts/lab/prep_launch.sh
#   (R400k builds mysql_build/data_150x400k first if it is missing, ~10 min)
# Rerunning resumes: complete probes exit at once, partial ones continue.
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
IFS=';' read -ra PROBES_ARR <<< "${PROBES:-S1;S2;S3;S4;C64}"
mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
echo "[PREP-LAUNCH] head=$(hostname) date=$(date -Iseconds) nodes=${#NODES[@]} probes=${PROBES_ARR[*]}"
if (( ${#NODES[@]} < ${#PROBES_ARR[@]} )); then
    echo "[PREP-LAUNCH][FATAL] need ${#PROBES_ARR[@]} nodes, got ${#NODES[@]}" >&2; exit 1
fi
for i in "${!PROBES_ARR[@]}"; do
    echo "[PREP-LAUNCH] -> vnode ${i} (${NODES[$i]}): ${PROBES_ARR[$i]}"
    pbsdsh -n "${i}" -- /bin/bash "${ROOT}/scripts/lab/prep_node_run.sh" \
        "${ROOT}" "${PROBES_ARR[$i]}" "${PREP_TIMEOUT_S:-10800}" &
done
wait
overall=0
for p in "${PROBES_ARR[@]}"; do
    rc="$(cat "${ROOT}/parallel/prep_${p}/.rc" 2>/dev/null || echo '?')"
    echo "[PREP-LAUNCH] ${p}: rc=${rc}"
    [[ "${rc}" == "0" ]] || overall=1
done
echo "[PREP-LAUNCH] done overall=${overall} $(date -Iseconds)"
exit "${overall}"
