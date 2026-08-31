#!/bin/bash -l
# Run ONE tuning task (method x workload x seed) on THIS compute node with a
# fully isolated MySQL instance. Intended to be launched (one per node) by
# par_launch.sh inside a select=N PBS job. Self-contained: node-local socket /
# pid / stack / gen-cnf; per-task Lustre datadir (copied fresh from the template).
#
#   usage: par_node_run.sh <root> <method:ot|dml> <wl:read|rw50|write> <seed>
set -uo pipefail
ROOT="$1"; METHOD="$2"; WL="$3"; SEED="$4"
TAG="${METHOD}_${WL}_s${SEED}"
PARAL="${ROOT}/parallel/${TAG}"
LOG="${ROOT}/logs/parallel/${TAG}.log"
mkdir -p "${ROOT}/logs/parallel"
exec >"${LOG}" 2>&1
echo "[${TAG}] node=$(hostname) start=$(date -Iseconds)"

# pbsdsh launches with a MINIMAL env (no USER/HOME/etc.). Set them defensively
# so start_stack.sh (set -u uses $USER) and python/joblib caches work.
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-/home/${USER}}"
export LANG="${LANG:-en_US.UTF-8}"
rm -f "${PARAL}/.rc"

# --- environment (mirror run_tuning.sh) ---
# Explicit, complete PATH: pbsdsh's minimal PATH omits /usr/sbin & /sbin where
# `ss` (used by mysqldb.py restart-wait) lives, which otherwise fails the trial.
export PATH="${ROOT}/sysbench_install/bin:${ROOT}/mysql_build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${ROOT}/mysql_build/lib:${LD_LIBRARY_PATH:-}"
export SYSBENCH_BIN="${ROOT}/sysbench_install/bin/sysbench"
export MYSQL_SOCK="/tmp/dbtune.sock"
export SYSBENCH_ZIPFIAN_EXP=0.2
# Per-node isolation of otherwise-shared paths:
export STACK_LOG_DIR="/tmp/dbtune_stack"
export EXPORTER_MY_CNF="${ROOT}/scripts/lab/exporter_my_parallel.cnf"
export DBTUNE_TMP_CNF="/tmp/dbtune_mysqld.cnf"
mkdir -p "${STACK_LOG_DIR}" "$(dirname "${DBTUNE_TMP_CNF}")"
# Experiment knobs: source-only surrogate + argmax-of-mean recommendation + GP + seed.
# 'otbo'/'rfbo' = standard BO loop (workload_map transfer + EI acquisition, multi-iter);
# they must NOT use the one-shot source-only / argmax-mean ablation hacks.
case "${METHOD}" in
    otbo|rfbo|otbo1|rfbo1|gpbo|osbo|osbo0|otop|rfop|spop|gpop) STD_BO_LOOP=1 ;;
    *) STD_BO_LOOP=0 ;;
esac
if [[ "${STD_BO_LOOP}" == "0" ]]; then
    export WM_SOURCE_ONLY=1
    export REC_ARGMAX_MEAN=1
    export REC_N_RANDOM="${REC_N_RANDOM:-2000}"
fi
export DBTUNE_SEED="${SEED}"

# For method=rf: identical OT pipeline but with the source PINNED to the RF's offline
# pick (precomputed by scripts/lab/compute_rf_picks.py -> rf_picks.json). This isolates
# the source-selection effect (the same trick that's used for the embedded 'dml' methods).
# RF now uses its own config_sysbench_rf_<wl>.ini (mapping_method=rf at runtime).
# OT uses its native binning on live iter-0 metrics — no forced source.
# 'top1' reuses the OT config but pins FORCE_SOURCE_CONTEXT to the source whose
# offline-computed top-1% Jaccard with the target is highest (test whether
# true top-1% overlap is even a useful selection target).
if [[ "${METHOD}" == "top1" ]]; then
    PICK="$(python3 -c "import json;print(json.load(open('${ROOT}/scripts/lab/top1_picks.json'))['${WL}']['picked_source'])")"
    if [[ -z "${PICK}" ]]; then echo "[${TAG}] could not resolve top1 pick"; exit 1; fi
    export FORCE_SOURCE_CONTEXT="history_${PICK}"
    echo "[${TAG}] FORCE_SOURCE_CONTEXT=${FORCE_SOURCE_CONTEXT}"
fi

source "${ROOT}/venv/bin/activate"

cleanup() {
    local rc=$?
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${PARAL}/data" 2>/dev/null || true   # reclaim 16GB Lustre after run
    echo "${rc}" > "${PARAL}/.rc"                 # success marker for the launcher
}
trap cleanup EXIT INT TERM

# 1) Generate this task's isolated cnf + config.
python3 "${ROOT}/scripts/lab/par_gen_task.py" "${METHOD}" "${WL}" "${SEED}" "${ROOT}" \
    || { echo "[${TAG}] gen_task failed"; exit 1; }

# 2) Fresh datadir = copy of the clean template (template mysqld is stopped).
echo "[${TAG}] copying 16GB datadir ..."; t0=$(date +%s)
rm -rf "${PARAL}/data"
cp -a "${ROOT}/mysql_build/data" "${PARAL}/data" || { echo "[${TAG}] datadir copy failed"; exit 1; }
echo "[${TAG}] datadir ready in $(( $(date +%s) - t0 ))s"

# 3) Reset to default config on this datadir (starts mysqld on /tmp/dbtune.sock).
bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" \
    || { echo "[${TAG}] reset_database failed"; exit 1; }

# 4) Start this node's Prometheus stack (node-local paths via env above).
bash "${ROOT}/scripts/lab/start_stack.sh" || { echo "[${TAG}] start_stack failed"; exit 1; }

# 5) Run the single tuning config (always fresh: drop any prior history so it
#    doesn't resume/no-op a previous partial run).
cd "${ROOT}/scripts"
rm -f "${ROOT}/scripts/DBTune_history/history_${TAG}.json"
echo "[${TAG}] optimize.py start=$(date -Iseconds)  seed=${SEED}"
if python optimize.py --config="${PARAL}/config.ini"; then
    echo "[${TAG}] OK end=$(date -Iseconds)"
    rc=0
else
    rc=$?
    echo "[${TAG}] FAIL rc=${rc} end=$(date -Iseconds)"
fi
exit "${rc}"
