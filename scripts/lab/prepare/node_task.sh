#!/bin/bash -l
# Run ONE prepare task (task.json in <run_dir>) on THIS compute node.
#
#   usage: node_task.sh <root> <run_dir>
#
# kind=prep : build a dataset snapshot (sysbench: collect_prep.sh with SBTEST_TABLE_SIZE)
# kind=tune : LlamaTune collection on the task's workload (optimize.py, online knobs)
# kind=eval : Sampler[replay] of the task's configs on the workload
# Resume-safe: a complete history exits at once; a partial one continues.
# Tune/Eval run in chunks (default 4 h) with a fresh datadir copy each chunk, so a
# restart-failure streak (sticky redo/recovery state) costs at most one chunk tail.
set -uo pipefail
ROOT="$1"; RUN="$2"
cd "${ROOT}" || exit 1
[[ -f "${RUN}/task.json" ]] || { echo "node_task.sh: no task.json in ${RUN}" >&2; exit 1; }
SPEC="${RUN}/task.json"
field() { python3 -c "import json,sys;print(json.load(open('${SPEC}')).get('$1',''))"; }
KIND="$(field kind)"; TASK_ID="$(field task_id)"; SNAP="${ROOT}/mysql_build/$(field snapshot)"
TIMEOUT_S="$(field timeout_s)"; HIST="$(field history)"
LOG="${RUN}/node.log"
exec >>"${LOG}" 2>&1
echo "[${TASK_ID}] node=$(hostname) start=$(date -Iseconds) kind=${KIND} timeout=${TIMEOUT_S}s"

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
source "${ROOT}/venv/bin/activate"
rm -f "${RUN}/.rc"

cleanup() {
    local rc=$?
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${RUN}/data" 2>/dev/null || true
    echo "[${TASK_ID}] end=$(date -Iseconds) rc=${rc}"
    echo "${rc}" > "${RUN}/.rc"
}
trap cleanup EXIT INT TERM

if [[ "${KIND}" == "prep" ]]; then
    BUILDER="$(python3 -c "import json;print(json.load(open('${SPEC}'))['dataset']['builder'])")"
    case "${BUILDER}" in
        sysbench)
            SIZE="$(python3 -c "import json;print(json.load(open('${SPEC}'))['dataset']['params']['table_size'])")"
            ( cd "${ROOT}" && SBTEST_TABLE_SIZE="${SIZE}" bash "${ROOT}/scripts/lab/collect_prep.sh" ) ;;
        *) echo "[${TASK_ID}] no dataset builder for ${BUILDER}"; exit 2 ;;
    esac
    [[ -d "${SNAP}" ]] && exit 0 || exit 1
fi

WANT="$(python3 -c "import configparser;c=configparser.ConfigParser();c.optionxform=str;c.read('${RUN}/config.ini');print(c['tune']['max_runs'])")"
[[ -n "${WANT}" ]] || { echo "[${TASK_ID}] config.ini has no [tune] max_runs"; exit 1; }
MIN_OK="$(python3 -c "import configparser;c=configparser.ConfigParser();c.optionxform=str;c.read('${RUN}/config.ini');print(c['tune'].get('min_success', 0))")"
CHUNK_S="${PREP_CHUNK_S:-14400}"     # fresh datadir every 4 h: a restart-failure streak costs at most one chunk tail

state() {   # echoes "<rows> <successes>"
    python3 - "${HIST}" <<'PY'
import json, os, sys
fn = sys.argv[1]
if not os.path.exists(fn):
    print('0 0'); raise SystemExit
d = json.load(open(fn))['data']
print(len(d), sum(1 for r in d if r['trial_state'] == 0))
PY
}
complete() {   # done when the attempt cap is reached or (tune only) enough successes exist
    read -r rows ok <<< "$(state)"
    (( rows >= WANT )) || { (( MIN_OK > 0 )) && (( ok >= MIN_OK )); }
}

[[ -d "${SNAP}" ]] || { echo "[${TASK_ID}] snapshot ${SNAP} missing"; exit 1; }
while :; do
    read -r rows ok <<< "$(state)"
    echo "[${TASK_ID}] state: ${rows}/${WANT} rows, ${ok} ok (min_success=${MIN_OK})"
    if complete; then echo "[${TASK_ID}] complete"; exit 0; fi
    remaining=$(( TIMEOUT_S - SECONDS ))
    if (( remaining < 900 )); then echo "[${TASK_ID}] out of time (${rows}/${WANT} rows)"; exit 124; fi
    this_chunk=$(( remaining < CHUNK_S ? remaining : CHUNK_S ))

    echo "[${TASK_ID}] chunk: fresh datadir from $(basename "${SNAP}") ..."; t0=$(date +%s)
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true; sleep 3
    rm -rf "${RUN}/data"
    cp -a "${SNAP}" "${RUN}/data" || { echo "[${TASK_ID}] datadir copy failed"; exit 1; }
    echo "[${TASK_ID}] copied in $(( $(date +%s) - t0 ))s"
    bash "${ROOT}/scripts/lab/reset_database.sh" "${RUN}/config.ini" || { echo "[${TASK_ID}] reset failed"; exit 1; }
    bash "${ROOT}/scripts/lab/start_stack.sh" || { echo "[${TASK_ID}] metrics stack failed"; exit 1; }

    ( cd "${ROOT}/scripts" && timeout --signal=TERM --kill-after=120 "${this_chunk}" \
        python optimize.py --config="${RUN}/config.ini" )
    echo "[${TASK_ID}] chunk rc=$?"
    # rc=0: stop rule reached inside the chunk -> loop re-checks and exits
    # rc=124: chunk timeout -> loop continues with a fresh datadir (history resumes)
done
