#!/bin/bash -l
# Run ONE S2-S4 collection cell on THIS node, in short chunks.
#
#   usage: collect2_node_run.sh <root> <source:S2|S3|S4> <strategy:random|lhs|llama> <budget_s>
#
# Differences from collect_node_run.sh (the S0/S1 runner):
#   - CHUNKED: repeats [fresh datadir copy -> reset -> stack -> timeout 4h optimize.py]
#     until the cell is done or the wall budget runs out. The S0/S1 campaign showed
#     failures come in long streaks tied to sticky datadir state (huge redo logs etc.);
#     a periodic fresh copy breaks the streaks. Resume makes chunking free.
#   - DONE = >=100 SUCCESS rows (min_success in the ini stops optimize.py) OR
#     >=300 attempts (max_runs).
#   - snapshot: S3 uses data_150x80k, S2/S4 use data_150x800k.
set -uo pipefail
ROOT="$1"; SRC="$2"; STRATEGY="$3"; BUDGET_S="${4:-165600}"
TAG="collect_${SRC}_${STRATEGY}"
PARAL="${ROOT}/parallel/${TAG}"
LOG="${ROOT}/logs/parallel/${TAG}.log"
CHUNK_S="${COLLECT_CHUNK_S:-14400}"   # 4h chunks
if [[ "${SRC}" == "S3" ]]; then SNAP="${ROOT}/mysql_build/data_150x80k"
else                            SNAP="${ROOT}/mysql_build/data_150x800k"; fi

mkdir -p "${ROOT}/logs/parallel" "${PARAL}"
exec >>"${LOG}" 2>&1
echo "[${TAG}] node=$(hostname) start=$(date -Iseconds) budget=${BUDGET_S}s chunk=${CHUNK_S}s"

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
[[ -d "${SNAP}" ]] || { echo "[${TAG}] snapshot ${SNAP} missing - run collect_prep.sh first"; echo 1 > "${PARAL}/.rc"; exit 1; }
source "${ROOT}/venv/bin/activate"
rm -f "${PARAL}/.rc"

tmp_probe() {  # node-local /tmp is 14 GB; log usage + biggest entries so exhaustion is attributable
    echo "[${TAG}] tmp-probe $(date -Iseconds): $(df -h /tmp | awk 'NR==2{print $3" used / "$2", "$5}')"
    du -sh /tmp/dbtune.err /tmp/dbtune_stack "${TMPDIR:-/tmp}"/prom-data-* /tmp/*.err 2>/dev/null | sort -h | tail -4 | sed "s/^/[${TAG}] tmp-probe   /"
    find /tmp -xdev -type f -size +300M 2>/dev/null | head -5 | xargs -r du -sh 2>/dev/null | sed "s/^/[${TAG}] tmp-probe   big: /"
}
( while sleep 3600; do tmp_probe; done ) &
PROBE_PID=$!

# task_id from the ini (histories are keyed by it)
TASK_ID="$(python3 -c "
import configparser; c=configparser.ConfigParser(); c.optionxform=str
c.read('${ROOT}/scripts/config_collect_${SRC}_${STRATEGY}.ini'); print(c['tune']['task_id'])")"
HIST="${ROOT}/scripts/DBTune_history/history_${TASK_ID}.json"

cell_state() {  # echoes "<ok> <attempts>"
    python3 - "${HIST}" <<'PY'
import json, sys, os
fn = sys.argv[1]
if not os.path.exists(fn):
    print('0 0'); raise SystemExit
d = json.load(open(fn))['data']
print(sum(1 for r in d if r['trial_state'] == 0), len(d))
PY
}

cleanup() {
    local rc=$?
    kill "${PROBE_PID}" 2>/dev/null || true
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    "${ROOT}/mysql_build/bin/mysqladmin" -uroot -S "${MYSQL_SOCK}" shutdown 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true
    rm -rf "${PARAL}/data" 2>/dev/null || true
    echo "${rc}" > "${PARAL}/.rc"
}
trap cleanup EXIT INT TERM

# isolated cnf + per-cell ini (paths retargeted; reuses the generic gen script)
python3 "${ROOT}/scripts/lab/collect_gen_task.py" "${SRC}" "${STRATEGY}" "${ROOT}" \
    || { echo "[${TAG}] gen_task failed"; exit 1; }

while :; do
    read -r ok attempts <<< "$(cell_state)"
    echo "[${TAG}] state: ${ok} SUCCESS / ${attempts} attempts"
    if (( ok >= 100 || attempts >= 300 )); then
        echo "[${TAG}] DONE (ok=${ok}, attempts=${attempts}) end=$(date -Iseconds)"
        exit 0
    fi
    remaining=$(( BUDGET_S - SECONDS ))
    if (( remaining < 1800 )); then
        echo "[${TAG}] PAUSED: wall budget exhausted (ok=${ok}, attempts=${attempts})"
        exit 124
    fi
    this_chunk=$(( remaining - 900 < CHUNK_S ? remaining - 900 : CHUNK_S ))

    echo "[${TAG}] chunk start ($(date -Iseconds)), timeout ${this_chunk}s: fresh datadir ..."
    tmp_probe
    bash "${ROOT}/scripts/lab/stop_stack.sh" 2>/dev/null || true
    pkill -x mysqld 2>/dev/null || true; sleep 3
    rm -rf "${PARAL}/data"
    cp -a "${SNAP}" "${PARAL}/data" || { echo "[${TAG}] datadir copy failed"; exit 1; }
    bash "${ROOT}/scripts/lab/reset_database.sh" "${PARAL}/config.ini" \
        || { echo "[${TAG}] reset_database failed"; exit 1; }
    bash "${ROOT}/scripts/lab/start_stack.sh" || { echo "[${TAG}] start_stack failed"; exit 1; }

    ( cd "${ROOT}/scripts" && timeout --signal=TERM --kill-after=120 "${this_chunk}" \
        python optimize.py --config="${PARAL}/config.ini" )
    rc=$?
    echo "[${TAG}] chunk rc=${rc}"
    # rc=0: min_success/max_runs reached inside the chunk -> loop re-checks & exits
    # rc=124: chunk timeout -> loop continues with a fresh datadir
    # other rc: log and keep going (resume is safe); persistent crashes will burn
    # the wall budget and surface as PAUSED with a low attempt count
done
