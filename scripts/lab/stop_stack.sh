#!/bin/bash -l
# Stop the Prometheus stack started by start_stack.sh. Idempotent.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STACK_LOG_DIR="${STACK_LOG_DIR:-${ROOT}/logs/stack}"

for proc in mysqld_exporter node_exporter prometheus; do
    pidfile="${STACK_LOG_DIR}/${proc}.pid"
    if [[ -f "${pidfile}" ]]; then
        pid="$(cat "${pidfile}")"
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
        rm -f "${pidfile}"
    fi
    # Catch any strays
    pkill -x "${proc}" 2>/dev/null || true
done
# drop the Prometheus TSDB this stack wrote (node-local /tmp is 14 GB; a long job
# with several chunks must not accumulate one TSDB per chunk)
if [[ -f "${STACK_LOG_DIR}/tsdb.path" ]]; then
    tsdb="$(cat "${STACK_LOG_DIR}/tsdb.path")"
    [[ -n "${tsdb}" && -d "${tsdb}" && "${tsdb}" == *prom-data-* ]] && rm -rf "${tsdb}"
    rm -f "${STACK_LOG_DIR}/tsdb.path"
fi
echo "[OK] stack stopped"
