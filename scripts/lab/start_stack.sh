#!/bin/bash -l
# Start the Prometheus metrics stack used by DBTune for OT/DML internal metrics.
# - mysqld_exporter listens on :9104  (scraped as instance "mysqld-exporter:9104")
# - node_exporter   listens on :9100  (scraped as instance "node-exporter:9100")
# - prometheus      listens on :9090

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

STACK_LOG_DIR="${STACK_LOG_DIR:-${ROOT}/logs/stack}"
TSDB_DIR="${TMPDIR:-/tmp}/prom-data-${USER:-$(id -un)}-$$"
mkdir -p "${STACK_LOG_DIR}" "${TSDB_DIR}"

# Tear down anything we may have left running (idempotent, never fails).
for proc in mysqld_exporter node_exporter prometheus; do
    pkill -x "${proc}" 2>/dev/null || true
done
# Give the kernel a moment to release listening sockets.
sleep 1

# 1) mysqld_exporter — DBTune's mysqldb.py expects pgrep -x mysqld_exporter
#    to match a real native process.
"${ROOT}/tools/mysqld_exporter_dir/mysqld_exporter" \
    --config.my-cnf="${EXPORTER_MY_CNF:-${ROOT}/scripts/lab/exporter_my.cnf}" \
    --web.listen-address=:9104 \
    >"${STACK_LOG_DIR}/mysqld_exporter.log" 2>&1 &
echo "$!" > "${STACK_LOG_DIR}/mysqld_exporter.pid"

# 2) node_exporter
"${ROOT}/tools/node_exporter_dir/node_exporter" \
    --web.listen-address=:9100 \
    >"${STACK_LOG_DIR}/node_exporter.log" 2>&1 &
echo "$!" > "${STACK_LOG_DIR}/node_exporter.pid"

# 3) Prometheus (TSDB scoped to job-local /tmp; PromQL queries hit :9090)
"${ROOT}/tools/prometheus/prometheus" \
    --config.file="${ROOT}/scripts/lab/prometheus.yml" \
    --storage.tsdb.path="${TSDB_DIR}" \
    --web.listen-address=:9090 \
    >"${STACK_LOG_DIR}/prometheus.log" 2>&1 &
echo "$!" > "${STACK_LOG_DIR}/prometheus.pid"
echo "${TSDB_DIR}" > "${STACK_LOG_DIR}/tsdb.path"

# Wait up to ~60s for Prometheus to report ready.
for _ in $(seq 1 60); do
    if curl -fsS -o /dev/null http://localhost:9090/-/ready 2>/dev/null; then
        echo "[OK] Prometheus ready at http://localhost:9090"
        echo "     mysqld_exporter pid=$(cat "${STACK_LOG_DIR}/mysqld_exporter.pid")"
        echo "     node_exporter   pid=$(cat "${STACK_LOG_DIR}/node_exporter.pid")"
        echo "     prometheus      pid=$(cat "${STACK_LOG_DIR}/prometheus.pid")"
        exit 0
    fi
    sleep 1
done
echo "[ERROR] Prometheus did not become ready within 60s" >&2
tail -20 "${STACK_LOG_DIR}/prometheus.log" >&2 || true
exit 1
