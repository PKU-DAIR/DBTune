#!/usr/bin/env bash
set -euo pipefail

# Run from repo root or anywhere
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/mysql_build"
CNF="${BUILD_DIR}/cnf/my.cnf"

# Prefer the built mysqld under mysql_build/bin. Fall back to PATH if needed.
MYSQLD="${BUILD_DIR}/bin/mysqld"
MYSQLADMIN="${BUILD_DIR}/bin/mysqladmin"

if [[ ! -x "${MYSQLD}" ]]; then
    echo "[ERROR] mysqld not found or not executable: ${MYSQLD}"
    echo "        Build MySQL first, or adjust MYSQLD path."
    exit 1
fi

if [[ ! -f "${CNF}" ]]; then
    echo "[ERROR] my.cnf not found: ${CNF}"
    exit 1
fi

DATA_DIR="${BUILD_DIR}/data"
SOCKET="${BUILD_DIR}/mysql.sock"
PIDFILE="${BUILD_DIR}/mysql.pid"

mkdir -p "${DATA_DIR}"

# Detect first run by presence of system tables
if [[ ! -d "${DATA_DIR}/mysql" ]]; then
    echo "[INFO] First run detected. Initializing insecure..."
    "${MYSQLD}" --defaults-file="${CNF}" --initialize-insecure
    echo "[INFO] Initialization complete."
fi
