#!/bin/bash -l
# Ensure the `sbtest` schema/dataset exists. DATA ONLY — assumes mysqld is
# already running (bring it up first with restart_default.sh). Does not start,
# stop, or initialize mysqld.
#
#   1. CREATE DATABASE sbtest if missing.
#   2. sysbench prepare to populate sbtest (default 64 tables x 100k rows, to
#      match autotune/dbenv.py defaults) if not already populated.
# Safe to re-run: gated on existing table count.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MYSQL="${ROOT}/mysql_build/bin/mysql"
MYSQLADMIN="${ROOT}/mysql_build/bin/mysqladmin"
SOCK="${SOCK:-${ROOT}/mysql_build/mysql.sock}"
SYSBENCH="${ROOT}/sysbench_install/bin/sysbench"
SBTEST_TABLES="${SBTEST_TABLES:-64}"
SBTEST_TABLE_SIZE="${SBTEST_TABLE_SIZE:-100000}"

for bin in "${MYSQL}" "${SYSBENCH}"; do
    [[ -x "${bin}" ]] || { echo "[ERROR] missing binary: ${bin}" >&2; exit 1; }
done

# Require a running server (started by restart_default.sh).
if ! "${MYSQLADMIN}" -uroot -S "${SOCK}" ping >/dev/null 2>&1; then
    echo "[ERROR] mysqld is not running at ${SOCK}." >&2
    echo "        Start it first: scripts/lab/restart_default.sh <config.ini>" >&2
    exit 1
fi

echo "[STEP] ensuring sbtest database exists ..."
"${MYSQL}" -uroot -S "${SOCK}" -e "CREATE DATABASE IF NOT EXISTS sbtest;"

TABLE_COUNT="$("${MYSQL}" -uroot -S "${SOCK}" -Nse \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='sbtest' AND table_name LIKE 'sbtest%';")"

# Detect a table-size change: if sbtest1 exists with a different row count than
# the requested SBTEST_TABLE_SIZE, drop & reload so the dataset matches.
NEED_RELOAD=0
if (( TABLE_COUNT >= SBTEST_TABLES )); then
    ROWS="$("${MYSQL}" -uroot -S "${SOCK}" -Nse "SELECT COUNT(*) FROM sbtest.sbtest1;" 2>/dev/null || echo 0)"
    if (( ROWS != SBTEST_TABLE_SIZE )); then
        echo "[INFO] sbtest1 has ${ROWS} rows but target is ${SBTEST_TABLE_SIZE}; dropping & reloading"
        "${MYSQL}" -uroot -S "${SOCK}" -e "DROP DATABASE sbtest; CREATE DATABASE sbtest;"
        NEED_RELOAD=1
    fi
else
    NEED_RELOAD=1
fi

if (( NEED_RELOAD == 0 )); then
    echo "[SKIP] sbtest already has ${SBTEST_TABLES} tables x ${SBTEST_TABLE_SIZE} rows"
else
    echo "[STEP] sysbench prepare (tables=${SBTEST_TABLES}, table-size=${SBTEST_TABLE_SIZE}) ..."
    "${SYSBENCH}" oltp_common \
        --mysql-socket="${SOCK}" \
        --mysql-user=root \
        --mysql-db=sbtest \
        --db-driver=mysql \
        --mysql-storage-engine=innodb \
        --tables="${SBTEST_TABLES}" \
        --table-size="${SBTEST_TABLE_SIZE}" \
        --threads="$(nproc)" \
        prepare
fi

echo "[DONE] sbtest ready (${SBTEST_TABLES} tables x ${SBTEST_TABLE_SIZE} rows)."
