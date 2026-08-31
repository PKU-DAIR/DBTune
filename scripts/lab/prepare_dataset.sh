#!/bin/bash -l
#PBS -q short-c
#PBS -l select=1
#PBS -l walltime=08:00:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -o logs/prepare_dataset.log
#PBS -N sbprep
# Regenerate the template datadir's sbtest dataset to SBTEST_TABLES x SBTEST_TABLE_SIZE.
# Starts mysqld on the shared template datadir (mysql_build/data), runs init_sbtest.sh
# (which DROPs & reloads sbtest), then shuts down. The per-task runner copies this
# template, so all tuning tasks inherit the new dataset.
set -uo pipefail
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "${PBS_O_WORKDIR}"
ROOT="$(pwd)"
export USER="${USER:-$(id -un)}"; export HOME="${HOME:-/home/${USER}}"
export PATH="${ROOT}/sysbench_install/bin:${ROOT}/mysql_build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${ROOT}/mysql_build/lib:${LD_LIBRARY_PATH:-}"

TABLES="${SBTEST_TABLES:-300}"
SIZE="${SBTEST_TABLE_SIZE:-800000}"
CNF="${ROOT}/mysql_build/cnf/my.cnf"
SOCK="${ROOT}/mysql_build/mysql.sock"
MYSQLD="${ROOT}/mysql_build/bin/mysqld"
MYSQLADMIN="${ROOT}/mysql_build/bin/mysqladmin"
echo "[prep] start $(date -Iseconds)  tables=${TABLES} size=${SIZE}"

# 1) start mysqld on the template datadir
"${MYSQLD}" --defaults-file="${CNF}" >"${ROOT}/logs/prep_mysqld.log" 2>&1 &
for _ in $(seq 1 600); do
    [[ -S "${SOCK}" ]] && "${MYSQLADMIN}" -uroot -S "${SOCK}" ping >/dev/null 2>&1 && break
    sleep 1
done
"${MYSQLADMIN}" -uroot -S "${SOCK}" ping >/dev/null 2>&1 || { echo "[prep][FATAL] mysqld did not start"; tail -30 "${ROOT}/logs/prep_mysqld.log"; exit 1; }
echo "[prep] mysqld up $(date -Iseconds)"

# 2) FORCE a clean drop + reload (init_sbtest only drops in its >=target branch,
#    which left a mixed old/new state; do it explicitly here).
MYSQL="${ROOT}/mysql_build/bin/mysql"
echo "[prep] dropping & recreating sbtest ..."
"${MYSQL}" -uroot -S "${SOCK}" -e "DROP DATABASE IF EXISTS sbtest; CREATE DATABASE sbtest;"
echo "[prep] sysbench prepare tables=${TABLES} size=${SIZE} threads=$(nproc) $(date -Iseconds)"
"${ROOT}/sysbench_install/bin/sysbench" oltp_common \
    --mysql-socket="${SOCK}" --mysql-user=root --mysql-db=sbtest \
    --db-driver=mysql --mysql-storage-engine=innodb \
    --tables="${TABLES}" --table-size="${SIZE}" --threads="$(nproc)" prepare
rc=$?
echo "[prep] sysbench prepare rc=${rc} $(date -Iseconds)"

# 3) verify + shut down
ROWS="$("${ROOT}/mysql_build/bin/mysql" -uroot -S "${SOCK}" -Nse 'SELECT COUNT(*) FROM sbtest.sbtest1;' 2>/dev/null || echo 0)"
NT="$("${ROOT}/mysql_build/bin/mysql" -uroot -S "${SOCK}" -Nse "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='sbtest' AND table_name LIKE 'sbtest%';" 2>/dev/null || echo 0)"
echo "[prep] sbtest now: ${NT} tables, sbtest1=${ROWS} rows"
"${MYSQLADMIN}" -uroot -S "${SOCK}" shutdown 2>/dev/null || true
sleep 3; pkill -x mysqld 2>/dev/null || true
du -sh "${ROOT}/mysql_build/data" 2>/dev/null
echo "[prep] done $(date -Iseconds) (expected ${TABLES} tables x ${SIZE} rows)"
[[ "${NT}" == "${TABLES}" && "${ROWS}" == "${SIZE}" ]] && echo "[prep] VERIFY OK" || echo "[prep] VERIFY MISMATCH"