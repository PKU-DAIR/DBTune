#!/bin/bash -l
#PBS -q short-c
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -N dbtune-diag
#PBS -o logs/diag_default.log

# Diagnostic: does iter-0 "default config" actually take effect in mysqld, and
# is rw50 TPS reproducible across mysqld restarts with that default config?
# We deliberately first apply a "tuned" config (large redo log), then the
# default, restarting between each, and inspect the *running* variables + the
# on-disk redo log — to catch knobs that don't apply cleanly on restart.

set -uo pipefail
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then cd "${PBS_O_WORKDIR}"; fi
ROOT="$(pwd)"
export LD_LIBRARY_PATH="${ROOT}/mysql_build/lib:${LD_LIBRARY_PATH:-}"
MYSQLD="${ROOT}/mysql_build/bin/mysqld"
MYSQL="${ROOT}/mysql_build/bin/mysql"
MYSQLADMIN="${ROOT}/mysql_build/bin/mysqladmin"
SYSBENCH="${ROOT}/sysbench_install/bin/sysbench"
CNF="${ROOT}/mysql_build/cnf/my.cnf"
CLEAN="${ROOT}/mysql_build/cnf/my.cnf.clean"
SOCK="${ROOT}/mysql_build/mysql.sock"
DATA="${ROOT}/mysql_build/data"

KNOBS=(innodb_buffer_pool_size innodb_read_io_threads innodb_write_io_threads \
       innodb_flush_log_at_trx_commit innodb_adaptive_hash_index sync_binlog \
       innodb_lru_scan_depth innodb_buffer_pool_instances innodb_change_buffer_max_size \
       innodb_io_capacity innodb_log_file_size table_open_cache)

cleanup_files() { rm -f "${SOCK}" "${SOCK}.lock" "${ROOT}/mysql_build/mysql.pid"; }
start() {
    "${MYSQLD}" --defaults-file="${CNF}" >"${ROOT}/logs/diag_mysqld.log" 2>&1 &
    MPID=$!
    for _ in $(seq 1 90); do
        "${MYSQLADMIN}" -uroot -S "${SOCK}" ping >/dev/null 2>&1 && return 0
        if ! kill -0 "${MPID}" 2>/dev/null; then return 1; fi
        sleep 1
    done
    return 1
}
stop() { "${MYSQLADMIN}" -uroot -S "${SOCK}" shutdown 2>/dev/null || true; wait "${MPID}" 2>/dev/null || true; }

# Write my.cnf = clean base + the knob set passed as "key=val key=val ..."
write_cnf() {
    cp "${CLEAN}" "${CNF}"
    for kv in "$@"; do
        echo "${kv/=/		= }" >> "${CNF}"
    done
}
show_runtime() {
    echo "--- running variables ---"
    for k in "${KNOBS[@]}"; do
        "${MYSQL}" -uroot -S "${SOCK}" -Nse "SHOW GLOBAL VARIABLES LIKE '${k}'" 2>/dev/null
    done
    echo "--- on-disk redo log ---"
    ls -la "${DATA}/#innodb_redo/" 2>/dev/null | head -6 || true
    ls -la "${DATA}"/ib_logfile* 2>/dev/null || true
    echo "--- redo capacity ---"
    "${MYSQL}" -uroot -S "${SOCK}" -Nse "SHOW GLOBAL VARIABLES LIKE 'innodb_redo_log_capacity'" 2>/dev/null
}
run_rw50() {
    "${SYSBENCH}" oltp_read_write --mysql-socket="${SOCK}" --mysql-user=root \
        --mysql-db=sbtest --db-driver=mysql --mysql-storage-engine=innodb \
        --tables=64 --table-size=100000 --threads=32 --time=15 --warmup-time=5 \
        --range-size=100 --rand-type=uniform --db-ps-mode=disable --report-interval=0 \
        run 2>&1 | grep -E 'transactions:|queries:' | head -2
}

echo "[INFO] node=$(hostname) date=$(date -Iseconds)"

###############################################################################
echo "############ PHASE 1: apply a TUNED config (large redo), restart ############"
cleanup_files
write_cnf innodb_buffer_pool_size=8589934592 innodb_log_file_size=1073741824 \
          innodb_buffer_pool_instances=4 innodb_flush_log_at_trx_commit=0
start || { echo "[ERR] start failed"; tail -30 "${ROOT}/logs/diag_mysqld.log"; exit 1; }
show_runtime
echo ">>> rw50 under tuned config:"; run_rw50
stop

###############################################################################
echo "############ PHASE 2: apply DEFAULT config, restart, measure x3 ############"
cleanup_files
write_cnf innodb_buffer_pool_size=1073741824 innodb_read_io_threads=2 \
          innodb_write_io_threads=2 innodb_flush_log_at_trx_commit=1 \
          innodb_adaptive_hash_index=1 sync_binlog=1 innodb_lru_scan_depth=1024 \
          innodb_buffer_pool_instances=1 innodb_change_buffer_max_size=25 \
          innodb_io_capacity=100 innodb_log_file_size=50331648 table_open_cache=4000
start || { echo "[ERR] start failed"; tail -30 "${ROOT}/logs/diag_mysqld.log"; exit 1; }
show_runtime
echo ">>> rw50 default run A:"; run_rw50
stop

# restart again with same default and re-measure (determinism across restarts)
cleanup_files
start || { echo "[ERR] restart failed"; exit 1; }
echo ">>> rw50 default run B (after restart, same cnf):"; run_rw50
echo ">>> rw50 default run C (no restart):"; run_rw50
stop

echo "[DONE] diagnostic complete"
