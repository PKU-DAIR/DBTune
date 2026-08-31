#!/bin/bash -l
#PBS -q short-c
#PBS -l select=1
#PBS -l walltime=00:40:00
#PBS -W group_list=xg26g002
#PBS -j oe
#PBS -N dbtune-warmup
#PBS -o logs/diag_warmup.log

# Test whether the 5s warmup is too short for a large (cold) buffer pool.
# For each configuration: restart mysqld (empties the pool), then run rw50 for
# 180s with 10s reporting and NO warmup. If TPS ramps up over the first
# minute or two, the short warmup is measuring a cold pool. Buffer-pool fill %
# is sampled at start and end to corroborate.

set -uo pipefail
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then cd "${PBS_O_WORKDIR}"; fi
ROOT="$(pwd)"
export LD_LIBRARY_PATH="${ROOT}/mysql_build/lib:${LD_LIBRARY_PATH:-}"
MYSQLD="${ROOT}/mysql_build/bin/mysqld"
MYSQL="${ROOT}/mysql_build/bin/mysql"
MYSQLADMIN="${ROOT}/mysql_build/bin/mysqladmin"
SYSBENCH="${ROOT}/sysbench_install/bin/sysbench"
CNF="${ROOT}/mysql_build/cnf/my.cnf"
CLEAN="${CNF}.clean"
SOCK="${ROOT}/mysql_build/mysql.sock"

write_cnf() { cp "${CLEAN}" "${CNF}"; for kv in "$@"; do echo "${kv/=/		= }" >> "${CNF}"; done; }
cleanup_files() { rm -f "${SOCK}" "${SOCK}.lock" "${ROOT}/mysql_build/mysql.pid"; }
start() {
    "${MYSQLD}" --defaults-file="${CNF}" >"${ROOT}/logs/warmup_mysqld.log" 2>&1 &
    MPID=$!
    for _ in $(seq 1 300); do
        "${MYSQLADMIN}" -uroot -S "${SOCK}" ping >/dev/null 2>&1 && return 0
        kill -0 "${MPID}" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}
stop() { "${MYSQLADMIN}" -uroot -S "${SOCK}" shutdown 2>/dev/null || true; wait "${MPID}" 2>/dev/null || true; }
poolfill() {
    "${MYSQL}" -uroot -S "${SOCK}" -Nse \
      "SELECT ROUND(100*
        (SELECT VARIABLE_VALUE FROM performance_schema.global_status WHERE VARIABLE_NAME='Innodb_buffer_pool_pages_data')/
        (SELECT VARIABLE_VALUE FROM performance_schema.global_status WHERE VARIABLE_NAME='Innodb_buffer_pool_pages_total'),1)" 2>/dev/null
}

run_ramp() {
    local label="$1"; shift
    cleanup_files
    write_cnf "$@"
    if ! start; then echo "[ERR] start failed for ${label}"; tail -20 "${ROOT}/logs/warmup_mysqld.log"; return 1; fi
    local bp; bp="$("${MYSQL}" -uroot -S "${SOCK}" -Nse 'SELECT @@innodb_buffer_pool_size, @@innodb_buffer_pool_instances')"
    echo "=== ${label}  (bp_size,inst = ${bp})  cold start, 180s @ 10s intervals ==="
    echo "    poolfill @ start = $(poolfill)%"
    "${SYSBENCH}" oltp_read_write --mysql-socket="${SOCK}" --mysql-user=root \
        --mysql-db=sbtest --db-driver=mysql --mysql-storage-engine=innodb \
        --tables=64 --table-size=100000 --threads=32 --time=180 --warmup-time=0 \
        --range-size=100 --rand-type=uniform --db-ps-mode=disable --report-interval=10 \
        run 2>&1 | grep -E '\[ *[0-9]+s \]' | sed 's/^/    /'
    echo "    poolfill @ end   = $(poolfill)%"
    stop
}

echo "[INFO] node=$(hostname) date=$(date -Iseconds)"
run_ramp "DEFAULT (1GB, inst=1)" \
    innodb_buffer_pool_size=1073741824 innodb_buffer_pool_instances=1
run_ramp "DML rw50 iter1 (12GB, inst=1)" \
    innodb_buffer_pool_size=12884901888 innodb_buffer_pool_instances=1 innodb_write_io_threads=3
echo "[DONE]"
