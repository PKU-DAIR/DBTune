#!/usr/bin/env bash
#
# Test: How long does InnoDB initialization take for different redo log sizes?
# Clean start (no crash recovery) — measures pure initialization time.
#
set -euo pipefail

MYSQL_BASE="/work/dpl-sfc/users/tatsu/tmp/DBTune/mysql_build"
MYSQLD="$MYSQL_BASE/bin/mysqld"
MYSQLADMIN="$MYSQL_BASE/bin/mysqladmin"
MYSQL_CLI="$MYSQL_BASE/bin/mysql"
CNF="$MYSQL_BASE/cnf/my.cnf"
CNF_CLEAN="$MYSQL_BASE/cnf/my.cnf.clean"
CLIENT_CNF="$MYSQL_BASE/cnf/mysql_client.cnf"
SOCK="$MYSQL_BASE/mysql.sock"
ERRLOG="$MYSQL_BASE/mysql.err"
TIMEOUT=1800

wait_for_mysql() {
    local elapsed=0
    while ! "$MYSQLADMIN" --socket="$SOCK" -u root ping &>/dev/null; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ "$elapsed" -ge "$TIMEOUT" ]; then
            echo "ERROR: MySQL did not start within ${TIMEOUT}s"
            tail -5 "$ERRLOG"
            exit 1
        fi
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "    ...waiting (${elapsed}s)"
        fi
    done
}

graceful_shutdown() {
    if "$MYSQLADMIN" --socket="$SOCK" -u root ping &>/dev/null; then
        "$MYSQLADMIN" --socket="$SOCK" -u root shutdown
        while pgrep -x mysqld &>/dev/null; do sleep 0.5; done
    fi
}

kill_all_mysqld() {
    for pid in $(pgrep -x mysqld 2>/dev/null || true); do
        kill -9 "$pid" 2>/dev/null && echo "    killed mysqld pid=$pid" || true
    done
    # Wait for processes to fully exit and port to be released
    while pgrep -x mysqld &>/dev/null; do sleep 0.5; done
    while ss -tlnp | grep -q ':3306 ' 2>/dev/null; do sleep 1; done
    # Clean up stale socket files
    rm -f "$SOCK" "$SOCK.lock"
}

write_cnf() {
    local log_file_size=$1
    local buf_pool_size=${2:-}
    cp "$CNF_CLEAN" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_log_file_size\t\t= $log_file_size" "$CNF"
    if [ -n "${buf_pool_size}" ]; then
        sed -i "/^\[mysqld\]/a innodb_buffer_pool_size\t\t= $buf_pool_size" "$CNF"
    fi
}

write_cnf_all_knobs() {
    # Write all 12 DML knobs to cnf, matching what DBTune does
    local knobs="$1"  # space-separated key=value pairs
    cp "$CNF_CLEAN" "$CNF"
    for kv in $knobs; do
        local key="${kv%%=*}"
        local val="${kv#*=}"
        sed -i "/^\[mysqld\]/a ${key}\t\t= ${val}" "$CNF"
    done
}

test_clean_start() {
    local label=$1
    local log_size=$2

    echo ""
    echo "--- $label (innodb_log_file_size = $log_size) ---"

    # Graceful shutdown first (clean state)
    graceful_shutdown
    kill_all_mysqld

    write_cnf "$log_size"

    local t_start=$(date +%s.%N)
    "$MYSQLD" --defaults-file="$CNF" &
    wait_for_mysql
    local t_end=$(date +%s.%N)

    local duration=$(python3 -c "print(f'{$t_end - $t_start:.1f}')")
    echo "    Clean start: ${duration}s"

    # Verify
    local actual=$("$MYSQL_CLI" --defaults-file="$CLIENT_CNF" -u root -N -e "SELECT @@innodb_log_file_size" 2>/dev/null)
    echo "    Verified innodb_log_file_size = $actual"
}

test_crash_recovery_start() {
    local label=$1
    local restart_knobs="$2"
    local benchmark_time=${3:-15}

    echo ""
    echo "--- $label ---"

    # Start with default config
    graceful_shutdown
    kill_all_mysqld
    write_cnf 50331648
    "$MYSQLD" --defaults-file="$CNF" &
    wait_for_mysql

    # Run benchmark with default config
    echo "    Running benchmark for ${benchmark_time}s (default config)..."
    ${SYSBENCH_BIN:-/usr/local/bin/sysbench} oltp_read_write \
        --mysql-host=localhost --mysql-socket="$SOCK" \
        --mysql-user=root --mysql-password="" \
        --mysql-db=sbtest --db-driver=mysql \
        --tables=64 --table-size=100000 \
        --threads=32 --time="$benchmark_time" \
        --warmup-time=5 --events=0 --rand-type=uniform \
        --db-ps-mode=disable \
        run 2>&1 | grep "transactions:"

    # Simulate DBTune's _kill_mysqld: start graceful shutdown, wait 60s, then kill -9
    echo "    Starting graceful shutdown (mysqladmin shutdown)..."
    "$MYSQLADMIN" --socket="$SOCK" -u root shutdown &
    echo "    Waiting 60s (partial flush in progress)..."
    sleep 60
    echo "    Force killing mysqld mid-flush..."
    kill_all_mysqld

    # Restart with all knobs applied
    write_cnf_all_knobs "$restart_knobs"
    echo "    Restarting with: $restart_knobs"
    local t_start=$(date +%s.%N)
    "$MYSQLD" --defaults-file="$CNF" &
    wait_for_mysql
    local t_end=$(date +%s.%N)

    local duration=$(python3 -c "print(f'{$t_end - $t_start:.1f}')")
    echo "    Startup time: ${duration}s"
}

echo "================================================================"
echo " InnoDB initialization time vs redo log size"
echo "================================================================"

# --- Part 1: Clean starts (no crash, just different log sizes) ---
echo ""
echo "=== PART 1: Clean start (graceful shutdown -> start with new log size) ==="

test_clean_start "48MB (default)"  50331648
test_clean_start "512MB"           536870912
test_clean_start "1GB"             1073741824
test_clean_start "3GB"             3221225472
test_clean_start "5GB"             5368709120

# --- Part 2: Crash recovery (force kill -> start with same log size) ---
echo ""
echo "=== PART 2: Benchmark with default, kill -9, restart with all knobs ==="

# Default knobs only (baseline)
test_crash_recovery_start "Default knobs" \
    "innodb_log_file_size=50331648"

# Actual knobs from a real OtterTune iteration 2 that took 103s
test_crash_recovery_start "OtterTune iter2 (11GB buf, 4.5GB redo)" \
    "innodb_adaptive_hash_index=1 innodb_buffer_pool_instances=6 innodb_buffer_pool_size=11397534467 innodb_change_buffer_max_size=13 innodb_flush_log_at_trx_commit=2 innodb_io_capacity=10742 innodb_log_file_size=4592021818 innodb_lru_scan_depth=8762 innodb_read_io_threads=23 innodb_write_io_threads=3 sync_binlog=1 table_open_cache=2470"

# Another real config (16GB buf, 2.4GB redo)
test_crash_recovery_start "OtterTune iter2 (16GB buf, 2.4GB redo)" \
    "innodb_adaptive_hash_index=0 innodb_buffer_pool_instances=5 innodb_buffer_pool_size=15984041432 innodb_change_buffer_max_size=37 innodb_flush_log_at_trx_commit=0 innodb_io_capacity=5407 innodb_log_file_size=2412460595 innodb_lru_scan_depth=4945 innodb_read_io_threads=15 innodb_write_io_threads=17 sync_binlog=1 table_open_cache=1490"

# Small config (1GB buf, 48MB redo, non-default other knobs)
test_crash_recovery_start "Small config (1GB buf, 48MB redo)" \
    "innodb_adaptive_hash_index=0 innodb_buffer_pool_instances=4 innodb_buffer_pool_size=1073741824 innodb_change_buffer_max_size=10 innodb_flush_log_at_trx_commit=0 innodb_io_capacity=5000 innodb_log_file_size=50331648 innodb_lru_scan_depth=100 innodb_read_io_threads=4 innodb_write_io_threads=4 sync_binlog=0 table_open_cache=2000"

# --- Cleanup ---
echo ""
graceful_shutdown
kill_all_mysqld
cp "$CNF_CLEAN" "$CNF"
echo "Restored clean config. Done."
