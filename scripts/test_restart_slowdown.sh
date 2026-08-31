#!/usr/bin/env bash
#
# Test: MySQL restart time — matches real DBTune _kill_mysqld behavior
# Tests 3 scenarios after a 15s benchmark with default config (48MB redo, 1GB buf):
#   A) Immediate kill -9
#   B) 60s partial graceful shutdown + kill -9 (what DBTune does when TIMEOUT_CLOSE=60)
#   C) Full graceful shutdown (baseline)
#
# Restart config uses real OtterTune iteration 2 knobs.
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

SYSBENCH_BIN="${SYSBENCH_BIN:-/usr/local/bin/sysbench}"

wait_for_mysql() {
    local elapsed=0
    while ! "$MYSQLADMIN" --socket="$SOCK" -u root ping &>/dev/null; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ "$elapsed" -ge "$TIMEOUT" ]; then
            echo "    ERROR: MySQL did not start within ${TIMEOUT}s"
            tail -5 "$ERRLOG"
            exit 1
        fi
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "    ...waiting (${elapsed}s)"
        fi
    done
    echo "    connected after ${elapsed}s"
}

kill_all_mysqld() {
    for pid in $(pgrep -x mysqld 2>/dev/null || true); do
        kill -9 "$pid" 2>/dev/null && echo "    killed mysqld pid=$pid" || true
    done
    while pgrep -x mysqld &>/dev/null; do sleep 0.5; done
    while ss -tlnp | grep -q ':3306 ' 2>/dev/null; do sleep 1; done
    rm -f "$SOCK" "$SOCK.lock"
}

write_cnf_default() {
    # Match DBTune iteration 1: clean cnf + default knobs from DML_12.json
    cp "$CNF_CLEAN" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_adaptive_hash_index\t\t= 1" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_buffer_pool_instances\t\t= 1" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_buffer_pool_size\t\t= 1073741824" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_change_buffer_max_size\t\t= 25" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_flush_log_at_trx_commit\t\t= 1" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_io_capacity\t\t= 100" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_log_file_size\t\t= 50331648" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_lru_scan_depth\t\t= 1024" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_read_io_threads\t\t= 2" "$CNF"
    sed -i "/^\[mysqld\]/a innodb_write_io_threads\t\t= 2" "$CNF"
    sed -i "/^\[mysqld\]/a sync_binlog\t\t= 1" "$CNF"
    sed -i "/^\[mysqld\]/a table_open_cache\t\t= 4000" "$CNF"
}

write_cnf_all_knobs() {
    # Write all 12 knobs to cnf
    local knobs="$1"
    cp "$CNF_CLEAN" "$CNF"
    for kv in $knobs; do
        local key="${kv%%=*}"
        local val="${kv#*=}"
        sed -i "/^\[mysqld\]/a ${key}\t\t= ${val}" "$CNF"
    done
}

run_benchmark() {
    echo "  Running sysbench oltp_read_write (5s warmup + 15s run, 32 threads)..."
    "$SYSBENCH_BIN" oltp_read_write \
        --mysql-host=localhost \
        --mysql-socket="$SOCK" \
        --mysql-user=root \
        --mysql-password="" \
        --mysql-db=sbtest \
        --db-driver=mysql \
        --mysql-storage-engine=innodb \
        --range-size=100 \
        --events=0 \
        --rand-type=uniform \
        --tables=64 \
        --table-size=100000 \
        --db-ps-mode=disable \
        --warmup-time=5 \
        --threads=32 \
        --time=15 \
        run 2>&1 | grep "transactions:"
}

log_dirty_pages() {
    local dirty=$("$MYSQL_CLI" --defaults-file="$CLIENT_CNF" -u root -N -e \
        "SELECT variable_value FROM performance_schema.global_status WHERE variable_name='Innodb_buffer_pool_pages_dirty'" 2>/dev/null || echo "N/A")
    echo "  Dirty pages: $dirty"
}

# Real OtterTune iteration 2 knobs (from a run that took 65s)
RESTART_KNOBS="innodb_adaptive_hash_index=0 innodb_buffer_pool_instances=4 innodb_buffer_pool_size=8974511804 innodb_change_buffer_max_size=9 innodb_flush_log_at_trx_commit=1 innodb_io_capacity=2665 innodb_log_file_size=1299599738 innodb_lru_scan_depth=2213 innodb_read_io_threads=11 innodb_write_io_threads=9 sync_binlog=0 table_open_cache=490"

echo "================================================================"
echo " MySQL restart time test (matching DBTune behavior)"
echo "================================================================"
echo ""
echo "Default config: 48MB redo, 1GB buffer pool"
echo "Restart config: $RESTART_KNOBS"
echo ""

# --- Setup function: start default MySQL, run benchmark ---
setup_and_benchmark() {
    kill_all_mysqld
    write_cnf_default
    "$MYSQLD" --defaults-file="$CNF" &
    wait_for_mysql
    run_benchmark
    log_dirty_pages
}

# =============================================
# TEST A: Immediate kill -9 after benchmark
# =============================================
echo "============================================"
echo " TEST A: Immediate kill -9 after benchmark"
echo "============================================"

setup_and_benchmark

echo "  Force killing immediately..."
T_KILL=$(date +%s.%N)
kill_all_mysqld
T_KILLED=$(date +%s.%N)

echo "  Restarting with OtterTune knobs..."
write_cnf_all_knobs "$RESTART_KNOBS"
T_START=$(date +%s.%N)
"$MYSQLD" --defaults-file="$CNF" &
wait_for_mysql
T_READY=$(date +%s.%N)

A_KILL=$(python3 -c "print(f'{$T_KILLED - $T_KILL:.1f}')")
A_STARTUP=$(python3 -c "print(f'{$T_READY - $T_START:.1f}')")
A_TOTAL=$(python3 -c "print(f'{$T_READY - $T_KILL:.1f}')")
echo ""
echo "  [TEST A] kill=${A_KILL}s  startup=${A_STARTUP}s  total=${A_TOTAL}s"

# Clean shutdown before next test
"$MYSQLADMIN" --socket="$SOCK" -u root shutdown
while pgrep -x mysqld &>/dev/null; do sleep 0.5; done

# =============================================
# TEST B: 60s partial graceful + kill -9
#         (what DBTune _kill_mysqld does)
# =============================================
echo ""
echo "============================================"
echo " TEST B: 60s partial graceful + kill -9"
echo "         (simulates DBTune TIMEOUT_CLOSE=60)"
echo "============================================"

setup_and_benchmark

echo "  Starting mysqladmin shutdown..."
T_SHUTDOWN=$(date +%s.%N)
"$MYSQLADMIN" --socket="$SOCK" -u root shutdown &
ADMIN_PID=$!
echo "  Waiting 60s for partial flush..."
sleep 60
echo "  Force killing after 60s..."
T_KILL=$(date +%s.%N)
kill_all_mysqld
# Also kill mysqladmin if still running
kill $ADMIN_PID 2>/dev/null || true
T_KILLED=$(date +%s.%N)

echo "  Restarting with OtterTune knobs..."
write_cnf_all_knobs "$RESTART_KNOBS"
T_START=$(date +%s.%N)
"$MYSQLD" --defaults-file="$CNF" &
wait_for_mysql
T_READY=$(date +%s.%N)

B_GRACEFUL=$(python3 -c "print(f'{$T_KILL - $T_SHUTDOWN:.1f}')")
B_KILL=$(python3 -c "print(f'{$T_KILLED - $T_KILL:.1f}')")
B_STARTUP=$(python3 -c "print(f'{$T_READY - $T_START:.1f}')")
B_TOTAL=$(python3 -c "print(f'{$T_READY - $T_SHUTDOWN:.1f}')")
echo ""
echo "  [TEST B] graceful_wait=${B_GRACEFUL}s  kill=${B_KILL}s  startup=${B_STARTUP}s  total=${B_TOTAL}s"

# Clean shutdown before next test
"$MYSQLADMIN" --socket="$SOCK" -u root shutdown
while pgrep -x mysqld &>/dev/null; do sleep 0.5; done

# =============================================
# TEST C: Full graceful shutdown (baseline)
# =============================================
echo ""
echo "============================================"
echo " TEST C: Full graceful shutdown (baseline)"
echo "============================================"

setup_and_benchmark

echo "  Graceful shutdown..."
T_SHUTDOWN=$(date +%s.%N)
"$MYSQLADMIN" --socket="$SOCK" -u root shutdown
while pgrep -x mysqld &>/dev/null; do sleep 0.5; done
T_DOWN=$(date +%s.%N)

echo "  Restarting with OtterTune knobs..."
write_cnf_all_knobs "$RESTART_KNOBS"
T_START=$(date +%s.%N)
"$MYSQLD" --defaults-file="$CNF" &
wait_for_mysql
T_READY=$(date +%s.%N)

C_SHUTDOWN=$(python3 -c "print(f'{$T_DOWN - $T_SHUTDOWN:.1f}')")
C_STARTUP=$(python3 -c "print(f'{$T_READY - $T_START:.1f}')")
C_TOTAL=$(python3 -c "print(f'{$T_READY - $T_SHUTDOWN:.1f}')")
echo ""
echo "  [TEST C] shutdown=${C_SHUTDOWN}s  startup=${C_STARTUP}s  total=${C_TOTAL}s"

# =============================================
# SUMMARY
# =============================================
echo ""
echo "============================================"
echo " SUMMARY"
echo "============================================"
echo ""
echo "  A) Immediate kill -9:           kill=${A_KILL}s + startup=${A_STARTUP}s = ${A_TOTAL}s"
echo "  B) 60s graceful + kill -9:      wait=${B_GRACEFUL}s + kill=${B_KILL}s + startup=${B_STARTUP}s = ${B_TOTAL}s"
echo "  C) Full graceful shutdown:      shutdown=${C_SHUTDOWN}s + startup=${C_STARTUP}s = ${C_TOTAL}s"
echo ""

# --- Restore ---
kill_all_mysqld
cp "$CNF_CLEAN" "$CNF"
"$MYSQLD" --defaults-file="$CNF" &
wait_for_mysql
echo "Restored clean config. MySQL running."
