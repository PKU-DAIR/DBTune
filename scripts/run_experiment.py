#!/usr/bin/env python3
"""
Automated DBTune experiment runner.

Usage:
    python run_experiment.py                          # run all benchmarks x all methods
    python run_experiment.py --benchmark readwrite    # only readwrite
    python run_experiment.py --method dml             # only DML
    python run_experiment.py --benchmark readwrite --method dml --iterations 3

Prerequisites:
    export SYSBENCH_BIN=/usr/local/bin/sysbench
    export MYSQL_SOCK=../mysql_build/mysql.sock
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
MYSQL_BASE = SCRIPT_DIR / ".." / "mysql_build"
MYSQLD = MYSQL_BASE / "bin" / "mysqld"
MYSQLADMIN = MYSQL_BASE / "bin" / "mysqladmin"
MYSQL_CLI = MYSQL_BASE / "bin" / "mysql"
CNF = MYSQL_BASE / "cnf" / "my.cnf"
CNF_CLEAN = MYSQL_BASE / "cnf" / "my.cnf.clean"
CLIENT_CNF = MYSQL_BASE / "cnf" / "mysql_client.cnf"
SOCK = MYSQL_BASE / "mysql.sock"
PYTHON = SCRIPT_DIR / ".." / "venv" / "bin" / "python"
HISTORY_DIR = SCRIPT_DIR / "DBTune_history"
TRASH_DIR = HISTORY_DIR / "trash"

SYSBENCH_BIN = os.environ.get("SYSBENCH_BIN", "/usr/local/bin/sysbench")
OLTPBENCH_HOME = SCRIPT_DIR / ".." / "third_party" / "oltpbench"

# --- Benchmark configs ---
SYSBENCH_TABLES = 64
SYSBENCH_TABLE_SIZE = 100000
SYSBENCH_THREADS = 32

# method -> benchmark -> config file
CONFIGS = {
    "dml": {
        "readwrite": "config_sysbench_dml_rw50.ini",
        "read": "config_sysbench_dml_read.ini",
        "write": "config_sysbench_dml_write.ini",
        "tpcc": "config_dml_test.ini",
    },
    "ottertune": {
        "readwrite": "config_sysbench_ot_rw50.ini",
        "read": "config_sysbench_ot_read.ini",
        "write": "config_sysbench_ot_write.ini",
        "tpcc": "config_ottertune_tpcc.ini",
    },
}

# Benchmark type: sysbench or oltpbench
BENCHMARK_TYPE = {
    "readwrite": "sysbench",
    "read": "sysbench",
    "write": "sysbench",
    "tpcc": "oltpbench",
}

BENCHMARKS = ["readwrite", "read", "write", "tpcc"]
METHODS = ["dml", "ottertune"]


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def mysql_cmd(sql):
    """Run a SQL command and return stdout."""
    result = subprocess.run(
        [str(MYSQL_CLI), f"--defaults-file={CLIENT_CNF}", "-u", "root", "-N", "-e", sql],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"MySQL error: {result.stderr.strip()}")
    return result.stdout.strip()


def mysql_ping():
    """Check if MySQL is reachable."""
    return subprocess.run(
        [str(MYSQLADMIN), f"--socket={SOCK}", "-u", "root", "ping"],
        capture_output=True, text=True,
    ).returncode == 0


def ensure_mysql_running():
    if mysql_ping():
        log("MySQL already running")
        return

    log("Starting MySQL...")
    import shutil
    shutil.copy(str(CNF_CLEAN), str(CNF))
    subprocess.Popen([str(MYSQLD), f"--defaults-file={CNF}"])

    for i in range(300):
        time.sleep(1)
        if mysql_ping():
            log(f"MySQL ready ({i+1}s)")
            return
    raise RuntimeError("MySQL did not start within 300s")


def assert_clean_state():
    """Drop all user databases and purge binlogs."""
    log("Asserting clean database state...")
    ensure_mysql_running()

    # Find and drop user databases
    dbs = mysql_cmd(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name NOT IN ('information_schema','performance_schema','sys','mysql')"
    )
    for db in dbs.splitlines():
        db = db.strip()
        if db:
            mysql_cmd(f"DROP DATABASE IF EXISTS `{db}`")
            log(f"  Dropped {db}")

    # Purge binlogs
    try:
        mysql_cmd("RESET MASTER")
    except RuntimeError:
        pass
    log("  Binlogs purged")
    log("Clean state verified")


def load_sysbench():
    """Load sysbench tables."""
    log(f"Loading sysbench: {SYSBENCH_TABLES} tables x {SYSBENCH_TABLE_SIZE} rows...")
    mysql_cmd("DROP DATABASE IF EXISTS sbtest; CREATE DATABASE sbtest")

    result = subprocess.run(
        [
            SYSBENCH_BIN, "oltp_common",
            "--mysql-host=localhost", f"--mysql-socket={SOCK}",
            "--mysql-user=root", "--mysql-password=",
            "--mysql-db=sbtest", "--db-driver=mysql",
            f"--tables={SYSBENCH_TABLES}",
            f"--table-size={SYSBENCH_TABLE_SIZE}",
            f"--threads={SYSBENCH_THREADS}",
            "prepare",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sysbench prepare failed: {result.stderr}")

    size_mb = mysql_cmd(
        "SELECT ROUND(SUM(data_length+index_length)/1024/1024) "
        "FROM information_schema.tables WHERE table_schema='sbtest'"
    )
    log(f"  Loaded: {size_mb}MB")


def load_oltpbench(benchmark):
    """Load OLTPBench data (TPC-C, etc.)."""
    # Map benchmark name to oltpbench benchmark name and config
    oltpbench_map = {
        "tpcc": {
            "name": "tpcc",
            "db": "tpcc",
            "config": OLTPBENCH_HOME / "config" / "tpcc_config_mysql.xml",
        },
    }
    info = oltpbench_map[benchmark]
    db = info["db"]
    config_xml = info["config"]

    log(f"Loading OLTPBench {benchmark} into database '{db}'...")
    mysql_cmd(f"DROP DATABASE IF EXISTS `{db}`; CREATE DATABASE `{db}`")

    result = subprocess.run(
        [
            str(OLTPBENCH_HOME / "oltpbenchmark"),
            "-b", info["name"],
            "-c", str(config_xml),
            "--create=true", "--load=true",
            "-s", "5",
        ],
        capture_output=True, text=True,
        cwd=str(OLTPBENCH_HOME),
    )
    if result.returncode != 0:
        raise RuntimeError(f"OLTPBench load failed: {result.stderr[-500:]}")

    size_mb = mysql_cmd(
        f"SELECT ROUND(SUM(data_length+index_length)/1024/1024) "
        f"FROM information_schema.tables WHERE table_schema='{db}'"
    )
    log(f"  Loaded: {size_mb}MB")


def load_benchmark(benchmark):
    """Load the appropriate benchmark data."""
    btype = BENCHMARK_TYPE[benchmark]
    if btype == "sysbench":
        load_sysbench()
    elif btype == "oltpbench":
        load_oltpbench(benchmark)


def drop_benchmark_db(benchmark):
    """Drop the benchmark database and purge binlogs."""
    db_map = {
        "readwrite": "sbtest",
        "read": "sbtest",
        "write": "sbtest",
        "tpcc": "tpcc",
    }
    db = db_map[benchmark]
    log(f"Dropping {db}...")
    try:
        mysql_cmd(f"DROP DATABASE IF EXISTS `{db}`")
        mysql_cmd("RESET MASTER")
    except RuntimeError:
        pass
    log("  Done")


def get_task_id(config_file):
    """Extract task_id from config file."""
    with open(SCRIPT_DIR / config_file) as f:
        for line in f:
            if line.strip().startswith("task_id"):
                return line.split("=")[1].strip()
    return None


def set_config_value(config_file, key, value):
    """Set a value in an ini config file."""
    config_path = SCRIPT_DIR / config_file
    lines = config_path.read_text().splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(key) and "=" in line:
            lines[i] = f"{key} = {value}"
            found = True
            break
    if not found:
        raise ValueError(f"Key '{key}' not found in {config_file}")
    config_path.write_text("\n".join(lines) + "\n")


def run_method(method, config_file, benchmark, iterations=None):
    """Run a single DBTune experiment."""
    task_id = get_task_id(config_file)

    # Override max_runs if iterations specified
    if iterations is not None:
        set_config_value(config_file, "max_runs", iterations)
        log(f"Set max_runs = {iterations} in {config_file}")

    # Move old history to trash
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    history_file = HISTORY_DIR / f"history_{task_id}.json"
    if history_file.exists():
        history_file.rename(TRASH_DIR / history_file.name)

    log(f"Running {method} on {benchmark} (config={config_file}, task={task_id}, iterations={iterations})")

    # Run DBTune — output goes to stdout (captured by tee)
    cmd = [str(PYTHON), "optimize.py", f"--config={config_file}"]

    t_start = time.time()
    exit_code = subprocess.call(
        cmd,
        cwd=str(SCRIPT_DIR),
        env={**os.environ, "SYSBENCH_BIN": SYSBENCH_BIN, "MYSQL_SOCK": str(SOCK)},
    )
    duration = int(time.time() - t_start)

    status = "OK" if exit_code == 0 else f"FAILED (exit {exit_code})"
    log(f"{method} on {benchmark}: {status}, {duration}s")

    return exit_code, duration


def main():
    parser = argparse.ArgumentParser(description="Run DBTune experiments")
    parser.add_argument("--benchmark", choices=BENCHMARKS + ["all"], default="all",
                        help="Which benchmark to test (sysbench: readwrite/read/write, oltpbench: tpcc)")
    parser.add_argument("--method", choices=METHODS + ["all"], default="all",
                        help="Which optimizer method to test")
    parser.add_argument("--iterations", type=int, default=2,
                        help="Override max_runs for each experiment run")
    args = parser.parse_args()

    benchmarks = BENCHMARKS if args.benchmark == "all" else [args.benchmark]
    methods = METHODS if args.method == "all" else [args.method]

    log("=" * 60)
    log("DBTune Experiment Runner")
    log("=" * 60)
    log(f"Benchmarks: {benchmarks}")
    log(f"Methods: {methods}")
    log(f"Tables: {SYSBENCH_TABLES} x {SYSBENCH_TABLE_SIZE}")
    log("")

    # Assert clean starting state
    assert_clean_state()

    results = []

    for benchmark in benchmarks:
        log("=" * 60)
        log(f"Benchmark: sysbench {benchmark}")
        log("=" * 60)

        for method in methods:
            config_file = CONFIGS.get(method, {}).get(benchmark)
            if not config_file or not (SCRIPT_DIR / config_file).exists():
                log(f"WARNING: No config for {method}/{benchmark}, skipping")
                continue

            # Fresh data for each method
            ensure_mysql_running()
            drop_benchmark_db(benchmark)
            t_load_start = time.time()
            load_benchmark(benchmark)
            t_load_end = time.time()
            load_duration = int(t_load_end - t_load_start)
            log(f"Benchmark load took {load_duration}s")

            ensure_mysql_running()
            exit_code, tune_duration = run_method(method, config_file, benchmark, iterations=args.iterations)
            total_duration = int(time.time() - t_load_start)
            results.append({
                "benchmark": benchmark,
                "method": method,
                "load_duration": load_duration,
                "tune_duration": tune_duration,
                "total_duration": total_duration,
                "exit_code": exit_code,
            })

        # Drop database after all methods tested on this benchmark
        ensure_mysql_running()
        drop_benchmark_db(benchmark)

    # Final summary
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    print()
    print(f"  {'Benchmark':<15} {'Method':<12} {'Load':>8} {'Tune':>8} {'Total':>8} {'Status':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        status = "OK" if r["exit_code"] == 0 else f"ERR({r['exit_code']})"
        print(f"  {r['benchmark']:<15} {r['method']:<12} {r['load_duration']:>6}s {r['tune_duration']:>6}s {r['total_duration']:>6}s {status:>8}")
    print()

    log("History files:")
    for f in sorted(HISTORY_DIR.glob("history_*.json")):
        if "trash" not in str(f):
            log(f"  {f}")


if __name__ == "__main__":
    main()
