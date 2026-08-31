# OtterTune Crash Investigation

## Problem

Running OtterTune (`config_sysbench_ot_rw50.ini`) sometimes gets stuck during MySQL restart at iteration 2, causing the experiment to hang or fail.

## Bugs Found and Fixed

### 1. Force kill commands were broken (FIXED)
`force_kill_cmd1` grepped for `mysql.sock` which never appears in `ps aux`. `force_kill_cmd2` had a race condition with grep matching itself. Result: force kill never actually killed mysqld.

**Fix:** Replaced with `pgrep -x mysqld` + `kill -9` per PID.

### 2. Stale socket lock file (FIXED)
After `kill -9`, `mysql.sock.lock` remains. New mysqld sees `Another process is using unix socket file` and aborts. `_start_mysqld` loops for 600s waiting for a connection that never comes.

**Fix:** Remove `self.sock` and `self.sock + '.lock'` after force kill.

### 3. Port conflict after kill -9 (FIXED in test script)
After `kill -9`, port 3306 may still be in TIME_WAIT. New mysqld fails with `Bind on TCP/IP port: Address already in use`.

**Fix in test script:** Wait for port to be released (`ss -tlnp | grep :3306`). Not yet added to `mysqldb.py`.

### 4. matplotlib crash on headless server (FIXED)
`tuner.py` crashed with exit code 1 when saving convergence plot.

**Fix:** `matplotlib.use('Agg')` before importing pyplot.

### 5. History loading takes 380s (FIXED)
Loading 254 source JSON files into HistoryContainers.

**Fix:** Pickle cache in `tuner.py:load_history()`. Subsequent loads take ~7s.

## Remaining Issue: Unexplained 65s Startup

### Observation
In the real DBTune run, iteration 2's `_start_mysqld` takes ~65s. Isolated tests with the same config take ~30s (immediate kill) or ~3s (after 60s partial graceful shutdown).

### What We Tested

| Test | Shutdown | Startup | Notes |
|------|----------|---------|-------|
| Clean start, any log size | N/A | **3s** | No crash recovery |
| Immediate kill -9, any knobs | instant | **25-33s** | Crash recovery on 48MB redo |
| 60s graceful + kill -9, any knobs | 60s | **2-4s** | Graceful flushed most data |
| Real DBTune run | 60s (timeout) | **65s** | Unknown why slower |
| `mysqladmin shutdown` after benchmark (isolated) | **26-30s** | N/A | Under TIMEOUT_CLOSE |

### What's Different in the Real Run
The real run goes through this path that the test doesn't:
1. `get_internal_metrics` — multiprocessing `mp.Process` that opens DB connections every 5s during benchmark
2. `get_states` → `ResourceMonitor` — monitors CPU/IO during benchmark  
3. `clear_cmd` — kills processlist after benchmark
4. Prometheus metrics collection — HTTP queries to localhost:9090
5. `step()` → `apply_knobs_offline()` → `_kill_mysqld()` — the actual shutdown

Between benchmark end and shutdown, there's ~4 seconds of metrics collection. During this time MySQL may start background operations (checkpoint, purge) that accumulate dirty state.

### How to Investigate Further
Set breakpoints in the debugger at:
- `mysqldb.py:177` — just before `Popen(kill_cmd)` — check `SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_pages_dirty'` to see dirty page count
- `mysqldb.py:180` — after `communicate()` returns — check if it timed out or succeeded
- `mysqldb.py:228` — in the connection loop — run `tail mysql.err` in another terminal to see InnoDB status

Or add instrumentation:
```python
# Before _kill_mysqld, log dirty page count
db_conn = MysqlConnector(**self.connection_info)
r = db_conn.fetch_results("SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_pages_dirty'")
logger.info("Dirty pages before shutdown: %s", r)
```

## Test Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_innodb_init_time.sh` | Tests clean start time and crash recovery time with various knob configs |
| `scripts/test_restart_slowdown.sh` | Tests graceful vs force kill with full benchmark (25GB buf + 5GB redo) |

## Config Changes Made

| File | Change |
|------|--------|
| `autotune/database/mysqldb.py` | Force kill via `pgrep -x mysqld`, socket/lock cleanup, timing logs, dead process detection |
| `autotune/dbenv.py` | `sysbench_tables` and `sysbench_table_size` configurable (was hardcoded 150/800000) |
| `autotune/tuner.py` | Pickle cache for history loading, matplotlib Agg backend |
| `.vscode/launch.json` | Debug config "OtterTune sysbench rw50 (crash repro)" |

## Database State
- Dropped all databases except `sbtest` (was 63GB total, now 603MB)
- Purged all binlogs (was 31GB)
- sbtest: 64 tables x 100K rows (was 150 tables x 800K rows)
- Config defaults in `dbenv.py`: `sysbench_tables=64`, `sysbench_table_size=100000`
