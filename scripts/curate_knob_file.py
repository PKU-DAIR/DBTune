#!/usr/bin/env python3
"""Generate mysql_perf_8.0.json: the performance-relevant subset of mysql_all_8.0.json.

CONSERVATIVE curation: a knob is dropped only when the reason is independent of the
workload — i.e. it would never be tuned for performance under ANY workload in this repo
(sysbench, oltpbench/TPCC/twitter, JOB, TPC-H). Categories:
  (A) breaks the server or the benchmark connection,
  (B) feature absent from every repo setup: InnoDB memcached plugin (not loaded),
      replication (no replicas anywhere), INSERT DELAYED (removed in MySQL 8.0),
  (C) pure observability/diagnostics,
  (D) changes query results / transaction semantics / security posture, not performance,
  (E) capacity/availability limits sized to the workload, never tuned for throughput
      (outage below the workload's need, inert above it).
Workload-specific no-ops (full-text, MyISAM-only, DDL-time-only, GIS, stored programs,
optimizer-stats knobs, ...) are KEPT: they waste a little sweep budget on sysbench but
stay meaningful for other repo workloads.

Run:  python curate_knob_file.py
Full drop list + reasons: claude_memory/KNOB_CURATION.md
"""
import json
import os

SRC = os.path.join(os.path.dirname(__file__), 'experiment/gen_knobs/mysql_all_8.0.json')
DST = os.path.join(os.path.dirname(__file__), 'experiment/gen_knobs/mysql_perf_8.0.json')

# A: server/benchmark breakers (workload-independent)
DROP_BREAKER = {
    'lower_case_table_names': 'init-time-only in 8.0; mysqld refuses to start on datadir mismatch',
    'skip_networking': 'kills the TCP benchmark connection',
    'offline_mode': 'refuses client connections',
    'require_secure_transport': 'kills non-TLS connections',
    'max_join_size': 'low values abort SELECTs with ER_TOO_BIG_SELECT (default = max, '
                     'cannot clamp without excluding the default)',
    'max_user_connections': 'values < client threads refuse connections; default 0 (=unlimited) '
                            'is below any safe clamp, so the range cannot be made safe',
}

# B: features absent from every repo setup
DROP_MEMCACHED = dict.fromkeys([
    'innodb_api_bk_commit_interval', 'innodb_api_disable_rowlock',
    'innodb_api_enable_binlog', 'innodb_api_enable_mdl',
], 'InnoDB memcached plugin is not loaded in any repo setup')
DROP_REPLICATION = dict.fromkeys([
    'log_slave_updates', 'master_verify_checksum', 'binlog_direct_non_transactional_updates',
    'binlog_error_action', 'binlog_rows_query_log_events', 'log_bin_use_v1_row_events',
    'log_statements_unsafe_for_binlog', 'innodb_replication_delay', 'expire_logs_days',
], 'replica-side / replication-maintenance semantics; no replicas in any repo setup '
   '(binlog performance knobs like sync_binlog/group-commit/cache sizes are kept)')
DROP_REMOVED_FEATURE = {
    'max_delayed_threads': 'INSERT DELAYED was removed in MySQL 8.0; the knob is inert',
}
DROP_MYISAM = dict.fromkeys([
    'key_buffer_size', 'key_cache_age_threshold', 'key_cache_block_size',
    'key_cache_division_limit', 'bulk_insert_buffer_size', 'concurrent_insert',
    'delay_key_write', 'preload_buffer_size', 'low_priority_updates',
    'max_write_lock_count',
], 'MyISAM/table-lock-engine only; every repo workload is InnoDB, and in 8.0 internal '
   'disk temp tables and system tables are InnoDB too (engine-agnostic flush/flush_time '
   'are kept)')

# E: capacity/availability limits — a DBA SIZES these to the workload, never tunes them
# for throughput: below the workload's need they refuse work (outage, no perf signal),
# above it they are inert (no resources are reserved for unused headroom)
DROP_CAPACITY = dict.fromkeys([
    'max_connections', 'back_log', 'open_files_limit', 'max_allowed_packet',
    'max_prepared_stmt_count', 'connect_timeout', 'net_read_timeout',
    'net_write_timeout',
], 'capacity/availability limit: outage below the workload need, inert above it; '
   'no throughput trade-off anywhere in the range for a fixed workload')

# C: observability/diagnostics
DROP_OBSERVABILITY = dict.fromkeys([
    'general_log', 'slow_query_log', 'log_output', 'log_queries_not_using_indexes',
    'log_slow_admin_statements', 'log_timestamps', 'long_query_time',
    'innodb_print_all_deadlocks', 'session_track_gtids', 'session_track_schema',
    'session_track_state_change', 'session_track_transaction_info',
    'max_digest_length', 'max_error_count', 'end_markers_in_json',
], 'logging/diagnostics output, not a performance trade-off a DBA would tune for throughput')

# D: semantics/security (change results or behavior, not performance)
DROP_SEMANTICS = dict.fromkeys([
    'autocommit', 'automatic_sp_privileges', 'check_proxy_users',
    'mysql_native_password_proxy_users', 'completion_type',
    'explicit_defaults_for_timestamp', 'innodb_strict_mode', 'innodb_rollback_on_timeout',
    'local_infile', 'log_bin_trust_function_creators', 'skip_name_resolve',
    'updatable_views_with_limit', 'keep_files_on_create', 'max_sort_length',
    'default_week_format', 'div_precision_increment',
], 'changes query results / transaction semantics / security posture '
   '(autocommit=OFF breaks benchmark transaction handling; max_sort_length truncates '
   'comparisons; default_week_format/div_precision_increment change query results)')

DROPS = {}
for d in (DROP_BREAKER, DROP_MEMCACHED, DROP_REPLICATION, DROP_REMOVED_FEATURE,
          DROP_MYISAM, DROP_CAPACITY, DROP_OBSERVABILITY, DROP_SEMANTICS):
    DROPS.update(d)



def main():
    with open(SRC) as f:
        knobs = json.load(f)

    unknown = set(DROPS) - set(knobs)
    assert not unknown, 'drop list names not in source file: %s' % sorted(unknown)
    out = {}
    for name, spec in knobs.items():
        if name in DROPS:
            continue
        out[name] = spec

    with open(DST, 'w') as f:
        json.dump(out, f, indent=4)
        f.write('\n')

    n_int = sum(1 for v in out.values() if v['type'] == 'integer')
    n_enum = len(out) - n_int
    print('source: %d knobs -> curated: %d knobs (%d integer, %d enum; %d enum choices)'
          % (len(knobs), len(out), n_int, n_enum,
             sum(len(v['enum_values']) for v in out.values() if v['type'] == 'enum')))
    print('dropped %d' % len(DROPS))
    print('wrote %s' % DST)


if __name__ == '__main__':
    main()
