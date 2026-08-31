"""
Collect 114 Prometheus metrics matching the DBMSTransferLearning CSV format.

These metrics are collected via mysqld_exporter and node_exporter, matching
the exact PromQL expressions used to generate the training data CSVs.

The column order matches DBMSTransferLearning/dataset/full_data/*.csv files
(after post-processing with unify_metrics + rename_columns from data_preprocess.py).
"""
import logging
import numpy as np
import requests

logger = logging.getLogger(__name__)


# Ordered list of 114 metric names matching the CSV column order
PROMETHEUS_METRIC_NAMES = [
    'Average CPU Usage (Idle Mode)',
    'Average CPU Usage (IO Wait Mode)',
    'Average CPU Usage (IRQ Mode)',
    'Average CPU Usage (Nice Mode)',
    'Average CPU Usage (SoftIRQ Mode)',
    'Average CPU Usage (Steal Mode)',
    'Average CPU Usage (System Mode)',
    'Average CPU Usage (User Mode)',
    'Average Memory Usage Percentage',
    'Total Memory Usage',
    'Total Memory',
    'Average Memory Swap In',
    'Average Memory Swap Out',
    'Average Network Retransmit Rate',
    'Average Swap Usage Percentage',
    'Average Pages Swap In/Out (60s Rate)',
    'Average Pages Swap Out (60s Rate)',
    'InnoDB Buffer Pool Cache Hit Rate',
    'Current QPS (Queries Per Second)',
    'InnoDB Buffer Pool (Data Pages)',
    'InnoDB Buffer Pool (Free Pages)',
    'InnoDB Buffer Pool (Misc Pages)',
    'InnoDB Dirty Buffer Pages',
    'Max CPU Usage (100 - Idle)',
    'MySQL Aborted Clients (60s Rate)',
    'MySQL Aborted Connections (60s Rate)',
    'MySQL Threads Running (Max Over Time)',
    'MySQL Client Threads Connected (Max Over Time)',
    'MySQL Threads Running (Max Over Time).1',
    'MySQL Threads Connected (Max Over Time)',
    'MySQL Max Used Connections',
    'MySQL Max Connections',
    'MySQL File Openings (60s Rate)',
    'MySQL Handlers (Delete)',
    'MySQL Handlers (Discover)',
    'MySQL Handlers (External Lock)',
    'MySQL Handlers (MRR Init)',
    'MySQL Handlers (Read First Record)',
    'MySQL Handlers (Key Read)',
    'MySQL Handlers (Read Last Record)',
    'MySQL Handlers (Read Next Record)',
    'MySQL Handlers (Read Previous Record)',
    'MySQL Handlers (Read Random)',
    'MySQL Handlers (Read Next Record).1',
    'MySQL Handlers (Update)',
    'MySQL Handlers (Write)',
    'InnoDB Rows Deleted (60s Rate)',
    'InnoDB Rows Inserted (60s Rate)',
    'InnoDB Rows Read (60s Rate)',
    'InnoDB Rows Updated (60s Rate)',
    'InnoDB Buffer Pool Total Pages',
    'InnoDB Log Buffer Size',
    'MySQL Internal Memory (Key Buffer Size)',
    'MySQL Network Traffic (Bytes Received)',
    'MySQL Network Traffic (Bytes Sent)',
    'MySQL Open Files (InnoDB)',
    'MySQL Open Files',
    'MySQL Open Files Limit',
    'MySQL Open Tables',
    'MySQL Table Open Cache Size',
    'MySQL Full Join Selects (60s Rate)',
    'MySQL Full Range Join Selects (60s Rate)',
    'MySQL Range Selects (60s Rate)',
    'MySQL Range Selects (60s Rate).1',
    'MySQL Full Table Scans (60s Rate)',
    'MySQL Slow Queries (60s Rate)',
    'MySQL Sort Merge Passes (60s Rate)',
    'MySQL Sort Range (60s Rate)',
    'MySQL Rows Sorted (60s Rate)',
    'MySQL Sort Scans (60s Rate)',
    'MySQL Table Definition Cache',
    'MySQL Table Definition Cache Size',
    'MySQL Opened Table Definitions (60s Rate)',
    'MySQL Table Locks Immediate (60s Rate)',
    'MySQL Table Locks Waited (60s Rate)',
    'MySQL Opened Tables (60s Rate)',
    'MySQL Table Open Cache Hits (60s Rate)',
    'MySQL Table Open Cache Efficiency',
    'MySQL Open Cache Misses (60s Rate)',
    'MySQL Open Cache Overflows (60s Rate)',
    'MySQL Temporary Disk Tables Created (60s Rate)',
    'MySQL Temporary Files Created (60s Rate)',
    'MySQL Temporary Tables Created (60s Rate)',
    'MySQL Threads Cached',
    'MySQL Thread Cache Size',
    'MySQL Threads Created (60s Rate)',
    'MySQL Transaction Handlers (Commit)',
    'MySQL Transaction Handlers (Prepare)',
    'MySQL Transaction Handlers (Rollback)',
    'MySQL Transaction Handlers (Savepoint)',
    'MySQL Transaction Handlers (Savepoint Rollback)',
    'Top 5 Command Usage (Delete)',
    'Top 5 Command Usage (Insert)',
    'Top 5 Command Usage (Select)',
    'Top 5 Command Usage (Statement Execute)',
    'Top 5 Command Usage (Update)',
    'MySQL Uptime',
    'Root Volume Disk Usage Percentage',
    'Root Volume Disk Usage',
    'Top 5 Command Usage (Begin Transaction)',
    'Top 5 Command Usage (Commit)',
    'Top 5 Command Usage (Show Status)',
    'Top 5 Command Usage (Admin Commands)',
    'Top 5 Command Usage (Show Variables)',
    'Top 5 Command Usage (Set Option)',
    'Top 5 Command Usage (Show Slave Status)',
    'Top 5 Command Usage (Statement Close)',
    'Top 5 Command Usage (Show Replica Status)',
    'Top 5 Command Usage (Statement Prepare)',
    'Average Network Traffic Sent',
    'Average Network Traffic Received',
    'Average Disk IOPS (Read)',
    'Average Disk IOPS (Write)',
    'Average Disk Busy',
]

NUM_PROMETHEUS_METRICS = len(PROMETHEUS_METRIC_NAMES)  # 114


def _query_prometheus(prometheus_url, query):
    """Execute a PromQL instant query and return the scalar value."""
    try:
        resp = requests.get(f"{prometheus_url}/api/v1/query", params={'query': query}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data['status'] != 'success' or not data['data']['result']:
            return 0.0
        return float(data['data']['result'][0]['value'][1])
    except Exception as e:
        logger.warning("Prometheus query failed for '%s': %s", query[:80], e)
        return 0.0


def collect_all_prometheus_metrics(prometheus_url, mysql_instance=None, node_instance=None):
    """
    Collect all 114 metrics from Prometheus matching the CSV training data format.

    Returns:
        numpy array of shape (114,) with metric values in CSV column order
    """
    mf = f'{{instance="{mysql_instance}"}}' if mysql_instance else ''
    nf = f'{{instance="{node_instance}"}}' if node_instance else ''
    q = lambda query: _query_prometheus(prometheus_url, query)

    metrics = np.zeros(NUM_PROMETHEUS_METRICS)

    # --- CPU Metrics (8) ---
    cpu_modes = ['idle', 'iowait', 'irq', 'nice', 'softirq', 'steal', 'system', 'user']
    for i, mode in enumerate(cpu_modes):
        mode_filter = nf.rstrip('}') + f',mode="{mode}"}}' if nf else f'{{mode="{mode}"}}'
        metrics[i] = q(f'avg without(cpu) (rate(node_cpu_seconds_total{mode_filter}[60s]) * 100)')

    # --- Memory Metrics (7) ---
    metrics[8] = q(f'100 - (node_memory_MemAvailable_bytes{nf} / node_memory_MemTotal_bytes{nf}) * 100')
    metrics[9] = q(f'node_memory_MemTotal_bytes{nf} - node_memory_MemAvailable_bytes{nf}')
    metrics[10] = q(f'node_memory_MemTotal_bytes{nf}')
    metrics[11] = q(f'node_vmstat_pswpin{nf}')
    metrics[12] = q(f'node_vmstat_pswpout{nf}')
    metrics[13] = q(f'(node_netstat_Tcp_RetransSegs{nf} / node_netstat_Tcp_OutSegs{nf})*100')
    metrics[14] = q(f'(1 - (node_memory_SwapFree_bytes{nf} / node_memory_SwapTotal_bytes{nf})) * 100')
    metrics[15] = q(f'rate(node_vmstat_pswpin{nf}[60s])')
    metrics[16] = q(f'rate(node_vmstat_pswpout{nf}[60s])')

    # --- InnoDB Buffer Pool & QPS ---
    bp_reads = q(f'mysql_global_status_innodb_buffer_pool_reads{mf}')
    bp_read_requests = q(f'mysql_global_status_innodb_buffer_pool_read_requests{mf}')
    metrics[17] = (1 - bp_reads / bp_read_requests) * 100 if bp_read_requests > 0 else 100.0
    metrics[18] = q(f'rate(mysql_global_status_queries{mf}[60s])')

    bp_filter_data = mf.rstrip('}') + ',state="data"}' if mf else '{state="data"}'
    bp_filter_free = mf.rstrip('}') + ',state="free"}' if mf else '{state="free"}'
    bp_filter_misc = mf.rstrip('}') + ',state="misc"}' if mf else '{state="misc"}'
    metrics[19] = q(f'mysql_global_status_buffer_pool_pages{bp_filter_data}')
    metrics[20] = q(f'mysql_global_status_buffer_pool_pages{bp_filter_free}')
    metrics[21] = q(f'mysql_global_status_buffer_pool_pages{bp_filter_misc}')
    metrics[22] = q(f'mysql_global_status_buffer_pool_dirty_pages{mf}')

    # --- Max CPU ---
    idle_filter = nf.rstrip('}') + ',mode="idle"}' if nf else '{mode="idle"}'
    metrics[23] = q(f'(1 - min without(cpu) (rate(node_cpu_seconds_total{idle_filter}[60s]))) * 100')

    # --- MySQL Connection & Thread Metrics ---
    metrics[24] = q(f'sum(rate(mysql_global_status_aborted_clients{mf}[60s]))')
    metrics[25] = q(f'sum(rate(mysql_global_status_aborted_connects{mf}[60s]))')
    metrics[26] = q(f'sum(avg_over_time(mysql_global_status_threads_running{mf}[60s]))')
    metrics[27] = q(f'sum(max_over_time(mysql_global_status_threads_connected{mf}[60s]))')
    metrics[28] = q(f'sum(max_over_time(mysql_global_status_threads_running{mf}[60s]))')
    metrics[29] = q(f'max_over_time(mysql_global_status_threads_connected{mf}[60s])')
    metrics[30] = q(f'mysql_global_status_max_used_connections{mf}')
    metrics[31] = q(f'mysql_global_variables_max_connections{mf}')
    metrics[32] = q(f'rate(mysql_global_status_opened_files{mf}[60s])')

    # --- MySQL Handlers (13) ---
    handlers = ['delete', 'discover', 'external_lock', 'mrr_init', 'read_first',
                'read_key', 'read_last', 'read_rnd_next', 'read_prev', 'read_rnd',
                'read_next', 'update', 'write']
    for i, h in enumerate(handlers):
        h_filter = mf.rstrip('}') + f',handler="{h}"}}' if mf else f'{{handler="{h}"}}'
        metrics[33 + i] = q(f'rate(mysql_global_status_handlers_total{h_filter}[60s])')

    # --- InnoDB Row Operations (4) ---
    for i, op in enumerate(['deleted', 'inserted', 'read', 'updated']):
        op_filter = mf.rstrip('}') + f',operation="{op}"}}' if mf else f'{{operation="{op}"}}'
        metrics[46 + i] = q(f'rate(mysql_global_status_innodb_row_ops_total{op_filter}[60s])')

    # --- InnoDB Internal Memory ---
    metrics[50] = q(f'sum(mysql_global_status_innodb_page_size{mf} * on (instance) mysql_global_status_buffer_pool_pages{mf})')
    metrics[51] = q(f'sum(mysql_global_variables_innodb_log_buffer_size{mf})')
    metrics[52] = q(f'sum(mysql_global_variables_key_buffer_size{mf})')

    # --- MySQL Network Traffic ---
    metrics[53] = q(f'sum(rate(mysql_global_status_bytes_received{mf}[60s]))')
    metrics[54] = q(f'sum(rate(mysql_global_status_bytes_sent{mf}[60s]))')

    # --- MySQL Open Files ---
    metrics[55] = q(f'mysql_global_status_innodb_num_open_files{mf}')
    metrics[56] = q(f'mysql_global_status_open_files{mf}')
    metrics[57] = q(f'mysql_global_variables_open_files_limit{mf}')
    metrics[58] = q(f'mysql_global_status_open_tables{mf}')
    metrics[59] = q(f'mysql_global_variables_table_open_cache{mf}')

    # --- MySQL Select Types (5) ---
    metrics[60] = q(f'sum(rate(mysql_global_status_select_full_join{mf}[60s]))')
    metrics[61] = q(f'sum(rate(mysql_global_status_select_full_range_join{mf}[60s]))')
    metrics[62] = q(f'sum(rate(mysql_global_status_select_range{mf}[60s]))')
    metrics[63] = q(f'sum(rate(mysql_global_status_select_range_check{mf}[60s]))')
    metrics[64] = q(f'sum(rate(mysql_global_status_select_scan{mf}[60s]))')

    # --- MySQL Slow Queries ---
    metrics[65] = q(f'sum(rate(mysql_global_status_slow_queries{mf}[60s]))')

    # --- MySQL Sorts (4) ---
    metrics[66] = q(f'sum(rate(mysql_global_status_sort_merge_passes{mf}[60s]))')
    metrics[67] = q(f'sum(rate(mysql_global_status_sort_range{mf}[60s]))')
    metrics[68] = q(f'sum(rate(mysql_global_status_sort_rows{mf}[60s]))')
    metrics[69] = q(f'sum(rate(mysql_global_status_sort_scan{mf}[60s]))')

    # --- MySQL Table Definition Cache ---
    metrics[70] = q(f'mysql_global_status_open_table_definitions{mf}')
    metrics[71] = q(f'mysql_global_variables_table_definition_cache{mf}')
    metrics[72] = q(f'rate(mysql_global_status_opened_table_definitions{mf}[60s])')

    # --- MySQL Table Locks ---
    metrics[73] = q(f'sum(rate(mysql_global_status_table_locks_immediate{mf}[60s]))')
    metrics[74] = q(f'sum(rate(mysql_global_status_table_locks_waited{mf}[60s]))')

    # --- MySQL Table Open Cache Status ---
    metrics[75] = q(f'rate(mysql_global_status_opened_tables{mf}[60s])')
    metrics[76] = q(f'rate(mysql_global_status_table_open_cache_hits{mf}[60s])')
    cache_hits = metrics[76]
    cache_misses = q(f'rate(mysql_global_status_table_open_cache_misses{mf}[60s])')
    metrics[77] = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
    metrics[78] = cache_misses
    metrics[79] = q(f'rate(mysql_global_status_table_open_cache_overflows{mf}[60s])')

    # --- MySQL Temporary Objects ---
    metrics[80] = q(f'sum(rate(mysql_global_status_created_tmp_disk_tables{mf}[60s]))')
    metrics[81] = q(f'sum(rate(mysql_global_status_created_tmp_files{mf}[60s]))')
    metrics[82] = q(f'sum(rate(mysql_global_status_created_tmp_tables{mf}[60s]))')

    # --- MySQL Thread Cache ---
    metrics[83] = q(f'sum(mysql_global_status_threads_cached{mf})')
    metrics[84] = q(f'sum(mysql_global_variables_thread_cache_size{mf})')
    metrics[85] = q(f'sum(rate(mysql_global_status_threads_created{mf}[60s]))')

    # --- MySQL Transaction Handlers ---
    tx_handlers = ['commit', 'prepare', 'rollback', 'savepoint', 'savepoint_rollback']
    for i, h in enumerate(tx_handlers):
        h_filter = mf.rstrip('}') + f',handler="{h}"}}' if mf else f'{{handler="{h}"}}'
        metrics[86 + i] = q(f'rate(mysql_global_status_handlers_total{h_filter}[60s])')

    # --- Top 5 Command Usage (first batch) ---
    commands_1 = ['delete', 'insert', 'select', 'stmt_execute', 'update']
    for i, cmd in enumerate(commands_1):
        cmd_filter = mf.rstrip('}') + f',command="{cmd}"}}' if mf else f'{{command="{cmd}"}}'
        metrics[91 + i] = q(f'topk(5, rate(mysql_global_status_commands_total{cmd_filter}[60s]))')

    # --- MySQL Uptime ---
    metrics[96] = q(f'mysql_global_status_uptime{mf}')

    # --- Root Volume Disk Usage ---
    metrics[97] = q(f'(1 - (node_filesystem_avail_bytes{{mountpoint="/"}}) / node_filesystem_size_bytes{{mountpoint="/"}}) * 100')
    metrics[98] = q(f'node_filesystem_size_bytes{{mountpoint="/"}} - node_filesystem_avail_bytes{{mountpoint="/"}}')

    # --- Top 5 Command Usage (second batch) ---
    commands_2 = ['begin', 'commit', 'show_status', 'admin_commands', 'show_variables',
                  'set_option', 'show_slave_status', 'stmt_close', 'show_replica_status', 'stmt_prepare']
    for i, cmd in enumerate(commands_2):
        cmd_filter = mf.rstrip('}') + f',command="{cmd}"}}' if mf else f'{{command="{cmd}"}}'
        metrics[99 + i] = q(f'topk(5, rate(mysql_global_status_commands_total{cmd_filter}[60s]))')

    # --- Network Traffic (unified across devices) ---
    metrics[109] = q(f'sum(rate(node_network_transmit_bytes_total{nf}[60s]))')
    metrics[110] = q(f'sum(rate(node_network_receive_bytes_total{nf}[60s]))')

    # --- Disk IOPS & Busy (unified across devices) ---
    metrics[111] = q(f'max(rate(node_disk_reads_completed_total{nf}[60s]))')
    metrics[112] = q(f'max(rate(node_disk_writes_completed_total{nf}[60s]))')
    metrics[113] = q(f'max(rate(node_disk_io_time_seconds_total{nf}[60s]) * 100)')

    logger.info("Collected %d Prometheus metrics", NUM_PROMETHEUS_METRICS)
    return metrics


def build_resource_from_prometheus(metrics):
    """Build the DBTune resource dict from the 114-element Prometheus metrics array."""
    return {
        'cpu': metrics[23],       # Max CPU Usage (100 - Idle)
        'readIO': metrics[111],   # Average Disk IOPS (Read)
        'writeIO': metrics[112],  # Average Disk IOPS (Write)
        'IO': metrics[111] + metrics[112],
        'virtualMem': 0.0,
        'physical': metrics[8],   # Average Memory Usage Percentage
        'dirty': metrics[22] / max(metrics[50], 1.0) if metrics[50] > 0 else 0.0,  # dirty pages / total pages
        'hit': metrics[17] / 100.0,  # Buffer Pool Cache Hit Rate (convert % to ratio)
        'data': 0.0,
    }
