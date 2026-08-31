"""
Metrics collection module for the DML (Deep Metric Learning) optimizer.

Provides two ways to collect the 12 model input metrics:
1. From Prometheus (preferred, exact unit match with training data)
2. From a DBTune Observation (fallback, approximate unit mapping)
"""
import numpy as np
import logging
import requests

logger = logging.getLogger(__name__)

METRIC_NAMES = [
    'Average Memory Usage Percentage',
    'InnoDB Buffer Pool Cache Hit Rate',
    'InnoDB Dirty Buffer Pages',
    'Current QPS (Queries Per Second)',
    'Max CPU Usage (100 - Idle)',
    'InnoDB Rows Deleted (60s Rate)',
    'InnoDB Rows Inserted (60s Rate)',
    'InnoDB Rows Read (60s Rate)',
    'InnoDB Rows Updated (60s Rate)',
    'Average Disk IOPS (Read)',
    'Average Disk IOPS (Write)',
]

# InnoDB metric names as they appear in information_schema.INNODB_METRICS (sorted alphabetically)
# These are used to find the correct index in the 65-element IM array
_IM_METRIC_NAMES = {
    'dml_deletes': 'InnoDB Rows Deleted (60s Rate)',
    'dml_inserts': 'InnoDB Rows Inserted (60s Rate)',
    'dml_reads': 'InnoDB Rows Read (60s Rate)',
    'dml_updates': 'InnoDB Rows Updated (60s Rate)',
    'buffer_pool_pages_dirty': 'InnoDB Dirty Buffer Pages',
}


def _query_prometheus(prometheus_url, query):
    """Execute a PromQL instant query and return the scalar value."""
    resp = requests.get(f"{prometheus_url}/api/v1/query", params={'query': query}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data['status'] != 'success' or not data['data']['result']:
        logger.warning("Prometheus query returned no data: %s", query)
        return 0.0
    return float(data['data']['result'][0]['value'][1])


def collect_metrics_from_prometheus(prometheus_url, mysql_instance=None, node_instance=None):
    """
    Collect the 11 model input metrics from Prometheus (TPS excluded).

    Args:
        prometheus_url: Base URL of Prometheus (e.g., http://localhost:9090)
        mysql_instance: MySQL exporter instance label filter (e.g., 'localhost:9104')
        node_instance: Node exporter instance label filter (e.g., 'localhost:9100')

    Returns:
        numpy array of shape (11,) with raw metric values
    """
    mysql_filter = f'{{instance="{mysql_instance}"}}' if mysql_instance else ''
    node_filter = f'{{instance="{node_instance}"}}' if node_instance else ''

    metrics = np.zeros(11)

    # 0: Average Memory Usage Percentage
    metrics[0] = _query_prometheus(
        prometheus_url,
        f'100 - (node_memory_MemAvailable_bytes{node_filter} / node_memory_MemTotal_bytes{node_filter}) * 100'
    )

    # 1: InnoDB Buffer Pool Cache Hit Rate
    reads = _query_prometheus(prometheus_url, f'mysql_global_status_innodb_buffer_pool_reads{mysql_filter}')
    read_requests = _query_prometheus(prometheus_url, f'mysql_global_status_innodb_buffer_pool_read_requests{mysql_filter}')
    if read_requests > 0:
        metrics[1] = (1 - reads / read_requests) * 100
    else:
        metrics[1] = 100.0

    # 2: InnoDB Dirty Buffer Pages
    metrics[2] = _query_prometheus(prometheus_url, f'mysql_global_status_buffer_pool_dirty_pages{mysql_filter}')

    # 3: Current QPS
    metrics[3] = _query_prometheus(prometheus_url, f'rate(mysql_global_status_queries{mysql_filter}[60s])')

    # 4: Max CPU Usage (100 - Idle)
    idle_filter = node_filter.rstrip('}') + ',mode="idle"}' if node_filter else '{mode="idle"}'
    metrics[4] = _query_prometheus(
        prometheus_url,
        f'(1 - min without(cpu) (rate(node_cpu_seconds_total{idle_filter}[60s]))) * 100'
    )

    # 5-8: InnoDB Row Operations (60s rate)
    for i, op in enumerate(['deleted', 'inserted', 'read', 'updated']):
        op_filter = mysql_filter.rstrip('}') + f',operation="{op}"}}' if mysql_filter else f'{{operation="{op}"}}'
        metrics[5 + i] = _query_prometheus(
            prometheus_url,
            f'rate(mysql_global_status_innodb_row_ops_total{op_filter}[60s])'
        )

    # 9: Average Disk IOPS (Read) - max across devices
    metrics[9] = _query_prometheus(
        prometheus_url,
        f'max(rate(node_disk_reads_completed_total{node_filter}[60s]))'
    )

    # 10: Average Disk IOPS (Write) - max across devices
    metrics[10] = _query_prometheus(
        prometheus_url,
        f'max(rate(node_disk_writes_completed_total{node_filter}[60s]))'
    )

    logger.info("Collected 11 metrics from Prometheus: %s", dict(zip(METRIC_NAMES, metrics)))
    return metrics


def extract_metrics_from_observation(observation, im_index_map=None):
    """
    Extract the 11 model input metrics from a DBTune Observation (TPS excluded).

    Note: Some metrics have unit differences vs Prometheus-collected training data:
    - Disk IOPS: DBTune collects bytes/sec via psutil, Prometheus collects ops/sec
    - Memory: DBTune collects GB via psutil, Prometheus collects percentage

    Args:
        observation: Observation namedtuple from HistoryContainer
        im_index_map: Dict mapping InnoDB metric name -> index in IM array.
                      If None, falls back to resource dict values.

    Returns:
        numpy array of shape (11,) with raw metric values
    """
    em = observation.EM if observation.EM else {}
    resource = observation.resource if observation.resource else {}
    im = observation.IM if observation.IM is not None else []

    metrics = np.zeros(11)

    # 0: Average Memory Usage Percentage
    try:
        import psutil
        total_mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        physical_gb = resource.get('physical', 0.0)
        metrics[0] = (physical_gb / total_mem_gb) * 100 if total_mem_gb > 0 else 0.0
    except (ImportError, Exception):
        metrics[0] = resource.get('physical', 0.0)

    # 1: InnoDB Buffer Pool Cache Hit Rate (percentage)
    hit = resource.get('hit', 0.0)
    metrics[1] = hit * 100 if hit <= 1.0 else hit

    # 2: InnoDB Dirty Buffer Pages (raw count)
    if im_index_map and 'buffer_pool_pages_dirty' in im_index_map and len(im) > im_index_map['buffer_pool_pages_dirty']:
        metrics[2] = im[im_index_map['buffer_pool_pages_dirty']]
    else:
        metrics[2] = resource.get('dirty', 0.0)

    # 3: Current QPS
    metrics[3] = em.get('qps', 0.0)

    # 4: Max CPU Usage
    metrics[4] = resource.get('cpu', 0.0)

    # 5-8: InnoDB Row Operations
    im_keys = ['dml_deletes', 'dml_inserts', 'dml_reads', 'dml_updates']
    for i, key in enumerate(im_keys):
        if im_index_map and key in im_index_map and len(im) > im_index_map[key]:
            metrics[5 + i] = im[im_index_map[key]]
        else:
            metrics[5 + i] = 0.0

    # 9-10: Disk IOPS (Read/Write)
    # Note: DBTune's readIO/writeIO are in bytes/sec, training data uses ops/sec
    metrics[9] = resource.get('readIO', 0.0)
    metrics[10] = resource.get('writeIO', 0.0)

    logger.info("Extracted 11 metrics from Observation: %s", dict(zip(METRIC_NAMES, metrics)))
    return metrics


def build_im_index_map(db_connection):
    """
    Build a mapping from InnoDB metric names to their sorted index position
    in the 65-element internal_metrics array.

    Args:
        db_connection: A MySQL connection object (e.g., from mysqldb.py)

    Returns:
        Dict mapping metric name -> index in sorted IM array
    """
    try:
        cursor = db_connection.cursor()
        cursor.execute("SELECT NAME FROM information_schema.INNODB_METRICS WHERE status='enabled' ORDER BY NAME")
        rows = cursor.fetchall()
        index_map = {row[0]: i for i, row in enumerate(rows)}
        cursor.close()
        return index_map
    except Exception as e:
        logger.warning("Could not build IM index map: %s", e)
        return {}
