"""
Convert DBMSTransferLearning full_data CSV files into DBTune-compatible JSON history files.

Each CSV row becomes one observation. Each unique {hardware}_{workload_label} context
becomes a separate JSON history file, matching DBTune's HistoryContainer format.

Usage:
    python convert_csv_to_history.py \
        --input_dir ../DBMSTransferLearning/dataset/full_data/ \
        --output_dir ./DBTune_history/csv_source/ \
        --knob_config ./experiment/gen_knobs/DML_12.json
"""
import argparse
import json
import os
import sys
import pandas as pd
import numpy as np

# 12 knob columns in CSV order
KNOB_COLS = [
    'innodb_buffer_pool_size', 'innodb_read_io_threads', 'innodb_write_io_threads',
    'innodb_flush_log_at_trx_commit', 'innodb_adaptive_hash_index', 'sync_binlog',
    'innodb_lru_scan_depth', 'innodb_buffer_pool_instances', 'innodb_change_buffer_max_size',
    'innodb_io_capacity', 'innodb_log_file_size', 'table_open_cache'
]

# Knobs stored as GB in CSV that need byte conversion
GB_KNOBS = {'innodb_buffer_pool_size'}
GB_TO_BYTES = 1073741824
# innodb_log_file_size is stored as MB/1000 in CSV (e.g., 48MB -> 0.048)
# See system_configuration.py line 109: int(val.removesuffix("MB")) / 1000
LOG_FILE_SIZE_KNOB = 'innodb_log_file_size'
LOG_FILE_MULTIPLIER = 1000 * 1024 * 1024  # * 1000 to get MB, * 1024^2 to get bytes

# Enum knobs: stored as int in CSV, need string representation in ConfigSpace
ENUM_KNOBS = {'innodb_flush_log_at_trx_commit', 'innodb_adaptive_hash_index', 'sync_binlog'}

# Metric columns: everything between tps and workload_label
# These are determined dynamically from the CSV header


def get_metric_columns(df):
    """Get the ordered list of metric column names (between 'tps' and 'workload_label')."""
    cols = list(df.columns)
    tps_idx = cols.index('tps')
    wl_idx = cols.index('workload_label')
    return cols[tps_idx + 1:wl_idx]


def convert_knob_value(name, value):
    """Convert a CSV knob value to DBTune's raw MySQL format."""
    if name == LOG_FILE_SIZE_KNOB:
        # CSV stores as MB/1000 (e.g., 0.048 = 48MB, 5.0 = 5000MB)
        return int(round(float(value) * LOG_FILE_MULTIPLIER))
    if name in GB_KNOBS:
        return int(round(float(value) * GB_TO_BYTES))
    if name in ENUM_KNOBS:
        return str(int(value))
    return int(value) if isinstance(value, (float, np.floating)) else value


def build_resource(row, metric_cols):
    """Build the resource dict from CSV metric values."""
    def get(name, default=0.0):
        return float(row[name]) if name in row.index else default

    # Buffer pool hit rate
    hit = get('InnoDB Buffer Pool Cache Hit Rate', 100.0) / 100.0

    # Dirty pages ratio
    dirty_pages = get('InnoDB Dirty Buffer Pages', 0.0)
    total_pages = get('InnoDB Buffer Pool Total Pages', 1.0)
    dirty = dirty_pages / total_pages if total_pages > 0 else 0.0

    # Memory percentage → physical (DBTune stores as percentage in 'physical')
    physical = get('Average Memory Usage Percentage', 0.0)

    cpu = get('Max CPU Usage (100 - Idle)', 0.0)
    read_io = get('Average Disk IOPS (Read)', 0.0)
    write_io = get('Average Disk IOPS (Write)', 0.0)

    return {
        'cpu': cpu,
        'readIO': read_io,
        'writeIO': write_io,
        'IO': read_io + write_io,
        'virtualMem': 0.0,
        'physical': physical,
        'dirty': dirty,
        'hit': hit,
        'data': 0.0,
    }


def convert_csv_file(csv_path, output_dir, knob_config=None):
    """Convert a single full_data CSV file into per-context JSON history files."""
    df = pd.read_csv(csv_path)
    hw_id = os.path.basename(csv_path).replace('-result.csv', '')
    metric_cols = get_metric_columns(df)

    print(f"Processing {csv_path}: {len(df)} rows, {df['workload_label'].nunique()} workloads, {len(metric_cols)} metrics")

    # Load knob config for sys.maxsize scaling if provided
    knob_scaling = {}
    if knob_config and os.path.exists(knob_config):
        with open(knob_config) as f:
            knobs = json.load(f)
        for name, info in knobs.items():
            if info.get('type') == 'integer' and info.get('max', 0) > sys.maxsize:
                knob_scaling[name] = 1000

    files_created = 0
    for workload_label, group in df.groupby('workload_label'):
        context_id = f"{hw_id}_{workload_label}"
        observations = []

        for _, row in group.iterrows():
            # Build configuration dict
            config = {}
            for knob in KNOB_COLS:
                val = convert_knob_value(knob, row[knob])
                # Apply sys.maxsize scaling if needed
                if knob in knob_scaling:
                    val = int(val / knob_scaling[knob])
                config[knob] = val

            # Build external metrics
            tps = float(row['tps'])
            external_metrics = {
                'tps': tps,
                'lat': -1,
                'qps': -1,
                'tpsVar': -1,
                'latVar': -1,
                'qpsVar': -1,
            }

            # Build internal metrics (114 values in CSV column order)
            internal_metrics = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in metric_cols]

            # Build resource dict
            resource = build_resource(row, metric_cols)

            observations.append({
                'configuration': config,
                'external_metrics': external_metrics,
                'internal_metrics': internal_metrics,
                'resource': resource,
                'context': None,
                'trial_state': 0,
                'elapsed_time': 0,
                'iter_time': 0,
            })

        # Write JSON history file
        history = {
            'info': {
                'objs': ['tps'],
                'constraints': [],
            },
            'data': observations,
        }

        safe_label = workload_label.replace('/', '_').replace(' ', '_')
        out_path = os.path.join(output_dir, f'history_{hw_id}_{safe_label}.json')
        with open(out_path, 'w') as f:
            json.dump(history, f, indent=2)

        files_created += 1

    print(f"  Created {files_created} JSON history files in {output_dir}")
    return files_created


def main():
    parser = argparse.ArgumentParser(description='Convert full_data CSVs to DBTune JSON history')
    parser.add_argument('--input_dir', default='../DBMSTransferLearning/dataset/full_data/',
                        help='Directory containing *-result.csv files')
    parser.add_argument('--output_dir', default='./DBTune_history/csv_source/',
                        help='Output directory for JSON history files')
    parser.add_argument('--knob_config', default='./experiment/gen_knobs/DML_12.json',
                        help='Knob config JSON for sys.maxsize scaling info')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('-result.csv')])
    if not csv_files:
        print(f"No *-result.csv files found in {args.input_dir}")
        return

    total = 0
    for csv_file in csv_files:
        csv_path = os.path.join(args.input_dir, csv_file)
        total += convert_csv_file(csv_path, args.output_dir, args.knob_config)

    print(f"\nDone! Created {total} JSON history files total.")


if __name__ == '__main__':
    main()
