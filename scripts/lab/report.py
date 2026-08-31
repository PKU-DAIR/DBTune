#!/usr/bin/env python3
"""Print per-workload result tables: selected_source + tps + all knob values,
one column per (method, iteration).

Usage: python report.py [workloads...]   (default: read rw50 write)
Run from the repo root (paths are relative to scripts/).
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(ROOT, 'scripts', 'DBTune_history')
LOGS = os.path.join(ROOT, 'logs', 'runs')

KNOBS = list(json.load(open(os.path.join(ROOT, 'scripts/experiment/gen_knobs/DML_12.json'))).keys())
SHORT = {
    'innodb_buffer_pool_size': 'bufpool_size', 'innodb_buffer_pool_instances': 'bufpool_inst',
    'innodb_read_io_threads': 'read_io_thr', 'innodb_write_io_threads': 'write_io_thr',
    'innodb_flush_log_at_trx_commit': 'flush_log_commit', 'innodb_adaptive_hash_index': 'adaptive_hash',
    'sync_binlog': 'sync_binlog', 'innodb_lru_scan_depth': 'lru_scan_depth',
    'innodb_change_buffer_max_size': 'chg_buf_max', 'innodb_io_capacity': 'io_capacity',
    'innodb_log_file_size': 'log_file_size', 'table_open_cache': 'table_open_cache',
}
import os as _os
# Methods to show, as (header, history/log short code). Override via env
# REPORT_METHODS, e.g. "dmlmap:ot" or "dml:dmlmap:ot".
_codes = _os.environ.get('REPORT_METHODS', 'dmlmap:ot').split(':')
_LABELS = {'dml': 'DML', 'ot': 'OT', 'dmlmap': 'DMLmap'}
METHODS = [(_LABELS.get(c, c), c) for c in _codes]


def cfg_and_tps(mk, wl, it):
    p = os.path.join(HIST, f'history_{mk}_sysbench_{wl}.json')
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))['data']
    if len(d) <= it:
        return None, None
    e = d[it]
    em = e.get('external_metrics') or {}
    tps = em.get('tps') if isinstance(em, dict) else None
    return e.get('configuration', {}), tps


COLW = 24  # data column width


def compact_ctx(ctx):
    # 24c32g_64-1000000-4-oltp_read_write_5-1.0      -> 24c32g_read_write_5-1.0
    # history_32c64g_64-100000-4-oltp_read_write_80-1 -> 32c64g_read_write_80-1
    ctx = ctx.replace('history_', '')
    ctx = re.sub(r'_\d+-\d+-\d+-oltp_', '_', ctx)
    return ctx


def matched_source_raw(mk, wl):
    """Return the FULL matched_context task_id (table size intact), or '?'/'(none)'."""
    # Prefer the matched context stored in the history JSON (reliable even when
    # the run's logs were overwritten by a later no-op run).
    p = os.path.join(HIST, f'history_{mk}_sysbench_{wl}.json')
    if os.path.exists(p):
        try:
            d = json.load(open(p))['data']
            if len(d) > 1:
                mc = (d[1].get('context') or {}).get('matched_context')
                if mc:
                    return mc
        except Exception:
            pass
    # Fall back to the run log.
    lp = os.path.join(LOGS, f'config_sysbench_{mk}_{wl}.log')
    if not os.path.exists(lp):
        return '?'
    txt = open(lp, errors='ignore').read()
    m = re.search(r'DML matched context: (\S+)', txt) if mk == 'dml' \
        else re.search(r'Matched context: (\S+)', txt)
    return m.group(1) if m else '(none)'


def matched_source(mk, wl):
    raw = matched_source_raw(mk, wl)
    return raw if raw in ('?', '(none)') else compact_ctx(raw)


def src_table_size(mk, wl):
    """Rows/table of the matched source, parsed from its full task_id.
    Format: <hw>_<tables>-<tablesize>-<clients>-oltp_... ; 'n/a' for tpcc/non-oltp."""
    raw = matched_source_raw(mk, wl)
    if raw in ('?', '(none)'):
        return raw
    m = re.search(r'_\d+-(\d+)-\d+-oltp_', raw)
    return m.group(1) if m else 'n/a'


def main(workloads):
    for wl in workloads:
        cols = []  # (header, configuration dict, tps, source, table_size)
        for mname, mk in METHODS:
            src = matched_source(mk, wl)
            tsize = src_table_size(mk, wl)
            for it in (0, 1):
                c, t = cfg_and_tps(mk, wl, it)
                hdr = f'{mname} i{it}'
                cols.append((hdr, c, t,
                             '(default)' if it == 0 else src,
                             '(default)' if it == 0 else tsize))

        title = f' sysbench {wl} '
        print('\n' + title.center(18 + COLW * len(cols), '='))
        header = f'{"row":18s}' + ''.join(f'{c[0]:>{COLW}s}' for c in cols)
        print(header)
        print('-' * len(header))
        # selected_source row
        print(f'{"selected_source":18s}' + ''.join(f'{(c[3] or "?"):>{COLW}s}' for c in cols))
        # source table size row
        print(f'{"src_table_size":18s}' + ''.join(f'{str(c[4]):>{COLW}s}' for c in cols))
        # tps row
        print(f'{"tps":18s}' + ''.join(f'{(f"{c[2]:.1f}" if c[2] is not None else "-"):>{COLW}s}' for c in cols))
        # knob rows
        for k in KNOBS:
            row = f'{SHORT.get(k, k):18s}'
            for col in cols:
                c = col[1]
                v = c.get(k) if c else None
                row += f'{str(v):>{COLW}s}'
            print(row)


if __name__ == '__main__':
    wls = sys.argv[1:] or ['read', 'rw50', 'write']
    main(wls)
