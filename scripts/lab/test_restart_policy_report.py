#!/usr/bin/env python3
"""Summarize scripts/lab/test_restart_policy.sh output (parallel/restart_test/).

Per cycle: which config was started after which shutdown policy, seconds to accept
connections, the error-log phase timeline of that startup, dirty pages / checkpoint age
right before the shutdown, the shutdown policy and its duration.  Then aggregates:
  startup time by the policy of the PRECEDING shutdown (recovery cost of that policy),
  shutdown time by policy, and dead time per iteration = shutdown(policy) + startup after it.
Run from anywhere:  python3 scripts/lab/test_restart_policy_report.py [--dir parallel/restart_test]
"""
import argparse
import csv
import os
import re
import statistics as st
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, '..', '..', 'parallel', 'restart_test')
PHASES = [  # (label, regex on error-log line)
    ('init', r'InnoDB initialization has started'),
    ('not-clean', r'was not shut down normally|Starting crash recovery'),
    ('apply', r'Applying a batch|Apply batch'),
    ('redo-resize', r'[Rr]esiz\w* redo|redo log .*resiz|Log file .*resized|Redo log'),
    ('pool', r'[Cc]ompleted initialization of buffer pool'),
    ('init-end', r'InnoDB initialization has ended'),
    ('ready', r'ready for connections'),
]


def ts(line):
    m = re.match(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z', line)
    return datetime.fromisoformat(m.group(1)) if m else None


def timeline(path):
    if not os.path.exists(path):
        return ''
    lines = open(path, errors='replace').read().splitlines()
    t0 = next((ts(l) for l in lines if ts(l)), None)
    if t0 is None:
        return ''
    out = []
    for label, pat in PHASES:
        for l in lines:
            if re.search(pat, l):
                t = ts(l)
                if t:
                    out.append('%s@%.0fs' % (label, (t - t0).total_seconds()))
                    break
    return ' '.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=DEFAULT_DIR)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.dir, 'results.csv'))))
    print('cycle cnf  after   startup_s  tps     dirty_pages ckpt_age_MB pool_MB policy shutdown_s fallback  startup phases')
    for r in rows:
        print('%5s %-4s %-7s %8.1f %7.0f %11s %11.0f %7s %-6s %9.1f %8s  %s' % (
            r['cycle'], r['start_cnf'], r['prev_policy'], float(r['startup_s']), float(r['tps'] or 0),
            r['dirty_pages'], float(r['checkpoint_age_bytes'] or 0) / 2**20, r['pool_mb'], r['policy'],
            float(r['shutdown_s']), r['fallback_kill9'],
            timeline(os.path.join(a.dir, 'errlog_%s_start.txt' % r['cycle']))))
    by_prev, by_pol = {}, {}
    for r in rows:
        by_prev.setdefault(r['prev_policy'], []).append(float(r['startup_s']))
        by_pol.setdefault(r['policy'], []).append(float(r['shutdown_s']))
    print('\nstartup time by PRECEDING shutdown policy (= recovery cost of that policy):')
    for k, v in by_prev.items():
        print('  after %-8s n=%d  mean %6.1f s  values %s' % (k, len(v), st.mean(v), ['%.0f' % x for x in v]))
    print('shutdown time by policy:')
    for k, v in by_pol.items():
        print('  %-8s n=%d  mean %6.1f s  values %s' % (k, len(v), st.mean(v), ['%.0f' % x for x in v]))
    print('\ndead time per iteration = shutdown(policy) + startup after it (excl. ~10 s of DBTune bookkeeping):')
    for k in by_pol:
        if k in by_prev:
            print('  %-8s %6.1f s' % (k, st.mean(by_pol[k]) + st.mean(by_prev[k])))
    print('\nsanity: tps per config across policies (should not differ systematically):')
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r['start_cnf'], []).append('%s:%.0f' % (r['policy'], float(r['tps'] or 0)))
    for k, v in sorted(by_cfg.items()):
        print('  %s  %s' % (k, '  '.join(v)))


if __name__ == '__main__':
    main()
