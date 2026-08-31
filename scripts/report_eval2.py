#!/usr/bin/env python3
"""Summarize the 1 h-deadline transfer-method comparison (the "tuning is slow,
high-variance, and surrogate transfer converts nothing" measurement).

Reads DBTune_history/history_eval2_<arm>_s<seed>.json (written by
scripts/lab/transfer_node_run.sh; each row carries `update_time` = wall-clock
seconds since optimize.py started) and prints, per arm and seed:
  best tps within the deadline, #iterations, time at which the best was found,
plus the per-arm mean/median. --trace prints every iteration (t, tps).

Run from anywhere:  python3 scripts/report_eval2.py [--deadline 3600] [--trace] [--hist-dir DIR]
Reference output (2026-07-07/08 runs) is in exp_notes/similar_workload.md, F7/F7b.
"""
import argparse
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(HERE, 'DBTune_history')
ARMS = ['opadviser', 'opadviser_ns', 'ottertune', 'rgpe', 'cold']
SEEDS = [42, 43, 44]


def load(arm, seed, hist_dir=HIST_DIR, prefix='eval2'):
    fn = os.path.join(hist_dir, 'history_%s_%s_s%d.json' % (prefix, arm, seed))
    if not os.path.exists(fn):
        return None
    rows = json.load(open(fn))['data']
    out = []
    for r in rows:
        t = r.get('update_time')
        ok = r['trial_state'] == 0 and bool(r['external_metrics'])
        out.append((t if t is not None else float('nan'),
                    r['external_metrics']['tps'] if ok else None))
    return out


def summarize(trace, deadline):
    within = [(t, tps) for t, tps in trace if t <= deadline]
    okay = [(t, tps) for t, tps in within if tps is not None]
    if not okay:
        return 0.0, len(within), float('nan')
    t_best, best = max(okay, key=lambda x: x[1])
    return best, len(within), t_best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deadline', type=float, default=3600.0, help='seconds (default 3600)')
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--trace', action='store_true', help='print every iteration')
    ap.add_argument('--hist-dir', default=HIST_DIR, help='directory holding history_<prefix>_*.json (e.g. a backup)')
    ap.add_argument('--prefix', default='eval2', help='task-id prefix: eval2 (offline kill -9, default) | eval3 (online) | eval4 (all knobs, clean shutdown)')
    a = ap.parse_args()

    print('best tps within %.0f s, per arm and seed (n = iterations started within the deadline; '
          't = time of best)\n' % a.deadline)
    hdr = '%-13s' % 'arm' + ''.join('%22s' % ('seed %d' % s) for s in a.seeds) + '%9s%9s' % ('mean', 'median')
    print(hdr)
    for arm in a.arms:
        cells, bests = [], []
        for seed in a.seeds:
            tr = load(arm, seed, a.hist_dir, a.prefix)
            if tr is None:
                cells.append('%22s' % '(missing)')
                continue
            best, n, t_best = summarize(tr, a.deadline)
            bests.append(best)
            cells.append('%22s' % ('%6.0f (n=%2d, t=%4.0fs)' % (best, n, t_best)))
        line = '%-13s' % arm + ''.join(cells)
        if bests:
            line += '%9.0f%9.0f' % (st.mean(bests), st.median(bests))
        print(line)
        if a.trace:
            for seed in a.seeds:
                tr = load(arm, seed, a.hist_dir, a.prefix)
                if tr is None:
                    continue
                print('    s%d: ' % seed + ' '.join(
                    '%.0fs:%s' % (t, ('%.0f' % tps) if tps is not None else 'FAIL') for t, tps in tr))


if __name__ == '__main__':
    main()
