#!/usr/bin/env python3
"""Summarize the warm-start evaluation runs.

Reads scripts/DBTune_history/history_eval_*.json and prints, per cell:
  - transplant (replay): tps of default + each transplanted top-3 source config,
    with the config's tps on the SOURCE side for comparison (from eval/replay_*.json)
  - warm BO: best-so-far tps after 5 and after 10 iterations, per seed + mean
  - cold BO baseline: same metrics
Run from scripts/:  python3 report_eval.py
"""
import glob
import json
import os
import statistics as st

CELLS = [('S0', s) for s in ('sweep', 'lhs', 'random', 'llama')] + \
        [('S1', s) for s in ('sweep', 'lhs', 'random', 'llama')]
SEEDS = [42, 43, 44]
TARGET_BEST = 19940.0  # best tps ever observed on the target (S0 collection)


def tps_list(task_id):
    fn = 'DBTune_history/history_%s.json' % task_id
    if not os.path.exists(fn):
        return None
    d = json.load(open(fn))['data']
    return [(r['external_metrics']['tps'] if r['trial_state'] == 0 and r['external_metrics']
             else 0.0) for r in d]


def best_after(tps, k):
    return max(tps[:k]) if tps and len(tps) >= 1 else 0.0


def fmt_run(tps, want):
    if tps is None:
        return '(missing)'
    tag = '' if len(tps) >= want else '  [partial %d/%d]' % (len(tps), want)
    return 'b5=%6.0f b10=%6.0f%s' % (best_after(tps, 5), best_after(tps, 10), tag)


def main():
    print('target best-known tps = %.0f (from S0 collection)\n' % TARGET_BEST)

    print('== transplant (replay: default + top-3 source configs on the target) ==')
    for cell, strat in CELLS:
        tps = tps_list('eval_replay_%s_%s' % (cell, strat))
        meta = json.load(open('eval/replay_%s_%s.json' % (cell, strat)))
        if tps is None:
            print('%s_%-6s : (missing)' % (cell, strat)); continue
        pairs = ', '.join('%.0f->%s' % (s, ('%.0f' % t) if t else 'FAIL')
                          for s, t in zip(meta['source_tps'], tps[1:]))
        print('%s_%-6s : default=%6.0f  src->tgt: %s'
              % (cell, strat, tps[0] if tps else 0, pairs))

    print('\n== warm-started BO (SMAC + ottertune mapping, 10 iters) ==')
    for cell, strat in CELLS:
        rows, b5s, b10s = [], [], []
        for seed in SEEDS:
            tps = tps_list('eval_warm_%s_%s_s%d' % (cell, strat, seed))
            rows.append('s%d: %s' % (seed, fmt_run(tps, 10)))
            if tps:
                b5s.append(best_after(tps, 5)); b10s.append(best_after(tps, 10))
        mean = ('mean b5=%6.0f b10=%6.0f' % (st.mean(b5s), st.mean(b10s))) if b5s else ''
        print('%s_%-6s : %s   %s' % (cell, strat, ' | '.join(rows), mean))

    print('\n== cold-start BO baseline (SMAC, no transfer, 10 iters) ==')
    b5s, b10s = [], []
    for seed in SEEDS:
        tps = tps_list('eval_cold_s%d' % seed)
        print('cold s%d  : %s' % (seed, fmt_run(tps, 10)))
        if tps:
            b5s.append(best_after(tps, 5)); b10s.append(best_after(tps, 10))
    if b5s:
        print('cold mean : b5=%6.0f b10=%6.0f' % (st.mean(b5s), st.mean(b10s)))


if __name__ == '__main__':
    main()
