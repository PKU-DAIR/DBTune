#!/usr/bin/env python3
"""Report the anchor-probe runs (Algorithm 1 probing test).

Run from scripts/:  python3 lab/prep_report.py [probe ...]
For each probe: per-config tps over repeats, then the Algorithm-1 ratio
p/qref for each anchor with three qref choices:
  pool  = the probe cell's own pool max (single draw, winner's-curse inflated)
  fresh = max over this session's re-measurements of the cell's own top-2
  best  = max over EVERY non-anchor config measured on this workload in this
          session (library + own) -- the only qref available for C64 / R400k
"""
import json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
PROBES = sys.argv[1:] or ['S1', 'S2', 'S3', 'S4', 'C64', 'R400k']
THETAS = (0.9, 0.8)
POOL_MAX = {'S1': 15781.0, 'S2': 15959.0, 'S3': 32095.0, 'S4': 15415.0}


def fmt(v):
    return '%7.0f' % v if v else '   FAIL'


for probe in PROBES:
    task = 'prep_anchor_%s' % probe
    meta_fn = os.path.join(SCRIPTS, 'eval', task + '.json')
    hist_fn = os.path.join(SCRIPTS, 'DBTune_history', 'history_%s.json' % task)
    print('=' * 78); print('probe %s   (%s)' % (probe, task))
    if not os.path.exists(meta_fn):
        print('  (no replay file - run prep_gen_task.py first)'); continue
    meta = json.load(open(meta_fn))
    if not os.path.exists(hist_fn):
        print('  (no history yet)'); continue
    rows = json.load(open(hist_fn))['data']
    tps = [(r['external_metrics'].get('tps', 0.0) if r['trial_state'] == 0 and r['external_metrics'] else 0.0)
           for r in rows]
    print('  evaluated %d/%d   default=%s' % (len(rows), 1 + len(meta['configs']), fmt(tps[0]) if tps else 'n/a'))
    per = {}
    for i, t in enumerate(tps[1:]):
        base = meta['labels'][i].split('/')[0]
        per.setdefault(base, dict(role=meta['roles'][i], src=meta['source_tps'][i], vals=[]))['vals'].append(t)
    print('  %-14s %-8s %8s  %s' % ('config', 'role', 'src_tps', 'measured on this workload (median)'))
    for lab, d in per.items():
        ok = [v for v in d['vals'] if v]
        med = st.median(ok) if ok else 0.0
        print('  %-14s %-8s %8.0f  %s  -> med %s' % (lab, d['role'], d['src'], ' '.join(fmt(v) for v in d['vals']), fmt(med)))
    own = [v for lab, d in per.items() if d['role'] == 'own' for v in d['vals'] if v]
    non_anchor = [v for lab, d in per.items() if d['role'] != 'anchor' for v in d['vals'] if v]
    qrefs = {'pool': POOL_MAX.get(probe), 'fresh': max(own) if own else None,
             'best': max(non_anchor) if non_anchor else None}
    print('  qref: ' + '  '.join('%s=%s' % (k, fmt(v) if v else 'n/a') for k, v in qrefs.items()))
    for lab, d in per.items():
        if d['role'] != 'anchor':
            continue
        ok = [v for v in d['vals'] if v]
        if not ok:
            print('  %-14s p=FAIL' % lab); continue
        p = st.median(ok)
        parts = []
        for k, q in qrefs.items():
            if not q:
                continue
            r = p / q
            verdict = ' '.join('%s@%.1f' % ('PASS' if r > th else 'fail', th) for th in THETAS)
            parts.append('%s %.3f (%s)' % (k, r, verdict))
        print('  %-14s p=%7.0f (n=%d, min %.0f max %.0f)  p/qref: %s' % (lab, p, len(ok), min(ok), max(ok), '; '.join(parts)))
