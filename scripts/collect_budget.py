#!/usr/bin/env python3
"""Print the exact 1-knob-sweep budget N for a knob file.

N (schedule length) is what goes into `max_runs` of ALL four collection cells of a
source workload (sweep/lhs/random/llama are budget-matched). Run from scripts/:

    python collect_budget.py --knobs ./experiment/gen_knobs/mysql_perf_8.0.json --levels 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_llamatune import build_space  # same space construction as DBTuner
from autotune.optimizer.sampler_optimizer import Sampler_Optimizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--knobs', default='./experiment/gen_knobs/mysql_perf_8.0.json')
    p.add_argument('--levels', type=int, default=5)
    p.add_argument('--breakdown', action='store_true', help='print per-knob run counts')
    args = p.parse_args()

    space, _ = build_space(args.knobs)
    sampler = Sampler_Optimizer(space, strategy='sweep', sweep_levels=args.levels)

    default = dict(space.get_default_configuration())
    per_knob = {}
    for config in sampler.schedule[1:]:
        changed = [k for k, v in dict(config).items() if v != default[k]]
        assert len(changed) == 1, changed
        per_knob[changed[0]] = per_knob.get(changed[0], 0) + 1

    if args.breakdown:
        for name in sorted(per_knob):
            print('%-45s %d' % (name, per_knob[name]))
    n_hps = len(space.get_hyperparameters())
    print('knob file : %s (%d knobs, %d swept)' % (args.knobs, n_hps, len(per_knob)))
    print('levels    : %d' % args.levels)
    print('sweep N   : %d  (1 default + %d sweep runs)  -> set max_runs = %d in all '
          'four cells of a source' % (len(sampler.schedule), len(sampler.schedule) - 1,
                                      len(sampler.schedule)))


if __name__ == '__main__':
    main()
