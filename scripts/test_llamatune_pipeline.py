"""Wiring smoke test: drive PipleLine with optimizer_type=LlamaTune and a synthetic
objective function (no MySQL). Verifies registration, the update dispatch, and that
the misconfiguration assertions fire. Run from scripts/:
    python test_llamatune_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
import shutil

from autotune.utils.constants import SUCCESS
from autotune.pipleline.pipleline import PipleLine
from test_llamatune import build_space, synthetic_objective, KNOB_FILE

logging.basicConfig(level=logging.WARNING)

TASK_ID = 'smoke_llamatune'


def objective_function(config):
    objs = [synthetic_objective(config)]
    em = {'tps': -objs[0], 'lat': 0.0, 'qps': 0.0}
    im = [0.0] * 65
    resource = {}
    info = {'objs': ['tps'], 'constraints': []}
    return objs, None, em, resource, im, info, SUCCESS


def make_pipeline(cs, **overrides):
    params = dict(
        objective_function=objective_function,
        config_space=cs,
        num_objs=1,
        num_constraints=0,
        optimizer_type='LlamaTune',
        max_runs=8,
        surrogate_type='prf',
        initial_runs=2,
        init_strategy='random_explore_first',
        task_id=TASK_ID,
        random_state=42,
        selector_type='shap',
        incremental='none',
        num_hps_init=-1,
        num_metrics=65,
        space_transfer=False,
        auto_optimizer=False,
        llamatune_low_dim=8,
        llamatune_max_num_values=100,
    )
    params.update(overrides)
    return PipleLine(**params)


def main():
    # stale history from a previous run would short-circuit iteration 0
    hist = os.path.join('DBTune_history', 'history_%s.json' % TASK_ID)
    if os.path.exists(hist):
        os.remove(hist)

    cs, knobs = build_space(KNOB_FILE, -1)

    # misconfigurations must be rejected
    # (space_transfer=True is unconstructable without history_bo_data, so not tested here)
    for bad in (dict(incremental='increase'), dict(num_hps_init=20)):
        try:
            make_pipeline(cs, **bad)
        except AssertionError:
            pass
        else:
            raise SystemExit('expected AssertionError for %s' % bad)
    print('misconfiguration assertions OK')

    pipe = make_pipeline(cs)
    history = pipe.run()

    n = len(history.configurations)
    assert n == 8, f'expected 8 evaluations, got {n}'
    assert len(pipe.optimizer.inner_history.configurations) == n, \
        'inner history must track every evaluation'
    assert history.configurations[0] == cs.get_default_configuration(), \
        'iteration 0 must evaluate the default config'
    inc_config, inc_perf = history.get_incumbents()[0]
    default_perf = history.perfs[0]
    print(f'pipeline ran {n} iterations; default perf {default_perf:.4f}, '
          f'incumbent {inc_perf:.4f}')
    assert inc_perf <= default_perf
    print('ALL PIPELINE WIRING TESTS PASSED')

    if os.path.exists(hist):
        os.remove(hist)


if __name__ == '__main__':
    main()
