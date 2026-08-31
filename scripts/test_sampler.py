"""Offline tests for Sampler_Optimizer (sweep / lhs / random schedules).

No MySQL needed. Run from scripts/:
    ../venv/bin/python test_sampler.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
import numpy as np

from autotune.utils.history_container import HistoryContainer, Observation
from autotune.utils.constants import SUCCESS
from autotune.optimizer.sampler_optimizer import Sampler_Optimizer
from autotune.optimizer.llamatune_optimizer import unit_to_knob_value
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from test_llamatune import build_space

logging.basicConfig(level=logging.WARNING, format='%(name)s - %(message)s')

KNOB_FILE = os.path.join(os.path.dirname(__file__),
                         'experiment', 'gen_knobs', 'mysql_perf_8.0.json')
SEED = 42
LEVELS = 5


def make(space, strategy, size=None):
    return Sampler_Optimizer(space, strategy=strategy, sweep_levels=LEVELS,
                             size=size, random_state=SEED)


def record(hc, config, perf):
    hc.update_observation(Observation(
        config=config, objs=[perf], constraints=None, trial_state=SUCCESS,
        elapsed_time=1.0, iter_time=1.0, EM=None, resource=None, IM=None,
        info=None, context=None))


def test_default_first(space, schedules):
    default = space.get_default_configuration()
    for strategy, schedule in schedules.items():
        assert schedule[0] == default, '%s: schedule[0] is not the default config' % strategy
    print('PASS default config first (all strategies)')


def test_determinism(space, schedules):
    for strategy, schedule in schedules.items():
        size = None if strategy == 'sweep' else len(schedule)
        again = make(space, strategy, size).schedule
        assert len(again) == len(schedule), strategy
        assert all(a == b for a, b in zip(again, schedule)), \
            '%s: schedule not reproducible from the same seed' % strategy
    print('PASS schedule determinism across re-instantiation')


def test_sweep_structure(space):
    schedule = make(space, 'sweep').schedule
    default = dict(space.get_default_configuration())
    per_knob = {}
    for config in schedule[1:]:
        changed = [(k, v) for k, v in dict(config).items() if v != default[k]]
        assert len(changed) == 1, 'sweep config changes %d knobs' % len(changed)
        name, value = changed[0]
        per_knob.setdefault(name, []).append(value)

    for hp in space.get_hyperparameters():
        values = per_knob.get(hp.name, [])
        if isinstance(hp, CategoricalHyperparameter):
            assert sorted(values) == sorted(c for c in hp.choices if c != default[hp.name]), hp.name
        else:
            # every swept value must be exactly LlamaTune's binning of some level
            expected = []
            for u in np.linspace(0.0, 1.0, LEVELS):
                v = unit_to_knob_value(hp, float(u))
                if v != default[hp.name] and v not in expected:
                    expected.append(v)
            assert values == expected, hp.name
    assert len(per_knob) == len(space.get_hyperparameters()), 'some knob never swept'
    print('PASS sweep structure: 1 knob per config, values = LlamaTune binning, all %d knobs covered'
          % len(per_knob))


def test_lhs_marginals(space, schedules):
    """LHS stratification: each dimension's u values must hit every decile."""
    schedule = schedules['lhs']
    hps = space.get_hyperparameters()
    n = len(schedule) - 1
    checked = 0
    for d, hp in enumerate(hps):
        if isinstance(hp, CategoricalHyperparameter):
            continue
        if hp.upper - hp.lower < 100:
            continue  # narrow ranges: value-snapping quantizes u, deciles can be empty
        checked += 1
        us = [(c[hp.name] - hp.lower) / (hp.upper - hp.lower) for c in schedule[1:]]
        bins = np.histogram(us, bins=10, range=(0.0, 1.0))[0]
        assert (bins > 0).all(), 'lhs: dim %s leaves a decile empty (%s)' % (hp.name, bins)
    print('PASS lhs marginal coverage: %d wide integer dims hit every decile (n=%d)'
          % (checked, n))


def test_resume(space, schedules):
    for strategy, schedule in schedules.items():
        size = None if strategy == 'sweep' else len(schedule)
        hc = HistoryContainer(task_id='test_resume_' + strategy, config_space=space)
        for config in schedule[:4]:
            record(hc, config, 1.0)
        sampler = make(space, strategy, size)
        assert sampler.get_suggestion(hc) == schedule[4], '%s: wrong resume index' % strategy

        # tampered history (different config at iter 2) must be refused
        hc_bad = HistoryContainer(task_id='test_resume_bad_' + strategy, config_space=space)
        record(hc_bad, schedule[0], 1.0)
        record(hc_bad, schedule[3], 1.0)
        try:
            make(space, strategy, size).get_suggestion(hc_bad)
            raise AssertionError('%s: diverged history was not detected' % strategy)
        except RuntimeError:
            pass
    print('PASS resume: continues at len(history), refuses diverged history')


def test_exhaustion(space, schedules):
    schedule = schedules['random']
    hc = HistoryContainer(task_id='test_exhaust', config_space=space)
    sampler = make(space, 'random', len(schedule))
    for config in schedule:
        record(hc, config, 1.0)
    try:
        sampler.get_suggestion(hc)
        raise AssertionError('exhausted schedule did not raise')
    except IndexError:
        pass
    print('PASS exhaustion raises IndexError with budget hint')


def main():
    space, _ = build_space(KNOB_FILE)
    n_hps = len(space.get_hyperparameters())
    sweep = make(space, 'sweep').schedule
    n = len(sweep)
    print('space: %d knobs; sweep budget N = %d (matched for lhs/random)' % (n_hps, n))
    schedules = {
        'sweep': sweep,
        'lhs': make(space, 'lhs', n).schedule,
        'random': make(space, 'random', n).schedule,
    }
    for s, sched in schedules.items():
        assert len(sched) == n, (s, len(sched))

    test_default_first(space, schedules)
    test_determinism(space, schedules)
    test_sweep_structure(space)
    test_lhs_marginals(space, schedules)
    test_resume(space, schedules)
    test_exhaustion(space, schedules)
    print('\nALL TESTS PASSED (N = %d)' % n)


if __name__ == '__main__':
    main()
