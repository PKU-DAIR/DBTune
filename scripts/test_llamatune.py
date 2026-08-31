"""Offline test for LlamaTune_Optimizer over the full validated MySQL 8.0 knob space.

No MySQL needed: drives suggest/update cycles with a synthetic objective, mirroring
exactly what PipleLine.iterate/evaluate do. Run from scripts/:
    python test_llamatune.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
import numpy as np
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter
from autotune.utils.config_space import ConfigurationSpace, Configuration
from autotune.utils.history_container import HistoryContainer, Observation
from autotune.utils.constants import SUCCESS, FAILED, MAXINT
from autotune.knobs import initialize_knobs
from autotune.optimizer.llamatune_optimizer import LlamaTune_Optimizer

logging.basicConfig(level=logging.WARNING, format='%(name)s - %(message)s')

KNOB_FILE = os.path.join(os.path.dirname(__file__),
                         'experiment', 'gen_knobs', 'mysql_all_8044_validated.json')


def build_space(knob_file, knob_num=-1):
    """Same loop as DBTuner.setup_configuration_space (autotune/tuner.py)."""
    knobs = initialize_knobs(knob_file, knob_num)
    cs = ConfigurationSpace()
    for name, value in knobs.items():
        if value['type'] == 'enum':
            hp = CategoricalHyperparameter(name, [str(i) for i in value['enum_values']],
                                           default_value=str(value['default']))
        elif value['type'] == 'integer':
            if value['max'] > sys.maxsize:  # same /1000 scaling as the tuner
                hp = UniformIntegerHyperparameter(name, int(value['min'] / 1000),
                                                  int(value['max'] / 1000),
                                                  default_value=int(value['default'] / 1000))
            else:
                hp = UniformIntegerHyperparameter(name, value['min'], value['max'],
                                                  default_value=value['default'])
        elif value['type'] in ('float', 'real'):
            hp = UniformFloatHyperparameter(name, value['min'], value['max'],
                                            default_value=value['default'])
        else:
            raise ValueError(value['type'])
        cs.add_hyperparameter(hp)
    return cs, knobs


def synthetic_objective(config):
    """Quadratic over two integer knobs + an enum penalty; minimized."""
    bp = config['innodb_buffer_pool_size']
    bp_hp = config.configuration_space.get_hyperparameter('innodb_buffer_pool_size')
    u_bp = (bp - bp_hp.lower) / (bp_hp.upper - bp_hp.lower)
    bl = config['back_log']
    bl_hp = config.configuration_space.get_hyperparameter('back_log')
    u_bl = (bl - bl_hp.lower) / (bl_hp.upper - bl_hp.lower)
    enum_pen = 0.0 if config['innodb_flush_log_at_trx_commit'] == '0' else 0.3
    return (u_bp - 0.8) ** 2 + (u_bl - 0.4) ** 2 + enum_pen


def make_obs(config, perf, trial_state=SUCCESS):
    return Observation(config=config, trial_state=trial_state, constraints=None,
                       objs=[perf], elapsed_time=1.0, iter_time=1.0,
                       EM={}, IM=[0.0] * 5, resource={},
                       info={'objs': ['synthetic'], 'constraints': []}, context=None)


def check_full_config(config, cs, knobs):
    assert config.configuration_space == cs
    vals = config.get_dictionary()
    assert set(vals.keys()) == set(knobs.keys()), 'config must assign every knob'
    for name, spec in knobs.items():
        v = vals[name]
        if spec['type'] == 'integer':
            assert isinstance(v, (int, np.integer)), (name, type(v))
            lo, hi = spec['min'], spec['max']
            if hi > sys.maxsize:
                lo, hi = int(lo / 1000), int(hi / 1000)
            assert lo <= v <= hi, (name, v)
        elif spec['type'] == 'enum':
            assert v in [str(i) for i in spec['enum_values']], (name, v)


def main():
    cs, knobs = build_space(KNOB_FILE, -1)
    n_knobs = len(knobs)
    print(f'knob space: {n_knobs} knobs '
          f'({sum(1 for k in knobs.values() if k["type"] == "integer")} int, '
          f'{sum(1 for k in knobs.values() if k["type"] == "enum")} enum)')
    assert n_knobs == 186

    opt = LlamaTune_Optimizer(cs, task_id='test', surrogate_type='prf',
                              initial_trials=5, random_state=42,
                              low_dim=8, max_num_values=100)
    hc = HistoryContainer('llamatune_test', config_space=cs)
    q = opt.projection.q

    perfs = []
    for it in range(25):
        cfg = opt.get_suggestion(history_container=hc)
        check_full_config(cfg, cs, knobs)

        if it == 0:
            assert cfg == cs.get_default_configuration(), 'iteration 0 must evaluate the default config'
            assert len(opt._pending) == 0, 'default config must not have a pending low-dim entry'
        else:
            low = opt._pending[cfg]
            for v in low.get_dictionary().values():
                assert abs(round(v / q) * q - v) < 1e-12, 'low-dim value off the q-grid'

        if it == 12:  # inject one failed trial
            obs = make_obs(cfg, MAXINT, trial_state=FAILED)
        else:
            perf = synthetic_objective(cfg)
            perfs.append(perf)
            obs = make_obs(cfg, perf)
        hc.update_observation(obs)
        opt.update(obs)
        assert len(opt.inner_history.configurations) == len(hc.configurations)

    assert len(opt._pending) == 0, 'pending map must be self-cleaning'
    assert min(perfs[10:]) < min(perfs[:5]), \
        f'incumbent should improve: early best {min(perfs[:5]):.4f}, late best {min(perfs[10:]):.4f}'
    print(f'25 iterations OK (1 FAILED injected); best perf {min(perfs):.4f} '
          f'(default config perf {perfs[0]:.4f})')

    # determinism: same seed -> same projection and same first BO suggestion
    opt2 = LlamaTune_Optimizer(cs, task_id='test2', surrogate_type='prf',
                               initial_trials=5, random_state=42,
                               low_dim=8, max_num_values=100)
    assert np.array_equal(opt.projection.h, opt2.projection.h)
    assert np.array_equal(opt.projection.sigma, opt2.projection.sigma)
    hc2 = HistoryContainer('llamatune_test2', config_space=cs)
    cfg0 = opt2.get_suggestion(history_container=hc2)
    obs0 = make_obs(cfg0, synthetic_objective(cfg0))
    hc2.update_observation(obs0)
    opt2.update(obs0)
    cfg1 = opt2.get_suggestion(history_container=hc2)

    opt3 = LlamaTune_Optimizer(cs, task_id='test3', surrogate_type='prf',
                               initial_trials=5, random_state=42,
                               low_dim=8, max_num_values=100)
    hc3 = HistoryContainer('llamatune_test3', config_space=cs)
    cfg0b = opt3.get_suggestion(history_container=hc3)
    assert cfg0b == cfg0
    obs0b = make_obs(cfg0b, synthetic_objective(cfg0b))
    hc3.update_observation(obs0b)
    opt3.update(obs0b)
    assert opt3.get_suggestion(history_container=hc3) == cfg1, 'same seed must reproduce suggestions'
    print('determinism OK (same seed -> identical projection and suggestions)')

    # unproject must cover the corners: synthetic low-dim configs at the bounds
    for y in (-1.0, 0.0, 1.0):
        low = Configuration(opt.projection.low_space,
                            values={f'hesbo_{j:03d}': y for j in range(8)})
        check_full_config(opt.projection.unproject(low), cs, knobs)
    print('bound unprojection OK')
    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
