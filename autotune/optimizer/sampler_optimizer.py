# License: MIT
# Non-adaptive sample-collection "optimizer": plays back a deterministic, precomputed
# schedule of configurations (1-knob sweep / LHS / uniform random) through the normal
# PipleLine evaluate/save/resume machinery. Used to collect source datasets for
# transfer-learning warm-start experiments — it does not optimize anything.
#
# All three strategies start with the default configuration (iteration 0) because the
# OtterTune workload mapping matches source/target on exact config overlap and every
# tuning session evaluates the default first. Sweep values are generated with the same
# binning as LlamaTune (unit_to_knob_value), so the sweep baseline tests exactly the
# values LlamaTune can propose.

import json
import os

import numpy as np
from skopt.sampler import Lhs

from ConfigSpace.hyperparameters import CategoricalHyperparameter
from autotune.utils.config_space import Configuration
from autotune.utils.util_funcs import check_random_state
from autotune.utils.logging_utils import get_logger
from autotune.optimizer.llamatune_optimizer import unit_to_knob_value


class Sampler_Optimizer:
    """get_suggestion() returns schedule[len(history)], so resume after a crash or a
    PBS walltime kill works as long as the schedule is regenerated identically:
    it is a pure function of (config_space, strategy, sweep_levels, size, random_state).
    A prefix check on the first call after resume turns any drift (changed knob file,
    seed, or library version) into a loud error instead of silent dataset corruption.
    """

    def __init__(self, config_space, strategy='sweep', sweep_levels=5, size=None,
                 random_state=None, task_id=None, replay_file=None):
        self.logger = get_logger(self.__class__.__name__)
        self.config_space = config_space
        self.strategy = strategy
        self.task_id = task_id
        self._prefix_checked = False
        # probed via hasattr by PipleLine.iterate / reset_context
        self.surrogate_model = None
        self.current_context = None

        if strategy == 'sweep':
            self.schedule = self._sweep_schedule(sweep_levels)
        elif strategy == 'lhs':
            self.schedule = self._lhs_schedule(size, random_state)
        elif strategy == 'random':
            self.schedule = self._random_schedule(size, random_state)
        elif strategy == 'replay':
            self.schedule = self._replay_schedule(replay_file)
        else:
            raise ValueError('Sampler: unknown sampler_method %r '
                             '(expected sweep | lhs | random | replay)' % strategy)
        self.logger.info('Sampler[%s]: schedule of %d configurations'
                         % (strategy, len(self.schedule)))
        self._write_manifest()

    def _default_values(self):
        return dict(self.config_space.get_default_configuration())

    def _sweep_schedule(self, levels):
        """[default] + one config per (knob, non-default level), all other knobs at
        default. Integer levels are u = linspace(0, 1, levels) mapped through
        LlamaTune's binning, deduplicated; enums sweep every non-default choice."""
        default = self._default_values()
        schedule = [self.config_space.get_default_configuration()]
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, CategoricalHyperparameter):
                values = [c for c in hp.choices if c != default[hp.name]]
            else:
                values, seen = [], {default[hp.name]}
                for u in np.linspace(0.0, 1.0, levels):
                    v = unit_to_knob_value(hp, float(u))
                    if v not in seen:
                        seen.add(v)
                        values.append(v)
            for v in values:
                knob_values = dict(default)
                knob_values[hp.name] = v
                schedule.append(Configuration(self.config_space, values=knob_values))
        return schedule

    def _lhs_schedule(self, size, random_state):
        """[default] + (size-1) maximin LHS points in [0,1]^D, mapped per dimension
        through LlamaTune's binning. The repo's LatinHypercubeSampler cannot be used:
        it rejects CategoricalHyperparameter and builds Configuration(vector=...)."""
        assert size and size > 1, 'Sampler[lhs]: max_runs must be set (budget N)'
        hps = self.config_space.get_hyperparameters()
        rng = check_random_state(random_state)
        lhs = Lhs(criterion='maximin', iterations=10000)
        X = np.asarray(lhs.generate([(0.0, 1.0)] * len(hps), size - 1, random_state=rng))
        schedule = [self.config_space.get_default_configuration()]
        for x in X:
            values = {hp.name: unit_to_knob_value(hp, float(u)) for hp, u in zip(hps, x)}
            schedule.append(Configuration(self.config_space, values=values))
        return schedule

    def _replay_schedule(self, replay_file):
        """[default] + an explicit list of configurations from a JSON file
        ({'configs': [ {knob: value, ...}, ... ]}). Used by the transplant
        evaluation: re-evaluate a source cell's top configs on the target."""
        assert replay_file, 'Sampler[replay]: replay_file must be set in the ini'
        with open(replay_file) as f:
            payload = json.load(f)
        schedule = [self.config_space.get_default_configuration()]
        for values in payload['configs']:
            # json stringifies enum values and may carry ints as floats
            typed = {}
            for hp in self.config_space.get_hyperparameters():
                if hp.name not in values:
                    continue
                v = values[hp.name]
                typed[hp.name] = str(v) if isinstance(hp, CategoricalHyperparameter) else int(v)
            schedule.append(Configuration(self.config_space, values=typed))
        self.logger.info('Sampler[replay]: %d configs from %s'
                         % (len(schedule) - 1, replay_file))
        return schedule

    def _random_schedule(self, size, random_state):
        assert size and size > 1, 'Sampler[random]: max_runs must be set (budget N)'
        self.config_space.seed(check_random_state(random_state).randint(10000))
        configs = self.config_space.sample_configuration(size=size - 1)
        if not isinstance(configs, list):  # size=1 returns a bare Configuration
            configs = [configs]
        return [self.config_space.get_default_configuration()] + configs

    def _write_manifest(self):
        """One-time human-readable schedule dump next to the history json."""
        if not self.task_id:
            return
        fn = os.path.join('DBTune_history', 'schedule_%s.json' % self.task_id)
        if os.path.exists(fn):
            return
        try:
            os.makedirs('DBTune_history', exist_ok=True)
            default = self._default_values()
            entries = []
            for i, config in enumerate(self.schedule):
                entry = {'index': i}
                diff = {k: v for k, v in dict(config).items() if v != default.get(k)}
                if self.strategy == 'sweep':
                    entry['changed'] = diff  # at most one knob; {} = default config
                else:
                    entry['config'] = dict(config)
                entries.append(entry)
            with open(fn, 'w') as f:
                json.dump({'strategy': self.strategy, 'size': len(self.schedule),
                           'entries': entries}, f, indent=2, default=str)
            self.logger.info('Sampler: wrote schedule manifest %s' % fn)
        except Exception as e:  # manifest is diagnostics only — never kill the run
            self.logger.warning('Sampler: could not write manifest: %s' % e)

    def _check_resume_prefix(self, history_container, idx):
        mismatch = [i for i in range(idx)
                    if history_container.configurations[i] != self.schedule[i]]
        if mismatch:
            raise RuntimeError(
                'Sampler[%s]: resumed history diverges from the regenerated schedule at '
                'iterations %s — knob file, rand_seed, sampler params or library versions '
                'changed since the original run. Refusing to continue (would corrupt the '
                'dataset).' % (self.strategy, mismatch[:5]))
        if idx:
            self.logger.info('Sampler: resume prefix of %d configurations verified' % idx)

    def get_suggestion(self, history_container, compact_space=None):
        assert compact_space is None, 'Sampler does not support space_transfer/compact_space'
        idx = len(history_container.configurations)
        if not self._prefix_checked:
            self._check_resume_prefix(history_container, idx)
            self._prefix_checked = True
        if idx >= len(self.schedule):
            raise IndexError(
                'Sampler[%s]: schedule exhausted (%d configurations evaluated). '
                'Set max_runs <= %d for this strategy/space.'
                % (self.strategy, idx, len(self.schedule)))
        return self.schedule[idx]
