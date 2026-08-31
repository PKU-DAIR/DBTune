# License: MIT
# LlamaTune (Kanellis et al., VLDB'22): run BO in a low-dimensional HeSBO random
# projection of the knob space and project suggestions up to the full space.
# Ported natively from uw-mad-dash/llamatune (MIT); special-value biased sampling
# (paper Section 4.1) is omitted because DBTune knob JSONs carry no special-value
# metadata. Quantization of the low space (Section 4.2) uses ConfigSpace's `q`.

import numpy as np

from autotune.utils.config_space import ConfigurationSpace, Configuration
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter
from autotune.utils.history_container import HistoryContainer
from autotune.utils.logging_utils import get_logger
from autotune.optimizer.bo_optimizer import BO_Optimizer


def unit_to_knob_value(hp, u):
    """Map u in [0, 1] onto a hyperparameter's value range (LlamaTune's binning).

    Integers: linear map + round, clamped (float64 rounding can overshoot huge ranges).
    Categoricals: equal-width buckets over the choices.
    Shared by HesBOProjection.unproject and Sampler_Optimizer so the sweep baseline
    tests exactly the values LlamaTune can propose.
    """
    if isinstance(hp, UniformIntegerHyperparameter):
        v = int(round(hp.lower + u * (hp.upper - hp.lower)))
        return min(max(v, hp.lower), hp.upper)
    if isinstance(hp, UniformFloatHyperparameter):
        return hp.lower + u * (hp.upper - hp.lower)
    if isinstance(hp, CategoricalHyperparameter):
        n = len(hp.choices)
        return hp.choices[min(int(u * n), n - 1)]
    raise TypeError('unsupported hyperparameter type %s' % type(hp))


class HesBOProjection:
    """HeSBO count-sketch embedding between a low-dim box [-1, 1]^d and a knob space.

    Each knob i is assigned a random low-dim coordinate h[i] and a sign sigma[i];
    unprojection computes v_i = sigma[i] * y[h[i]] and maps [-1, 1] linearly onto
    the knob's range (categoricals: equal-width buckets over the choices).
    """

    def __init__(self, target_space, low_dim=16, max_num_values=10000, seed=None):
        assert max_num_values % 2 == 0, 'max_num_values must be even so that -1/0/+1 lie on the q-grid'
        self.target_space = target_space
        self.low_dim = low_dim
        self.q = 2.0 / max_num_values
        self.hps = target_space.get_hyperparameters()
        rng = np.random.RandomState(seed)
        self.h = rng.randint(low_dim, size=len(self.hps))
        self.sigma = rng.choice([-1, 1], size=len(self.hps))

        self.low_space = ConfigurationSpace()
        self.low_space.add_hyperparameters([
            UniformFloatHyperparameter('hesbo_%03d' % j, -1.0, 1.0, q=self.q, default_value=0.0)
            for j in range(low_dim)])
        if seed is not None:
            self.low_space.seed(seed)

    def unproject(self, low_config):
        y = np.array([low_config['hesbo_%03d' % j] for j in range(self.low_dim)])
        values = {}
        for i, hp in enumerate(self.hps):
            u = float(np.clip((self.sigma[i] * y[self.h[i]] + 1.0) / 2.0, 0.0, 1.0))
            values[hp.name] = unit_to_knob_value(hp, u)
        return Configuration(self.target_space, values=values)

    def approx_project(self, full_config):
        """Least-squares inverse of the embedding (per low-dim mean of sigma_i * x_i).

        Used for observations that were not produced by unproject() — e.g. the
        default configuration evaluated at iteration 0.
        """
        arr = full_config.get_array()  # alphabetical hp order; numeric in [0,1], categorical = index
        order = self.target_space.get_hyperparameters()
        sums = np.zeros(self.low_dim)
        counts = np.zeros(self.low_dim)
        for i, hp in enumerate(order):
            if isinstance(hp, CategoricalHyperparameter):
                u = (arr[i] + 0.5) / len(hp.choices)
            else:
                u = arr[i]
            x = 2.0 * u - 1.0
            sums[self.h[i]] += self.sigma[i] * x
            counts[self.h[i]] += 1
        y = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        y = np.clip(y, -1.0, 1.0)
        y = np.round(y / self.q) * self.q  # snap to the quantization grid
        values = {'hesbo_%03d' % j: float(y[j]) for j in range(self.low_dim)}
        return Configuration(self.low_space, values=values)


class LlamaTune_Optimizer:
    """Wraps BO_Optimizer: BO state lives in an internal low-dim HistoryContainer;
    the pipeline-facing interface (full-space Configurations) matches the other
    optimizers. Requires update(observation) to be called after each evaluation.

    Known limitation: PipleLine.load_history refills only the outer container, so
    after a history resume the inner BO restarts its initial design.
    """

    def __init__(self,
                 config_space,
                 task_id='llamatune',
                 surrogate_type='prf',
                 acq_type='ei',
                 acq_optimizer_type='auto',
                 init_strategy='random_explore_first',
                 initial_trials=3,
                 random_state=None,
                 low_dim=16,
                 max_num_values=10000,
                 eval_default_first=True):
        self.logger = get_logger(self.__class__.__name__)
        self.config_space = config_space
        self.eval_default_first = eval_default_first
        self.projection = HesBOProjection(config_space, low_dim=low_dim,
                                          max_num_values=max_num_values, seed=random_state)
        self.inner_history = HistoryContainer(task_id=str(task_id) + '_llamatune_low',
                                              num_constraints=0,
                                              config_space=self.projection.low_space)
        self.inner_optimizer = BO_Optimizer(self.projection.low_space,
                                            self.inner_history,
                                            surrogate_type=surrogate_type,
                                            acq_type=acq_type,
                                            acq_optimizer_type=acq_optimizer_type,
                                            init_strategy=init_strategy,
                                            initial_trials=initial_trials,
                                            random_state=random_state,
                                            num_objs=1,
                                            num_constraints=0)
        self._pending = {}  # full-space Configuration -> low-dim Configuration
        # probed via hasattr by PipleLine.iterate / reset_context
        self.surrogate_model = None
        self.current_context = None
        self.logger.info('LlamaTune: %d knobs -> %d dims, q=%g, surrogate=%s' %
                         (len(config_space.get_hyperparameters()), low_dim,
                          self.projection.q, surrogate_type))

    def get_suggestion(self, history_container, compact_space=None):
        assert compact_space is None, 'LlamaTune does not support space_transfer/compact_space'
        if self.eval_default_first and len(history_container.configurations) == 0:
            self.logger.info('LlamaTune: iteration 0, return default configuration')
            return self.config_space.get_default_configuration()
        low_config = self.inner_optimizer.get_suggestion(self.inner_history)
        full_config = self.projection.unproject(low_config)
        self._pending[full_config] = low_config
        return full_config

    def update(self, observation):
        low_config = self._pending.pop(observation.config, None)
        if low_config is None:
            self.logger.info('LlamaTune: no pending low-dim config for observation '
                             '(default config or resumed history), using approximate projection')
            low_config = self.projection.approx_project(observation.config)
        self.inner_history.update_observation(observation._replace(config=low_config))
