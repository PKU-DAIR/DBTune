# License: MIT

import os
import abc
import numpy as np
import sys
import time
import traceback
import math
from typing import List
from collections import OrderedDict
from tqdm import tqdm
from autotune.utils.util_funcs import check_random_state
from autotune.utils.logging_utils import get_logger
from autotune.utils.history_container import HistoryContainer, MOHistoryContainer
from autotune.utils.constants import MAXINT, SUCCESS
from autotune.utils.samplers import SobolSampler, LatinHypercubeSampler
from autotune.utils.multi_objective import get_chebyshev_scalarization, NondominatedPartitioning
from autotune.utils.config_space.util import convert_configurations_to_array
from autotune.utils.history_container import Observation
from autotune.pipleline.base import BOBase
from autotune.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from autotune.utils.limit import time_limit, TimeoutException, no_time_limit_func
from autotune.utils.util_funcs import get_result
from autotune.utils.config_space import ConfigurationSpace
from autotune.selector.selector import SHAPSelector, fANOVASelector, GiniSelector, AblationSelector, LASSOSelector
from autotune.optimizer.surrogate.core import build_surrogate, surrogate_switch
from autotune.optimizer.core import build_acq_func, build_optimizer
import pdb
from autotune.knobs import ts, logger

class PipleLine(BOBase):
    """
    Basic Advisor Class, which adopts a policy to sample a configuration.
    """

    def __init__(self, objective_function: callable,
                 config_space,
                 num_objs,
                 num_constraints=0,
                 optimizer_type='MBO',
                 sample_strategy: str = 'bo',
                 max_runs=200,
                 runtime_limit=None,
                 time_limit_per_trial=180,
                 surrogate_type='auto',
                 acq_type='auto',
                 acq_optimizer_type='auto',
                 initial_runs=3,
                 init_strategy='random_explore_first',
                 initial_configurations=None,
                 ref_point=None,
                 history_bo_data: List[OrderedDict] = None,
                 logging_dir='logs',
                 task_id='default_task_id',
                 random_state=None,
                 selector_type='shap',
                 incremental='decrease',
                 incremental_every=4,
                 incremental_num=1,
                 num_hps_init=5,
                 num_metrics=65,
                 advisor_kwargs: dict = None,
                 **kwargs
                 ):


        super().__init__(objective_function, config_space, task_id=task_id, logging_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         runtime_limit=runtime_limit, sample_strategy=sample_strategy,
                         time_limit_per_trial=time_limit_per_trial, history_bo_data=history_bo_data)

        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.FAILED_PERF = [MAXINT] * self.num_objs

        self.selector_type = selector_type
        self.optimizer_type = optimizer_type
        self.config_space_all = config_space
        self.incremental = incremental  # none, increase, decrease
        self.incremental_every = incremental_every  # how often increment the number of knobs
        self.incremental_num = incremental_num  # how many knobs to increment each time
        self.num_hps_init = num_hps_init
        self.num_hps_max = len(self.config_space_all.get_hyperparameters())
        self.num_metrics = num_metrics
        self.init_selector()
        advisor_kwargs = advisor_kwargs or {}

        # init history container
        if self.num_objs == 1:
            self.history_container = HistoryContainer(task_id=self.task_id,
                                                      num_constraints=self.num_constraints,
                                                      config_space=self.config_space)
        else:
            self.history_container = MOHistoryContainer(task_id=self.task_id,
                                                        num_objs=self.num_objs,
                                                        num_constraints=self.num_constraints,
                                                        config_space=self.config_space,
                                                        ref_point=ref_point)
        # load history container if exists
        self.load_history()

        if optimizer_type == 'MBO' or optimizer_type == 'SMAC':
            from autotune.optimizer.bo_optimizer import BO_Optimizer
            self.optimizer = BO_Optimizer(config_space,
                                          self.history_container,
                                          num_objs=self.num_objs,
                                          num_constraints=num_constraints,
                                          initial_trials=initial_runs,
                                          init_strategy=init_strategy,
                                          initial_configurations=initial_configurations,
                                          surrogate_type=surrogate_type,
                                          acq_type=acq_type,
                                          acq_optimizer_type=acq_optimizer_type,
                                          ref_point=ref_point,
                                          history_bo_data=history_bo_data,
                                          random_state=random_state,
                                          **advisor_kwargs)
        elif optimizer_type == 'TPE':
            assert self.num_objs == 1 and num_constraints == 0

            from autotune.optimizer.tpe_optimizer import TPE_Optimizer
            self.optimizer = TPE_Optimizer(config_space,
                                           **advisor_kwargs)
        elif optimizer_type == 'GA':
            assert self.num_objs == 1 and num_constraints == 0
            assert self.incremental == 'none'

            from autotune.optimizer.ga_optimizer import GA_Optimizer
            self.optimizer = GA_Optimizer(config_space,
                                          self.history_container,
                                          num_objs=self.num_objs,
                                          num_constraints=num_constraints,
                                          optimization_strategy=sample_strategy,
                                          output_dir=logging_dir,
                                          random_state=random_state,
                                          **advisor_kwargs)
        elif optimizer_type == 'TurBO':
            # TODO: assertion? warm start?
            assert self.num_objs == 1 and num_constraints == 0
            assert self.incremental == 'none'

            from autotune.optimizer.turbo_optimizer import TURBO_Optimizer
            self.optimizer = TURBO_Optimizer(config_space,
                                             initial_trials=initial_runs,
                                             init_strategy=init_strategy,
                                             **advisor_kwargs)
        elif optimizer_type == 'DDPG':
            assert self.num_objs == 1 and num_constraints == 0
            assert self.incremental == 'none'

            from autotune.optimizer.ddpg_optimizer import DDPG_Optimizer
            self.optimizer = DDPG_Optimizer(config_space,
                                            self.history_container,
                                            knobs_num=self.num_hps_init,
                                            metrics_num=num_metrics,
                                            task_id=task_id,
                                            params=kwargs['params'],
                                            batch_size=kwargs['batch_size'],
                                            mean_var_file=kwargs['mean_var_file']
                                            )
        else:
            raise ValueError('Invalid advisor type!')

    def get_history(self):
        return self.history_container

    def get_incumbent(self):
        return self.history_container.get_incumbents()

    def init_selector(self):
        if self.selector_type == 'shap':
            self.selector = SHAPSelector()
        elif self.selector_type == 'fanova':
            self.selector = fANOVASelector()
        elif self.selector_type == 'gini':
            self.selector = GiniSelector()
        elif self.selector_type == 'ablation':
            self.selector = AblationSelector()
        elif self.selector_type == 'lasso':
            self.selector = LASSOSelector()

    def run(self):
        for _ in tqdm(range(self.iteration_id, self.max_iterations)):
            if self.budget_left < 0:
                self.logger.info('Time %f elapsed!' % self.runtime_limit)
                break
            start_time = time.time()
            self.iterate()
            self.save_history()
            runtime = time.time() - start_time
            self.budget_left -= runtime

        return self.get_history()

    def knob_selection(self):
        assert self.num_objs == 1

        if self.iteration_id < self.init_num:
            return

        if self.incremental == 'none':
            if self.num_hps_init == len(self.config_space.get_hyperparameter_names()):
                return

            new_config_space, _ = self.selector.knob_selection(
                self.config_space_all, self.history_container, self.num_hps_init)

            if not self.config_space == new_config_space:
                logger.info("new configuration space: {}".format(new_config_space))
                self.history_container.alter_configuration_space(new_config_space)
                self.config_space = new_config_space

        else:
            # rank all knobs after init
            if self.iteration_id == self.init_num:
                _, self.knob_rank = self.selector.knob_selection(
                    self.config_space_all, self.history_container, self.num_hps_max)

            incremental_step = int((self.iteration_id - self.init_num)/self.incremental_every)
            if self.incremental == 'increase':
                num_hps = self.num_hps_init + incremental_step * self.incremental_every
                num_hps = min(num_hps, self.num_hps_max)

                new_config_space = ConfigurationSpace()
                for knob in self.knob_rank[:num_hps]:
                    new_config_space.add_hyperparameter(self.config_space_all[knob])

                if not self.config_space == new_config_space:
                    logger.info("new configuration space: {}".format(new_config_space))
                    self.history_container.alter_configuration_space(new_config_space)
                    self.config_space = new_config_space

            elif self.incremental == 'decrease':
                num_hps = self.num_hps_init - incremental_step * self.incremental_every
                num_hps = max(num_hps, 1)

                new_config_space = ConfigurationSpace()
                for knob in self.knob_rank[:num_hps]:
                    new_config_space.add_hyperparameter(self.config_space_all[knob])

                # fix the knobs that no more to tune
                inc_config = self.history_container.incumbents[0][0]
                self.objective_function(inc_config)

                if not self.config_space == new_config_space:
                    logger.info("new configuration space: {}".format(new_config_space))
                    self.history_container.alter_configuration_space(new_config_space)
                    self.config_space = new_config_space

    def iterate(self):
        self.knob_selection()
        # get configuration suggestion
        config = self.optimizer.get_suggestion(history_container=self.history_container)
        _, trial_state, constraints, objs = self.evaluate(config)

        return config, trial_state, constraints, objs

    def save_history(self):
        dir_path = os.path.join('DBTune_history')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = 'history_%s.json' % self.task_id
        return self.history_container.save_json(os.path.join(dir_path, file_name))

    def load_history(self):
        # TODO: check info
        fn = os.path.join('DBTune_history', 'history_%s.json' % self.task_id)
        if not os.path.exists(fn):
            self.logger.info('Start new DBTune task')
        else:
            self.history_container.load_history_from_json(fn)
            self.iteration_id = len(self.history_container.configurations)
            self.logger.info('Load {} iterations from {}'.format(self.iteration_id, fn))

    def evaluate(self, config):
        trial_state = SUCCESS
        start_time = time.time()

        objs, constraints, em, resource, im, info, trial_state = self.objective_function(config)

        if trial_state == FAILED:
            objs = self.FAILED_PERF

        elapsed_time = time.time() - start_time

        observation = Observation(
            config=config, objs=objs, constraints=constraints,
            trial_state=trial_state, elapsed_time=elapsed_time, EM=em, resource=resource, IM=im, info=info
        )
        self.history_container.update_observation(observation)

        if self.optimizer_type in ['GA', 'TurBO', 'DDPG']:
            self.optimizer.update(observation)

        self.iteration_id += 1
        # Logging.
        if self.num_constraints > 0:
            self.logger.info('Iteration %d, objective value: %s. constraints: %s.'
                             % (self.iteration_id, objs, constraints))
        else:
            self.logger.info('Iteration %d, objective value: %s.' % (self.iteration_id, objs))

        return config, trial_state, constraints, objs



