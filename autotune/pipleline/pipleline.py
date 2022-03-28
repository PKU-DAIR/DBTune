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
from openbox.utils.util_funcs import check_random_state
from openbox.utils.logging_utils import get_logger
from autotune.utils.history_container import HistoryContainer, MOHistoryContainer, MultiStartHistoryContainer
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.samplers import SobolSampler, LatinHypercubeSampler
from openbox.utils.multi_objective import get_chebyshev_scalarization, NondominatedPartitioning
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.core.base import Observation
from autotune.pipleline.base import BOBase
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.util_funcs import get_result
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
                 num_objs=1,
                 num_constraints=0,
                 selector_type='shap',
                 incremental='decrease',
                 incremental_step=4,
                 incremental_num=1,
                 num_hps=5,
                 advisor_kwargs: dict = None,
                 **kwargs
                 ):


        self.FAILED_PERF = [MAXINT] * num_objs
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         runtime_limit=runtime_limit, sample_strategy=sample_strategy,
                         time_limit_per_trial=time_limit_per_trial, history_bo_data=history_bo_data)

        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.FAILED_PERF = [MAXINT] * num_objs
        advisor_kwargs = advisor_kwargs or {}
        self.selector_type = selector_type
        self.optimizer_type = optimizer_type
        self.config_space_all = config_space
        self.incremental = incremental  # None, increase, decrease
        self.incremental_step = incremental_step  # how often increment the number of knobs
        self.incremental_num = incremental_num  # how many knobs to increment each time
        self.num_hps = num_hps  # initial number of knobs before incremental tuning
        self.num_hps_max = len(self.config_space_all.get_hyperparameters())
        self.init_selector()

        if optimizer_type == 'MBO' or optimizer_type == 'SMAC':
            from autotune.optimizer.bo_optimizer import BO_Optimizer
            self.optimizer = BO_Optimizer(config_space,
                                          num_objs=num_objs,
                                          num_constraints=num_constraints,
                                          initial_trials=initial_runs,
                                          init_strategy=init_strategy,
                                          initial_configurations=initial_configurations,
                                          surrogate_type=surrogate_type,
                                          acq_type=acq_type,
                                          acq_optimizer_type=acq_optimizer_type,
                                          ref_point=ref_point,
                                          history_bo_data=history_bo_data,
                                          task_id=task_id,
                                          random_state=random_state,
                                          **advisor_kwargs)
        elif optimizer_type == 'TPE':
            from autotune.optimizer.tpe_optimizer import TPE_Optimizer
            assert num_objs == 1 and num_constraints == 0
            self.optimizer = TPE_Optimizer(config_space, task_id=task_id, random_state=random_state,
                                              **advisor_kwargs)
        elif advisor_type == 'ea':
            from openbox.core.ea_advisor import EA_Advisor
            assert num_objs == 1 and num_constraints == 0
            self.config_advisor = EA_Advisor(config_space,
                                             num_objs=num_objs,
                                             num_constraints=num_constraints,
                                             optimization_strategy=sample_strategy,
                                             batch_size=1,
                                             task_id=task_id,
                                             output_dir=logging_dir,
                                             random_state=random_state,
                                             **advisor_kwargs)
        elif advisor_type == 'random':
            from openbox.core.random_advisor import RandomAdvisor
            self.config_advisor = RandomAdvisor(config_space,
                                                num_objs=num_objs,
                                                num_constraints=num_constraints,
                                                initial_trials=initial_runs,
                                                init_strategy=init_strategy,
                                                initial_configurations=initial_configurations,
                                                surrogate_type=surrogate_type,
                                                acq_type=acq_type,
                                                acq_optimizer_type=acq_optimizer_type,
                                                ref_point=ref_point,
                                                history_bo_data=history_bo_data,
                                                task_id=task_id,
                                                output_dir=logging_dir,
                                                random_state=random_state,
                                                **advisor_kwargs)
        else:
            raise ValueError('Invalid advisor type!')


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
            self.iterate(budget_left=self.budget_left)
            runtime = time.time() - start_time
            self.budget_left -= runtime
        return self.get_history()



    def knob_selection(self):
        if self.incremental \
                and self.iteration_id > self.init_num \
                and (self.iteration_id - self.init_num) % self.incremental_step == 0:
            if self.incremental == 'increase':
                self.num_hps = min(self.num_hps + 1, self.num_hps_max)
            elif self.incremental == 'decrease':
                self.num_hps = max(self.num_hps - 1, 1)

            new_config_space = self.selector.knob_selection(self.config_space_all, self.optimizer.history_container, self.num_hps)
            logger.info("new configuration space: {}".format(new_config_space))
            self.optimizer.alter_config_space(new_config_space)
        else:
            if self.iteration_id == self.init_num:
                new_config_space = self.selector.knob_selection(
                    self.config_space_all, self.optimizer.history_container, self.num_hps)
                self.optimizer.alter_config_space(new_config_space)

    def iterate(self, budget_left=None):
        self.knob_selection()
        # get configuration suggestion
        config = self.optimizer.get_suggestion()
        _, trial_state, constraints, objs = self.evaluate(config, budget_left)

        return config, trial_state, constraints, objs



    def save_history(self, dir_path: str = None, file_name: str = None):
        """
        Save history to a json file.
        """
        if dir_path is None:
            dir_path = os.path.join(self.output_dir, 'bo_history')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if file_name is None:
            file_name = 'bo_history_%s.json' % self.task_id
        return self.optimizer.history_container.save_json(os.path.join(dir_path, file_name))

    def load_history_from_json(self, fn=None):
        """
        Load history from a json file.
        """
        if fn is None:
            fn = os.path.join(self.output_dir, 'bo_history', 'bo_history_%s.json' % self.task_id)
        return self.history_container.load_history_from_json(self.config_space, fn)

    def get_suggestions(self):
        raise NotImplementedError


    def evaluate(self, config, budget_left=None):
        trial_state = SUCCESS
        _budget_left = int(1e10) if budget_left is None else budget_left
        _time_limit_per_trial = math.ceil(min(self.time_limit_per_trial, _budget_left))

        # only evaluate non duplicate configuration
        if config not in self.optimizer.history_container.configurations:
            start_time = time.time()
            try:
                # evaluate configuration on objective_function within time_limit_per_trial
                args, kwargs = (config,), dict()
                timeout_status, _result = time_limit(self.objective_function,
                                                     _time_limit_per_trial,
                                                     args=args, kwargs=kwargs)
                if timeout_status:
                    raise TimeoutException(
                        'Timeout: time limit for this evaluation is %.1fs' % _time_limit_per_trial)
                else:
                    # parse result
                    objs, constraints = get_result(_result)
            except Exception as e:
                # parse result of failed trial
                if isinstance(e, TimeoutException):
                    self.logger.warning(str(e))
                    trial_state = TIMEOUT
                else:
                    self.logger.warning('Exception when calling objective function: %s' % str(e))
                    trial_state = FAILED
                objs = self.FAILED_PERF
                constraints = None

            elapsed_time = time.time() - start_time
            # update observation to advisor
            observation = Observation(
                config=config, objs=objs, constraints=constraints,
                trial_state=trial_state, elapsed_time=elapsed_time,
            )
            if _time_limit_per_trial != self.time_limit_per_trial and trial_state == TIMEOUT:
                # Timeout in the last iteration.
                pass
            else:
                self.optimizer.update_observation(observation)
        else:
            self.logger.info('This configuration has been evaluated! Skip it: %s' % config)
            history = self.get_history()
            config_idx = history.configurations.index(config)
            trial_state = history.trial_states[config_idx]
            objs = history.perfs[config_idx]
            constraints = history.constraint_perfs[config_idx] if self.num_constraints > 0 else None
            if self.num_objs == 1:
                objs = (objs,)

        self.iteration_id += 1
        # Logging.
        if self.num_constraints > 0:
            self.logger.info('Iteration %d, objective value: %s. constraints: %s.'
                             % (self.iteration_id, objs, constraints))
        else:
            self.logger.info('Iteration %d, objective value: %s.' % (self.iteration_id, objs))

        # Visualization.
        # for idx, obj in enumerate(objs):
        #     if obj < self.FAILED_PERF[idx]:
        #         self.writer.add_scalar('data/objective-%d' % (idx + 1), obj, self.iteration_id)
        return config, trial_state, constraints, objs



