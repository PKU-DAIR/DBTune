# License: MIT

import os
import abc
import numpy as np
import sys
import time
import traceback
import math
import random
from typing import List
from collections import OrderedDict, defaultdict
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
from autotune.selector.selector import KnobSelector
from autotune.optimizer.surrogate.core import build_surrogate, surrogate_switch
from autotune.optimizer.core import build_acq_func, build_optimizer
from autotune.transfer.tlbo.rgpe import RGPE
from autotune.utils.util_funcs import check_random_state
from autotune.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter

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
                 space_transfer=False,
                 advisor_kwargs: dict = None,
                 **kwargs
                 ):


        super().__init__(objective_function, config_space, task_id=task_id, logging_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         runtime_limit=runtime_limit, sample_strategy=sample_strategy,
                         time_limit_per_trial=time_limit_per_trial, surrogate_type=surrogate_type, history_bo_data=history_bo_data)

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
        self.selector = KnobSelector(self.selector_type)
        self.current_context = None
        self.space_transfer = space_transfer
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

        if optimizer_type in ('MBO', 'SMAC', 'auto'):
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
            if self.num_hps_init == -1 or self.num_hps_init == len(self.config_space.get_hyperparameter_names()):
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
                    new_config_space.add_hyperparameter(self.config_space_all.get_hyperparameter(knob) )

                if not self.config_space == new_config_space:
                    logger.info("new configuration space: {}".format(new_config_space))
                    self.history_container.alter_configuration_space(new_config_space)
                    self.config_space = new_config_space

            elif self.incremental == 'decrease':
                num_hps = self.num_hps_init - incremental_step * self.incremental_every
                num_hps = max(num_hps, 1)

                new_config_space = ConfigurationSpace()
                for knob in self.knob_rank[:num_hps]:
                    new_config_space.add_hyperparameter(self.config_space_all.get_hyperparameter(knob))

                # fix the knobs that no more to tune
                inc_config = self.history_container.incumbents[0][0]
                self.objective_function(inc_config)

                if not self.config_space == new_config_space:
                    logger.info("new configuration space: {}".format(new_config_space))
                    self.history_container.alter_configuration_space(new_config_space)
                    self.config_space = new_config_space

    def iterate(self):
        self.knob_selection()
        compact_space = None
        if self.space_transfer and self.iteration_id > 1 and self.iteration_id % 3 == 0:
            compact_space = self.get_compact_space()
        # get configuration suggestion
        config = self.optimizer.get_suggestion(history_container=self.history_container, compact_space=compact_space)
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

    def reset_context(self, context):
        self.current_context = context
        self.optimizer.current_context = context
        if self.optimizer.surrogate_model:
            self.optimizer.surrogate_model.current_context =  context

    def evaluate(self, config):
        trial_state = SUCCESS
        start_time = time.time()

        objs, constraints, em, resource, im, info, trial_state = self.objective_function(config)

        if trial_state == FAILED :
            objs = self.FAILED_PERF

        elapsed_time = time.time() - start_time

        if self.surrogate_type == 'context_prf' and config == self.history_container.config_space.get_default_configuration():
            self.reset_context(np.array(im))

        observation = Observation(
            config=config, objs=objs, constraints=constraints,
            trial_state=trial_state, elapsed_time=elapsed_time, EM=em, resource=resource, IM=im, info=info, context=self.current_context
        )
        self.history_container.update_observation(observation)

        if self.optimizer_type in ['GA', 'TurBO', 'DDPG']:
            self.optimizer.update(observation)

        self.iteration_id += 1
        # Logging.
        if self.num_constraints > 0:
            self.logger.info('Iteration %d, objective value: %s, constraints: %s.'
                             % (self.iteration_id, objs, constraints))
        else:
            #self.logger.info('Iteration %d, objective value: %s ,improvement,: :.2%' % (self.iteration_id, objs, (objs-self.default_obj))/self.default_obj)
            self.logger.info('Iteration %d, objective value: %s.' % (self.iteration_id, objs))
        return config, trial_state, constraints, objs


    def get_compact_space(self):
        if not hasattr(self, 'rgpe'):
            rng = check_random_state(100)
            seed = rng.randint(MAXINT)
            self.rgpe = RGPE(self.config_space, self.history_bo_data, seed, num_src_hpo_trial=-1, only_source=True)

        rank_loss_list = self.rgpe.get_ranking_loss(self.history_container)[:-1]
        similarity_list = [1 - i for i in rank_loss_list]

        ### filter unsimilar task
        sample_list = list()
        similarity_threhold = np.quantile(similarity_list, 0.5)
        candidate_list, weight = list(), list()
        for i in range(len(self.history_bo_data)):
            if similarity_list[i] > similarity_threhold:
                candidate_list.append(i)
                weight.append(similarity_list[i] / sum(similarity_list))

        if not len(candidate_list):
            self.logger.info("Remain the space:{}".format(self.config_space))
            return self.config_space

        ### determine the number of sampled task
        if len(self.history_bo_data) > 60:
            k = int(len(self.history_bo_data) / 15)
        else:
            k = 4

        k = min(k, len(candidate_list))

        # sample k task s
        for j in range(k):
            item = random.choices(candidate_list, weights=weight, k=1)[0]
            sample_list.append(item)
            del (weight[candidate_list.index(item)])
            candidate_list.remove(item)

        # obtain the pruned space for the sampled tasks
        important_dict = defaultdict(int)
        pruned_space_list = list()
        quantile_min = 1 / 1e9
        quantile_max = 1 - 1 / 1e9
        for j in range(len(self.history_bo_data)):
            if not j in sample_list:
                continue
            quantile = quantile_max - (1 - 2 * max(similarity_list[j] - 0.5, 0)) * (quantile_max - quantile_min)
            ys_source = - self.history_bo_data[j].get_transformed_perfs()
            performance_threshold = np.quantile(ys_source, quantile)
            default_performance = - self.history_bo_data[j].get_deafult_performance()
            self.logger.info("[{}] similarity:{} default:{}, quantile:{}, threshold:{}".format(self.history_bo_data[j].task_id, similarity_list[j], default_performance, quantile, performance_threshold))
            if performance_threshold < default_performance:
                quantile = 0

            pruned_space = self.history_bo_data[j].get_promising_space(quantile)
            pruned_space_list.append(pruned_space)
            total_imporve = sum([pruned_space[key][2] for key in list(pruned_space.keys())])
            for key in pruned_space.keys():
                if not pruned_space[key][0] == pruned_space[key][1]:
                    if pruned_space[key][2] > 0.01 or pruned_space[key][2] > 0.1 * total_imporve:
                        # print((key,pruned_space[key] ))
                        important_dict[key] = important_dict[key] + similarity_list[j] / sum(
                            [similarity_list[i] for i in sample_list])

        # remove unimportant knobs
        important_knobs = list()
        for key in important_dict.keys():
            if important_dict[key] >= 1/3:
                important_knobs.append(key)

        # generate target pruned space
        default_array = self.config_space.get_default_configuration().get_array()
        default_knobs = self.config_space.get_hyperparameter_names()
        target_space = ConfigurationSpace()
        for knob in important_knobs:
            # CategoricalHyperparameter
            if isinstance(self.config_space.get_hyperparameters_dict()[knob], CategoricalHyperparameter):
                values_dict = defaultdict(int)
                for space in pruned_space_list:
                    values = space[knob][0]
                    for v in values:
                        values_dict[v] += similarity_list[sample_list[pruned_space_list.index(space)]] / sum(
                            [similarity_list[t] for t in sample_list])

                feasible_value = list()
                for v in values_dict.keys():
                    if values_dict[v] > 1/3:
                        feasible_value.append(v)

                default = self.config_space.get_default_configuration()[knob]
                if not default in feasible_value:
                    default = feasible_value[0]

                knob_add = CategoricalHyperparameter(knob, feasible_value, default_value=default)
                target_space.add_hyperparameter(knob_add)
                continue

            # Integer
            index_list = set()
            for space in pruned_space_list:
                info = space[knob]
                if not info[0] == info[1]:
                    index_list.add(info[0])
                    index_list.add(info[1])
            index_list = sorted(index_list)
            count_array = np.array([index_list[:-1], index_list[1:]]).T
            count_array = np.hstack((count_array, np.zeros((count_array.shape[0], 1))))
            for space in pruned_space_list:
                info = space[knob]
                if not info[0] == info[1]:
                    for i in range(count_array.shape[0]):
                        if count_array[i][0] >= info[0] and count_array[i][1] <= info[1]:
                            count_array[i][2] += similarity_list[sample_list[pruned_space_list.index(space)]] / sum(
                                [similarity_list[t] for t in sample_list])

            max_index, min_index = 0, 1
            # vote
            for i in range(count_array.shape[0]):
                if count_array[i][2] > 1/3 :
                    if count_array[i][0] < min_index:
                        min_index = count_array[i][0]
                    if count_array[i][1] > max_index:
                        max_index = count_array[i][1]

            if max_index == 0 and min_index == 1:
                continue
            default = default_array[default_knobs.index(knob)]
            if default < min_index:
                default = min_index
            if default > max_index:
                default = max_index
            transform = self.config_space.get_hyperparameters_dict()[knob]._transform
            knob_add = UniformIntegerHyperparameter(knob, transform(min_index), transform(max_index),
                                                    transform(default))
            target_space.add_hyperparameter(knob_add)

        print(target_space)
        return target_space