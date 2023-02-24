# License: MIT

import abc
import pdb

import numpy as np
from typing import List
from collections import OrderedDict
from autotune.utils.util_funcs import check_random_state
from autotune.utils.logging_utils import get_logger
from autotune.utils.history_container import HistoryContainer
from autotune.utils.samplers import SobolSampler, LatinHypercubeSampler
from autotune.utils.multi_objective import get_chebyshev_scalarization, NondominatedPartitioning
from autotune.utils.config_space.util import convert_configurations_to_array, impute_incumb_values, max_min_distance
from autotune.utils.constants import MAXINT
from autotune.optimizer.surrogate.core import build_surrogate
from autotune.optimizer.core import build_acq_func, build_optimizer


class BO_Optimizer(object, metaclass=abc.ABCMeta):

    def __init__(self,
                 config_space,
                 history_container: HistoryContainer,
                 surrogate_type='auto',
                 acq_type='auto',
                 acq_optimizer_type='auto',
                 init_strategy='random_explore_first',
                 initial_configurations=None,
                 ref_point=None,
                 history_bo_data: List[OrderedDict] = None,
                 random_state=None,
                 num_objs=1,
                 num_constraints=0,
                 initial_trials=3,
                 rand_prob=0.1):

        # Create output (logging) directory.
        # Init logging module.
        # Random seed generator.
        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.init_strategy = init_strategy
        self.rng = check_random_state(random_state)
        self.logger = get_logger(self.__class__.__name__)

        # Basic components in Advisor.
        self.rand_prob = rand_prob
        self.optimization_strategy = 'bo'

        # Init the basic ingredients in Bayesian optimization.
        self.history_bo_data = history_bo_data
        self.surrogate_type = surrogate_type
        self.constraint_surrogate_type = None
        self.acq_type = acq_type
        self.acq_optimizer_type = acq_optimizer_type
        self.init_num = initial_trials
        self.config_space = config_space
        self.config_space_seed = self.rng.randint(MAXINT)
        self.config_space.seed(self.config_space_seed)
        self.ref_point = ref_point
        self.current_context = None

        # initial design
        if initial_configurations is not None and len(initial_configurations) > 0:
            self.initial_configurations = initial_configurations
            self.init_num = len(initial_configurations)
        else:
            self.initial_configurations = self.create_initial_design(self.init_strategy, excluded_configs=history_container.configurations)
            self.init_num = len(self.initial_configurations)

        self.surrogate_model = None
        self.constraint_models = None
        self.acquisition_function = None
        self.optimizer = None
        self.auto_alter_model = False
        self.algo_auto_selection()
        self.setup_bo_basics(self.config_space)
        self.check_setup()


    def algo_auto_selection(self):
        from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
            CategoricalHyperparameter, OrdinalHyperparameter
        # analyze config space
        cont_types = (UniformFloatHyperparameter, UniformIntegerHyperparameter)
        cat_types = (CategoricalHyperparameter, OrdinalHyperparameter)
        n_cont_hp, n_cat_hp, n_other_hp = 0, 0, 0
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, cont_types):
                n_cont_hp += 1
            elif isinstance(hp, cat_types):
                n_cat_hp += 1
            else:
                n_other_hp += 1
        n_total_hp = n_cont_hp + n_cat_hp + n_other_hp

        info_str = ''

        if self.surrogate_type == 'auto':
            self.auto_alter_model = True
            if n_total_hp >= 100:
                self.optimization_strategy = 'random'
                self.surrogate_type = 'prf'  # for setup procedure
            elif n_total_hp >= 10:
                self.surrogate_type = 'prf'
            elif n_cat_hp > n_cont_hp:
                self.surrogate_type = 'prf'
            else:
                self.surrogate_type = 'gp'
            info_str += ' surrogate_type: %s.' % self.surrogate_type

        if self.acq_type == 'auto':
            if self.num_objs == 1:  # single objective
                if self.num_constraints == 0:
                    self.acq_type = 'ei'
                else:   # with constraints
                    self.acq_type = 'eic'
            elif self.num_objs <= 4:    # multi objective (<=4)
                if self.num_constraints == 0:
                    self.acq_type = 'ehvi'
                else:   # with constraints
                    self.acq_type = 'ehvic'
            else:   # multi objective (>4)
                if self.num_constraints == 0:
                    self.acq_type = 'mesmo'
                else:   # with constraints
                    self.acq_type = 'mesmoc'
                self.surrogate_type = 'gp_rbf'
                info_str = ' surrogate_type: %s.' % self.surrogate_type
            info_str += ' acq_type: %s.' % self.acq_type

        if self.acq_optimizer_type == 'auto':
            if n_cat_hp + n_other_hp == 0:  # todo: support constant hp in scipy optimizer
                self.acq_optimizer_type = 'random_scipy'
            else:
                self.acq_optimizer_type = 'local_random'
            info_str += ' acq_optimizer_type: %s.' % self.acq_optimizer_type

        if info_str != '':
            info_str = '=== [BO auto selection] ===' + info_str
            self.logger.info(info_str)

    def alter_model(self, history_container):
        if not self.auto_alter_model:
            return

        num_config_evaluated = len(history_container.configurations)
        num_config_successful = len(history_container.successful_perfs)

        if num_config_evaluated == 300:
            if self.surrogate_type == 'gp':
                self.surrogate_type = 'prf'
                self.logger.info('n_observations=300, change surrogate model from GP to PRF!')
                if self.acq_optimizer_type == 'random_scipy':
                    self.acq_optimizer_type = 'local_random'
                    self.logger.info('n_observations=300, change acq optimizer from random_scipy to local_random!')
                self.setup_bo_basics(history_container.config_space)

    def check_setup(self):
        """
        Check optimization_strategy, num_objs, num_constraints, acq_type, surrogate_type.
        Returns
        -------
        None
        """
        assert self.optimization_strategy in ['bo', 'random']
        assert isinstance(self.num_objs, int) and self.num_objs >= 1
        assert isinstance(self.num_constraints, int) and self.num_constraints >= 0

        # single objective
        if self.num_objs == 1:
            if self.num_constraints == 0:
                assert self.acq_type in ['ei', 'eips', 'logei', 'pi', 'lcb', 'lpei', ]
            else:  # with constraints
                assert self.acq_type in ['eic', ]
                if self.constraint_surrogate_type is None:
                    self.constraint_surrogate_type = 'gp'

        # multi-objective
        else:
            if self.num_constraints == 0:
                assert self.acq_type in ['ehvi', 'mesmo', 'usemo', 'parego']
                if self.acq_type == 'mesmo' and self.surrogate_type != 'gp_rbf':
                    self.surrogate_type = 'gp_rbf'
                    self.logger.warning('Surrogate model has changed to Gaussian Process with RBF kernel '
                                        'since MESMO is used. Surrogate_type should be set to \'gp_rbf\'.')
            else:  # with constraints
                assert self.acq_type in ['ehvic', 'mesmoc', 'mesmoc2']
                if self.constraint_surrogate_type is None:
                    if self.acq_type == 'mesmoc':
                        self.constraint_surrogate_type = 'gp_rbf'
                    else:
                        self.constraint_surrogate_type = 'gp'
                if self.acq_type == 'mesmoc' and self.surrogate_type != 'gp_rbf':
                    self.surrogate_type = 'gp_rbf'
                    self.logger.warning('Surrogate model has changed to Gaussian Process with RBF kernel '
                                        'since MESMOC is used. Surrogate_type should be set to \'gp_rbf\'.')
                if self.acq_type == 'mesmoc' and self.constraint_surrogate_type != 'gp_rbf':
                    self.surrogate_type = 'gp_rbf'
                    self.logger.warning('Constraint surrogate model has changed to Gaussian Process with RBF kernel '
                                        'since MESMOC is used. Surrogate_type should be set to \'gp_rbf\'.')

            # Check reference point is provided for EHVI methods
            if 'ehvi' in self.acq_type and self.ref_point is None:
                raise ValueError('Must provide reference point to use EHVI method!')

    def setup_bo_basics(self, config_space: None):
        """
                Prepare the basic BO components.
                Returns
                -------
                An optimizer object.
                """
        if config_space is None:
            config_space = self.config_space
        else:
            self.config_space = config_space

        if self.num_objs == 1 or self.acq_type == 'parego':
            self.surrogate_model = build_surrogate(func_str=self.surrogate_type,
                                                   config_space=config_space,
                                                   rng=self.rng,
                                                   history_hpo_data=self.history_bo_data,
                                                   context=self.current_context
                                                   )
        else:  # multi-objectives
            self.surrogate_model = [build_surrogate(func_str=self.surrogate_type,
                                                    config_space=config_space,
                                                    rng=self.rng,
                                                    history_hpo_data=self.history_bo_data)
                                    for _ in range(self.num_objs)]

        if self.num_constraints > 0:
            self.constraint_models = [build_surrogate(func_str=self.constraint_surrogate_type,
                                                      config_space=config_space,
                                                      rng=self.rng) for _ in range(self.num_constraints)]

        if self.acq_type in ['mesmo', 'mesmoc', 'mesmoc2', 'usemo']:
            self.acquisition_function = build_acq_func(func_str=self.acq_type,
                                                       model=self.surrogate_model,
                                                       constraint_models=self.constraint_models,
                                                       config_space=config_space)
        else:
            self.acquisition_function = build_acq_func(func_str=self.acq_type,
                                                       model=self.surrogate_model,
                                                       constraint_models=self.constraint_models,
                                                       ref_point=self.ref_point)
        if self.acq_type == 'usemo':
            self.acq_optimizer_type = 'usemo_optimizer'
        self.optimizer = build_optimizer(func_str=self.acq_optimizer_type,
                                         acq_func=self.acquisition_function,
                                         config_space=config_space,
                                         rng=self.rng)

    def create_initial_design(self, init_strategy='default', excluded_configs=None):
        default_config = self.config_space.get_default_configuration()
        num_random_config = self.init_num - 1
        if init_strategy == 'random':
            initial_configs = self.sample_random_configs(self.init_num, excluded_configs)
            return initial_configs
        elif init_strategy == 'default':
            initial_configs = [default_config] + self.sample_random_configs(num_random_config, excluded_configs)
            return initial_configs
        elif init_strategy == 'random_explore_first':
            candidate_configs = self.sample_random_configs(100, excluded_configs)
            return max_min_distance(default_config, candidate_configs, num_random_config)
        elif init_strategy == 'sobol':
            sobol = SobolSampler(self.config_space, num_random_config, random_state=self.rng)
            initial_configs = [default_config] + sobol.generate(return_config=True)
            return initial_configs
        elif init_strategy == 'latin_hypercube':
            lhs = LatinHypercubeSampler(self.config_space, num_random_config, criterion='maximin')
            initial_configs = [default_config] + lhs.generate(return_config=True)
            return initial_configs
        else:
            raise ValueError('Unknown initial design strategy: %s.' % init_strategy)



    def get_surrogate(self, history_container: HistoryContainer):
        if not history_container.config_space == self.surrogate_model.config_space:
            self.setup_bo_basics(history_container.config_space)

        X = convert_configurations_to_array(history_container.configurations)
        Y = history_container.get_transformed_perfs()
        cY = history_container.get_transformed_constraint_perfs()

        if self.num_objs == 1:
            if self.surrogate_type.startswith('tlbo_'):
                self.surrogate_model.train(history_container)
            elif self.surrogate_type.startswith('context_'):
                contexts = history_container.get_contexts()
                self.surrogate_model.train(X, Y, contexts= contexts)
            else:
                self.surrogate_model.train(X, Y)
        elif self.acq_type == 'parego':
            weights = self.rng.random_sample(self.num_objs)
            weights = weights / np.sum(weights)
            scalarized_obj = get_chebyshev_scalarization(weights, Y)
            self.surrogate_model.train(X, scalarized_obj(Y))
        else:  # multi-objectives
            for i in range(self.num_objs):
                self.surrogate_model[i].train(X, Y[:, i])

            # train constraint model
        for i in range(self.num_constraints):
            self.constraint_models[i].train(X, cY[:, i])

        return X, Y, cY

    def get_suggestion(self, history_container: HistoryContainer, return_list=False, compact_space=None):
        # if have enough data, get_suggorate
        num_config_evaluated = len(history_container.configurations)
        num_config_successful = len(history_container.successful_perfs)

        if num_config_evaluated < self.init_num:
            return self.initial_configurations[num_config_evaluated]

        X, Y, cY = self.get_surrogate(history_container)

        self.alter_model(history_container)

        if self.optimization_strategy == 'random':
            return self.sample_random_configs(num_configs=1, excluded_configs=history_container.configurations)[0]

        if (not return_list) and self.rng.random() < self.rand_prob:
            self.logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
            return self.sample_random_configs(num_configs=1, excluded_configs=history_container.configurations)[0]

        if self.optimization_strategy == 'bo':
            # update acquisition function
            if self.num_objs == 1:
                incumbent_value = history_container.get_incumbents()[0][1]
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 eta=incumbent_value,
                                                 num_data=num_config_evaluated,
                                                 compact_space=compact_space,
                                                 incumbent = history_container.get_incumbents()[0][0]
                                                 )
            else:  # multi-objectives
                mo_incumbent_value = history_container.get_mo_incumbent_value()
                if self.acq_type == 'parego':
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     eta=scalarized_obj(np.atleast_2d(mo_incumbent_value)),
                                                     num_data=num_config_evaluated,
                                                     compact_space=compact_space,
                                                     incumbent=history_container.get_incumbents()[0][0]
                                                     )
                elif self.acq_type.startswith('ehvi'):
                    partitioning = NondominatedPartitioning(self.num_objs, Y)
                    cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     cell_lower_bounds=cell_bounds[0],
                                                     cell_upper_bounds=cell_bounds[1],
                                                     compact_space=compact_space,
                                                     incumbent=history_container.get_incumbents()[0][0]
                                                     )
                else:
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     constraint_perfs=cY,  # for MESMOC
                                                     eta=mo_incumbent_value,
                                                     num_data=num_config_evaluated,
                                                     X=X, Y=Y,
                                                     compact_space=compact_space,
                                                     incumbent=history_container.get_incumbents()[0][0]
                                                     )


            # optimize acquisition function
            if not compact_space is None:
                self.optimizer.set_compact_space(compact_space)
            challengers = self.optimizer.maximize(runhistory=history_container, num_points=5000)
            if return_list:
                # Caution: return_list doesn't contain random configs sampled according to rand_prob
                return challengers.challengers

            for config in challengers.challengers:
                if config not in history_container.configurations:
                    return config
            self.logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                                'Sample random config.' % (len(challengers.challengers), ))
            return self.sample_random_configs(num_configs=1, excluded_configs=history_container.configurations)[0]
        else:
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)

    def sample_random_configs(self, num_configs=1, excluded_configs=None):
        if excluded_configs is None:
            excluded_configs = set()

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = self.config_space.sample_configuration()
            sample_cnt += 1
            if config not in configs and config not in excluded_configs:
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                self.logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs
