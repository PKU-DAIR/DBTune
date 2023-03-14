import logging
import numpy as np
import os
import math
import pickle
import pdb
import copy
from autotune.optimizer.surrogate.ddpg.ddpg import DDPG
from autotune.utils.history_container import Observation, HistoryContainer
from autotune.utils.config_space import Configuration, CategoricalHyperparameter
from autotune.utils.config_space.util import configs2space, max_min_distance
from autotune.utils.samplers import SobolSampler, LatinHypercubeSampler


def create_output_folders():
    output_folders = ['ddpg', 'ddpg/save_memory', 'ddpg/model_params']
    for folder in output_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)


def config2action(config, config_space):
    action = copy.deepcopy(config.get_array())
    keys = config_space.get_hyperparameter_names()
    for key in keys:
        if type(config_space.get_hyperparameters_dict()[key]) == CategoricalHyperparameter:
            action[keys.index(key)] = action[keys.index(key)] / (config_space.get_hyperparameters_dict()[key].num_choices - 1)

    return action

def action2config(action, config_space):
    keys = config_space.get_hyperparameter_names()
    knob_dict = dict()
    for i, key in enumerate(keys):
        transform = config_space.get_hyperparameters_dict()[key]._transform
        if type(config_space.get_hyperparameters_dict()[key]) == CategoricalHyperparameter:
            action[i] = np.round( action[i] * (config_space.get_hyperparameters_dict()[key].num_choices - 1))
        knob_dict[key] = transform(action[i])
        if not type(config_space.get_hyperparameters_dict()[key]) == CategoricalHyperparameter:
            if knob_dict[key] < config_space.get_hyperparameters_dict()[key].lower:
                knob_dict[key] = config_space.get_hyperparameters_dict()[key].lower
            if knob_dict[key] > config_space.get_hyperparameters_dict()[key].upper:
                knob_dict[key] = config_space.get_hyperparameters_dict()[key].upper
    return Configuration(config_space, values=knob_dict)


class DDPG_Optimizer:
    def __init__(self, config_space,
                 history_container: HistoryContainer,
                 metrics_num,
                 task_id,
                 initial_trials=5,
                 init_strategy="random_explore_first",
                 mean_var_file='',
                 batch_size=16,
                 params=''):

        self.task_id = task_id
        self.config_space = config_space
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_num = metrics_num
        self.batch_size = batch_size
        self.params = params
        self.init_num = initial_trials
        self.initial_configurations = self.create_initial_design(init_strategy, excluded_configs=history_container.configurations)

        self.mean_var_file = mean_var_file
        self.internal_metrics = []
        self.state_mean = None
        self.state_var = None
        self.model = None
        self.init_step = 0
        self.episode = 0
        self.global_t = 0
        self.t = 0
        self.score = 0
        self.episode_init = True
        create_output_folders()
        self.initialize(history_container)

    def initialize(self, history_container):
        if self.mean_var_file != '' and os.path.exists(self.mean_var_file):
            with open(self.mean_var_file, 'rb') as f:
                self.state_mean = pickle.load(f)
                self.state_var = pickle.load(f)
                self.logger.info('Load state mean and var.')
        else:
            for im in history_container.internal_metrics:
                self.internal_metrics.append(im)
            if len(self.internal_metrics) >= self.init_num:
                self.gen_mean_var()

        self.init_num = len(history_container.configurations)
        if self.create_model():
            configurations = configs2space(history_container.configurations, self.config_space)
            for i in range(len(configurations)):
                objs = [history_container.perfs[i]]

                observation = Observation(config=configurations[i],
                                      objs=objs,
                                      constraints=history_container.constraint_perfs,
                                      trial_state=history_container.trial_states[i],
                                      elapsed_time=history_container.elapsed_times[i],
                                      iter_time=history_container.iter_times[i],
                                      EM=history_container.external_metrics[i],
                                      resource=history_container.resource[i],
                                      IM=history_container.internal_metrics[i],
                                      info=history_container.info, context=history_container.contexts[i])
                try:
                    self.update(observation)
                except:
                    pass


    def create_model(self):
        if (self.state_mean is None) or (self.state_var is None) :
            return False

        ddpg_opt = {
            'tau': 0.002,
            'alr': 0.001,
            'clr': 0.001,
            'gamma': 0.9,
            'memory_size': 100000,
            'batch_size': self.batch_size,
            'model': self.params,
        }

        self.model = DDPG(n_states=self.metrics_num,
                          n_actions=len(self.config_space.get_hyperparameter_names()),
                          opt=ddpg_opt,
                          ouprocess=True,
                          mean=self.state_mean,
                          var=self.state_var)

        return True

    def gen_mean_var(self):
        r = list()
        for im in self.internal_metrics:
            if len(im) == 65:
                r.append(im)
        r = np.array(r)
        self.state_mean = r.mean(axis=0)
        self.state_var = r.var(axis=0)

        if self.mean_var_file == '':
            self.mean_var_file = '{}_mean_var.pkl'.format(self.task_id)

        with open(self.mean_var_file, 'wb') as f:
            pickle.dump(self.state_mean, f)
            pickle.dump(self.state_var, f)

        self.logger.info('Calculate state mean and var.')

    def get_suggestion(self, history_container=None, compact_space=None):
        if self.model is None:
            init_config = self.initial_configurations[self.init_step]
            self.init_step += 1
            return init_config

        if self.episode_init:
            self.t = 0
            self.score = 0
            return self.config_space.get_default_configuration()

        if np.random.random() < 0.7:  # avoid too nus reward in the fisrt 100 step
            X_next = self.model.choose_action(self.state, 1 / (self.global_t + 1))
        else:
            X_next = self.model.choose_action(self.state, 1)

        return action2config(X_next, self.config_space)

    def update(self, observation: Observation):
        if self.model is None:
            self.internal_metrics.append(observation.IM)
            if len(self.internal_metrics) >= self.init_num:
                self.gen_mean_var()
                self.create_model()
                self.logger.info('Iteration {} create model'.format(self.init_step))
            return

        if self.episode_init:
            self.state = observation.IM
            self.default_external_metrics = observation.objs[0]
            self.last_external_metrics = observation.objs[0]

            self.episode_init = False
            self.episode += 1
            self.t = 0
            self.logger.info('New Episode-%d, initialize' % (self.episode))
            return

        reward = self.get_reward(observation.objs[0])
        self.last_external_metrics = observation.objs[0]
        next_state = observation.IM
        self.t += 1
        self.global_t += 1

        done = False
        if self.t >= 100:
            done = True
        if done or self.score < -50:
            self.episode_init = True

        self.model.add_sample(self.state, config2action(observation.config, self.config_space), reward, next_state, done)
        self.state = next_state

        if len(self.model.replay_memory) > self.batch_size:
            losses = []

            for _ in range(4):
                losses.append(self.model.update())

        if self.global_t % 5 == 0:
            self.model.save_model('model_params', title='{}_{}'.format(self.task_id, self.global_t))
            self.logger.info('Save model_params to %s_%s' % (self.task_id, self.global_t))

    def get_reward(self, external_metrics):

        def calculate_reward(delta0, deltat):
            if delta0 > 0:
                _reward = ((1 + delta0) ** 2 - 1) * math.fabs(1 + deltat)
            else:
                _reward = - ((1 - delta0) ** 2 - 1) * math.fabs(1 - deltat)

            if _reward > 0 and deltat < 0:
                _reward = 0
            return _reward

        if external_metrics == 0 or self.default_external_metrics == 0:
            # bad case, not enough time to restart mysql or bad knobs
            return 0
        # tps
        delta_0 = float((external_metrics - self.default_external_metrics)) / self.default_external_metrics
        delta_t = float((external_metrics - self.last_external_metrics)) / self.last_external_metrics
        reward = calculate_reward(delta_0, delta_t)
        self.score += reward

        return reward

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



