import abc
import pdb
import random
from typing import List

from autotune.utils.util_funcs import check_random_state
from autotune.utils.logging_utils import get_logger
from autotune.utils.history_container import HistoryContainer
from autotune.utils.constants import MAXINT, SUCCESS
from autotune.utils.config_space import get_one_exchange_neighbourhood, Configuration
from autotune.utils.history_container import Observation
from autotune.utils.config_space.util import configs2space

class GA_Optimizer(object, metaclass=abc.ABCMeta):

    def __init__(self, config_space,
                 history_container: HistoryContainer,
                 num_objs=1,
                 num_constraints=0,
                 population_size=20,
                 subset_size=10,
                 epsilon=0.2,
                 strategy='worst',  # 'worst', 'oldest'
                 optimization_strategy='ea',
                 output_dir='logs',
                 random_state=None):

        # Create output (logging) directory.
        # Init logging module.
        # Random seed generator.
        self.num_objs = num_objs
        self.num_constraints = num_constraints
        assert self.num_objs == 1 and self.num_constraints == 0
        self.output_dir = output_dir
        self.rng = check_random_state(random_state)
        self.config_space = config_space
        self.config_space_seed = self.rng.randint(MAXINT)
        self.config_space.seed(self.config_space_seed)
        self.logger = get_logger(self.__class__.__name__)

        # Basic components in Advisor.
        self.optimization_strategy = optimization_strategy

        # Init the basic ingredients
        self.running_configs = list()
        self.all_configs = set()
        self.age = 0
        self.population = list()
        self.population_size = population_size
        self.subset_size = subset_size
        assert 0 < self.subset_size <= self.population_size
        self.epsilon = epsilon
        self.strategy = strategy
        assert self.strategy in ['worst', 'oldest']

        self.last_suggestions = []
        self.last_observations = []

        # initialize
        self.initialize(history_container)


    def initialize(self, history_container: HistoryContainer):
        all_configs = configs2space(history_container.get_all_configs(), self.config_space)
        all_perfs = history_container.get_all_perfs()
        self.all_configs = set(all_configs)

        num_config_evaluated = len(all_perfs)
        for i in range(num_config_evaluated):
            self.population.append(dict(config=all_configs[i], age=self.age, perf=all_perfs[i]))
            self.age += 1

        if self.strategy == 'oldest':
            self.population.sort(key=lambda x: x['age'])
        elif self.strategy == 'worst':
            self.population.sort(key=lambda x: x['perf'])

        while len(self.population) > self.population_size:
            if self.strategy == 'oldest':
                self.population.pop(0)
            elif self.strategy == 'worst':
                self.population.pop(-1)
            else:
                raise ValueError('Unknown strategy: %s' % self.strategy)


    def get_suggestion(self, history_container: HistoryContainer, compact_space=None):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        # Get next suggestion if already
        if not self.last_suggestions:
            self.update_observations(self.last_observations)
            self.last_observations = []
            self.last_suggestions = self.get_suggestions(history_container, compact_space)
        return self.last_suggestions.pop()
        

    def get_suggestions(self, history_container: HistoryContainer, compact_space=None):
        next_configs = []
        if len(self.population) < self.population_size:
            # Initialize population
            miu = self.population_size - len(self.population)
            for t in range(miu):
                next_config = self.sample_random_config(excluded_configs=self.all_configs)
                self.all_configs.add(next_config)
                self.running_configs.append(next_config)
                next_configs.append(next_config)
        else:
            # Mutation here
            for individual in self.population:
                next_config = self.mutation(individual['config'])
                self.all_configs.add(next_config)
                self.running_configs.append(next_config)
                next_configs.append(next_config)
            # Cross-over here
            for parent_1 in range((self.population_size - 1) // 2):
                parent_2 = self.population_size - 1 - parent_1
                next_config = self.cross_over(self.population[parent_1]['config'], self.population[parent_2]['config'])
                self.all_configs.add(next_config)
                self.running_configs.append(next_config)
                next_configs.append(next_config)

        return next_configs


    def update_observations(self, observations: List[Observation]):
        print("Update GA")
        self.age += 1
        for observation in observations:
            config = observation.config
            perf = observation.objs[0]
            trial_state = observation.trial_state
            assert config in self.running_configs
            self.running_configs.remove(config)

            # update population
            if trial_state == SUCCESS and perf < MAXINT:
                self.population.append(dict(config=config, perf=perf, age=self.age))

        # Eliminate samples
        if len(self.population) > self.population_size:
            layers = self.pareto_layers(self.population)
            laynum = len(layers)
            tot = 0
            self.population = []
            for t in range(laynum):
                if tot + len(layers[t]) > self.population_size:
                    miu = self.population_size - tot
                    self.population += self.crowding_select(layers[t], miu)
                    break
                else:
                    self.population += layers[t]
                    tot += len(layers[t])


        # Eliminate worst samples
        if len(self.population) > self.population_size:
            if self.strategy == 'oldest':
                self.population.sort(key=lambda x: x['age'])
                self.population.pop(0)
            elif self.strategy == 'worst':
                self.population.sort(key=lambda x: x['perf'])
                self.population.pop(-1)
            else:
                raise ValueError('Unknown strategy: %s' % self.strategy)


    def update(self, observation: Observation):
        """
        Update the current observations.
        Parameters
        ----------
        observation
        Returns
        -------
        """
        self.last_observations.append(observation)


    def sample_random_config(self, excluded_configs=None):
        if excluded_configs is None:
            excluded_configs = set()

        sample_cnt = 0
        max_sample_cnt = 1000
        while True:
            config = self.config_space.sample_configuration()
            sample_cnt += 1
            if config not in excluded_configs:
                break
            if sample_cnt >= max_sample_cnt:
                self.logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                break
        return config
    

    def cross_over(self, config_a: Configuration, config_b: Configuration):
        a1, a2 = config_a.get_array(), config_b.get_array()
        s_len = len(self.config_space.keys())
        for i in range(s_len):
            if s_len < 3:
                a1[i] = (a1[i] + a2[i]) * 0.5
            else:
                if self.rng.random() < 0.5:
                    a1[i] = a2[i]

        return Configuration(self.config_space, vector=a1)


    def mutation(self, config: Configuration):
        ret_config = None
        neighbors_gen = get_one_exchange_neighbourhood(config, seed=self.rng.randint(MAXINT))
        for neighbor in neighbors_gen:
            if neighbor not in self.all_configs:
                ret_config = neighbor
                break
        if ret_config is None:
            ret_config = self.sample_random_config(excluded_configs=self.all_configs)
        return ret_config


    def crowding_select(self, xs, num):
        INF = 1e7
        llen = len(xs)
        for i in range(llen):
            if isinstance(xs[0]['perf'], float):
                xs[i]['perf'] = [xs[i]['perf']]
        dim = len(xs[0]['perf'])
        xs = [[x, 0] for x in xs]
        for k in range(dim):
            xs = sorted(xs, key=lambda xv: xv[0]['perf'][k], reverse=True)
            xs[0][1] += INF
            xs[-1][1] += INF
            for t in range(1, llen - 1):
                xs[t][1] += xs[t - 1][0]['perf'][k] - xs[t + 1][0]['perf'][k]
        xs = sorted(xs, key=lambda xv: xv[1], reverse=True)
        xs = xs[:num]
        return [x for (x, v) in xs]
    

    def pareto_layers(self, population):
        remain = [x for x in population]
        res = []
        while remain:
            front = self.pareto_frontier(remain)
            assert len(front) > 0
            res.append(front)
            remain = [x for x in remain if x not in front]
        return res


    # Naive Implementation
    def pareto_frontier(self, population):
        if isinstance(population[0]['perf'], float):
            return [x for x in population if
                    not [y for y in population if y['perf'] < x['perf']]]
        return [x for x in population if
                not [y for y in population if not [i for i in range(len(x['perf'])) if y['perf'][i] >= x['perf'][i]]]]
