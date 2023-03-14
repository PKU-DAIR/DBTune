import abc
import pdb
import random

from autotune.utils.util_funcs import check_random_state
from autotune.utils.logging_utils import get_logger
from autotune.utils.history_container import HistoryContainer
from autotune.utils.constants import MAXINT, SUCCESS
from autotune.utils.config_space import get_one_exchange_neighbourhood
from autotune.utils.history_container import Observation
from autotune.utils.config_space.util import configs2space

class GA_Optimizer(object, metaclass=abc.ABCMeta):

    def __init__(self, config_space,
                 history_container: HistoryContainer,
                 num_objs=1,
                 num_constraints=0,
                 population_size=10,
                 subset_size=6,
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
        if len(self.population) < self.population_size:
            # Initialize population
            next_config = self.sample_random_config(excluded_configs=self.all_configs)
        else:
            # Select a parent by subset tournament and epsilon greedy
            if self.rng.random() < self.epsilon:
                parent_config = random.sample(self.population, 1)[0]['config']
            else:
                subset = random.sample(self.population, self.subset_size)
                subset.sort(key=lambda x: x['perf'])    # minimize
                parent_config = subset[0]['config']

            # Mutation to 1-step neighbors
            next_config = None
            neighbors_gen = get_one_exchange_neighbourhood(parent_config, seed=self.rng.randint(MAXINT))
            for neighbor in neighbors_gen:
                if neighbor not in self.all_configs:
                    next_config = neighbor
                    break
            if next_config is None:  # If all the neighors are evaluated, sample randomly!
                next_config = self.sample_random_config(excluded_configs=self.all_configs)

        return next_config

    def update(self, observation: Observation):
        """
        Update the current observations.
        Parameters
        ----------
        observation
        Returns
        -------
        """

        config = observation.config
        perf = observation.objs[0]
        trial_state = observation.trial_state

        self.all_configs.add(config)


        # update population
        if trial_state == SUCCESS and perf < MAXINT:
            self.population.append(dict(config=config, age=self.age, perf=perf))
            self.age += 1

        # Eliminate samples
        if len(self.population) > self.population_size:
            if self.strategy == 'oldest':
                self.population.sort(key=lambda x: x['age'])
                self.population.pop(0)
            elif self.strategy == 'worst':
                self.population.sort(key=lambda x: x['perf'])
                self.population.pop(-1)
            else:
                raise ValueError('Unknown strategy: %s' % self.strategy)

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
