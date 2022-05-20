import logging
import numpy as np

import math
import sys
from copy import deepcopy

import gpytorch
import torch
from torch.quasirandom import SobolEngine
from autotune.optimizer.surrogate.base.gp_for_turbo import train_gp
import pdb

from autotune.utils.history_container import HistoryContainer, Observation
from autotune.utils.samplers import SobolSampler, LatinHypercubeSampler
from autotune.utils.config_space import Configuration, UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter


class TURBO_Optimizer:
    def __init__(self, config_space,
                 initial_trials=3,
                 n_trust_regions=3,
                 init_strategy='random_explore_first',
                 batch_size=1,
                 use_ard=True,
                 verbose=True,
                 max_cholesky_size=2000,
                 n_training_steps=50,
                 min_cuda=1024,
                 device="cpu",
                 dtype="float64"):

        self.config_space = config_space
        self.dim = len(config_space.get_hyperparameter_names())
        # Settings
        self.n_trust_regions = n_trust_regions
        self.init_num = initial_trials
        self.init_strategy = init_strategy
        self.batch_size = batch_size
        self.use_ard = use_ard
        self.verbose = verbose
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = min(100 * self.dim, 5000)
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()


        # Remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]
        self.trust_regin_id = 0
        # Initialize parameters
        #if os.path.exists(self.save_file):
        #    self.load_paremeter()
        #else:
        self._restart()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.initial_configurations = self.create_initial_design(self.init_num, self.init_strategy)
        self.init_append = False
        self.init_append_step = 0

    def _restart(self):
        self._idx = np.zeros((0, 1), dtype=int)  # Track what trust region proposed what using an index vector
        self.failcount = np.zeros(self.n_trust_regions, dtype=int)
        self.succcount = np.zeros(self.n_trust_regions, dtype=int)
        self.length = self.length_init * np.ones(self.n_trust_regions)

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates from samples from all trust regions."""
        assert X_cand.shape == (self.n_trust_regions, self.n_cand, self.dim)
        assert y_cand.shape == (self.n_trust_regions, self.n_cand, self.batch_size)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        X_next = np.zeros((self.batch_size, self.dim))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)
        for k in range(self.batch_size):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (self.n_trust_regions, self.n_cand))
            assert y_cand[:, :, k].min() == y_cand[i, j, k]
            X_next[k, :] = deepcopy(X_cand[i, j, :])
            idx_next[k, 0] = i
            assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf

            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf

        return X_next, idx_next

    def _adjust_length(self, fX_next, i):
        assert i >= 0 and i <= self.n_trust_regions - 1

        fX_min = self.fX[self._idx[:, 0] == i, 0].min()  # Target value
        if fX_next.min() < fX_min - 1e-3 * math.fabs(fX_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += len(fX_next)  # NOTE: Add size of the batch for this TR

        if self.succcount[i] == self.succtol:  # Expand trust region
            self.length[i] = min([2.0 * self.length[i], self.length_max])
            self.succcount[i] = 0
        elif self.failcount[i] >= self.failtol:  # Shrink trust region (we may have exceeded the failtol)
            self.length[i] /= 2.0
            self.failcount[i] = 0

    def create_initial_design(self, init_num, init_strategy='default', excluded_configs=None):
        """
            Create several configurations as initial design.
            Parameters
            ----------
            init_strategy: str

            Returns
            -------
            Initial configurations.
        """
        default_config = self.config_space.get_default_configuration()
        num_random_config = init_num - 1
        if init_strategy == 'random':
            initial_configs = self.sample_random_configs(init_num, excluded_configs)
            return initial_configs
        elif init_strategy == 'default':
            initial_configs = [default_config] + self.sample_random_configs(num_random_config, excluded_configs)
            return initial_configs
        elif init_strategy == 'random_explore_first':
            candidate_configs = self.sample_random_configs(100, excluded_configs)
            return self.max_min_distance(default_config, candidate_configs, num_random_config)
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

    def get_suggestion(self, history_container: HistoryContainer):
        # if have enough data, get_suggorate
        num_config_evaluated = len(history_container.configurations)
        if num_config_evaluated < self.init_num:
            self.trust_regin_id = num_config_evaluated % self.n_trust_regions
            self.evluate_init = True
            return self.initial_configurations[num_config_evaluated]
        elif self.init_append:
            self.evluate_init = True
            config = self.X_init_append[self.init_append_step]
            self.init_append_step += 1
            if self.init_append_step >= len(self.X_init_append):
                self.init_append = False
                self.init_append_step = 0
            return config

        else:
            self.evluate_init = False

        #self.save_parameter()
        # Generate candidates from each TR
        X_cand = np.zeros((self.n_trust_regions, self.n_cand, self.dim))
        y_cand = np.inf * np.ones((self.n_trust_regions, self.n_cand, self.batch_size))
        for i in range(self.n_trust_regions):
            idx = np.where(self._idx == i)[0]  # Extract all "active" indices

            # Get the points, values the active values
            X = deepcopy(self.X[idx, :])
            #X = to_unit_cube(X, self.lb, self.ub)

            # Get the values from the standardized data
            fX = deepcopy(self.fX[idx, 0].ravel())

            # Don't retrain the model if the training data hasn't changed
            n_training_steps = 0 if self.hypers[i] else self.n_training_steps

            # Create new candidates
            X_cand[i, :, :], y_cand[i, :, :], self.hypers[i] = self._create_candidates(
                X, fX, length=self.length[i], n_training_steps=n_training_steps, hypers=self.hypers[i]
            )

        # Select the next candidates
        X_next, idx_next = self._select_candidates(X_cand, y_cand)
        assert X_next.min() >= 0.0 and X_next.max() <= 1.0

        keys = self.config_space.get_hyperparameter_names()
        for key in keys:
            if type(self.config_space.get_hyperparameters_dict()[key]) == CategoricalHyperparameter:
                X_next[:, keys.index(key)] = np.round(X_next[:, keys.index(key)] * (self.config_space.get_hyperparameters_dict()[key].num_choices - 1))

        # Undo the warping
        #X_next = from_unit_cube(X_next, self.lb, self.ub)
        return  Configuration(self.config_space, vector=X_next.reshape(-1, self.batch_size))

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            try:
                X_torch = torch.tensor(X.astype('float')).to(device=device, dtype=dtype)
            except:
                pdb.set_trace()
            y_torch = torch.tensor(fX.astype('float')).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand.astype('float')).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def update(self, observation: Observation):
        """
        Update the current observations.
        Parameters
        ----------
        observation

        Returns
        -------

        """
        x = observation.config.get_array()
        keys = self.config_space.get_hyperparameter_names()

        for key in keys:
            if type(self.config_space.get_hyperparameters_dict()[key]) == CategoricalHyperparameter:
                x[keys.index(key)] = x[keys.index(key)] / (self.config_space.get_hyperparameters_dict()[key].num_choices - 1)

        self.X = np.vstack((self.X, x))
        fX_next = np.array([observation.objs])
        self.fX = np.vstack((self.fX, fX_next))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)
        idx_next[0, 0] = self.trust_regin_id
        self._idx = np.vstack((self._idx, idx_next))
        self.n_evals += 1
        if self.evluate_init:
            return

        # Update trust regions
        for i in range(self.n_trust_regions):
            idx_i = np.where(idx_next == i)[0]
            if len(idx_i) > 0:
                self.hypers[i] = {}  # Remove model hypers
                fX_i = fX_next[idx_i]

                if self.verbose and fX_i.min() < self.fX.min() - 1e-3 * math.fabs(self.fX.min()):
                    n_evals, fbest = self.n_evals, fX_i.min()
                    print(f"{n_evals}) New best @ TR-{i}: {fbest:.4}")
                    sys.stdout.flush()
                self._adjust_length(fX_i, i)

        # Check if any TR needs to be restarted
        for i in range(self.n_trust_regions):
            if self.length[i] < self.length_min:  # Restart trust region if converged
                idx_i = self._idx[:, 0] == i

                if self.verbose:
                    n_evals, fbest = self.n_evals, self.fX[idx_i, 0].min()
                    print(f"{n_evals}) TR-{i} converged to: : {fbest:.4}")
                    sys.stdout.flush()

                # Reset length and counters, remove old data from trust region
                self.length[i] = self.length_init
                self.succcount[i] = 0
                self.failcount[i] = 0
                self._idx[idx_i, 0] = -1  # Remove points from trust region
                self.hypers[i] = {}  # Remove model hypers

                # Create a new initial design
                self.init_append = True
                self.X_init_append = self.create_initial_design(max(int(self.init_num/self.n_trust_regions),1) , self.init_strategy)

                #fX_init = np.array([[self.f(x)] for x in X_init])

                # Print progress
                if self.verbose:
                    n_evals = self.n_evals
                    print(f"{n_evals}) TR-{i} is restarting ")
                    sys.stdout.flush()

        return

    def sample_random_configs(self, num_configs=1, excluded_configs=None):
        """
        Sample a batch of random configurations.
        Parameters
        ----------
        num_configs

        history_container

        Returns
        -------

        """
        if excluded_configs is None:
            excluded_configs = []

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

    def max_min_distance(self, default_config, src_configs, num):
        min_dis = list()
        initial_configs = list()
        initial_configs.append(default_config)

        for config in src_configs:
            dis = np.linalg.norm(config.get_array() - default_config.get_array())
            min_dis.append(dis)
        min_dis = np.array(min_dis)

        for i in range(num):
            furthest_config = src_configs[np.argmax(min_dis)]
            initial_configs.append(furthest_config)
            min_dis[np.argmax(min_dis)] = -1

            for j in range(len(src_configs)):
                if src_configs[j] in initial_configs:
                    continue
                updated_dis = np.linalg.norm(src_configs[j].get_array() - furthest_config.get_array())
                min_dis[j] = min(updated_dis, min_dis[j])

        return initial_configs





