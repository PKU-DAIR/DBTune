#
# Autotune- gp_torch.py
#
#
import numpy as np
import math
import torch
import random
import botorch
import gpytorch
import time
from gpytorch.priors import GammaPrior
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from torch.optim import SGD
from matplotlib import pyplot as plt
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.fit import fit_gpytorch_model
from botorch.sampling import IIDNormalSampler
from .knobs import gen_continuous
from gpytorch.kernels.scale_kernel import ScaleKernel
# from ax import ParameterType, RangeParameter, SearchSpace
# from ax import SimpleExperiment
# from ax.modelbridge import get_sobol
# from ax.modelbridge.factory import get_botorch

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# N_DIM = 6
# BATCH_SIZE = 1
# MC_SAMPLES = 10


# NOISE_SE = 0.5
# train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

# put constraints in constraints terms


def optimize_acqf_and_get_observation(acq_func,
                                      candidate_size,
                                      bounds=None,
                                      inequality_constraints=None,
                                      equality_constraints=None):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    if bounds is None:
        bounds = torch.tensor([[0.0] * 6, [1.0] * 6],
                              device=device, dtype=torch.double)

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=candidate_size,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True
    )
    new_x = candidates.detach()
    return new_x


'''
Get acquisition func
'''


def get_acqf(func_name, model, train_obj, **kwargs):
    if func_name == 'EI':
        EI = ExpectedImprovement(
            model=model,
            best_f=train_obj.max(),
            maximize=True
        )
        return EI
    elif func_name == 'UCB':
        beta = kwargs['beta'] if 'beta' in kwargs else 0.2
        UCB = UpperConfidenceBound(
            model=model,
            beta=beta
        )
    else:
        return None


def anlytic_optimize_acqf_and_get_observation(acq_func,
                                              candidate_size,
                                              bounds,
                                              inequality_constraints=None,
                                              equality_constraints=None):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=candidate_size,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200})
    new_x = candidates.detach()
    return new_x


def initialize_GP_model(train_x, train_y, state_dict=None, kernel=None):
    if kernel:
        model = SingleTaskGP(train_x, train_y, covar_module=kernel)
    else:
        model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model
