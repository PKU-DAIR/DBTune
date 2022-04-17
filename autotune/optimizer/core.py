import sys
from autotune.utils.constants import MAXINT, SUCCESS
from autotune.optimizer.acquisition_function import *
from autotune.utils.util_funcs import get_types


acq_dict = {
    'ei': EI,
    'eips': EIPS,
    'logei': LogEI,
    'pi': PI,
    'lcb': LCB,
    'lpei': LPEI,
    'ehvi': EHVI,
    'ehvic': EHVIC,
    'mesmo': MESMO,
    'usemo': USeMO,     # todo single acq type
    'mcei': MCEI,
    'parego': EI,
    'mcparego': MCParEGO,
    'mcparegoc': MCParEGOC,
    'mcehvi': MCEHVI,
    'mcehvic': MCEHVIC,
    'eic': EIC,
    'mesmoc': MESMOC,
    'mesmoc2': MESMOC2,
    'mceic': MCEIC,
}



def build_acq_func(func_str='ei', model=None, constraint_models=None, **kwargs):
    func_str = func_str.lower()
    acq_func = acq_dict.get(func_str)
    if acq_func is None:
        raise ValueError('Invalid string %s for acquisition function!' % func_str)
    if constraint_models is None:
        return acq_func(model=model, **kwargs)
    else:
        return acq_func(model=model, constraint_models=constraint_models, **kwargs)



def build_optimizer(func_str='local_random', acq_func=None, config_space=None, rng=None):
    assert config_space is not None
    func_str = func_str.lower()

    if func_str == 'local_random':
        from autotune.optimizer.acq_maximizer.ei_optimization import InterleavedLocalAndRandomSearch
        optimizer = InterleavedLocalAndRandomSearch
    elif func_str == 'random_scipy':
        from autotune.optimizer.acq_maximizer.ei_optimization import RandomScipyOptimizer
        optimizer = RandomScipyOptimizer
    elif func_str == 'scipy_global':
        from autotune.optimizer.acq_maximizer.ei_optimization import ScipyGlobalOptimizer
        optimizer = ScipyGlobalOptimizer
    elif func_str == 'mesmo_optimizer':
        from autotune.optimizer.acq_maximizer.ei_optimization import MESMO_Optimizer
        optimizer = MESMO_Optimizer
    elif func_str == 'usemo_optimizer':
        from autotune.optimizer.acq_maximizer.ei_optimization import USeMO_Optimizer
        optimizer = USeMO_Optimizer
    elif func_str == 'cma_es':
        from autotune.optimizer.acq_maximizer.ei_optimization import CMAESOptimizer
        optimizer = CMAESOptimizer
    elif func_str == 'batchmc':
        from autotune.optimizer.acq_maximizer.ei_optimization import batchMCOptimizer
        optimizer = batchMCOptimizer
    elif func_str == 'staged_batch_scipy':
        from autotune.optimizer.acq_maximizer.ei_optimization import StagedBatchScipyOptimizer
        optimizer = StagedBatchScipyOptimizer
    else:
        raise ValueError('Invalid string %s for acq_maximizer!' % func_str)

    return optimizer(acquisition_function=acq_func,
                     config_space=config_space,
                     rng=rng)


