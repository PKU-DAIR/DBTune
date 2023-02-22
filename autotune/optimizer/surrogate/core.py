import sys
from autotune.utils.constants import MAXINT, SUCCESS
from autotune.optimizer.acquisition_function import *
from autotune.utils.util_funcs import get_types




def build_surrogate(func_str='gp', config_space=None, rng=None, history_hpo_data=None, context=None):
    assert config_space is not None
    func_str = func_str.lower()
    types, bounds = get_types(config_space)
    seed = rng.randint(MAXINT)

    if func_str == 'prf':
        try:
            from autotune.optimizer.surrogate.base.rf_with_instances import RandomForestWithInstances
            return RandomForestWithInstances(types=types, bounds=bounds, seed=seed, config_space=config_space)
        except ModuleNotFoundError:
            from autotune.optimizer.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
            print('[Build Surrogate] Use probabilistic random forest based on scikit-learn. For better performance, '
                  'please install pyrfr: '
                  'https://open-box.readthedocs.io/en/latest/installation/install_pyrfr.html')
            return skRandomForestWithInstances(types=types, bounds=bounds, seed=seed)

    elif func_str == 'sk_prf':
        from autotune.optimizer.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
        return skRandomForestWithInstances(types=types, bounds=bounds, seed=seed)

    elif func_str == 'lightgbm':
        from autotune.optimizer.surrogate.lightgbm import LightGBM
        return LightGBM(config_space, types=types, bounds=bounds, seed=seed)
    elif func_str == 'context_prf':
        from autotune.optimizer.surrogate.base.rf_with_contexts import RandomForestWithContexts
        return  RandomForestWithContexts(types=types, bounds=bounds, seed=seed, current_context=context)

    if func_str == 'random_forest':
        from autotune.optimizer.surrogate.skrf import RandomForestSurrogate
        return RandomForestSurrogate(config_space, types=types, bounds=bounds, seed=seed)

    elif func_str.startswith('gp'):
        from autotune.optimizer.surrogate.base.build_gp import create_gp_model
        return create_gp_model(model_type=func_str,
                               config_space=config_space,
                               types=types,
                               bounds=bounds,
                               rng=rng)
    elif func_str.startswith('mfgpe'):
        from autotune.transfer.tlbo.mfgpe import MFGPE
        inner_surrogate_type = 'prf'
        return MFGPE(config_space, history_hpo_data, seed,
                     surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1)
    elif func_str.startswith('tlbo'):
        print('the current surrogate is', func_str)
        if 'rgpe' in func_str:
            from autotune.transfer.tlbo.rgpe import RGPE
            inner_surrogate_type = func_str.split('_')[-1]
            return RGPE(config_space, history_hpo_data, seed,
                        surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1)
        elif 'sgpr' in func_str:
            from autotune.transfer.tlbo.stacking_gpr import SGPR
            inner_surrogate_type = func_str.split('_')[-1]
            return SGPR(config_space, history_hpo_data, seed,
                        surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1)
        elif 'topov3' in func_str:
            from autotune.transfer.tlbo.topo_variant3 import TOPO_V3
            inner_surrogate_type = func_str.split('_')[-1]
            return TOPO_V3(config_space, history_hpo_data, seed,
                           surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1)
        elif 'mapping' in func_str:
            from autotune.transfer.tlbo.workload_map import WorkloadMapping
            inner_surrogate_type = func_str.split('_')[-1]
            return WorkloadMapping(config_space, history_hpo_data, seed,
                                   surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1)
        else:
            raise ValueError('Invalid string %s for tlbo surrogate!' % func_str)
    else:
        raise ValueError('Invalid string %s for surrogate!' % func_str)


def surrogate_switch():
    pass