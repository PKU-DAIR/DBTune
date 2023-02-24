# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

from typing import List
from collections import defaultdict
import numpy as np
import pandas as pd

from autotune.utils.config_space import Configuration, ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter


def convert_configurations_to_array(configs: List[Configuration]) -> np.ndarray:
    """Impute inactive hyperparameters in configurations with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configs : List[Configuration]
        List of configuration objects.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    configs_array = np.array([config.get_array() for config in configs],
                             dtype=np.float64)
    configuration_space = configs[0].configuration_space
    return impute_default_values(configuration_space, configs_array)


def impute_default_values(
        configuration_space: ConfigurationSpace,
        configs_array: np.ndarray
) -> np.ndarray:
    """Impute inactive hyperparameters in configuration array with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configuration_space : ConfigurationSpace
    
    configs_array : np.ndarray
        Array of configurations.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    for hp in configuration_space.get_hyperparameters():
        default = hp.normalized_default_value
        idx = configuration_space.get_idx_by_hyperparameter_name(hp.name)
        nonfinite_mask = ~np.isfinite(configs_array[:, idx])
        configs_array[nonfinite_mask, idx] = default

    return configs_array


def impute_incumb_values(configurations, incumbent):
    if isinstance(configurations, list):
        configurations_ = list()
        for configuration in configurations:
            knob_dict = configuration.get_dictionary()
            for knob in incumbent.keys():
                if not knob in knob_dict.keys():
                    knob_dict[knob] = incumbent[knob]
            configurations_.append(Configuration(incumbent.configuration_space, values=knob_dict))

        return configurations_
    else:
        knob_dict = configurations.get_dictionary()
        for knob in incumbent.keys():
            if not knob in knob_dict.keys():
                knob_dict[knob] = incumbent[knob]
        return  Configuration(incumbent.configuration_space, values=knob_dict)



def config2df(configs):
    config_dic = defaultdict(list)
    for config in configs:
        for k in config:
            config_dic[k].append(config[k])

    return pd.DataFrame.from_dict(config_dic)


def configs2space(configs, space):
    configs_new = list()
    for i, config in enumerate(configs):
        config_new = {}
        for name in config.keys():
            if name in space.get_hyperparameter_names():
                value = config[name]
                if isinstance(space.get_hyperparameters_dict()[name],  CategoricalHyperparameter):
                    if value not in space.get_hyperparameters_dict()[name].choices:
                        value = space.get_hyperparameters_dict()[name].default_value
                else:
                    if value < space.get_hyperparameters_dict()[name].lower:
                        value =  space.get_hyperparameters_dict()[name].lower
                    if value > space.get_hyperparameters_dict()[name].upper:
                        value =  space.get_hyperparameters_dict()[name].upper

                config_new[name] = value

        c_new = Configuration(space, config_new)
        configs_new.append(c_new)

    return configs_new

def max_min_distance(default_config, src_configs, num):
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
