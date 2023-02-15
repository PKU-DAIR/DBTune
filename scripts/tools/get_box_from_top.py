import os
import sys
import json
import pandas as pd
import numpy as np
import random
from plot import parse_data_onefile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  roc_auc_score
from collections import defaultdict
from autotune.utils.constants import MAXINT, SUCCESS
from autotune.utils.history_container import HistoryContainer
from autotune.tuner import DBTuner
from autotune.knobs import initialize_knobs, get_default_knobs
from autotune.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter
from autotune.selector.selector import KnobSelector
from openbox.utils.config_space.util import convert_configurations_to_array
from autotune.utils.logging_utils import setup_logger, get_logger
from autotune.utils.config_space import Configuration, ConfigurationSpace
from autotune.transfer.tlbo.rgpe import RGPE
from autotune.utils.util_funcs import check_random_state

import pdb
import pandas as pd
test_workload = 'job'
setup_logger('SPACE_FILTER')

def load_history(fileL, config_space):

    data_mutipleL = list()
    for fn in fileL:
        try:
            with open(fn) as fp:
                all_data = json.load(fp)
        except Exception as e:
            print('Encountered exception %s while reading runhistory from %s. '
                  'Not adding any runs!', e, fn, )
            return

        info = all_data["info"]
        data = all_data["data"]
        data_mutipleL = data_mutipleL + data


    file_out = 'history_{}_{}.json'.format(workload, knob_num)
    with open(file_out, "w") as fp:
        json.dump({"info": info, "data": data_mutipleL}, fp, indent=2)

    task_id = workload
    history_container = HistoryContainer(task_id, config_space=config_space)
    history_container.load_history_from_json(file_out)

    return history_container

def setup_configuration_space(knob_config_file, knob_num):
    knobs = initialize_knobs(knob_config_file, knob_num)
    knobs_list = []
    config_space = ConfigurationSpace()

    for name in knobs.keys():
        value = knobs[name]
        knob_type = value['type']
        if knob_type == 'enum':
            knob = CategoricalHyperparameter(name, [str(i) for i in value["enum_values"]],
                                             default_value=str(value['default']))
        elif knob_type == 'integer':
            min_val, max_val = value['min'], value['max']
            if  max_val > sys.maxsize:
                knob = UniformIntegerHyperparameter(name, int(min_val / 1000), int(max_val / 1000),
                                                    default_value=int(value['default'] / 1000))
            else:
                knob = UniformIntegerHyperparameter(name, min_val, max_val, default_value=value['default'])
        elif knob_type == 'float':
            min_val, max_val = value['min'], value['max']
            knob = UniformFloatHyperparameter(name, min_val, max_val, default_value=value['default'])
        else:
            raise ValueError('Invalid knob type!')

        knobs_list.append(knob)

    config_space.add_hyperparameters(knobs_list)

    return config_space




if __name__ == '__main__':
    history_path = '/data2/ruike/DBTune/scripts/DBTune_history/repob'
    knob_config_file = '/data2/ruike/DBTune/scripts/experiment/gen_knobs/mysql_all_197_32G.json'
    knob_details = initialize_knobs(knob_config_file, -1)
    workloadL = [ 'sysbench', 'twitter', 'job', 'tpch']
    workloadL.remove(test_workload)
    task = [ 'smac', 'mbo', 'ddpg', 'ga']
    spaceL = [6, 197, 100, 50, 25, 12]
    config_space = setup_configuration_space(knob_config_file, -1)

    topL = list()
    for workload in workloadL:
        for method in task:
            for knob_num in spaceL:
                file = 'history_{}_{}_{}.json'.format(workload, method, knob_num)
                file = os.path.join(history_path, file)
                task_id = "{}_{}_{}".format(workload, method, knob_num)
                history_container = HistoryContainer(task_id, config_space=config_space)
                history_container.load_history_from_json(file)
                topL.append(history_container.get_incumbents()[0][0])

    X = convert_configurations_to_array(topL)

    knob_dict = dict()
    for j in range(X.shape[1]):
        knob = config_space.get_hyperparameter_names()[j]
        transform = config_space.get_hyperparameters_dict()[knob]._transform
        if knob_details[knob]['type'] == 'enum':
            values = np.unique(X[:, j])
            feasiable_values = list()
            for t in range(values.shape[0]):
                value = values[t]
                true_value = transform(value)
                feasiable_values.append(true_value)

            if not knob_details[knob]['default'] in feasiable_values:
                knob_details[knob]['default'] = feasiable_values[0]
            knob_dict[knob] = {'type':'enum',
                               'enum_values': feasiable_values,
                               'default': knob_details[knob]['default']}
        else:
            v_min = transform( X[:,j].min())
            v_max =  transform(X[:,j].max())
            if v_max == v_min:
                print(knob)
                continue
            if knob_details[knob]['default'] < v_min:
                knob_details[knob]['default'] = v_min
            elif knob_details[knob]['default'] > v_max:
                knob_details[knob]['default'] = v_max

            knob_dict[knob] = {'type':'integer',
                               'min':v_min,
                               'max':v_max,
                               'default':knob_details[knob]['default']}

    output_file = '../experiment/gen_knobs/box_space_{}.json'.format(test_workload)
    with open(output_file, 'w') as fp:
        json.dump(knob_dict, fp, indent=4)

