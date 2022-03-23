import logging
from .knobs import logger
import numpy as np
import bisect
import json
import os
import sys
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from smac.smac_cli import SMACCLI
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from .knobs import initialize_knobs
from smac.utils.io.output_directory import create_output_directory
boston = load_boston()


class SMAC(object):
    def __init__(self, knobs_detail):
        self.cs = ConfigurationSpace()
        self.scenario = None
        self.knobs_detail = knobs_detail

    def init_Configuration(self):
        KNOBS = list(self.knobs_detail.keys())
        for idx in range(len(KNOBS)):
            name = KNOBS[idx]
            value = self.knobs_detail[name]
            knob_type = value['type']
            if knob_type == 'enum':
                knob = CategoricalHyperparameter(name, [ str(i) for i in value["enum_values"]], default_value=str(value['default']))
            elif knob_type == 'integer':
                min_val, max_val = value['min'], value['max']
                if value.get('stride'):
                    knob = UniformIntegerHyperparameter(name, min_val, max_val, default_value=value['default'], q=value['stride'])
                else:
                    if self.knobs_detail[name]['max'] > sys.maxsize:#name == "innodb_online_alter_log_max_size":
                        knob = UniformIntegerHyperparameter(name, int(min_val / 1000), int(max_val / 1000),
                                                            default_value=int(value['default'] / 1000))
                    else:
                        knob = UniformIntegerHyperparameter(name, min_val, max_val, default_value=value['default'])
            elif knob_type == 'float':
                min_val, max_val = value['min'], value['max']
                if value.get('stride'):
                    knob = UniformFloatHyperparameter(name, min_val, max_val, default_value=value['default'], q=value['stride'])
                else:
                    knob = UniformFloatHyperparameter(name, min_val, max_val, default_value=value['default'])
            self.cs.add_hyperparameter(knob)

        self.scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                  "runcount-limit": 210,  # max. number of function evaluations; for this example set to a low number
                  "cs": self.cs,  # configuration space
                  "deterministic": "true",
                  "memory_limit": 3072,  # adapt this to reasonable value for your hardware
                  "output_dir": "restore_me",
                  })


    def restore(self, run_id, new_run_id, load_num=-1):
        loader = SMACCLI()
        old_output_dir = "restore_me/run_" + str(run_id)
        if not load_num == -1:
            old_run_f = old_output_dir+'/runhistory.json'
            f = open(old_run_f)
            f_json = json.load(f)
            old_history_cut = {}
            old_history_cut['data'] = f_json['data'][:load_num]
            old_history_cut['config_origins'] = {}
            old_history_cut['configs'] = {}
            for i in range(load_num):
                old_history_cut['config_origins'][str(i+1)] = f_json['config_origins'][str(i+1)]
                old_history_cut['configs'][str(i+1)] = f_json['configs'][str(i+1)]
            cmd = "mv {} {}".format(old_run_f, old_run_f+'.copy')
            os.system(cmd)
            with open(old_run_f, 'w') as fp:
                json.dump(old_history_cut, fp, indent=4)
        rh, stats, traj_list_aclib, traj_list_old = loader.restore_state(self.scenario, old_output_dir)

        self.scenario.output_dir_for_this_run = create_output_directory(
            self.scenario, int(new_run_id), logger)
        self.scenario.write()
        incumbent = loader.restore_state_after_output_dir(self.scenario, stats, traj_list_aclib, traj_list_old)
        return rh, stats, incumbent, self.scenario
