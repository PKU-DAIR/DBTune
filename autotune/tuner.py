import os
import sys
from openbox.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter
from autotune.workload_map import WorkloadMapping
from autotune.pipleline.pipleline import PipleLine
from .knobs import ts, logger
from .utils.parser import  get_hist_json
import pdb

class DBTuner:
    def __init__(self, args_db, args_tune, env):
        self.model = None
        self.env = env
        self.create_output_folders()
        self.args_tune = args_tune
        self.method = args_tune['optimize_method']
        self.y_variable = env.y_variable
        self.transfer_framework =  args_tune['transfer_framework']
        self.odL = []

        self.setup_configuration_space()
        self.setup_transfer()

    def setup_configuration_space(self):
        KNOBS = self.env.knobs_detail
        config_space = ConfigurationSpace()
        for name in KNOBS.keys():
            value = KNOBS[name]
            knob_type = value['type']
            if knob_type == 'enum':
                knob = CategoricalHyperparameter(name, [str(i) for i in value["enum_values"]],
                                                 default_value=str(value['default']))
            elif knob_type == 'integer':
                min_val, max_val = value['min'], value['max']
                if self.env.knobs_detail[name]['max'] > sys.maxsize:
                    knob = UniformIntegerHyperparameter(name, int(min_val / 1000), int(max_val / 1000),
                                                        default_value=int(value['default'] / 1000))
                else:
                    knob = UniformIntegerHyperparameter(name, min_val, max_val, default_value=value['default'])
            elif knob_type == 'float':
                min_val, max_val = value['min'], value['max']
                knob = UniformFloatHyperparameter(name, min_val, max_val, default_value=value['default'])

            config_space.add_hyperparameters([knob])

        self.config_space = config_space



    def setup_transfer(self):
        if self.transfer_framework == 'workload_map':
            self.mapper = WorkloadMapping(args_tune['data_repo'], self.env.knobs_detail, self.y_variable)

        elif self.transfer_framework == 'rgpe':
            odL = []
            files = os.listdir(self.args_tune['data_repo'])
            for f in files:
                try:
                    od = get_hist_json(os.path.join(self.args_tune['data_repo'], f), cs=self.config_space,
                                       y_variable=self.y_variable, knob_detail=self.env.knobs_detail)
                    odL.append(od)
                except:
                    logger.info('build base surrogate failed for {}'.format(f))
            if self.method == 'MBO':
                rgpe_method = 'rgpe_gp'
            else:
                rgpe_method = 'rgpe_prf'
            self.surrogate_type = 'tlbo_rgpe_' + rgpe_method
            self.odL = odL
        else:
            if self.method == 'SMAC':
                self.surrogate_type = 'gp'
            else:
                self.surrogate_type = 'prf'



    def tune(self):
        bo = PipleLine(self.env.step_openbox,
                       self.config_space,
                       optimizer_type=self.method,
                       num_objs=1,
                       num_constraints=0,
                       max_runs=210,
                       surrogate_type=self.surrogate_type,
                       history_bo_data=self.odL,
                       acq_optimizer_type='local_random',  # 'random_scipy',#
                       selector_type=self.args_tune['selector_type'],
                       initial_runs=eval(self.args_tune['initial_runs']),
                       incremental=self.args_tune['incremental'],
                       incremental_step=eval(self.args_tune['incremental_step']),
                       incremental_num=eval(self.args_tune['incremental_num']),
                       init_strategy='random_explore_first',
                       task_id=self.args_tune['task_id'],
                       time_limit_per_trial=60 * 200,
                       num_hps=int(self.args_tune['initial_tunable_knob_num'])
                       )

        save_file = self.args_tune['task_id'] + '.pkl'
        advisor = bo.config_advisor
        history = bo.run()
        '''try:
            history = bo.run()
        except:
            with open(save_file, 'wb') as f:
                pickle.dump(bo.config_advisor.history_container, f)
                print("Save history recorde to {}".format(save_file))'''



    @staticmethod
    def create_output_folders():
        output_folders = ['log', 'save_memory', 'save_knobs', 'save_state_actions', 'model_params']
        for folder in output_folders:
            if not os.path.exists(folder):
                os.mkdir(folder)

