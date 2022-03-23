import os
import sys
from openbox.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter
from autotune.workload_map import WorkloadMapping
from autotune.pipleline.pipleline import PipleLine
from .knobs import ts, logger
from .utils.parser import  get_hist_json

class DBTuner:
    def __init__(self, args_db, args_tune, env):
        self.model = None
        self.env = env
        self.batch_size = args_tune['batch_size']
        self.episodes = args_tune['episodes']
        if args_tune['replay_memory']:
            self.model.replay_memory.load_memory(replay_memory)
            logger.info('Load memory: {}'.format(self.model.replay_memory))

        self.train_step = 0
        self.accumulate_loss = [0, 0]
        self.step_counter = 0
        self.expr_name = 'train_{}'.format(ts)
        # ouprocess
        self.sigma = 0.2
        # decay rate
        self.sigma_decay_rate = 0.99
        self.create_output_folders()
        self.step_times, self.train_step_times = [], []
        self.env_step_times, self.env_restart_times, self.action_step_times = [], [], []
        self.noisy = False
        '''self.method = args_tune['method']
        self.lhs_log = env.lhs_log
        self.restore_state = args_tune['restore_state']
        self.workload_map =eval(args_tune['workload_map'])
        self.data_repo = args_tune['data_repo']
        self.lhs_num = int(args_tune['lhs_num'])
        self.y_variable = env.y_variable
        self.trials_file = args_tune['trials_file']
        self.tr_init = args_tune['tr_init']
        self.task_id = args_tune['task_id']
        self.rgpe = eval(args_tune['rgpe'])'''
        self.args_tune = args_tune
        self.method = args_tune['method']
        self.y_variable = env.y_variable
        if eval(args_tune['workload_map']):
            self.mapper = WorkloadMapping(args_tune['data_repo'], self.env.knobs_detail, self.y_variable)



    def tune(self):
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

        task_id = self.args_tune['task_id']
        odL = []

        if eval(self.args_tune['rgpe']):
            files = os.listdir(self.args_tune['data_repo'])
            for f in files:
                try:
                    od = get_hist_json(os.path.join(self.args_tune['data_repo'], f), cs=config_space, y_variable=self.y_variable)
                    odL.append(od)
                except:
                    logger.info('build base surrogate failed for {}'.format(f))
            if self.method == 'MBO':
                rgpe_method = 'rgpe_gp'
            else:
                rgpe_method = 'rgpe_prf'
            surrogate_type = 'tlbo_rgpe_' + rgpe_method
        else:
            if self.method == 'SMAC':
                surrogate_type = 'gp'
            else:
                surrogate_type = 'prf'

        bo = PipleLine(self.env.step_openbox,
                       config_space,
                       num_objs=1,
                       num_constraints=0,
                       max_runs=210,
                       surrogate_type=surrogate_type,
                       history_bo_data=odL,
                       acq_optimizer_type='local_random',  # 'random_scipy',#
                       initial_runs=10,
                       init_strategy='random_explore_first',
                       task_id=task_id,
                       time_limit_per_trial=60 * 200,
                       num_hps=len(self.env.default_knobs.keys()))

        save_file = task_id + '.pkl'
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

