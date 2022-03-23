# Usage:
#python train.py  --method=GP_BOTORCH --benchmark=sysbench --workload_type=readwrite --knobs_config=mysql_all_cdbtune.json --dbname=sbtest --lhs_log=xxx.res

import os
import argparse

from autotune.dbenv import MySQLEnv
from autotune.dbenv_bench import BenchEnv
from autotune.workload import SYSBENCH_WORKLOAD, TPCC_WORKLOAD, JOB_WORKLOAD
from autotune.workload import OLTPBENCH_WORKLOADS, WORKLOAD_ZOO_WORKLOADS
from autotune.tuner import MySQLTuner
from autotune.knobs import logger
from autotune.utils.helper import check_env_setting
from autotune.utils.autotune_exceptions import AutotuneError
from autotune.model.ddpg import DDPG
import pdb
from autotune.workload_map import WorkloadMapping

HOST='localhost'
THREADS=64
PASSWD=''
PORT=3306
USER='root'
SOCK=os.environ.get("MYSQL_SOCK")
LOG_PATH="./log"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='GP_BOTORCH', help='tunning method[GP_BOTORCH, WGPE]')
    parser.add_argument('--workload', type=str, default='tpcc', help='[sysbench, tpcc, workload_zoo, \
                        oltpbench_wikipedia, oltpbench_syntheticresourcestresser, oltpbench_twitter, oltpbench_tatp, \
                        oltpbench_auctionmark, oltpbench_seats, oltpbench_ycsb, oltpbench_jpab, \
                        oltpbench_ch-benchmark, oltpbench_voter, oltpbench_slbench, oltpbench_smallbank, oltpbench_linkbench]')
    parser.add_argument('--workload_type', type=str, default='write', help='[`read`, `write`, `readwrite`]')
    parser.add_argument('--knobs_config', type=str, default='', help='knobs configuration file in json format')
    parser.add_argument('--lhs_log', type=str, default='', help='log file generated from lhs for GP traning,like xxx.res')
    parser.add_argument('--lhs_num', type=int, default=10, help='mysql pid')

    # ddpg
    parser.add_argument('--params', type=str, default='', help='Load existing parameters')
    parser.add_argument('--batch_size', type=int, default=16, help='Training Batch Size')
    parser.add_argument('--metric_num', type=int, default=65, help='metric nums')
    parser.add_argument('--knobs_num', type=int, default=-1, help='knobs num')
    parser.add_argument('--mean_var_file', type=str, default='mean_var_file.plk',
                        help='mean_var file for normalizer')
    parser.add_argument('--memory', type=str, default='', help='add replay memory')
    #
    parser.add_argument('--y_variable', type=str, default='tps', help='[tps, lat]')
    parser.add_argument('--restore_state', type=str, default='', help='run_id for SMAC to restore state')
    parser.add_argument('--trials_file', type=str, default='', help='trials save file for TPE')
    parser.add_argument('--tr_init', action='store_true', dest='tr_init', default=False, help='init for trust region')
    parser.add_argument('--load_num', type=int, default=-1, help='num of loaded configs')
    parser.add_argument('--task_id', type=str, default='box_db', help='task name for openbox')
    # rgpe
    opt = parser.parse_args()

    if opt.knobs_config == '':
        err_msg = 'You must specify the knobs_config file for tuning: --knobs_config=config.json'
        logger.error(err_msg)
        raise AutotuneError(err_msg)

    # model
    model = ''
    if opt.method == 'DDPG':
        ddpg_opt = dict()
        ddpg_opt['tau'] = 0.002
        ddpg_opt['alr'] = 0.001  # 0.0005
        ddpg_opt['clr'] = 0.001  # 0.0001
        ddpg_opt['model'] = opt.params
        ddpg_opt['gamma'] = 0.9
        ddpg_opt['batch_size'] = opt.batch_size
        ddpg_opt['memory_size'] = 100000
        model = DDPG(n_states=opt.metric_num,
                     n_actions=opt.knobs_num,
                     opt=ddpg_opt,
                     ouprocess=True,
                     mean_var_path=opt.mean_var_file)
        logger.info('model initialized with {} metric and {} actions, and option is {}'.format(
            opt.metric_num, opt.knobs_num, ddpg_opt
        ))

    env = BenchEnv(workload=opt.workload,
                       knobs_config=opt.knobs_config,
                       num_metrics=65,
                       log_path=LOG_PATH,
                       knob_num=opt.knobs_num,
                       y_variable=opt.y_variable,
                       lhs_log=opt.lhs_log,
                       )

    logger.info('env initialized with the following options: {}'.format(opt))

    tuner = MySQLTuner(model=model,
                           env=env,
                           batch_size=16,
                           episodes=100,
                           replay_memory=opt.memory,
                           idx_sql='',
                           source_data_path='',
                           dst_data_path='',
                           method=opt.method,
                           lhs_log=opt.lhs_log,
                           lhs_num=opt.lhs_num,
                           y_variable=opt.y_variable,
                           restore_state=opt.restore_state,
                           trials_file=opt.trials_file,
                           tr_init=opt.tr_init)
    tuner.task_id = opt.task_id
    tuner.load_num = opt.load_num
    tuner.tune()


