import os
import time
import math
import threading
import subprocess
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import Manager
from collections import defaultdict
from typing import Any
from .knobs import gen_continuous
import pdb
import sys
import math
import copy
import joblib
from .dbconnector import MysqlConnector, PolarOConnector
from .knobs import logger
from .utils.parser import ConfigParser
from .utils.parser import parse_tpcc, parse_sysbench, parse_oltpbench, parse_cloudbench, parse_job
from .knobs import initialize_knobs, save_knobs, get_default_knobs, knob2action
from dynaconf import settings
import re
import psutil
import multiprocessing as mp
from .resource_monitor import ResourceMonitor

im_alive = mp.Value('b', False)
CPU_CORE = 8
TIMEOUT=180
TIMEOUT_CLOSE=90

if 0:
    RETRY_WAIT_TIME = 10
    RESTART_WAIT_TIME = 10
    BENCHMARK_RUNNING_TIME = 20
    DATABASE_REINIT_TIME = 600
    BENCHMARK_WARMING_TIME = 0
    REBALANCE_FREQUENCY = 5
else:
    RETRY_WAIT_TIME = 60
    RESTART_WAIT_TIME = 60
    BENCHMARK_RUNNING_TIME = 120
    DATABASE_REINIT_TIME = 600
    BENCHMARK_WARMING_TIME = 30
    REBALANCE_FREQUENCY = 10

RESTART_FREQUENCY = 20

value_type_metrics = [
    'lock_deadlocks', 'lock_timeouts', 'lock_row_lock_time_max',
    'lock_row_lock_time_avg', 'buffer_pool_size', 'buffer_pool_pages_total',
    'buffer_pool_pages_misc', 'buffer_pool_pages_data', 'buffer_pool_bytes_data',
    'buffer_pool_pages_dirty', 'buffer_pool_bytes_dirty', 'buffer_pool_pages_free',
    'trx_rseg_history_len', 'file_num_open_files', 'innodb_page_size']

dst_data_path = ' /data1/ruike/mysql/data'
src_data_path = ' /data1/ruike/mysql/data_copy'
socket_path = '/data1/ruike/mysql/base/mysql.soc*'
log_num_default = 2
log_size_default = 50331648
def generate_knobs(action, method):
    if method in ['ddpg', 'ppo', 'sac', 'gp']:
        return gen_continuous(action)
    else:
        raise NotImplementedError("Not implemented generate_knobs")



class DatabaseType(Enum):
    Mysql = 1
    Postgresql = 2


class DBEnv(ABC):
    def __init__(self, workload):
        self.score = 0.
        self.steps = 0
        self.terminate = False
        self.workload = workload

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self, knobs, episode, step):
        pass

    @abstractmethod
    def terminate(self):
        return False


class BenchEnv(DBEnv):
    def __init__(self,
                 workload,
                 knobs_config,
                 num_metrics,
                 log_path='',
                 threads=8,
                 host='localhost',
                 port=3392,
                 user='root',
                 passwd='',
                 dbname='tpcc',
                 sock='',
                 rds_mode=False,
                 workload_zoo_config='',
                 workload_zoo_app='',
                 oltpbench_config_xml='',
                 disk_name='nvme1n1',
                 tps_constraint=0,
                 latency_constraint=0,
                 pid=9999,
                 knob_num=-1,
                 y_variable='tps',
                 lhs_log='output.res'
                 ):
        super().__init__(workload)
        self.knobs_config = knobs_config
        self.mysqld = os.environ.get('MYSQLD')
        self.mycnf = os.environ.get('MYCNF')
        if not self.mysqld:
            logger.error('You should set MYSQLD env var before running the code.')
        if not self.mycnf:
            logger.error('You should set MYCNF env var before running the code.')
        self.workload = workload
        self.log_path = log_path
        self.num_metrics = num_metrics
        self.external_metricsdefault_ = []
        self.last_external_metrics = []
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
        self.sock = sock
        self.threads = threads
        self.best_result = './autotune_best.res'
        self.knobs_config = knobs_config
        self.knobs_detail = initialize_knobs(knobs_config, knob_num)
        model_path = '../tuning_benchmark/surrogate/RF_{}_{}knob.joblib'.format(self.workload.upper(), knob_num)
        functions = joblib.load(model_path)
        self.model = functions['model']
        self.encoderDir = functions['encoder']
        self.names = functions['X-name']
        self.default_knobs = get_default_knobs()
        self.rds_mode = rds_mode
        self.oltpbench_config_xml = oltpbench_config_xml
        self.step_count = 0
        self.disk_name = disk_name
        self.workload_zoo_config = workload_zoo_config
        self.workload_zoo_app = workload_zoo_app
        self.tps_constraint = tps_constraint
        self.latency_constraint = latency_constraint
        self.pre_combine_log_file_size = 0
        self.connect_sucess = True
        self.pid = pid
        self.reinit_interval = 0
        self.reinit = True
        if self.rds_mode:
            self.reinit = False
        self.y_variable = y_variable
        self.lhs_log = lhs_log

    def generate_time(self):
        global BENCHMARK_RUNNING_TIME
        global BENCHMARK_WARMING_TIME
        global TIMEOUT
        global RESTART_FREQUENCY

        if self.workload['name'] == 'sysbench' or self.workload['name'] == 'oltpbench':
            BENCHMARK_RUNNING_TIME = 120
            BENCHMARK_WARMING_TIME = 30
            TIMEOUT = BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME + 15
            RESTART_FREQUENCY = 200
        if self.workload['name'] == 'job':
            BENCHMARK_RUNNING_TIME = 240
            BENCHMARK_WARMING_TIME = 0
            TIMEOUT = BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME
            RESTART_FREQUENCY = 30000



    def get_reward(self, external_metrics, y_variable):
        """Get the reward that is used in reinforcement learning algorithm.

        The reward is calculated by tps and rt that are external metrics.
        """

        def calculate_reward(delta0, deltat):
            if delta0 > 0:
                _reward = ((1 + delta0) ** 2 - 1) * math.fabs(1 + deltat)
            else:
                _reward = - ((1 - delta0) ** 2 - 1) * math.fabs(1 - deltat)

            if _reward > 0 and deltat < 0:
                _reward = 0
            return _reward

        if sum(external_metrics) == 0 or self.default_external_metrics[0] == 0:
            # bad case, not enough time to restart mysql or bad knobs
            return 0
        # tps
        if y_variable == 'tps':
            delta_0_tps = float((external_metrics[0] - self.default_external_metrics[0])) / self.default_external_metrics[0]
            delta_t_tps = float((external_metrics[0] - self.last_external_metrics[0])) / self.last_external_metrics[0]
            reward = calculate_reward(delta_0_tps, delta_t_tps)

        # latency
        else:
            delta_0_lat = float((-external_metrics[1] + self.default_external_metrics[1])) / self.default_external_metrics[
            1]
            delta_t_lat = float((-external_metrics[1] + self.last_external_metrics[1])) / self.last_external_metrics[1]
            reward = calculate_reward(delta_0_lat, delta_t_lat)

        #reward = tps_reward * 0.95 + 0.05 * lat_reward
        #   reward = tps_reward * 0.6 + 0.4 * lat_reward
        self.score += reward

        # if reward > 0:
        #    reward = reward*1000000
        return reward

    def get_reward2(self, external_metrics):
        return float(external_metrics[0] / self.last_external_metrics[0])

    def get_reward3(self, external_metrics):
        return float(external_metrics[0] / self.default_external_metrics[0])


    def get_benchmark_cmd(self):
        timestamp = int(time.time())
        filename = self.log_path + '/{}.log'.format(timestamp)
        dirname, _ = os.path.split(os.path.abspath(__file__))
        if self.workload['name'] == 'sysbench':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_sysbench.sh',
                                              self.workload['type'],
                                              self.host,
                                              self.port,
                                              self.user,
                                              150,
                                              800000,
                                              BENCHMARK_WARMING_TIME,
                                              self.threads,
                                              BENCHMARK_RUNNING_TIME,
                                              filename,
                                              self.dbname)
            cmd = "sudo cgexec -g cpuset,memory:client " + cmd

        elif self.workload['name'] == 'tpcc':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_tpcc.sh',
                                              self.host,
                                              self.port,
                                              self.user,
                                              self.threads,
                                              BENCHMARK_WARMING_TIME,
                                              BENCHMARK_RUNNING_TIME,
                                              filename,
                                              self.dbname)
            cmd = "sudo cgexec -g cpuset,memory:client " + cmd

        elif self.workload['name'] == 'oltpbench':
            filename = filename.split('/')[-1].split('.')[0]
            cmd = self.workload['cmd'].format(dirname + '/cli/run_oltpbench.sh',
                                              self.dbname,
                                              self.oltpbench_config_xml,
                                              filename)
            cmd = "sudo cgexec -g cpuset,memory:client " + cmd
        elif self.workload['name'] == 'workload_zoo':
            filename = filename.split('/')[-1].split('.')[0]
            cmd = self.workload['cmd'].format(dirname + '/cli/run_workload_zoo.sh',
                                              self.workload_zoo_app,
                                              self.workload_zoo_config,
                                              filename)
            cmd = "sudo cgexec -g cpuset,memory:client " + cmd
        elif self.workload['name'] == 'job':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_job_mysql.sh',
                                              dirname + '/cli/selectedList.txt',
                                              dirname + '/job_query/queries-mysql-new',
                                              filename,
                                              self.sock
                                              )
            cmd = "sudo cgexec -g cpuset,memory:client " + cmd

        logger.info('[DBG]. {}'.format(cmd))
        return cmd, filename


    def get_states(self, knobs):
        x = []
        for k in self.names:
            if self.knobs_detail[k]['type'] == 'integer':
                x.append(knobs[k])
            else:
                #pdb.set_trace()
                tmp = copy.deepcopy(self.knobs_detail[k]['enum_values'])
                tmp.sort()
                v = tmp.index(str(knobs[k]))
                x.append(v)

        y_variable = self.model.predict(np.array(x).reshape(1, -1))[0]
        external_metrics = [y_variable] * 6
        internal_metrics = [0] * 65
        resource = [0] * 8
        return  external_metrics, internal_metrics, resource

    def initialize(self, collect_CPU=0):
        #return np.random.rand(65), np.random.rand(6), np.random.rand(8)
        self.score = 0.
        self.steps = 0
        self.terminate = False

        logger.info('[DBG]. default tuning knobs: {}'.format(self.default_knobs))

        s = self.get_states(self.default_knobs)

        while s == None:
            logger.info('retrying: sleep for {} seconds'.format(RETRY_WAIT_TIME))
            time.sleep(RETRY_WAIT_TIME)
            logger.info('try getting_states again')
            s = self.get_states(collect_CPU)

        external_metrics, internal_metrics, resource = s

        logger.info('[DBG]. get_state done: {}|{}|{}'.format(external_metrics, internal_metrics, resource))

        # TODO(HONG): check external_metrics[0]

        self.last_external_metrics = external_metrics
        self.default_external_metrics = external_metrics
        state = internal_metrics
        save_knobs(self.default_knobs, external_metrics)
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(self.default_knobs, str(external_metrics[0]), str(external_metrics[1]), str(external_metrics[2]),
                                external_metrics[3], external_metrics[4],
                                external_metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        logger.info("[step {}] default:{}".format(self.step_count, res))
        return state, external_metrics, resource

    def step(self, knobs, episode, step, best_action_applied=False, file=None):
        metrics, internal_metrics, resource = self.step_GP(knobs, best_action_applied)
        try:
            format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
            res = format_str.format(knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        except:
            format_str = '{}|tps_0|lat_300|qps_0|[]|65d\n'
            res = format_str.format(knobs)
        with open(file, 'a') as f:
            f.write(res)
        reward = self.get_reward(metrics, self.y_variable)
        if not (self.y_variable == 'tps' and metrics[0] <= 0) and not (self.y_variable == 'lat' and metrics[1] <= 0):
            self.last_external_metrics = metrics

        #flag = self._record_best(metrics)
        #if flag:
        #    logger.info('Better performance changed!')
        #else:
        #    logger.info('Performance remained!')
        # get the best performance so far to calculate the reward
        # best_now_performance = self._get_best_now()
        # self.last_external_metrics = best_now_performance

        next_state = internal_metrics
        # TODO(Hong)
        terminate = False

        return reward, next_state, terminate, self.score, metrics



    def step_GP(self, knobs, best_action_applied=False):
        #return np.random.rand(6), np.random.rand(65), np.random.rand(8)

        self.step_count = self.step_count + 1
        self.reinit_interval = self.reinit_interval + 1



        external_metrics, internal_metrics, resource = self.get_states(knobs)

        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(knobs, str(external_metrics[0]), str(external_metrics[1]), str(external_metrics[2]), external_metrics[3], external_metrics[4],
                                external_metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        if best_action_applied:
            logger.info("[step {}] best:{}".format(self.step_count, res))
        else:
            logger.info("[step {}] result:{}".format(self.step_count, res))
        return external_metrics, internal_metrics, resource


    def step_CobBo(self, **action_cobbo):#**knob_cobbo):
        '''knobs = knob_cobbo.copy()
        for key in knobs.keys():
            knobs[key] = math.floor(knobs[key])
        external_metrics, internal_metrics, resource = self.step_GP(knobs)'''
        action = np.zeros((len(list(action_cobbo.keys()))))
        count = 0
        for key in action_cobbo:
            action[count] = action_cobbo[key]
            count = count + 1
        knobs = generate_knobs(action, 'gp')
        external_metrics, internal_metrics, resource = self.step_GP(knobs)
        return external_metrics[0]




    def terminate(self):
        return False


    def step_SMAC(self, knobs, seed):
        #f = open(self.lhs_log, 'a')
        knobs_display = {}
        for key in knobs.keys():
            knobs_display[key] = knobs[key]
        for k in knobs_display.keys():
            if self.knobs_detail[k]['type'] == 'integer' and self.knobs_detail[k]['max'] > sys.maxsize:
                knobs_display[k] = knobs_display[k] * 1000

        logger.info('[SMAC][Episode: 1][Step: {}] knobs generated: {}'.format(self.step_count, knobs_display))
        external_metrics, internal_metrics, resource = self.step_GP(knobs_display)
        #record = "{}|{}\n".format(knobs_display, list(internal_metrics))
        #f.write(record)
        #f.close()
        f = open(self.lhs_log, 'a')
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(knobs_display, str(external_metrics[0]), str(external_metrics[1]), str(external_metrics[2]),
                                external_metrics[3], external_metrics[4],
                                external_metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        f.write(res)
        f.close()
        if self.y_variable == 'tps':
            return -external_metrics[0]
        elif self.y_variable == 'lat':
            return external_metrics[1]

    def step_TPE(self, knobs):
        knobs_display = {}
        for key in knobs.keys():
            if self.knobs_detail[key]['type'] == 'integer':
                knobs_display[key] = int(knobs[key])
                if self.knobs_detail[key]['type'] == 'integer' and self.knobs_detail[key]['max'] > sys.maxsize:
                    knobs_display[key] = knobs_display[key] * 1000
            else:
                knobs_display[key] = knobs[key]

        logger.info('[TPE][Episode: 1][Step: {}] knobs generated: {}'.format(self.step_count, knobs_display))
        metrics, internal_metrics, resource = self.step_GP(knobs_display)
        f = open(self.lhs_log, 'a')
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        f.write(res)
        f.close()

        if self.y_variable == 'tps':
            return -metrics[0]
        elif self.y_variable == 'lat':
            return metrics[1]


    def step_turbo(self, action):#**knob_cobbo):
        knobs = generate_knobs(action, 'gp')
        metrics, internal_metrics, resource = self.step_GP(knobs)
        f = open(self.lhs_log, 'a')
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        f.write(res)
        f.close()

        if self.y_variable == 'tps':
            return  -float(metrics[0])
        elif self.y_variable == 'lat':
            return float(metrics[1])


    def step(self, config):
        f = open(self.lhs_log, 'a')
        knobs = config.get_dictionary().copy()
        for k in knobs.keys():
            if self.knobs_detail[k]['type'] == 'integer' and  self.knobs_detail[k]['max'] > sys.maxsize:
                knobs[k] = knobs[k] * 1000

        metrics, internal_metrics, resource = self.step_GP(knobs)
        # record = "{}|{}\n".format(knobs, list(internal_metrics))
        # f.write(record)
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        f.write(res)
        f.close()
        if self.y_variable == 'tps':
            return  -float(metrics[0])
        elif self.y_variable == 'lat':
            return float(metrics[1])
