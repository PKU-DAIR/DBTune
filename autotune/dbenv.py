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
from .dbconnector import MysqlConnector
from .knobs import logger
from .utils.parser import ConfigParser
from .utils.parser import parse_tpcc, parse_sysbench, parse_oltpbench, parse_cloudbench, parse_job
from .knobs import initialize_knobs, save_knobs, get_default_knobs, knob2action
from dynaconf import settings
import re
import psutil
import multiprocessing as mp
from .resource_monitor import ResourceMonitor
from autotune.utils.config import knob_config
from autotune.workload import SYSBENCH_WORKLOAD, JOB_WORKLOAD, OLTPBENCH_WORKLOADS

RETRY_WAIT_TIME = 60

def generate_knobs(action, method):
    if method in ['ddpg', 'ppo', 'sac', 'gp']:
        return gen_continuous(action)
    else:
        raise NotImplementedError("Not implemented generate_knobs")

class DBEnv():
    def __init__(self, args, args_tune, db):
        self.db = db
        self.args = args
        self.workload = self.get_worklaod()
        self.log_path = "./log"
        self.num_metrics = self.db.num_metrics
        self.threads = int(args['thread_num'])
        self.best_result = './autotune_best.res'
        self.knobs_detail = initialize_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = get_default_knobs()
        self.online_mode = eval(args['online_mode'])
        self.oltpbench_config_xml =args['oltpbench_config_xml']
        self.step_count = 0
        self.tps_constraint = float(args_tune['tps_constraint'])
        self.latency_constraint = float(args_tune['latency_constraint'])
        self.connect_sucess = True
        self.reinit_interval = 0
        self.reinit = False
        if self.reinit_interval:
            self.reinit = False
        self.generate_time()
        self.y_variable = eval(args['performance_metric'])[0]
        self.lhs_log = args['lhs_log']
        self.cpu_core = args['cpu_core']

    def get_worklaod(self):
        wl = None
        dbname = self.db.dbname
        if self.args['workload'] == 'sysbench':
            wl = dict(SYSBENCH_WORKLOAD)
            wl['type'] = self.args['workload_type']
        elif self.args['workload'].startswith('oltpbench_'):
            wl = dict(OLTPBENCH_WORKLOADS)
            dbname = self.args['workload']  # strip oltpbench_
            logger.info('use database name {} by default'.format(dbname))
        # elif self.args['workload'] == 'workload_zoo':
        #     wl = dict(WORKLOAD_ZOO_WORKLOADS)
        elif self.args['workload']== 'job':
            wl = dict(JOB_WORKLOAD)

        return wl

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



    def get_external_metrics(self, filename=''):
        """Get the external metrics including tps and rt"""
        result = ''
        if self.workload['name'] == 'tpcc':
            result = parse_tpcc(filename)
        elif self.workload['name'] == 'tpcc_rds':
            result = parse_tpcc(filename)
        elif self.workload['name'] == 'sysbench':
            result = parse_sysbench(filename)
        elif self.workload['name'] == 'sysbench_rds':
            result = parse_sysbench(filename)
        elif self.workload['name'] == 'oltpbench':
            result = parse_oltpbench('results/{}.summary'.format(filename))
        elif self.workload['name'] == 'job':
            dirname, _ = os.path.split(os.path.abspath(__file__))
            select_file = dirname + '/cli/selectedList.txt'
            result = parse_job(filename, select_file)
        else:
            # logger.error('unsupported workload {}'.format(self.workload['name']))
            result = parse_cloudbench(filename)
        return result


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


    def _get_best_now(self):
        with open(self.best_result) as f:
            lines = f.readlines()
        best_now = lines[0].split(',')
        return [float(best_now[0]), float(best_now[1]), float(best_now[0])]

    def _record_best(self, external_metrics):
        best_flag = False
        if os.path.exists(self.best_result):
            tps_best = external_metrics[0]
            lat_best = external_metrics[1]
            rate = 0
            if int(lat_best) != 0:
                rate = float(tps_best) / lat_best
                with open(self.best_result) as f:
                    lines = f.readlines()
                best_now = lines[0].split(',')
                rate_best_now = float(best_now[0]) / float(best_now[1])
                if rate > rate_best_now:
                    best_flag = True
                    with open(self.best_result, 'w') as f:
                        f.write(str(tps_best) + ',' + str(lat_best) + ',' + str(rate))
        else:
            file = open(self.best_result, 'w')
            tps_best = external_metrics[0]
            lat_best = external_metrics[1]
            rate = 0
            if int(lat_best) == 0:
                rate = 0
            else:
                rate = float(tps_best) / lat_best
            file.write(str(tps_best) + ',' + str(lat_best) + ',' + str(rate))
        return best_flag


    def get_benchmark_cmd(self):
        timestamp = int(time.time())
        filename = self.log_path + '/{}.log'.format(timestamp)
        dirname, _ = os.path.split(os.path.abspath(__file__))
        if self.workload['name'] == 'sysbench':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_sysbench.sh',
                                              self.db.workload['type'],
                                              self.db.host,
                                              self.db.port,
                                              self.db.user,
                                              150,
                                              800000,
                                              BENCHMARK_WARMING_TIME,
                                              self.threads,
                                              BENCHMARK_RUNNING_TIME,
                                              filename,
                                              self.db.dbname)

        elif self.workload['name'] == 'tpcc':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_tpcc.sh',
                                              self.db.host,
                                              self.db.port,
                                              self.db.user,
                                              self.threads,
                                              BENCHMARK_WARMING_TIME,
                                              BENCHMARK_RUNNING_TIME,
                                              filename,
                                              self.dbname)
        elif self.workload['name'] == 'oltpbench':
            filename = filename.split('/')[-1].split('.')[0]
            cmd = self.workload['cmd'].format(dirname + '/cli/run_oltpbench.sh',
                                              self.db.dbname,
                                              self.oltpbench_config_xml,
                                              filename)
        elif self.workload['name'] == 'job':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_job_{}.sh'.format(self.db.args['db']),
                                              dirname + '/cli/selectedList.txt',
                                              dirname + '/job_query/queries-{}-new'.format(self.db.args['db']),
                                              filename,
                                              self.db.sock)


        logger.info('[DBG]. {}'.format(cmd))
        return cmd, filename

    def get_states(self, collect_cpu=0):
        start = time.time()
        self.connect_sucess = True
        if not self.online_mode:
            p = psutil.Process(self.db.pid)
            if len(p.cpu_affinity())!= self.cpu_core:
                command = 'sudo cgclassify -g memory,cpuset:sever ' + str(self.db.pid)
                os.system(command)

        internal_metrics = Manager().list()
        im = mp.Process(target=self.db.get_internal_metrics, args=(internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME))
        self.db.set_im_alive(True)
        im.start()
        if collect_cpu:
            rm = ResourceMonitor(self.db.pid, 1, BENCHMARK_WARMING_TIME, BENCHMARK_RUNNING_TIME)
            rm.run()
        cmd, filename = self.get_benchmark_cmd()
        # v = p.cpu_percent()
        print("[{}] benchmark start!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        p_benchmark = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
        try:
            outs, errs = p_benchmark.communicate(timeout=TIMEOUT)
            ret_code = p_benchmark.poll()
            if ret_code == 0:
                print("[{}] benchmark finished!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        except subprocess.TimeoutExpired:
            print("[{}] benchmark timeout!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        # TODO: move to database.py
        if not self.online_mode:
            if self.db.args['db'] == 'mysql':
                clear_cmd = """mysqladmin processlist -uroot -S$MYSQL_SOCK | awk '$2 ~ /^[0-9]/ {print "KILL "$2";"}' | mysql -uroot -S$MYSQL_SOCK """
            elif self.db.args['db'] == 'postgresql':
                clear_cmd = """psql -c \"select pg_terminate_backend(pid) from pg_stat_activity where datname = 'imdbload';\" """
            subprocess.Popen(clear_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
            print("[{}] clear processlist".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        self.db.set_im_alive(False)
        im.join()
        if collect_cpu:
            rm.terminate()

        if not self.connect_sucess:
            logger.info("connection failed")
            return None
        #try:
        external_metrics = self.get_external_metrics(filename)
        internal_metrics, dirty_pages, hit_ratio, page_data = self.db._post_handle(internal_metrics)
        logger.info('internal metrics: {}.'.format(list(internal_metrics)))
        ##except:
        #    logger.info("connection failed")
        #    return None
        if collect_cpu:
            monitor_data_dict = rm.get_monitor_data()
            interval = time.time() - start
            avg_read_io = sum(monitor_data_dict['io_read']) / (len(monitor_data_dict['io_read']) + 1e-9)
            avg_write_io = sum(monitor_data_dict['io_write']) / (len(monitor_data_dict['io_write']) + 1e-9)
            avg_virtual_memory = sum(monitor_data_dict['mem_virtual']) / (len(monitor_data_dict['mem_virtual']) + 1e-9)
            avg_physical_memory = sum(monitor_data_dict['mem_physical']) / (
                        len(monitor_data_dict['mem_physical']) + 1e-9)
            resource = (None, avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory, dirty_pages, hit_ratio, page_data)
            logger.info(external_metrics)
            return external_metrics, internal_metrics, resource
        else:
            return external_metrics, internal_metrics, [0]*8


    def initialize(self, collect_CPU=0):
        #return np.random.rand(65), np.random.rand(6), np.random.rand(8)
        self.score = 0.
        self.steps = 0
        self.terminate = False
        logger.info('[DBG]. default tuning knobs: {}'.format(self.default_knobs))
        if self.online_mode:
            flag = self.db.apply_rds_knobs(self.default_knobs)
        else:
            flag = self.db.apply_knobs(self.default_knobs)


        while not flag:
            logger.info('retrying: sleep for {} seconds'.format(RETRY_WAIT_TIME))
            time.sleep(RETRY_WAIT_TIME)
            logger.info('try apply default knobs again')
            if self.online_mode:
                flag = self.db.apply_rds_knobs(self.default_knobs)
            else:
                flag = self.db.apply_knobs(self.default_knobs)

        logger.info('[DBG]. apply default knobs done')

        s = self.get_states(collect_CPU)

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

    def get_reward_cpu(self, tps, cpu):
        """Get the reward that is used in reinforcement learning algorithm.

        The reward is calculated by tps and rt that are external metrics.
        """

        def calculate_reward(delta0, deltat, tps):
            if delta0 > 0:
                _reward = ((1 + delta0) ** 2 - 1) * math.fabs(1 + deltat)
            else:
                _reward = - ((1 - delta0) ** 2 - 1) * math.fabs(1 - deltat)

            if _reward > 0 and deltat < 0:
                _reward = 0
            if _reward < 0 and tps - self.constraint > 0:
                _reward = 0
            if _reward > 0 and tps - self.constraint < 0:
                _reward = 0
            return _reward

        if tps == 0 or cpu == 0:
            # bad case, not enough time to restart mysql or bad knobs
            return 0
        # latency
        delta_0_lat = float((-cpu + self.default_cpu)) / self.default_cpu
        delta_t_lat = float((-cpu + self.last_cpu)) / self.last_cpu
        reward = calculate_reward(delta_0_lat, delta_t_lat, tps)
        return reward

    def get_reward_cpu_latency(self, tps, cpu, latency):
        """Get the reward that is used in reinforcement learning algorithm.

        The reward is calculated by tps and rt that are external metrics.
        """

        def calculate_reward(delta0, deltat, tps, latency):
            if delta0 > 0:
                _reward = ((1 + delta0) ** 2 - 1) * math.fabs(1 + deltat)
            else:
                _reward = - ((1 - delta0) ** 2 - 1) * math.fabs(1 - deltat)

            if _reward > 0 and deltat < 0:
                _reward = 0
            if _reward < 0 and (tps - self.tps_constraint > 0 and latency - self.latency_constraint < 0 ):
                _reward = 0
            if _reward > 0 and (tps - self.tps_constraint < 0 or latency - self.latency_constraint > 0):
                _reward = 0
            return _reward

        if tps == 0 or cpu == 0:
            # bad case, not enough time to restart mysql or bad knobs
            return 0
        # latency
        delta_0_lat = float((-cpu + self.default_cpu)) / self.default_cpu
        delta_t_lat = float((-cpu + self.last_cpu)) / self.last_cpu
        reward = calculate_reward(delta_0_lat, delta_t_lat, tps, latency)
        return reward


    def step_GP(self, knobs, best_action_applied=False):
        #return np.random.rand(6), np.random.rand(65), np.random.rand(8)

        if self.reinit_interval > 0 and self.reinit_interval % RESTART_FREQUENCY == 0:
            if self.reinit:
                logger.info('reinitializing database begin')
                self.reinitdb_magic()
                logger.info('database reinitialized')
        self.step_count = self.step_count + 1
        self.reinit_interval = self.reinit_interval + 1
        for key in knobs.keys():
            value = knobs[key]
            if not key in self.knobs_detail.keys() or  not self.knobs_detail[key]['type'] == 'integer':
                continue
            if value > self.knobs_detail[key]['max']:
                knobs[key] = self.knobs_detail[key]['max']
                logger.info("{} with value of is larger than max, adjusted".format(key))
            elif value < self.knobs_detail[key]['min']:
                knobs[key] = self.knobs_detail[key]['min']
                logger.info("{} with value of is smaller than min, adjusted".format(key))
        logger.info("[step {}] generate knobs: {}\n".format(self.step_count, knobs))
        collect_resource = False
        if self.online_mode:
            try:
                flag = self.db.apply_rds_knobs(knobs)
            except:
                flag = self.db.apply_knobs(knobs)
        else:
            flag = self.db.apply_knobs(knobs)

        if not flag:
            if best_action_applied:
                logger.info("[step {}] best:{}|tps_0|lat_300|[]|65d\n".format(self.step_count,knobs))
            else:
                logger.info("[step {}] result:{}|tps_0|lat_300|[]|65d\n".format(self.step_count, knobs))
            if self.reinit:
                logger.info('reinitializing database begin')
                self.reinitdb_magic()
                logger.info('database reinitialized')
            return [-1, 300, -1 ], np.array([0]*65), 0

        s = self.get_states(collect_resource)

        if s == None:
            if best_action_applied:
                logger.info("[step {}] best:{}|tps_0|[]|65d\n".format(self.step_count,knobs))
            else:
                logger.info("[step {}] result:{}|tps_0|[]|65d\n".format(self.step_count, knobs))
            if self.reinit:
                logger.info('reinitializing database begin')
                self.reinitdb_magic()
                logger.info('database reinitialized')
            return [-1, 300, -1 ], np.array([0]*65), 0

        external_metrics, internal_metrics, resource = s

        '''while external_metrics[0] == 0 or sum(internal_metrics) == 0:
            logger.info('retrying because got invalid metrics. Sleep for {} seconds.'.format(RETRY_WAIT_TIME))
            time.sleep(RETRY_WAIT_TIME)
            logger.info('try get_states again')
            external_metrics, internal_metrics ,resource= self.get_states(collect_resource)
            logger.info('metrics got again, {}|{}'.format(external_metrics, internal_metrics))'''

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
        f = open(self.lhs_log, 'a')
        knobs_display = {}
        for key in knobs.keys():
            knobs_display[key] = knobs[key]
        for k in knobs_display.keys():
            if self.knobs_detail[k]['type'] == 'integer' and self.knobs_detail[k]['max'] > sys.maxsize:
                knobs_display[k] = knobs_display[k] * 1000

        logger.info('[SMAC][Episode: 1][Step: {}] knobs generated: {}'.format(self.step_count, knobs_display))
        metrics, internal_metrics, resource = self.step_GP(knobs_display)
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
        external_metrics, internal_metrics, resource = self.step_GP(knobs_display)

        if self.y_variable == 'tps':
            return -external_metrics[0]
        elif self.y_variable == 'lat':
            return external_metrics[1]


    def step_turbo(self, action):#**knob_cobbo):
        knobs = generate_knobs(action, 'gp')
        external_metrics, internal_metrics, resource = self.step_GP(knobs)
        if self.y_variable == 'tps':
            return  -float(external_metrics[0])
        elif self.y_variable == 'lat':
            return float(external_metrics[1])


    def step_openbox(self, config):
        f = open(self.lhs_log, 'a')
        knobs = config.get_dictionary().copy()
        for k in knobs.keys():
            if self.knobs_detail[k]['type'] == 'integer' and  self.knobs_detail[k]['max'] > sys.maxsize:
                knobs[k] = knobs[k] * 1000

        metrics, internal_metrics, resource = self.step_GP(knobs)
        # record = "{}|{}\n".format(knobs, list(internal_metrics))
        # f.write(record)
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|{}d\n'
        res = format_str.format(knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics), self.db.num_metrics)
        f.write(res)
        f.close()

        if self.y_variable == 'tps':
            obj = -float(metrics[0])
        elif self.y_variable == 'lat':
            obj = float(metrics[1])

        constraint = None

        res = {
            'tps': metrics[0],
            'lat': metrics[1],
            'qps': metrics[2],
            'tpsVar': metrics[3],
            'latVar': metrics[4],
            'qpsVar': metrics[5],
            'cpu': resource[0],
            'readIO': resource[1],
            'writeIO': resource[2],
            'virtualMem': resource[3],
            'physical': resource[4],
            'dirty': resource[5],
            'hit': resource[6],
            'data': resource[7],
            'IM': list(internal_metrics)
        }

        return (obj, ), constraint, res
