import logging
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
from .dbconnector import MysqlConnector, PostgresqlConnector
from .knobs import logger
from .utils.parser import ConfigParser
from .utils.parser import parse_tpcc, parse_sysbench, parse_oltpbench, parse_cloudbench, parse_job
from .knobs import initialize_knobs, save_knobs, get_default_knobs, knob2action
from dynaconf import settings
import re
import psutil
import multiprocessing as mp
from .resource_monitor import ResourceMonitor
from autotune.workload import SYSBENCH_WORKLOAD, JOB_WORKLOAD, OLTPBENCH_WORKLOADS

dst_data_path = os.environ.get("DATADST")
src_data_path = os.environ.get("DATASRC")
log_num_default = 2
log_size_default = 50331648

RESTART_WAIT_TIME = 20
TIMEOUT_CLOSE = 60


class MysqlDB():
    def __init__(self, args):
        self.args = args
        self.mysqld = os.environ.get('MYSQLD')
        if not self.mysqld:
            logger.error('You should set MYSQLD env var before running the code.')

        self.mycnf = os.environ.get('MYCNF')
        if not self.mycnf:
            logger.error('You should set MYCNF env var before running the code.')
        self.host = args['host']
        self.port = args['port']
        self.user = args['user']
        self.passwd = args['passwd']
        self.dbname = args['dbname']
        self.sock = args['sock']
        self.pid = args['pid']
        self.num_metrics = 65
        self.value_type_metrics = [
            'lock_deadlocks', 'lock_timeouts', 'lock_row_lock_time_max',
            'lock_row_lock_time_avg', 'buffer_pool_size', 'buffer_pool_pages_total',
            'buffer_pool_pages_misc', 'buffer_pool_pages_data', 'buffer_pool_bytes_data',
            'buffer_pool_pages_dirty', 'buffer_pool_bytes_dirty', 'buffer_pool_pages_free',
            'trx_rseg_history_len', 'file_num_open_files', 'innodb_page_size'
        ]
        self.knobs_detail = initialize_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = get_default_knobs()
        self.pre_combine_log_file_size = log_num_default * log_size_default


        if args['workload']  == 'sysbench':
            wl = dict(SYSBENCH_WORKLOAD)
            wl['type'] = args['workload_type']
        elif args['workload'] .startswith('oltpbench_'):
            wl = dict(OLTPBENCH_WORKLOADS)
            dbname = args['workload'] [10:] # strip oltpbench_
            logger.info('use database name {} by default'.format(dbname))
        elif args['workload']  == 'job':
            wl = dict(JOB_WORKLOAD)

        self.workload = wl
        self.im_alive_init()


    def _gen_config_file(self, knobs):
        cnf_parser = ConfigParser(self.mycnf)
        # for k, v in knobs.items():
        #    cnf_parser.set(k, v)
        konbs_not_in_mycnf = []
        for key in knobs.keys():
            if not key in self.knobs_detail.keys():
                konbs_not_in_mycnf.append(key)
                continue
            cnf_parser.set(key, knobs[key])
        cnf_parser.replace()
        logger.info('generated config file done')
        return konbs_not_in_mycnf


    def _kill_mysqld(self):
        mysqladmin = os.path.dirname(self.mysqld) + '/mysqladmin'
        cmd = '{} -u{} -S {} shutdown'.format(mysqladmin, self.user, self.sock)
        p_close = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
        try:
            outs, errs = p_close.communicate(timeout=TIMEOUT_CLOSE)
            ret_code = p_close.poll()
            if ret_code == 0:
                print("Close database successfully")
        except subprocess.TimeoutExpired:
            print("Force close!")
            os.system("ps aux|grep '" + self.sock + "'|awk '{print $2}'|xargs kill -9")
            os.system("ps aux|grep '" + self.mycnf + "'|awk '{print $2}'|xargs kill -9")
        logger.info('mysql is shut down')

    def _start_mysqld(self):
        proc = subprocess.Popen([self.mysqld, '--defaults-file={}'.format(self.mycnf)])
        self.pid = proc.pid
        command = 'sudo cgclassify -g memory,cpuset:sever ' + str(self.pid)
        p = os.system(command)
        if not p:
            logger.info('add {} to memory,cpuset:sever'.format(self.pid))
        else:
            logger.info('Failed: add {} to memory,cpuset:sever'.format(self.pid))
        # os.popen("sudo -S %s" % (command), 'w').write('mypass')
        count = 0
        start_sucess = True
        logger.info('wait for connection')
        error, db_conn = None, None
        while True:
            try:

                dbc = MysqlConnector(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         passwd=self.passwd,
                                         name=self.dbname,
                                         socket=self.sock)
                db_conn = dbc.conn
                if db_conn.is_connected():
                    logger.info('Connected to MySQL database')
                    db_conn.close()
                    break
            except:
                pass

            time.sleep(1)
            count = count + 1
            if count > 600:
                start_sucess = False
                logger.info("can not connect to DB")
                break

        logger.info('finish {} seconds waiting for connection'.format(count))
        logger.info('{} --defaults-file={}'.format(self.mysqld, self.mycnf))
        logger.info('mysql is up')
        return start_sucess


    def reinitdb_magic(self):
        self._kill_mysqld()
        time.sleep(10)
        os.system('rm -rf {}'.format(self.sock))
        os.system('rm -rf {}'.format(dst_data_path))  # avoid moving src into dst
        logger.info('remove all files in {}'.format(dst_data_path))
        os.system('cp -r {} {}'.format(src_data_path, dst_data_path))
        logger.info('cp -r {} {}'.format(src_data_path, dst_data_path))
        self.pre_combine_log_file_size = log_num_default * log_size_default
        self.apply_knobs(self.default_knobs)
        self.reinit_interval = 0

    def apply_rds_knobs(self, knobs):

        # self.restart_rds()
        # apply knobs remotely
        db_conn = MysqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)
        if 'innodb_io_capacity' in knobs.keys():
            self.set_rds_param(db_conn, 'innodb_io_capacity_max', 2 * int(knobs['innodb_io_capacity']))
        # for k, v in knobs.items():
        #   self.set_rds_param(db_conn, k, v)

        for key in knobs.keys():
            self.set_rds_param(db_conn, key, knobs[key])
        db_conn.close_db()
        return True


    def apply_knobs(self, knobs):
        self._kill_mysqld()
        modify_concurrency = False
        if 'innodb_thread_concurrency' in knobs.keys() and knobs['innodb_thread_concurrency'] * (
                200 * 1024) > self.pre_combine_log_file_size:
            true_concurrency = knobs['innodb_thread_concurrency']
            modify_concurrency = True
            knobs['innodb_thread_concurrency'] = int(self.pre_combine_log_file_size / (200 * 1024.0)) - 2
            logger.info("modify innodb_thread_concurrency")

        if 'innodb_log_file_size' in knobs.keys():
            log_size = knobs['innodb_log_file_size']
        else:
            log_size = log_size_default
        if 'innodb_log_files_in_group' in knobs.keys():
            log_num = knobs['innodb_log_files_in_group']
        else:
            log_num = log_num_default

        if 'innodb_thread_concurrency' in knobs.keys() and knobs['innodb_thread_concurrency'] * (
                200 * 1024) > log_num * log_size:
            logger.info("innodb_thread_concurrency is set too large")
            return False

        knobs_rdsL = self._gen_config_file(knobs)
        sucess = self._start_mysqld()
        try:
            logger.info('sleeping for {} seconds after restarting mysql'.format(RESTART_WAIT_TIME))
            time.sleep(RESTART_WAIT_TIME)

            db_conn = MysqlConnector(host=self.host,
                                     port=self.port,
                                     user=self.user,
                                     passwd=self.passwd,
                                     name=self.dbname,
                                     socket=self.sock)
            sql1 = 'SHOW VARIABLES LIKE "innodb_log_file_size";'
            sql2 = 'SHOW VARIABLES LIKE "innodb_log_files_in_group";'
            r1 = db_conn.fetch_results(sql1)
            file_size = r1[0]['Value'].strip()
            r2 = db_conn.fetch_results(sql2)
            file_num = r2[0]['Value'].strip()
            self.pre_combine_log_file_size = int(file_num) * int(file_size)
            if len(knobs_rdsL) > 0:
                tmp_rds = {}
                for knob_rds in knobs_rdsL:
                    tmp_rds[knob_rds] = knobs[knob_rds]
                self.apply_rds_knobs(tmp_rds)
            if modify_concurrency:
                tmp = {}
                tmp['innodb_thread_concurrency'] = true_concurrency
                self.apply_rds_knobs(tmp)
                knobs['innodb_thread_concurrency'] = true_concurrency
        except:
            sucess = False

        return sucess

    def _check_apply(self, db_conn, k, v, v0, IsSession=False):
        if IsSession:
            sql = 'SHOW VARIABLES LIKE "{}";'.format(k)
            r = db_conn.fetch_results(sql)
            if r[0]['Value'] == 'ON':
                vv = 1
            elif r[0]['Value'] == 'OFF':
                vv = 0
            else:
                vv = r[0]['Value'].strip()
            if vv == v0:
                return False
            return True

        sql = 'SHOW GLOBAL VARIABLES LIKE "{}";'.format(k)
        r = db_conn.fetch_results(sql)
        if r[0]['Value'] == 'ON':
            vv = 1
        elif r[0]['Value'] == 'OFF':
            vv = 0
        else:
            vv = r[0]['Value'].strip()
        if vv == v0:
            return False
        return True

    def set_rds_param(self, db_conn, k, v):
        sql = 'SHOW GLOBAL VARIABLES LIKE "{}";'.format(k)
        r = db_conn.fetch_results(sql)
        if v == 'ON':
            v = 1
        elif v == 'OFF':
            v = 0
        if r[0]['Value'] == 'ON':
            v0 = 1
        elif r[0]['Value'] == 'OFF':
            v0 = 0
        else:
            try:
                v0 = eval(r[0]['Value'])
            except:
                v0 = r[0]['Value'].strip()
        if v0 == v:
            return True

        IsSession = False
        if str(v).isdigit():
            sql = "SET GLOBAL {}={}".format(k, v)
        else:
            sql = "SET GLOBAL {}='{}'".format(k, v)
        try:
            db_conn.execute(sql)
        except:
            logger.info("Failed: execute {}".format(sql))
            IsSession = True
            if str(v).isdigit():
                sql = "SET {}={}".format(k, v)
            else:
                sql = "SET {}='{}'".format(k, v)
            db_conn.execute(sql)
        while not self._check_apply(db_conn, k, v, v0, IsSession):
            time.sleep(1)
        return True

    def im_alive_init(self):
        global im_alive
        im_alive = mp.Value('b', True)

    def set_im_alive(self, value):
        im_alive.value = value


    def get_internal_metrics(self, internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME):
        """Get the all internal metrics of MySQL, like io_read, physical_read.

        This func uses a SQL statement to lookup system table: information_schema.INNODB_METRICS
        and returns the lookup result.
        """
        self.connect_sucess = True
        _counter = 0
        _period = 5
        count = (BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME) / _period - 1
        warmup = BENCHMARK_WARMING_TIME / _period

        # self.reset_internal_state()
        # double reset
        # TODO: fix this
        # self.reset_internal_state()

        def collect_metric(counter):
            counter += 1
            print(counter)
            timer = threading.Timer(float(_period), collect_metric, (counter,))
            timer.start()
            if counter >= count or not im_alive.value:
                timer.cancel()
            try:
                db_conn = MysqlConnector(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         passwd=self.passwd,
                                         name=self.dbname,
                                         socket=self.sock)
            except:
                if counter > warmup:
                    self.connect_sucess = False
                    logger.info("connection failed during internal metrics collection")
                    return

            try:
                if counter > warmup:

                    sql = 'SELECT NAME, COUNT from information_schema.INNODB_METRICS where status="enabled" ORDER BY NAME'
                    res = db_conn.fetch_results(sql, json=False)
                    res_dict = {}
                    for (k, v) in res:
                        # if not k in BLACKLIST:
                        res_dict[k] = v
                    internal_metrics.append(res_dict)

            except Exception as err:
                self.connect_sucess = False
                logger.info("connection failed during internal metrics collection")
                logger.info(err)

        collect_metric(_counter)
        # f.close()
        return internal_metrics

    def _post_handle(self, metrics):
        result = np.zeros(65)

        def do(metric_name, metric_values):
            metric_type = 'counter'
            if metric_name in self.value_type_metrics:
                metric_type = 'value'
            if metric_type == 'counter':
                return float(metric_values[-1] - metric_values[0]) * 23 / len(metric_values)
            else:
                return float(sum(metric_values)) / len(metric_values)

        keys = list(metrics[0].keys())
        keys.sort()
        total_pages = 0
        dirty_pages = 0
        request = 0
        reads = 0
        page_data = 0
        page_size = 0
        page_misc = 0
        for idx in range(len(keys)):
            key = keys[idx]
            data = [x[key] for x in metrics]
            result[idx] = do(key, data)
            if key == 'buffer_pool_pages_total':
                total_pages = result[idx]
            elif key == 'buffer_pool_pages_dirty':
                dirty_pages = result[idx]
            elif key == 'buffer_pool_read_requests':
                request = result[idx]
            elif key == 'buffer_pool_reads':
                reads = result[idx]
            elif key == 'buffer_pool_pages_data':
                page_data = result[idx]
            elif key == 'innodb_page_size':
                page_size = result[idx]
            elif key == 'buffer_pool_pages_misc':
                page_misc = result[idx]
        dirty_pages_per = dirty_pages / total_pages
        hit_ratio = request / float(request + reads)
        page_data = (page_data + page_misc) * page_size / (1024.0 * 1024.0 * 1024.0)

        return result, dirty_pages_per, hit_ratio, page_data


    def get_db_size(self):
        db_conn = MysqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)
        sql = 'SELECT CONCAT(round(sum((DATA_LENGTH + index_length) / 1024 / 1024), 2), "MB") as data from information_schema.TABLES where table_schema="{}"'.format(
            self.dbname)
        res = db_conn.fetch_results(sql, json=False)
        db_size = float(res[0][0][:-2])
        db_conn.close_db()
        return db_size




class PostgresqlDB():
    def __init__(self, args):
        self.args = args
        self.pg_ctl = os.environ.get('PG_CTL')
        if not self.pg_ctl:
            logger.error('You should set PG_CTL env var before running the code.')

        self.pgdata = os.environ.get('PGDATA')
        if not self.pgdata:
            logger.error('You should set PGDATA env var before running the code.')

        self.pgcnf = os.environ.get('PGCNF')
        if not self.pgcnf:
            logger.error('You should set PGCNF env var before running the code.')

        self.host = args['host']
        self.port = args['port']
        self.user = args['user']
        self.passwd = args['passwd']
        self.dbname = args['dbname']
        self.sock = args['sock']
        self.pid = args['pid']
        self.num_metrics = 60
        self.PG_STAT_VIEWS = [
            "pg_stat_archiver", "pg_stat_bgwriter",             # global
            "pg_stat_database", "pg_stat_database_conflicts",   # local
            "pg_stat_user_tables", "pg_statio_user_tables",     # local
            "pg_stat_user_indexes", "pg_statio_user_indexes"    # local
        ]
        self.PG_STAT_VIEWS_LOCAL_DATABASE = [ "pg_stat_database", "pg_stat_database_conflicts"]
        self.PG_STAT_VIEWS_LOCAL_TABLE = ["pg_stat_user_tables", "pg_statio_user_tables"]
        self.PG_STAT_VIEWS_LOCAL_INDEX = ["pg_stat_user_indexes", "pg_statio_user_indexes"]
        self.NUMERIC_METRICS = [ # counter
            # global
            'buffers_alloc', 'buffers_backend', 'buffers_backend_fsync', 'buffers_checkpoint', 'buffers_clean',
            'checkpoints_req', 'checkpoints_timed', 'checkpoint_sync_time', 'checkpoint_write_time', 'maxwritten_clean',
            'archived_count', 'failed_count',
            # database
            'blk_read_time', 'blks_hit', 'blks_read', 'blk_write_time', 'conflicts', 'deadlocks', 'temp_bytes',
            'temp_files', 'tup_deleted', 'tup_fetched', 'tup_inserted', 'tup_returned', 'tup_updated', 'xact_commit',
            'xact_rollback', 'confl_tablespace', 'confl_lock', 'confl_snapshot', 'confl_bufferpin', 'confl_deadlock',
            # table
            'analyze_count', 'autoanalyze_count', 'autovacuum_count', 'heap_blks_hit', 'heap_blks_read', 'idx_blks_hit',
            'idx_blks_read', 'idx_scan', 'idx_tup_fetch', 'n_dead_tup', 'n_live_tup', 'n_tup_del', 'n_tup_hot_upd',
            'n_tup_ins', 'n_tup_upd', 'n_mod_since_analyze', 'seq_scan', 'seq_tup_read', 'tidx_blks_hit', 'tidx_blks_read',
            'toast_blks_hit', 'toast_blks_read', 'vacuum_count',
            # index
            'idx_blks_hit', 'idx_blks_read', 'idx_scan', 'idx_tup_fetch', 'idx_tup_read'
        ]

        self.knobs_detail = initialize_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = get_default_knobs()

        if args['workload'] == 'job':
            wl = dict(JOB_WORKLOAD)
        # elif args['workload'] == 'sysbench':
        #     wl = dict(SYSBENCH_WORKLOAD)
        #     wl['type'] = args['workload_type']
        # elif args['workload'].startswith('oltpbench_'):
        #     wl = dict(OLTPBENCH_WORKLOADS)
        #     dbname = args['workload'][10:]  # strip oltpbench_
        #     logger.info('use database name {} by default'.format(dbname))

        self.workload = wl
        self.im_alive_init()


    def _kill_postgres(self):
        cmd = 'pg_ctl stop'
        p_close = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
        try:
            outs, errs = p_close.communicate(timeout=TIMEOUT_CLOSE)
            ret_code = p_close.poll()
            if ret_code == 0:
                print("Close database successfully")
        except subprocess.TimeoutExpired:
            print("Force close!")
            os.system("ps aux|grep '" + self.pgcnf + "'|awk '{print $2}'|xargs kill -9")
        logger.info('postgresql is shut down')

    def _start_postgres(self):
        # proc = subprocess.Popen(['pg_ctl', 'start', '-o "--config_file={}"'.format(self.pgcnf)])
        proc = subprocess.Popen(['postgres', '--config_file={}'.format(self.pgcnf)])
        self.pid = proc.pid
        command = 'sudo cgclassify -g memory,cpuset:sever ' + str(self.pid)
        p = os.system(command)
        if not p:
            logger.info('add {} to memory,cpuset:server'.format(self.pid))
        else:
            logger.info('Failed: add {} to memory,cpuset:server'.format(self.pid))
        # os.popen("sudo -S %s" % (command), 'w').write('mypass')
        count = 0
        start_sucess = True
        logger.info('wait for connection')
        error, db_conn = None, None
        while True:
            try:
                dbc = PostgresqlConnector(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         passwd=self.passwd,
                                         name=self.dbname,
                                         socket=self.sock)
                db_conn = dbc.conn
                if db_conn.closed == 0:
                    logger.info('Connected to PostgreSQL database')
                    db_conn.close()
                    break
            except:
                pass

            time.sleep(1)
            count = count + 1
            if count > 90:
                start_sucess = False
                logger.info("can not connect to DB")
                clear_cmd = """ps -ef|grep postgres|grep -v grep|cut -c 9-15|xargs kill -9"""
                subprocess.Popen(clear_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                 close_fds=True)
                logger.info("kill all postgres process")
                break

        logger.info('finish {} seconds waiting for connection'.format(count))
        # logger.info('pg_ctl start -o "--config_file{}"'.format(self.pgcnf))
        logger.info('postgres --config_file={}'.format(self.pgcnf))
        logger.info('postgres is up')
        return start_sucess


    def reinitdb_magic(self):
        self._kill_postgres()
        time.sleep(10)
        # os.system('rm -rf {}'.format(self.sock))
        os.system('rm -rf {}'.format(dst_data_path))  # avoid moving src into dst
        logger.info('remove all files in {}'.format(dst_data_path))
        os.system('cp -r {} {}'.format(src_data_path, dst_data_path))
        logger.info('cp -r {} {}'.format(src_data_path, dst_data_path))
        # self.pre_combine_log_file_size = log_num_default * log_size_default
        self.apply_knobs(self.default_knobs)
        self.reinit_interval = 0

    def apply_rds_knobs(self, knobs):
        # self.restart_rds()
        # apply knobs remotely
        db_conn = PostgresqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)

        for key in knobs.keys():
            self.set_rds_param(db_conn, key, knobs[key])
        db_conn.close_db()
        return True

    def _gen_config_file(self, knobs):
        cnf_parser = ConfigParser(self.pgcnf)
        knobs_not_in_pgcnf = []
        for key in knobs.keys():
            # if not key in self.knobs_detail.keys():
            #     knobs_not_in_pgcnf.append(key)
            #     continue
            cnf_parser.set(key, knobs[key])
        cnf_parser.replace()
        logger.info('generating config file done')
        return knobs_not_in_pgcnf

    def apply_knobs(self, knobs):
        self._kill_postgres()

        if 'min_wal_size' in knobs.keys():
            if 'wal_segment_size' in knobs.keys():
                wal_segment_size = knobs['wal_segment_size']
            else:
                wal_segment_size = 16
            if knobs['min_wal_size'] < 2 * wal_segment_size:
                knobs['min_wal_size'] = 2 * wal_segment_size
                logging.info('"min_wal_size" must be at least twice "wal_segment_size"')

        knobs_rdsL = self._gen_config_file(knobs)
        sucess = self._start_postgres()
        try:
            logger.info('sleeping for {} seconds after restarting postgres'.format(RESTART_WAIT_TIME))
            time.sleep(RESTART_WAIT_TIME)

            db_conn = PostgresqlConnector(host=self.host,
                                     port=self.port,
                                     user=self.user,
                                     passwd=self.passwd,
                                     name=self.dbname,
                                     socket=self.sock)

            if len(knobs_rdsL) > 0:
                tmp_rds = {}
                for knob_rds in knobs_rdsL:
                    tmp_rds[knob_rds] = knobs[knob_rds]
                self.apply_rds_knobs(tmp_rds)
        except:
            sucess = False

        return sucess

    def _check_apply(self, db_conn, k, v, v0, IsSession=False):
        if IsSession:
            sql = 'SHOW {};'.format(k)
            r = db_conn.fetch_results(sql)
            if r[0]['Value'] == 'ON':
                vv = 1
            elif r[0]['Value'] == 'OFF':
                vv = 0
            else:
                vv = r[0]['Value'].strip()
            if vv == v0:
                return False
            return True

        sql = 'SHOW {};'.format(k)
        r = db_conn.fetch_results(sql)
        if r[0]['Value'] == 'ON':
            vv = 1
        elif r[0]['Value'] == 'OFF':
            vv = 0
        else:
            vv = r[0]['Value'].strip()
        if vv == v0:
            return False
        return True

    def set_rds_param(self, db_conn, k, v):
        sql = 'SHOW {};'.format(k)
        r = db_conn.fetch_results(sql)
        if v == 'ON':
            v = 1
        elif v == 'OFF':
            v = 0
        if r[0]['Value'] == 'ON':
            v0 = 1
        elif r[0]['Value'] == 'OFF':
            v0 = 0
        else:
            try:
                v0 = eval(r[0]['Value'])
            except:
                v0 = r[0]['Value'].strip()
        if v0 == v:
            return True

        IsSession = False
        if str(v).isdigit():
            sql = "SET {}={}".format(k, v)
        else:
            sql = "SET {}='{}'".format(k, v)
        try:
            db_conn.execute(sql)
        except:
            logger.info("Failed: execute {}".format(sql))
            IsSession = True
            if str(v).isdigit():
                sql = "SET {}={}".format(k, v)
            else:
                sql = "SET {}='{}'".format(k, v)
            db_conn.execute(sql)
        while not self._check_apply(db_conn, k, v, v0, IsSession):
            time.sleep(1)
        return True

    def im_alive_init(self):
        global im_alive
        im_alive = mp.Value('b', True)

    def set_im_alive(self, value):
        im_alive.value = value


    def get_internal_metrics(self, internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME):
        """Get the all internal metrics of MySQL, like io_read, physical_read.

        This func uses a SQL statement to lookup system table: information_schema.INNODB_METRICS
        and returns the lookup result.
        """
        self.connect_sucess = True
        _counter = 0
        _period = 5
        count = (BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME) / _period - 1
        warmup = BENCHMARK_WARMING_TIME / _period


        def collect_metric(counter):
            counter += 1
            print(counter)
            timer = threading.Timer(float(_period), collect_metric, (counter,))
            timer.start()
            if counter >= count or not im_alive.value:
                print('clock shutdown')
                timer.cancel()
            try:
                db_conn = PostgresqlConnector(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         passwd=self.passwd,
                                         name=self.dbname,
                                         socket=self.sock)
            except:
                if counter > warmup:
                    self.connect_sucess = False
                    logger.info("connection failed during internal metrics collection")
                    return

            try:
                if counter > warmup:
                    metrics_dict = {
                        'global': {},
                        'local': {
                            'database': {},
                            'table': {},
                            'index': {}
                        }
                    }
                    for view in self.PG_STAT_VIEWS:
                        sql = 'SELECT * from {}'.format(view)
                        results = db_conn.fetch_results(sql, json=True)
                        if view in ["pg_stat_archiver", "pg_stat_bgwriter"]:
                            metrics_dict['global'][view] = results[0]
                        else:
                            if view in self.PG_STAT_VIEWS_LOCAL_DATABASE:
                                type = 'database'
                                type_key = 'datname'
                            elif view in self.PG_STAT_VIEWS_LOCAL_TABLE:
                                type ='table'
                                type_key = 'relname'
                            elif view in self.PG_STAT_VIEWS_LOCAL_INDEX:
                                type = 'index'
                                type_key = 'relname'
                            metrics_dict['local'][type][view] = {}
                            for res in results:
                                type_name = res[type_key]
                                metrics_dict['local'][type][view][type_name] = res

                    metrics = {}
                    for scope, sub_vars in list(metrics_dict.items()):
                        if scope == 'global':
                            metrics.update(self.parse_helper(metrics, sub_vars))
                        elif scope == 'local':
                            for _, viewnames in list(sub_vars.items()):
                                for viewname, objnames in list(viewnames.items()):
                                    for _, view_vars in list(objnames.items()):
                                        metrics.update(self.parse_helper(metrics, {viewname: view_vars}))

                    # Combine values
                    valid_metrics = {}
                    for name, values in list(metrics.items()):
                        if name.split('.')[-1] in self.NUMERIC_METRICS:
                            values = [float(v) for v in values if v is not None]
                            if len(values) == 0:
                                valid_metrics[name] = 0
                            else:
                                valid_metrics[name] = str(sum(values))

                    internal_metrics.append(valid_metrics)

            except Exception as err:
                self.connect_sucess = False
                logger.info("connection failed during internal metrics collection")
                logger.info(err)

        collect_metric(_counter)
        # f.close()
        return internal_metrics

    def parse_helper(self, valid_variables, view_variables):
        for view_name, variables in list(view_variables.items()):
            for var_name, var_value in list(variables.items()):
                full_name = '{}.{}'.format(view_name, var_name)
                if full_name not in valid_variables:
                    valid_variables[full_name] = []
                valid_variables[full_name].append(var_value)
        return valid_variables

    def _post_handle(self, metrics):
        result = np.zeros(self.num_metrics)
        return result, 0, 0, 0

        def do(metric_name, metric_values):
            return float(float(metric_values[-1]) - float(metric_values[0])) / len(metric_values)

        result = np.zeros(self.num_metrics)
        keys = list(metrics[0].keys())
        keys.sort()
        for idx in range(len(keys)):
            key = keys[idx]
            data = [x[key] for x in metrics]
            result[idx] = do(key, data)
        return result, 0, 0, 0


    def get_db_size(self):
        db_conn = PostgresqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)
        sql = "select pg_database_size('{}')/1024/1024;".format(self.dbname)
        res = db_conn.fetch_results(sql, json=False)
        db_size = float(res[0][0])
        db_conn.close_db()
        return db_size
