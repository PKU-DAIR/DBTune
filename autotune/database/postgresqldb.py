import os
import time
import threading
import subprocess
import paramiko
import logging
import numpy as np
import multiprocessing as mp
from getpass import getpass
from autotune.dbconnector import PostgresqlConnector
from autotune.knobs import logger
from autotune.utils.parser import ConfigParser
from autotune.knobs import initialize_knobs, get_default_knobs

dst_data_path = os.environ.get("DATADST")
src_data_path = os.environ.get("DATASRC")
RESTART_WAIT_TIME = 20
TIMEOUT_CLOSE = 60

logging.getLogger("paramiko").setLevel(logging.ERROR)


class PostgresqlDB:
    def __init__(self, args):
        self.args = args

        # PostgreSQL configuration
        self.host = args['host']
        self.port = args['port']
        self.user = args['user']
        self.passwd = args['passwd']
        self.dbname = args['dbname']
        self.sock = args['sock']
        self.pid = int(args['pid'])
        self.pgcnf = args['cnf']
        self.pg_ctl = args['pg_ctl']
        self.pgdata = args['pgdata']
        self.postgres = os.path.join(os.path.split(os.path.abspath(self.pg_ctl))[0], 'postgres')

        # remote information
        self.remote_mode = eval(args['remote_mode'])
        if self.remote_mode and self.remote_mode:
            self.ssh_user = args['ssh_user']
            self.ssh_pk_file = os.path.expanduser('~/.ssh/id_rsa')
            self.pk = paramiko.RSAKey.from_private_key_file(self.ssh_pk_file)

        # resource isolation information
        self.isolation_mode = eval(args['isolation_mode'])
        if self.isolation_mode:
            self.ssh_passwd = getpass(prompt='Password on host for cgroups commands: ')

        # PostgreSQL Internal Metrics
        self.num_metrics = 60
        self.PG_STAT_VIEWS = [
            "pg_stat_archiver", "pg_stat_bgwriter",  # global
            "pg_stat_database", "pg_stat_database_conflicts",  # local
            "pg_stat_user_tables", "pg_statio_user_tables",  # local
            "pg_stat_user_indexes", "pg_statio_user_indexes"  # local
        ]
        self.PG_STAT_VIEWS_LOCAL_DATABASE = ["pg_stat_database", "pg_stat_database_conflicts"]
        self.PG_STAT_VIEWS_LOCAL_TABLE = ["pg_stat_user_tables", "pg_statio_user_tables"]
        self.PG_STAT_VIEWS_LOCAL_INDEX = ["pg_stat_user_indexes", "pg_statio_user_indexes"]
        self.NUMERIC_METRICS = [  # counter
            # global
            'buffers_alloc', 'buffers_backend', 'buffers_backend_fsync', 'buffers_checkpoint', 'buffers_clean',
            'checkpoints_req', 'checkpoints_timed', 'checkpoint_sync_time', 'checkpoint_write_time', 'maxwritten_clean',
            'archived_count', 'failed_count',
            # db
            'blk_read_time', 'blks_hit', 'blks_read', 'blk_write_time', 'conflicts', 'deadlocks', 'temp_bytes',
            'temp_files', 'tup_deleted', 'tup_fetched', 'tup_inserted', 'tup_returned', 'tup_updated', 'xact_commit',
            'xact_rollback', 'confl_tablespace', 'confl_lock', 'confl_snapshot', 'confl_bufferpin', 'confl_deadlock',
            # table
            'analyze_count', 'autoanalyze_count', 'autovacuum_count', 'heap_blks_hit', 'heap_blks_read', 'idx_blks_hit',
            'idx_blks_read', 'idx_scan', 'idx_tup_fetch', 'n_dead_tup', 'n_live_tup', 'n_tup_del', 'n_tup_hot_upd',
            'n_tup_ins', 'n_tup_upd', 'n_mod_since_analyze', 'seq_scan', 'seq_tup_read', 'tidx_blks_hit',
            'tidx_blks_read',
            'toast_blks_hit', 'toast_blks_read', 'vacuum_count',
            # index
            'idx_blks_hit', 'idx_blks_read', 'idx_scan', 'idx_tup_fetch', 'idx_tup_read'
        ]
        self.im_alive_init()

        # PostgreSQL Knobs
        self.knobs_detail = initialize_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = get_default_knobs()

        self.clear_cmd = """psql -c \"select pg_terminate_backend(pid) from pg_stat_activity where datname = 'imdbload';\" """

    def _gen_config_file(self, knobs):
        if self.remote_mode:
            cnf = '/tmp/pglocal.cnf'
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, pkey=self.pk,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
            sftp = ssh.open_sftp()
            try:
                sftp.get(self.pgcnf, cnf)
            except IOError:
                logger.info('PGCNF not exists!')
            if sftp: sftp.close()
            if ssh: ssh.close()
        else:
            cnf = self.pgcnf

        cnf_parser = ConfigParser(cnf)
        knobs_not_in_cnf = []
        for key in knobs.keys():
            if key not in self.knobs_detail.keys():
                knobs_not_in_cnf.append(key)
                continue
            cnf_parser.set(key, knobs[key])
        cnf_parser.replace('/tmp/postgres.cnf')

        if self.remote_mode:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, pkey=self.pk,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
            sftp = ssh.open_sftp()
            try:
                sftp.put(cnf, self.pgcnf)
            except IOError:
                logger.info('PGCNF not exists!')
            if sftp: sftp.close()
            if ssh: ssh.close()

        logger.info('generated config file done')
        return knobs_not_in_cnf

    def _kill_postgres(self):
        kill_cmd = '{} stop -D {}'.format(self.pg_ctl, self.pgdata)
        force_kill_cmd1 = "ps aux|grep '" + self.sock + "'|awk '{print $2}'|xargs kill -9"
        force_kill_cmd2 = "ps aux|grep '" + self.pgcnf + "'|awk '{print $2}'|xargs kill -9"

        if self.remote_mode:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, pkey=self.pk,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(kill_cmd)
            ret_code = ssh_stdout.channel.recv_exit_status()
            if ret_code == 0:
                logger.info("Close db successfully")
            else:
                logger.info("Force close DB!")
                ssh.exec_command(force_kill_cmd1)
                ssh.exec_command(force_kill_cmd2)
            ssh.close()
            logger.info('postgresql is shut down remotely')
        else:
            p_close = subprocess.Popen(kill_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
            try:
                outs, errs = p_close.communicate(timeout=TIMEOUT_CLOSE)
                ret_code = p_close.poll()
                if ret_code == 0:
                    logger.info("Close db successfully")
            except subprocess.TimeoutExpired:
                logger.info("Force close!")
                os.system(force_kill_cmd1)
                os.system(force_kill_cmd2)
            logger.info('postgresql is shut down')

    def _start_postgres(self):
        if self.remote_mode:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, pkey=self.pk,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})

            start_cmd = '{} --config_file={} -D {}'.format(self.postgres, self.pgcnf, self.pgdata)
            wrapped_cmd = 'echo $$; exec ' + start_cmd
            _, start_stdout, _ = ssh.exec_command(wrapped_cmd)
            self.pid = int(start_stdout.readline())

            if self.isolation_mode:
                cgroup_cmd = 'sudo -S cgclassify -g memory,cpuset:server ' + str(self.pid)
                ssh_stdin, ssh_stdout, _ = ssh.exec_command(cgroup_cmd)
                ssh_stdin.write(self.ssh_passwd + '\n')
                ssh_stdin.flush()
                ret_code = ssh_stdout.channel.recv_exit_status()
                ssh.close()
                if not ret_code:
                    logger.info('add {} to memory,cpuset:server'.format(self.pid))
                else:
                    logger.info('Failed: add {} to memory,cpuset:server'.format(self.pid))

        else:
            proc = subprocess.Popen([self.postgres, '--config_file={}'.format(self.pgcnf), '-D',  self.pgdata])
            self.pid = proc.pid
            if self.isolation_mode:
                command = 'sudo cgclassify -g memory,cpuset:server ' + str(self.pid)
                p = os.system(command)
                if not p:
                    logger.info('add {} to memory,cpuset:server'.format(self.pid))
                else:
                    logger.info('Failed: add {} to memory,cpuset:server'.format(self.pid))

        count = 0
        start_sucess = True
        logger.info('wait for connection')
        while True:
            try:
                dbc = PostgresqlConnector(host=self.host,
                                          port=self.port,
                                          user=self.user,
                                          passwd=self.passwd,
                                          name=self.dbname)
                db_conn = dbc.conn
                if db_conn.closed == 0:
                    logger.info('Connected to PostgreSQL db')
                    db_conn.close()
                    break
            except:
                pass

            time.sleep(1)
            count = count + 1
            if count > 600:
                start_sucess = False
                logger.info("can not connect to DB")
                clear_cmd = """ps -ef|grep postgres|grep -v grep|cut -c 9-15|xargs kill -9"""
                subprocess.Popen(clear_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                 close_fds=True)
                logger.info("kill all postgres process")
                break

        logger.info('finish {} seconds waiting for connection'.format(count))
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
        self.apply_knobs_offline(self.default_knobs)
        self.reinit_interval = 0

    def apply_knobs_online(self, knobs):
        # self.restart_rds()
        # apply knobs remotely
        db_conn = PostgresqlConnector(host=self.host,
                                      port=self.port,
                                      user=self.user,
                                      passwd=self.passwd,
                                      name=self.dbname)

        for key in knobs.keys():
            self.set_knob_value(db_conn, key, knobs[key])
        db_conn.close_db()
        logger.info("[{}] Knobs applied online!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        return True

    def apply_knobs_offline(self, knobs):
        self._kill_postgres()

        if 'min_wal_size' in knobs.keys():
            if 'wal_segment_size' in knobs.keys():
                wal_segment_size = knobs['wal_segment_size']
            else:
                wal_segment_size = 16
            if knobs['min_wal_size'] < 2 * wal_segment_size:
                knobs['min_wal_size'] = 2 * wal_segment_size
                logger.info('"min_wal_size" must be at least twice "wal_segment_size"')

        knobs_not_in_cnf = self._gen_config_file(knobs)
        sucess = self._start_postgres()
        try:
            logger.info('sleeping for {} seconds after restarting postgres'.format(RESTART_WAIT_TIME))
            time.sleep(RESTART_WAIT_TIME)

            if len(knobs_not_in_cnf) > 0:
                tmp_rds = {}
                for knob_rds in knobs_not_in_cnf:
                    tmp_rds[knob_rds] = knobs[knob_rds]
                self.apply_knobs_online(tmp_rds)
        except:
            sucess = False

        return sucess

    def _check_apply(self, db_conn, k, v0):
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

    def set_knob_value(self, db_conn, k, v):
        sql = 'SHOW {};'.format(k)
        r = db_conn.fetch_results(sql)

        # type convert
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


        if str(v).isdigit():
            sql = "SET {}={}".format(k, v)
        else:
            sql = "SET {}='{}'".format(k, v)
        try:
            db_conn.execute(sql)
        except:
            logger.info("Failed: execute {}".format(sql))

        while not self._check_apply(db_conn, k, v0):
            time.sleep(1)
        return True

    def im_alive_init(self):
        global im_alive
        im_alive = mp.Value('b', True)

    def set_im_alive(self, value):
        im_alive.value = value

    def get_internal_metrics(self, internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME):
        _counter = 0
        _period = 5
        count = (BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME) / _period - 1
        warmup = BENCHMARK_WARMING_TIME / _period

        def collect_metric(counter):
            counter += 1
            timer = threading.Timer(float(_period), collect_metric, (counter,))
            timer.start()
            if counter >= count or not im_alive.value:
                timer.cancel()
            if counter > warmup:
                try:
                    # print('collect internal metrics {}'.format(counter))
                    db_conn = PostgresqlConnector(host=self.host,
                                                  port=self.port,
                                                  user=self.user,
                                                  passwd=self.passwd,
                                                  name=self.dbname)
                    metrics_dict = {
                        'global': {},
                        'local': {
                            'db': {},
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
                                type = 'db'
                                type_key = 'datname'
                            elif view in self.PG_STAT_VIEWS_LOCAL_TABLE:
                                type = 'table'
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
        def do(metric_name, metric_values):
            return float(float(metric_values[-1]) - float(metric_values[0])) / len(metric_values)

        result = np.zeros(self.num_metrics)
        keys = list(metrics[0].keys())
        keys.sort()
        for idx in range(len(keys)):
            key = keys[idx]
            data = [x[key] for x in metrics]
            result[idx] = do(key, data)
        # TODO:
        dirty_pages_per, hit_ratio, page_data = 0, 0, 0
        return result, dirty_pages_per, hit_ratio, page_data

    def get_db_size(self):
        db_conn = PostgresqlConnector(host=self.host,
                                      port=self.port,
                                      user=self.user,
                                      passwd=self.passwd,
                                      name=self.dbname)
        sql = "select pg_database_size('{}')/1024/1024;".format(self.dbname)
        res = db_conn.fetch_results(sql, json=False)
        db_size = float(res[0][0])
        db_conn.close_db()
        return db_size
