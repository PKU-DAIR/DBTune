import os
import pdb
import time
import threading
import subprocess
import paramiko
import logging
import numpy as np
import multiprocessing as mp
from getpass import getpass
from autotune.dbconnector import MysqlConnector
from autotune.knobs import logger
from autotune.utils.parser import ConfigParser
from autotune.knobs import initialize_knobs, get_default_knobs

dst_data_path = os.environ.get("DATADST")
src_data_path = os.environ.get("DATASRC")
log_num_default = 2
log_size_default = 50331648

RESTART_WAIT_TIME = 5
TIMEOUT_CLOSE = 60

logging.getLogger("paramiko").setLevel(logging.ERROR)


class MysqlDB:
    def __init__(self, args):
        self.args = args

        # MySQL configuration
        self.host = args['host']
        self.port = args['port']
        self.user = args['user']
        self.passwd = args['passwd']
        self.dbname = args['dbname']
        self.sock = args['sock']
        self.pid = int(args['pid'])
        self.mycnf = args['cnf']
        self.mysqld = args['mysqld']

        # remote information
        self.remote_mode = eval(args['remote_mode'])
        if self.remote_mode:
            self.ssh_user = args['ssh_user']
            self.ssh_pk_file = os.path.expanduser('~/.ssh/id_rsa')
            self.pk = paramiko.RSAKey.from_private_key_file(self.ssh_pk_file)

        self.connection_info = {'host': self.host,
                                'port': self.port,
                                'user': self.user,
                                'passwd': self.passwd,
                                'name': self.dbname}
        if not self.remote_mode:
            self.connection_info['socket'] = self.sock
        # resource isolation information
        self.isolation_mode = eval(args['isolation_mode'])
        if self.isolation_mode and self.remote_mode:
            self.ssh_passwd = getpass(prompt='Password on host for cgroups commands: ')

        # MySQL Internal Metrics
        self.num_metrics = 65
        self.value_type_metrics = [
            'lock_deadlocks', 'lock_timeouts', 'lock_row_lock_time_max',
            'lock_row_lock_time_avg', 'buffer_pool_size', 'buffer_pool_pages_total',
            'buffer_pool_pages_misc', 'buffer_pool_pages_data', 'buffer_pool_bytes_data',
            'buffer_pool_pages_dirty', 'buffer_pool_bytes_dirty', 'buffer_pool_pages_free',
            'trx_rseg_history_len', 'file_num_open_files', 'innodb_page_size'
        ]
        self.im_alive_init()  # im collection signal

        # MySQL Knobs
        self.knobs_detail = initialize_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = get_default_knobs()
        self.pre_combine_log_file_size = log_num_default * log_size_default

        self.clear_cmd = """mysqladmin processlist -uroot -S$MYSQL_SOCK | awk '$2 ~ /^[0-9]/ {print "KILL "$2";"}' | mysql -uroot -S$MYSQL_SOCK """

    def _gen_config_file(self, knobs):
        if self.remote_mode:
            cnf = '/tmp/mylocal.cnf'
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, pkey=self.pk,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
            sftp = ssh.open_sftp()
            try:
                sftp.get(self.mycnf, cnf)
            except IOError:
                logger.error('MYCNF not exists!')
            if sftp: sftp.close()
            if ssh: ssh.close()
        else:
            cnf = self.mycnf

        cnf_parser = ConfigParser(cnf)
        knobs_not_in_cnf = []
        for key in knobs.keys():
            if key not in self.knobs_detail.keys():
                knobs_not_in_cnf.append(key)
                continue
            cnf_parser.set(key, knobs[key])

        cnf_parser.replace('/data2/ruike/tmpdir/mysql.cnf')

        if self.remote_mode:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, pkey=self.pk,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
            sftp = ssh.open_sftp()
            try:
                sftp.put(cnf, self.mycnf)
            except IOError:
                logger.error('MYCNF not exists!')
            if sftp: sftp.close()
            if ssh: ssh.close()

        logger.info('generated config file done')
        return knobs_not_in_cnf

    def _kill_mysqld(self):
        mysqladmin = os.path.dirname(self.mysqld) + '/mysqladmin'
        kill_cmd = '{} -u{} -S {} shutdown'.format(mysqladmin, self.user, self.sock)
        force_kill_cmd1 = "ps aux|grep '" + self.sock + "'|awk '{print $2}'|xargs kill -9"
        force_kill_cmd2 = "ps aux|grep '" + self.mycnf + "'|awk '{print $2}'|xargs kill -9"

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
            logger.info('mysql is shut down remotely')

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
            logger.info('mysql is shut down')

    def _start_mysqld(self):
        if self.remote_mode:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, pkey=self.pk,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})

            start_cmd = '{} --defaults-file={}'.format(self.mysqld, self.mycnf)
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
            proc = subprocess.Popen([self.mysqld, '--defaults-file={}'.format(self.mycnf)])
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
        error, db_conn = None, None
        while True:
            try:
                dbc = MysqlConnector(**self.connection_info)
                db_conn = dbc.conn
                if db_conn.is_connected():
                    logger.info('Connected to MySQL db')
                    db_conn.close()
                    break
            except Exception as result:
                if count > 30:
                    logger.info(result)
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
        self.apply_knobs_offline(self.default_knobs)
        self.reinit_interval = 0

    def apply_knobs_online(self, knobs):
        db_conn = MysqlConnector(**self.connection_info)
        if 'innodb_io_capacity' in knobs.keys():
            self.set_knob_value(db_conn, 'innodb_io_capacity_max', 2 * int(knobs['innodb_io_capacity']))

        for key in knobs.keys():
            self.set_knob_value(db_conn, key, knobs[key])
        db_conn.close_db()
        logger.info("[{}] Knobs applied online!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        return True

    def apply_knobs_offline(self, knobs):
        # modify cnf and restart db
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
            db_conn = MysqlConnector(**self.connection_info)
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
                self.apply_knobs_online(tmp_rds)
            if modify_concurrency:
                tmp = {}
                tmp['innodb_thread_concurrency'] = true_concurrency
                self.apply_knobs_online(tmp)
                knobs['innodb_thread_concurrency'] = true_concurrency
        except:
            sucess = False

        return sucess

    def _check_apply(self, db_conn, k, v0):
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

    def set_knob_value(self, db_conn, k, v):
        sql = 'SHOW GLOBAL VARIABLES LIKE "{}";'.format(k)
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
            sql = "SET GLOBAL {}={}".format(k, v)
        else:
            sql = "SET GLOBAL {}='{}'".format(k, v)
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
        """Get the all internal metrics of MySQL, like io_read, physical_read.

        This func uses a SQL statement to lookup system table: information_schema.INNODB_METRICS
        and returns the lookup result.
        """
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
                    db_conn = MysqlConnector(**self.connection_info)

                    sql = 'SELECT NAME, COUNT from information_schema.INNODB_METRICS where status="enabled" ORDER BY NAME'
                    res = db_conn.fetch_results(sql, json=False)
                    res_dict = {}
                    for (k, v) in res:
                        res_dict[k] = v
                    internal_metrics.append(res_dict)

                except Exception as err:
                    self.connect_sucess = False
                    logger.info("connection failed during internal metrics collection")
                    logger.info(err)

        collect_metric(_counter)
        return internal_metrics

    def _post_handle(self, metrics):
        def do(metric_name, metric_values):
            metric_type = 'counter'
            if metric_name in self.value_type_metrics:
                metric_type = 'value'
            if metric_type == 'counter':
                return float(metric_values[-1] - metric_values[0]) * 23 / len(metric_values)
            else:
                return float(sum(metric_values)) / len(metric_values)

        result = np.zeros(65)
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
        db_conn = MysqlConnector(**self.connection_info)
        sql = 'SELECT CONCAT(round(sum((DATA_LENGTH + index_length) / 1024 / 1024), 2), "MB") as data from information_schema.TABLES where table_schema="{}"'.format(
            self.dbname)
        res = db_conn.fetch_results(sql, json=False)
        db_size = float(res[0][0][:-2])
        db_conn.close_db()
        return db_size
