import os
import pdb
import time
import threading
import subprocess
import paramiko
import logging
import numpy as np
import multiprocessing as mp
from shutil import copyfile
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
GRACEFUL_CLOSE_TIMEOUT = 600   # clean-shutdown budget before falling back to kill -9
# Clean shutdown vs kill -9, measured on Miyabi-C (Lustre datadir, scripts/lab/test_restart_policy.sh):
#   doublewrite=OFF: 500-880k dirty pages flush in 14-21 s (~40k pages/s); kill -9 recovery 118-282 s.
#   doublewrite=ON : flush only 1.4-2.7k pages/s (per-batch fsync) -> 300-390k pages take 111-225 s;
#                    kill -9 recovery for the same states 169-271 s, historically up to ~600 s.
# The flush cost is predictable from the dirty-page count, recovery is not; so shut down cleanly whenever
# the estimated flush stays well inside the budget, and kill -9 only for pathological dirty sets.
GRACEFUL_FLUSH_PAGES_PER_S = {'OFF': 40000.0, 'ON': 1400.0}
GRACEFUL_MAX_EST_FLUSH_S = 450.0

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
        # If pid is 0 (online mode), try to detect from pid-file or process list
        if self.pid == 0:
            pid_file = args.get('sock', '').replace('.sock', '.pid')
            if pid_file and os.path.exists(pid_file):
                try:
                    with open(pid_file) as f:
                        self.pid = int(f.read().strip())
                except Exception:
                    pass
            if self.pid == 0:
                try:
                    import subprocess
                    result = subprocess.run(['pgrep', '-f', args.get('mysqld', 'mysqld')],
                                          capture_output=True, text=True)
                    pids = result.stdout.strip().split('\n')
                    if pids and pids[0]:
                        self.pid = int(pids[0])
                except Exception:
                    pass
        self.mycnf = args['cnf']
        # clean shutdown between trials (innodb_fast_shutdown=1 + mysqladmin shutdown) is the default;
        # [database] graceful_shutdown = False restores the historical kill -9 (crash recovery at every
        # start; needed to reproduce the eval2 sessions byte-for-byte)
        self.graceful_shutdown = str(args.get('graceful_shutdown', 'True')).strip().lower() != 'false'
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
        # Canonical 65 INNODB_METRICS names (alphabetical) shared with the OpAdviser
        # source histories. MySQL 8.0.44 enables a superset (~74); we project the
        # collected metrics onto exactly these 65 in this order so target IM vectors
        # align dimensionally with the source pool. (scripts/DBTune_history/TRASH/
        # innodb_metrics_65.txt)
        self.canonical_im_names = [
            'buffer_pool_bytes_data', 'buffer_pool_bytes_dirty', 'buffer_pool_pages_data', 'buffer_pool_pages_dirty',
            'buffer_pool_pages_free', 'buffer_pool_pages_misc', 'buffer_pool_pages_total', 'buffer_pool_read_requests',
            'buffer_pool_reads', 'buffer_pool_size', 'dml_deletes', 'dml_inserts',
            'dml_reads', 'dml_updates', 'file_num_open_files', 'ibuf_free_list',
            'ibuf_merges', 'ibuf_merged_delete_marks', 'ibuf_merged_deletes', 'ibuf_merged_inserts',
            'ibuf_merges_insert', 'ibuf_merges_delete_mark', 'ibuf_merges_delete', 'ibuf_segment_leaf',
            'ibuf_segment_non_leaf', 'ibuf_size', 'innodb_adaptive_hash_searches', 'innodb_adaptive_hash_searches_btree',
            'innodb_buffer_pool_dump_status', 'innodb_buffer_pool_load_status', 'innodb_buffer_pool_resize_status', 'innodb_dblwr_pages_written',
            'innodb_dblwr_writes', 'innodb_log_waits', 'innodb_os_log_fsyncs', 'innodb_os_log_pending_fsyncs',
            'innodb_os_log_pending_writes', 'innodb_os_log_written', 'innodb_page_size', 'innodb_pages_created',
            'innodb_pages_read', 'innodb_pages_written', 'innodb_row_lock_time', 'innodb_row_lock_time_avg',
            'innodb_row_lock_time_max', 'innodb_rows_deleted', 'innodb_rows_inserted', 'innodb_rows_read',
            'innodb_rows_updated', 'lock_deadlocks', 'lock_number_of_waits', 'lock_object_lock_created',
            'lock_object_lock_requests', 'lock_rec_lock_created', 'lock_rec_lock_requests', 'lock_row_lock_time',
            'lock_row_lock_time_avg', 'lock_row_lock_time_max', 'lock_table_lock_created', 'lock_table_lock_requests',
            'lock_timeouts', 'trx_commits_insert_update', 'trx_id_counter', 'trx_rseg_history_len',
            'trx_rw_commits',
        ]
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

        # Save a clean copy of my.cnf so we can restore it before each config write
        self.mycnf_clean = self.mycnf + '.clean'
        if not os.path.exists(self.mycnf_clean):
            copyfile(self.mycnf, self.mycnf_clean)
            logger.info("Saved clean cnf backup: %s", self.mycnf_clean)

    def _gen_config_file(self, knobs):
        # Restore clean cnf before writing new knobs to avoid stale values
        if os.path.exists(self.mycnf_clean):
            copyfile(self.mycnf_clean, self.mycnf)
            logger.info("Restored clean cnf from %s", self.mycnf_clean)

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
    
        cnf_parser.replace(os.environ.get('DBTUNE_TMP_CNF', './tmp/mysqld.cnf'))

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
        kill_start = time.time()
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
            logger.info('mysql is shut down remotely (%.1fs)', time.time() - kill_start)

        else:
            # p_close = subprocess.Popen(kill_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
            #                            close_fds=True)
            # try:
            #     outs, errs = p_close.communicate(timeout=TIMEOUT_CLOSE)
            #     ret_code = p_close.poll()
            #     if ret_code == 0:
            #         logger.info("Close db successfully (graceful, %.1fs)", time.time() - kill_start)
            # except subprocess.TimeoutExpired:
            closed = False
            if self.graceful_shutdown:
                # Clean shutdown: flush dirty pages now so the next start skips crash recovery.
                # Measured on Miyabi-C (scripts/lab/test_restart_policy.sh, 2026-08-29): kill -9 ->
                # 118-282 s of recovery at the next start; innodb_fast_shutdown=1 -> 14-21 s here +
                # 6-10 s there. The buffer-pool dump is disabled for this shutdown so no page list is
                # loaded at the next start (kill -9 never produced one -> warm-up semantics unchanged).
                try:
                    db_conn = MysqlConnector(**self.connection_info)
                    dw = str(db_conn.fetch_results("SHOW GLOBAL VARIABLES LIKE 'innodb_doublewrite'")[0]['Value']).strip().upper()
                    dirty = int(db_conn.fetch_results("SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_pages_dirty'")[0]['Value'])
                    est = dirty / GRACEFUL_FLUSH_PAGES_PER_S.get(dw, GRACEFUL_FLUSH_PAGES_PER_S['ON'])
                    use_clean = est <= GRACEFUL_MAX_EST_FLUSH_S
                    logger.info('shutdown policy: doublewrite=%s dirty_pages=%d est_flush=%.0fs -> %s', dw, dirty, est,
                                'clean shutdown' if use_clean else 'kill -9 (estimated flush exceeds %.0fs)' % GRACEFUL_MAX_EST_FLUSH_S)
                    if use_clean:
                        db_conn.execute('SET GLOBAL innodb_fast_shutdown=1')
                        db_conn.execute('SET GLOBAL innodb_buffer_pool_dump_at_shutdown=OFF')
                    db_conn.close_db()
                    if use_clean:
                        p_close = subprocess.run(kill_cmd, shell=True, capture_output=True, timeout=GRACEFUL_CLOSE_TIMEOUT)
                        closed = (p_close.returncode == 0)
                        logger.info('graceful shutdown %s (%.1fs)',
                                    'done' if closed else 'rc=%d, falling back to kill -9' % p_close.returncode,
                                    time.time() - kill_start)
                except subprocess.TimeoutExpired:
                    logger.info('graceful shutdown exceeded %ds; falling back to kill -9', GRACEFUL_CLOSE_TIMEOUT)
                except Exception as e:
                    logger.info('graceful shutdown failed (%s); falling back to kill -9', e)
            if not closed:
                logger.info("Force close! (timeout after %ds, elapsed %.1fs)", TIMEOUT_CLOSE, time.time() - kill_start)
            result = subprocess.run(['pgrep', '-x', 'mysqld'], capture_output=True, text=True)
            if result.stdout:
                for pid in result.stdout.strip().split('\n'):
                    subprocess.run(['kill', '-9', pid], capture_output=True, text=True)
                    logger.info("Killed mysqld pid=%s", pid)
            # Remove stale socket and lock files so new mysqld can bind
            for f in [self.sock, self.sock + '.lock']:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                        logger.info("Removed stale %s", f)
                except PermissionError:
                    logger.warning("Cannot remove %s (permission denied)", f)
            # Wait for port to be released
            while subprocess.run(['ss', '-tlnp'], capture_output=True, text=True).stdout.find(':3306 ') >= 0:
                time.sleep(0.5)
            assert subprocess.run(['pgrep', '-x', 'mysqld_exporter'], capture_output=True).returncode == 0, \
                "mysqld_exporter was killed! Check pgrep -x matching."
            logger.info('mysql is shut down (total %.1fs)', time.time() - kill_start)

    def _start_mysqld(self):
        start_time = time.time()
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
            logger.info('launched mysqld pid=%d (%.1fs after start)', self.pid, time.time() - start_time)
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
                if count > 30 and count % 30 == 0:
                    logger.info(result)
                # Check if mysqld process is still alive
                if not self.remote_mode and proc.poll() is not None:
                    logger.error('mysqld process (pid=%d) died with exit code %d after %.1fs',
                                 self.pid, proc.returncode, time.time() - start_time)
                    start_sucess = False
                    break
                pass

            time.sleep(1)
            count = count + 1
            if count > 600:
                start_sucess = False
                logger.info("can not connect to DB after 600s (%.1fs total)", time.time() - start_time)
                break

        logger.info('finish %d seconds waiting for connection (%.1fs total)', count, time.time() - start_time)
        logger.info('{} --defaults-file={}'.format(self.mysqld, self.mycnf))
        logger.info('mysql is up' if start_sucess else 'mysql FAILED to start')
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

    def ensure_default_config(self, strong=True):
        """Verify MySQL is running with default knobs.

        If strong=True, raise an error if any knob differs from default.
        If strong=False, restart MySQL with defaults if any knob differs.
        """
        try:
            db_conn = MysqlConnector(**self.connection_info)
            # Knobs MySQL silently clamps/rounds away from the requested default
            # (chunk-size rounding, system fd limit) — comparing them is meaningless.
            clamp_skip = {'innodb_buffer_pool_size', 'open_files_limit', 'innodb_open_files'}

            def _vals_match(cur, exp):
                cur, exp = str(cur).strip(), str(exp).strip()
                normalize = {'ON': '1', 'OFF': '0'}
                if normalize.get(cur, cur).lower() == normalize.get(exp, exp).lower():
                    return True            # case-insensitive / ON-OFF / exact string
                try:
                    c, e = float(cur), float(exp)
                    # tolerate float formatting (75.000000==75), huge-int precision
                    # loss, and MySQL's rounding to multiples (≤5% relative).
                    return c == e or abs(c - e) <= 0.05 * max(abs(c), abs(e), 1.0)
                except ValueError:
                    return False

            mismatches = []
            for knob_name, detail in self.knobs_detail.items():
                if knob_name in clamp_skip:
                    continue
                sql = 'SHOW GLOBAL VARIABLES LIKE "{}";'.format(knob_name)
                result = db_conn.fetch_results(sql)
                if result:
                    current = str(result[0]['Value']).strip()
                    expected = str(detail['default'])
                    if not _vals_match(current, expected):
                        mismatches.append((knob_name, current, expected))
            db_conn.close_db()

            if not mismatches:
                logger.info("All knobs at default values")
                return

            for knob_name, current, expected in mismatches:
                logger.error("Knob {} is {}, expected default {}".format(knob_name, current, expected))

            if strong:
                raise RuntimeError(
                    "MySQL is not running with default config. {} knob(s) differ: {}".format(
                        len(mismatches),
                        ", ".join("{}={} (expected {})".format(k, c, e) for k, c, e in mismatches)))
            else:
                logger.info("Restarting MySQL with default knobs")
                self.apply_knobs_offline(self.default_knobs)

        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("Cannot verify config: {}".format(e))
            if strong:
                raise
            else:
                self.apply_knobs_offline(self.default_knobs)
            self.apply_knobs_offline(self.default_knobs)

    def apply_knobs_online(self, knobs):
        db_conn = MysqlConnector(**self.connection_info)
        if 'innodb_io_capacity' in knobs.keys():
            self.set_knob_value(db_conn, 'innodb_io_capacity_max', 2 * int(knobs['innodb_io_capacity']))

        for key in knobs.keys():
            self.set_knob_value(db_conn, key, knobs[key])
        self._wait_for_async_resizes(db_conn)
        db_conn.close_db()
        logger.info("[{}] Knobs applied online!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        return True

    def _wait_for_async_resizes(self, db_conn, max_wait=900):
        """innodb_buffer_pool_size and innodb_redo_log_capacity are dynamic but resize
        asynchronously: MySQL reports the new size only when the resize completes and
        blocks page access during its critical phase. Benchmarking while a resize is in
        flight measures the stall, not the configuration (collect_S0_llama_online rows
        3/8: 'benchmark timeout', tps=-1 recorded as SUCCESS). Poll the status variables
        until both report completion (capped at max_wait seconds), and log the wait."""
        checks = {
            'Innodb_buffer_pool_resize_status': lambda v: v == '' or 'completed' in v.lower(),
            'Innodb_redo_log_resize_status': lambda v: v == '' or v.upper() == 'OK',
        }
        t0 = time.time()
        for name, settled in checks.items():
            while True:
                try:
                    r = db_conn.fetch_results("SHOW GLOBAL STATUS LIKE '{}';".format(name))
                    v = r[0]['Value'].strip() if r else ''
                except Exception as e:
                    logger.info('[resize-wait] %s: status query failed (%s); not waiting', name, e)
                    break
                if settled(v):
                    break
                if time.time() - t0 > max_wait:
                    logger.info('[resize-wait] %s still "%s" after %d s; proceeding', name, v, max_wait)
                    break
                time.sleep(5)
        waited = time.time() - t0
        if waited >= 1:
            logger.info('[resize-wait] async resizes settled after %.0f s', waited)

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

            # --- Config verification: intended (DBTune) vs actual (MySQL) ---
            # Logs one parseable line per knob so we can confirm the applied
            # configuration matches what DBTune intended for each iteration.
            for k in sorted(knobs.keys()):
                try:
                    rr = db_conn.fetch_results('SHOW GLOBAL VARIABLES LIKE "{}";'.format(k))
                    raw = rr[0]['Value'] if rr else None
                except Exception as e:
                    raw = 'ERR:{}'.format(e)
                # Normalize ON/OFF to 1/0 for comparison with enum knob values
                actual = raw
                if raw == 'ON':
                    actual = '1'
                elif raw == 'OFF':
                    actual = '0'
                intended = str(knobs[k])
                match = (str(actual).strip() == intended.strip())
                logger.info('[KNOB-VERIFY] %s intended=%s actual=%s match=%s',
                            k, intended, raw, match)
            # --- end config verification ---

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

    @staticmethod
    def _norm(value):
        """Canonical form for comparing knob values: ON/OFF -> 1/0, numbers -> str, text -> lower."""
        text = str(value).strip()
        if text.upper() == 'ON':
            return '1'
        if text.upper() == 'OFF':
            return '0'
        return text.lower()

    def _check_apply(self, db_conn, k, v0):
        """True once the variable no longer reads as its previous value v0 (case-insensitive:
        MySQL reports enum values in its own case, e.g. FULL vs full)."""
        sql = 'SHOW GLOBAL VARIABLES LIKE "{}";'.format(k)
        r = db_conn.fetch_results(sql)
        return self._norm(r[0]['Value']) != self._norm(v0)

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

        if self._norm(v0) == self._norm(v):
            return True   # already at the requested value (case-insensitive)

        if str(v).isdigit():
            sql = "SET GLOBAL {}={}".format(k, v)
        else:
            sql = "SET GLOBAL {}='{}'".format(k, v)
        try:
            db_conn.execute(sql)
        except:
            logger.info("Failed: execute {} (read-only variable?)".format(sql))
            return True  # Skip waiting for read-only variables

        count = 0
        while not self._check_apply(db_conn, k, v0):
            time.sleep(1)
            count += 1
            if count > 30:
                logger.info("Timeout waiting for {} to apply".format(k))
                break
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

        # Project onto the canonical 65 names (in order) so the vector aligns with
        # the OpAdviser source pool regardless of how many metrics the server enables.
        # Names absent on this server stay 0; extra enabled metrics are ignored.
        keys = [k for k in self.canonical_im_names if k in metrics[0]]
        result = np.zeros(len(self.canonical_im_names))
        idx_of = {name: i for i, name in enumerate(self.canonical_im_names)}
        total_pages = 0
        dirty_pages = 0
        request = 0
        reads = 0
        page_data = 0
        page_size = 0
        page_misc = 0
        for key in keys:
            pos = idx_of[key]
            data = [x[key] for x in metrics]
            result[pos] = do(key, data)
            if key == 'buffer_pool_pages_total':
                total_pages = result[pos]
            elif key == 'buffer_pool_pages_dirty':
                dirty_pages = result[pos]
            elif key == 'buffer_pool_read_requests':
                request = result[pos]
            elif key == 'buffer_pool_reads':
                reads = result[pos]
            elif key == 'buffer_pool_pages_data':
                page_data = result[pos]
            elif key == 'innodb_page_size':
                page_size = result[pos]
            elif key == 'buffer_pool_pages_misc':
                page_misc = result[pos]
        dirty_pages_per = dirty_pages / total_pages if total_pages else 0.0
        hit_ratio = request / float(request + reads) if (request + reads) else 0.0
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
