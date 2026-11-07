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

        # Deployment mode: local | kubernetes
        self.deployment_mode = args.get('deployment_mode', 'local')

        # PostgreSQL configuration (LOCAL mode)
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

        # Kubernetes configuration (KUBERNETES mode)
        if self.deployment_mode == 'kubernetes':
            # Prometheus
            self.k8s_prometheus_namespace = args.get('k8s_prometheus_namespace', 'monitoring')
            self.k8s_prometheus_service = args.get('k8s_prometheus_service', 'prometheus-kube-prometheus-prometheus')
            self.k8s_prometheus_port = int(args.get('k8s_prometheus_port', 9090))
            
            # PostgreSQL CNPG
            self.k8s_pg_namespace = args.get('k8s_pg_namespace', 'pg')
            self.k8s_pg_cluster_name = args.get('k8s_pg_cluster_name', 'pg-cluster')
            self.k8s_pg_primary_node_instance = args.get('k8s_pg_primary_node_instance', '')
            self.k8s_pg_database = args.get('k8s_pg_database', 'app')
            self.k8s_benchmark_namespace = args.get("k8s_benchmark_namespace", "benchmark")

            # Benchmark parameters
            self.benchmark_type = args.get('benchmark_type', 'tpcc')
            self.benchmark_tpcc_vus = args.get('benchmark_tpcc_vus', 20)
            self.benchmark_tpcc_duration_min = args.get('benchmark_tpcc_duration_min', 5)
            self.benchmark_tpch_vus = args.get('benchmark_tpch_vus', 1)
            self.benchmark_tpch_querysets = args.get('benchmark_tpch_querysets', 1)
            self.benchmark_tpch_dop = args.get('benchmark_tpch_dop', 2)
            self.benchmark_tpch_scale = args.get('benchmark_tpch_scale', 1)

            # Tier for tier-specific hardware constants (set externally by dbenv/optimizer)
            self._fog_tier = None

            # Auto-discover Primary Node IP for Fog Reward
            self.k8s_pg_primary_node_instance = args.get('k8s_pg_primary_node_instance', '')
            if not self.k8s_pg_primary_node_instance:
                try:
                    # [Fixed] Dynamic discovery via kubectl with correct labels and kubeconfig
                    import subprocess
                    
                    # Check for local kubeconfig
                    kubeconfig_flag = ""
                    if os.path.exists("./kubeconfig"):
                        kubeconfig_flag = "--kubeconfig ./kubeconfig"
                    
                    cmd = f"kubectl {kubeconfig_flag} get pod -n {self.k8s_pg_namespace} -l cnpg.io/cluster={self.k8s_pg_cluster_name},cnpg.io/instanceRole=primary -o jsonpath='{{.items[0].status.hostIP}}'"
                    
                    # Redirect stderr to avoid polluting logs if it fails
                    node_ip = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()
                    
                    if node_ip:
                        self.k8s_pg_primary_node_instance = f"{node_ip}:9100"
                        logger.info(f"[Auto-Discovery] ✓ Found Primary Node IP: {self.k8s_pg_primary_node_instance}")
                    else:
                         logger.warning(f"[Auto-Discovery] ✗ Kubectl returned empty IP. Keeping default.")
                except Exception as e:
                    logger.warning(f"[Auto-Discovery] ✗ Failed to find primary node IP: {e}. Fallback to config.")

            # Port-forward process
            self.prometheus_portforward_process = None
            self.prometheus_session = None  # Session persistente de requests
            
            logger.info(f'=== Kubernetes Mode Enabled ===')
            logger.info(f'Prometheus: {self.k8s_prometheus_namespace}/{self.k8s_prometheus_service}')
            logger.info(f'PostgreSQL: {self.k8s_pg_namespace}/{self.k8s_pg_cluster_name}')
            logger.info(f'Primary node: {self.k8s_pg_primary_node_instance}')        

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
        # [DBTune 2.0] Updated metric count (20 -> 24)
        # [DBTune 2.0] Updated metric count (Dynamic)
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
            'buffers_alloc', 
            # 'buffers_backend', 'buffers_backend_fsync',
            'buffers_checkpoint',   # [RESTORED] Checkpoint pressure
            'buffers_clean', 
            'checkpoints_req', 'checkpoints_timed', 
            # 'checkpoint_sync_time', 'checkpoint_write_time',                    # [DEAD]
            'maxwritten_clean',
            # 'archived_count',                                                   # [DEAD]
            'failed_count',
            # db
            # 'blk_read_time',
            'blks_hit',             # [RESTORED] Vital for Cache Hit Ratio
            'blks_read', 'blk_write_time', 
            # 'conflicts', 'deadlocks', 'temp_bytes', 'temp_files',               # [DEAD]
            # 'tup_deleted', 'tup_fetched', 'tup_inserted', 'tup_returned',       # [DEAD]
            'tup_deleted', 'tup_fetched', 'tup_inserted', 'tup_returned', 'tup_updated', # [RESTORED] Workload Type
            'xact_commit',
            # xact_rollback, deadlocks, temp_bytes, temp_files — ELIMINADOS (v5):
            # Con 20-40 VUs de TPCC, estas métricas son siempre 0.0 en todos los steps.
            # Deadlocks y rollbacks son extremadamente raros; temp_bytes requeriría
            # work_mem < 1MB con sorts muy grandes. Dims muertos = ruido puro para el Actor.
            # WAL (PostgreSQL 14+)
            'wal_records', 'wal_fpi', 'wal_bytes', 
            'wal_buffers_full', 'wal_write', 'wal_sync', 'wal_write_time', 'wal_sync_time',
            # table
            # 'analyze_count', 'autoanalyze_count', 'autovacuum_count',           # [DEAD]
            # 'heap_blks_hit', 'heap_blks_read', 'idx_blks_hit', 'idx_blks_read', # [DEAD]
            # 'idx_scan', 'idx_tup_fetch', 'n_dead_tup', 'n_live_tup',            # [DEAD]
            # 'n_tup_del', 'n_tup_hot_upd', 'n_tup_ins', 'n_tup_upd',             # [DEAD]
            # 'n_mod_since_analyze', 'seq_scan', 'seq_tup_read', 'tidx_blks_hit', # [DEAD]
            # 'tidx_blks_read', 'toast_blks_hit', 'toast_blks_read',              # [DEAD]
            # 'vacuum_count',                                                     # [DEAD]
            # index
            # 'idx_blks_hit', 'idx_blks_read', 'idx_scan', 'idx_tup_fetch',       # [DEAD]
            # 'idx_tup_read',                                                     # [DEAD]
            # system (FogAwareReward)
            'cpu_percent', 'memory_mb', 'io_wait', 'swap_used_mb',
            # system context (Contextual RL)
            'total_memory_mb', 'cpu_count', 'container_memory_mb'

        ]
        self.num_metrics = len(self.NUMERIC_METRICS)
        logger.info(f"[PostgresqlDB] Internal Metrics Count: {self.num_metrics}")
        
        self.im_alive_init()

        # PostgreSQL Knobs
        self.knobs_detail = initialize_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = get_default_knobs()

        self.clear_cmd = """psql -c \"select pg_terminate_backend(pid) from pg_stat_activity where datname = 'imdbload';\" """

        if self.pid == 0:
            self.pid = self._get_postmaster_pid()
            if self.pid > 0:
                logger.info(f"Auto-detected PostgreSQL PID: {self.pid}")        

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
        """Apply knobs online (SIGHUP parameters)"""
        if self.deployment_mode == 'kubernetes':
            # En kubernetes, todos los knobs requieren patch al CR
            return self.apply_knobs(knobs)
        
        # Modo local original:
        from autotune.dbconnector import PostgresqlConnector

        db_conn = PostgresqlConnector(
            host=self.host,
            port=self.port,
            user=self.user,
            passwd=self.passwd,
            name=self.dbname,
            socket=self.sock
        )
        # Ya no necesitamos connect_db() porque lo hace automáticamente en __init__

        db_conn.conn.autocommit = True
 
        for key in knobs.keys():
            if key not in self.knobs_detail.keys():
                logger.info('{} is not in knobs_detail'.format(key))
                continue

            success = self.set_knob_value(db_conn, key, knobs[key])
            if not success:
                db_conn.close_db()
                return False

        # Reload configuration for SIGHUP parameters
        try:
            db_conn.execute("SELECT pg_reload_conf();")
            logger.info("Configuration reloaded successfully")
            time.sleep(2)  # Wait for reload to take effect
        except Exception as e:
            logger.info(f"Failed to reload configuration: {e}")

        db_conn.close_db()
        return True


    def apply_knobs_offline(self, knobs):
        """Apply knobs offline (requires restart)"""
        if self.deployment_mode == 'kubernetes':
            # En kubernetes, CNPG maneja el restart automáticamente
            return self.apply_knobs(knobs)
        
        # Modo local original:
        """Apply knobs using ALTER SYSTEM + restart PostgreSQL"""
        import subprocess
        import time
        from autotune.dbconnector import PostgresqlConnector
        
        try:
            logger.info("[PostgreSQL] Applying knobs with restart...")
            
            # 1. Crear conexión para aplicar knobs
            db_conn = PostgresqlConnector(
                host=self.host,
                port=self.port,
                user=self.user,
                passwd=self.passwd,
                name=self.dbname,
                socket=self.sock
            )
            db_conn.conn.autocommit = True
            
            # 2. Aplicar knobs con ALTER SYSTEM
            for key, value in knobs.items():
                if key not in self.knobs_detail.keys():
                    logger.info(f'{key} is not in knobs_detail')
                    continue
                    
                try:
                    if isinstance(value, str):
                        sql = f"ALTER SYSTEM SET {key} = '{value}';"
                    else:
                        sql = f"ALTER SYSTEM SET {key} = {value};"
                    
                    db_conn.execute(sql)
                    logger.info(f"Set {key}={value} using ALTER SYSTEM")
                except Exception as e:
                    logger.error(f"Failed to ALTER SYSTEM SET {key}: {e}")
                    db_conn.close_db()
                    return False
            
            # 3. Cerrar conexión antes de reiniciar
            db_conn.close_db()
            
            # 4. Reiniciar PostgreSQL con sudo
            logger.info("Restarting PostgreSQL to apply postmaster parameters...")
            result = subprocess.run(
                ['sudo', 'systemctl', 'restart', 'postgresql'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Restart failed: {result.stderr}")
                return False
            
            # 5. Esperar a que PostgreSQL esté listo
            logger.info("Waiting for PostgreSQL to be ready...")
            time.sleep(8)
            
            # 6. Verificar que PostgreSQL está arriba
            max_retries = 10
            for i in range(max_retries):
                try:
                    test_conn = PostgresqlConnector(
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        passwd=self.passwd,
                        name=self.dbname,
                        socket=self.sock
                    )
                    test_conn.execute("SELECT 1;")
                    test_conn.close_db()
                    logger.info("PostgreSQL restart completed successfully")
                    
                    # 7. IMPORTANTE: Actualizar el PID después del reinicio
                    self.pid = self._get_postmaster_pid()
                    logger.info(f"Updated PostgreSQL PID to {self.pid}")
                    
                    return True
                except Exception as e:
                    if i < max_retries - 1:
                        logger.warning(f"Reconnection attempt {i+1} failed, retrying...")
                        time.sleep(2)
                    else:
                        logger.error(f"Failed to reconnect after {max_retries} attempts: {e}")
                        return False
            
        except Exception as e:
            logger.error(f"Error in apply_knobs_offline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _check_apply(self, db_conn, k, v0):
        sql = 'SHOW {};'.format(k)
        r = db_conn.fetch_results(sql)

        # PostgreSQL returns column name as the key, not 'Value'
        if 'Value' in r[0]:
            current_value = r[0]['Value']
        else:
            # PostgreSQL returns the parameter name as the key
            current_value = r[0][k]

        if current_value == 'ON':
            vv = 1
        elif current_value == 'OFF':
            vv = 0
        else:
            vv = current_value.strip()

        if vv == v0:
            return False
        return True

    def set_knob_value(self, db_conn, k, v):
        # Check parameter context
        context_sql = "SELECT context FROM pg_settings WHERE name = '{}'".format(k)
        try:
            context_result = db_conn.fetch_results(context_sql)
            context = context_result[0]['context'] if context_result else 'user'
        except:
            context = 'user'
        
        sql = 'SHOW {};'.format(k)
        try:
            r = db_conn.fetch_results(sql)
        except Exception as e:
            db_conn.execute('ROLLBACK')
            r = db_conn.fetch_results(sql)
        
        # PostgreSQL returns column name as the key, not 'Value'
        if 'Value' in r[0]:
            current_value = r[0]['Value']
        else:
            current_value = r[0][k]
        
        # type convert
        if v == 'ON':
            v = 1
        elif v == 'OFF':
            v = 0
        
        if current_value == 'ON':
            v0 = 1
        elif current_value == 'OFF':
            v0 = 0
        else:
            try:
                v0 = eval(current_value)
            except:
                v0 = current_value.strip()
        
        if v0 == v:
            return True
        
        # For SIGHUP parameters, we need to modify postgresql.auto.conf
        if context in ['sighup', 'postmaster']:
            try:
                if str(v).isdigit() or isinstance(v, (int, float)):
                    alter_sql = "ALTER SYSTEM SET {} = {}".format(k, v)
                else:
                    alter_sql = "ALTER SYSTEM SET {} = '{}'".format(k, v)
                db_conn.execute(alter_sql)
                logger.info(f"Set {k}={v} using ALTER SYSTEM")
                return True
            except Exception as e:
                logger.info(f"Failed to ALTER SYSTEM SET {k}: {e}")
                return False
        else:
            # For user/superuser context, use SET
            if str(v).isdigit():
                sql = "SET {}={}".format(k, v)
            else:
                sql = "SET {}='{}'".format(k, v)
            try:
                db_conn.execute(sql)
            except Exception as e:
                logger.info("Failed: execute {} - Error: {}".format(sql, e))
                try:
                    db_conn.execute('ROLLBACK')
                except:
                    pass
                return False
        
        return True    



    def im_alive_init(self):
        global im_alive
        im_alive = mp.Value('b', True)

    def set_im_alive(self, value):
        im_alive.value = value

    # ========================================
    # KUBERNETES MODE - Prometheus Integration
    # ========================================
    
    def _start_prometheus_portforward(self):
        """Inicia port-forward a Prometheus"""
        if self.deployment_mode != 'kubernetes':
            return
        
        if self.prometheus_portforward_process is not None:
            # Verificar si el proceso sigue vivo
            if self.prometheus_portforward_process.poll() is None:
                return  # Está vivo, no reiniciar
            # Proceso murió, limpiar
            logger.warning('Port-forward process died, restarting...')
            self.prometheus_portforward_process = None
        
        import subprocess
        cmd = [
            'kubectl', 'port-forward',
            f'-n', self.k8s_prometheus_namespace,
            f'svc/{self.k8s_prometheus_service}',
            f'{self.k8s_prometheus_port}:{self.k8s_prometheus_port}'
        ]
        
        logger.info(f'Starting port-forward to Prometheus: {" ".join(cmd)}')
        self.prometheus_portforward_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Esperar a que esté listo
        import time
        time.sleep(3)
        
        # Crear session persistente de requests
        import requests
        self.prometheus_session = requests.Session()
        logger.info('Port-forward to Prometheus ready')
    
    def _stop_prometheus_portforward(self):
        """Detiene port-forward a Prometheus"""
        if self.prometheus_portforward_process:
            self.prometheus_portforward_process.terminate()
            self.prometheus_portforward_process.wait()
            self.prometheus_portforward_process = None
            logger.info('Port-forward to Prometheus stopped')
    
    def _query_prometheus(self, query):
        """Ejecuta query en Prometheus y devuelve resultados"""
        import json
        
        # Crear session si no existe (fallback)
        if self.prometheus_session is None:
            import requests
            self.prometheus_session = requests.Session()
        
        url = f'http://localhost:{self.k8s_prometheus_port}/api/v1/query'
        params = {'query': query}
        
        try:
            response = self.prometheus_session.get(url, params=params, timeout=10)
            data = response.json()
            
            if data['status'] == 'success':
                return data['data']['result']
            else:
                logger.error(f'Prometheus query failed: {query}')
                return []
        except Exception as e:
            logger.error(f'Error querying Prometheus: {e}')
            return []
    
    def _collect_prometheus_metrics(self):
        """Colecta métricas de Prometheus y las mapea al formato DBTune"""
        metrics_dict = {}
        
        # 1. Métricas globales - bgwriter
        bgwriter_metrics = [
            'buffers_clean', 'maxwritten_clean', 'buffers_alloc'
        ]
        for metric in bgwriter_metrics:
            # Use rate() to get per-second speed, not total accumulation
            query = f'rate(custom_pg_stat_bgwriter_{metric}[2m])'
            results = self._query_prometheus(query)
            if results:
                metrics_dict[f'pg_stat_bgwriter.{metric}'] = [float(results[0]['value'][1])]
        
        # 2. Métricas globales - checkpointer (PostgreSQL 17+)
        checkpointer_map = {
            'checkpoints_timed': 'checkpoints_timed',
            'checkpoints_req': 'checkpoints_req',
            'buffers_checkpoint': 'buffers_checkpoint', # [RESTORED]
        }
        for prom_name, dbtune_name in checkpointer_map.items():
            query = f'rate(custom_pg_stat_checkpointer_{prom_name}[2m])'
            results = self._query_prometheus(query)
            if results:
                metrics_dict[f'pg_stat_bgwriter.{dbtune_name}'] = [float(results[0]['value'][1])]
        
        # 3. Métricas globales - archiver
        archiver_metrics = ['archived_count', 'failed_count']
        for metric in archiver_metrics:
            query = f'rate(custom_pg_stat_archiver_{metric}[2m])'
            results = self._query_prometheus(query)
            if results:
                metrics_dict[f'pg_stat_archiver.{metric}'] = [float(results[0]['value'][1])]

        # 3.5. Métricas globales - WAL (pg_stat_wal) [DBTune 2.0]
        # Crucial for detecting I/O storms and FPI overhead
        wal_metrics = [
            'wal_records', 'wal_fpi', 'wal_bytes', 
            'wal_buffers_full', 'wal_write', 'wal_sync', 'wal_write_time', 'wal_sync_time'
        ]
        for metric in wal_metrics:
             query = f'rate(custom_pg_stat_wal_{metric}[2m])'
             results = self._query_prometheus(query)
             if results:
                  metrics_dict[f'pg_stat_wal.{metric}'] = [float(results[0]['value'][1])]
             else:
                 # Fallback/Safe Default
                 metrics_dict[f'pg_stat_wal.{metric}'] = [0.0]
        
        # 4. Métricas por database
        # [RESTORED] Extended metrics for unblinded agent
        db_metrics = [
            'blks_read', 'blks_hit', 'blk_write_time', 'xact_commit', 'xact_rollback',
            'tup_fetched', 'tup_returned', 'tup_inserted', 'tup_updated', 'tup_deleted',
            'deadlocks', 'temp_bytes', 'temp_files'
        ]
        for metric in db_metrics:
            # Crucial: Rate per second for learning stability
            query = f'rate(custom_pg_stat_database_{metric}{{datname="{self.k8s_pg_database}"}}[2m])'
            results = self._query_prometheus(query)
            if results:
                metrics_dict[f'pg_stat_database.{metric}'] = [float(results[0]['value'][1])]
        
        # [PRUNED] Metrics removed: conflicts, user_tables, statio_user_tables, indexes, statio_indexes
        # Code removed to improve readability. Context: These metrics were deemed "Dead" or unstable.

        # 10. Métricas de sistema del nodo PRIMARY (para FogAwareReward)
        if self.k8s_pg_primary_node_instance:
            # CPU utilization % (Raw query: 100 - idle%)
            # Using 2m window for better stability (1m is too tight if scrapes are >30s)
            # Relaxed regex to match IP ignoring port if needed
            ip_part = self.k8s_pg_primary_node_instance.split(':')[0]
            cpu_query = f'100 - (avg(rate(node_cpu_seconds_total{{mode="idle", instance=~".*{ip_part}.*"}}[2m])) * 100)'
            cpu_results = self._query_prometheus(cpu_query)
            if cpu_results:
                cpu_percent = float(cpu_results[0]['value'][1])
                metrics_dict['system.cpu_percent'] = [str(cpu_percent)]
            else:
                logger.warning(f"[Prometheus] CPU query returned EMPTY! Query: {cpu_query}")
                metrics_dict['system.cpu_percent'] = ['0.0']
            
            # Memory total bytes
            mem_total_query = f'node_memory_MemTotal_bytes{{instance=~".*{ip_part}.*"}}'
            mem_total_results = self._query_prometheus(mem_total_query)
            
            # Memory utilization ratio
            mem_util_query = f'instance:node_memory_utilisation:ratio{{instance=~".*{ip_part}.*"}}'
            mem_util_results = self._query_prometheus(mem_util_query)
            
            if mem_total_results and mem_util_results:
                mem_total_bytes = float(mem_total_results[0]['value'][1])
                mem_util_ratio = float(mem_util_results[0]['value'][1])
                mem_used_mb = (mem_total_bytes * mem_util_ratio) / 1024 / 1024
                metrics_dict['system.memory_mb'] = [str(mem_used_mb)]

            # I/O Wait % (Critical for Edge/SSD)
            # avg(rate(node_cpu_seconds_total{mode="iowait"}[2m])) * 100
            iowait_query = f'avg(rate(node_cpu_seconds_total{{mode="iowait", instance=~".*{ip_part}.*"}}[2m])) * 100'
            iowait_results = self._query_prometheus(iowait_query)
            if iowait_results:
                 metrics_dict['system.io_wait'] = [iowait_results[0]['value'][1]]
            else:
                 metrics_dict['system.io_wait'] = ['0.0']

            # Swap Usage (Critical for FogAwareReward)
            # node_memory_SwapTotal_bytes - node_memory_SwapFree_bytes
            swap_total_query = f'node_memory_SwapTotal_bytes{{instance="{self.k8s_pg_primary_node_instance}"}}'
            swap_free_query = f'node_memory_SwapFree_bytes{{instance="{self.k8s_pg_primary_node_instance}"}}'
            
            swap_total_res = self._query_prometheus(swap_total_query)
            swap_free_res = self._query_prometheus(swap_free_query)
            
            if swap_total_res and swap_free_res:
                swap_total = float(swap_total_res[0]['value'][1])
                swap_free = float(swap_free_res[0]['value'][1])
                swap_used = swap_total - swap_free
                swap_used_mb = swap_used / 1024 / 1024
                # logger.info(f"[DEBUG-METRICS] Swap Used: {swap_used_mb} MB")
                metrics_dict['system.swap_used_mb'] = [str(swap_used_mb)]
            else:
                 metrics_dict['system.swap_used_mb'] = ['0.0']
                 
            # 12. Context Metrics (Total Memory & CPU Count) for Generalist Agent
            # Querying kube_pod_container_resource_limits (from kube-state-metrics)
            try:
                # Memory Limit (Bytes -> MB) — kube-state-metrics >= 2.0 naming
                query_mem_limit = f'max(kube_pod_container_resource_limits{{resource="memory", pod=~"{self.k8s_pg_cluster_name}.*", container="postgres"}}) / 1024 / 1024'
                res_mem = self._query_prometheus(query_mem_limit)
                if res_mem:
                     metrics_dict['system.total_memory_mb'] = [float(res_mem[0]['value'][1])]
                else:
                     # Fallback: Node Memory
                     if self.k8s_pg_primary_node_instance:
                         query_node_mem = f'node_memory_MemTotal_bytes{{instance="{self.k8s_pg_primary_node_instance}"}} / 1024 / 1024'
                         res_node = self._query_prometheus(query_node_mem)
                         if res_node:
                             metrics_dict['system.total_memory_mb'] = [float(res_node[0]['value'][1])]

                # CPU Limit (Cores) — kube-state-metrics >= 2.0 naming
                query_cpu_limit = f'max(kube_pod_container_resource_limits{{resource="cpu", pod=~"{self.k8s_pg_cluster_name}.*", container="postgres"}})'
                res_cpu = self._query_prometheus(query_cpu_limit)
                if res_cpu:
                     metrics_dict['system.cpu_count'] = [float(res_cpu[0]['value'][1])]
                else:
                     metrics_dict['system.cpu_count'] = [4.0] # Fallback
                
                # Container Memory Working Set (True usage for OOM killer)
                # query_mem_limit above was for LIMITS. This is for USAGE.
                query_container_mem = f'max(container_memory_working_set_bytes{{pod=~"{self.k8s_pg_cluster_name}.*", container="postgres"}})'
                res_cont_mem = self._query_prometheus(query_container_mem)
                if res_cont_mem:
                    cm_mb = float(res_cont_mem[0]['value'][1]) / 1024 / 1024
                    metrics_dict['system.container_memory_mb'] = [cm_mb]
                else:
                    metrics_dict['system.container_memory_mb'] = [0.0]

            except Exception as e:
                logger.warning(f"Failed to fetch context metrics: {e}")
                metrics_dict['system.swap_used_mb'] = ['0.0']
        
        return metrics_dict



    def get_internal_metrics(self, internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME):
        """Colecta métricas internas según deployment_mode"""
        
        if self.deployment_mode == 'kubernetes':
            # Modo Kubernetes: usar Prometheus
            return self._get_internal_metrics_kubernetes(internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME)
        else:
            # Modo Local: usar código original
            return self._get_internal_metrics_local(internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME)
    
    def _get_internal_metrics_kubernetes(self, internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME):
        """Colecta métricas de Prometheus (modo kubernetes)"""
        _counter = 0
        _period = 30
        count = (BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME) / _period - 1
        warmup = BENCHMARK_WARMING_TIME / _period
        
        # Iniciar port-forward a Prometheus
        # DISABLED:         self._start_prometheus_portforward()
        
        def collect_metric(counter):
            counter += 1
            timer = threading.Timer(float(_period), collect_metric, (counter,))
            timer.start()
            if counter >= count or not im_alive.value:
                timer.cancel()
                # NO cerrar port-forward - mantenerlo vivo para próximas iteraciones
            
            if counter > warmup:
                try:
                    logger.info(f'[K8s] Collecting metrics from Prometheus (sample {counter})')
                    
                    # Colectar métricas de Prometheus
                    metrics_dict = self._collect_prometheus_metrics()
                    
                    # Procesar y combinar valores (igual que modo local)
                    valid_metrics = {}
                    for name, values in list(metrics_dict.items()):
                        metric_name = name.split('.')[-1]
                        if metric_name in self.NUMERIC_METRICS:
                            values = [float(v) for v in values if v is not None]
                            if len(values) == 0:
                                valid_metrics[name] = 0
                            else:
                                valid_metrics[name] = str(sum(values))
                    
                    internal_metrics.append(valid_metrics)
                    logger.info(f'[K8s] Collected {len(valid_metrics)} metrics')
                    
                except Exception as err:
                    self.connect_sucess = False
                    logger.error(f"[K8s] Error collecting metrics from Prometheus: {err}")
                    import traceback
                    traceback.print_exc()
        
        collect_metric(_counter)
        return internal_metrics
    
    def _get_internal_metrics_local(self, internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME):
        """Colecta métricas locales de PostgreSQL (modo local) - CÓDIGO ORIGINAL"""
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
                                # For system metrics (Gauges), we want AVERAGE
                                if name.startswith('system.'):
                                    valid_metrics[name] = str(sum(values) / len(values))
                                else:
                                    # For DB metrics (Counters/Accumulators), SUM is default (legacy behavior)
                                    # TODO: Verify if some DB metrics should also be Averaged or Maxed
                                    valid_metrics[name] = str(sum(values))

                    internal_metrics.append(valid_metrics)

                except Exception as err:
                    self.connect_sucess = False
                    logger.info("connection failed during internal metrics collection")
                    logger.info(err)

        collect_metric(_counter)
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

        # Manejar caso de métricas vacías
        if not metrics or len(metrics) == 0:
            logger.warning('[_post_handle] No metrics collected, returning zeros')
            dirty_pages_per, hit_ratio, page_data = 0, 0, 0
            return result, dirty_pages_per, hit_ratio, page_data

        # Build reverse lookup: metric_short_name -> full key in metrics dict
        # e.g. 'cpu_percent' -> 'system.cpu_percent'
        key_lookup = {}
        for key in metrics[0].keys():
            short = key.split('.')[-1]
            key_lookup[short] = key

        # Assign to positions in NUMERIC_METRICS order (not alphabetical)
        for i, metric_name in enumerate(self.NUMERIC_METRICS):
            full_key = key_lookup.get(metric_name)
            if full_key is None:
                result[i] = 0.0
                continue
            try:
                data = [x.get(full_key, 0) for x in metrics]
                result[i] = sum(float(x) for x in data) / len(data)
            except Exception as e:
                logger.warning(f'[_post_handle] Error processing {metric_name}: {e}')
                result[i] = 0.0

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

    def _get_postmaster_pid(self):
        """Get the PID of PostgreSQL postmaster process for resource monitoring"""
        try:
            # Try using pg_ctl status
            import subprocess
            result = subprocess.run(
                [self.pg_ctl, 'status', '-D', self.pgdata],
                capture_output=True,
                text=True
            )
            
            # Parse output like: "PID: 12345"
            for line in result.stdout.split('\n'):
                if 'PID:' in line or 'pid:' in line:
                    pid = int(line.split(':')[1].strip())
                    return pid
            
            # Fallback: use pgrep
            result = subprocess.run(
                ['pgrep', '-f', f'postgres.*-D.*{self.pgdata}'],
                capture_output=True,
                text=True
            )
            if result.stdout:
                return int(result.stdout.strip().split('\n')[0])
                
        except Exception as e:
            logger.warning(f"Could not auto-detect PostgreSQL PID: {e}")
        
        return 0    

    def _run_benchmark_kubernetes(self):
        """Ejecuta benchmark usando K8s Jobs (modo kubernetes)"""
        from autotune.workload.factory import WorkloadFactory
        
        # Leer tipo de benchmark desde config
        benchmark_type = getattr(self, 'benchmark_type', 'tpcc')
        
        # Crear runner apropiado
        runner = WorkloadFactory.create_runner(
            benchmark_type=benchmark_type,
            namespace=self.k8s_benchmark_namespace
        )
        
        # Obtener parámetros según tipo
        if benchmark_type == 'tpcc':
            vus = int(getattr(self, 'benchmark_tpcc_vus', 20))
            rampup_min = int(getattr(self, 'benchmark_tpcc_rampup_min', 2))
            duration_min = int(getattr(self, 'benchmark_tpcc_duration_min', 5))
            
            results = runner.run(vus=vus, rampup_min=rampup_min, duration_min=duration_min)
            
            # TPC-C retorna TPM, convertir a TPS
            tps = results['tpm'] / 60.0
            return tps
            
        elif benchmark_type == 'tpch':
            vus = int(getattr(self, 'benchmark_tpch_vus', 1))
            querysets = int(getattr(self, 'benchmark_tpch_querysets', 1))
            dop = int(getattr(self, 'benchmark_tpch_dop', 2))
            scale = int(getattr(self, 'benchmark_tpch_scale', 1))
            
            results = runner.run(vus=vus, querysets=querysets, dop=dop, scale=scale)
            
            # TPC-H retorna QPH, convertir a QPS (queries per second)
            qps = results['qph'] / 3600.0
            return qps
            
        else:
            raise ValueError(f"Benchmark type '{benchmark_type}' no soportado")

    def _run_checkpoint(self):
        """Run CHECKPOINT before benchmark to reset WAL counter to zero.

        Without this, WAL accumulates across warmup + consecutive steps. At ~5 MB/s
        (TPC-C 20VUs + wal_compression=on), a 5-min warmup generates ~1.5 GB of WAL.
        The next 5-min benchmark then hits max_wal_size=4GB at ~3.5 min into measurement.
        CHECKPOINT here resets the counter so each benchmark starts with WAL=0.
        Takes ~5-15s. Accepted overhead vs. a random forced checkpoint mid-measurement.
        """
        if self.deployment_mode != 'kubernetes':
            return
        import subprocess
        try:
            pod_cmd = (
                f"kubectl get pod -n {self.k8s_pg_namespace} "
                f"-l cnpg.io/cluster={self.k8s_pg_cluster_name},cnpg.io/instanceRole=primary "
                f"-o jsonpath='{{.items[0].metadata.name}}'"
            )
            result = subprocess.run(pod_cmd, shell=True, capture_output=True, text=True, timeout=15)
            pod = result.stdout.strip()
            if pod:
                subprocess.run(
                    ['kubectl', 'exec', '-n', self.k8s_pg_namespace, pod,
                     '--', 'psql', '-U', 'postgres', '-c', 'CHECKPOINT'],
                    check=True, capture_output=True, timeout=180
                )
                logger.info("[Benchmark] CHECKPOINT done — WAL counter reset to 0")
        except Exception as e:
            logger.warning(f"[Benchmark] CHECKPOINT failed (continuing): {e}")

    def _run_vacuum_hot_tables(self):
        """VACUUM ANALYZE on hot TPC-C tables before measurement.

        After the warmup benchmark, dead tuples accumulate on District and Warehouse
        (small tables updated on every new order). Autovacuum can fire on these tables
        mid-benchmark, competing for I/O and causing lock contention that drops TPS by
        ~30-50%. A targeted VACUUM before measurement prevents this without the overhead
        of a full-database VACUUM.
        """
        if self.deployment_mode != 'kubernetes':
            return
        import subprocess
        try:
            pod_cmd = (
                f"kubectl get pod -n {self.k8s_pg_namespace} "
                f"-l cnpg.io/cluster={self.k8s_pg_cluster_name},cnpg.io/instanceRole=primary "
                f"-o jsonpath='{{.items[0].metadata.name}}'"
            )
            result = subprocess.run(pod_cmd, shell=True, capture_output=True, text=True, timeout=15)
            pod = result.stdout.strip()
            if pod:
                sql = "VACUUM (ANALYZE) district, warehouse, item;"
                subprocess.run(
                    ['kubectl', 'exec', '-n', self.k8s_pg_namespace, pod,
                     '--', 'psql', '-U', 'postgres', '-d', 'tpcc', '-c', sql],
                    check=True, capture_output=True, timeout=300
                )
                logger.info("[Benchmark] VACUUM ANALYZE on hot tables done")
        except Exception as e:
            logger.warning(f"[Benchmark] VACUUM failed (continuing): {e}")

    def _prewarm_shared_buffers(self):
        """Load hot TPC-C tables into shared_buffers via pg_prewarm before measurement.

        Baselines are measured back-to-back in steady-state (buffers already warm from
        prior run). Training steps that follow a pod restart start with cold shared_buffers.
        pg_prewarm equalizes this asymmetry: it reads the most-accessed TPC-C tables
        directly into the buffer pool in ~10-30s without generating dead tuples.

        Called after VACUUM + CHECKPOINT so all loaded pages are clean and won't be
        evicted before the benchmark starts.
        """
        if self.deployment_mode != 'kubernetes':
            return
        import subprocess
        try:
            pod_cmd = (
                f"kubectl get pod -n {self.k8s_pg_namespace} "
                f"-l cnpg.io/cluster={self.k8s_pg_cluster_name},cnpg.io/instanceRole=primary "
                f"-o jsonpath='{{.items[0].metadata.name}}'"
            )
            result = subprocess.run(pod_cmd, shell=True, capture_output=True, text=True, timeout=15)
            pod = result.stdout.strip()
            if pod:
                sql = (
                    "CREATE EXTENSION IF NOT EXISTS pg_prewarm; "
                    "SELECT pg_prewarm('warehouse'), pg_prewarm('district'), "
                    "pg_prewarm('item'), pg_prewarm('stock'), "
                    "pg_prewarm('customer'), pg_prewarm('orders'), "
                    "pg_prewarm('new_order'), pg_prewarm('order_line'), "
                    "pg_prewarm('stock_i1'), pg_prewarm('customer_i1'), pg_prewarm('customer_i2');"
                )
                subprocess.run(
                    ['kubectl', 'exec', '-n', self.k8s_pg_namespace, pod,
                     '--', 'psql', '-U', 'postgres', '-d', 'tpcc', '-c', sql],
                    check=True, capture_output=True, timeout=300
                )
                logger.info("[Benchmark] pg_prewarm complete — hot TPC-C tables loaded into shared_buffers")
        except Exception as e:
            logger.warning(f"[Benchmark] pg_prewarm failed (continuing): {e}")

    def get_benchmark_result(self):
        """Obtiene TPS del benchmark"""
        if self.deployment_mode == 'kubernetes':
            self._run_vacuum_hot_tables()
            # SKIP_PREWARM=1 env var disables pg_prewarm for cold-cache baseline measurements.
            # By default (training + formal eval) prewarm is always active.
            if not os.environ.get('SKIP_PREWARM'):
                self._prewarm_shared_buffers()
            else:
                import logging as _lg
                _lg.getLogger('autotune').info("[Benchmark] pg_prewarm SKIPPED (SKIP_PREWARM=1)")
            return self._run_benchmark_kubernetes()
        else:
            raise NotImplementedError("Benchmark en modo local no implementado")

    def run_warmup_benchmark(self):
        """Run a short TPC-C benchmark to warm up shared_buffers after a restart.

        Uses rampup=1min + duration=1min (total ~3min with K8s overhead).
        The result (TPS) is returned for logging but should be discarded
        by the caller -- the purpose is filling the buffer pool, not measurement.
        """
        if self.deployment_mode != 'kubernetes':
            raise NotImplementedError("Warmup benchmark only supported in kubernetes mode")

        from autotune.workload.factory import WorkloadFactory

        runner = WorkloadFactory.create_runner(
            benchmark_type='tpcc',
            namespace=self.k8s_benchmark_namespace
        )

        vus = int(getattr(self, 'benchmark_tpcc_vus', 20))
        # Flush WAL and dirty pages before warmup so it starts from a clean state.
        # Without this, WAL from prior steps accumulates and spikes I/O mid-warmup.
        self._run_checkpoint()
        # rampup=1min + duration=1min: sufficient to seed the OS page cache.
        # Reduced from 2+3 to limit dead-tuple generation — the 5-min warmup was
        # writing ~270K transactions per step, causing autovacuum pressure that
        # degraded the subsequent 5-min measurement window monotonically.
        results = runner.run(vus=vus, rampup_min=1, duration_min=1)
        tps = results['tpm'] / 60.0
        return tps

    def apply_knobs_kubernetes(self, knobs: dict):
        """
        Aplica knobs al cluster CNPG en Kubernetes
        
        Args:
            knobs: Dict de {knob_name: value}
            
        Returns:
            result dict from cnpg_applier
        """
        from autotune.knob_management.cnpg_applier import CnpgKnobApplier
        
        applier = CnpgKnobApplier(
            namespace=self.k8s_pg_namespace,
            cluster_name=self.k8s_pg_cluster_name,
            tier=self._fog_tier
        )
        
        # Convertir valores a strings
        knobs_str = {k: str(v) for k, v in knobs.items()}
        
        result = applier.apply_knobs(knobs_str)
        
        if result['success']:
            logger.info(f"✅ Knobs aplicados: {list(knobs_str.keys())}")
            if result.get('restart_required'):
                logger.warning("⚠️  Algunos knobs requieren restart")
        else:
            logger.error(f"❌ Error aplicando knobs: {result.get('message')}")
        
        # [Fix] Return the full dictionary, not a tuple, so dbenv can read 'applied' knobs
        return result

    def apply_knobs(self, knobs: dict):
        """Aplica knobs según modo de deployment"""
        if self.deployment_mode == 'kubernetes':
            return self.apply_knobs_kubernetes(knobs)
        else:
            raise NotImplementedError("apply_knobs en modo local no implementado")
