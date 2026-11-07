import os
import pdb
import time
import json
import subprocess
import numpy as np
from datetime import datetime
from multiprocessing import Manager
from multiprocessing.connection import Client
import sys
from .knobs import logger
from .utils.parser import parse_sysbench, parse_oltpbench, parse_job
from .knobs import initialize_knobs, get_default_knobs
import psutil
import multiprocessing as mp
from .resource_monitor import ResourceMonitor
from autotune.workload import SYSBENCH_WORKLOAD, JOB_WORKLOAD, OLTPBENCH_WORKLOADS, TPCH_WORKLOAD, TPCC_WORKLOAD
from autotune.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from autotune.utils.parser import is_number
from autotune.database.postgresqldb import PostgresqlDB

from autotune.reward.fog_aware_reward_v2 import FogAwareRewardV2 as FogAwareReward
from autotune.constraints.tier_profiles import get_profile

class DBEnv:
    def __init__(self, args, args_tune, db):
        self.db = db
        self.args = args
        self.workload = self.get_workload()
        self.log_path = "./log"
        self.num_metrics = self.db.num_metrics
        self.threads = int(args['thread_num'])
        self.best_result = './autotune_best.res'
        self.knobs_detail = initialize_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = get_default_knobs()
        self.online_mode = eval(args['online_mode'])
        self.remote_mode = eval(args['remote_mode'])
        self.oltpbench_config_xml = args['oltpbench_config_xml']
        self.step_count = 0
        self.connect_sucess = True
        self.reinit_interval = 0
        self.reinit = False
        if self.reinit_interval:
            self.reinit = False
        self.generate_time()
        self.y_variable = eval(args_tune['performance_metric'])
        self.reference_point = self.generate_reference_point(eval(args_tune['reference_point']))

        if args_tune['constraints'] is None or args_tune['constraints'] == '':
            self.constraints = []
        else:
            self.constraints = eval(args_tune['constraints'])
        self.lhs_log = args['lhs_log']
        self.cpu_core = args['cpu_core']
        self.info =  {
            'objs': self.y_variable,
            'constraints': self.constraints
        }

        use_fog_str = args_tune.get('use_fog_reward', 'False')
        self.use_fog_reward = use_fog_str.lower() in ['true', '1', 'yes']
        self.fog_reward = None
        # Steps per phase for alpha scaling.
        # Must equal STEPS_PER_CYCLE in the launch script (default 50).
        # Do NOT use max_runs here: max_runs is cumulative across phases and
        # causes alpha to shrink toward 0 in later cycles (v3 bug).
        self.max_steps_total = int(args_tune.get('steps_per_phase', 50))

        if self.use_fog_reward:
            fog_tier = args_tune.get('fog_tier', 'rpi_4gb')
            self.fog_tier = fog_tier
            self.global_t_base = int(args_tune.get('global_t_base', 0))
            # Propagate tier to PostgresqlDB so CnpgKnobApplier can fix shared_buffers
            if hasattr(self.db, '_fog_tier'):
                self.db._fog_tier = fog_tier

            try:
                # Load tier profile
                profile = get_profile(fog_tier)

                # Use tier-specific measured baseline (overrides INI value)
                baseline_tps = float(profile.get('baseline_tps', args_tune.get('baseline_tps', 2000.0)))

                # Create fog-aware reward
                self.fog_reward = FogAwareReward(
                    constraints=profile,
                    baseline_tps=baseline_tps
                )

                logger.info(f"[FogPG-Tuner] ✓ Fog-aware reward enabled")
                logger.info(f"[FogPG-Tuner]   Tier: {fog_tier}")
                logger.info(f"[FogPG-Tuner]   Baseline TPS (tier-specific): {baseline_tps}")
                logger.info(f"[FogPG-Tuner]   Max Memory: {profile['max_memory_mb']}MB")
                logger.info(f"[FogPG-Tuner]   Max Swap: {profile['max_swap_mb']}MB")
            except Exception as e:
                logger.error(f"[FogPG-Tuner] ✗ Failed to init fog reward: {e}")
                import traceback
                traceback.print_exc()
                self.use_fog_reward = False




    def generate_reference_point(self, user_defined_reference_point):
        if len(self.y_variable) <= 1:
            return None

        reference_point_dir = {
            'tps': 0,
            'lat': BENCHMARK_RUNNING_TIME,
            'qps': 0,
            'cpu': 0,
            'readIO': 0,
            'writeIO': 0,
            'virtualMem': 0,
            'physical': 0,
        }
        reference_point = []
        for key in self.y_variable:
            use_defined_value = user_defined_reference_point[self.y_variable.index(key)]
            if is_number(use_defined_value):
                reference_point.append(use_defined_value)
            else:
                key = key.strip().strip('-')
                reference_point.append(reference_point_dir[key])

        return reference_point

    def get_workload(self):
        if self.args['workload'] == 'sysbench':
            wl = dict(SYSBENCH_WORKLOAD)
            wl['type'] = self.args['workload_type']
        elif self.args['workload'].startswith('oltpbench_'):
            wl = dict(OLTPBENCH_WORKLOADS)
        elif self.args['workload'] == 'job':
            wl = dict(JOB_WORKLOAD)
        elif self.args['workload'] == 'tpch':
            wl = dict(TPCH_WORKLOAD)
        elif self.args['workload'] == 'tpcc':
            wl = dict(TPCC_WORKLOAD)
        else:
            raise ValueError('Invalid workload!')
        return wl

    def generate_time(self):
        global BENCHMARK_RUNNING_TIME
        global BENCHMARK_WARMING_TIME
        global TIMEOUT_TIME
        global RESTART_FREQUENCY

        if self.workload['name'] == 'sysbench' or self.workload['name'] == 'oltpbench':
            try:
                BENCHMARK_RUNNING_TIME = int(self.args['workload_time'])
            except:
                BENCHMARK_RUNNING_TIME = 60
            try:
                BENCHMARK_WARMING_TIME = int(self.args['workload_warmup_time'])
            except:
                BENCHMARK_WARMING_TIME = 10
            TIMEOUT_TIME = BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME + 30
            RESTART_FREQUENCY = 200

        elif self.workload['name'] == 'job' or self.workload['name'] == 'tpch' or self.workload['name'] == 'tpcc':
            try:
                BENCHMARK_RUNNING_TIME = int(self.args['workload_time'])
            except:
                BENCHMARK_RUNNING_TIME = 240
            try:
                BENCHMARK_WARMING_TIME = int(self.args['workload_warmup_time'])
            except:
                BENCHMARK_WARMING_TIME = 0
            TIMEOUT_TIME = BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME
            RESTART_FREQUENCY = 30000

        else:
            raise ValueError('Invalid workload name!')

    def get_external_metrics(self, filename=''):
        if self.workload['name'] == 'sysbench':
            for _ in range(60):
                if os.path.exists(filename):
                    break
                time.sleep(1)
            if not os.path.exists(filename):
                print("benchmark result file does not exist!")
            result = parse_sysbench(filename)
        elif self.workload['name'] == 'oltpbench':
            for _ in range(60):
                if os.path.exists('results/{}.summary'.format(filename)):
                    break
                time.sleep(1)
            if not os.path.exists('results/{}.summary'.format(filename)):
                print("benchmark result file does not exist!")
            result = parse_oltpbench('results/{}.summary'.format(filename))
        elif self.workload['name'] == 'job' or self.workload['name'] == 'tpch':
            for _ in range(60):
                if os.path.exists(filename):
                    break
                time.sleep(1)
            if not os.path.exists(filename):
                print("benchmark result file does not exist!")
            dirname, _ = os.path.split(os.path.abspath(__file__))
            select_file = dirname + '/cli/selectedList_{}.txt'.format(self.workload['name'])
            result = parse_job(filename, select_file, timeout=TIMEOUT_TIME)
        elif self.workload['name'] == 'tpcc':
            # In K8s mode, this is handled by _get_states_kubernetes.
            # In Local mode, we would parse the log.
            # For now, just return empty/dummy to avoid crash if called.
            result = {'tps': 0, 'lat': 0, 'qps': 0, 'tpsVar': 0, 'latVar': 0, 'qpsVar': 0}
        else:
            raise ValueError('Invalid workload name!')
        return result

    def get_benchmark_cmd(self):
        timestamp = int(time.time())
        filename = self.log_path + '/{}.log'.format(timestamp)
        dirname, _ = os.path.split(os.path.abspath(__file__))
        if self.workload['name'] == 'sysbench':
            if(isinstance(self.db, PostgresqlDB)):
                exe_file =  '/cli/run_sysbench_postgresql.sh'
            else:
                exe_file = '/cli/run_sysbench.sh'
            cmd = self.workload['cmd'].format(dirname + exe_file,
                                              self.workload['type'],
                                              self.db.host,
                                              self.db.port,
                                              self.db.user,
                                              self.db.passwd,
                                              10,
                                              10000,
                                              BENCHMARK_WARMING_TIME,
                                              self.threads,
                                              BENCHMARK_RUNNING_TIME,
                                              filename,
                                              self.db.dbname)
        elif self.workload['name'] == 'oltpbench':
            filename = filename.split('/')[-1].split('.')[0]
            cmd = self.workload['cmd'].format(dirname + '/cli/run_oltpbench.sh',
                                              self.db.dbname,
                                              self.oltpbench_config_xml,
                                              filename)
        elif self.workload['name'] == 'job':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_job_{}.sh'.format(self.db.args['db']),
                                              dirname + '/cli/selectedList_job.txt',
                                              dirname + '/job_query/queries-{}-new'.format(self.db.args['db']),
                                              filename,
                                              self.db.sock)
        elif self.workload['name'] == 'tpch':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_tpch_{}.sh'.format(self.db.args['db']),
                                              dirname + '/cli/selectedList_tpch.txt',
                                              dirname + '/tpch_query/queries-{}-new'.format(self.db.args['db']),
                                              filename,
                                              self.db.sock)
        else:
            raise ValueError('Invalid workload name!')

        logger.info('[DBG]. {}'.format(cmd))
        return cmd, filename

    def get_states(self, collect_resource=0):

        if hasattr(self.db, 'deployment_mode') and self.db.deployment_mode == 'kubernetes':
            return self._get_states_kubernetes(collect_resource)

        # start Internal Metrics Collection
        internal_metrics = Manager().list()
        im = mp.Process(target=self.db.get_internal_metrics,
                        args=(internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME))
        self.db.set_im_alive(True)
        im.start()

        # start Resource Monition (if activated)
        if collect_resource:
            if self.remote_mode:
                # start remote Resource Monitor
                clientDB_address = (self.db.host, 6001)
                clientDB_conn = Client(clientDB_address, authkey=b'DBTuner')
                clientDB_conn.send(self.db.pid)
            else:
                rm = ResourceMonitor(self.db.pid, 1, BENCHMARK_WARMING_TIME, BENCHMARK_RUNNING_TIME)
                rm.run()

        # start Benchmark
        benchmark_timeout = False
        cmd, filename = self.get_benchmark_cmd()
        print("="*80)
        print(f"DEBUG: Command = {cmd}")
        print(f"DEBUG: Filename = {filename}")
        print("="*80)
        print(cmd)
        print("[{}] benchmark start!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        p_benchmark = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
        try:
            outs, errs = p_benchmark.communicate(timeout=TIMEOUT_TIME)
            ret_code = p_benchmark.poll()
            if ret_code == 0:
                print("[{}] benchmark finished!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            else:
                print("run benchmark get error {}".format(ret_code))
        except subprocess.TimeoutExpired:
            #benchmark_timeout = True
            print("[{}] benchmark timeout!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        # terminate Benchmark
        if not self.remote_mode:
            subprocess.Popen(self.db.clear_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                             close_fds=True)
            print("[{}] clear processlist".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        # stop Internal Metrics Collection
        self.db.set_im_alive(False)
        im.join()

        # stop Resource Monition (if activated)
        if collect_resource:
            if self.remote_mode:
                # send Benchmark-Finish msg to remote Resource Monitor Process
                clientDB_conn.send('benchmark_finished')
                # receive remote Monitor Data
                monitor_data = clientDB_conn.recv()
                cpu, avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory = monitor_data
                # close connection
                clientDB_conn.close()

            else:
                rm.terminate()
                cpu, avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory = rm.get_monitor_data_avg()
        else:
            cpu, avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory = 0, 0, 0, 0, 0

        external_metrics = self.get_external_metrics(filename)
        internal_metrics, dirty_pages, hit_ratio, page_data = self.db._post_handle(internal_metrics)
        logger.info('internal metrics: {}.'.format(list(internal_metrics)))

        return benchmark_timeout, external_metrics, internal_metrics, (
            cpu, avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory, dirty_pages, hit_ratio, page_data)

    def _get_states_kubernetes(self, collect_resource=0):
        """get_states para modo kubernetes"""
        # Start metrics collection
        internal_metrics = Manager().list()
        im = mp.Process(target=self.db.get_internal_metrics,
                        args=(internal_metrics, BENCHMARK_RUNNING_TIME, BENCHMARK_WARMING_TIME))
        self.db.set_im_alive(True)
        im.start()

        # Run K8s benchmark
        try:
            tps = self.db.get_benchmark_result()
            logger.info(f'[K8s] Benchmark TPS: {tps}')
            external_metrics = [tps, 0, 0, 0, 0, 0]
            benchmark_timeout = False
        except Exception as e:
            logger.error(f'[K8s] Benchmark failed: {e}')
            external_metrics = [0, 0, 0, 0, 0, 0]
            benchmark_timeout = True

        # Stop metrics
        self.db.set_im_alive(False)
        im.join()
        
        # Extract system metrics BEFORE _post_handle (while still in dict format)
        cpu, avg_physical_memory, avg_io_wait, avg_total_memory, avg_swap = 0, 0, 0, 0, 0
        if len(internal_metrics) > 0:
            # Calculate AVERAGE over all samples
            def get_avg_val(key):
                values = []
                for sample in internal_metrics:
                    val = sample.get(key, 0)
                    if isinstance(val, list):
                        # Should not happen with new logic, but safe fallback
                        for v in val:
                            if v is not None: values.append(float(v))
                    else:
                        try:
                            values.append(float(val))
                        except:
                            pass

                if len(values) > 0:
                    return sum(values) / len(values)
                return 0.0

            cpu = get_avg_val('system.cpu_percent')
            # Use container working-set bytes (not node memory) to correctly compare
            # against tier_profile['max_memory_mb'] which is the K8s container limit.
            # Node memory includes OS overhead and would inflate the ratio.
            avg_physical_memory = get_avg_val('system.container_memory_mb')
            if avg_physical_memory == 0.0:
                avg_physical_memory = get_avg_val('system.memory_mb')
                logger.warning('[K8s] container_memory_mb=0, falling back to node memory_mb for reward')
            avg_io_wait = get_avg_val('system.io_wait')
            avg_total_memory = get_avg_val('system.total_memory_mb')
            avg_swap = get_avg_val('system.swap_used_mb')

            logger.info(f'[K8s] Extracted resources: CPU={cpu:.1f}%, MemUsed={avg_physical_memory:.1f}MB, MemLimit={avg_total_memory:.1f}MB, I/O Wait={avg_io_wait:.1f}%, Swap={avg_swap:.1f}MB')

        avg_read_io, avg_write_io, avg_virtual_memory = 0, 0, 0

        # Post-process (converts dicts to array)
        internal_metrics, dirty_pages, hit_ratio, page_data = self.db._post_handle(internal_metrics)
        logger.info('internal metrics: {}.'.format(list(internal_metrics)))

        # Return extended resource tuple (indices 0-10)
        return benchmark_timeout, external_metrics, internal_metrics, (cpu, avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory, dirty_pages, hit_ratio, page_data, avg_io_wait, avg_total_memory, avg_swap)

    

    def measure_default_tps(self, n=1):
        """Apply Default config, run n benchmark(s), return avg TPS.
        Called by the optimizer for rolling baseline recalibration.
        After this call, PostgreSQL is running with Default config.
        """
        default_knobs = {k: v['default'] for k, v in self.knobs_detail.items()}
        logger.info(f'[Recalibration] Applying Default knobs: {default_knobs}')
        try:
            if self.online_mode:
                res = self.db.apply_knobs_online(default_knobs)
            else:
                res = self.db.apply_knobs_offline(default_knobs)
            if isinstance(res, dict):
                restart_happened = res.get('restart_required', False)
            elif isinstance(res, tuple):
                _, restart_happened = res
            else:
                restart_happened = False
            if restart_happened:
                logger.info('[Recalibration] Restart after Default config, waiting 30s...')
                time.sleep(30)
            tps_values = []
            for i in range(n):
                tps = self.db.get_benchmark_result()
                if tps and tps > 0:
                    tps_values.append(tps)
                    logger.info(f'[Recalibration] Run {i+1}/{n}: TPS={tps:.1f}')
            if tps_values:
                avg = sum(tps_values) / len(tps_values)
                logger.info(f'[Recalibration] Default TPS={avg:.1f}')
                return avg
            logger.warning('[Recalibration] No valid TPS measurements, keeping old baseline')
            return None
        except Exception as e:
            logger.warning(f'[Recalibration] Failed: {e}')
            return None

    def apply_knobs(self, knobs):
        for key in knobs.keys():
            value = knobs[key]
            if not key in self.knobs_detail.keys() or not self.knobs_detail[key]['type'] == 'integer':
                continue
            if value > self.knobs_detail[key]['max']:
                knobs[key] = self.knobs_detail[key]['max']
                logger.info("{} with value of is larger than max, adjusted".format(key))
            elif value < self.knobs_detail[key]['min']:
                knobs[key] = self.knobs_detail[key]['min']
                logger.info("{} with value of is smaller than min, adjusted".format(key))

        logger.info("[step {}] generate knobs: {}\n".format(self.step_count, knobs))
        if self.online_mode:
            flag = self.db.apply_knobs_online(knobs)
        else:
            flag = self.db.apply_knobs_offline(knobs)

        if not flag:
            if self.reinit:
                logger.info('reinitializing db begin')
                self.db.reinitdb_magic(self.remote_mode)
                logger.info('db reinitialized')

            raise Exception('Apply knobs failed!')


    def normalize_knobs(self, knobs):
        """
        [DBTune Optimization]
        Normalize memory-related knobs to the current hardware tier.
        Interprets the agent's action (relative to global max) as a fraction
        and scales it to the current tier's safe limits.
        """
        if not self.use_fog_reward or not self.fog_reward:
            return knobs

        constraints = self.fog_reward.constraints
        max_mem_mb = constraints.get('max_memory_mb', 2048)
        
        # 0. shared_buffers (Unit: 8KB blocks)
        # Global action space is [128MB–2048MB]. The cap must leave room not only for
        # OS + K8s sidecar but also for the rest of PostgreSQL's RAM (work_mem pool,
        # maintenance_work_mem, wal_buffers, per-connection overhead). On 2GB
        # (max_memory_mb=1536) a -500 margin (cap 1036MB) still pushed total RSS to
        # ~1887MB, above the penalty threshold. -836 gives cap 700MB on 2GB (~4x the
        # 160MB optimum). Also lowers 3GB cap (2300-836=1464MB, vs optimal ~640MB —
        # still safe, no penalty at that level). 4GB+ caps exceed action space max (2048MB)
        # so never bind.
        if 'shared_buffers' in knobs:
            safe_max_sb_mb = max_mem_mb - 836
            max_blocks = int(safe_max_sb_mb * 1024 / 8)
            if knobs['shared_buffers'] > max_blocks:
                logger.info(f"[Norm] shared_buffers: {knobs['shared_buffers']} -> {max_blocks} (OOM cap: {safe_max_sb_mb}MB for max_mem={max_mem_mb}MB)")
                knobs['shared_buffers'] = max_blocks

        # 1. work_mem (Unit: KB)
        # Goal: Scale to [1MB, MaxSafePerConn]
        if 'work_mem' in knobs:
            wm_val = knobs['work_mem']
            wm_def = self.knobs_detail['work_mem']
            
            fraction = (wm_val - wm_def['min']) / (wm_def['max'] - wm_def['min'])
            fraction = max(0.0, min(1.0, fraction))
            
            # Use max_work_mem_mb from tier_profiles if available (tuned per tier).
            # Fallback: (Total * 0.25) / connections — but this gives 67.5MB on 6GB/20VUs,
            # which is 4x too high vs PGTune recommendations and saturates CPU.
            max_conns = knobs.get('max_connections', 20)
            tier_max_wm = constraints.get('max_work_mem_mb', None)
            if tier_max_wm:
                safe_max_mb = float(tier_max_wm)
            else:
                safe_max_mb = (max_mem_mb * 0.25) / max_conns
            safe_min_mb = 1 # 1MB minimum
            
            if safe_max_mb < safe_min_mb: safe_max_mb = safe_min_mb
            
            new_mb = safe_min_mb + fraction * (safe_max_mb - safe_min_mb)
            new_kb = int(new_mb * 1024)
            
            logger.info(f"[Norm] work_mem: {wm_val} -> {new_kb} (Frac: {fraction:.2f}, Limit: {safe_max_mb:.1f}MB)")
            knobs['work_mem'] = new_kb
            
        # 2. maintenance_work_mem (Unit: KB)
        # Goal: Scale to [16MB, tier_profile max or 20% RAM]
        if 'maintenance_work_mem' in knobs:
            mwm_val = knobs['maintenance_work_mem']
            mwm_def = self.knobs_detail['maintenance_work_mem']

            fraction = (mwm_val - mwm_def['min']) / (mwm_def['max'] - mwm_def['min'])
            fraction = max(0.0, min(1.0, fraction))

            tier_max_mwm = constraints.get('max_maintenance_work_mem_mb', None)
            if tier_max_mwm:
                safe_max_mb = float(tier_max_mwm)
            else:
                safe_max_mb = min(1024, max_mem_mb * 0.20)
            safe_min_mb = mwm_def['min'] / 1024  # JSON min in KB → MB (postgres_fog_v6: 64MB)

            new_mb = safe_min_mb + fraction * (safe_max_mb - safe_min_mb)
            new_kb = int(new_mb * 1024)

            logger.info(f"[Norm] maintenance_work_mem: {mwm_val} -> {new_kb} (Frac: {fraction:.2f}, Range: [{safe_min_mb:.0f}..{safe_max_mb:.0f}]MB)")
            knobs['maintenance_work_mem'] = new_kb

        return knobs

    def step_GP(self, knobs, collect_resource=True):
        #return False, np.random.rand(6), np.random.rand(65), np.random.rand(8)
        # re-init database if activated
        if self.reinit_interval > 0 and self.reinit_interval % RESTART_FREQUENCY == 0:
            if self.reinit:
                logger.info('reinitializing db begin')
                self.db.reinitdb_magic(self.remote_mode)
                logger.info('db reinitialized')
        self.step_count = self.step_count + 1
        self.reinit_interval = self.reinit_interval + 1

        # modify and apply knobs
        for key in knobs.keys():
            value = knobs[key]
            if not key in self.knobs_detail.keys() or not self.knobs_detail[key]['type'] == 'integer':
                continue
            if value > self.knobs_detail[key]['max']:
                knobs[key] = self.knobs_detail[key]['max']
                logger.info("{} with value of is larger than max, adjusted".format(key))
            elif value < self.knobs_detail[key]['min']:
                knobs[key] = self.knobs_detail[key]['min']
                logger.info("{} with value of is smaller than min, adjusted".format(key))
        
        # [DBTune Optimization] Normalize Knobs based on Tier
        knobs = self.normalize_knobs(knobs)


        logger.info("[step {}] generate knobs: {}\n".format(self.step_count, knobs))

        restart_happened = False
        applied_knobs = {}
        
        if self.online_mode:
            res = self.db.apply_knobs_online(knobs)
            if isinstance(res, dict):
                 flag = res.get('success', False)
                 restart_happened = res.get('restart_required', False)
                 applied_knobs = res.get('applied', {})
            elif isinstance(res, tuple):
                flag, restart_happened = res
            else:
                flag = res
        else:
            res = self.db.apply_knobs_offline(knobs)
            if isinstance(res, dict):
                 flag = res.get('success', False)
                 restart_happened = res.get('restart_required', False)
                 applied_knobs = res.get('applied', {})
            elif isinstance(res, tuple):
                flag, restart_happened = res
            else:
                 flag = res

        if not flag:
            if self.reinit:
                logger.info('reinitializing db begin')
                self.db.reinitdb_magic(self.remote_mode)
                logger.info('db reinitialized')

            raise Exception('Apply knobs failed!')
            
        # [DBTune Optimization] Enhanced Logging
        if applied_knobs:
             logger.info(f"[Optimization] ✅ Actual Knobs Applied (Diff): {list(applied_knobs.keys())}")
        else:
             logger.info(f"[Optimization] ℹ️  No Knobs Changed (All matched existing config)")
        
        # [v7] Warm-up after restart: stabilize + fill buffer pool before measurement.
        # Post-restart shared_buffers are empty → TPS 30-50% lower than steady-state.
        # A short TPC-C (rampup=1min, duration=1min, ~3min total) fills the buffer
        # pool with the real workload pages before the actual measurement.
        if restart_happened:
            logger.info(f"🔥 Restart detected — waiting 30s for PostgreSQL to stabilize...")
            time.sleep(30)
            # Warmup benchmark REMOVED (v11 C2 fix): baselines do not run a warmup TPC-C
            # between measurements, so running one here after restart would inflate agent TPS
            # vs baselines. Both conditions now use only pg_prewarm (called in get_benchmark_result).
            # This matches OtterTune/LlamaTune protocol: cold-start after config change.

        s = self.get_states(collect_resource=collect_resource)

        if s is None:
            if self.reinit:
                logger.info('reinitializing db begin')
                self.db.reinitdb_magic(self.remote_mode)
                logger.info('db reinitialized')

            raise Exception('Get states failed!')

        timeout, external_metrics, internal_metrics, resource = s

        # Adjust formatting to include io_wait and total_memory
        # cpu, readIO, writeIO, virtualMem, physical, dirty, hit, data, io_wait, total_mem
        io_wait = resource[8] if len(resource) > 8 else 0.0
        total_mem = resource[9] if len(resource) > 9 else 0.0

        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|ioWait_{}|totMem_{}|{}|65d\n'
        res = format_str.format(knobs, str(external_metrics[0]), str(external_metrics[1]), str(external_metrics[2]),
                                external_metrics[3], external_metrics[4],
                                external_metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], io_wait, total_mem, list(internal_metrics))
        
        return timeout, external_metrics, internal_metrics, resource

    def get_objs(self, res):
            """
            Calculate objectives (minimize by default)
            FogPG-Tuner: Can use fog-aware reward if enabled
            """
            # ⭐ FogPG-Tuner: Use fog-aware reward if enabled
            # ⭐ FogPG-Tuner: Use fog-aware reward if enabled
            if self.use_fog_reward and self.fog_reward is not None:
                # Delegate calculation to FogAwareReward (Decoupled Logic)
                # Now dbenv doesn't need to know about alpha scaling or specific metric keys
                reward_value = self.fog_reward.get_reward_from_metrics(
                    res_dict=res,
                    step_count=self.step_count,
                    max_steps=self.max_steps_total
                )
                
                # Check if configuration violates hard constraints (kept for safety, though reward should handle it)
                # if not self.fog_reward.is_valid_config(res.get('physical', 0), 0):
                #     logger.warning("[FogPG-Tuner] ✗ Config violates hard constraints")
                
                # Return as single objective (maximize reward = minimize negative reward)
                return [-reward_value]
            
            # ⭐ Original DBTune behavior (when fog reward is disabled)
            objs = []
            for y_variable in self.y_variable:
                key = y_variable.strip().strip('-')
                value = res[key]
                if not y_variable.strip()[0] == '-':
                    value = - value
                objs.append(value)
            return objs

    def get_constraints(self, res):
        if len(self.constraints) == 0:
            return None

        locals().update(res)
        constraintL = []
        for constraint in self.constraints:
            value = eval(constraint)
            constraintL.append(value)

        return constraintL

    def step(self, config):

        knobs = config.get_dictionary().copy()
        for k in self.knobs_detail.keys():
            if k in knobs.keys():
                if self.knobs_detail[k]['type'] == 'integer' and self.knobs_detail[k]['max'] > sys.maxsize:
                    knobs[k] = knobs[k] * 1000
            else:
                knobs[k] = self.knobs_detail[k]['default']

        try:
            timeout, metrics, internal_metrics, resource = self.step_GP(knobs, collect_resource=True)

            if timeout:
                trial_state = TIMEOUT
            else:
                trial_state = SUCCESS

            external_metrics = {
                'tps': metrics[0],
                'lat': metrics[1],
                'qps': metrics[2],
                'tpsVar': metrics[3],
                'latVar': metrics[4],
                'qpsVar': metrics[5],
            }

            resource_dict = {
                'cpu': resource[0],
                'readIO': resource[1],
                'writeIO': resource[2],
                'IO': resource[1] + resource[2],
                'virtualMem': resource[3],
                'physical': resource[4],
                'dirty': resource[5],
                'hit': resource[6],
                'data': resource[7],
                'io_wait': resource[8] if len(resource) > 8 else 0.0,
                'total_mem': resource[9] if len(resource) > 9 else 0.0,
                'swap': resource[10] if len(resource) > 10 else 0.0,
            }

            res = dict(external_metrics, **resource_dict)
            objs = self.get_objs(res)
            constraints = self.get_constraints(res)
            self._append_step_record(knobs, res, objs)
            return objs, constraints, external_metrics, resource, list(internal_metrics), self.info, trial_state
        except Exception as e:
            print("="*80)
            print(f"ERROR in step(): {e}")
            import traceback
            traceback.print_exc()
            print("="*80)
            print("="*80)
            return None, None, {}, {}, [0.0] * self.num_metrics, self.info, FAILED

    def _append_step_record(self, knobs, res, objs):
        try:
            breakdown = getattr(self.fog_reward, 'last_breakdown', None) if self.use_fog_reward else None
            record = {
                'global_t':   getattr(self, 'step_offset', getattr(self, 'global_t_base', 0)) + getattr(self, 'step_count', 0),
                'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'tier':       getattr(self, 'fog_tier', None),
                'cycle':      os.environ.get('TRAINING_CYCLE'),
                'tps':        round(res.get('tps', 0), 4),
                'reward':     round(-objs[0], 4) if objs else None,
                'penalties':  breakdown,
                'config':     {k: round(v, 6) if isinstance(v, float) else v for k, v in knobs.items()},
                'resources': {
                    'cpu_pct':      round(res.get('cpu', 0), 2),
                    'mem_used_mb':  round(res.get('physical', 0), 1),
                    'mem_limit_mb': round(res.get('total_mem', 0), 1),
                    'io_wait_pct':  round(res.get('io_wait', 0), 2),
                    'swap_mb':      round(res.get('swap', 0), 1),
                },
            }
            task_id = getattr(self, 'task_id', None) or os.environ.get('TASK_ID', 'unknown')
            logfile = f'training_steps_{task_id}.jsonl'
            with open(logfile, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.warning(f'[StepRecord] Failed to write step record: {e}')
